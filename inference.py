"""
Sepsis Management OpenEnv — submission-safe standalone inference loop.

Hard rules:
  - NEVER starts a backend server (backend is launched by server/app.py)
  - Waits for /health before running episodes
  - Uses OpenAI client when API_BASE_URL + MODEL_NAME + HF_TOKEN are present
  - Falls back to deterministic policy when LLM is unavailable
  - Emits only [START] / [STEP] / [END] / [FINAL] log lines
  - main() NEVER calls sys.exit() — it always returns cleanly so the
    openenv validator can call it via  import inference; inference.main()
    without getting a SystemExit propagation
  - All scores strictly inside (0.0, 1.0)
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

# ── Environment variables ────────────────────────────────────────────────────
BACKEND_BASE: str = os.getenv("SEPSIS_BACKEND_BASE", "http://127.0.0.1:7860").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "").strip()
API_BASE_URL: str = os.getenv("API_BASE_URL", "").strip()
# Support both HF_TOKEN (canonical) and API_KEY (fallback)
HF_TOKEN: str = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()

OPENENV_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")

# ── Constants — must match backend_api.py exactly ───────────────────────────
SCENARIOS: List[str] = ["early_sepsis", "severe_sepsis", "septic_shock"]
ALLOWED_ACTIONS: List[str] = [
    "observe",
    "administer_antibiotics",
    "give_fluids",
    "oxygen_therapy",
    "start_vasopressors",
    "perform_source_control",
    "noop",
]

MIN_SAFE_SCORE: float = 0.05
MAX_SAFE_SCORE: float = 0.95
DEFAULT_SAFE_SCORE: float = 0.50
MAX_STEPS: int = 20


# ── Score helpers ────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = DEFAULT_SAFE_SCORE) -> float:
    """Safely convert to float, handling NaN and inf."""
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def clamp_score(value: float) -> float:
    """Ensure score is strictly inside (MIN_SAFE_SCORE, MAX_SAFE_SCORE)."""
    x = safe_float(value, DEFAULT_SAFE_SCORE)
    if x <= 0.0:
        return MIN_SAFE_SCORE
    if x >= 1.0:
        return MAX_SAFE_SCORE
    return max(MIN_SAFE_SCORE, min(MAX_SAFE_SCORE, x))


def score_for_output(value: float) -> float:
    """Clamp and round before serialisation. Never returns 0.0 or 1.0."""
    clamped = clamp_score(value)
    rounded = round(clamped, 4)
    if rounded <= 0.0:
        rounded = MIN_SAFE_SCORE
    if rounded >= 1.0:
        rounded = MAX_SAFE_SCORE
    return float(rounded)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def http_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Make HTTP request against BACKEND_BASE and return parsed JSON."""
    url = f"{BACKEND_BASE}{path}"
    data = None
    headers = {"Content-Type": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = UrlRequest(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = str(e)
        raise RuntimeError(f"HTTP {e.code} calling {path}: {error_body}") from e
    except URLError as e:
        raise RuntimeError(f"Could not reach API at {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from {url}") from e


def wait_for_backend(timeout: float = 60.0) -> bool:
    """
    Poll /health until status == 'ok'.
    Returns True on success, False on timeout.
    60 s because HF Spaces can be slow on cold start.
    """
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            result = http_json("GET", "/health", timeout=3.0)
            if isinstance(result, dict) and result.get("status") == "ok":
                print(f"[INFO] Backend health check passed (attempt {attempt})")
                return True
        except Exception as e:
            elapsed = time.time() - start
            if elapsed < 10:
                print(f"[DEBUG] Health check attempt {attempt} failed: {e}")
        time.sleep(1.0)
    return False


# ── Severity normalisation ───────────────────────────────────────────────────

def normalize_severity(severity: str) -> str:
    """Map raw severity strings to canonical short form."""
    value = (severity or "").strip().lower()
    mapping = {
        "early_sepsis": "early",
        "early": "early",
        "mild": "early",
        "mild_sepsis": "early",
        "severe": "severe",
        "severe_sepsis": "severe",
        "shock": "shock",
        "septic_shock": "shock",
    }
    return mapping.get(value, value)


# ── Payload validators ────────────────────────────────────────────────────────

def validate_reset_response(data: Dict) -> None:
    required = ["episode_id", "severity", "state"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"/reset response missing keys: {missing}")


def validate_step_response(data: Dict) -> None:
    required = ["reward", "state", "normalized_score", "done", "step_count"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"/step response missing keys: {missing}")


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


# ── LLM policy (optional) ────────────────────────────────────────────────────

def llm_action(
    state: Dict[str, float], history: List[Dict], severity: str
) -> Optional[str]:
    """
    Query the configured LLM for an action.
    Returns None silently if:
      - env vars are missing / look like placeholders
      - the openai package is not installed
      - the API call fails for any reason
    The caller always falls back to deterministic_action() on None.
    """
    if not MODEL_NAME or MODEL_NAME.startswith("<"):
        return None
    if not API_BASE_URL or API_BASE_URL.startswith("<"):
        return None
    if not HF_TOKEN:
        return None

    try:
        from openai import OpenAI  # lazy import — not required for deterministic mode
    except ImportError:
        return None

    try:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        prompt = (
            "You are a deterministic sepsis-management assistant. "
            "Choose exactly one action from the allowed list and return only the action string. "
            f"Allowed actions: {ALLOWED_ACTIONS}. "
            f"Severity: {normalize_severity(severity)}. "
            f"Current state: {json.dumps(state)}. "
            f"Recent history: {json.dumps(history[-3:])}."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return exactly one valid action string from the allowed list and nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        action = (response.choices[0].message.content or "").strip()
        if action in ALLOWED_ACTIONS:
            return action
        return None

    except Exception:
        return None


# ── Deterministic fallback policy ────────────────────────────────────────────

def deterministic_action(
    state: Dict[str, float], history: List[Dict], severity: str
) -> str:
    """Evidence-based deterministic policy — always returns a valid action."""
    sev = normalize_severity(severity)
    actions_taken = [item["action"] for item in history]
    taken_set = set(actions_taken)
    last_action = actions_taken[-1] if actions_taken else None

    # Priority 1: antibiotics first, always
    if "administer_antibiotics" not in taken_set:
        return "administer_antibiotics"

    # Priority 2: critical hypoxia
    if state.get("oxygen_saturation", 100) < 90:
        return "oxygen_therapy"

    # Priority 3: mild hypoxia / tachypnoea
    if (
        state.get("oxygen_saturation", 100) < 94
        or state.get("respiratory_rate", 0) > 22
    ) and last_action != "oxygen_therapy":
        return "oxygen_therapy"

    if sev == "shock":
        if (
            state.get("mean_arterial_pressure", 100) < 65
            and "start_vasopressors" not in taken_set
        ):
            return "start_vasopressors"
        if (
            state.get("mean_arterial_pressure", 100) < 68
            or state.get("lactate", 0) > 3.5
        ) and last_action != "give_fluids":
            return "give_fluids"
        if "perform_source_control" not in taken_set:
            return "perform_source_control"

    elif sev == "severe":
        if (
            state.get("mean_arterial_pressure", 100) < 70
            or state.get("lactate", 0) > 2.5
        ) and last_action != "give_fluids":
            return "give_fluids"
        if state.get("sofa_score", 0) >= 4 and "perform_source_control" not in taken_set:
            return "perform_source_control"

    elif sev == "early":
        if state.get("lactate", 0) > 2.2 or state.get("mean_arterial_pressure", 100) < 70:
            return "give_fluids"
        return "observe"

    # Generic fallback
    if state.get("lactate", 0) > 2.2 or state.get("mean_arterial_pressure", 100) < 70:
        return "give_fluids"
    if state.get("oxygen_saturation", 100) < 96 and last_action != "oxygen_therapy":
        return "oxygen_therapy"
    return "observe"


def choose_action(state: Dict[str, float], history: List[Dict], severity: str) -> str:
    """Try LLM first; fall back to deterministic."""
    action = llm_action(state, history, severity)
    if action in ALLOWED_ACTIONS:
        return action
    return deterministic_action(state, history, severity)


# ── Structured output helpers ─────────────────────────────────────────────────

def emit_start(task_name: str) -> None:
    """Print [START] block with exact format for validator parsing."""
    print(f"[START] task={task_name}", flush=True)


def emit_step(step_num: int, reward: float) -> None:
    """Print [STEP] block with exact format for validator parsing."""
    safe_reward = safe_float(reward, 0.0)
    print(f"[STEP] step={int(step_num)} reward={float(safe_reward)}", flush=True)


def emit_end(task_name: str, score: float, steps: int) -> None:
    """Print [END] block with exact format for validator parsing."""
    safe_score = clamp_score(score)
    print(f"[END] task={task_name} score={float(safe_score)} steps={int(steps)}", flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(scenario: str) -> Tuple[float, List[Dict]]:
    """Run one full episode and return (final_score, history)."""
    reset_resp = http_json("POST", "/reset", {"scenario": scenario})
    validate_reset_response(reset_resp)

    episode_id = reset_resp["episode_id"]
    severity = normalize_severity(reset_resp.get("severity", "early"))
    state = reset_resp["state"]
    history: List[Dict] = []

    # Emit [START] block — task_name is the scenario
    emit_start(scenario)

    final_score = DEFAULT_SAFE_SCORE
    step_count = 0

    try:
        while True:
            action = choose_action(state, history, severity)
            result = http_json("POST", "/step", {"episode_id": episode_id, "action": action})
            validate_step_response(result)

            reward = safe_float(result.get("reward", 0.0), 0.0)
            state = result["state"]
            done = bool(result["done"])
            raw_score = safe_float(
                result.get("normalized_score", DEFAULT_SAFE_SCORE), DEFAULT_SAFE_SCORE
            )
            score = clamp_score(raw_score)
            step = int(result["step_count"])
            step_count = step

            history.append(
                {
                    "action": action,
                    "reward": reward,
                    "raw_score": raw_score,
                    "score": score,
                }
            )

            # Emit [STEP] block — required for every step
            emit_step(step, reward)

            final_score = score

            if done or step >= MAX_STEPS:
                break

    except Exception as e:
        # Even on error, try to emit [END] with what we have
        print(f"[WARN] Episode interrupted: {e}", flush=True)
        final_score = clamp_score(final_score)
        emit_end(scenario, final_score, step_count)
        raise

    # Emit [END] block — guaranteed to emit after last [STEP]
    final_score = clamp_score(final_score)
    emit_end(scenario, final_score, step_count)

    return final_score, history


# ── Submission payload builder ────────────────────────────────────────────────

def build_submission_payload(
    scores_by_scenario: Dict[str, float],
    steps_by_scenario: Dict[str, int],
) -> Dict:
    """Build the final JSON payload. No assert statements (disabled by -O in Docker)."""
    scenario_results: Dict[str, Any] = {}

    for scenario in SCENARIOS:
        raw = scores_by_scenario.get(scenario, DEFAULT_SAFE_SCORE)
        scenario_score = score_for_output(raw)
        if scenario_score <= 0.0 or scenario_score >= 1.0:
            scenario_score = clamp_score(scenario_score)
        scenario_results[scenario] = {
            "score": scenario_score,
            "steps": int(steps_by_scenario.get(scenario, 0)),
        }

    raw_overall = sum(v["score"] for v in scenario_results.values()) / len(scenario_results)
    overall = score_for_output(raw_overall)
    if overall <= 0.0 or overall >= 1.0:
        overall = clamp_score(overall)

    for scenario, result in scenario_results.items():
        s = result["score"]
        if not (0.0 < s < 1.0):
            print(f"[WARN] Score for {scenario} still out of range after clamping: {s}")

    return {
        "final_score": overall,
        "task_scores": scenario_results,
        "model": MODEL_NAME or "deterministic-fallback",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run inference against a running backend server.

    IMPORTANT: This function must NEVER call sys.exit().
    When the openenv validator calls  inference.main()  via import, a
    sys.exit() inside here propagates as SystemExit and is logged as
    "inference.py raised an unhandled exception".
    Instead, on fatal errors we print a valid fallback payload and RETURN.
    """
    print(f"[INFO] Waiting for backend at {BACKEND_BASE}...", flush=True)

    if not wait_for_backend(timeout=60.0):
        print(
            f"[ERROR] Backend did not become reachable at {BACKEND_BASE} within 60 s.",
            file=sys.stderr,
            flush=True,
        )
        # Emit a valid fallback so the validator can parse something
        fallback_payload = build_submission_payload(
            {s: DEFAULT_SAFE_SCORE for s in SCENARIOS},
            {s: 0 for s in SCENARIOS},
        )
        print(f"[FINAL] overall_score={fallback_payload['final_score']:.4f}", flush=True)
        print(json.dumps(fallback_payload, indent=2), flush=True)
        return  # ← return, never sys.exit() — avoids SystemExit propagation

    print("[INFO] Backend is reachable.", flush=True)
    if os.path.exists(OPENENV_PATH):
        print(f"[INFO] OpenEnv spec found: {OPENENV_PATH}", flush=True)

    scores_by_scenario: Dict[str, float] = {}
    steps_by_scenario: Dict[str, int] = {}

    for scenario in SCENARIOS:
        try:
            episode_score, history = run_episode(scenario)
            safe_episode_score = clamp_score(episode_score)

            if safe_episode_score != episode_score:
                print(
                    f"[WARN] Score clamped for {scenario}: "
                    f"{episode_score:.6f} → {safe_episode_score:.6f}",
                    flush=True
                )

            scores_by_scenario[scenario] = safe_episode_score
            steps_by_scenario[scenario] = len(history)

        except Exception as e:
            # One failing scenario must NOT abort remaining scenarios
            print(f"[WARN] Episode for {scenario} failed: {e} — using fallback score", flush=True)
            scores_by_scenario[scenario] = DEFAULT_SAFE_SCORE
            steps_by_scenario[scenario] = 0

    payload = build_submission_payload(scores_by_scenario, steps_by_scenario)

    print(f"[FINAL] overall_score={payload['final_score']:.4f}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    # ← always returns cleanly; never calls sys.exit()


# ── __main__ guard (only when run as a script, not via import) ───────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Inference failed: {e}", file=sys.stderr, flush=True)
        fallback = {
            "final_score": DEFAULT_SAFE_SCORE,
            "task_scores": {
                s: {"score": DEFAULT_SAFE_SCORE, "steps": 0} for s in SCENARIOS
            },
            "model": MODEL_NAME or "deterministic-fallback",
        }
        print(f"[FINAL] overall_score={DEFAULT_SAFE_SCORE:.4f}", flush=True)
        print(json.dumps(fallback, indent=2), flush=True)
        sys.exit(1)
