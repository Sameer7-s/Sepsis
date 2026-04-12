"""
Sepsis Management OpenEnv — submission-safe standalone inference loop
Never starts a backend server — waits for one provided by Docker/HuggingFace.
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Environment variables
BACKEND_BASE = os.getenv("SEPSIS_BACKEND_BASE", "http://127.0.0.1:7860").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
API_KEY = (os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "").strip()

OPENENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")

# Scenarios and actions must match backend_api.py exactly
SCENARIOS = ["early_sepsis", "severe_sepsis", "septic_shock"]
ALLOWED_ACTIONS = [
    "observe",
    "administer_antibiotics",
    "give_fluids",
    "oxygen_therapy",
    "start_vasopressors",
    "perform_source_control",
    "noop",
]

# Score safety: keep strictly inside (0, 1) with comfortable margin
MIN_SAFE_SCORE = 0.05
MAX_SAFE_SCORE = 0.95
DEFAULT_SAFE_SCORE = 0.50
MAX_STEPS = 20


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
    """Clamp and round before serialization. Never returns 0.0 or 1.0."""
    clamped = clamp_score(value)
    rounded = round(clamped, 4)
    if rounded <= 0.0:
        rounded = MIN_SAFE_SCORE
    if rounded >= 1.0:
        rounded = MAX_SAFE_SCORE
    return float(rounded)


def http_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Make HTTP request and return parsed JSON."""
    url = f"{BACKEND_BASE}{path}"
    data = None
    headers = {"Content-Type": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url, data=data, headers=headers, method=method)

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


def wait_for_backend(timeout: float = 30.0) -> bool:
    """Wait for backend health check to pass."""
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            result = http_json("GET", "/health", timeout=2.0)
            if isinstance(result, dict) and result.get("status") == "ok":
                print(f"[INFO] Backend health check passed (attempt {attempt})")
                return True
        except Exception as e:
            elapsed = time.time() - start
            if elapsed < 5:  # Log first few attempts
                print(f"[DEBUG] Health check attempt {attempt} failed: {e}")
        time.sleep(0.5)
    return False
        "early_sepsis": "early",
        "mild": "early",
        "mild_sepsis": "early",
        "severe": "severe",
        "severe_sepsis": "severe",
        "shock": "shock",
        "septic_shock": "shock",
    }
    return mapping.get(value, value)


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


def format_scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


def llm_action(state: Dict[str, float], history: List[Dict], severity: str) -> Optional[str]:
    """
    Optional LLM policy.
    Falls back to deterministic logic if model endpoint is unavailable.
    """
    if not MODEL_NAME or MODEL_NAME.strip().startswith("<"):
        return None

    if not API_BASE_URL or API_BASE_URL.strip().startswith("<"):
        return None

    api_key = HF_TOKEN or "dummy-key"

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
        )

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
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        action = (response.choices[0].message.content or "").strip()
        if action in ALLOWED_ACTIONS:
            return action
        return None

    except Exception:
        return None


def deterministic_action(state: Dict[str, float], history: List[Dict], severity: str) -> str:
    severity = normalize_severity(severity)
    actions_taken = [item["action"] for item in history]
    taken_set = set(actions_taken)
    last_action = actions_taken[-1] if actions_taken else None

    if "administer_antibiotics" not in taken_set:
        return "administer_antibiotics"

    if state.get("oxygen_saturation", 100) < 90:
        return "oxygen_therapy"

    if (
        state.get("oxygen_saturation", 100) < 94
        or state.get("respiratory_rate", 0) > 22
    ) and last_action != "oxygen_therapy":
        return "oxygen_therapy"

    if severity == "shock":
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

    elif severity == "severe":
        if (
            state.get("mean_arterial_pressure", 100) < 70
            or state.get("lactate", 0) > 2.5
        ) and last_action != "give_fluids":
            return "give_fluids"

        if state.get("sofa_score", 0) >= 4 and "perform_source_control" not in taken_set:
            return "perform_source_control"

    elif severity == "early":
        if state.get("lactate", 0) > 2.2 or state.get("mean_arterial_pressure", 100) < 70:
            return "give_fluids"
        return "observe"

    if state.get("lactate", 0) > 2.2 or state.get("mean_arterial_pressure", 100) < 70:
        return "give_fluids"

    if state.get("oxygen_saturation", 100) < 96 and last_action != "oxygen_therapy":
        return "oxygen_therapy"

    return "observe"


def choose_action(state: Dict[str, float], history: List[Dict], severity: str) -> str:
    action = llm_action(state, history, severity)
    if action in ALLOWED_ACTIONS:
        return action
    return deterministic_action(state, history, severity)


def run_episode(scenario: str) -> Tuple[float, List[Dict]]:
    reset = http_json("POST", "/reset", {"scenario": scenario})
    validate_reset_response(reset)

    episode_id = reset["episode_id"]
    severity = normalize_severity(reset["severity"])
    state = reset["state"]
    history: List[Dict] = []

    print(
        f"[START] scenario={format_scenario_label(scenario)} "
        f"severity={severity} episode={episode_id[:8]}"
    )

    final_score = DEFAULT_SAFE_SCORE

    while True:
        action = choose_action(state, history, severity)
        result = http_json("POST", "/step", {"episode_id": episode_id, "action": action})
        validate_step_response(result)

        reward = safe_float(result.get("reward", 0.0), 0.0)
        state = result["state"]
        done = bool(result["done"])
        raw_score = safe_float(result.get("normalized_score", DEFAULT_SAFE_SCORE), DEFAULT_SAFE_SCORE)
        score = clamp_score(raw_score)
        step = int(result["step_count"])

        history.append(
            {
                "action": action,
                "reward": reward,
                "raw_score": raw_score,
                "score": score,
            }
        )

        print(
            f"[STEP] step={step} action={action} reward={reward:+.4f} "
            f"raw_score={raw_score:.4f} safe_score={score:.4f} done={done}"
        )

        final_score = score

        if done or step >= MAX_STEPS:
            final_score = clamp_score(final_score)
            print(
                f"[END] scenario={format_scenario_label(scenario)} "
                f"final_score={final_score:.4f} steps={step}"
            )
            return final_score, history


def build_submission_payload(
    scores_by_scenario: Dict[str, float],
    steps_by_scenario: Dict[str, int],
) -> Dict:
    """
    Build the final JSON payload.
    All scores are hard-clamped here — NO assertions are used because
    Python -O (used in many Docker images) silently disables assert statements.
    """
    scenario_results = {}

    for scenario in SCENARIOS:
        raw = scores_by_scenario.get(scenario, DEFAULT_SAFE_SCORE)
        scenario_score = score_for_output(raw)

        # Hard clamp — does not rely on assert
        if scenario_score <= 0.0 or scenario_score >= 1.0:
            scenario_score = clamp_score(scenario_score)

        scenario_results[scenario] = {
            "score": scenario_score,
            "steps": int(steps_by_scenario.get(scenario, 0)),
        }

    raw_overall = sum(v["score"] for v in scenario_results.values()) / len(scenario_results)
    overall = score_for_output(raw_overall)

    # Hard clamp on overall too
    if overall <= 0.0 or overall >= 1.0:
        overall = clamp_score(overall)

    # Final sanity log (non-fatal)
    for scenario, result in scenario_results.items():
        s = result["score"]
        if not (0.0 < s < 1.0):
            print(f"[WARN] Score for {scenario} is still out of range after clamping: {s}")

    return {
        "final_score": overall,
        "task_scores": scenario_results,
        "model": MODEL_NAME,
    }


def main() -> None:
    """Run inference against a running backend server."""
    try:
        print(f"[INFO] Waiting for backend at {BACKEND_BASE}...")
        if not wait_for_backend(timeout=30.0):
            print(
                f"[ERROR] Backend did not become reachable at {BACKEND_BASE} within 30 seconds.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("[INFO] Backend is reachable.")
        if os.path.exists(OPENENV_PATH):
            print(f"[INFO] OpenEnv spec: {OPENENV_PATH}")

        scores_by_scenario: Dict[str, float] = {}
        steps_by_scenario: Dict[str, int] = {}

        for scenario in SCENARIOS:
            episode_score, history = run_episode(scenario)
            safe_episode_score = clamp_score(episode_score)

            # Log if clamping had to correct the score
            if safe_episode_score != episode_score:
                print(
                    f"[WARN] Score for {scenario} was clamped: "
                    f"{episode_score:.6f} → {safe_episode_score:.6f}"
                )

            scores_by_scenario[scenario] = safe_episode_score
            steps_by_scenario[scenario] = len(history)

        payload = build_submission_payload(scores_by_scenario, steps_by_scenario)

        print(f"[FINAL] overall_score={payload['final_score']:.4f}")
        print(json.dumps(payload, indent=2))

    except Exception as e:
        print(f"[FATAL] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
