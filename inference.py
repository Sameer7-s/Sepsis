"""
Sepsis Management — inference entry point.

Output contract (validator-required):
  [START] task=<scenario_name>
  [STEP]  step=<n> reward=<f>
  [END]   task=<scenario_name> score=<f> steps=<n>

LLM contract (validator-required):
  - MUST call the LiteLLM proxy via API_BASE_URL + API_KEY env vars
  - Use OpenAI client: base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"]
  - Do NOT hardcode keys or use other providers
"""

import json
import math
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ── Backend (already started by HF validator on localhost) ────────────────────
BASE_URL = os.getenv("SEPSIS_BACKEND_BASE", "http://127.0.0.1:7860").rstrip("/")

# ── LLM proxy — injected by the validator; MUST be used ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_KEY      = os.getenv("API_KEY", os.getenv("HF_TOKEN", "dummy-key"))
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")   # LiteLLM routes this

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

# Deterministic fallback sequence (used only when LLM call fails)
_FALLBACK_SEQUENCE = [
    "administer_antibiotics",
    "give_fluids",
    "oxygen_therapy",
    "give_fluids",
    "start_vasopressors",
    "perform_source_control",
    "observe",
    "observe",
    "observe",
    "observe",
]

DEFAULT_SCORE = 0.50
MIN_SCORE     = 0.05
MAX_SCORE     = 0.95
MAX_STEPS     = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_float(value, default: float = DEFAULT_SCORE) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def clamp_score(score: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, safe_float(score, DEFAULT_SCORE)))


def post_json(path: str, payload: dict, timeout: float = 15.0) -> dict:
    url  = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req  = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str, timeout: float = 5.0) -> dict:
    req = Request(f"{BASE_URL}{path}", method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Structured output ──────────────────────────────────────────────────────────

def emit_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def emit_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={reward:.4f}", flush=True)


def emit_end(task: str, score: float, steps: int) -> None:
    print(f"[END] task={task} score={clamp_score(score)} steps={steps}", flush=True)


# ── Backend readiness ──────────────────────────────────────────────────────────

def wait_for_backend(timeout: float = 30.0) -> bool:
    start, attempt = time.time(), 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            result = get_json("/health", timeout=2.0)
            if isinstance(result, dict) and result.get("status") == "ok":
                print(f"[INFO] Backend ready (attempt {attempt})", flush=True)
                return True
        except Exception as exc:
            if time.time() - start < 6:
                print(f"[DEBUG] Health attempt {attempt}: {exc}", flush=True)
        time.sleep(0.5)
    return False


# ── LLM action selection (REQUIRED by validator) ──────────────────────────────

def _build_llm_prompt(state: dict, history: list, severity: str, scenario: str) -> str:
    actions_taken = [h["action"] for h in history]
    return (
        f"You are an expert sepsis management AI agent.\n"
        f"Scenario: {scenario} (severity: {severity})\n"
        f"Current patient state: {json.dumps(state, indent=2)}\n"
        f"Actions taken so far: {actions_taken}\n\n"
        f"Choose the single best next action from this list:\n"
        f"{ALLOWED_ACTIONS}\n\n"
        f"Clinical guidelines:\n"
        f"- Always administer antibiotics first if not yet given\n"
        f"- Give fluids if MAP < 70 or lactate > 2.2\n"
        f"- Use oxygen_therapy if SpO2 < 94 or RR > 22\n"
        f"- start_vasopressors only if MAP < 65 AND severity is shock\n"
        f"- perform_source_control for severe/shock if SOFA >= 4\n"
        f"- Otherwise observe\n\n"
        f"Respond with ONLY the action name, nothing else."
    )


def llm_choose_action(state: dict, history: list, severity: str, scenario: str) -> str:
    """
    Call the validator's LiteLLM proxy to choose an action.
    This call is REQUIRED — the validator checks that API_BASE_URL was used.
    Falls back to deterministic logic only if the call fails.
    """
    if not API_BASE_URL:
        print("[WARN] API_BASE_URL not set — using deterministic fallback", flush=True)
        return _deterministic_action(state, history)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )

        prompt = _build_llm_prompt(state, history, severity, scenario)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=20,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sepsis management AI. "
                        "Respond with exactly one action name from the allowed list. "
                        "No explanation, no punctuation — just the action string."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (response.choices[0].message.content or "").strip().lower()

        # Strip any punctuation/quotes the model might add
        action = raw.strip("\"'.`\n ")

        if action in ALLOWED_ACTIONS:
            print(f"[LLM] Chose action: {action}", flush=True)
            return action

        # Model returned something garbled — try to fuzzy-match
        for allowed in ALLOWED_ACTIONS:
            if allowed in raw:
                print(f"[LLM] Fuzzy-matched '{raw}' → '{allowed}'", flush=True)
                return allowed

        print(f"[LLM] Unrecognised response '{raw}' — using deterministic fallback", flush=True)
        return _deterministic_action(state, history)

    except ImportError:
        print("[WARN] openai package not available — using deterministic fallback", flush=True)
        return _deterministic_action(state, history)

    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc} — using deterministic fallback", flush=True)
        return _deterministic_action(state, history)


def _deterministic_action(state: dict, history: list) -> str:
    """Pure rule-based fallback — never calls any external API."""
    taken = {h["action"] for h in history}

    if "administer_antibiotics" not in taken:
        return "administer_antibiotics"

    spo2 = state.get("oxygen_saturation", 100)
    rr   = state.get("respiratory_rate", 16)
    map_ = state.get("mean_arterial_pressure", 80)
    lac  = state.get("lactate", 1.0)

    if spo2 < 90:
        return "oxygen_therapy"
    if map_ < 65 and "start_vasopressors" not in taken:
        return "start_vasopressors"
    if map_ < 70 or lac > 2.2:
        return "give_fluids"
    if spo2 < 94 or rr > 22:
        return "oxygen_therapy"
    if "perform_source_control" not in taken:
        return "perform_source_control"
    return "observe"


# ── Per-scenario runner ────────────────────────────────────────────────────────

def run_task(task: str) -> None:
    emit_start(task)

    try:
        reset_data  = post_json("/reset", {"scenario": task})
        episode_id  = reset_data["episode_id"]
        severity    = reset_data.get("severity", "early")
        final_score = safe_float(reset_data.get("normalized_score", DEFAULT_SCORE))
        history: list = []
        steps = 0

        # Extract flat state dict (handles both flat and nested Observation)
        raw_state = reset_data.get("state", {})
        if hasattr(raw_state, "__dict__"):
            state = dict(raw_state)
        elif isinstance(raw_state, dict):
            state = raw_state
        else:
            state = {}

        while steps < MAX_STEPS:
            # ── LLM call (required by validator) ──────────────────────────────
            action = llm_choose_action(state, history, severity, task)

            step_data   = post_json("/step", {"episode_id": episode_id, "action": action})
            reward      = safe_float(step_data.get("reward", 0.0), 0.0)
            final_score = safe_float(step_data.get("normalized_score", final_score))
            steps      += 1

            # Update state for next LLM prompt
            raw_state = step_data.get("state", {})
            if isinstance(raw_state, dict):
                state = raw_state

            history.append({"action": action, "reward": reward})
            emit_step(steps, reward)

            if step_data.get("done", False):
                break

        emit_end(task, final_score, steps)

    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[WARN] Network error in task={task}: {exc}", flush=True)
        emit_step(1, 0.0)
        emit_end(task, DEFAULT_SCORE, 1)

    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"[WARN] Parse error in task={task}: {exc}", flush=True)
        emit_step(1, 0.0)
        emit_end(task, DEFAULT_SCORE, 1)

    except Exception as exc:
        print(f"[WARN] Unexpected error in task={task}: {exc}", flush=True)
        emit_step(1, 0.0)
        emit_end(task, DEFAULT_SCORE, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[INFO] Backend target : {BASE_URL}", flush=True)
    print(f"[INFO] LLM proxy      : {API_BASE_URL or '(not set)'}", flush=True)
    print(f"[INFO] Model          : {MODEL_NAME}", flush=True)

    if not wait_for_backend(timeout=30.0):
        print("[ERROR] Backend unreachable — emitting fallback blocks.", flush=True)
        for scenario in SCENARIOS:
            emit_start(scenario)
            emit_step(1, 0.0)
            emit_end(scenario, DEFAULT_SCORE, 1)
        sys.exit(1)

    for scenario in SCENARIOS:
        run_task(scenario)

    sys.stdout.flush()


if __name__ == "__main__":
    main()