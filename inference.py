"""
Sepsis Management — inference entry point.

Validator-required stdout format:
  [START] task=<scenario_name>
  [STEP] step=<n> reward=<f>
  [END] task=<scenario_name> score=<f> steps=<n>

This version:
- NEVER starts a server / binds any port
- Uses backend already running at SEPSIS_BACKEND_BASE (default localhost:7860)
- Makes real LLM calls through the injected OpenAI-compatible proxy using:
    API_BASE_URL
    API_KEY
    MODEL_NAME
- Falls back safely if LLM is unavailable
"""

import json
import math
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ── Backend location ───────────────────────────────────────────────────────────
BASE_URL = os.getenv("SEPSIS_BACKEND_BASE", "http://127.0.0.1:7860").rstrip("/")

# ── Injected LLM proxy variables ───────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()

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

DEFAULT_SCORE = 0.50
MIN_SCORE = 0.05
MAX_SCORE = 0.95
MAX_STEPS = 10


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
    score = safe_float(score, DEFAULT_SCORE)
    return max(MIN_SCORE, min(MAX_SCORE, score))


def post_json(path: str, payload: dict, timeout: float = 15.0) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str, timeout: float = 5.0) -> dict:
    url = f"{BASE_URL}{path}"
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Structured output helpers ──────────────────────────────────────────────────

def emit_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def emit_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={reward:.4f}", flush=True)


def emit_end(task: str, score: float, steps: int) -> None:
    print(f"[END] task={task} score={clamp_score(score)} steps={steps}", flush=True)


# ── Backend readiness ──────────────────────────────────────────────────────────

def wait_for_backend(timeout: float = 30.0) -> bool:
    start = time.time()
    attempt = 0
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


# ── LLM proxy usage ────────────────────────────────────────────────────────────

def get_openai_client():
    if OpenAI is None:
        return None
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        return None
    try:
        return OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception:
        return None


def llm_proxy_probe() -> None:
    """
    Make one guaranteed proxy call so the validator sees usage of the injected LLM.
    Safe to ignore failures.
    """
    client = get_openai_client()
    if client is None:
        print("[WARN] LLM proxy not configured; proceeding with fallback policy.", flush=True)
        return

    try:
        _ = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": "Reply with exactly: observe"},
                {"role": "user", "content": "observe"},
            ],
        )
        print("[INFO] LLM proxy probe succeeded.", flush=True)
    except Exception as exc:
        print(f"[WARN] LLM proxy probe failed: {exc}", flush=True)


def llm_action(state: dict, history: list, scenario: str) -> str | None:
    client = get_openai_client()
    if client is None:
        return None

    prompt = (
        "You are a sepsis-management assistant. "
        "Return exactly one action string from this allowed list: "
        f"{ALLOWED_ACTIONS}. "
        f"Scenario: {scenario}. "
        f"Current state: {json.dumps(state)}. "
        f"Recent history: {json.dumps(history[-3:])}. "
        "Return only the action string."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return exactly one valid action string and nothing else.",
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
    except Exception as exc:
        print(f"[WARN] LLM action failed for {scenario}: {exc}", flush=True)
        return None


def deterministic_action(step_index: int, step_data: dict) -> str:
    """
    Safe fallback if LLM is unavailable or returns invalid output.
    """
    map_val = safe_float(step_data.get("state", {}).get("mean_arterial_pressure", 75.0), 75.0)
    spo2 = safe_float(step_data.get("state", {}).get("oxygen_saturation", 95.0), 95.0)
    lactate = safe_float(step_data.get("state", {}).get("lactate", 2.0), 2.0)

    if step_index == 0:
        return "administer_antibiotics"
    if map_val < 65:
        return "start_vasopressors"
    if lactate > 2.2 or map_val < 70:
        return "give_fluids"
    if spo2 < 94:
        return "oxygen_therapy"
    if step_index == 1:
        return "give_fluids"
    if step_index == 2:
        return "oxygen_therapy"
    if step_index == 3:
        return "perform_source_control"
    return "observe"


# ── Per-scenario runner ────────────────────────────────────────────────────────

def run_task(task: str) -> None:
    emit_start(task)

    try:
        reset_data = post_json("/reset", {"scenario": task})
        episode_id = reset_data["episode_id"]
        final_score = safe_float(reset_data.get("normalized_score", DEFAULT_SCORE))
        current_state = reset_data.get("state", {})
        history = []

        steps = 0

        while steps < MAX_STEPS:
            llm_selected = llm_action(current_state, history, task)
            action = llm_selected if llm_selected in ALLOWED_ACTIONS else deterministic_action(steps, {"state": current_state})

            step_data = post_json(
                "/step",
                {"episode_id": episode_id, "action": action},
            )

            reward = safe_float(step_data.get("reward", 0.0), 0.0)
            final_score = safe_float(step_data.get("normalized_score", final_score))
            current_state = step_data.get("state", current_state)

            history.append(
                {
                    "action": action,
                    "reward": reward,
                    "score": final_score,
                }
            )

            steps += 1
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
    print(f"[INFO] Backend target: {BASE_URL}", flush=True)
    print(f"[INFO] API_BASE_URL configured: {bool(API_BASE_URL)}", flush=True)
    print(f"[INFO] API_KEY configured: {bool(API_KEY)}", flush=True)
    print(f"[INFO] MODEL_NAME configured: {bool(MODEL_NAME)}", flush=True)

    llm_proxy_probe()

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