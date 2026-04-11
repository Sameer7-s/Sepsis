"""
Sepsis Management OpenEnv — optimized submission-safe inference loop.

Goals:
- Never starts a backend server.
- Waits for backend health only.
- Avoids unhandled crashes where possible.
- Uses deterministic policy by default.
- Optionally uses LLM if API_BASE_URL + API_KEY are set.
- Prints validator-friendly final JSON payload.
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BACKEND_BASE = os.getenv("SEPSIS_BACKEND_BASE", "http://127.0.0.1:7860").rstrip("/")
# Validate BACKEND_BASE is not empty
if not BACKEND_BASE:
    print("[ERROR] BACKEND_BASE is empty or not set", file=sys.stderr)
    sys.exit(1)
    
OPENENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")

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

DEFAULT_SAFE_SCORE = 0.5
SAFE_MIN_SCORE = 0.01
SAFE_MAX_SCORE = 0.99
MAX_STEPS = 20

MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
API_KEY = (os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "").strip()


def safe_float(value: Any, default: float = DEFAULT_SAFE_SCORE) -> float:
    try:
        if isinstance(value, bool):
            return float(default)
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def clamp_open_interval(value: float) -> float:
    x = safe_float(value, DEFAULT_SAFE_SCORE)
    if x <= 0.0:
        return SAFE_MIN_SCORE
    if x >= 1.0:
        return SAFE_MAX_SCORE
    if x < SAFE_MIN_SCORE:
        return SAFE_MIN_SCORE
    if x > SAFE_MAX_SCORE:
        return SAFE_MAX_SCORE
    return x


def http_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Make HTTP request and validate JSON response."""
    # Validate inputs
    if not method or not isinstance(method, str):
        raise RuntimeError(f"Invalid HTTP method: {method}")
    if not path or not isinstance(path, str):
        raise RuntimeError(f"Invalid path: {path}")
    if not path.startswith("/"):
        raise RuntimeError(f"Path must start with '/', got: {path}")
    
    url = f"{BACKEND_BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    req = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req, timeout=timeout) as resp:
            # Validate response status
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            
            body = resp.read().decode("utf-8")
            if not body:
                raise RuntimeError(f"Empty response body from {url}")
                
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
            return parsed
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = str(exc)
        raise RuntimeError(f"HTTP {exc.code} calling {path}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach API at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


def wait_for_backend(timeout: float = 30.0) -> bool:
    """Wait for backend health check with detailed logging."""
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            result = http_json("GET", "/health", timeout=2.0)
            # Validate response structure
            if not isinstance(result, dict):
                raise RuntimeError(f"Expected dict response, got {type(result).__name__}")
            if result.get("status") == "ok":
                print(f"[INFO] Backend health check passed (attempt {attempt})")
                return True
            else:
                print(f"[DEBUG] Backend health check returned status: {result.get('status')}")
        except Exception as exc:
            elapsed = time.time() - start
            if elapsed < 5:  # Only log first few attempts to avoid spam
                print(f"[DEBUG] Health check attempt {attempt} failed: {exc}")
        time.sleep(0.5)
    return False


def normalize_severity(severity: str) -> str:
    value = (severity or "").strip().lower()
    mapping = {
        "early": "early",
        "early_sepsis": "early",
        "mild": "early",
        "mild_sepsis": "early",
        "severe": "severe",
        "severe_sepsis": "severe",
        "shock": "shock",
        "septic_shock": "shock",
    }
    return mapping.get(value, value)


def validate_reset_response(data: Dict[str, Any]) -> None:
    """Validate /reset response has all required fields and correct types."""
    required = ["episode_id", "severity", "state", "normalized_score"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"/reset response missing keys: {missing}")
    
    # Validate types
    if not isinstance(data["episode_id"], str) or not data["episode_id"].strip():
        raise RuntimeError(f"/reset response field 'episode_id' must be a non-empty string")
    if not isinstance(data["severity"], str) or not data["severity"].strip():
        raise RuntimeError(f"/reset response field 'severity' must be a non-empty string")
    if not isinstance(data["state"], dict) or not data["state"]:
        raise RuntimeError("/reset response field 'state' must be a non-empty dict")
    if not isinstance(data["normalized_score"], (int, float)):
        raise RuntimeError("/reset response field 'normalized_score' must be numeric")
    if data["normalized_score"] < 0.0 or data["normalized_score"] > 1.0:
        raise RuntimeError(f"/reset response 'normalized_score' out of range: {data['normalized_score']}")


def validate_step_response(data: Dict[str, Any]) -> None:
    """Validate /step response has all required fields and correct types."""
    required = ["reward", "state", "normalized_score", "done", "step_count", "message"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"/step response missing keys: {missing}")
    
    # Validate types
    if not isinstance(data["state"], dict) or not data["state"]:
        raise RuntimeError("/step response field 'state' must be a non-empty dict")
    if not isinstance(data["done"], bool):
        raise RuntimeError("/step response field 'done' must be a bool")
    if not isinstance(data["step_count"], int) or data["step_count"] < 0:
        raise RuntimeError("/step response field 'step_count' must be non-negative int")
    if not isinstance(data["reward"], (int, float)):
        raise RuntimeError("/step response field 'reward' must be numeric")
    if not isinstance(data["normalized_score"], (int, float)):
        raise RuntimeError("/step response field 'normalized_score' must be numeric")
    if not isinstance(data["message"], str) or not data["message"].strip():
        raise RuntimeError("/step response field 'message' must be a non-empty string")
    if data["normalized_score"] < 0.0 or data["normalized_score"] > 1.0:
        raise RuntimeError(f"/step response 'normalized_score' out of range: {data['normalized_score']}")


def get_llm_client() -> Optional[Any]:
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        return None

    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[LLM] unavailable, falling back to deterministic policy: {exc}")
        return None


def normalize_action_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip().lower()
    for ch in ("-", " ", ".", ",", ":", ";"):
        normalized = normalized.replace(ch, "_" if ch in ("-", " ") else "")
    return normalized


def match_allowed_action(raw: str) -> Optional[str]:
    """Match raw action string to allowed actions with validation."""
    if not raw:
        return None

    normalized = normalize_action_text(raw)
    if not normalized:
        return None

    if normalized in ALLOWED_ACTIONS:
        return normalized

    alias_map = {
        "antibiotics": "administer_antibiotics",
        "administer_antibiotic": "administer_antibiotics",
        "administer_antibiotics": "administer_antibiotics",
        "give_antibiotics": "administer_antibiotics",
        "fluids": "give_fluids",
        "give_fluid": "give_fluids",
        "give_fluids": "give_fluids",
        "oxygen": "oxygen_therapy",
        "oxygen_therapy": "oxygen_therapy",
        "vasopressors": "start_vasopressors",
        "start_vasopressor": "start_vasopressors",
        "start_vasopressors": "start_vasopressors",
        "source_control": "perform_source_control",
        "perform_source_control": "perform_source_control",
        "observe": "observe",
        "noop": "noop",
    }

    mapped = alias_map.get(normalized)
    if mapped in ALLOWED_ACTIONS:
        return mapped

    # Fuzzy matching as fallback
    for action in ALLOWED_ACTIONS:
        if normalized.startswith(action) or action in normalized:
            return action

    return None


def llm_choose_action(
    client: Optional[Any],
    state: Dict[str, float],
    history: List[Dict[str, Any]],
    severity: str,
) -> str:
    if client is None:
        return deterministic_action(state, history, severity)

    recent_actions = [item.get("action", "") for item in history[-5:]]

    system_prompt = (
        "You are a clinical decision-support AI for sepsis management. "
        "Choose exactly one next action from the allowed list. "
        "Return only the exact action string."
    )
    user_prompt = (
        f"Severity: {severity}\n"
        f"State: {json.dumps(state, sort_keys=True)}\n"
        f"Recent actions: {recent_actions}\n"
        f"Allowed actions: {ALLOWED_ACTIONS}\n"
        "Return only one exact allowed action."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        matched = match_allowed_action(raw)
        if matched:
            return matched
        print(f"[LLM] invalid action {raw!r}, using deterministic fallback")
    except Exception as exc:
        print(f"[LLM] call failed, using deterministic fallback: {exc}")

    return deterministic_action(state, history, severity)


def deterministic_action(
    state: Dict[str, float],
    history: List[Dict[str, Any]],
    severity: str,
) -> str:
    severity = normalize_severity(severity)
    actions_taken = [str(item.get("action", "")) for item in history]
    taken_set = set(actions_taken)
    last_action = actions_taken[-1] if actions_taken else None

    if "administer_antibiotics" not in taken_set:
        return "administer_antibiotics"

    spo2 = safe_float(state.get("oxygen_saturation"), 100.0)
    rr = safe_float(state.get("respiratory_rate"), 18.0)
    map_val = safe_float(state.get("mean_arterial_pressure"), 75.0)
    lactate = safe_float(state.get("lactate"), 2.0)
    sofa = safe_float(state.get("sofa_score"), 2.0)

    if spo2 < 90.0:
        return "oxygen_therapy"

    if (spo2 < 94.0 or rr > 22.0) and last_action != "oxygen_therapy":
        return "oxygen_therapy"

    if severity == "shock":
        if map_val < 65.0 and "start_vasopressors" not in taken_set:
            return "start_vasopressors"
        if (map_val < 68.0 or lactate > 3.5) and last_action != "give_fluids":
            return "give_fluids"
        if "perform_source_control" not in taken_set:
            return "perform_source_control"

    elif severity == "severe":
        if (map_val < 70.0 or lactate > 2.5) and last_action != "give_fluids":
            return "give_fluids"
        if sofa >= 4.0 and "perform_source_control" not in taken_set:
            return "perform_source_control"

    elif severity == "early":
        if lactate > 2.2 or map_val < 70.0:
            return "give_fluids"
        return "observe"

    if lactate > 2.2 or map_val < 70.0:
        return "give_fluids"

    if spo2 < 96.0 and last_action != "oxygen_therapy":
        return "oxygen_therapy"

    return "observe"


def format_scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


def run_episode(
    scenario: str,
    client: Optional[Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run a single episode with comprehensive validation."""
    # Validate scenario
    if scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario '{scenario}'. Must be one of: {SCENARIOS}")
    
    reset = http_json("POST", "/reset", {"scenario": scenario})
    validate_reset_response(reset)

    episode_id = str(reset["episode_id"])
    severity = normalize_severity(str(reset["severity"]))
    state = dict(reset["state"])
    history: List[Dict[str, Any]] = []

    print(
        f"[START] scenario={format_scenario_label(scenario)} "
        f"severity={severity} episode={episode_id[:8]}"
    )

    final_score = clamp_open_interval(reset["normalized_score"])
    local_step_count = 0

    while True:
        local_step_count += 1

        if client is not None and local_step_count == 1:
            action = llm_choose_action(client, state, history, severity)
            if action not in ALLOWED_ACTIONS:
                print(f"[WARNING] LLM returned invalid action '{action}', using deterministic fallback")
                action = deterministic_action(state, history, severity)
            print(f"[LLM ] step={local_step_count} action={action}")
        else:
            action = deterministic_action(state, history, severity)
        
        # Final validation before sending to backend
        if action not in ALLOWED_ACTIONS:
            raise RuntimeError(f"Invalid action selected: {action}. Must be one of: {ALLOWED_ACTIONS}")

        result = http_json("POST", "/step", {"episode_id": episode_id, "action": action})
        validate_step_response(result)

        reward = safe_float(result.get("reward"), 0.0)
        state = dict(result["state"])
        done = bool(result["done"])
        backend_score = clamp_open_interval(result["normalized_score"])
        backend_step = int(result["step_count"])
        message = str(result.get("message", ""))

        history.append(
            {
                "action": action,
                "reward": float(reward),
                "normalized_score": float(backend_score),
                "message": message,
            }
        )

        print(
            f"[STEP] step={backend_step} action={action} reward={reward:+.4f} "
            f"score={backend_score:.4f} done={done} msg={message}"
        )

        final_score = backend_score

        if done or backend_step >= MAX_STEPS:
            final_score = clamp_open_interval(final_score)
            print(
                f"[END] scenario={format_scenario_label(scenario)} "
                f"final_score={final_score:.4f} steps={backend_step}"
            )
            return final_score, history


def build_final_payload(scores_by_scenario: Dict[str, float]) -> Dict[str, Any]:
    """Build and validate final submission payload."""
    early = clamp_open_interval(scores_by_scenario.get("early_sepsis", DEFAULT_SAFE_SCORE))
    severe = clamp_open_interval(scores_by_scenario.get("severe_sepsis", DEFAULT_SAFE_SCORE))
    shock = clamp_open_interval(scores_by_scenario.get("septic_shock", DEFAULT_SAFE_SCORE))
    final_score = clamp_open_interval((early + severe + shock) / 3.0)

    payload = {
        "final_score": float(final_score),
        "task_scores": {
            "early_sepsis": float(early),
            "severe_sepsis": float(severe),
            "septic_shock": float(shock),
        },
    }
    
    # Validate payload structure
    validate_payload(payload)
    return payload


def validate_payload(payload: Dict[str, Any]) -> None:
    """Validate final submission payload has all required fields and correct types."""
    # Check required top-level keys
    required_keys = ["final_score", "task_scores"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise RuntimeError(f"Payload missing required keys: {missing}")
    
    # Check final_score is a valid number
    final_score = payload["final_score"]
    if not isinstance(final_score, (int, float)):
        raise RuntimeError(f"final_score must be numeric, got {type(final_score).__name__}")
    if final_score < 0.0 or final_score > 1.0:
        raise RuntimeError(f"final_score must be in range [0.0, 1.0], got {final_score}")
    
    # Check task_scores structure
    task_scores = payload["task_scores"]
    if not isinstance(task_scores, dict):
        raise RuntimeError(f"task_scores must be a dict, got {type(task_scores).__name__}")
    
    # Check all required scenarios are present
    required_scenarios = {"early_sepsis", "severe_sepsis", "septic_shock"}
    missing_scenarios = required_scenarios - set(task_scores.keys())
    if missing_scenarios:
        raise RuntimeError(f"task_scores missing scenarios: {missing_scenarios}")
    
    # Validate each scenario score
    for scenario, score in task_scores.items():
        if scenario not in required_scenarios:
            raise RuntimeError(f"Unknown scenario '{scenario}' in task_scores")
        if not isinstance(score, (int, float)):
            raise RuntimeError(f"task_scores[{scenario}] must be numeric, got {type(score).__name__}")
        if score < 0.0 or score > 1.0:
            raise RuntimeError(f"task_scores[{scenario}] must be in range [0.0, 1.0], got {score}")
    
    print("[VALIDATION] ✓ Payload structure is valid")


def main() -> None:
    try:
        print(f"[INFO] Waiting for backend at {BACKEND_BASE} ...")
        if not wait_for_backend(timeout=30.0):
            print(
                f"[ERROR] Backend did not become reachable at {BACKEND_BASE} within 30 seconds.",
                file=sys.stderr,
            )
            fallback = build_final_payload({})
            print("[FINAL] submission payload (backend unavailable - using fallback):")
            print(json.dumps(fallback, indent=2, sort_keys=True))
            sys.exit(1)

        print("[INFO] Backend is reachable.")
        if os.path.exists(OPENENV_PATH):
            print(f"[INFO] OpenEnv spec: {OPENENV_PATH}")

        client = get_llm_client()
        if client:
            print("[INFO] LLM client initialized - will use LLM decision-making")
        else:
            print("[INFO] Using deterministic policy (no LLM available)")
        
        scores_by_scenario: Dict[str, float] = {}

        # Run all scenarios
        for scenario in SCENARIOS:
            try:
                print(f"[SCENARIO] Starting {scenario}...")
                score, _history = run_episode(scenario, client)
                scores_by_scenario[scenario] = clamp_open_interval(score)
                print(f"[SCENARIO] Completed {scenario}: score={score:.4f}")
            except Exception as exc:
                print(f"[ERROR] scenario={scenario} failed: {exc}", file=sys.stderr)
                scores_by_scenario[scenario] = DEFAULT_SAFE_SCORE
                print(f"[WARNING] Using default safe score for {scenario}")

        # Build and validate final payload
        payload = build_final_payload(scores_by_scenario)
        print("[FINAL] submission payload:")
        print(json.dumps(payload, indent=2, sort_keys=True))

    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] inference failed: {exc}", file=sys.stderr)
        fallback = build_final_payload({})
        print("[FINAL] fallback payload (error occurred during execution):")
        print(json.dumps(fallback, indent=2, sort_keys=True))
        sys.exit(1)


if __name__ == "__main__":
    main()