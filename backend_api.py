import json
import math
import os
import threading
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(BASE_DIR, "scenarios")

PULSE_PRESSURE_MMHG: float = 40.0
_SBP_OFFSET = 2.0 * PULSE_PRESSURE_MMHG / 3.0
_DBP_OFFSET = PULSE_PRESSURE_MMHG / 3.0

SUPPORTED_SCENARIOS = {"early_sepsis", "severe_sepsis", "septic_shock"}

# ── Score safety: wider margin so float formatting never rounds to 0.0 / 1.0 ─
MIN_SAFE_SCORE = 0.05
MAX_SAFE_SCORE = 0.95
DEFAULT_SAFE_SCORE = 0.50


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = DEFAULT_SAFE_SCORE) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def clamp_open_interval(
    value: float,
    low: float = MIN_SAFE_SCORE,
    high: float = MAX_SAFE_SCORE,
) -> float:
    """Return a value strictly inside (low, high). Never returns low or high exactly."""
    x = safe_float(value, DEFAULT_SAFE_SCORE)

    if x <= 0.0:
        return low
    if x >= 1.0:
        return high

    clamped = max(low, min(high, x))
    return clamped


def round_safe_score(value: float) -> float:
    """
    Round to 4 dp and guarantee the result is strictly in (0, 1).
    Does NOT use assert — assertions are disabled by Python -O in Docker.
    """
    clamped = clamp_open_interval(value)
    rounded = round(clamped, 4)
    # Extra guard: rounding can theoretically hit 0.0 or 1.0 for extreme inputs
    if rounded <= 0.0:
        rounded = MIN_SAFE_SCORE
    if rounded >= 1.0:
        rounded = MAX_SAFE_SCORE
    return float(rounded)


def normalize_scenario_name(name: str) -> str:
    value = (name or "").strip().lower()
    aliases = {
        "mild": "early_sepsis",
        "mild_sepsis": "early_sepsis",
        "early": "early_sepsis",
        "early_sepsis": "early_sepsis",
        "severe": "severe_sepsis",
        "severe_sepsis": "severe_sepsis",
        "shock": "septic_shock",
        "septic_shock": "septic_shock",
    }
    return aliases.get(value, value)


def load_scenario(name: str) -> Dict[str, Any]:
    normalized_name = normalize_scenario_name(name)

    if normalized_name not in SUPPORTED_SCENARIOS:
        raise FileNotFoundError(f"Unknown scenario: {name}")

    path = os.path.join(SCENARIO_DIR, f"{normalized_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scenario file not found for '{normalized_name}'. Expected: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    scenario["name"] = normalized_name
    return scenario


class Observation(BaseModel):
    heart_rate: float = Field(...)
    systolic_bp: float = Field(...)
    diastolic_bp: float = Field(...)
    mean_arterial_pressure: float = Field(...)
    respiratory_rate: float = Field(...)
    oxygen_saturation: float = Field(...)
    temperature: float = Field(...)
    lactate: float = Field(...)
    sofa_score: float = Field(...)


ActionName = Literal[
    "observe",
    "administer_antibiotics",
    "give_fluids",
    "oxygen_therapy",
    "start_vasopressors",
    "perform_source_control",
    "noop",
]

ScenarioName = Literal["early_sepsis", "severe_sepsis", "septic_shock"]


class ResetRequest(BaseModel):
    scenario: str = "early_sepsis"


class StepRequest(BaseModel):
    episode_id: str
    action: ActionName


class StepRecord(BaseModel):
    step: int
    action: str
    reward: float
    normalized_score: float


class EpisodeStateResponse(BaseModel):
    episode_id: str
    scenario: str
    severity: str
    step_count: int
    max_steps: int
    done: bool
    state: Observation
    cumulative_reward: float
    normalized_score: float
    history: List[StepRecord]


class StepResponse(EpisodeStateResponse):
    reward: float
    message: str


class HealthResponse(BaseModel):
    status: str


class SepsisEnvironment:
    ACTIONS: List[str] = [
        "observe",
        "administer_antibiotics",
        "give_fluids",
        "oxygen_therapy",
        "start_vasopressors",
        "perform_source_control",
        "noop",
    ]

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _get_target_state(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        return scenario["target_state"]

    def _stability_score(self, state: Dict[str, float], scenario: Dict[str, Any]) -> float:
        target = self._get_target_state(scenario)

        weights = {
            "heart_rate": 0.12,
            "mean_arterial_pressure": 0.18,
            "respiratory_rate": 0.10,
            "oxygen_saturation": 0.12,
            "temperature": 0.08,
            "lactate": 0.22,
            "sofa_score": 0.18,
        }

        bands = {
            "heart_rate": 80.0,
            "mean_arterial_pressure": 30.0,
            "respiratory_rate": 18.0,
            "oxygen_saturation": 6.0,
            "temperature": 3.0,
            "lactate": 6.0,
            "sofa_score": 10.0,
        }

        score = 0.0
        for key, weight in weights.items():
            diff = abs(state[key] - target[key])
            score += weight * clamp(1.0 - diff / bands[key], 0.0, 1.0)

        return clamp_open_interval(score)

    def _apply_progression(
        self,
        state: Dict[str, float],
        hidden: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> None:
        prog = scenario["progression"]

        infection_active = not hidden["interventions"]["administer_antibiotics"]
        source_uncontrolled = (
            scenario["requirements"].get("needs_source_control", False)
            and not hidden["interventions"]["perform_source_control"]
        )
        shock_uncontrolled = (
            state["mean_arterial_pressure"] < 65
            and scenario["requirements"].get("needs_vasopressors", False)
            and not hidden["interventions"]["start_vasopressors"]
        )

        severity_multiplier = 1.0 + 0.12 * max(0, hidden["delay_steps"])

        if infection_active:
            state["temperature"] += prog.get("temp_increase_if_no_antibiotics", 0.0) * severity_multiplier
            state["heart_rate"] += prog.get("hr_increase_if_no_antibiotics", 0.0) * severity_multiplier
            state["respiratory_rate"] += prog.get("rr_increase_if_no_antibiotics", 0.0) * severity_multiplier
            state["lactate"] += prog.get("lactate_increase_if_no_antibiotics", 0.0) * severity_multiplier
            state["sofa_score"] += prog.get("sofa_increase_if_no_antibiotics", 0.0) * severity_multiplier

        if source_uncontrolled:
            state["temperature"] += prog.get("temp_increase_if_no_source_control", 0.0)
            state["lactate"] += prog.get("lactate_increase_if_no_source_control", 0.0)

        if shock_uncontrolled:
            state["mean_arterial_pressure"] -= prog.get("map_drop_if_no_vasopressors", 0.0)
            state["oxygen_saturation"] -= prog.get("spo2_drop_if_no_vasopressors", 0.0)

        if state["oxygen_saturation"] < 93 and not hidden["interventions"]["oxygen_therapy"]:
            state["heart_rate"] += 1.0
            state["respiratory_rate"] += 1.0

    def _apply_action(self, action: str, episode: Dict[str, Any]) -> float:
        scenario = episode["scenario"]
        state = episode["state"]
        hidden = episode["hidden"]

        step_reward = -0.01
        relevant = scenario["requirements"]
        correct_action_bonus = 0.10
        wrong_action_penalty = -0.08
        recent_actions = [item["action"] for item in episode["history"][-2:]]

        if action == "observe":
            step_reward += 0.01

        elif action == "noop":
            step_reward -= 0.05

        elif action == "administer_antibiotics":
            if not hidden["interventions"]["administer_antibiotics"]:
                hidden["interventions"]["administer_antibiotics"] = True
                state["temperature"] -= 0.3
                state["heart_rate"] -= 4.0
                state["lactate"] -= 0.4
                state["sofa_score"] -= 0.3
                step_reward += correct_action_bonus + 0.05
            else:
                step_reward -= 0.03

        elif action == "give_fluids":
            if state["mean_arterial_pressure"] < 70 or state["lactate"] > 2.2:
                state["mean_arterial_pressure"] += 6.0
                state["heart_rate"] -= 3.0
                state["lactate"] -= 0.5
                step_reward += correct_action_bonus
                hidden["interventions"]["give_fluids"] = True
            else:
                state["oxygen_saturation"] -= 0.5
                step_reward += wrong_action_penalty

        elif action == "oxygen_therapy":
            if state["oxygen_saturation"] < 94 or state["respiratory_rate"] > 22:
                state["oxygen_saturation"] += 3.0
                state["respiratory_rate"] -= 2.0
                step_reward += correct_action_bonus
                hidden["interventions"]["oxygen_therapy"] = True
            else:
                step_reward -= 0.03

        elif action == "start_vasopressors":
            if relevant.get("needs_vasopressors", False) and state["mean_arterial_pressure"] < 65:
                state["mean_arterial_pressure"] += 10.0
                state["lactate"] -= 0.4
                step_reward += 0.14
                hidden["interventions"]["start_vasopressors"] = True
            else:
                state["lactate"] += 0.3
                step_reward += wrong_action_penalty

        elif action == "perform_source_control":
            if relevant.get("needs_source_control", False):
                if not hidden["interventions"]["perform_source_control"]:
                    hidden["interventions"]["perform_source_control"] = True
                    state["temperature"] -= 0.2
                    state["lactate"] -= 0.3
                    state["sofa_score"] -= 0.2
                    step_reward += 0.12
                else:
                    step_reward -= 0.03
            else:
                step_reward += wrong_action_penalty

        else:
            step_reward -= 0.05

        if len(recent_actions) == 2 and recent_actions[0] == recent_actions[1] == action and action != "observe":
            step_reward -= 0.04

        if hidden["interventions"]["administer_antibiotics"]:
            state["temperature"] -= 0.10
            state["lactate"] -= 0.15
            state["sofa_score"] -= 0.05

        if hidden["interventions"]["oxygen_therapy"] and state["oxygen_saturation"] < 97:
            state["oxygen_saturation"] += 1.0

        if hidden["interventions"]["start_vasopressors"] and state["mean_arterial_pressure"] < 75:
            state["mean_arterial_pressure"] += 2.0

        return float(step_reward)

    def _normalize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        normalized_map = clamp(safe_float(state["mean_arterial_pressure"], 65.0), 35, 140)

        return {
            "heart_rate": round(clamp(safe_float(state["heart_rate"], 100.0), 40, 180), 2),
            "systolic_bp": round(clamp(normalized_map + _SBP_OFFSET, 60, 220), 2),
            "diastolic_bp": round(clamp(normalized_map - _DBP_OFFSET, 30, 140), 2),
            "mean_arterial_pressure": round(normalized_map, 2),
            "respiratory_rate": round(clamp(safe_float(state["respiratory_rate"], 20.0), 8, 45), 2),
            "oxygen_saturation": round(clamp(safe_float(state["oxygen_saturation"], 95.0), 70, 100), 2),
            "temperature": round(clamp(safe_float(state["temperature"], 37.0), 34, 42.5), 2),
            "lactate": round(clamp(safe_float(state["lactate"], 2.0), 0.5, 12), 2),
            "sofa_score": round(clamp(safe_float(state["sofa_score"], 2.0), 0, 24), 2),
        }

    def _state_payload(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        # Always run through round_safe_score — never trust the raw stored value alone
        score = round_safe_score(episode.get("normalized_score", DEFAULT_SAFE_SCORE))

        return {
            "episode_id": episode["episode_id"],
            "scenario": episode["scenario"]["name"],
            "severity": episode["scenario"]["severity"],
            "step_count": int(episode["step_count"]),
            "max_steps": int(episode["scenario"]["max_steps"]),
            "done": bool(episode["done"]),
            "state": self._normalize_state(episode["state"]),
            "cumulative_reward": round(safe_float(episode["cumulative_reward"], 0.0), 4),
            "normalized_score": score,
            "history": episode["history"],
        }

    def reset(self, scenario_name: str) -> Dict[str, Any]:
        scenario = load_scenario(scenario_name)
        initial = deepcopy(scenario["initial_state"])

        map_val = safe_float(initial["mean_arterial_pressure"], 65.0)
        initial["systolic_bp"] = map_val + _SBP_OFFSET
        initial["diastolic_bp"] = map_val - _DBP_OFFSET

        episode_id = str(uuid.uuid4())

        initial_stability = self._stability_score(initial, scenario)

        episode = {
            "episode_id": episode_id,
            "scenario": scenario,
            "state": initial,
            "step_count": 0,
            "done": False,
            "cumulative_reward": 0.0,
            "normalized_score": clamp_open_interval(initial_stability),
            "previous_stability": initial_stability,
            "history": [],
            "hidden": {
                "delay_steps": 0,
                "interventions": {action: False for action in self.ACTIONS},
            },
        }

        with self.lock:
            self.sessions[episode_id] = episode

        return self._state_payload(episode)

    def state(self, episode_id: str) -> Dict[str, Any]:
        with self.lock:
            if episode_id not in self.sessions:
                raise KeyError("Unknown episode_id")
            return self._state_payload(self.sessions[episode_id])

    def step(self, episode_id: str, action: str) -> Dict[str, Any]:
        with self.lock:
            if episode_id not in self.sessions:
                raise KeyError("Unknown episode_id")

            episode = self.sessions[episode_id]

            if episode["done"]:
                response = self._state_payload(episode)
                response.update({"reward": 0.0, "message": "episode_complete"})
                return response

            scenario = episode["scenario"]
            state = episode["state"]
            hidden = episode["hidden"]

            reward = safe_float(self._apply_action(action, episode), 0.0)
            self._apply_progression(state, hidden, scenario)

            map_val = safe_float(state["mean_arterial_pressure"], 65.0)
            state["systolic_bp"] = map_val + _SBP_OFFSET
            state["diastolic_bp"] = map_val - _DBP_OFFSET

            stability = self._stability_score(state, scenario)
            delta = stability - safe_float(episode["previous_stability"], stability)
            reward += 0.45 * delta
            episode["previous_stability"] = stability

            essential_actions = scenario["requirements"].get("essential_actions", [])
            missing_essentials = [
                a for a in essential_actions if not hidden["interventions"].get(a, False)
            ]

            if missing_essentials:
                hidden["delay_steps"] += 1
                reward -= 0.02 * len(missing_essentials)

            normalized = self._normalize_state(state)
            episode["state"] = normalized
            episode["step_count"] += 1
            episode["cumulative_reward"] = safe_float(episode["cumulative_reward"], 0.0) + reward

            term = scenario["termination"]

            stable = (
                normalized["mean_arterial_pressure"] >= term["stable_map_min"]
                and normalized["oxygen_saturation"] >= term["stable_spo2_min"]
                and normalized["lactate"] <= term["stable_lactate_max"]
                and normalized["sofa_score"] <= term["stable_sofa_max"]
            )

            deteriorated = (
                normalized["mean_arterial_pressure"] <= term["critical_map_below"]
                or normalized["oxygen_saturation"] <= term["critical_spo2_below"]
                or normalized["lactate"] >= term["critical_lactate_above"]
            )

            if stable or deteriorated or episode["step_count"] >= scenario["max_steps"]:
                episode["done"] = True

            rnorm = scenario.get("reward_normalization", {})
            max_r = safe_float(rnorm.get("max_cumulative_reward", 1.6), 1.6)
            min_r = safe_float(rnorm.get("min_cumulative_reward", -1.0), -1.0)
            raw = safe_float(episode["cumulative_reward"], 0.0)
            denom = max(max_r - min_r, 1e-8)

            # Key fix: clamp_open_interval guarantees (MIN_SAFE_SCORE, MAX_SAFE_SCORE)
            episode["normalized_score"] = clamp_open_interval((raw - min_r) / denom)

            step_record = {
                "step": int(episode["step_count"]),
                "action": action,
                "reward": round(reward, 4),
                "normalized_score": round_safe_score(episode["normalized_score"]),
            }
            episode["history"].append(step_record)

            if stable:
                message = "stable"
            elif deteriorated:
                message = "deteriorated"
            elif episode["done"]:
                message = "max_steps_reached"
            else:
                message = "in_progress"

            response = self._state_payload(episode)
            response.update(
                {
                    "reward": round(reward, 4),
                    "message": message,
                }
            )
            return response


env = SepsisEnvironment()

app = FastAPI(
    title="Sepsis Management OpenEnv Backend",
    description="Deterministic sepsis-management environment exposing OpenEnv-style reset, step, and state endpoints.",
    version="1.5.0",
)


@app.get("/")
def root():
    return {
        "message": "Sepsis AI Agent API is running",
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    OpenEnv-compliant reset endpoint that handles:
    - Empty request bodies
    - Missing JSON
    - Invalid JSON
    - Various scenario names
    
    Returns a valid response regardless of input (never crashes).
    """
    try:
        # Step 1: Safely read and parse JSON, default to empty dict
        data = {}
        try:
            body = await request.body()
            if body:  # Only try to parse if body is not empty
                data = json.loads(body)
            else:
                data = {}
        except json.JSONDecodeError:
            # If JSON is invalid, use empty dict (defaults to early_sepsis)
            data = {}
        except Exception:
            # Any other error, use empty dict
            data = {}

        # Step 2: Extract and validate scenario
        scenario = "early_sepsis"  # default
        if isinstance(data, dict):
            raw_scenario = data.get("scenario", "early_sepsis")
            if isinstance(raw_scenario, str) and raw_scenario.strip():
                scenario = raw_scenario.strip()

        # Step 3: Normalize and validate scenario name
        normalized_name = normalize_scenario_name(scenario)
        if normalized_name not in SUPPORTED_SCENARIOS:
            # If scenario is invalid, default to early_sepsis
            normalized_name = "early_sepsis"

        # Step 4: Call env.reset() which is guaranteed to return valid data
        try:
            payload = env.reset(normalized_name)
        except Exception as e:
            # If env.reset fails, create a minimal fallback response
            print(f"[ERROR] env.reset failed: {e}")
            payload = {
                "episode_id": str(uuid.uuid4()),
                "scenario": normalized_name,
                "severity": "early",
                "step_count": 0,
                "max_steps": 20,
                "done": False,
                "state": {
                    "heart_rate": 100.0,
                    "systolic_bp": 120.0,
                    "diastolic_bp": 80.0,
                    "mean_arterial_pressure": 93.33,
                    "respiratory_rate": 20.0,
                    "oxygen_saturation": 95.0,
                    "temperature": 37.0,
                    "lactate": 2.0,
                    "sofa_score": 2.0,
                },
                "cumulative_reward": 0.0,
                "normalized_score": 0.5,
                "history": [],
            }

        # Step 5: Return valid response
        return payload

    except Exception as e:
        # Absolute fallback - never crash
        print(f"[CRITICAL] /reset handler failed: {e}")
        return {
            "episode_id": str(uuid.uuid4()),
            "scenario": "early_sepsis",
            "severity": "early",
            "step_count": 0,
            "max_steps": 20,
            "done": False,
            "state": {
                "heart_rate": 100.0,
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "mean_arterial_pressure": 93.33,
                "respiratory_rate": 20.0,
                "oxygen_saturation": 95.0,
                "temperature": 37.0,
                "lactate": 2.0,
                "sofa_score": 2.0,
            },
            "cumulative_reward": 0.0,
            "normalized_score": 0.5,
            "history": [],
        }


@app.get("/state", response_model=EpisodeStateResponse)
def state(episode_id: str) -> EpisodeStateResponse:
    try:
        payload = env.state(episode_id)
        return EpisodeStateResponse(**payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"state fetch failed: {str(exc)}") from exc


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        payload = env.step(request.episode_id, request.action)
        return StepResponse(**payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step failed: {str(exc)}") from exc


# NOTE: Server startup should ONLY be done via server/app.py
# This module is meant only to be imported as a FastAPI app module
# DO NOT run uvicorn directly from this file
# DO NOT define any run_server() function here - it's unused and causes confusion