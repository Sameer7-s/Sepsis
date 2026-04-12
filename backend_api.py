"""
Sepsis Management OpenEnv Backend
FastAPI app only — never starts uvicorn.
"""

import json
import math
import os
import threading
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR  = os.path.join(BASE_DIR, "scenarios")

PULSE_PRESSURE_MMHG: float = 40.0
_SBP_OFFSET = 2.0 * PULSE_PRESSURE_MMHG / 3.0
_DBP_OFFSET = PULSE_PRESSURE_MMHG / 3.0

SUPPORTED_SCENARIOS = {"early_sepsis", "severe_sepsis", "septic_shock"}

MIN_SAFE_SCORE    = 0.05
MAX_SAFE_SCORE    = 0.95
DEFAULT_SAFE_SCORE = 0.50

_BUILTIN_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "early_sepsis": {
        "severity": "early", "max_steps": 20,
        "initial_state": {
            "heart_rate": 102.0, "mean_arterial_pressure": 78.0,
            "respiratory_rate": 20.0, "oxygen_saturation": 95.0,
            "temperature": 38.4, "lactate": 1.8, "sofa_score": 1.0,
        },
        "target_state": {
            "heart_rate": 80.0, "mean_arterial_pressure": 85.0,
            "respiratory_rate": 16.0, "oxygen_saturation": 98.0,
            "temperature": 37.0, "lactate": 1.0, "sofa_score": 0.0,
        },
        "requirements": {
            "essential_actions": ["administer_antibiotics"],
            "needs_vasopressors": False, "needs_source_control": False,
        },
        "progression": {
            "temp_increase_if_no_antibiotics": 0.15,
            "hr_increase_if_no_antibiotics": 2.0,
            "rr_increase_if_no_antibiotics": 0.5,
            "lactate_increase_if_no_antibiotics": 0.1,
            "sofa_increase_if_no_antibiotics": 0.1,
            "temp_increase_if_no_source_control": 0.0,
            "lactate_increase_if_no_source_control": 0.0,
            "map_drop_if_no_vasopressors": 0.0,
            "spo2_drop_if_no_vasopressors": 0.0,
        },
        "termination": {
            "stable_map_min": 70.0, "stable_spo2_min": 94.0,
            "stable_lactate_max": 2.0, "stable_sofa_max": 2.0,
            "critical_map_below": 50.0, "critical_spo2_below": 80.0,
            "critical_lactate_above": 8.0,
        },
        "reward_normalization": {"max_cumulative_reward": 1.6, "min_cumulative_reward": -1.0},
    },
    "severe_sepsis": {
        "severity": "severe", "max_steps": 20,
        "initial_state": {
            "heart_rate": 118.0, "mean_arterial_pressure": 68.0,
            "respiratory_rate": 26.0, "oxygen_saturation": 91.0,
            "temperature": 39.1, "lactate": 2.8, "sofa_score": 4.0,
        },
        "target_state": {
            "heart_rate": 80.0, "mean_arterial_pressure": 80.0,
            "respiratory_rate": 16.0, "oxygen_saturation": 97.0,
            "temperature": 37.0, "lactate": 1.2, "sofa_score": 1.0,
        },
        "requirements": {
            "essential_actions": ["administer_antibiotics", "give_fluids"],
            "needs_vasopressors": False, "needs_source_control": True,
        },
        "progression": {
            "temp_increase_if_no_antibiotics": 0.20,
            "hr_increase_if_no_antibiotics": 3.0,
            "rr_increase_if_no_antibiotics": 0.8,
            "lactate_increase_if_no_antibiotics": 0.15,
            "sofa_increase_if_no_antibiotics": 0.15,
            "temp_increase_if_no_source_control": 0.1,
            "lactate_increase_if_no_source_control": 0.1,
            "map_drop_if_no_vasopressors": 0.0,
            "spo2_drop_if_no_vasopressors": 0.0,
        },
        "termination": {
            "stable_map_min": 70.0, "stable_spo2_min": 94.0,
            "stable_lactate_max": 2.0, "stable_sofa_max": 3.0,
            "critical_map_below": 50.0, "critical_spo2_below": 78.0,
            "critical_lactate_above": 9.0,
        },
        "reward_normalization": {"max_cumulative_reward": 1.8, "min_cumulative_reward": -1.2},
    },
    "septic_shock": {
        "severity": "shock", "max_steps": 20,
        "initial_state": {
            "heart_rate": 132.0, "mean_arterial_pressure": 55.0,
            "respiratory_rate": 30.0, "oxygen_saturation": 86.0,
            "temperature": 39.6, "lactate": 4.5, "sofa_score": 7.0,
        },
        "target_state": {
            "heart_rate": 85.0, "mean_arterial_pressure": 75.0,
            "respiratory_rate": 18.0, "oxygen_saturation": 96.0,
            "temperature": 37.2, "lactate": 1.5, "sofa_score": 2.0,
        },
        "requirements": {
            "essential_actions": ["administer_antibiotics", "give_fluids", "start_vasopressors"],
            "needs_vasopressors": True, "needs_source_control": True,
        },
        "progression": {
            "temp_increase_if_no_antibiotics": 0.25,
            "hr_increase_if_no_antibiotics": 4.0,
            "rr_increase_if_no_antibiotics": 1.0,
            "lactate_increase_if_no_antibiotics": 0.20,
            "sofa_increase_if_no_antibiotics": 0.20,
            "temp_increase_if_no_source_control": 0.15,
            "lactate_increase_if_no_source_control": 0.15,
            "map_drop_if_no_vasopressors": 3.0,
            "spo2_drop_if_no_vasopressors": 1.5,
        },
        "termination": {
            "stable_map_min": 65.0, "stable_spo2_min": 92.0,
            "stable_lactate_max": 2.5, "stable_sofa_max": 5.0,
            "critical_map_below": 40.0, "critical_spo2_below": 75.0,
            "critical_lactate_above": 10.0,
        },
        "reward_normalization": {"max_cumulative_reward": 2.0, "min_cumulative_reward": -1.5},
    },
}

_SEVERITY_MAP = {
    "early_sepsis": "early",
    "severe_sepsis": "severe",
    "septic_shock": "shock",
}


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


def clamp_open_interval(value: float, low: float = MIN_SAFE_SCORE, high: float = MAX_SAFE_SCORE) -> float:
    x = safe_float(value, DEFAULT_SAFE_SCORE)
    if x <= 0.0: return low
    if x >= 1.0: return high
    return max(low, min(high, x))


def round_safe_score(value: float) -> float:
    clamped = clamp_open_interval(value)
    rounded = round(clamped, 4)
    if rounded <= 0.0: rounded = MIN_SAFE_SCORE
    if rounded >= 1.0: rounded = MAX_SAFE_SCORE
    return float(rounded)


def normalize_scenario_name(name: str) -> str:
    value = (name or "").strip().lower()
    aliases = {
        "mild": "early_sepsis", "mild_sepsis": "early_sepsis",
        "early": "early_sepsis", "early_sepsis": "early_sepsis",
        "severe": "severe_sepsis", "severe_sepsis": "severe_sepsis",
        "shock": "septic_shock", "septic_shock": "septic_shock",
    }
    return aliases.get(value, value)


def load_scenario(name: str) -> Dict[str, Any]:
    normalized = normalize_scenario_name(name)
    if normalized not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unknown scenario: '{name}'")
    path = os.path.join(SCENARIO_DIR, f"{normalized}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            scenario["name"] = normalized
            return scenario
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc} — using built-in.")
    scenario = deepcopy(_BUILTIN_SCENARIOS[normalized])
    scenario["name"] = normalized
    return scenario


class Observation(BaseModel):
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    mean_arterial_pressure: float
    respiratory_rate: float
    oxygen_saturation: float
    temperature: float
    lactate: float
    sofa_score: float


ActionName = Literal[
    "observe", "administer_antibiotics", "give_fluids",
    "oxygen_therapy", "start_vasopressors", "perform_source_control", "noop",
]


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
        "observe", "administer_antibiotics", "give_fluids",
        "oxygen_therapy", "start_vasopressors", "perform_source_control", "noop",
    ]

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _stability_score(self, state: Dict[str, float], scenario: Dict[str, Any]) -> float:
        target = scenario["target_state"]
        weights = {"heart_rate": 0.12, "mean_arterial_pressure": 0.18, "respiratory_rate": 0.10,
                   "oxygen_saturation": 0.12, "temperature": 0.08, "lactate": 0.22, "sofa_score": 0.18}
        bands   = {"heart_rate": 80.0, "mean_arterial_pressure": 30.0, "respiratory_rate": 18.0,
                   "oxygen_saturation": 6.0, "temperature": 3.0, "lactate": 6.0, "sofa_score": 10.0}
        score = 0.0
        for key, weight in weights.items():
            diff = abs(state.get(key, target[key]) - target[key])
            score += weight * clamp(1.0 - diff / bands[key], 0.0, 1.0)
        return clamp_open_interval(score)

    def _apply_progression(self, state, hidden, scenario):
        prog = scenario["progression"]
        req  = scenario["requirements"]
        infection_active    = not hidden["interventions"]["administer_antibiotics"]
        source_uncontrolled = req.get("needs_source_control", False) and not hidden["interventions"]["perform_source_control"]
        shock_uncontrolled  = (state.get("mean_arterial_pressure", 65) < 65 and
                               req.get("needs_vasopressors", False) and
                               not hidden["interventions"]["start_vasopressors"])
        mult = 1.0 + 0.12 * max(0, hidden["delay_steps"])
        if infection_active:
            state["temperature"]      += prog.get("temp_increase_if_no_antibiotics", 0.0) * mult
            state["heart_rate"]       += prog.get("hr_increase_if_no_antibiotics", 0.0) * mult
            state["respiratory_rate"] += prog.get("rr_increase_if_no_antibiotics", 0.0) * mult
            state["lactate"]          += prog.get("lactate_increase_if_no_antibiotics", 0.0) * mult
            state["sofa_score"]       += prog.get("sofa_increase_if_no_antibiotics", 0.0) * mult
        if source_uncontrolled:
            state["temperature"] += prog.get("temp_increase_if_no_source_control", 0.0)
            state["lactate"]     += prog.get("lactate_increase_if_no_source_control", 0.0)
        if shock_uncontrolled:
            state["mean_arterial_pressure"] -= prog.get("map_drop_if_no_vasopressors", 0.0)
            state["oxygen_saturation"]      -= prog.get("spo2_drop_if_no_vasopressors", 0.0)
        if state.get("oxygen_saturation", 95) < 93 and not hidden["interventions"]["oxygen_therapy"]:
            state["heart_rate"] += 1.0; state["respiratory_rate"] += 1.0

    def _apply_action(self, action: str, episode: Dict[str, Any]) -> float:
        scenario = episode["scenario"]; state = episode["state"]; hidden = episode["hidden"]
        req = scenario["requirements"]
        step_reward = -0.01
        recent_actions = [item["action"] for item in episode["history"][-2:]]
        if action == "observe":
            step_reward += 0.01
        elif action == "noop":
            step_reward -= 0.05
        elif action == "administer_antibiotics":
            if not hidden["interventions"]["administer_antibiotics"]:
                hidden["interventions"]["administer_antibiotics"] = True
                state["temperature"] -= 0.3; state["heart_rate"] -= 4.0
                state["lactate"] -= 0.4; state["sofa_score"] -= 0.3
                step_reward += 0.15
            else:
                step_reward -= 0.03
        elif action == "give_fluids":
            if state.get("mean_arterial_pressure", 65) < 70 or state.get("lactate", 2) > 2.2:
                state["mean_arterial_pressure"] += 6.0; state["heart_rate"] -= 3.0
                state["lactate"] -= 0.5; step_reward += 0.10
                hidden["interventions"]["give_fluids"] = True
            else:
                state["oxygen_saturation"] -= 0.5; step_reward -= 0.08
        elif action == "oxygen_therapy":
            if state.get("oxygen_saturation", 95) < 94 or state.get("respiratory_rate", 16) > 22:
                state["oxygen_saturation"] += 3.0; state["respiratory_rate"] -= 2.0
                step_reward += 0.10; hidden["interventions"]["oxygen_therapy"] = True
            else:
                step_reward -= 0.03
        elif action == "start_vasopressors":
            if req.get("needs_vasopressors", False) and state.get("mean_arterial_pressure", 65) < 65:
                state["mean_arterial_pressure"] += 10.0; state["lactate"] -= 0.4
                step_reward += 0.14; hidden["interventions"]["start_vasopressors"] = True
            else:
                state["lactate"] += 0.3; step_reward -= 0.08
        elif action == "perform_source_control":
            if req.get("needs_source_control", False):
                if not hidden["interventions"]["perform_source_control"]:
                    hidden["interventions"]["perform_source_control"] = True
                    state["temperature"] -= 0.2; state["lactate"] -= 0.3
                    state["sofa_score"] -= 0.2; step_reward += 0.12
                else:
                    step_reward -= 0.03
            else:
                step_reward -= 0.08
        else:
            step_reward -= 0.05
        if len(recent_actions) == 2 and recent_actions[0] == recent_actions[1] == action and action != "observe":
            step_reward -= 0.04
        if hidden["interventions"]["administer_antibiotics"]:
            state["temperature"] -= 0.10; state["lactate"] -= 0.15; state["sofa_score"] -= 0.05
        if hidden["interventions"]["oxygen_therapy"] and state.get("oxygen_saturation", 95) < 97:
            state["oxygen_saturation"] += 1.0
        if hidden["interventions"]["start_vasopressors"] and state.get("mean_arterial_pressure", 65) < 75:
            state["mean_arterial_pressure"] += 2.0
        return float(step_reward)

    def _normalize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        norm_map = clamp(safe_float(state.get("mean_arterial_pressure", 65.0), 65.0), 35, 140)
        return {
            "heart_rate":             round(clamp(safe_float(state.get("heart_rate", 100.0), 100.0), 40, 180), 2),
            "systolic_bp":            round(clamp(norm_map + _SBP_OFFSET, 60, 220), 2),
            "diastolic_bp":           round(clamp(norm_map - _DBP_OFFSET, 30, 140), 2),
            "mean_arterial_pressure": round(norm_map, 2),
            "respiratory_rate":       round(clamp(safe_float(state.get("respiratory_rate", 20.0), 20.0), 8, 45), 2),
            "oxygen_saturation":      round(clamp(safe_float(state.get("oxygen_saturation", 95.0), 95.0), 70, 100), 2),
            "temperature":            round(clamp(safe_float(state.get("temperature", 37.0), 37.0), 34, 42.5), 2),
            "lactate":                round(clamp(safe_float(state.get("lactate", 2.0), 2.0), 0.5, 12), 2),
            "sofa_score":             round(clamp(safe_float(state.get("sofa_score", 2.0), 2.0), 0, 24), 2),
        }

    def _state_payload(self, episode: Dict[str, Any]) -> Dict[str, Any]:
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
        initial  = deepcopy(scenario["initial_state"])
        map_val  = safe_float(initial.get("mean_arterial_pressure", 65.0), 65.0)
        initial["systolic_bp"]  = map_val + _SBP_OFFSET
        initial["diastolic_bp"] = map_val - _DBP_OFFSET
        episode_id        = str(uuid.uuid4())
        initial_stability = self._stability_score(initial, scenario)
        episode = {
            "episode_id": episode_id, "scenario": scenario, "state": initial,
            "step_count": 0, "done": False, "cumulative_reward": 0.0,
            "normalized_score": clamp_open_interval(initial_stability),
            "previous_stability": initial_stability, "history": [],
            "hidden": {"delay_steps": 0, "interventions": {a: False for a in self.ACTIONS}},
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
            scenario = episode["scenario"]; state = episode["state"]; hidden = episode["hidden"]
            reward = safe_float(self._apply_action(action, episode), 0.0)
            self._apply_progression(state, hidden, scenario)
            map_val = safe_float(state.get("mean_arterial_pressure", 65.0), 65.0)
            state["systolic_bp"]  = map_val + _SBP_OFFSET
            state["diastolic_bp"] = map_val - _DBP_OFFSET
            stability = self._stability_score(state, scenario)
            delta = stability - safe_float(episode["previous_stability"], stability)
            reward += 0.45 * delta
            episode["previous_stability"] = stability
            essential_actions  = scenario["requirements"].get("essential_actions", [])
            missing_essentials = [a for a in essential_actions if not hidden["interventions"].get(a, False)]
            if missing_essentials:
                hidden["delay_steps"] += 1
                reward -= 0.02 * len(missing_essentials)
            normalized = self._normalize_state(state)
            episode["state"] = normalized; episode["step_count"] += 1
            episode["cumulative_reward"] = safe_float(episode["cumulative_reward"], 0.0) + reward
            term = scenario["termination"]
            stable = (normalized["mean_arterial_pressure"] >= term["stable_map_min"] and
                      normalized["oxygen_saturation"] >= term["stable_spo2_min"] and
                      normalized["lactate"] <= term["stable_lactate_max"] and
                      normalized["sofa_score"] <= term["stable_sofa_max"])
            deteriorated = (normalized["mean_arterial_pressure"] <= term["critical_map_below"] or
                            normalized["oxygen_saturation"] <= term["critical_spo2_below"] or
                            normalized["lactate"] >= term["critical_lactate_above"])
            if stable or deteriorated or episode["step_count"] >= scenario["max_steps"]:
                episode["done"] = True
            rnorm = scenario.get("reward_normalization", {})
            max_r = safe_float(rnorm.get("max_cumulative_reward", 1.6), 1.6)
            min_r = safe_float(rnorm.get("min_cumulative_reward", -1.0), -1.0)
            raw   = safe_float(episode["cumulative_reward"], 0.0)
            episode["normalized_score"] = clamp_open_interval((raw - min_r) / max(max_r - min_r, 1e-8))
            step_record = {"step": int(episode["step_count"]), "action": action,
                           "reward": round(reward, 4),
                           "normalized_score": round_safe_score(episode["normalized_score"])}
            episode["history"].append(step_record)
            message = ("stable" if stable else "deteriorated" if deteriorated else
                       "max_steps_reached" if episode["done"] else "in_progress")
            response = self._state_payload(episode)
            response.update({"reward": round(reward, 4), "message": message})
            return response


env = SepsisEnvironment()

app = FastAPI(title="Sepsis Management OpenEnv Backend", version="1.7.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "Sepsis AI Agent API is running", "endpoints": ["/health", "/reset", "/step", "/state"]}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset", response_model=EpisodeStateResponse)
async def reset_endpoint(request: Request) -> EpisodeStateResponse:
    data: Dict[str, Any] = {}
    try:
        body = await request.body()
        if body:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                data = parsed
    except Exception:
        pass
    raw_scenario = data.get("scenario", "early_sepsis")
    if not isinstance(raw_scenario, str) or not raw_scenario.strip():
        raw_scenario = "early_sepsis"
    normalized_name = normalize_scenario_name(raw_scenario)
    if normalized_name not in SUPPORTED_SCENARIOS:
        normalized_name = "early_sepsis"
    try:
        payload = env.reset(normalized_name)
    except Exception as exc:
        print(f"[ERROR] env.reset failed: {exc}")
        # FIX: register fallback episode so /step works
        payload = _create_registered_fallback(normalized_name)
    return EpisodeStateResponse(**payload)


def _create_registered_fallback(scenario_name: str) -> Dict[str, Any]:
    """Emergency fallback that REGISTERS the episode so /step doesn't 404."""
    scenario   = deepcopy(_BUILTIN_SCENARIOS.get(scenario_name, _BUILTIN_SCENARIOS["early_sepsis"]))
    scenario["name"] = scenario_name
    initial    = deepcopy(scenario["initial_state"])
    map_val    = safe_float(initial.get("mean_arterial_pressure", 65.0), 65.0)
    initial["systolic_bp"]  = map_val + _SBP_OFFSET
    initial["diastolic_bp"] = map_val - _DBP_OFFSET
    episode_id = str(uuid.uuid4())
    episode = {
        "episode_id": episode_id, "scenario": scenario, "state": initial,
        "step_count": 0, "done": False, "cumulative_reward": 0.0,
        "normalized_score": DEFAULT_SAFE_SCORE, "previous_stability": DEFAULT_SAFE_SCORE,
        "history": [],
        "hidden": {"delay_steps": 0, "interventions": {a: False for a in SepsisEnvironment.ACTIONS}},
    }
    with env.lock:
        env.sessions[episode_id] = episode
    return env._state_payload(episode)


@app.get("/state", response_model=EpisodeStateResponse)
def get_state(episode_id: str) -> EpisodeStateResponse:
    try:
        return EpisodeStateResponse(**env.state(episode_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"state fetch failed: {exc!s}") from exc


@app.post("/step", response_model=StepResponse)
def step_endpoint(request: StepRequest) -> StepResponse:
    try:
        return StepResponse(**env.step(request.episode_id, request.action))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step failed: {exc!s}") from exc