"""
Graders for HospitalBedAllocation-v1

Each grader runs a full episode and returns a score in [0.0, 1.0].
Partial credit is awarded at each threshold so agents get a learning signal
even when far from optimal.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

from environment import HospitalBedEnv, PatientPriority, PatientStatus


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """
    Runs the environment with a provided policy and returns a score + report.
    """
    level: str = "base"
    max_score: float = 1.0

    def __init__(self, seed: int = 42):
        self.seed = seed

    def grade(self, policy: Callable[[Dict], Dict]) -> Tuple[float, Dict]:
        """
        Args:
            policy: callable (observation) -> action dict

        Returns:
            (score in [0, 1], report dict with sub-scores)
        """
        env = HospitalBedEnv(task_level=self.level, seed=self.seed)
        obs = env.reset(seed=self.seed)
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        full_state = env.state()
        score, report = self._compute_score(full_state, total_reward)
        report["total_steps"] = step
        report["total_reward"] = total_reward
        report["level"] = self.level
        return score, report

    def _compute_score(self, state: Dict, total_reward: float) -> Tuple[float, Dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Easy grader  (target score ≥ 0.5 for a decent policy)
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Easy task: 30 steps, no surges, no deterioration.
    Score based on:
        40% — discharge rate  (discharged / total spawned)
        40% — no deteriorations
        20% — utilization in sweet spot (70-90%)
    """
    level = "easy"

    def _compute_score(self, state: Dict, total_reward: float) -> Tuple[float, Dict]:
        patients = state["patients"]
        total = max(1, state["step_stats"][-1]["admitted"] + len(state["discharged"]) + 1)
        discharged = len(state["discharged"])
        deteriorated = sum(
            1 for p in patients.values()
            if p["status"] == PatientStatus.DETERIORATED.value
        )

        discharge_rate = min(1.0, discharged / max(1, len(state["discharged"]) + len(state["waiting_queue"]) + len([
            p for p in patients.values() if p["status"] == PatientStatus.ADMITTED.value
        ])))
        no_deterirate_score = 1.0 if deteriorated == 0 else max(0.0, 1.0 - deteriorated * 0.15)

        # Utilization: average across all steps
        util_scores = []
        for ss in state["step_stats"]:
            # approximate: admitted / 56 total beds
            u = ss["admitted"] / 56
            if 0.70 <= u <= 0.90:
                util_scores.append(1.0)
            elif u < 0.70:
                util_scores.append(u / 0.70)
            else:
                util_scores.append(max(0.0, 1.0 - (u - 0.90) * 5))
        util_score = sum(util_scores) / max(1, len(util_scores))

        score = 0.40 * discharge_rate + 0.40 * no_deterirate_score + 0.20 * util_score
        report = {
            "discharge_rate": round(discharge_rate, 3),
            "no_deterioration_score": round(no_deterirate_score, 3),
            "utilization_score": round(util_score, 3),
            "deteriorated_count": deteriorated,
            "discharged_count": discharged,
        }
        return round(score, 4), report


# ---------------------------------------------------------------------------
# Medium grader  (target score ≥ 0.5 requires handling surges)
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Medium: 50 steps, surges, deterioration enabled, transfers available.
    Score:
        30% — discharge rate
        30% — critical/urgent patient handling (zero deteriorations for these)
        20% — surge response (# of urgent admitted within 2 steps of surge)
        20% — utilization efficiency
    """
    level = "medium"

    def grade(self, policy: Callable[[Dict], Dict]) -> Tuple[float, Dict]:
        env = HospitalBedEnv(task_level=self.level, seed=self.seed)
        obs = env.reset(seed=self.seed)
        done = False
        total_reward = 0.0
        step = 0

        surge_admission_windows: list = []   # (surge_step, admissions_next_2)
        surge_step = None
        post_surge_admitted: list = []

        while not done:
            action = policy(obs)
            prev_admitted = set(p["patient_id"] for p in obs.get("admitted", []))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if obs.get("surge_active") and surge_step is None:
                surge_step = step
                post_surge_admitted = []
            if surge_step is not None:
                new_admitted = set(p["patient_id"] for p in obs.get("admitted", []))
                newly = new_admitted - prev_admitted
                post_surge_admitted.extend(newly)
                if step >= surge_step + 2:
                    surge_admission_windows.append(len(post_surge_admitted))
                    surge_step = None

        full_state = env.state()
        score, report = self._compute_score(full_state, total_reward, surge_admission_windows)
        report["total_steps"] = step
        report["total_reward"] = total_reward
        report["level"] = self.level
        return score, report

    def _compute_score(self, state, total_reward, surge_windows=None) -> Tuple[float, Dict]:
        patients = state["patients"]
        discharged = len(state["discharged"])
        total_ever = state["step_stats"][-1].get("admitted", 1) + discharged
        discharge_rate = min(1.0, discharged / max(1, total_ever))

        # Critical/urgent deterioration
        hi_pri_deter = sum(
            1 for p in patients.values()
            if p["status"] == PatientStatus.DETERIORATED.value
            and p["priority"] in (PatientPriority.CRITICAL.value, PatientPriority.URGENT.value)
        )
        critical_score = max(0.0, 1.0 - hi_pri_deter * 0.25)

        # Surge response
        surge_score = 0.5   # default if no surges occurred
        if surge_windows:
            avg_surge_admit = sum(surge_windows) / len(surge_windows)
            surge_score = min(1.0, avg_surge_admit / 3.0)

        # Utilization
        util_scores = []
        for ss in state["step_stats"]:
            u = ss["admitted"] / 56
            if 0.70 <= u <= 0.90:
                util_scores.append(1.0)
            else:
                util_scores.append(max(0.0, 1.0 - abs(u - 0.80) * 3))
        util_score = sum(util_scores) / max(1, len(util_scores))

        score = (0.30 * discharge_rate + 0.30 * critical_score
                 + 0.20 * surge_score + 0.20 * util_score)
        return round(score, 4), {
            "discharge_rate": round(discharge_rate, 3),
            "critical_urgency_score": round(critical_score, 3),
            "surge_response_score": round(surge_score, 3),
            "utilization_score": round(util_score, 3),
            "hi_priority_deteriorated": hi_pri_deter,
            "surge_windows": surge_windows or [],
        }


# ---------------------------------------------------------------------------
# Hard grader  (requires sophisticated policy to score > 0.5)
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Hard: 80 steps, high surge probability, maintenance events, all actions.
    Score:
        25% — discharge rate
        25% — zero critical deteriorations
        20% — throughput efficiency (patients/step)
        15% — maintenance management (proper use, no abuse)
        15% — queue depth management (avg queue length below threshold)
    """
    level = "hard"

    def _compute_score(self, state, total_reward) -> Tuple[float, Dict]:
        patients = state["patients"]
        discharged = len(state["discharged"])
        steps = max(1, len(state["step_stats"]))

        # Discharge rate
        total_ever = sum(ss["admitted"] for ss in state["step_stats"][-1:]) + discharged
        discharge_rate = min(1.0, discharged / max(1, total_ever))

        # Critical deterioration (hard zero-tolerance)
        crit_deter = sum(
            1 for p in patients.values()
            if p["status"] == PatientStatus.DETERIORATED.value
            and p["priority"] == PatientPriority.CRITICAL.value
        )
        zero_crit_score = max(0.0, 1.0 - crit_deter * 0.5)

        # Throughput
        throughput = discharged / steps
        throughput_score = min(1.0, throughput / 0.5)  # 0.5 discharges/step = perfect

        # Maintenance management
        maint_patients = [p for p in patients.values() if "MAINT" in p["patient_id"]]
        abuse = sum(1 for p in patients.values()
                    if "MAINT" in p.get("patient_id", "") and p.get("steps_waiting", 0) > 0)
        maintenance_score = max(0.0, 1.0 - abuse * 0.2)

        # Queue depth
        avg_queue = sum(ss["waiting"] for ss in state["step_stats"]) / steps
        queue_score = max(0.0, 1.0 - avg_queue / 10.0)

        score = (0.25 * discharge_rate + 0.25 * zero_crit_score
                 + 0.20 * throughput_score + 0.15 * maintenance_score
                 + 0.15 * queue_score)
        return round(score, 4), {
            "discharge_rate": round(discharge_rate, 3),
            "zero_critical_deterioration_score": round(zero_crit_score, 3),
            "throughput_score": round(throughput_score, 3),
            "maintenance_score": round(maintenance_score, 3),
            "queue_management_score": round(queue_score, 3),
            "avg_queue_depth": round(avg_queue, 2),
            "critical_deteriorated": crit_deter,
            "throughput_per_step": round(throughput, 3),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}


def get_grader(level: str, seed: int = 42) -> BaseGrader:
    return GRADERS[level](seed=seed)
