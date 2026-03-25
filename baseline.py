"""
Baseline inference script for HospitalBedAllocation-v1

Implements a rule-based greedy policy that provides a reproducible
non-trivial baseline score across all task levels.

Run:
    python baseline.py                     # all levels
    python baseline.py --level easy        # single level
    python baseline.py --seeds 42 123 999  # multiple seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from environment import (
    HospitalBedEnv,
    Department,
    PatientPriority,
    PatientStatus,
)
from graders import get_grader


# ---------------------------------------------------------------------------
# Greedy rule-based policy
# ---------------------------------------------------------------------------

class GreedyPolicy:
    """
    Priority-based greedy policy:
    1. Discharge any ready patients to free beds.
    2. Admit the highest-priority waiting patient to any compatible free bed.
    3. If no compatible free bed, try transfer to make room.
    4. Otherwise hold.
    """

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        beds = {b["bed_id"]: b for b in obs["beds"]}
        waiting = obs["waiting_queue"]
        admitted = obs["admitted"]

        # --- 1. Discharge ready patients ---
        for p in admitted:
            if p["status"] == "ready_discharge":
                return {
                    "action_type": "discharge",
                    "patient_id": p["patient_id"],
                }

        # --- 2. Admit highest-priority waiting patient ---
        sorted_waiting = sorted(
            waiting,
            key=lambda p: (p["priority"], p["steps_waiting"] * -1),  # priority ASC, wait DESC
        )

        for patient in sorted_waiting:
            dept = patient["required_dept"]
            # Find a free, compatible bed
            free_bed = self._find_free_bed(beds, dept, obs.get("task_level", "easy"))
            if free_bed:
                return {
                    "action_type": "admit",
                    "patient_id": patient["patient_id"],
                    "bed_id": free_bed,
                }

        # --- 3. Transfer if a patient is in wrong dept and better bed exists ---
        for p in admitted:
            if p["status"] == "admitted":
                correct_dept = p["required_dept"]
                current_bed = beds.get(p.get("assigned_bed", ""), {})
                if current_bed.get("department") != correct_dept:
                    free_bed = self._find_free_bed(beds, correct_dept)
                    if free_bed:
                        return {
                            "action_type": "transfer",
                            "patient_id": p["patient_id"],
                            "bed_id": free_bed,
                        }

        # --- 4. Hold ---
        return {"action_type": "hold"}

    def _find_free_bed(
        self,
        beds: Dict,
        required_dept: str,
        task_level: str = "easy",
    ) -> Optional[str]:
        # Primary: exact department match
        for bid, bed in beds.items():
            if (
                bed["department"] == required_dept
                and not bed["occupied"]
                and not bed["maintenance"]
            ):
                return bid

        # Overflow: GENERAL ↔ EMERGENCY
        overflow_map = {"GENERAL": ["EMERGENCY"], "EMERGENCY": ["GENERAL"]}
        for alt_dept in overflow_map.get(required_dept, []):
            for bid, bed in beds.items():
                if (
                    bed["department"] == alt_dept
                    and not bed["occupied"]
                    and not bed["maintenance"]
                ):
                    return bid
        return None


# ---------------------------------------------------------------------------
# Smarter policy for medium/hard (adds surge awareness)
# ---------------------------------------------------------------------------

class SurgeAwarePolicy(GreedyPolicy):
    """
    Extends greedy policy to prioritize critical/urgent during surges
    and proactively discharge elective patients to free capacity.
    """

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        beds = {b["bed_id"]: b for b in obs["beds"]}
        waiting = obs["waiting_queue"]
        admitted = obs["admitted"]
        surge = obs.get("surge_active", False)

        # During surge: proactively discharge non-critical ready patients first
        for p in admitted:
            if p["status"] == "ready_discharge":
                return {"action_type": "discharge", "patient_id": p["patient_id"]}

        # During surge: also consider discharging elective patients slightly early
        # to make room for critical/urgent arrivals
        if surge:
            for p in admitted:
                if (p["status"] == "admitted"
                        and p["priority"] == PatientPriority.ELECTIVE.value
                        and p.get("steps_admitted", 0) >= p.get("length_of_stay", 99) * 0.8):
                    return {"action_type": "discharge", "patient_id": p["patient_id"]}

        # Admit in strict priority order
        sorted_waiting = sorted(
            waiting,
            key=lambda p: (p["priority"], p["steps_waiting"] * -1),
        )
        for patient in sorted_waiting:
            dept = patient["required_dept"]
            free_bed = self._find_free_bed(beds, dept)
            if free_bed:
                return {
                    "action_type": "admit",
                    "patient_id": patient["patient_id"],
                    "bed_id": free_bed,
                }

        return {"action_type": "hold"}


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def evaluate(level: str, seeds: List[int], verbose: bool = True) -> Dict:
    policy_map = {
        "easy":   GreedyPolicy,
        "medium": SurgeAwarePolicy,
        "hard":   SurgeAwarePolicy,
    }
    PolicyClass = policy_map[level]
    policy = PolicyClass()

    scores = []
    reports = []

    for seed in seeds:
        grader = get_grader(level, seed=seed)
        score, report = grader.grade(policy)
        scores.append(score)
        reports.append(report)
        if verbose:
            print(f"  seed={seed}  score={score:.4f}  "
                  f"discharged={report.get('discharged_count', report.get('discharge_rate', '?'))}  "
                  f"reward={report.get('total_reward', '?'):.2f}")

    mean_score = sum(scores) / len(scores)
    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

    return {
        "level": level,
        "policy": PolicyClass.__name__,
        "seeds": seeds,
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description="HospitalBedAllocation-v1 baseline")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999, 2024, 7])
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    levels = [args.level] if args.level else ["easy", "medium", "hard"]
    results = {}

    print("=" * 60)
    print("HospitalBedAllocation-v1  —  Baseline Evaluation")
    print("=" * 60)

    for level in levels:
        print(f"\n[{level.upper()}]")
        result = evaluate(level, args.seeds, verbose=not args.json)
        results[level] = result
        if not args.json:
            print(f"  → mean={result['mean_score']:.4f}  std={result['std_score']:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for level, r in results.items():
        print(f"  {level:8s}  mean={r['mean_score']:.4f}  std={r['std_score']:.4f}")

    if args.json:
        print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
