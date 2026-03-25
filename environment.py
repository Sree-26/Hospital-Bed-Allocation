"""
Hospital Bed Allocation Environment
OpenEnv-compliant environment for AI agent training.

An agent manages bed assignments across hospital departments,
handling patient intake, discharges, transfers, and emergency surges.
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class Department(str, Enum):
    ICU = "ICU"
    GENERAL = "GENERAL"
    SURGICAL = "SURGICAL"
    PEDIATRIC = "PEDIATRIC"
    EMERGENCY = "EMERGENCY"


class PatientPriority(int, Enum):
    CRITICAL = 1   # must be placed immediately
    URGENT = 2
    STANDARD = 3
    ELECTIVE = 4


class PatientStatus(str, Enum):
    WAITING = "waiting"
    ADMITTED = "admitted"
    READY_DISCHARGE = "ready_discharge"
    DISCHARGED = "discharged"
    TRANSFERRED = "transferred"
    DETERIORATED = "deteriorated"   # waited too long → escalated


@dataclass
class Patient:
    patient_id: str
    priority: PatientPriority
    required_dept: Department
    arrival_step: int
    length_of_stay: int          # expected steps until discharge
    steps_waiting: int = 0
    steps_admitted: int = 0
    assigned_bed: Optional[str] = None
    status: PatientStatus = PatientStatus.WAITING

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["required_dept"] = self.required_dept.value
        d["status"] = self.status.value
        return d


@dataclass
class Bed:
    bed_id: str
    department: Department
    occupied: bool = False
    patient_id: Optional[str] = None
    maintenance: bool = False       # unavailable for cleaning/repair

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["department"] = self.department.value
        return d


@dataclass
class DepartmentConfig:
    name: Department
    capacity: int
    overflow_allowed: bool = False   # can temporarily take other depts


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

DEPT_CONFIGS: Dict[Department, DepartmentConfig] = {
    Department.ICU:       DepartmentConfig(Department.ICU,       8,  False),
    Department.GENERAL:   DepartmentConfig(Department.GENERAL,   20, True),
    Department.SURGICAL:  DepartmentConfig(Department.SURGICAL,  12, False),
    Department.PEDIATRIC: DepartmentConfig(Department.PEDIATRIC, 10, False),
    Department.EMERGENCY: DepartmentConfig(Department.EMERGENCY, 6,  True),
}

# Compatible overflow departments (key can send to values)
OVERFLOW_MAP: Dict[Department, List[Department]] = {
    Department.GENERAL:   [Department.EMERGENCY],
    Department.EMERGENCY: [Department.GENERAL],
}

MAX_WAIT_STEPS = {
    PatientPriority.CRITICAL: 1,
    PatientPriority.URGENT:   3,
    PatientPriority.STANDARD: 6,
    PatientPriority.ELECTIVE: 12,
}


class _RNG(random.Random):
    def poisson_approx(self, lam: float) -> int:
        """Approximate Poisson using Knuth method (no numpy needed)."""
        import math
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.random()
            if k > 30:
                break
        return max(0, k - 1)

    def choices(self, population, weights=None, k=1):
        return random.choices(population, weights=weights, k=k)


class HospitalBedEnv:
    """
    OpenEnv-compatible Hospital Bed Allocation environment.

    Action space (dict):
        action_type : str  — one of: "admit", "discharge", "transfer", "hold", "mark_maintenance"
        patient_id  : str  — target patient (for admit / discharge / transfer)
        bed_id      : str  — target bed     (for admit / transfer)
        source_bed  : str  — source bed     (for transfer, optional)

    Observation space (dict):
        beds          : list[Bed.to_dict()]
        waiting_queue : list[Patient.to_dict()]
        admitted      : list[Patient.to_dict()]
        step          : int
        utilization   : dict[dept -> float]
        surge_active  : bool
        info          : dict  — misc diagnostics

    Reward:
        Shaped continuous reward in [−1, 1] per step combining:
        - +0.4  per critical/urgent patient admitted this step
        - +0.2  per standard patient admitted
        - +0.1  per elective patient admitted
        - −0.3  per patient who deteriorated (waited too long)
        - +0.15 per successful discharge (frees capacity)
        - −0.05 per step a critical patient waits
        - +0.05 utilization bonus when 70–90% beds filled (efficiency sweet spot)
        - −0.1  per unnecessary maintenance mark (abuse penalty)
    """

    metadata = {"version": "1.0.0", "env_id": "HospitalBedAllocation-v1"}

    def __init__(self, task_level: str = "easy", seed: Optional[int] = None):
        assert task_level in ("easy", "medium", "hard"), \
            "task_level must be 'easy', 'medium', or 'hard'"
        self.task_level = task_level
        self.seed = seed
        self._rng = _RNG(seed)
        self._configure_task()
        self.reset()

    # ------------------------------------------------------------------
    # Task configuration
    # ------------------------------------------------------------------

    def _configure_task(self):
        level = self.task_level
        if level == "easy":
            self.max_steps = 30
            self.surge_probability = 0.0
            self.arrival_rate = 1.5          # avg patients per step
            self.maintenance_events = False
            self.patient_deterioration = False
            self.allowed_actions = {"admit", "discharge", "hold"}
        elif level == "medium":
            self.max_steps = 50
            self.surge_probability = 0.08
            self.arrival_rate = 2.5
            self.maintenance_events = True
            self.patient_deterioration = True
            self.allowed_actions = {"admit", "discharge", "transfer", "hold"}
        else:  # hard
            self.max_steps = 80
            self.surge_probability = 0.15
            self.arrival_rate = 4.0
            self.maintenance_events = True
            self.patient_deterioration = True
            self.allowed_actions = {"admit", "discharge", "transfer",
                                    "hold", "mark_maintenance"}

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self._rng = _RNG(seed)

        self._step_count = 0
        self._total_reward = 0.0
        self._surge_active = False
        self._surge_steps_remaining = 0
        self._maintenance_abuse_count = 0

        # Build beds
        self._beds: Dict[str, Bed] = {}
        for dept, cfg in DEPT_CONFIGS.items():
            for i in range(cfg.capacity):
                bid = f"{dept.value[:3]}-{i+1:02d}"
                self._beds[bid] = Bed(bed_id=bid, department=dept)

        self._patients: Dict[str, Patient] = {}
        self._waiting_queue: List[str] = []   # patient_ids
        self._admitted: List[str] = []
        self._discharged: List[str] = []
        self._patient_counter = 0
        self._step_stats: List[Dict] = []

        # Pre-populate some admitted patients so env isn't empty
        self._prepopulate()

        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one action.
        Returns: (observation, reward, done, info)
        """
        reward = 0.0
        info: Dict[str, Any] = {"action": action, "events": []}

        # 1. Validate action
        action_type = action.get("action_type", "hold")
        if action_type not in self.allowed_actions:
            info["error"] = f"action '{action_type}' not allowed at level '{self.task_level}'"
            reward -= 0.1
            action_type = "hold"

        # 2. Execute action
        if action_type == "admit":
            r, msg = self._do_admit(action)
            reward += r
            info["events"].append(msg)

        elif action_type == "discharge":
            r, msg = self._do_discharge(action)
            reward += r
            info["events"].append(msg)

        elif action_type == "transfer":
            r, msg = self._do_transfer(action)
            reward += r
            info["events"].append(msg)

        elif action_type == "mark_maintenance":
            r, msg = self._do_mark_maintenance(action)
            reward += r
            info["events"].append(msg)

        # hold → do nothing, just advance time

        # 3. Advance environment time
        step_reward, step_events = self._advance_time()
        reward += step_reward
        info["events"].extend(step_events)

        # 4. Utilization bonus
        util_reward = self._utilization_bonus()
        reward += util_reward

        # 5. Clamp reward
        reward = max(-1.0, min(1.0, reward))

        self._step_count += 1
        self._total_reward += reward

        obs = self._get_observation()
        done = self._step_count >= self.max_steps or self._all_served()

        info["step"] = self._step_count
        info["total_reward"] = self._total_reward
        info["utilization"] = obs["utilization"]

        self._step_stats.append({
            "step": self._step_count,
            "reward": reward,
            "waiting": len(self._waiting_queue),
            "admitted": len(self._admitted),
        })

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Full internal state snapshot (for graders / debugging)."""
        return {
            "step": self._step_count,
            "task_level": self.task_level,
            "beds": {bid: b.to_dict() for bid, b in self._beds.items()},
            "patients": {pid: p.to_dict() for pid, p in self._patients.items()},
            "waiting_queue": list(self._waiting_queue),
            "admitted": list(self._admitted),
            "discharged": list(self._discharged),
            "total_reward": self._total_reward,
            "surge_active": self._surge_active,
            "step_stats": self._step_stats,
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_admit(self, action: Dict) -> Tuple[float, str]:
        pid = action.get("patient_id")
        bid = action.get("bed_id")

        if pid not in self._patients or pid not in self._waiting_queue:
            return -0.05, f"admit failed: patient {pid} not in queue"
        if bid not in self._beds:
            return -0.05, f"admit failed: bed {bid} not found"

        patient = self._patients[pid]
        bed = self._beds[bid]

        if bed.occupied or bed.maintenance:
            return -0.1, f"admit failed: bed {bid} unavailable"

        # Check department compatibility
        if not self._dept_compatible(patient.required_dept, bed.department):
            return -0.15, f"admit failed: dept mismatch {patient.required_dept} → {bed.department}"

        # Admit
        bed.occupied = True
        bed.patient_id = pid
        patient.assigned_bed = bid
        patient.status = PatientStatus.ADMITTED
        self._waiting_queue.remove(pid)
        self._admitted.append(pid)

        reward_map = {
            PatientPriority.CRITICAL: 0.4,
            PatientPriority.URGENT:   0.3,
            PatientPriority.STANDARD: 0.2,
            PatientPriority.ELECTIVE: 0.1,
        }
        return reward_map[patient.priority], f"admitted {pid} (p{patient.priority.value}) to {bid}"

    def _do_discharge(self, action: Dict) -> Tuple[float, str]:
        pid = action.get("patient_id")
        if pid not in self._patients or pid not in self._admitted:
            return -0.05, f"discharge failed: {pid} not admitted"

        patient = self._patients[pid]
        if patient.status not in (PatientStatus.ADMITTED, PatientStatus.READY_DISCHARGE):
            return -0.05, f"discharge failed: {pid} not ready"

        # Only discharge if LOS complete or explicitly ready
        if patient.status != PatientStatus.READY_DISCHARGE and patient.steps_admitted < patient.length_of_stay:
            return -0.1, f"discharge failed: {pid} LOS not complete ({patient.steps_admitted}/{patient.length_of_stay})"

        self._free_bed(patient)
        patient.status = PatientStatus.DISCHARGED
        self._admitted.remove(pid)
        self._discharged.append(pid)
        return 0.15, f"discharged {pid}"

    def _do_transfer(self, action: Dict) -> Tuple[float, str]:
        pid = action.get("patient_id")
        bid = action.get("bed_id")   # destination bed

        if pid not in self._patients or pid not in self._admitted:
            return -0.05, f"transfer failed: {pid} not admitted"
        if bid not in self._beds:
            return -0.05, f"transfer failed: bed {bid} not found"

        patient = self._patients[pid]
        dest_bed = self._beds[bid]

        if dest_bed.occupied or dest_bed.maintenance:
            return -0.1, f"transfer failed: dest bed {bid} unavailable"
        if not self._dept_compatible(patient.required_dept, dest_bed.department):
            return -0.1, f"transfer failed: dept incompatible"

        self._free_bed(patient)
        dest_bed.occupied = True
        dest_bed.patient_id = pid
        patient.assigned_bed = bid
        patient.status = PatientStatus.TRANSFERRED
        return 0.05, f"transferred {pid} to {bid}"

    def _do_mark_maintenance(self, action: Dict) -> Tuple[float, str]:
        bid = action.get("bed_id")
        if bid not in self._beds:
            return -0.05, f"maintenance failed: bed {bid} not found"
        bed = self._beds[bid]
        if bed.occupied:
            return -0.2, f"maintenance failed: {bid} occupied"
        if bed.maintenance:
            self._maintenance_abuse_count += 1
            return -0.1, f"maintenance abuse: {bid} already in maintenance"
        bed.maintenance = True
        # Maintenance clears after 3 steps (tracked externally via patient-less scheduling)
        # Store as a special "maintenance patient"
        mid = f"MAINT-{bid}"
        self._patients[mid] = Patient(
            patient_id=mid,
            priority=PatientPriority.ELECTIVE,
            required_dept=bed.department,
            arrival_step=self._step_count,
            length_of_stay=3,
            status=PatientStatus.ADMITTED,
            assigned_bed=bid,
        )
        self._patients[mid].steps_admitted = 0
        self._admitted.append(mid)
        return 0.0, f"bed {bid} marked for maintenance (3 steps)"

    # ------------------------------------------------------------------
    # Time advance
    # ------------------------------------------------------------------

    def _advance_time(self) -> Tuple[float, List[str]]:
        reward = 0.0
        events = []

        # Tick admitted patients
        for pid in list(self._admitted):
            p = self._patients[pid]
            p.steps_admitted += 1
            if p.steps_admitted >= p.length_of_stay:
                if pid.startswith("MAINT-"):
                    # Release maintenance
                    bid = p.assigned_bed
                    if bid and bid in self._beds:
                        self._beds[bid].maintenance = False
                        self._beds[bid].occupied = False
                        self._beds[bid].patient_id = None
                    self._admitted.remove(pid)
                    del self._patients[pid]
                    events.append(f"maintenance complete: {bid}")
                else:
                    p.status = PatientStatus.READY_DISCHARGE
                    events.append(f"patient {pid} ready for discharge")

        # Tick waiting patients
        for pid in list(self._waiting_queue):
            p = self._patients[pid]
            p.steps_waiting += 1
            max_wait = MAX_WAIT_STEPS[p.priority]
            if self.patient_deterioration and p.steps_waiting > max_wait:
                p.status = PatientStatus.DETERIORATED
                self._waiting_queue.remove(pid)
                reward -= 0.3
                events.append(f"DETERIORATED: {pid} (waited {p.steps_waiting} steps)")

        # Arrivals
        n_arrivals = self._rng.poisson_approx(self.arrival_rate)
        for _ in range(n_arrivals):
            p = self._spawn_patient()
            self._patients[p.patient_id] = p
            self._waiting_queue.append(p.patient_id)
            events.append(f"arrived: {p.patient_id} ({p.required_dept.value}, p{p.priority.value})")

        # Surge event
        if not self._surge_active and self._rng.random() < self.surge_probability:
            self._trigger_surge(events)
        if self._surge_active:
            self._surge_steps_remaining -= 1
            if self._surge_steps_remaining <= 0:
                self._surge_active = False
                events.append("surge ended")

        # Penalty per step for waiting critical patients
        for pid in self._waiting_queue:
            p = self._patients[pid]
            if p.priority == PatientPriority.CRITICAL:
                reward -= 0.05

        return reward, events

    def _trigger_surge(self, events: List[str]):
        """Mass-casualty or epidemic wave — dumps 4-8 urgent patients."""
        self._surge_active = True
        self._surge_steps_remaining = self._rng.randint(3, 6)
        n = self._rng.randint(4, 8)
        for _ in range(n):
            p = self._spawn_patient(priority_override=PatientPriority.URGENT)
            self._patients[p.patient_id] = p
            self._waiting_queue.append(p.patient_id)
        events.append(f"!!! SURGE: {n} urgent patients arrived !!!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepopulate(self):
        """Fill ~40% of beds with already-admitted patients."""
        for dept, cfg in DEPT_CONFIGS.items():
            fill_count = max(1, cfg.capacity // 3)
            dept_beds = [b for b in self._beds.values() if b.department == dept]
            for bed in dept_beds[:fill_count]:
                p = self._spawn_patient(dept_override=dept)
                p.status = PatientStatus.ADMITTED
                p.steps_admitted = self._rng.randint(1, max(1, p.length_of_stay - 1))
                p.assigned_bed = bed.bed_id
                self._patients[p.patient_id] = p
                self._admitted.append(p.patient_id)
                bed.occupied = True
                bed.patient_id = p.patient_id
        # Seed queue
        for _ in range(self._rng.randint(2, 5)):
            p = self._spawn_patient()
            self._patients[p.patient_id] = p
            self._waiting_queue.append(p.patient_id)

    def _spawn_patient(
        self,
        priority_override: Optional[PatientPriority] = None,
        dept_override: Optional[Department] = None,
    ) -> Patient:
        self._patient_counter += 1
        pid = f"P{self._patient_counter:04d}"
        priority = priority_override or self._rng.choices(
            list(PatientPriority),
            weights=[0.10, 0.25, 0.45, 0.20],
        )[0]
        dept = dept_override or self._rng.choice(list(Department))
        los = self._rng.randint(3, 12)
        return Patient(
            patient_id=pid,
            priority=priority,
            required_dept=dept,
            arrival_step=self._step_count,
            length_of_stay=los,
        )

    def _dept_compatible(self, required: Department, available: Department) -> bool:
        if required == available:
            return True
        overflow_targets = OVERFLOW_MAP.get(required, [])
        return available in overflow_targets

    def _free_bed(self, patient: Patient):
        if patient.assigned_bed and patient.assigned_bed in self._beds:
            bed = self._beds[patient.assigned_bed]
            bed.occupied = False
            bed.patient_id = None
        patient.assigned_bed = None

    def _utilization_bonus(self) -> float:
        total = len(self._beds)
        occupied = sum(1 for b in self._beds.values() if b.occupied and not b.maintenance)
        util = occupied / max(1, total)
        if 0.70 <= util <= 0.90:
            return 0.05
        return 0.0

    def _all_served(self) -> bool:
        return (
            len(self._waiting_queue) == 0
            and all(
                self._patients[pid].status
                in (PatientStatus.READY_DISCHARGE, PatientStatus.DISCHARGED)
                for pid in self._admitted
                if not pid.startswith("MAINT-")
            )
        )

    def _get_observation(self) -> Dict[str, Any]:
        utilization = {}
        for dept in Department:
            dept_beds = [b for b in self._beds.values() if b.department == dept]
            if dept_beds:
                occ = sum(1 for b in dept_beds if b.occupied)
                utilization[dept.value] = round(occ / len(dept_beds), 3)

        return {
            "beds": [b.to_dict() for b in self._beds.values()],
            "waiting_queue": [
                self._patients[pid].to_dict()
                for pid in self._waiting_queue
                if pid in self._patients
            ],
            "admitted": [
                self._patients[pid].to_dict()
                for pid in self._admitted
                if pid in self._patients and not pid.startswith("MAINT-")
            ],
            "step": self._step_count,
            "max_steps": self.max_steps,
            "utilization": utilization,
            "surge_active": self._surge_active,
            "task_level": self.task_level,
            "info": {
                "total_patients_spawned": self._patient_counter,
                "discharged_count": len(self._discharged),
                "deteriorated_count": sum(
                    1 for p in self._patients.values()
                    if p.status == PatientStatus.DETERIORATED
                ),
            },
        }



