---
title: HospitalBedAllocation-v1
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# 🏥 HospitalBedAllocation-v1

> **OpenEnv submission** — a real-world hospital operations environment for training and evaluating AI agents via the standard `step()` / `reset()` / `state()` API.

---

## Overview

An AI agent acts as a **hospital bed coordinator**, managing patient flow across five departments:

| Department  | Beds | Overflow |
|-------------|------|----------|
| ICU         | 8    | ✗        |
| General     | 20   | ↔ Emergency |
| Surgical    | 12   | ✗        |
| Pediatric   | 10   | ✗        |
| Emergency   | 6    | ↔ General |

Patients arrive continuously with varying **priority** (Critical → Urgent → Standard → Elective) and must be admitted to a compatible department bed. The agent must balance utilization efficiency, handle unexpected surge events, and prevent high-priority patients from deteriorating in the queue.

---

## Action Space

```python
action = {
    "action_type": str,   # "admit" | "discharge" | "transfer" | "hold" | "mark_maintenance"
    "patient_id":  str,   # required for admit / discharge / transfer
    "bed_id":      str,   # required for admit / transfer / mark_maintenance
}
```

| Action            | Available   | Description |
|-------------------|-------------|-------------|
| `admit`           | All levels  | Assign a waiting patient to a free compatible bed |
| `discharge`       | All levels  | Release a ready patient, freeing their bed |
| `hold`            | All levels  | No operation this step |
| `transfer`        | Medium+     | Move admitted patient to a different bed |
| `mark_maintenance`| Hard only   | Take a free bed offline for 3 steps |

---

## Observation Space

```python
obs = {
    "beds":          list[BedDict],      # all 56 beds with status
    "waiting_queue": list[PatientDict],  # patients waiting for assignment
    "admitted":      list[PatientDict],  # currently admitted patients
    "step":          int,
    "max_steps":     int,
    "utilization":   dict[dept, float],  # per-dept occupancy 0.0–1.0
    "surge_active":  bool,               # mass-casualty event flag
    "task_level":    str,
    "info":          dict,               # counts: spawned / discharged / deteriorated
}
```

### Patient fields
```python
{
    "patient_id":    str,   # e.g. "P0042"
    "priority":      int,   # 1=CRITICAL, 2=URGENT, 3=STANDARD, 4=ELECTIVE
    "required_dept": str,   # one of: ICU, GENERAL, SURGICAL, PEDIATRIC, EMERGENCY
    "arrival_step":  int,
    "length_of_stay":int,   # expected steps until discharge-eligible
    "steps_waiting": int,
    "steps_admitted":int,
    "status":        str,   # waiting | admitted | ready_discharge | deteriorated
}
```

---

## Reward Function

Continuous reward in **[−1.0, +1.0]** per step with partial progress signals:

| Event | Reward |
|-------|--------|
| Admit CRITICAL patient | +0.40 |
| Admit URGENT patient | +0.30 |
| Admit STANDARD patient | +0.20 |
| Admit ELECTIVE patient | +0.10 |
| Patient deteriorated (waited too long) | −0.30 |
| Successful discharge (frees bed) | +0.15 |
| CRITICAL patient in queue each step | −0.05 |
| Utilization in 70–90% sweet spot | +0.05 |
| Invalid action / bed incompatibility | −0.05 to −0.20 |

---

## Tasks & Graders

Three tasks with increasing complexity — all scored **0.0 → 1.0** with partial credit:

### Easy (30 steps)
- Stable arrivals (~1.5 patients/step)
- No surge events, no deterioration
- Actions: `admit`, `discharge`, `hold`
- **Score** = 40% discharge rate + 40% zero deteriorations + 20% utilization

### Medium (50 steps)
- Higher arrivals (~2.5/step)
- Surge events (8% chance/step), deterioration enabled
- Actions: + `transfer`
- **Score** = 30% discharge rate + 30% critical handling + 20% surge response + 20% utilization

### Hard (80 steps)
- High arrivals (~4/step), frequent surges (15%/step)
- Maintenance events, full action set
- **Score** = 25% discharge + 25% zero critical deterioration + 20% throughput + 15% maintenance mgmt + 15% queue depth

---

## Quickstart

```bash
git clone <repo>
cd hospital_bed_env
pip install -r requirements.txt   # only: gradio, pyyaml
```

### Run baseline evaluation
```bash
python baseline.py                    # all three levels, seeds 42 123 999 2024 7
python baseline.py --level easy       # single level
python baseline.py --seeds 1 2 3 --json  # JSON output
```

### Use the environment in your own agent
```python
from environment import HospitalBedEnv

env = HospitalBedEnv(task_level="medium", seed=42)
obs = env.reset()

done = False
while not done:
    # Your policy here
    action = {"action_type": "hold"}
    obs, reward, done, info = env.step(action)

state = env.state()   # full internal snapshot
```

### Grade your policy
```python
from graders import get_grader

def my_policy(obs):
    # ... your logic ...
    return {"action_type": "hold"}

grader = get_grader("medium", seed=42)
score, report = grader.grade(my_policy)
print(score, report)
```

---

## Baseline Scores

Rule-based greedy + surge-aware policy:

| Level  | Mean Score | Std  | Policy |
|--------|-----------|------|--------|
| Easy   | 0.62      | 0.04 | GreedyPolicy |
| Medium | 0.51      | 0.06 | SurgeAwarePolicy |
| Hard   | 0.38      | 0.08 | SurgeAwarePolicy |

Scores are reproducible across the default seed set `[42, 123, 999, 2024, 7]`.

---

## Docker / Hugging Face Spaces

```bash
docker build -t hospital-bed-env .
docker run -p 7860:7860 hospital-bed-env
```

The Gradio app exposes an interactive demo on `http://localhost:7860` where you can step through episodes manually or run the full baseline evaluation.

---

## Design Decisions

**Why hospital bed allocation?**
- Real operational pain point — misallocation costs lives and millions in wasted capacity
- Natural multi-objective tradeoff (urgency vs. utilization vs. throughput)
- Rich partial observability and stochastic arrivals
- Surge events create non-stationarity that rewards adaptive policies

**No external dependencies** — the environment uses only Python stdlib (`random`, `dataclasses`, `enum`). Install just `gradio` and `pyyaml` for the demo.

---

## License
MIT
