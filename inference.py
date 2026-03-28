"""
inference.py — HospitalBedAllocation-v1
Required filename per hackathon rules.

Uses OpenAI-compatible client pointed at API_BASE_URL / MODEL_NAME / HF_TOKEN
to run an LLM-based policy across all three task levels and produce graded scores.

Environment variables required:
    API_BASE_URL   — LLM API endpoint  (e.g. https://api.openai.com/v1)
    MODEL_NAME     — model identifier  (e.g. gpt-4o-mini)
    HF_TOKEN       — Hugging Face / API key

Runtime target: <20 min on vcpu=2, memory=8gb
"""

from __future__ import annotations

import json
import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

# ── OpenAI client (required by hackathon rules) ──────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai", "-q", "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from openai import OpenAI
        _OPENAI_AVAILABLE = True
    except Exception:
        _OPENAI_AVAILABLE = False
        OpenAI = None  # type: ignore

from environment import HospitalBedEnv, PatientPriority
from graders import get_grader


# ── Client setup ─────────────────────────────────────────────────────────────

def get_client() -> "OpenAI":
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available")
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    hf_token = os.environ.get("HF_TOKEN", "")
    return OpenAI(base_url=api_base, api_key=hf_token)


MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# ── LLM Policy ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a hospital bed coordinator AI. Your job is to allocate beds to patients
efficiently, prioritise critical patients, and prevent deterioration.

At each step you receive the current hospital state as JSON and must respond
with a single JSON action object. No prose, no markdown — only valid JSON.

Action schema:
{
  "action_type": "admit" | "discharge" | "transfer" | "hold" | "mark_maintenance",
  "patient_id":  "<id>",   // required for admit, discharge, transfer
  "bed_id":      "<id>"    // required for admit, transfer, mark_maintenance
}

Strategy guidelines:
- ALWAYS discharge patients whose status is "ready_discharge" first (frees beds).
- ALWAYS admit CRITICAL patients (priority=1) before anyone else.
- Prefer exact department matches; overflow is allowed GENERAL↔EMERGENCY only.
- During a surge (surge_active=true), clear beds aggressively.
- If nothing useful can be done, return {"action_type": "hold"}.
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    """Compress observation to a compact JSON string for the LLM."""
    waiting = obs["waiting_queue"]
    admitted = obs["admitted"]

    # Summarise free beds per dept
    beds = obs["beds"]
    free: Dict[str, List[str]] = {}
    for b in beds:
        if not b["occupied"] and not b["maintenance"]:
            free.setdefault(b["department"], []).append(b["bed_id"])

    # Patients ready to discharge
    ready_discharge = [p for p in admitted if p["status"] == "ready_discharge"]

    compact = {
        "step": obs["step"],
        "max_steps": obs["max_steps"],
        "surge_active": obs.get("surge_active", False),
        "utilization": obs["utilization"],
        "free_beds_by_dept": {dept: ids[:5] for dept, ids in free.items()},  # cap 5 per dept
        "ready_discharge": [{"patient_id": p["patient_id"]} for p in ready_discharge],
        "waiting_queue": [
            {
                "patient_id": p["patient_id"],
                "priority": p["priority"],
                "required_dept": p["required_dept"],
                "steps_waiting": p["steps_waiting"],
            }
            for p in sorted(waiting, key=lambda x: (x["priority"], -x["steps_waiting"]))[:10]
        ],
    }
    return json.dumps(compact, separators=(",", ":"))


def llm_action(client: OpenAI, obs: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    """Call the LLM and parse a valid action. Falls back to hold on failure."""
    user_msg = build_user_prompt(obs)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=100,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(raw)
            if "action_type" not in action:
                raise ValueError("Missing action_type")
            return action
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  LLM attempt {attempt+1} failed ({exc}), retrying in {wait}s…", flush=True)
            time.sleep(wait)
    return {"action_type": "hold"}


# ── Greedy fallback (used when LLM unavailable) ───────────────────────────────

class GreedyFallback:
    def __call__(self, obs: Dict) -> Dict:
        beds = {b["bed_id"]: b for b in obs["beds"]}
        for p in obs["admitted"]:
            if p["status"] == "ready_discharge":
                return {"action_type": "discharge", "patient_id": p["patient_id"]}
        sorted_q = sorted(obs["waiting_queue"], key=lambda p: (p["priority"], -p["steps_waiting"]))
        for patient in sorted_q:
            dept = patient["required_dept"]
            for bid, bed in beds.items():
                if not bed["occupied"] and not bed["maintenance"]:
                    if bed["department"] == dept:
                        return {"action_type": "admit", "patient_id": patient["patient_id"], "bed_id": bid}
            # overflow
            overflow = {"GENERAL": "EMERGENCY", "EMERGENCY": "GENERAL"}
            alt = overflow.get(dept)
            if alt:
                for bid, bed in beds.items():
                    if not bed["occupied"] and not bed["maintenance"] and bed["department"] == alt:
                        return {"action_type": "admit", "patient_id": patient["patient_id"], "bed_id": bid}
        return {"action_type": "hold"}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    client: Optional[OpenAI],
    level: str,
    seed: int,
    use_llm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = HospitalBedEnv(task_level=level, seed=seed)
    obs = env.reset(seed=seed)
    done = False
    step = 0
    fallback = GreedyFallback()

    t0 = time.time()
    while not done:
        if use_llm and client is not None:
            action = llm_action(client, obs)
        else:
            action = fallback(obs)

        obs, reward, done, info = env.step(action)
        step += 1

        if verbose and step % 10 == 0:
            util = obs["utilization"]
            u_str = " ".join(f"{k[:3]}:{v:.0%}" for k, v in util.items())
            print(f"    step={step:3d}  {u_str}  "
                  f"q={len(obs['waiting_queue'])}  "
                  f"reward={reward:+.3f}", flush=True)

    elapsed = time.time() - t0
    state = env.state()
    return {
        "level": level,
        "seed": seed,
        "steps": step,
        "total_reward": round(state["total_reward"], 4),
        "elapsed_sec": round(elapsed, 1),
        "discharged": len(state["discharged"]),
        "deteriorated": sum(
            1 for p in state["patients"].values() if p["status"] == "deteriorated"
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("HospitalBedAllocation-v1  —  Inference & Grading", flush=True)
    print("=" * 60, flush=True)

    # Validate env vars
    api_base  = os.environ.get("API_BASE_URL", "")
    model     = os.environ.get("MODEL_NAME", "")
    hf_token  = os.environ.get("HF_TOKEN", "")

    use_llm = bool(api_base and model and hf_token and _OPENAI_AVAILABLE)
    client: Optional[OpenAI] = None

    if use_llm:
        print(f"LLM mode: {model} @ {api_base}", flush=True)
        client = get_client()
    else:
        print("WARNING: API_BASE_URL / MODEL_NAME / HF_TOKEN not set — using greedy fallback.", flush=True)

    levels = ["easy", "medium", "hard"]
    seeds  = [42, 123, 999]   # 3 seeds keeps runtime well under 20 min

    all_results: Dict[str, Any] = {}
    grand_t0 = time.time()

    for level in levels:
        print(f"\n{'─'*40}", flush=True)
        print(f"LEVEL: {level.upper()}", flush=True)
        print(f"{'─'*40}", flush=True)

        level_scores = []
        level_reports = []

        for seed in seeds:
            print(f"\n  Grading seed={seed}…", flush=True)

            # Run episode first (for diagnostics)
            ep = run_episode(client, level, seed, use_llm=use_llm, verbose=True)
            print(f"  Episode done: {ep}", flush=True)

            # Official grade via grader
            grader = get_grader(level, seed=seed)
            if use_llm and client is not None:
                def llm_policy(obs, _c=client):
                    return llm_action(_c, obs)
                score, report = grader.grade(llm_policy)
            else:
                score, report = grader.grade(GreedyFallback())

            level_scores.append(score)
            level_reports.append(report)
            print(f"  Score: {score:.4f}", flush=True)

        mean_score = sum(level_scores) / len(level_scores)
        std_score  = (sum((s - mean_score) ** 2 for s in level_scores) / len(level_scores)) ** 0.5

        all_results[level] = {
            "scores":     level_scores,
            "mean_score": round(mean_score, 4),
            "std_score":  round(std_score,  4),
            "reports":    level_reports,
        }

        print(f"\n  {level} → mean={mean_score:.4f}  std={std_score:.4f}", flush=True)

    total_elapsed = round(time.time() - grand_t0, 1)
    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for level, r in all_results.items():
        print(f"  {level:8s}  mean={r['mean_score']:.4f}  std={r['std_score']:.4f}", flush=True)
    print(f"\nTotal runtime: {total_elapsed}s", flush=True)

    # Write scores.json (for automated validation)
    scores_out = {
        "env_id":        "HospitalBedAllocation-v1",
        "model":         model or "greedy-fallback",
        "total_elapsed": total_elapsed,
        "results":       all_results,
    }
    with open("scores.json", "w") as f:
        json.dump(scores_out, f, indent=2)
    print("\nScores written to scores.json", flush=True)

    return scores_out


if __name__ == "__main__":
    main()
