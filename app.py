"""
app.py — HospitalBedAllocation-v1
Gradio app for Hugging Face Spaces.

Requirements met:
  ✓ GET /          → 200 (automated ping)
  ✓ POST /reset    → JSON observation  (automated reset() check)
  ✓ POST /step     → JSON (obs, reward, done, info)
  ✓ POST /state    → JSON full state
  ✓ Interactive UI for manual testing
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from environment import HospitalBedEnv
from graders import get_grader
from baseline import GreedyPolicy, SurgeAwarePolicy

# ── FastAPI sub-app for HTTP API endpoints ────────────────────────────────────
api = FastAPI()

# Shared environment state
_lock = threading.Lock()
_envs: Dict[str, HospitalBedEnv] = {}


@api.get("/")
async def health():
    """Ping endpoint — automated checker expects HTTP 200."""
    return {"status": "ok", "env_id": "HospitalBedAllocation-v1", "version": "1.0.0"}


@api.get("/health")
async def health2():
    return {"status": "ok"}


@api.post("/reset")
async def http_reset(request: Request):
    """
    POST /reset
    Body (optional): {"task_level": "easy"|"medium"|"hard", "seed": 42, "session_id": "..."}
    Returns: observation dict
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_level = body.get("task_level", "easy")
    seed       = body.get("seed", 42)
    session_id = body.get("session_id", "default")

    with _lock:
        env = HospitalBedEnv(task_level=task_level, seed=seed)
        obs = env.reset(seed=seed)
        _envs[session_id] = env

    return JSONResponse(content=obs)


@api.post("/step")
async def http_step(request: Request):
    """
    POST /step
    Body: {"action": {...}, "session_id": "..."}
    Returns: {"observation": ..., "reward": ..., "done": ..., "info": ...}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    session_id = body.get("session_id", "default")
    action     = body.get("action", {"action_type": "hold"})

    with _lock:
        env = _envs.get(session_id)
        if env is None:
            return JSONResponse({"error": "no active session — call /reset first"}, status_code=400)
        obs, reward, done, info = env.step(action)

    return JSONResponse({
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
    })


@api.post("/state")
async def http_state(request: Request):
    """POST /state — returns full internal state snapshot."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = body.get("session_id", "default")
    with _lock:
        env = _envs.get(session_id)
        if env is None:
            return JSONResponse({"error": "no active session"}, status_code=400)
        state = env.state()
    return JSONResponse(content=state)


@api.post("/grade")
async def http_grade(request: Request):
    """
    POST /grade
    Body: {"level": "easy"|"medium"|"hard", "seed": 42}
    Returns grader score + report.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    level = body.get("level", "easy")
    seed  = body.get("seed", 42)

    PolicyClass = SurgeAwarePolicy if level != "easy" else GreedyPolicy
    policy = PolicyClass()
    grader = get_grader(level, seed=seed)
    score, report = grader.grade(policy)
    return JSONResponse({"level": level, "seed": seed, "score": score, "report": report})


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_ui_env: Optional[HospitalBedEnv] = None
_ui_obs = None
_ui_done = False
_ui_policy = None
_ui_log: list = []


def _fmt(obs):
    if obs is None:
        return "No active episode. Press Start Episode."
    free  = sum(1 for b in obs["beds"] if not b["occupied"] and not b["maintenance"])
    total = len(obs["beds"])
    util  = "  ".join(f"{k[:3]}:{v:.0%}" for k, v in obs["utilization"].items())
    surge = "  ⚠️ SURGE" if obs.get("surge_active") else ""
    q     = obs["waiting_queue"]
    q_str = ", ".join(f"{p['patient_id']}(p{p['priority']})" for p in q[:6])
    if len(q) > 6:
        q_str += f" +{len(q)-6}"
    return (
        f"Step {obs['step']}/{obs['max_steps']}  |  Free: {free}/{total}{surge}\n"
        f"Utilization: {util}\n"
        f"Waiting ({len(q)}): {q_str or 'none'}\n"
        f"Admitted: {len(obs['admitted'])}  "
        f"Discharged: {obs['info']['discharged_count']}  "
        f"Deteriorated: {obs['info']['deteriorated_count']}"
    )


def ui_start(level, seed):
    global _ui_env, _ui_obs, _ui_done, _ui_policy, _ui_log
    seed = int(seed)
    _ui_env    = HospitalBedEnv(task_level=level, seed=seed)
    _ui_obs    = _ui_env.reset(seed=seed)
    _ui_done   = False
    _ui_policy = SurgeAwarePolicy() if level != "easy" else GreedyPolicy()
    _ui_log    = [f"▶ Started — level={level}, seed={seed}"]
    return _fmt(_ui_obs), "\n".join(_ui_log[-25:])


def ui_auto_step():
    global _ui_obs, _ui_done, _ui_log
    if _ui_env is None or _ui_done:
        return _fmt(_ui_obs), "\n".join(_ui_log[-25:])
    action = _ui_policy(_ui_obs)
    _ui_obs, reward, _ui_done, info = _ui_env.step(action)
    _ui_log.append(
        f"[{_ui_obs['step']:02d}] {action['action_type']:20s} "
        f"reward={reward:+.3f} {'✅ DONE' if _ui_done else ''}"
    )
    for e in info.get("events", []):
        _ui_log.append(f"      {e}")
    return _fmt(_ui_obs), "\n".join(_ui_log[-25:])


def ui_run_to_end():
    global _ui_done
    if _ui_env is None:
        return _fmt(None), "No episode started."
    while not _ui_done:
        ui_auto_step()
    state = _ui_env.state()
    _ui_log.append(f"=== END  total_reward={state['total_reward']:.3f} ===")
    return _fmt(_ui_obs), "\n".join(_ui_log[-30:])


def ui_manual(action_json):
    global _ui_obs, _ui_done, _ui_log
    if _ui_env is None or _ui_done:
        return _fmt(_ui_obs), "\n".join(_ui_log[-25:])
    try:
        action = json.loads(action_json)
    except Exception as e:
        _ui_log.append(f"❌ JSON error: {e}")
        return _fmt(_ui_obs), "\n".join(_ui_log[-25:])
    _ui_obs, reward, _ui_done, info = _ui_env.step(action)
    _ui_log.append(f"[{_ui_obs['step']:02d}] manual: {action}  reward={reward:+.3f}")
    return _fmt(_ui_obs), "\n".join(_ui_log[-25:])


def ui_grade(level, seed):
    seed = int(seed)
    PolicyClass = SurgeAwarePolicy if level != "easy" else GreedyPolicy
    grader = get_grader(level, seed=seed)
    score, report = grader.grade(PolicyClass())
    return f"Level={level}  seed={seed}  score={score:.4f}\n\n{json.dumps(report, indent=2)}"


with gr.Blocks(title="HospitalBedAllocation-v1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Hospital Bed Allocation — OpenEnv v1
    AI agent allocates beds across **ICU · General · Surgical · Pediatric · Emergency**.

    **HTTP API** (for automated evaluation):
    `GET /` — ping  |  `POST /reset`  |  `POST /step`  |  `POST /state`  |  `POST /grade`
    """)

    with gr.Row():
        lvl   = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Level")
        seed  = gr.Number(value=42, label="Seed", precision=0)
        start = gr.Button("▶ Start Episode", variant="primary")

    obs_box = gr.Textbox(label="Observation", lines=5, interactive=False)
    log_box = gr.Textbox(label="Event Log",   lines=12, interactive=False)

    with gr.Row():
        step_btn = gr.Button("Step (policy)")
        run_btn  = gr.Button("⏩ Run to End")

    with gr.Row():
        manual_in  = gr.Textbox(value='{"action_type":"hold"}', label="Manual Action (JSON)", scale=4)
        manual_btn = gr.Button("Execute", scale=1)

    gr.Markdown("### 📊 Grade a level")
    with gr.Row():
        g_lvl  = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Level")
        g_seed = gr.Number(value=42, label="Seed", precision=0)
        g_btn  = gr.Button("Run Grader")
    g_out = gr.Textbox(label="Grade Result", lines=15, interactive=False)

    start.click(ui_start,     inputs=[lvl, seed],    outputs=[obs_box, log_box])
    step_btn.click(ui_auto_step,                      outputs=[obs_box, log_box])
    run_btn.click(ui_run_to_end,                      outputs=[obs_box, log_box])
    manual_btn.click(ui_manual, inputs=[manual_in],   outputs=[obs_box, log_box])
    g_btn.click(ui_grade,    inputs=[g_lvl, g_seed],  outputs=[g_out])


# ── Mount Gradio onto FastAPI ─────────────────────────────────────────────────
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
