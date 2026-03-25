"""
Gradio demo app for HospitalBedAllocation-v1
Deploys on Hugging Face Spaces at port 7860.

Users can:
  • Run the baseline policy interactively
  • Step through an episode manually via JSON actions
  • View live bed utilization and patient queue
"""

import json
import gradio as gr
from environment import HospitalBedEnv
from baseline import GreedyPolicy, SurgeAwarePolicy
from graders import get_grader

# Global episode state
_env: HospitalBedEnv | None = None
_obs = None
_done = False
_policy = None
_log = []


def _format_obs(obs):
    if obs is None:
        return "No active episode."
    free_beds = sum(1 for b in obs["beds"] if not b["occupied"] and not b["maintenance"])
    total_beds = len(obs["beds"])
    util = obs["utilization"]
    util_str = "  ".join(f"{k}:{v:.0%}" for k, v in util.items())
    queue = obs["waiting_queue"]
    q_str = ", ".join(f"{p['patient_id']}(p{p['priority']})" for p in queue[:8])
    if len(queue) > 8:
        q_str += f" +{len(queue)-8} more"
    surge = "⚠️ SURGE ACTIVE" if obs.get("surge_active") else ""
    return (
        f"Step {obs['step']}/{obs['max_steps']}  |  "
        f"Free beds: {free_beds}/{total_beds}  {surge}\n"
        f"Utilization: {util_str}\n"
        f"Waiting queue ({len(queue)}): {q_str or 'empty'}\n"
        f"Admitted: {len(obs['admitted'])}  |  "
        f"Discharged: {obs['info']['discharged_count']}  |  "
        f"Deteriorated: {obs['info']['deteriorated_count']}"
    )


def start_episode(level, seed):
    global _env, _obs, _done, _policy, _log
    seed = int(seed)
    _env = HospitalBedEnv(task_level=level, seed=seed)
    _obs = _env.reset(seed=seed)
    _done = False
    _policy = SurgeAwarePolicy() if level in ("medium", "hard") else GreedyPolicy()
    _log = [f"Episode started — level={level}, seed={seed}"]
    return _format_obs(_obs), "\n".join(_log[-20:]), gr.update(interactive=True)


def auto_step():
    global _obs, _done, _log
    if _env is None or _done:
        return _format_obs(_obs), "\n".join(_log[-20:])
    action = _policy(_obs)
    _obs, reward, _done, info = _env.step(action)
    events = info.get("events", [])
    _log.append(
        f"[{_obs['step']:02d}] action={action['action_type']}  "
        f"reward={reward:+.3f}  {'DONE' if _done else ''}"
    )
    for e in events:
        _log.append(f"      {e}")
    if _done:
        state = _env.state()
        _log.append(f"=== EPISODE DONE  total_reward={state['total_reward']:.3f} ===")
    return _format_obs(_obs), "\n".join(_log[-30:])


def manual_step(action_json):
    global _obs, _done, _log
    if _env is None or _done:
        return _format_obs(_obs), "\n".join(_log[-20:])
    try:
        action = json.loads(action_json)
    except json.JSONDecodeError as e:
        _log.append(f"JSON error: {e}")
        return _format_obs(_obs), "\n".join(_log[-20:])
    _obs, reward, _done, info = _env.step(action)
    _log.append(f"[{_obs['step']:02d}] manual={action}  reward={reward:+.3f}")
    return _format_obs(_obs), "\n".join(_log[-30:])


def run_full_baseline(level, seed):
    results = []
    for lvl in (["easy", "medium", "hard"] if level == "all" else [level]):
        grader = get_grader(lvl, seed=int(seed))
        policy = SurgeAwarePolicy() if lvl != "easy" else GreedyPolicy()
        score, report = grader.grade(policy)
        results.append(f"[{lvl}] score={score:.4f}\n{json.dumps(report, indent=2)}")
    return "\n\n".join(results)


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="HospitalBedAllocation-v1") as demo:
    gr.Markdown("""
    # 🏥 Hospital Bed Allocation — OpenEnv v1
    An AI agent allocates beds across ICU, General, Surgical, Pediatric, and Emergency departments.
    **Actions**: `admit`, `discharge`, `transfer`, `hold`, `mark_maintenance`
    """)

    with gr.Row():
        level_dd = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Level")
        seed_box = gr.Number(value=42, label="Seed", precision=0)
        start_btn = gr.Button("Start Episode", variant="primary")

    obs_box = gr.Textbox(label="Observation", lines=6, interactive=False)
    log_box = gr.Textbox(label="Event Log", lines=12, interactive=False)

    with gr.Row():
        auto_btn = gr.Button("▶ Auto Step (policy)", interactive=False)
        auto_all_btn = gr.Button("⏩ Run to End", interactive=False)

    with gr.Row():
        manual_action = gr.Textbox(
            value='{"action_type": "hold"}',
            label="Manual Action (JSON)",
            lines=2,
        )
        manual_btn = gr.Button("Execute Manual Step", interactive=False)

    gr.Markdown("### 📊 Run Full Baseline Evaluation")
    with gr.Row():
        bl_level = gr.Dropdown(["all", "easy", "medium", "hard"], value="all", label="Level")
        bl_seed = gr.Number(value=42, label="Seed", precision=0)
        bl_btn = gr.Button("Run Baseline", variant="secondary")
    bl_out = gr.Textbox(label="Baseline Results", lines=20, interactive=False)

    # Wiring
    start_btn.click(
        start_episode,
        inputs=[level_dd, seed_box],
        outputs=[obs_box, log_box, auto_btn],
    )
    auto_btn.click(auto_step, outputs=[obs_box, log_box])

    def run_to_end():
        global _done
        logs = []
        while not _done:
            o, l = auto_step()
        return o, l

    auto_all_btn.click(run_to_end, outputs=[obs_box, log_box])
    manual_btn.click(manual_step, inputs=[manual_action], outputs=[obs_box, log_box])
    bl_btn.click(run_full_baseline, inputs=[bl_level, bl_seed], outputs=[bl_out])

    # Enable buttons on start
    start_btn.click(
        lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[auto_btn, auto_all_btn, manual_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
