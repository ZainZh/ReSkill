# Orchestrator specification

This is the specification read by the orchestrating agent. The orchestrator is a long-running Claude Code session
(Claude Opus 5.5). Development agents are separate headless Claude Code sessions (same model) started by the
orchestrator. Humans give only occasional directional instructions (for example, adding tasks); they do not edit
program code or decide adoption. Paths are relative to `$RESKILL_ROOT/code`.

## Goal

Using only observation inputs (head and wrist RGB-D images from the official observation, camera calibration, joint
encoders, the robot model, instruction text), maximise the native whole-task success rate of each RoboDojo task:
classify_objects, classify_objects_by_language, pack_objects_into_box, organize_table, build_tower, stack_blocks,
stack_blocks_by_language, stack_bowls, store_tools_in_toolbox, general_pickup.

## Inviolable rules

1. The run-time controller never reads simulator state, `Assets/`, layout JSON, metadata, `points.npz` or native
   results. The input audit of `runner/audit.py` applies; an episode that fails the audit is invalid.
2. Only the native score counts (`_result.json` with `eval_time >= 1`), within the native action limit (`step_lim`);
   no rewinding or mid-episode resets. The controller never declares success itself.
3. Final validation uses seed-2 layouts that are not used in revision or model selection (`tools/arconf.py`
   `TEST_LAYOUTS`). They stay sealed until the orchestrator declares the final version and take no part in any
   selection. `run_episode.py --split test` requires `--unseal-test`; the queue refuses it.
4. Public demonstrations may be used only offline during development, by candidates marked
   `privileged_demo_assisted`, to help structure graphs and set parameters; the controller still uses only
   observations at run time. Offline debugging tools may not read layout JSON, asset metadata or simulator ground
   truth; diagnose with native scores, node records, perception frames and episode videos. Report non-privileged and
   demo-assisted results separately.
5. Keep every scored episode; never delete failures. Interrupted episodes count as neither success nor failure.
6. Frozen programs make no Claude or other remote API call at run time (`semantic_calls == 0` in every counted
   episode; check with `tools/zero_api_check.py`).

## Data splits

| split | native seed | layouts | use |
|---|---|---|---|
| DEV | 0 | 0–7 | development set, free use by development agents; task score J |
| dev2 | 0 | 8–23 | robustness layouts, group 1 |
| dev3 | 1 | 0–39 | robustness layouts, group 2 (selection uses L10–39) |
| val (test layouts) | 1 | 40–47 | model selection; run only by the orchestrator |
| val2 (second test set) | 1 | 48–54 | model selection once a task's failures on val have been inspected |
| TEST | 2 | up to 50 per task | sealed final validation |

- The test sets are reserved for the orchestrator: `tools/queue.py enqueue` refuses val/val2 unless `AR_COORD=1`,
  and their execution records are not shown to development agents. Once a task's failures on val have been
  inspected, selection for that task switches to val2, with the incumbent also run on val2 for pairing.
- Episode wall-clock cap 45 minutes; a timeout is recorded as unresolved and counts as a failure with score 0.

## Task score J (per task, per version)

J = 100 × SR_DEV + mean native partial score (0–100) over DEV, so 0 ≤ J ≤ 200. With equal success counts, the partial
score decides. `tools/task_score.py` computes J only for versions whose counted episodes cover all DEV layouts.

## Search: fork-and-flush

- A workspace W is a branch directory `branches/<task>/<branch_id>/` (create with `tools/branch.py`): a full copy of
  the program, `NOTES.md`, and the branch's results. A development-agent session works in one W; a *flush* starts a
  fresh agent on the same W, which continues from `NOTES.md` and the completed evaluations (the notes are the only
  memory across sessions).
- New tasks start from the existing task with the closest skill requirements (its branch or frozen program), not from
  an empty skeleton; the candidate keeps that task's graph structure and skill implementations and changes only what
  the new task requires.
- Initial forks per task: K0 different starting points. Each gets b_explore iterations; the highest J becomes the
  incumbent.
- The incumbent is developed by an exploitation line (`tools/loop_next.py`) for b_exploit iterations. One iteration =
  change → screening on 3 DEV layouts → if the screening is not worse than the branch's best, full DEV → robustness
  layouts.
- Parallel exploration: up to two structurally different forks per task (`tools/fork_next.py`), each with its own
  workspace and budget b_explore. A fork that fails to surpass the incumbent is stopped; validated local changes of a
  fork may be transferred to the main line.
- Parameters K0=3, b_explore=2, b_exploit=4; the orchestrator may change them and logs why.
- Cross-task reuse: shared skills validated on one task (`agraph/`) are ported by the orchestrator to other tasks'
  incumbents, and rules confirmed by native evaluation are added to `docs/LESSONS.md` (the cross-task lessons file
  every new prompt cites).

## Multi-stage evaluation

1. Screening: the candidate runs on 3 DEV layouts.
2. Full development set: if the screening is no worse than the current best, the candidate runs on all 8 DEV layouts
   and is scored with J.
3. Robustness layouts: promising candidates run on the two groups of robustness layouts (dev2 and dev3 L10–39), with
   the incumbent run in the same time window and the two enqueued interleaved in groups of 3 layouts.
4. Paired comparison: the orchestrator runs the candidate and the incumbent on the same test layouts (val, or val2 once
   the task's val failures have been inspected) and decides adoption with the rules below.
5. Final validation: the selected program is evaluated once on the sealed TEST layouts.

## Adoption rules

Adoption is decided by the orchestrator, never by a development agent.

1. The challenger must be strictly better than the incumbent on the model-selection test set, without regression on
   the robustness layouts.
2. When the incumbent already succeeds on every test layout, the challenger must be strictly better on the robustness
   layouts.
3. Simulation is not step-deterministic. When two versions differ by at most four successes in a full evaluation of
   64 episodes, each must be rerun at least twice on the layouts where their outcomes disagree, and adoption uses the
   aggregate comparison rather than individual episodes.
4. Changes must be general mechanisms; no special cases keyed to layout ids.
5. After every adoption the orchestrator re-runs the new version in the deployment (policy-server) structure on
   robustness layouts (`tools/make_ps_branch.py`, then enqueue); it passes if successes differ by at most 1 from the
   same layouts in the development structure and no new failure type appears.
6. At the end each task's selected version is frozen into a content-addressed directory with `CANDIDATE.json`, and
   the sealed TEST layouts are run once.

## Skill and model restrictions

- Skills are pure code (geometry, inverse kinematics, rules, closed-loop checks) or existing generic pretrained tools.
  The only learned model used at run time is the segmentation model SAM 3 with public, frozen weights
  (`tools/sam3_service.py`). Training models on this project's data is not allowed.
- Depth comes only from the simulator depth image in the official observation.
- Finger state comes only from commanded gripper values; held / empty checks are visual (pure code on wrist
  and head images).
- Deployment structure: the controller runs in a separate policy-server process that exchanges observations and
  actions with the simulator only through the official client; it may depend only on what the official observation
  contains.
- Development agents may read the native scoring code and task configuration offline to understand success criteria,
  but must not hard-code configuration ranges or initial poses as run-time priors.

## Infrastructure rules

- All simulations go through one queue (`tools/queue.py`): agents only enqueue, never start Isaac directly. Processes
  are stopped only by PIDs recorded in `state/pids*.json`.
- Per-task fair share of GPU slots; per-task queue cap; at most two versions of one agent in the queue at once; a
  second pass only after the first pass finished.
- Every result batch is appended to `state/results.jsonl` and summarised by `tools/task_score.py`; adoptions are
  recorded in `state/incumbents.json`.

## Heartbeat (every 30 minutes)

The orchestrator is woken every 30 minutes (Claude Code scheduled prompt) and then:

1. checks agents (`tools/agent_tmux.sh list`), the queue and node status (`tools/queue.py status`) and services
   (`tools/services.sh status`); keeps the GPUs loaded;
2. for every new `REPORT.md`: runs the remaining evaluation stages and the paired comparison, applies the adoption
   rules, runs the policy-server check, updates `state/incumbents.json`, and flushes the line
   (`tools/loop_next.py`, `tools/fork_next.py` with an optional note) or stops it;
3. relaunches agents that exited abnormally (`tools/agent_tmux.sh resume`);
4. promotes validated shared skills across tasks and adds confirmed rules to `docs/LESSONS.md`;
5. regenerates the task scores (`tools/task_score.py`) and writes its log entry.
