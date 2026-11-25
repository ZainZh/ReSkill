# Controller interface (for development agents)

This describes the in-simulator controller harness used for development (`base/runner`, `base/agraph`, restored by
`base/materialize.py`). For deployment the same task graphs run in a separate policy-server process
(`tools/make_ps_branch.py`). It applies to any bundle under `code/branches/<task>/<id>/` (created by
`tools/branch.py create`). The rules in `../prompts/ORCHESTRATOR.md` take precedence.

## 1. How an episode is launched

```
queue.py enqueue --bundle branches/T/ID --task T --split dev --layouts 0-7     (--branch defaults to ID)
 → supervisor (simulation host) → tools/run_episode.py: freeze the bundle (frozen/<T>/<sha16>/), layout pre-check, launch gate
 → runner/bootstrap.py: native src/eval_client/main.py --task_name T --seed S, with only this one layout file allowed through
 → runner/deploy.py:eval_one_episode (called once, after the native reset)
     static input audit runner/audit.py review (agraph/, runner/, tasks/T/*.py, excluding test_*.py)
     read the native instruction → bind it at first observation (the same scene must later show the same instruction, otherwise control is not handed over)
     open the head camera cam_head and the wrist cameras cam_left_wrist / cam_right_wrist
 → runner/controller.py:run_controller → importlib('tasks.T.graph'), requires TASK_ID == T
     executor = GraphExecutor(build_graph(ctx)); executor.run(ctx)
 → after the controller returns: the harness holds in place (io.step()) until the native end_flag or action count = step_lim; the native evaluator scores the episode
```

- The bundle freezes only `agraph/ runner/ runtime_params.json tasks/__init__.py tasks/T/`. `NOTES.md`, `debug/` and `tests/` exist only in the workspace and are not visible at run time.
- Changing any file in the bundle produces a new sha (a new version); `tools/task_score.py` aggregates by (task, branch, sha).
- `tasks/T/params.json` is `ctx.params`; its `skills` and `perception` keys override the parameters of the same name in `runtime_params.json`.

### Graph contract (agraph/contract.py)
- `TASK_ID: str`, `build_graph(ctx) -> Graph(task, nodes, start, edges, max_stagnant_visits, max_transitions, unlimited=False)`.
- `Node(name, run, ready=lambda ctx: PASS, postcondition='', requires=())`; `run(ctx) -> NodeResult(status, cause, evidence, actions, progress)`.
  `status` must be one of `PASS / FAIL / UNKNOWN`; `edges[name] = {PASS: next, FAIL: ..., UNKNOWN: ...}`, and a target may be `DONE` or `ABORT`.
- Termination: reaching `DONE`/`ABORT`; running out of actions (`BudgetExhausted` or `ctx.remaining_actions() <= 0`);
  a node failing to PASS more than `max_stagnant_visits` consecutive times, or total visits exceeding `max_transitions` (→ ABORT).
  A node that raises `EvidenceUnavailable` (including `TargetNotObserved`) is recorded as UNKNOWN.
- The controller **never** claims success (`controller_claims_success=None`); only the native score counts. All remaining actions after termination are in-place holds,
  so tasks that require returning to the home pose must call `home()` themselves before `DONE`.
- In a skeleton task (created by branch.py from a source that has no graph for that task), `graph.py` captures a single frame and goes to `DONE`. Its native score should be 0; it is used to validate the whole pipeline first.

## 2. Allowed inputs (only these; episodes that fail the audit are invalid)

| Source | Access | Content |
|---|---|---|
| Instruction | `ctx.instruction` | Native instruction text (a fixed template per task; `*_by_language` and `general_pickup` contain scene-dependent category/colour/target descriptions) |
| Head RGB-D | `ctx.perception.observe(stage)` | `rgb` uint8 HxWx3, `depth` float32 (simulator depth image from the official observation, distance to the image plane in m), intrinsics `K`, `T_world_camera`; frames are saved to `RUN_DIR/perception/frame_NNNNNN/` |
| Wrist RGB-D | `ctx.perception.observe_wrist(arm, stage)` | Same as above, with the frame's own K/pose; the pose is cross-checked against encoder FK (>5 cm discrepancy marks it unusable) |
| Segmentation | `agraph/local_vision.py` (`Sam3Client`) | SAM 3 masks on the observation images (loopback service, §5) |
| Encoders + URDF | `ctx.robot.tcp(arm)`, `base_transform(arm)`, `action_count`, `action_limit`; `ctx.skills.io.joints(arm)`, `link_transforms(arm, q)`, `hand_collision_world_hulls()`, `jaw_span(arm)` | World-frame 4x4; arm ∈ {`left`,`right`} |
| Table prior | `runtime_params.json skills.table_z_prior = 0.74` | Used only as a height lower bound / fallback support surface; run-time depth measurements take precedence |

Finger state comes only from the commanded gripper values; held / empty checks are visual.
Forbidden: simulator state (object poses, labels, ground-truth segmentation), `Assets/`, layout JSON, `task/RoboDojo/config|tasks`, native results, `HARNESS_BINDING.json`;
no `import env / task / src`; control code (.py) must not contain `success`, `seed_info`, `scene_layout`, `get_score`, `reward_manager`, etc.
(full list in `runner/audit.py FORBIDDEN_TOKENS/STRINGS/PATHS`). No run-time API calls (`ctx.agent` must not be used; checked by `tools/zero_api_check.py`).
Offline debugging tools go in `W/debug/`; they may read simulator logs but must not be imported at run time.

## 3. Execution interface and action semantics

- **One native action** = one `take_action`: 6 joint targets for each of the two arms + 1 gripper command each, natively interpolated over 10 physics steps (dt 0.004 s),
  i.e. 1/25 s. Both arms are commanded on every step: an arm that is not moving holds its previous target.
- Low level `ctx.skills.io` (`agraph/vendor/robot_io.py: RobotIO`):
  - `move(arm, T_tcp, grip)`: IK + FK check (position ≤6 mm, orientation ≤0.06 rad); the joint delta is **clipped to 0.10 rad/action** on its largest component and 1 action is issued;
    long moves require multiple calls. Raises `IKFailure` on failure. Planning and execution should share the URDF damped-least-squares IK of `agraph/kinematics.py`,
    seeded at the current joints and constrained to the real joint limits.
  - `step()`: issues 1 more action at the current target (hold / wait for the gripper to close). `q[arm]` and `g[arm]` are the current targets and can be modified directly before calling `step()`.
  - Gripper command `grip`: **1 = fully open, 0 = fully closed**.
- Skill layer `ctx.skills` (`agraph/skills.py: RobotSkills`, parameters in `runtime_params.json skills`): `pick(obj, arm)`, `verify_held(arm, query)`,
  `transport(arm, xyz)`, `place(arm, xyz, release, drop_height)`, `push(arm, start, end, contact_height)`, `release(arm)`,
  `retreat(arm, h)`, `home(arm=None)`, `reachable(arm, xyz)`, `plan_route(...)`, `plan_carry/execute_carry`, `clear_arm`, `update_world`, etc.
  All return a `NodeResult` with `actions` and `evidence['arm']`. `handover` is not implemented in the shared library (returns FAIL `skill_unsupported_v1`, 0 actions).
- Planning cost: route search can take very long in `mode: unlimited`; use a bounded mode with time tiers in `params.json search`, or bypass it and use IK + simple paths directly.

### Action budget (native `step_lim`; hard constraint, cannot be changed)

| Task | step_lim | Full score requires both arms back at home (all_robot_back_to_origin) |
|---|--:|---|
| classify_objects | 1100 | Yes |
| classify_objects_by_language | 1100 | Yes (grippers must also be open) |
| pack_objects_into_box | 1300 | Yes |
| organize_table | 1000 | Yes |
| build_tower | 1050 | Yes |
| stack_blocks | 550 | Yes |
| stack_blocks_by_language | 400 | Yes |
| stack_bowls | 800 | Yes |
| store_tools_in_toolbox | 900 | Yes |
| general_pickup | 200 | No (success as soon as the target is lifted 10 cm; the episode ends immediately) |

Once the native success condition is met, `end_flag` is set and the episode ends early. Tasks with tight budgets
(stack_blocks_by_language 400, general_pickup 200) must run at demonstration speed, with almost no room for retries.

Known pitfalls: planning from the first frame misses objects occluded by the robot's own gripper (move the arms out of
view first or rescan with a wrist camera); PASS signals inside the graph cannot substitute for the native score.

## 4. Inspecting failed episodes

`queue.py results --branch ID --task T --json` or `runs/<T>/<split><II>/<job>_a<n>/RESULT.json`, which contains all paths:

| File | Content |
|---|---|
| `RESULT.json` / `RUN.json` | Status (verified/infra/unresolved/interrupted), native score and success, action count, wall-clock time, failure type, launch-gate wait |
| `video_trace/trace.mp4` | **Look at this first**: visual trace of the whole episode, one frame every 10 native actions (left: head; top right/bottom right: left/right wrist), action count in the top-left corner, played at 5 fps |
| `video_trace/f_NNNNN.jpg`, `index.jsonl` | Individual frames of the same trace (NNNNN = native action count at that moment, `_end` = native terminal frame); can be aligned with `step` in `trace.jsonl` |
| `eval_result/**/*.mp4` | Native evaluation videos (head/wrist, native resolution) |
| `perception/frame_NNNNNN/rgb.png` (plus crops, masks) | Every frame the controller actually saw |
| `trace.jsonl` | Per-event log: nodes, perception, planning diagnostics (each line has `step` = native action count at that moment) |
| `graph_trace.json` | Every node visit: status, cause, evidence, actions, elapsed_s |
| `result.json` | Deploy-side episode record (controller exception traceback, control_actions, terminal_hold_actions) |
| `process.log` | Native process output (crashes, PhysX errors) |
| `CONTROL_SOURCE_AUDIT.json`, `RUNTIME_AUDIT.json` | Input audits |

The visual trace is recorded by the harness inside the simulator process (`tools/simsite/ar_video_trace.py`; not part
of the bundle, and it changes neither the controller's inputs nor the native score), and applies to all branches
automatically; the interval is set by the environment variable `AR_VIDEO_EVERY` (supervisor environment, 0 = off).

## 5. Services

- **SAM 3 (local to each simulation host)**: `http://127.0.0.1:18891` (`tools/sam3_service.py`, started by
  `tools/services.sh start sam3`). `POST /segment` (single-box instance mask), `/detect` (text-prompted detection),
  `/ground` (query-table grounding); `GET /health`. In a controller, use `agraph.local_vision.Sam3Client`.
- When the service is unhealthy, the supervisor does not dispatch new episodes; run_episode also pre-checks before an
  episode starts (a failed pre-check is recorded as `infra` and re-queued).

## 6. Time limits and resources

- Per-episode wall-clock limit **45 minutes** (counted from simulator process start, including Isaac startup and
  teardown); a timeout is recorded as `unresolved` and counts as 0 in J.
- Isaac launches are serialised per GPU on the same host (launch gate: a minimum gap between launches, and the gate is
  held until shortly after the previous one reaches app ready); the gate wait does not count toward the 45 minutes.
- Native crashes (Kit crash reporter) are killed within seconds; a crash before the controller starts is recorded as
  `infra` and retried automatically once.

## 7. Demonstration data (privileged_demo_assisted branches only)

Public RoboDojo demonstrations (joint/end-effector trajectories and video; no depth, no object poses, no seed) may be
placed under `demos/` in `$RESKILL_ROOT`. Branches developed with reference to demonstrations must be marked
`privileged_demo_assisted` and are accounted for separately. Demonstrations are only for offline development (stage
structure, grasp heights, pacing, placement points and similar parameters); the run-time controller **must not read
`demos/`**; any values needed are written into `params.json` with their source noted.
