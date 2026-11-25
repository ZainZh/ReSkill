# ReSkill development loop

This directory contains the code and agent instructions of the iterative self-improvement loop that produces the
frozen task programs in [`../programs/`](../programs). A long-running **orchestrator** agent and several
**development agents** (Claude Code sessions) revise task programs between episodes, evaluate every candidate
through a shared GPU job queue on RoboDojo, and adopt a candidate only after multi-stage evaluation and a paired
comparison with the incumbent. The paper's Methods section describes the loop; this README shows where each part
lives in the code.

## Requirements

This is research code. It shows how the loop works and can be run with the following components:

- a RoboDojo / Isaac Sim installation (`ROBODOJO_ROOT`) with its conda environment;
- one or more GPU hosts that share a file system holding the loop directory (`RESKILL_ROOT`);
- the loopback SAM 3 segmentation service on every simulation host (`tools/sam3_service.py`; it needs the SAM 3
  source and public weights plus an offline grounding query bundle that is not included here);
- the Claude Code CLI (`claude`) with credentials on the orchestration host. Both the orchestrator and the development
  agents run on it. The frozen programs make no API calls at run time.

The tests in `tools/tests/` run on CPU without a simulator, except for the layout pre-check tests, which need the
RoboDojo layout files.

## Layout

```
devloop/
  README.md
  prompts/
    ORCHESTRATOR.md          specification for the orchestrator: rules, splits, J, fork-and-flush, multi-stage
                             evaluation, adoption rules, model restrictions, 30-minute heartbeat
    DEVELOPMENT_AGENT.md     prompt template of a development agent (workspace assignment + per-session prompt)
  docs/
    CONTROLLER_INTERFACE.md  controller contract, allowed inputs, action semantics, services, failure inspection
    LESSONS.md               the cross-task lessons file maintained by the orchestrator
  tools/
    arconf.py                paths, tasks, data splits, sealed final-validation lists, timeouts, service endpoint
    branch.py                create/list candidate workspaces branches/<task>/<id>/
    loop_next.py             flush the exploitation line of a task (next session, previous REPORT attached)
    fork_next.py             start/flush structurally different fork lines (parallel exploration, pruning)
    agent_tmux.sh            start/list/stop/resume headless development agents in tmux
    agent_run.sh             wrapper around `claude -p` for one agent session
    queue.py                 file-based job queue + per-host GPU-slot supervisor (fair share per task)
    queue_ctl.sh             start/stop/status of the supervisor on a GPU host
    pidreg.py                registry of started processes (the queue stops processes only by recorded PID)
    run_episode.py           one native episode of a bundle: freeze, preflight, launch, verify -> RESULT.json
    run_episode_ps.py        same for policy-server bundles
    make_ps_branch.py        derive a policy-server (deployment-structure) branch from an adopted version
    task_score.py            per-version aggregation and the task score J
    zero_api_check.py        verifies zero run-time API calls per episode
    services.sh              start/stop/status of the SAM 3 service
    sam3_service.py, sam3_v2_queries.json   SAM 3 segmentation service and its per-task query table
    simsite/                 simulator-process hooks: visual trace every 10 actions, GC crash guard
    tests/test_infra_tasks.py
  base/                      seed workspace (template that branch.py copies)
    materialize.py           restores the shared library (agraph/, runner/, runtime_params.json) from ../programs
    SHARED_FILES.json        file -> frozen program it is byte-identical to, with sha256
```

### Configuration

| variable | meaning | default |
|---|---|---|
| `RESKILL_ROOT` (or `AR_EVIDENCE`) | loop directory holding `code/` (this directory), `runs/`, `queue/`, `frozen/`, `mailbox/`, `instruction_bindings/`, `deployment/`, `envs/` | two levels above `tools/` |
| `ROBODOJO_ROOT` | RoboDojo checkout (layouts under `Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/`) | `~/github/RoboDojo` |
| `AR_PYTHON`, `AR_CONDA_SH`, `AR_RUNTIME_ENV` | simulator Python, conda activation script, env file sourced before the simulator | `~/miniconda3/...` |
| `AR_SAM_ENDPOINT` | SAM 3 service checked before every episode | `http://127.0.0.1:18891` |
| `AR_SAM3_EVAL`, `AR_SAM3_PY`, `AR_SAM3_GPU` | SAM 3 source/weights/query bundle, its Python environment, its GPU | under `$RESKILL_ROOT` |
| `AR_RS_HOST` | host name of the orchestration host (per-host PID registry naming) | unset |
| `AR_AGENT_MODEL` | model of the development agents | `claude-opus-5-5` |
| `AR_COORD=1` | marks the orchestrator: only then does the queue accept the test sets val / val2 | unset |

To use the directory, check it out as `$RESKILL_ROOT/code` and run `python3 base/materialize.py` once. This copies the
shared library from `../programs` into `base/`.

## How the paper's concepts map to the code

| Paper concept | Where it lives |
|---|---|
| **Task program** P = (G, Π, Λ) | A *bundle*: `agraph/` (skill library Π, graph contract), `runner/` (harness), `tasks/<T>/graph.py` (task graph G), `tasks/<T>/params.json` + `runtime_params.json` (bindings, parameters, budgets Λ). Only these paths are hashed and frozen (`agraph/candidates.py manifest`). |
| **Task graph construction / candidate creation** | The development agent writes `tasks/<T>/graph.py` (`TASK_ID`, `build_graph(ctx) -> Graph`) against the contract in `agraph/contract.py` (`Node`, `NodeResult` with `PASS/FAIL/UNKNOWN`, `edges`, `max_stagnant_visits`, `max_transitions`, `GraphExecutor`). `tools/branch.py create` starts a candidate from the incumbent, from the closest existing task (any bundle directory such as `../programs/<task>`), or from a skeleton graph (`--from base`). See `docs/CONTROLLER_INTERFACE.md` §1–3. |
| **Node records / failure localisation** | `GraphExecutor` records every node visit (status, cause, evidence, actions, elapsed time) in `RUN_DIR/graph_trace.json`; `trace.jsonl` holds events with the native action count; `tools/simsite/ar_video_trace.py` records a multi-view visual trace every 10 actions (`video_trace/trace.mp4`); `perception/frame_*` holds what the controller actually saw. The development agent follows the graph to the earliest node that deviates (`prompts/DEVELOPMENT_AGENT.md`, rule 7 and the root-cause section). |
| **Task-level vs skill-level revision** | Task level: edits to `tasks/<T>/graph.py` (stages, hand-off conditions, bindings, recovery routes). Skill level: edits to `agraph/*.py` (e.g. `skills.py`, `perception.py`, `kinematics.py`, `carry.py`) behind the same interface. Both produce a new bundle sha. The orchestrator ports validated shared skills to other tasks' incumbents. |
| **Input restrictions / audit** | `runner/audit.py` (static check of every controller `.py`, forbidden tokens and paths), run-time audit reports `CONTROL_SOURCE_AUDIT.json` / `RUNTIME_AUDIT.json`, environment boundary in `runner/deploy.py`. `tools/run_episode.py` marks a verified episode `unresolved` if the audit did not pass. |
| **Multi-stage evaluation: 3-layout screening → full development set** | Enqueue 3 DEV layouts, then `--layouts 0-7` (`tools/queue.py enqueue`). `tools/task_score.py` lists versions with incomplete DEV coverage as screening rows and computes J only for complete DEV. |
| **Task score J = 100·SR_DEV + mean native partial score** | `tools/task_score.py summarise`: per-layout means (repeated runs averaged first), `J = 100 * success_rate + mean_score`; 45-minute watchdog timeouts count as 0, infrastructure failures are excluded (`arconf.effective_status`). |
| **Two groups of robustness layouts** | `arconf.SPLITS`: `dev2` (seed 0 L8–23) and `dev3` (seed 1 L0–39; selection uses L10–39). |
| **Test layouts for model selection, hidden from development agents; second test set** | `val` (seed 1 L40–47) and `val2` (seed 1 L48–54). `tools/queue.py enqueue` refuses them unless `AR_COORD=1` (the orchestrator), and development agents are told never to use them. Selection switches to `val2` once a task's `val` failures have been inspected. |
| **Final validation on unused layouts** | Sealed TEST (`arconf.TEST_LAYOUTS`, seed 2): `run_episode.py` requires `--unseal-test` and the queue never accepts it. |
| **Paired comparison and adoption rules** | Written as rules in `prompts/ORCHESTRATOR.md` ("Multi-stage evaluation", "Adoption rules": strict improvement on the model-selection test set without regression on robustness layouts; strictly better on robustness layouts when the incumbent already succeeds on every test layout; ≥ 2 reruns of disagreeing layouts and an aggregate decision when versions differ by ≤ 4 of 64). The orchestrator carries them out with `tools/queue.py enqueue/results` (candidate and incumbent interleaved in groups of 3 layouts) and `tools/task_score.py`; adoptions are recorded in `state/incumbents.json`. |
| **Parallel exploration** | Fork-and-flush: `tools/fork_next.py` (up to two structurally different forks per task, stopped when they fail to surpass the incumbent within their budget) next to the exploitation line `tools/loop_next.py`. |
| **Orchestrator vs development agents** | Orchestrator: an interactive Claude Code session following `prompts/ORCHESTRATOR.md`, woken every 30 minutes by a scheduled prompt (heartbeat section). Development agents: headless `claude -p` sessions started by `tools/agent_tmux.sh` (`tools/agent_run.sh`), each working only in its own workspace `branches/<task>/<id>/` and ending each session with `REPORT.md`. |
| **Notes as cross-session memory** | `branches/<task>/<id>/NOTES.md` (template in `tools/branch.py`; a child branch receives `PARENT_NOTES.md`). A flush (`loop_next.py`, `fork_next.py`, `agent_tmux.sh resume`) starts a new session in the same workspace; the new prompt also carries the previous `REPORT.md`. |
| **Cross-task lessons file** | `docs/LESSONS.md` (maintained by the orchestrator, cited by every new prompt). |
| **Shared job queue on GPUs; per-episode records** | `tools/queue.py` (jobs, per-task fair share, cancel, infra retry), `tools/queue_ctl.sh`, `tools/run_episode.py` (RUN_DIR with native result, `RESULT.json`, `graph_trace.json`, visual trace, audit reports). |
| **No run-time agent calls** | `tools/zero_api_check.py` (four independent counters per episode). |
| **Policy-server deployment** | `tools/make_ps_branch.py` and `tools/run_episode_ps.py`. The policy-server layer (`runner/ps_server.py`, `agraph/grasp_verify.py`, ...) ships inside the programs in `../programs/`. |
| **Freezing into a content-addressed program** | `tools/run_episode.py ensure_frozen` freezes an enqueued bundle into `frozen/<task>/<sha16>/<task>/v001` with `CANDIDATE.json` (per-file sha256, read-only files; `agraph/candidates.py freeze/verify`); a bundle that already carries a `CANDIDATE.json` is verified in place. The hash is re-verified before and after every episode. The directories in `../programs/` are such frozen programs. |

## How to run the loop

Paths are relative to `$RESKILL_ROOT/code`.

```bash
# 0. once: restore the shared library of the seed workspace
python3 base/materialize.py --programs /path/to/ReSkill/programs

# 1. SAM 3 service on every simulation host
tools/services.sh start sam3

# 2. queue: shared config, then one supervisor per GPU host
python3 -B tools/queue.py config --gpus 0-7 --slots-per-gpu 4
tools/queue_ctl.sh start --gpus 0-7 --slots-per-gpu 4

# 3. orchestrator: create a candidate workspace from the closest existing task (or the incumbent / a skeleton),
#    then start a development agent on it
python3 -B tools/branch.py create --task stack_blocks --id A_lean --from /path/to/ReSkill/programs/stack_blocks_by_language --note "lean act-observe-retry"
tools/agent_tmux.sh start loop.stack_blocks.s1 state/prompts/loop/stack_blocks.md   # filled from prompts/DEVELOPMENT_AGENT.md

# 4. development agent (inside its workspace): screening, full DEV, robustness layouts
python3 -B tools/queue.py enqueue --bundle branches/stack_blocks/A_lean --task stack_blocks --split dev --layouts 0,3,6 --tag it1_screen
python3 -B tools/queue.py enqueue --bundle branches/stack_blocks/A_lean --task stack_blocks --split dev --layouts 0-7 --tag it1_full
python3 -B tools/queue.py enqueue --bundle branches/stack_blocks/A_lean --task stack_blocks --split dev3 --layouts 10-39 --tag it1_rob
python3 -B tools/queue.py results --branch A_lean --task stack_blocks
python3 -B tools/task_score.py --task stack_blocks --print

# 5. orchestrator heartbeat (every 30 min): agents, queue, new reports
tools/agent_tmux.sh list
python3 -B tools/queue.py status
#    paired comparison on the model-selection test set (orchestrator only)
AR_COORD=1 python3 -B tools/queue.py enqueue --bundle <candidate> --task stack_blocks --split val --layouts 40-47 --priority 5 --tag sel_cand
AR_COORD=1 python3 -B tools/queue.py enqueue --bundle <incumbent> --task stack_blocks --split val --layouts 40-47 --priority 5 --tag sel_inc
#    after adoption: zero-API check and deployment-structure check
python3 -B tools/zero_api_check.py --tag sel_cand
python3 -B tools/make_ps_branch.py --task stack_blocks --source <frozen dir> --id ps_<name> --ps-base <policy-server layer>
#    next session of the line, or a new fork
python3 -B tools/loop_next.py loop.stack_blocks.s1 note.md
python3 -B tools/fork_next.py start stack_blocks F1 direction.md

# 6. final: freeze (already done at enqueue) and the single sealed TEST run per selected version
python3 -B tools/run_episode.py --freeze-only --bundle branches/stack_blocks/<id> --task stack_blocks
python3 -B tools/run_episode.py --bundle ../frozen/stack_blocks/<sha16>/stack_blocks/v001 --task stack_blocks \
    --split test --layout <L> --gpu 0 --out runs/stack_blocks/test<L> --unseal-test
```
