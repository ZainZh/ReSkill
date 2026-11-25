# ReSkill: frozen task programs

Code release for **ReSkill: Selective Self-Improvement for Robot Manipulation**
([arXiv:2611.10510](https://arxiv.org/abs/2611.10510)).

This repository contains the ten frozen RoboDojo task programs reported in Table 1 of the paper.
Each program is the retained version whose recorded evaluations give the Table 1 result.
The files are byte-identical to the content-addressed snapshots that were evaluated; see
[Verifying the release](#verifying-the-release).

The development loop (the coding agent, evaluation queue, paired comparison and adoption tooling)
is not part of this release. These programs are its output: ordinary Python that runs without any
agent or language-model call.

## Programs

| Task | Snapshot | Table 1 successes | `graph.py` lines |
|---|---|---:|---:|
| `stack_blocks` | `e817b73e6998892a` | 31/32 | 473 |
| `stack_blocks_by_language` | `b8a260478925a255` | 27/27 | 1235 |
| `general_pickup` | `5eb933dbc932a3f4` | 51/57 | 791 |
| `stack_bowls` | `81c378aaca6f1850` | 21/22 | 799 |
| `build_tower` | `2010e662ed1cc314` | 37/37 | 574 |
| `classify_objects` | `9950b11220511205` | 22/25 | 1291 |
| `classify_objects_by_language` | `a2f8f05067982363` | 20/27 | 1775 |
| `organize_table` | `92e4230a4149b37b` | 15/36 | 2835 |
| `pack_objects_into_box` | `a9d7eb2c6afbd2d3` | 17/27 | 2777 |
| `store_tools_in_toolbox` | `133f59b339c421d1` | 41/47 | 1888 |

Successes are counted over the recorded robustness and model-selection evaluations
(seed 1, layouts 10–54), as in the paper. These include episodes used to select the
retained versions, so they measure performance during development rather than an
independent final validation.

## Layout of one program

```
programs/<task>/
  tasks/<task>/graph.py    object-centric task graph: nodes, conditions, edges, recovery routes
  tasks/<task>/params.json task parameters
  agraph/                  skill library: perception, kinematics/motion, manipulation (skills.py,
                           carry.py), verification, and the graph executor (contract.py)
  runner/                  RoboDojo integration: policy hook, episode lifecycle, input audit
  runtime_params.json      runtime parameters
  CANDIDATE.json           snapshot metadata, including the source sha256
```

Each program is self-contained and carries its own copy of `agraph/` and `runner/`, exactly as
it was frozen. Shared skill files are mostly identical across tasks; where a task improved a
shared skill, its copy differs. This duplication is deliberate: every program runs the code that
was evaluated.

## Inputs and run-time calls

At run time a program uses only head and wrist RGB-D images, camera calibration, joint states,
the robot model and the instruction text. It does not read simulator state, and the benchmark's
native criterion alone scores the episode. Object masks come from frozen SAM segmentation models.

The skill library still contains a client for an optional semantic query service
(`agraph/agent_bridge.py`, `agraph/mailbox.py`, `runner/rs_semantic_service.py`). None of the
ten retained programs used it: in all 337 recorded evaluation episodes, the logged number of
semantic calls and their cost were zero.

## Running

The programs plug into the RoboDojo evaluation client
([RoboDojo](https://arxiv.org/abs/2607.04434)) in one of two ways:

- `stack_blocks`, `stack_bowls`, `build_tower`, `classify_objects`, `organize_table`,
  `store_tools_in_toolbox`: `runner/bootstrap.py` starts the native client and registers
  `runner/deploy.py:eval_one_episode` as the policy, which builds the task graph from
  `tasks/<task>/graph.py` and executes it in-process.
- `general_pickup`, `stack_blocks_by_language`, `classify_objects_by_language`,
  `pack_objects_into_box`: `runner/bootstrap_ps.py` starts a separate policy-server process
  (`runner/ps_server.py`, `ps_model.py`); the task graph runs there through `runner/controller.py`,
  and `deploy.py` only forwards observations and actions over the official client.

At start-up `agraph/candidates.py` re-verifies every file against `CANDIDATE.json`, so the program
directories must be used unmodified (this includes three lint-cache files in `build_tower`).

Paths in the code are the defaults of the original evaluation machines. Set `ROBODOJO_ROOT` to
your RoboDojo checkout. Segmentation runs as a local service: the bundled SAM 2.1 service
(`agraph/vendor/sam_service.py`, endpoint `sam_endpoint` in `runtime_params.json`) for all tasks,
and additionally a SAM 3 service (`sam3_endpoint`, see `agraph/local_ground.py`) for
`general_pickup`, `stack_blocks_by_language`, `classify_objects`, `classify_objects_by_language`
and `pack_objects_into_box`. The SAM 3 service itself is not included. Some comments and task READMEs are in
Chinese and refer to development tooling that is not included here.

## Verifying the release

`SHA256SUMS` lists every file; `MANIFEST.json` maps each task to its snapshot and Table 1 result.

```
shasum -a 256 -c SHA256SUMS
```

The first 16 hex digits of each `CANDIDATE.json` `source_sha256` are the snapshot identifier above.

## Citation

```bibtex
@article{sun2026reskill,
  title   = {ReSkill: Selective Self-Improvement for Robot Manipulation},
  author  = {Sun, Zheng and Li, Zhihao and Li, Zhuo and Liu, Wenrui and Wang, Xin and Chen, Fei},
  journal = {arXiv preprint arXiv:2611.10510},
  year    = {2026}
}
```

> Note for maintainers: `programs/build_tower/tasks/build_tower/.ruff_cache/` contains its own
> `.gitignore`. Its three files are part of the verified snapshot and must be committed with
> `git add -f`.
