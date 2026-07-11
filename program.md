# gnn-autoresearch

Autonomous GNN research loop for the study: **"How well do different
message-passing choices mitigate over-squashing on graphs with bottlenecks?"**
(Tree-NeighborsMatch task, Alon & Yahav ICLR 2021 setting.)

## Setup

1. **Run tag**: work on branch `autoresearch/<tag>` (e.g. `jul11`). Create it from
   `main` if it does not exist.
2. **Read the in-scope files**: `README.md`, `prepare.py` (fixed harness, do not
   modify), `train.py` (the only file you edit).
3. **Initialize `results.tsv`** with just the header row if it does not exist.
   Leave it untracked by git.
4. The very first run establishes the **baseline**: run `train.py` unmodified.

## Experimentation

Each experiment runs `uv run train.py` with a **fixed total training budget of
240 seconds** (shared across radii 2-7). Launch it as:

```
EXPERIMENT_DESC="short description" uv run train.py > run.log 2>&1
```

The metric is **score_mean_acc** (mean root accuracy across radii, HIGHER is
better; chance level is 0.125). Extract results with:

```
grep "^acc_r\|^score_mean_acc" run.log
```

If the grep is empty, the run crashed: `tail -n 50 run.log`, fix if trivial,
otherwise log a crash and move on.

**What you CAN do:** modify `train.py` only. Architecture (GIN, GAT-style
attention, gating, skip/residual connections, a fully-adjacent last layer,
virtual nodes via extra adjacency, normalization, depth, width), optimizer,
LR schedule, batch size — all fair game. Scientifically interesting directions
for THIS study are architectural: the point is which message-passing mechanisms
beat the bottleneck, not just squeezing the optimizer.

**What you CANNOT do:**
- Modify `prepare.py` (`evaluate_accuracy` is the ground-truth metric, the time
  budget and radii are fixed).
- Add dependencies. Only what is in `pyproject.toml`.
- Peek at TEST_SEED data during training.

**Simplicity criterion:** all else equal, simpler is better. Weigh complexity
cost against improvement size.

## Logging

Log every experiment to `results.tsv` (tab-separated):

```
commit	score	acc_per_radius	status	description
a1b2c3d	0.4521	1.00/1.00/0.61/0.13/0.13/0.13	keep	baseline GCN
```

- status: `keep`, `discard`, or `crash` (score 0.000000 for crashes)
- MLflow tracking happens automatically inside `train.py` (server at
  http://127.0.0.1:5000). Always set `EXPERIMENT_DESC` when launching.

## The loop

LOOP FOREVER:

1. Check git state (branch, commit).
2. Modify `train.py` with one experimental idea. Keep a comment block at the top
   of the file noting what the current variant is.
3. `git commit` the change.
4. Run: `EXPERIMENT_DESC="..." uv run train.py > run.log 2>&1`
5. `grep "^acc_r\|^score_mean_acc" run.log`
6. Record in `results.tsv`.
7. If score improved (higher), keep the commit and advance.
8. If equal or worse, `git reset --hard` back to the previous good commit.

If a run exceeds 10 minutes, kill it and treat as a crash. Do not stop to ask
whether to continue; run until interrupted. If out of ideas: revisit
over-squashing literature you know (Alon & Yahav 2021; Topping et al. 2022
curvature rewiring; Gilmer et al. virtual nodes; GIN/Xu et al. expressiveness),
combine previous near-misses, or try more radical architectures. Prefer changes
that say something *scientific* about information propagation — each keep/discard
is a data point for the final report.
