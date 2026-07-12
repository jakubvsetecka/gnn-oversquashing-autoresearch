# gnn-autoresearch

Controlled study of **over-squashing / information propagation** in message-passing
GNNs, in the spirit of [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
an autonomous agent iterates on `train.py` under a fixed wall-clock training budget,
keeping changes that improve the metric and discarding the rest.

**Research question:** How well do different message-passing choices mitigate
over-squashing on graphs with bottlenecks?

**Task:** Tree-NeighborsMatch (Alon & Yahav, ICLR 2021). A perfect binary tree of
radius r; the root must retrieve the label of the leaf whose key matches the
root's query. The information required at the root grows exponentially with r.

**Files:**

- `prepare.py` — fixed harness: dataset generation, evaluation, constants. Not modified.
- `train.py` — model + training loop. The single file the agent edits (HEAD holds the winning variant).
- `program.md` — instructions for the autonomous research agent.
- `results.tsv` — log of all 17 loop experiments (keep/discard/crash), one git commit each.
- `replication.tsv` — seed-replication study: the 6 headline variants re-run with 4 fresh seeds each.
- `NOTES.md` — the agent's experiment log and scientific interpretation.
- `analysis.py` — regenerates the report figures and seed statistics (`uv run analysis.py`).
- `variants/` — snapshot of `train.py` for every experiment in `results.tsv`
  (numbered by experiment), so no git-history access is needed to reproduce any
  variant: copy one over `train.py` and run `uv run train.py`.

**Metric:** `score_mean_acc`, mean root accuracy over radii 2-7 on a fixed held-out
set (higher is better, chance = 1/8). Total training budget fixed at 240 s per
experiment, split equally across radii.

**Tracking:** every run logs per-radius accuracy to a local MLflow server
(`http://127.0.0.1:5000`).

```bash
uv sync
uv run train.py                       # one baseline experiment (~4 min)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
