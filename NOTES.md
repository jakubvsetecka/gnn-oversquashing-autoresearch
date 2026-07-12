# Autoresearch session notes (branch autoresearch/jul11, 2026-07-11)

Study: how well do different message-passing choices mitigate over-squashing
on Tree-NeighborsMatch (Alon & Yahav 2021)? Metric: score_mean_acc, mean root
accuracy over radii 2-7, 240 s total training budget. Chance = 0.125.

Baseline: 0.279134. Final best: **1.000000** (commit 9ebb46d).

## Experiment log (16 runs)

| # | commit | score | per-radius acc (r2..r7) | status | change |
|---|--------|-------|-------------------------|--------|--------|
| 0 | 89cbcfd | 0.2791 | 0.64/0.32/0.27/0.19/0.14/0.13 | keep | baseline GCN (r+1 sym-norm layers) |
| 1 | 2557db2 | 0.3070 | 0.78/0.31/0.27/0.19/0.16/0.13 | keep | fully-adjacent (uniform mean) last layer |
| 2 | 5fbbe09 | 0.3212 | 1.00/0.33/0.20/0.16/0.12/0.12 | keep | GIN sum aggregation + 2-layer MLPs |
| 3 | b2396e2 | 0.4663 | 1.00/1.00/0.26/0.23/0.17/0.14 | keep | + residual connections |
| 4 | 4d469d3 | 0.8549 | 1.00 x5 /0.13 | keep | FA layer -> dot-product self-attention |
| 5 | f0b717f | 0.8550 | 1.00 x5 /0.13 | keep | tree depth r+1 -> constant 2 layers |
| 6 | 224f5cb | 0.8566 | 1.00 x5 /0.14 | keep | 0 tree layers (pure attention retrieval) |
| 7 | 96a7419 | 0.8532 | 1.00 x5 /0.12 | discard | HIDDEN 64 -> 128 (hurt: fewer steps, no r7 gain) |
| 8 | 9fd1a4a | 0.8577 | 1.00 x5 /0.15 | keep | GPU-side train-batch sampler (more steps) |
| 9 | 1ba95b9 | 0.8675 | 1.00 x5 /0.21 | keep | tied Q=K projection (task-symmetric similarity) |
| 10 | a9b86ed | 0.9999 | 1.00 x6 | keep | + raw-feature bias tau*(X X^T) in attention logits, diagonal masked |
| 11 | 9ebb46d | **1.0000** | 1.00 x6 | keep | drop learned QK entirely: logits = tau*(X X^T), diag masked (WINNER) |
| 12 | 54cebfb | 0.9984 | 1.00 x5 /0.99 | discard | ablation: winner without diagonal mask |
| 13 | 9d30db1 | 0.8406 | 1.00/1.00/1.00/1.00/0.90/0.14 | discard | ablation: plain baseline GCN + only last layer swapped to raw-X attention |
| 14 | d8f7be0 | 0.3298 | 0.99/0.31/0.24/0.17/0.14/0.13 | discard | ablation: baseline + virtual node (Gilmer) |
| 15 | b02bc16 | 0.3203 | 1.00/0.32/0.21/0.15/0.12/0.12 | discard | ablation: GAT-style attention restricted to tree edges |
| 16 | 923766e | 0.8612 | 1.00 x5 /0.17 | discard | ablation: tied-QK + diag mask WITHOUT raw-X bias (attribution check) |

## Winning architecture (9ebb46d)

embed(X) -> one global attention layer with logits = tau * (X X^T)
(tau learnable, diagonal masked to -inf) over v(h), residual + 2-layer MLP,
readout at root. No tree message passing at all. GPU-side training sampler
(same distribution as prepare.make_batch, separate RNG stream; evaluation
uses the untouched prepare.evaluate_accuracy).

## Scientific interpretation

1. **Over-squashing is real and severe for local MP.** The plain GCN decays
   from 0.64 (r2) to chance (r7): the root's fixed-width vector must compress
   2^r leaf (key,label) pairs.

2. **Uniform global shortcuts barely help.** An FA mean last layer (+0.03) and
   a virtual node (+0.05) both give the root a 1-2 hop path to every leaf, but
   a *uniform* aggregate is itself a bottleneck: the root receives the average
   of 2^r leaves, still O(2^r) items squeezed into one vector. Global
   *connectivity* without *selectivity* does not fix over-squashing on
   retrieval-style tasks.

3. **Adaptive local weighting does not help either.** GAT-style attention on
   tree edges only (0.320) matches GIN, far below global attention: however
   messages are weighted, all information still has to pass through the same
   O(1)-width path near the root. Over-squashing is a topological capacity
   problem, not a weighting problem.

4. **Content-based global selection is the decisive mechanism.** Replacing the
   uniform FA layer with dot-product self-attention jumped 0.47 -> 0.85,
   solving r2-r6 outright. Attention lets the root *select* the one relevant
   leaf instead of averaging all of them: the amount of information crossing
   the bottleneck drops from O(2^r) to O(1).

5. **Once a global retrieval layer exists, tree message passing is useless
   here.** Cutting the r+1 tree layers to 2 and then to 0 never hurt (scores
   ticked up via more optimization steps). Conversely, keeping the deep GCN
   stack *in front of* the attention layer (ablation 13) actively hurt r6-r7:
   deep pre-smoothing corrupts the leaf representations the retrieval layer
   needs, and burns the step budget.

6. **The residual finding (exp 3, +0.145)** shows part of the "baseline
   failure" is trainability, not only information capacity: residuals let
   deeper stacks optimize within the budget (r3 went 0.33 -> 1.00).

7. **The last mile at r7 was an alignment/learning-speed problem, not
   capacity.** Learned attention (even tied Q=K) was slowly *learning* the
   query-key correspondence at r7 (0.13 -> 0.21 with more steps/tying) but the
   budget was too small for 128 keys. Exploiting the task symmetry - query and
   key one-hots share feature indices, so raw X X^T is exactly 1 for the
   (root, matching-leaf) pair and 0 for other leaves - makes correct retrieval
   available at initialization; only the label decoder must be learned, and r7
   goes to 1.00. Ablation 16 confirms the attribution: with the diagonal mask
   but without the raw-X bias, r7 stays at 0.17, so the bias term (not the
   mask) is causal. The mask itself is worth little (ablation 12: 0.998
   without it - the model learns around root self-attention).

8. **Width was not the binding constraint** (exp 7): doubling HIDDEN to 128 to
   "fit" 128 keys did nothing for r7 and cost throughput everywhere.

Overall narrative for the report: on bottlenecked graphs the ranking is
local MP (any flavor: GCN/GIN/local GAT) << uniform global shortcuts
(FA layer, virtual node) << content-based global attention, and with an
inductive bias that aligns the attention similarity with the task's key-query
structure the task is solved exactly at every radius within budget.

## Bookkeeping

- Kept chain: 89cbcfd -> 2557db2 -> 5fbbe09 -> b2396e2 -> 4d469d3 -> f0b717f
  -> 224f5cb -> 9fd1a4a -> 1ba95b9 -> a9b86ed -> 9ebb46d (HEAD).
- Discarded commits remain in reflog; results.tsv has all 17 rows (incl. baseline).
- All runs logged to MLflow experiment "oversquashing" at http://127.0.0.1:5000.
- No crashes, no OOM (6 GB VRAM never stressed).
