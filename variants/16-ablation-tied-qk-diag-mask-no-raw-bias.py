"""Model + training loop for the over-squashing study. THIS is the file agents edit.

Current variant: ABLATION - tied-QK attention + diagonal mask, NO raw-X bias.
Disentangles exp10's two simultaneous changes: keeps the learned tied-QK
similarity and the -inf diagonal mask, drops the raw-feature bias term.
If this solves r7, the mask (removing trivial self-attention) was the key;
if not, the credit belongs to the task-symmetric raw-X similarity bias.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DEVICE,
    NUM_CLASSES,
    PER_RADIUS_BUDGET,
    RADII,
    evaluate_accuracy,
    feature_dim,
    make_batch,
    tree_adjacency,
)

torch.manual_seed(0)

HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 128
NUM_TREE_LAYERS = 0


class GCN(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        A = tree_adjacency(r)  # raw adjacency, sum aggregation
        self.register_buffer("A", A)
        n = A.shape[0]
        self.register_buffer("A_full", torch.ones(n, n) / n)  # FA layer adjacency
        self.eps = nn.Parameter(torch.zeros(NUM_TREE_LAYERS))
        self.q = nn.Linear(HIDDEN, HIDDEN)
        self.k = nn.Linear(HIDDEN, HIDDEN)
        self.v = nn.Linear(HIDDEN, HIDDEN)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.embed = nn.Linear(feature_dim(r), HIDDEN)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN)
            )
            for _ in range(NUM_TREE_LAYERS + 1)
        )
        self.out = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, X):
        h = self.embed(X)
        last = len(self.layers) - 1
        n = X.shape[1]
        eye = torch.eye(n, device=X.device, dtype=torch.bool)
        for i, mlp in enumerate(self.layers):
            if i == last:
                qk = self.q(h)  # tied query/key projection
                logits = (qk @ qk.transpose(-2, -1)) / HIDDEN**0.5
                logits = logits.masked_fill(eye, float("-inf"))
                attn = torch.softmax(logits, dim=-1)
                agg = attn @ self.v(h)
            else:
                agg = self.A @ h + (1 + self.eps[i]) * h
            h = h + F.relu(mlp(agg))
        return self.out(h[:, 0])  # prediction at the root


def fast_batch(r: int, batch_size: int, generator: torch.Generator):
    """GPU-side equivalent of prepare.make_batch (train stream only)."""
    n = 2 ** (r + 1) - 1
    L = 2**r
    first_leaf = L - 1
    Fdim = feature_dim(r)
    dev = generator.device
    X = torch.zeros(batch_size, n, Fdim, device=dev)
    keys = torch.rand(batch_size, L, device=dev, generator=generator).argsort(-1)
    labels = torch.randint(
        0, NUM_CLASSES, (batch_size, L), device=dev, generator=generator
    )
    target_leaf = torch.randint(0, L, (batch_size,), device=dev, generator=generator)
    batch_idx = torch.arange(batch_size, device=dev)
    query = keys[batch_idx, target_leaf]
    y = labels[batch_idx, target_leaf]
    X[batch_idx, 0, query] = 1.0
    X[:, 0, L + NUM_CLASSES] = 1.0
    b = batch_idx.unsqueeze(1).expand(-1, L)
    leaf = torch.arange(first_leaf, n, device=dev).unsqueeze(0).expand(batch_size, -1)
    X[b, leaf, keys] = 1.0
    X[b, leaf, L + labels] = 1.0
    X[:, first_leaf:, L + NUM_CLASSES + 1] = 1.0
    return X, y


def train_one_radius(r: int) -> tuple[float, int]:
    model = GCN(r).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator(device=DEVICE).manual_seed(r)  # train stream, != TEST_SEED
    start = time.monotonic()
    steps = 0
    while time.monotonic() - start < PER_RADIUS_BUDGET:
        X, y = fast_batch(r, BATCH_SIZE, generator)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    return evaluate_accuracy(model, r), steps


def log_mlflow(description: str, accs: dict, score: float):
    try:
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        )
        mlflow.set_experiment("oversquashing")
        with mlflow.start_run(run_name=description[:60]):
            mlflow.log_params(
                {
                    "description": description,
                    "hidden": HIDDEN,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "device": DEVICE,
                }
            )
            for r, acc in accs.items():
                mlflow.log_metric(f"acc_r{r}", acc)
            mlflow.log_metric("score_mean_acc", score)
    except Exception as e:  # tracking must never kill an experiment
        print(f"mlflow logging skipped: {e}")


if __name__ == "__main__":
    description = os.environ.get("EXPERIMENT_DESC", "unnamed")
    accs = {}
    for r in RADII:
        acc, steps = train_one_radius(r)
        accs[r] = acc
        print(f"acc_r{r}: {acc:.4f}  (steps: {steps})", flush=True)
    score = sum(accs.values()) / len(accs)
    print("---")
    print(f"score_mean_acc: {score:.6f}")
    log_mlflow(description, accs, score)
