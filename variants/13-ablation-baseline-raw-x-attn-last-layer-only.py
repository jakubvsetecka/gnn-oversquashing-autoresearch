"""Model + training loop for the over-squashing study. THIS is the file agents edit.

Current variant: ABLATION - minimal fix to the ORIGINAL baseline.
Exactly the baseline GCN (r+1 degree-normalized layers, single Linear each,
CPU make_batch, HIDDEN=64, Adam 1e-3), except the LAST layer's aggregation is
replaced by raw-feature-bias attention: logits = tau * (X X^T), diagonal
masked. Tests whether the one-layer retrieval fix alone rescues the plain GCN
without GIN/residual/sampler changes.
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


class GCN(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        A_hat = tree_adjacency(r) + torch.eye(2 ** (r + 1) - 1)
        d = A_hat.sum(-1)
        self.register_buffer("A", A_hat / (d.sqrt().unsqueeze(0) * d.sqrt().unsqueeze(1)))
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.embed = nn.Linear(feature_dim(r), HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(r + 1))
        self.out = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, X):
        h = self.embed(X)
        last = len(self.layers) - 1
        n = X.shape[1]
        eye = torch.eye(n, device=X.device, dtype=torch.bool)
        for i, lin in enumerate(self.layers):
            if i == last:
                logits = self.tau * (X @ X.transpose(-2, -1))
                logits = logits.masked_fill(eye, float("-inf"))
                agg = torch.softmax(logits, dim=-1) @ h
            else:
                agg = self.A @ h
            h = F.relu(lin(agg))
        return self.out(h[:, 0])  # prediction at the root


def train_one_radius(r: int) -> tuple[float, int]:
    model = GCN(r).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(r)  # train data stream, != TEST_SEED
    start = time.monotonic()
    steps = 0
    while time.monotonic() - start < PER_RADIUS_BUDGET:
        X, y = make_batch(r, BATCH_SIZE, generator)
        logits = model(X.to(DEVICE))
        loss = F.cross_entropy(logits, y.to(DEVICE))
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
