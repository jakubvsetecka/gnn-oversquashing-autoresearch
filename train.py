"""Model + training loop for the over-squashing study. THIS is the file agents edit.

Current variant: GCN + fully-adjacent (FA) last layer (Alon & Yahav ICLR 2021).
The final message-passing layer operates on a complete graph (mean-normalized
all-ones adjacency) instead of the tree, giving the root direct access to all
leaves in one hop and bypassing the exponential bottleneck.
Otherwise baseline: r+1 layers, Adam, on-the-fly data, one model per radius.
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
        A_hat = tree_adjacency(r) + torch.eye(tree_num := 2 ** (r + 1) - 1)
        d = A_hat.sum(-1)
        self.register_buffer("A", A_hat / (d.sqrt().unsqueeze(0) * d.sqrt().unsqueeze(1)))
        n = A_hat.shape[0]
        self.register_buffer("A_full", torch.ones(n, n) / n)  # FA layer adjacency
        self.embed = nn.Linear(feature_dim(r), HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(r + 1))
        self.out = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, X):
        h = self.embed(X)
        for i, lin in enumerate(self.layers):
            A = self.A_full if i == len(self.layers) - 1 else self.A
            h = F.relu(lin(A @ h))
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
