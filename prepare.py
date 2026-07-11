"""Fixed experiment harness for the over-squashing study. DO NOT MODIFY.

Task: Tree-NeighborsMatch style information propagation (Alon & Yahav, ICLR 2021).
A perfect binary tree of radius r. Each leaf carries a unique one-hot key and a
random class label. The root carries a query key. The model must predict, at the
root, the label of the leaf whose key matches the query. The information that
must flow to the root grows exponentially with r -> over-squashing.

This file defines the datasets, the evaluation, and the fixed constants.
Agents modify train.py only.
"""

import os

import torch

# ----------------------------------------------------------------------------
# Fixed constants
RADII = [2, 3, 4, 5, 6, 7]
NUM_CLASSES = 8
TEST_SAMPLES = 2048
TEST_BATCH = 256
TEST_SEED = 1234
# Total wall-clock training budget in seconds, shared equally across radii.
# SMOKE_BUDGET env var overrides it for quick sanity checks only.
TIME_BUDGET_SECONDS = float(os.environ.get("SMOKE_BUDGET", 240.0))
PER_RADIUS_BUDGET = TIME_BUDGET_SECONDS / len(RADII)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tree_num_nodes(r: int) -> int:
    return 2 ** (r + 1) - 1


def tree_adjacency(r: int) -> torch.Tensor:
    """Dense symmetric adjacency (no self loops) of a perfect binary tree.

    Node 0 is the root; children of node i are 2i+1 and 2i+2.
    Leaves are the last 2^r nodes.
    """
    n = tree_num_nodes(r)
    A = torch.zeros(n, n)
    for i in range((n - 1) // 2):
        for c in (2 * i + 1, 2 * i + 2):
            A[i, c] = 1.0
            A[c, i] = 1.0
    return A


def feature_dim(r: int) -> int:
    # [key one-hot (2^r) | label one-hot (NUM_CLASSES) | is_root | is_leaf]
    return 2**r + NUM_CLASSES + 2


def make_batch(r: int, batch_size: int, generator: torch.Generator):
    """Sample a batch of NeighborsMatch instances of radius r.

    Returns X [B, N, F] node features and y [B] root targets.
    """
    n = tree_num_nodes(r)
    L = 2**r  # number of leaves = number of keys
    first_leaf = L - 1
    F = feature_dim(r)

    X = torch.zeros(batch_size, n, F)
    keys = torch.stack(
        [torch.randperm(L, generator=generator) for _ in range(batch_size)]
    )  # [B, L] unique key per leaf
    labels = torch.randint(0, NUM_CLASSES, (batch_size, L), generator=generator)
    target_leaf = torch.randint(0, L, (batch_size,), generator=generator)
    batch_idx = torch.arange(batch_size)
    query = keys[batch_idx, target_leaf]  # [B]
    y = labels[batch_idx, target_leaf]  # [B]

    # Root: query key + is_root flag.
    X[batch_idx, 0, query] = 1.0
    X[:, 0, L + NUM_CLASSES] = 1.0
    # Leaves: own key, own label, is_leaf flag.
    b = batch_idx.unsqueeze(1).expand(-1, L)
    leaf = torch.arange(first_leaf, n).unsqueeze(0).expand(batch_size, -1)
    X[b, leaf, keys] = 1.0
    X[b, leaf, L + labels] = 1.0
    X[:, first_leaf:, L + NUM_CLASSES + 1] = 1.0
    return X, y


@torch.no_grad()
def evaluate_accuracy(model, r: int) -> float:
    """Root-prediction accuracy on a fixed held-out test set. Ground truth metric."""
    generator = torch.Generator().manual_seed(TEST_SEED + r)
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for _ in range(TEST_SAMPLES // TEST_BATCH):
        X, y = make_batch(r, TEST_BATCH, generator)
        logits = model(X.to(DEVICE))
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += y.numel()
    if was_training:
        model.train()
    return correct / total
