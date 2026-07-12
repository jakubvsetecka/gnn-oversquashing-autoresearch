"""Report figures from results.tsv: accuracy vs radius per kept variant,
and experiment history. Run: uv run analysis.py  ->  figures/*.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RADII = [2, 3, 4, 5, 6, 7]
CHANCE = 1 / 8
# Validated categorical palette (fixed slot order, light surface).
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = {"primary": "#1a1a19", "secondary": "#5f5e56", "grid": "#e5e4dd"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK["secondary"],
        "axes.labelcolor": INK["primary"],
        "text.color": INK["primary"],
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def load_results(path="results.tsv"):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            accs = [float(a) for a in row["acc_per_radius"].split("/")] if row["acc_per_radius"] not in ("", "0") else []
            rows.append(
                {
                    "commit": row["commit"],
                    "score": float(row["score"]),
                    "accs": accs,
                    "status": row["status"],
                    "description": row["description"],
                }
            )
    return rows


# Curated subset for the report: one curve per mechanism class.
# description -> short label; discarded ablations drawn dashed.
REPORT_VARIANTS = {
    "baseline GCN": "GCN",
    "ablation: local GAT on tree edges": "GAT (tree edges)",
    "ablation: baseline + virtual node": "GCN + virtual node",
    "GIN + residual, FA last": "GIN + residual + FA",
    "attention FA last layer": "attention FA layer",
    "pure raw-X attention (no QK)": "raw-X attention",
}


def load_replication(rows, path="replication.tsv"):
    """Per-variant list of per-radius accuracy vectors: seed 0 from results.tsv
    plus fresh seeds from replication.tsv. Returns {description: [accs, ...]}."""
    seeds = {}
    for r in rows:
        if r["description"] in REPORT_VARIANTS:
            seeds[r["description"]] = [r["accs"]]
    if Path(path).exists():
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                accs = [float(a) for a in row["acc_per_radius"].split("/")]
                seeds.setdefault(row["label"], []).append(accs)
    return seeds


def print_stats(rows):
    seeds = load_replication(rows)
    print(f"\n{'variant':38s} n  mean    std     min     max")
    for desc, runs in seeds.items():
        scores = [sum(a) / len(a) for a in runs]
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / (n - 1)) ** 0.5 if n > 1 else 0.0
        print(f"{REPORT_VARIANTS[desc]:38s} {n}  {mean:.4f}  {std:.4f}  {min(scores):.4f}  {max(scores):.4f}")


def fig_report(rows, out):
    sel = [r for r in rows if r["description"] in REPORT_VARIANTS]
    seeds = load_replication(rows)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axhline(CHANCE, color=INK["secondary"], lw=1, ls=":", zorder=1)
    ax.annotate("chance", (RADII[0], CHANCE), textcoords="offset points", xytext=(0, 4),
                fontsize=8, color=INK["secondary"])
    offsets = [(i - (len(sel) - 1) / 2) * 0.07 for i in range(len(sel))]
    for i, r in enumerate(sel):
        c, m = PALETTE[i % 8], MARKERS[i % 8]
        ls = "--" if r["status"] == "discard" else "-"
        runs = seeds.get(r["description"], [r["accs"]])
        n = len(runs)
        mean = [sum(a[j] for a in runs) / n for j in range(len(RADII))]
        x = [rad + offsets[i] for rad in RADII]
        seed_x = [xj for xj in x for _ in runs]
        seed_y = [a[j] for j in range(len(RADII)) for a in runs]
        ax.scatter(seed_x, seed_y, s=7, color=c, alpha=0.45, lw=0, zorder=2)
        ax.plot(x, mean, color=c, marker=m, ms=4, lw=1.6, ls=ls, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8,
                label=REPORT_VARIANTS[r["description"]])
    ax.set_xlabel("tree radius $r$")
    ax.set_ylabel("root accuracy")
    ax.set_xticks(RADII)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=INK["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.52, 0.62))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_accuracy_vs_radius(rows, out):
    kept = [r for r in rows if r["status"] == "keep" and len(r["accs"]) == len(RADII)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axhline(CHANCE, color=INK["secondary"], lw=1, ls=":", zorder=1)
    ax.annotate("chance", (RADII[-1], CHANCE), textcoords="offset points", xytext=(0, 4),
                ha="right", fontsize=8, color=INK["secondary"])
    for i, r in enumerate(kept):
        c, m = PALETTE[i % 8], MARKERS[i % 8]
        ax.plot(RADII, r["accs"], color=c, marker=m, ms=4, lw=2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8, label=r["description"])
    ax.set_xlabel("tree radius $r$")
    ax.set_ylabel("root accuracy")
    ax.set_xticks(RADII)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=INK["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_history(rows, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    xs = range(1, len(rows) + 1)
    best = []
    b = 0.0
    for r in rows:
        if r["status"] == "keep":
            b = max(b, r["score"])
        best.append(b)
    ax.plot(xs, best, color=PALETTE[0], lw=2, label="best kept score", zorder=3)
    style = {"keep": (PALETTE[0], "kept"), "discard": (INK["secondary"], "discarded"), "crash": (PALETTE[5], "crashed")}
    seen = set()
    for x, r in zip(xs, rows):
        c, name = style.get(r["status"], (INK["secondary"], r["status"]))
        ax.scatter([x], [r["score"]], s=18, color=c, zorder=4, edgecolor="white", lw=0.8,
                   label=name if name not in seen else None)
        seen.add(name)
    ax.axhline(CHANCE, color=INK["secondary"], lw=1, ls=":", zorder=1)
    ax.set_xlabel("experiment #")
    ax.set_ylabel("mean accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=INK["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    rows = load_results()
    print(f"{len(rows)} experiments loaded")
    fig_accuracy_vs_radius(rows, "figures/accuracy_vs_radius.png")
    fig_history(rows, "figures/history.png")
    fig_report(rows, "figures/report_accuracy_vs_radius.png")
    print_stats(rows)
