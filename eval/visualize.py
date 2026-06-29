"""Generate publication-quality evaluation charts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
METRICS_PATH = OUTPUTS / "metrics.json"

with open(METRICS_PATH) as f:
    data = json.load(f)

lexical = data["lexical"]
hybrid = data["hybrid"]

ks = [5, 10, 20, 50]

# --- Load per-query metrics for distribution plots ---
def _load_raw(filepath):
    recs, lats, mrrs = [], [], []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            recs.append(r)
            lats.append(r.get("latency_ms", 0))
            mrrs.append(r.get("mrr", 0))
    return recs, np.array(lats), np.array(mrrs)

lex_recs, lex_lats, lex_mrr = _load_raw(OUTPUTS / "results_lexical.jsonl")
hyb_recs, hyb_lats, hyb_mrr = _load_raw(OUTPUTS / "results_hybrid.jsonl")

# Palette
LEX_C = "#78909c"
HYB_C = "#1e88e5"
LEX_L = "#cfd8dc"
HYB_L = "#bbdefb"
GRID_C = "#e0e0e0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.color": GRID_C,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(name):
    path = OUTPUTS / name
    plt.savefig(path)
    print(f"  {path}")


# ============================================================
# 1. Cumulative Recall Curve (IR classic)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

lex_rec_vals = [lexical[f"mean_recall@{k}"] for k in ks]
hyb_rec_vals = [hybrid[f"mean_recall@{k}"] for k in ks]

ax.plot(ks, lex_rec_vals, "o-", color=LEX_C, linewidth=2, markersize=7, label="Lexical (BM25)")
ax.plot(ks, hyb_rec_vals, "s-", color=HYB_C, linewidth=2, markersize=7, label="KnowWhere")
ax.fill_between(ks, lex_rec_vals, hyb_rec_vals, alpha=0.08, color=HYB_C)

for k, v in zip(ks, hyb_rec_vals):
    ax.annotate(f"{v:.1%}", (k, v), textcoords="offset points", xytext=(8, 6), fontsize=8, color=HYB_C)

ax.set_xlabel("Rank Cutoff (k)")
ax.set_ylabel("Recall")
ax.set_title("Cumulative Recall Curve")
ax.set_ylim(0, 0.85)
ax.set_xticks(ks)
ax.legend(loc="lower right")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
_save("chart_cumulative_recall.png")
plt.close(fig)


# ============================================================
# 2. Precision-Recall Curve (at cutoffs)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

lex_prec_vals = [lexical[f"mean_precision@{k}"] for k in ks]
hyb_prec_vals = [hybrid[f"mean_precision@{k}"] for k in ks]

ax.plot(lex_rec_vals, lex_prec_vals, "o-", color=LEX_C, linewidth=2, markersize=7, label="Lexical (BM25)")
ax.plot(hyb_rec_vals, hyb_prec_vals, "s-", color=HYB_C, linewidth=2, markersize=7, label="KnowWhere")

for k, px, py in zip(ks, hyb_prec_vals, hyb_rec_vals):
    ax.annotate(f"@{k}", (py, px), textcoords="offset points", xytext=(6, 4), fontsize=7, color=HYB_C)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall by Cutoff Depth")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
_save("chart_pr_curve.png")
plt.close(fig)


# ============================================================
# 3. MRR Distribution (violin + swarm)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))

violin_parts = ax.violinplot(
    [lex_mrr, hyb_mrr], positions=[1, 2], showmeans=True,
    showmedians=True, widths=0.6,
)
for body, color in zip(violin_parts["bodies"], [LEX_C, HYB_C]):
    body.set_facecolor(color)
    body.set_alpha(0.7)
for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
    if part in violin_parts:
        violin_parts[part].set_color("#333333")
        violin_parts[part].set_linewidth(1)

# Overlay jittered points (sample 200 for readability)
rng = np.random.default_rng(42)
for i, (arr, color) in enumerate([(lex_mrr, LEX_C), (hyb_mrr, HYB_C)]):
    sample = rng.choice(arr, min(200, len(arr)), replace=False)
    jitter = rng.normal(0, 0.04, len(sample))
    ax.scatter(np.full_like(sample, i + 1) + jitter, sample, alpha=0.25, s=8, color=color, edgecolors="none")

ax.set_xticks([1, 2])
ax.set_xticklabels(["Lexical", "KnowWhere"])
ax.set_ylabel("MRR")
ax.set_title("MRR Distribution (per query)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
_save("chart_mrr_dist.png")
plt.close(fig)


# ============================================================
# 4. Latency Distribution (log-scale box + strip)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))

bp = ax.boxplot(
    [lex_lats, hyb_lats], positions=[1, 2], widths=0.5,
    patch_artist=True, showfliers=True, flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
)
for box, color in zip(bp["boxes"], [LEX_C, HYB_C]):
    box.set_facecolor(color)
    box.set_alpha(0.7)

# Overlay strip plot (sample)
for i, (arr, color) in enumerate([(lex_lats, LEX_C), (hyb_lats, HYB_C)]):
    sample = rng.choice(arr, min(200, len(arr)), replace=False)
    jitter = rng.normal(0, 0.06, len(sample))
    ax.scatter(np.full_like(sample, i + 1) + jitter, sample, alpha=0.2, s=6, color=color, edgecolors="none")

ax.set_xticks([1, 2])
ax.set_xticklabels(["Lexical", "KnowWhere"])
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency Distribution")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
# Add median annotations
for i, (name, arr) in enumerate([("Lexical", lex_lats), ("KnowWhere", hyb_lats)]):
    med = np.median(arr)
    ax.annotate(f"P50={med:.0f} ms", (i + 1.35, med), fontsize=8, color="#333", va="center")
_save("chart_latency_dist.png")
plt.close(fig)


# ============================================================
# 5. Improvement Factor (log-scale horizontal bar)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

metrics_labels = [
    "MRR", "Precision@5", "Precision@10", "Precision@20", "Precision@50",
    "Recall@5", "Recall@10", "Recall@20", "Recall@50",
]
lex_vals = (
    [lexical["mean_mrr"]]
    + [lexical[f"mean_precision@{k}"] for k in ks]
    + [lexical[f"mean_recall@{k}"] for k in ks]
)
hyb_vals = (
    [hybrid["mean_mrr"]]
    + [hybrid[f"mean_precision@{k}"] for k in ks]
    + [hybrid[f"mean_recall@{k}"] for k in ks]
)
improvements = [h / l if l > 1e-9 else float("inf") for h, l in zip(hyb_vals, lex_vals)]
# Cap for display and sort
improvements_disp = [min(v, 1000) for v in improvements]
y_pos = range(len(metrics_labels))

colors = [HYB_C if v >= 100 else "#ff9800" for v in improvements_disp]
ax.barh(y_pos, improvements_disp, color=colors, alpha=0.85, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_labels)
ax.set_xlabel("Improvement Factor (KnowWhere / Lexical)")
ax.set_title("Relative Improvement over Lexical Baseline")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.invert_yaxis()
# Annotate bars
for y, val, raw in zip(y_pos, improvements_disp, improvements):
    label = f"  ×{raw:.0f}" if raw < 1e4 else f"  ×{raw:.1e}"
    ax.text(val, y, label, va="center", fontsize=8, color="#333")
_save("chart_improvement.png")
plt.close(fig)


# ============================================================
# 6. Small multiples: Precision + Recall by cutoff (facetted)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

for ax, title, lex_arr, hyb_arr in [
    (ax1, "Precision@k", lex_prec_vals, hyb_prec_vals),
    (ax2, "Recall@k", lex_rec_vals, hyb_rec_vals),
]:
    ax.plot(ks, lex_arr, "o--", color=LEX_C, linewidth=1.8, markersize=7, label="Lexical")
    ax.plot(ks, hyb_arr, "s-", color=HYB_C, linewidth=2.2, markersize=8, label="KnowWhere")
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # Annotate values on hybrid line
    for k, v in zip(ks, hyb_arr):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, ha="center", color=HYB_C)

fig.suptitle("Precision & Recall by Rank Cutoff", fontsize=13, y=1.01)
_save("chart_precision_recall_lines.png")
plt.close(fig)


# ============================================================
# 7. Latency CDF (cumulative distribution)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

for arr, color, label in [(lex_lats, LEX_C, "Lexical"), (hyb_lats, HYB_C, "KnowWhere")]:
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax.plot(sorted_arr, cdf, color=color, linewidth=2, label=label)
    # P50, P95 markers
    for pct, ls in [(50, "--"), (95, ":")]:
        val = np.percentile(arr, pct)
        ax.axvline(val, color=color, linestyle=ls, alpha=0.5, linewidth=1)
        ax.annotate(f"P{pct}={val:.0f}ms", (val, 0.02 if pct == 50 else 0.90),
                    color=color, fontsize=7, rotation=90, va="bottom")

ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Latency Cumulative Distribution Function")
ax.legend(loc="lower right")
ax.set_xlim(left=0)
_save("chart_latency_cdf.png")
plt.close(fig)


print("\nCharts saved:")
for p in sorted(OUTPUTS.glob("chart_*.png")):
    print(f"  {p}")
