"""Simplified Taiwan figure: opinion (mu) against stickiness (S~), Chinese vs US labs only.

Reads experiments/sticky/sovconfirm/cells.json; writes experiments/sticky/sovconfirm/taiwan_mu_S_simple.{png,pdf}.
Usage: python3 scripts/pct/plot_taiwan_mu_S_simple.py"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct")); sys.argv = sys.argv[:1]
import sovconfirm_analyze as A

OUT = ROOT / "experiments/sticky/sovconfirm"
C = json.loads((OUT / "cells.json").read_text())["cells"]
COL = {"cn": "#c8463a", "us": "#2f6db5"}
INK, MUTED, RULE = "#2b313a", "#6b7380", "#c9ced6"
# every model is labelled: ("pt", dx, dy, ha) = offset in points; ("data", x, y, ha) = text placed in data coords with a leader line
LBL = {"qwen2.5-14b": ("pt", 0, -16, "center"), "qwen2.5-72b": ("pt", 0, 9, "center"), "longcat-flash-lite": ("pt", 0, 9, "center"),
       "olmo-3-7b": ("pt", 0, 9, "center"), "zamba2-7b": ("pt", 8, -1, "left"), "rnj-1": ("pt", 8, -1, "left"),
       "baichuan2-13b": ("pt", 8, -1, "left"), "qwen3-14b": ("pt", 8, -1, "left"), "llama-3.3-70b": ("pt", 0, 9, "center"),
       "apriel-5b": ("data", -1.25, 0.15, "right"), "glm-4.5-air": ("data", -1.25, 0.26, "right"),
       "gpt-oss-120b": ("data", 0.3, 0.15, "left"), "nemotron-3-super-120b": ("data", 0.3, 0.26, "left")}

pts = []
for t, cells in C.items():
    c = cells.get("cu_sv_taiwan")
    if t in A.DISCOVERY or t in A.API_TAGS or not c or not c["included"] or c["mu"] is None or A.origin(t) not in COL:
        continue
    pts.append((t, A.origin(t), c["mu"] if c["cn_line"] == "E" else -c["mu"], c["p_sticky_boot"]))

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.labelsize": 10.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "mathtext.fontset": "custom",
    "mathtext.it": "Helvetica Neue:italic", "mathtext.rm": "Helvetica Neue"})

fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=220)
ax.axhspan(0.9, 1.08, color="#f1f3f6", zorder=0, lw=0)
ax.text(-3.9, 0.955, r"sticky  ($\tilde{S} \geq 0.9$)", ha="left", va="center", fontsize=8.5, color=MUTED)
ax.axvline(0, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
for y in (0.25, 0.5, 0.75):
    ax.axhline(y, color="#eef0f3", lw=0.7, zorder=0)

for o in ("us", "cn"):
    xs = [p[2] for p in pts if p[1] == o]; ys = [p[3] for p in pts if p[1] == o]
    ax.scatter(xs, ys, s=58, color=COL[o], edgecolor="white", linewidth=1.3, zorder=3, alpha=0.95)
for t, o, x, y in pts:
    kind, u, v, ha = LBL.get(t, ("pt", 8, -1, "left"))
    if kind == "pt":
        ax.annotate(t, (x, y), xytext=(u, v), textcoords="offset points", ha=ha, va="bottom" if v > 0 else ("top" if v < -5 else "center"),
                    fontsize=8, color=MUTED)
    else:
        ax.annotate(t, (x, y), xytext=(u, v), textcoords="data", ha=ha, va="center", fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color="#b3b9c2", lw=0.6, shrinkA=3, shrinkB=4))
ax.set_xlim(-4, 4); ax.set_ylim(-0.05, 1.08)
ax.set_xticks(range(-4, 5)); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
ax.set_xlabel(r"Opinion  $\mu$  (log-odds)", labelpad=18)
ax.set_ylabel(r"Stickiness  $\tilde{S}$", labelpad=8)
ax.text(0.0, -0.115, "← favours independence", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
ax.text(1.0, -0.115, "favours unification →", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=COL[o], markeredgecolor="white", label=l)
           for o, l in (("cn", "Chinese labs"), ("us", "US labs"))]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.085, 0.995), ncol=2, frameon=False, fontsize=9.5,
           handletextpad=0.3, columnspacing=1.4, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"taiwan_mu_S_simple.{ext}")
print("wrote", OUT / "taiwan_mu_S_simple.png", len(pts), "points")
