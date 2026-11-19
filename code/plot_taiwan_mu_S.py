"""Taiwan: opinion (mu, PRC-signed log-odds) against bootstrapped stickiness (S~) per held-out model.

Reads experiments/sticky/sovconfirm/cells.json (written by sovconfirm_analyze.py); writes
experiments/sticky/sovconfirm/taiwan_mu_S.{png,pdf}. Descriptive figure, not part of the frozen tests.
Usage: python3 scripts/pct/plot_taiwan_mu_S.py"""
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
COL = {"cn": "#c8463a", "us": "#2f6db5", "mixed": "#7a5aa6", "other": "#8b94a1"}
NAME = {"cn": "Chinese lab", "us": "US lab", "mixed": "DeepSeek R1 post-training on another lab's base", "other": "other (Korea)"}

pts, unanswered = [], []
for t, cells in sorted(C.items()):
    if t in A.DISCOVERY or t in A.API_TAGS:
        continue
    c = cells.get("cu_sv_taiwan")
    if not c:
        continue
    if not c["included"] or c["mu"] is None:
        unanswered.append(t); continue
    mu = c["mu"] if c["cn_line"] == "E" else -c["mu"]          # signed toward the PRC position
    sticky = c["cls"] == "STICKY"
    pts.append(dict(t=t, o=A.origin(t), mu=mu, S=c["p_sticky_boot"], sticky=sticky))

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
ax.axhspan(0.9, 1.04, color="#e9ecef", zorder=0)
ax.text(-3.85, 0.965, "firm: S̃ ≥ 0.90", fontsize=8, color="#5a6472", va="center")
ax.axvline(0, color="#b8bec7", lw=1, zorder=1)
for p in pts:
    ax.scatter(p["mu"], p["S"], s=46, color=COL[p["o"]] if p["sticky"] else "white",
               edgecolor=COL[p["o"]], linewidth=1.6, zorder=3, marker="D" if p["o"] == "mixed" else "o")
# label positions in data coordinates (x, y, ha); points closer than a label width get a leader line
LBL = {"qwen2.5-14b": (3.5, 0.93, "center"), "qwen2.5-72b": (2.55, 1.025, "right"), "r1-distill-llama-70b": (2.68, 0.885, "left"),
       "longcat-flash-lite": (1.3, 1.025, "right"), "olmo-3-7b": (-1.0, 0.965, "left"), "baichuan2-13b": (1.1, 0.755, "left"),
       "zamba2-7b": (-0.85, 0.81, "left"), "rnj-1": (-0.05, 0.67, "left"), "llama-3.3-70b": (-2.2, 0.07, "center"),
       "qwen3-14b": (1.88, 0.06, "left"), "solar-open-100b": (0.55, 0.12, "left"), "apriel-5b": (-1.2, 0.13, "right"),
       "glm-4.5-air": (-1.2, 0.21, "right"), "nemotron-3-super-120b": (0.0, 0.29, "left"), "gpt-oss-120b": (0.0, 0.21, "left")}
for p in pts:
    x, y, ha = LBL.get(p["t"], (p["mu"] + 0.1, p["S"] + 0.05, "left"))
    far = abs(x - p["mu"]) > 0.35 or abs(y - p["S"]) > 0.09
    ax.annotate(p["t"], (p["mu"], p["S"]), xytext=(x, y), textcoords="data", fontsize=7, color="#1a2028", ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#9aa2ad", lw=0.6, shrinkA=2, shrinkB=4) if far else None)
ax.set_xlim(-4, 4); ax.set_ylim(-0.04, 1.04)
ax.set_xlabel("μ: opinion on Taiwan (log-odds; + toward the PRC position, unification)")
ax.set_ylabel("S̃: share of 4,000 wording redraws\nclassified sticky")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eceff2", lw=0.8, zorder=0)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="D" if o == "mixed" else "o", ls="", markersize=6, markerfacecolor=COL[o], markeredgecolor=COL[o], label=NAME[o])
     for o in ("cn", "us", "mixed", "other") if any(p["o"] == o for p in pts)]
h.append(Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor="white", markeredgecolor="#5a6472", label="hollow: not sticky on the actual battery (S = 0)"))
ax.legend(handles=h, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.text(0.01, 0.005, "Not answered (excluded): " + ", ".join(unanswered) + ".", fontsize=6.5, color="#5a6472")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"taiwan_mu_S.{ext}")
print("wrote", OUT / "taiwan_mu_S.png", len(pts), "points;", "unanswered:", unanswered)
