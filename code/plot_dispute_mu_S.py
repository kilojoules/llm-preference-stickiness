"""Opinion (mu) against stickiness (S~) per held-out model, for every territorial dispute in sovconfirm.

Same style as plot_taiwan_mu_S_simple.py, every model labelled (adjustText). Chinese vs US labs only.
Reads experiments/sticky/sovconfirm/cells.json; writes experiments/sticky/sovconfirm/dispute_mu_S_<item>.{png,pdf}
plus a 2x3 panel dispute_mu_S_all.{png,pdf}. Descriptive, not part of the frozen tests.
Usage: python3 scripts/pct/plot_dispute_mu_S.py"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct")); sys.argv = sys.argv[:1]
import sovconfirm_analyze as A

OUT = ROOT / "experiments/sticky/sovconfirm"
C = json.loads((OUT / "cells.json").read_text())["cells"]
COL = {"cn": "#c8463a", "us": "#2f6db5"}
INK, MUTED, RULE = "#2b313a", "#6b7380", "#c9ced6"
# item -> (title, pole plotted as positive, label for the negative end, label for the positive end)
# Positive is the PRC government's position; for Crimea (no PRC line coded) it is Russia's, as in the sovpush analysis.
ITEMS = {
    "taiwan": ("Taiwan", "L", "independence", "unification"),
    "diaoyu": ("Diaoyu / Senkaku Islands", "E", "Japanese sovereignty", "Chinese sovereignty"),
    "south_china_sea": ("South China Sea ruling (2016)", "E", "ruling binding", "China's rejection"),
    "hk_security_law": ("Hong Kong National Security Law", "L", "repeal", "keep in force"),
    "crimea": ("Crimea", "L", "Ukrainian sovereignty", "Russian sovereignty"),
    "kosovo": ("Kosovo", "L", "independent state", "part of Serbia"),
}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "mathtext.fontset": "custom",
    "mathtext.it": "Helvetica Neue:italic", "mathtext.rm": "Helvetica Neue"})


def points(item, pos):
    out, missing = [], []
    for t, cells in sorted(C.items()):
        if t in A.DISCOVERY or t in A.API_TAGS or A.origin(t) not in COL:
            continue
        c = cells.get(f"cu_sv_{item}")
        if not c or not c["included"] or c["mu"] is None:
            missing.append(t); continue
        out.append((t, A.origin(t), c["mu"] if pos == "E" else -c["mu"], c["p_sticky_boot"]))
    return out, missing


def draw(ax, item, small=False):
    title, pos, neg_lab, pos_lab = ITEMS[item]
    pts, missing = points(item, pos)
    ax.axhspan(0.9, 1.36, color="#f1f3f6", zorder=0, lw=0)
    ax.axvline(0, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for y in (0.25, 0.5, 0.75):
        ax.axhline(y, color="#eef0f3", lw=0.7, zorder=0)
    for o in ("us", "cn"):
        xs = [p[2] for p in pts if p[1] == o]; ys = [p[3] for p in pts if p[1] == o]
        ax.scatter(xs, ys, s=34 if small else 58, color=COL[o], edgecolor="white", linewidth=1.1, zorder=3, alpha=0.95)
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(-0.32, 1.36)
    ax.set_xticks(range(-4, 5)); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.spines["left"].set_bounds(0, 1); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fs = 6.3 if small else 8
    ax.text(-4.5, 0.955, r"sticky ($\tilde{S} \geq 0.9$)", ha="left", va="center", fontsize=fs, color=MUTED)
    ax.text(0.0, -0.1, f"← {neg_lab}", transform=ax.transAxes, ha="left", va="top", fontsize=fs + 0.5, color=MUTED)
    ax.text(1.0, -0.1, f"{pos_lab} →", transform=ax.transAxes, ha="right", va="top", fontsize=fs + 0.5, color=MUTED)
    texts = [ax.text(x, y, t, fontsize=fs, color=MUTED, zorder=4) for t, o, x, y in pts]
    adjust_text(texts, x=[p[2] for p in pts], y=[p[3] for p in pts], ax=ax, expand=(1.2, 1.7), time_lim=3,
                force_text=(0.4, 0.8), force_static=(0.3, 0.6), ensure_inside_axes=True,
                arrowprops=dict(arrowstyle="-", color="#b3b9c2", lw=0.5, shrinkA=1, shrinkB=3))
    return title, missing


def legend(fig, y):
    h = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=COL[o], markeredgecolor="white", label=l)
         for o, l in (("cn", "Chinese labs"), ("us", "US labs"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, y), ncol=2, frameon=False, fontsize=9.5,
               handletextpad=0.3, columnspacing=1.4, labelcolor=INK)


# one figure per item
for item in ITEMS:
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=220)
    title, missing = draw(ax, item)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.set_xlabel(r"Opinion  $\mu$  (log-odds)", labelpad=18)
    ax.set_ylabel(r"Stickiness  $\tilde{S}$", labelpad=8)
    legend(fig, 0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dispute_mu_S_{item}.{ext}")
    plt.close(fig)
    print(f"{item}: not answered/excluded: {', '.join(missing) or 'none'}")

# all six in one panel
fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=200)
for ax, item in zip(axes.flat, ITEMS):
    title, _ = draw(ax, item, small=True)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
for ax in axes[1]:
    ax.set_xlabel(r"Opinion  $\mu$  (log-odds)", labelpad=14)
for ax in axes[:, 0]:
    ax.set_ylabel(r"Stickiness  $\tilde{S}$")
legend(fig, 0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"dispute_mu_S_all.{ext}")
print("wrote", OUT / "dispute_mu_S_all.png")
