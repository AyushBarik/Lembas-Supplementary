"""Supplementary sensitivity figure, three panels in one full-width figure: (a) attention weight omega
(MT-Video-Bench), (b) subtitle weight beta (Video-MME w/ subtitles), (c) relevance temperature T (Video-MME w/
subtitles, 300 questions; the T = 0.63 point is the default run on the same questions, sens_default_T.json).
Same data, axes and colors as the earlier one-per-column figures (sens_split.py): sens_weights.json,
sens_sweeps.json, sens_default_T.json.
Regenerate: uv run --no-project --with matplotlib --with numpy python fig/sens_panels.py"""
import json
from pathlib import Path

import orx_figstyle as fs

here = Path(__file__).parent
w = json.load(open(here / "sens_weights.json"))
s = json.load(open(here / "sens_sweeps.json"))
dT = json.load(open(here / "sens_default_T.json"))
fs.use_style()

PANELS = [  # symbol, color, points (value, tflops, acc), label offsets, x label, x ticks, y ticks
    (r"\omega", "blue", [(r["value"], r["tflops"], r["acc"]) for r in w["omega"]],
     {0.0: (0, 6, "center"), 0.25: (6, 0, "left"), 0.5: (0, 6, "center"), 0.75: (0, -6, "center"),
      1.0: (0, -6, "center")},
     "TFLOPs per conversation", range(240, 321, 20), range(56, 61)),
    (r"\beta", "orange", [(r["value"], r["tflops"], r["acc"]) for r in w["beta"]],
     {0.0: (0, 6, "center"), 0.25: (0, -6, "center"), 0.5: (6, 0, "left"), 1.0: (0, 6, "center"), 2.0: (-6, 0, "right")},
     "TFLOPs per question", range(230, 235), range(65, 86, 5)),
    ("T", "green", [(0.3, s["T0.3"]["tflops"], s["T0.3"]["acc"]), (0.63, dT["tflops"], dT["acc"]),
                    (1.0, s["T1"]["tflops"], s["T1"]["acc"]), (2.0, s["T2"]["tflops"], s["T2"]["acc"])],
     {0.3: (0, 6, "center"), 0.63: (0, 6, "center"), 1.0: (6, 0, "left"), 2.0: (0, -6, "center")},
     "TFLOPs per question", range(226, 237, 2), range(74, 81, 2)),
]

def better_arrow(ax):
    """The "Better" arrow of Fig. 1 of the main paper (fig/budget_curve.py of the paper repository), unchanged:
    a slim outlined arrow at 45 degrees towards the top-left corner (less compute, higher accuracy)."""
    TIP, L, HEAD = (0.03, 0.975), 24, 8.5  # L = tip-to-tail offset per axis; HEAD = head length (points)
    ax.annotate("", xy=TIP, xycoords="axes fraction", xytext=(L, -L), textcoords="offset points",
                arrowprops=dict(arrowstyle=f"simple,tail_width=9,head_width=15,head_length={HEAD}", fc="white",
                                ec="black", lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=1, joinstyle="miter"))
    mid = (HEAD / 2 ** 0.5 + L) / 2  # middle of the shaft, measured from the tip
    ax.annotate("Better", xy=TIP, xycoords="axes fraction", xytext=(mid, -mid), textcoords="offset points",
                ha="center", va="center_baseline", rotation=-45, rotation_mode="anchor", fontsize=6.5)


fig, axes = fs.figure_grid(1, 3, width=fs.WIDE, ratio=0.25)
for ax, (sym, col, pts, off, xl, xt, yt) in zip(axes, PANELS):
    c = fs.PALETTE[col]
    for v, tf, a in pts:
        ax.plot([tf], [a], marker="o", ms=4.5, color=c, ls="none", clip_on=False, zorder=4)
        dx, dy, ha = off[v]
        ax.annotate(rf"${sym}={v:g}$", (tf, a), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="bottom" if dy > 0 else ("top" if dy < 0 else "center"), fontsize=7, zorder=5)
    xt, yt = list(xt), list(yt)
    ax.set_xlabel(xl)
    ax.set_xticks(xt); ax.set_xlim(xt[0], xt[-1]); ax.set_yticks(yt); ax.set_ylim(yt[0], yt[-1])
    ax.set_ylabel("Accuracy (%)")
    better_arrow(ax)
fs.panel_labels(axes)
print(fs.save(fig, str(here / "sens_panels")))
