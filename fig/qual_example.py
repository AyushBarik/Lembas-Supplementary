"""Qualitative example (full width): the token allocation of Lembas over video time for one question, against Top-K.

Source: Video-MME question 378-1 (video WOGaS19VZ8Q, medium), subtitled setting, Qwen3-VL-8B, budget 12,288 tokens;
the same question as the case study of the main paper.
  qual/alloc_case.json   Lembas: tokens of each of the 128 candidate frames and of each subtitle cue (CPU replay of
                         the allocation, artifacts alloc-profile/replay_case.py)
  qual/records.jsonl     logged per-question records: "naive" = Top-K (frames, tokens per frame, prediction) and the
                         Lembas prediction
  qual/frames/<i>.jpg    candidate frames (640x360)
The text of the highlighted cue is from the Video-MME subtitle file of the video (cue starting at 170.64 s).

Regenerate:  uv run --no-project --with matplotlib --with numpy --with pillow python fig/qual_example.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

from orx_figstyle import PALETTE, WIDE, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "qual")
QUESTION = "What is the color of the pneumatic air gun?"
OPTIONS = "A. Green     B. Red     C. Blue     D. Orange"
THUMBS = [0, 14, 28, 42, 60, 77, 100, 125]   # candidate-frame indices shown as thumbnails, in time order
KEY_FRAME, KEY_CUE_START = 42, 170.64          # the frame that shows the air gun; the cue that names it
KEY_CUE_TEXT = "“Frankie uses a pneumatic air gun to tack the joist”"
BLUE, ORANGE, GRAY, INK = PALETTE["blue"], PALETTE["orange"], "#8C8C8C", "#111111"
RED, GREEN = "#C0392B", "#1E8449"


def main():
    use_style()
    d = json.load(open(os.path.join(Q, "alloc_case.json")))
    recs = {r["method"]: r for r in map(json.loads, open(os.path.join(Q, "records.jsonl")))}
    topk = recs["naive"]
    t = np.array(d["frame_times"]) / 60
    tok = np.array(d["frame_tokens"])
    step = float(np.median(np.diff(t)))
    cues = [c for c in d["cues"] if c["shown"]]

    W, H = WIDE, 3.55
    fig = plt.figure(figsize=(W, H))
    left, right = 0.52 / W, 1 - 0.1 / W
    aw = right - left

    def row(bottom, height):
        return fig.add_axes([left, bottom / H, aw, height / H])

    # ---- question band
    fig.patches.append(Rectangle((0.04 / W, 1 - 0.47 / H), 1 - 0.08 / W, 0.43 / H, transform=fig.transFigure,
                                 fc="#F3F3F3", ec="#4D4D4D", lw=0.6))
    fig.text(0.14 / W, 1 - 0.17 / H, "Question:", fontsize=7.5, fontweight="bold", va="center", color=INK)
    fig.text(0.78 / W, 1 - 0.17 / H, QUESTION, fontsize=7.5, va="center", color=INK)
    fig.text(0.78 / W, 1 - 0.35 / H, OPTIONS, fontsize=7.5, va="center", color=INK)
    for y, name, ans, mark, col in [(0.17, "Top-K:", "D. Orange", "\u2717", RED), (0.35, "Lembas:", "C. Blue", "\u2713", GREEN)]:
        fig.text(1 - 2.3 / W, 1 - y / H, name, fontsize=7.5, fontweight="bold", va="center")
        fig.text(1 - 1.8 / W, 1 - y / H, ans, fontsize=7.5, va="center", color=col)
        fig.text(1 - 1.18 / W, 1 - y / H, mark, fontsize=8, va="center", color=col, fontname="DejaVu Sans")

    # ---- time axes: Lembas frames, Lembas subtitles, Top-K frames
    ax_f = row(1.50, 0.62)
    ax_s = row(0.98, 0.36)
    ax_k = row(0.44, 0.36)
    ax_f.bar(t, tok, width=step * 0.8, color=BLUE, lw=0)
    ax_s.bar([(c["start"] + c["end"]) / 120 for c in cues], [c["tokens"] for c in cues],
             width=[(c["end"] - c["start"]) / 60 * 0.8 for c in cues], color=ORANGE, lw=0)
    ax_k.bar(np.array(topk["frame_times"]) / 60, topk["tokens_per_frame"], width=step * 0.8, color=GRAY, lw=0)
    rows = [(ax_f, [0, 320, 640], "Lembas, frames: 101 frames with 8 to 640 tokens"),
            (ax_s, [0, 8, 16], "Lembas, subtitles: 195 of 247 cues"),
            (ax_k, [0, 320, 640], "Top-K, frames: 64 frames with 170 tokens each")]
    for ax, ticks, label in rows:
        ax.set_xlim(0, 9)
        ax.set_xticks(range(10))
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])
        ax.tick_params(labelsize=6.5)
        ax.grid(False)
        ax.axvspan(KEY_CUE_START / 60 - 0.04, t[KEY_FRAME] + 0.04, color="#000000", alpha=0.08, lw=0, zorder=0)
        ax.annotate(label, (1, 1), xycoords="axes fraction", xytext=(-2, -1), textcoords="offset points",
                    ha="right", va="top", fontsize=6.5, color=INK, bbox=dict(fc="white", ec="none", pad=0.8))
        if ax is not ax_k:
            ax.set_xticklabels([])
    ax_k.set_xlabel("Video time (min)")
    ax_s.set_ylabel("Tokens")
    ax_s.annotate(KEY_CUE_TEXT, (KEY_CUE_START / 60, 16), xytext=(8, -1), textcoords="offset points", ha="left",
                  va="top", fontsize=6.5, color=INK, style="italic", bbox=dict(fc="white", ec="none", pad=0.8))

    # ---- thumbnails with leader lines to their time on the Lembas frame row
    n, gap = len(THUMBS), 0.05
    tw = (aw * W - gap * (n - 1)) / n
    th = tw * 9 / 16
    top_thumbs = H - 0.56
    label_h = 0.14
    for k, i in enumerate(THUMBS):
        x0 = left * W + k * (tw + gap)
        ax = fig.add_axes([x0 / W, (top_thumbs - th) / H, tw / W, th / H])
        ax.imshow(np.asarray(Image.open(os.path.join(Q, "frames", f"{i:03d}.jpg")).convert("RGB").resize((400, 225))))
        ax.set_xticks([]); ax.set_yticks([])
        key = i == KEY_FRAME
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(BLUE if key else "#4D4D4D"); sp.set_linewidth(1.6 if key else 0.5)
        cx = (x0 + tw / 2) / W
        fig.text(cx, (top_thumbs - th - 0.03) / H, f"{tok[i]} tokens", ha="center", va="top",
                 fontsize=6.5, color=BLUE, fontweight="bold" if key else "normal")
        fig.add_artist(ConnectionPatch((cx, (top_thumbs - th - 0.03 - label_h) / H), (t[i], 640), "figure fraction",
                                       "data", axesB=ax_f, color="#9A9A9A", lw=0.4, zorder=0))
    print(save(fig, os.path.join(HERE, "qual_example")))


if __name__ == "__main__":
    main()
