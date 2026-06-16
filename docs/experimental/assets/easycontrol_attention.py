"""Render an EasyControl attention-flow figure styled after the unified
multimodal-transformer infographic (clean AR/cond subsequence + noisy
DM/target subsequence sharing one attention with a block-structured mask).

Run:  uv run python docs/experimental/assets/easycontrol_attention.py
Out:  docs/experimental/assets/easycontrol_attention.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.pyplot as plt

# ---- palette (matched to the reference infographic) --------------------------
COND_FILL = "#cfe8ee"  # clean reference image  (≈ Vision-AR teal)
COND_EDGE = "#4a90a4"
TGT_FILL = "#fbe2c4"  # noisy target latent     (≈ Vision-DM orange)
TGT_EDGE = "#d98a3d"
TEXT_FILL = "#d9d2ec"  # text / cross-attn        (≈ Language purple)
TEXT_EDGE = "#7a6ba8"
BOX_FILL = "#f2f2f2"
BOX_EDGE = "#888888"
MASK_FULL = "#fbe2c4"  # full-attend tiles
MASK_COND = "#cfe8ee"  # cond self-attn tile
MASK_ZERO = "#ececec"  # masked / zero
INK = "#333333"

plt.rcParams["font.family"] = "DejaVu Sans"


def rbox(ax, x, y, w, h, fc, ec, text="", fs=9, weight="normal", tc=INK, lw=1.3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
        lw=lw, color=color, linestyle=ls, zorder=1,
    ))


fig = plt.figure(figsize=(14.5, 7.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.16)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# =============================== LEFT: two-stream block ========================
axL.text(5, 9.78, "EasyControl — two-stream shared attention (frozen DiT)",
         ha="center", fontsize=12.5, weight="bold", color=INK)

# token rows ----------------------------------------------------------------
def token_row(ax, x0, y, labels, fc, ec):
    w = 0.62; g = 0.07
    for i, lb in enumerate(labels):
        rbox(ax, x0 + i * (w + g), y, w, 0.5, fc, ec, lb, fs=8)
    return x0, x0 + len(labels) * (w + g) - g

# cond tokens (clean ref image, t=0)
cx0, cx1 = token_row(axL, 0.35, 8.55,
                     ["$c_1$", "$c_2$", "$\\cdots$", "$c_m$"], COND_FILL, COND_EDGE)
# target tokens (noisy latent, t) + text tokens
tx0, _ = token_row(axL, 4.7, 8.55,
                   ["$\\tilde z_1$", "$\\tilde z_2$", "$\\cdots$", "$\\tilde z_n$"], TGT_FILL, TGT_EDGE)
ttx0, tx1 = token_row(axL, 7.55, 8.55,
                      ["$l_1$", "$\\cdots$"], TEXT_FILL, TEXT_EDGE)

axL.text((cx0 + cx1) / 2, 9.42, "Cond stream\n(reference image, VAE→$x_{emb}$,  t = 0)",
         ha="center", va="center", fontsize=8.3, color=COND_EDGE, weight="bold",
         linespacing=1.35)
axL.text(6.55, 9.42, "Target stream\n(noisy latent, t)        + text",
         ha="center", va="center", fontsize=8.3, color=TGT_EDGE, weight="bold",
         linespacing=1.35)

# brace-ish underlines
axL.plot([cx0, cx1], [8.43, 8.43], color=COND_EDGE, lw=1.2)
axL.plot([tx0, tx1], [8.43, 8.43], color=TGT_EDGE, lw=1.2)

# column x-centers for the two stacks
LC = (cx0 + cx1) / 2          # cond column center
TC = 6.0                       # target column center

# --- stack helper ---
def stack(ax, xc, items, y_top, fc, ec):
    """items: list of (label, fontsize, height). returns list of (y_lo,y_hi)."""
    w = 3.2; y = y_top; spans = []
    for lb, fs, h in items:
        rbox(ax, xc - w / 2, y - h, w, h, fc, ec, lb, fs=fs)
        spans.append((y - h, y))
        y -= h + 0.30
    return spans

# target stack
arrow(axL, TC, 8.43, TC, 8.18)
t_spans = stack(axL, TC, [
    ("LayerNorm  ·  AdaLN$_{self}$(t)", 8.5, 0.46),
], 8.18, BOX_FILL, BOX_EDGE)

# cond stack (mirrors target, but AdaLN at t=0 + cond-LoRA)
arrow(axL, LC, 8.43, LC, 8.18)
c_spans = stack(axL, LC, [
    ("LayerNorm  ·  AdaLN$_{self}$(0)", 8.5, 0.46),
], 8.18, BOX_FILL, BOX_EDGE)

# shared attention bar spanning both columns
attn_y = 6.55; attn_h = 0.95
attn_x0 = LC - 1.7; attn_x1 = TC + 1.7
arrow(axL, LC, t_spans[-1][0] + 0.0 if False else c_spans[-1][0], LC, attn_y + attn_h)
arrow(axL, LC, c_spans[-1][0], LC, attn_y + attn_h)
arrow(axL, TC, t_spans[-1][0], TC, attn_y + attn_h)
rbox(axL, attn_x0, attn_y, attn_x1 - attn_x0, attn_h, "#e9e9e9", "#777777",
     "", lw=1.5)
axL.text((attn_x0 + attn_x1) / 2, attn_y + attn_h - 0.24, "Shared Self-Attention",
         ha="center", fontsize=10.5, weight="bold", color=INK)
axL.text(LC, attn_y + 0.30, "cond self-attn\n$\\mathrm{Attn}(Q_c, K_c, V_c)$",
         ha="center", fontsize=7.6, color=COND_EDGE)
axL.text(TC, attn_y + 0.30,
         "extended attn\n$\\mathrm{Attn}(Q_t,\\,[K_t;K_c],\\,[V_t;V_c])+b_{cond}$",
         ha="center", fontsize=7.6, color=TGT_EDGE)
# the read arrow cond -> target inside attention
arrow(axL, LC + 1.05, attn_y + 0.55, TC - 1.95, attn_y + 0.55,
      color=COND_EDGE, lw=1.4, style="-|>")
axL.text((LC + TC) / 2, attn_y + 0.78, "K$_c$,V$_c$", ha="center",
         fontsize=7.2, color=COND_EDGE, style="italic")

# post-attn: cross-attn (target only) + mlp, both columns
arrow(axL, LC, attn_y, LC, 5.62)
arrow(axL, TC, attn_y, TC, 5.62)
rbox(axL, TC - 1.6, 5.10, 3.2, 0.5, BOX_FILL, BOX_EDGE,
     "AdaLN$_{cross}$ · cross-attn(text)", fs=8.0)
rbox(axL, LC - 1.6, 5.10, 3.2, 0.5, "#f7f7f7", "#bbbbbb",
     "(cross-attn skipped)", fs=8.0, tc="#999999")

arrow(axL, LC, 5.10, LC, 4.55); arrow(axL, TC, 5.10, TC, 4.55)
rbox(axL, TC - 1.6, 4.05, 3.2, 0.5, BOX_FILL, BOX_EDGE, "AdaLN$_{mlp}$ · MLP", fs=8.5)
rbox(axL, LC - 1.6, 4.05, 3.2, 0.5, BOX_FILL, BOX_EDGE,
     "AdaLN$_{mlp}$(0) · MLP", fs=8.5)

# cond-LoRA callout
axL.annotate("+ cond-LoRA  (q/k/v/o, ffn)\n+ cond_gate · residual",
             xy=(LC - 1.6, 4.3), xytext=(0.05, 1.75),
             fontsize=7.8, color=COND_EDGE,
             arrowprops=dict(arrowstyle="-|>", color=COND_EDGE, lw=1.1))

# x L
axL.add_patch(mpatches.FancyBboxPatch(
    (LC - 1.85, 3.78), (TC + 1.85) - (LC - 1.85), 8.55 - 3.78,
    boxstyle="round,pad=0.02,rounding_size=0.06",
    fill=False, edgecolor="#bcbcbc", lw=1.1, linestyle=(0, (5, 3)), zorder=0))
axL.text(TC + 2.05, 6.3, "× L", fontsize=12, color="#888888", weight="bold")

# outputs
arrow(axL, LC, 4.05, LC, 3.45); arrow(axL, TC, 4.05, TC, 3.45)
token_row(axL, cx0, 2.92, ["$c_1'$", "$c_2'$", "$\\cdots$", "$c_m'$"], COND_FILL, COND_EDGE)
token_row(axL, tx0, 2.92, ["$\\hat z_1$", "$\\hat z_2$", "$\\cdots$", "$\\hat z_n$"], TGT_FILL, TGT_EDGE)
axL.text(LC, 2.55, "(cached at inference)", ha="center", fontsize=7.4,
         color=COND_EDGE, style="italic")
axL.text(TC, 2.55, "denoised target  →  velocity", ha="center", fontsize=8,
         color=TGT_EDGE)

# =============================== RIGHT: attention mask =========================
axR.text(5, 9.78, "Attention Mask", ha="center", fontsize=13, weight="bold", color=INK)

# grid box
gx0, gy0, gx1, gy1 = 2.0, 1.7, 9.0, 8.7
mid_x = (gx0 + gx1) / 2
mid_y = (gy0 + gy1) / 2

# four tiles  (note y is inverted: queries top->down)
# top-left: cond x cond -> full (bidirectional)
rbox(axR, gx0, mid_y, mid_x - gx0, gy1 - mid_y, MASK_COND, "#ffffff", lw=0.0)
# top-right: cond x target -> masked
rbox(axR, mid_x, mid_y, gx1 - mid_x, gy1 - mid_y, MASK_ZERO, "#ffffff", lw=0.0)
# bottom-left: target x cond -> full + b_cond
rbox(axR, gx0, gy0, mid_x - gx0, mid_y - gy0, MASK_FULL, "#ffffff", lw=0.0)
# bottom-right: target x target -> full
rbox(axR, mid_x, gy0, gx1 - mid_x, mid_y - gy0, MASK_FULL, "#ffffff", lw=0.0)

# tile labels
axR.text((gx0 + mid_x) / 2, (mid_y + gy1) / 2, "cond self-attn\n(FULL,\nbidirectional)",
         ha="center", va="center", fontsize=9.5, color=COND_EDGE, weight="bold")
axR.text((mid_x + gx1) / 2, (mid_y + gy1) / 2 + 0.25, "Masked\n(zero)",
         ha="center", va="center", fontsize=10, color="#999999")
axR.text((mid_x + gx1) / 2, (mid_y + gy1) / 2 - 0.85,
         "clean stream firewalled\nfrom noise → $K_c,V_c$ KV-cached",
         ha="center", va="center", fontsize=7.2, color="#aaaaaa", style="italic")
axR.text((gx0 + mid_x) / 2, (gy0 + mid_y) / 2, "Full attend\n+ $b_{cond}$ gate",
         ha="center", va="center", fontsize=9.5, color=TGT_EDGE, weight="bold")
axR.text((mid_x + gx1) / 2, (gy0 + mid_y) / 2, "Full\nattend",
         ha="center", va="center", fontsize=10, color=TGT_EDGE, weight="bold")

# outer + divider lines
axR.add_patch(plt.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                            edgecolor="#555555", lw=1.6, zorder=4))
axR.plot([mid_x, mid_x], [gy0, gy1], color="#555555", lw=1.6, zorder=4)
axR.plot([gx0, gx1], [mid_y, mid_y], color="#555555", lw=1.6, zorder=4)

# axis labels
axR.text(mid_x, gy1 + 0.55, "Keys (K)", ha="center", fontsize=11, weight="bold")
axR.text((gx0 + mid_x) / 2, gy1 + 0.18, "$K_{cond}$", ha="center", fontsize=10, color=COND_EDGE)
axR.text((mid_x + gx1) / 2, gy1 + 0.18, "$K_{target}$", ha="center", fontsize=10, color=TGT_EDGE)
axR.text(gx0 - 0.95, mid_y, "Queries (Q)", ha="center", va="center",
         fontsize=11, weight="bold", rotation=90)
axR.text(gx0 - 0.42, (mid_y + gy1) / 2, "$Q_{cond}$", ha="center", va="center",
         fontsize=10, color=COND_EDGE, rotation=90)
axR.text(gx0 - 0.42, (gy0 + mid_y) / 2, "$Q_{target}$", ha="center", va="center",
         fontsize=10, color=TGT_EDGE, rotation=90)

# legend
ly = 1.05
axR.add_patch(plt.Rectangle((2.0, ly), 0.34, 0.34, fc=MASK_COND, ec="#ffffff"))
axR.text(2.46, ly + 0.17, "cond self-attention (clean ref, t=0)", va="center", fontsize=8.2)
axR.add_patch(plt.Rectangle((2.0, ly - 0.5), 0.34, 0.34, fc=MASK_FULL, ec="#ffffff"))
axR.text(2.46, ly - 0.33, "target full attention (noisy latent, t)", va="center", fontsize=8.2)


# ---- cross-panel "zoom-in" connectors: Shared Self-Attention → mask ----------
from matplotlib.patches import ConnectionPatch

for (yA, yB) in [(attn_y + attn_h, gy1), (attn_y, gy0)]:
    con = ConnectionPatch(
        xyA=(attn_x1, yA), coordsA=axL.transData,
        xyB=(gx0, yB), coordsB=axR.transData,
        arrowstyle="-|>", mutation_scale=13, lw=1.2,
        color="#9a9a9a", linestyle=(0, (5, 3)), zorder=5,
    )
    fig.add_artist(con)

Path("docs/experimental/assets").mkdir(parents=True, exist_ok=True)
out = Path("docs/experimental/assets/easycontrol_attention.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
