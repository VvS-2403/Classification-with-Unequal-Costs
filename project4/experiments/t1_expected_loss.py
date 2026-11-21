"""T1 figure: posterior expected loss of each action against p = P(y=1 | x).

Member 1 (T1, Section 2).  No data or models: the curves follow from the loss
matrix alone, R(0|x) = c_fn p, R(1|x) = c_fp (1-p), R(r|x) = c_I.

Outputs
  figures/t1_expected_loss.{pdf,png}   left: our costs (c_fp=10, c_fn=1); right: equal costs
                                       (c_fp=c_fn=1, Chow's rule); both at c_I = C_INSP
"""
import numpy as np

from project4.config import C_FN, C_FP, C_INSP, FIGURES_DIR
from project4.decision import LossMatrix, expected_losses, thresholds
from project4.plotting import COLORS, INK, INK_MUTED, apply_style, plt, save

# ---- configuration --------------------------------------------------------
EQUAL_COST = 1.0                    # sanity-check panel: c_fp = c_fn, so t* = 1/2
P_GRID = np.linspace(0.0, 1.0, 1001)
LINE_LABELS = ("deliver: $c_{FN}\\,p$", "spam folder: $c_{FP}(1-p)$", "inspect: $c_I$")
REGION_ALPHA = 0.07


def panel(ax, L: LossMatrix):
    R = expected_losses(P_GRID, L)
    th0, th1 = thresholds(L)
    # Shade where each action is optimal (same colour as its line).
    edges = (0.0, th0, th1, 1.0) if L.has_reject_band else (0.0, th0, th0, 1.0)
    for k, (lo, hi) in zip((0, 2, 1), zip(edges[:-1], edges[1:])):
        if hi > lo:
            ax.axvspan(lo, hi, color=COLORS[k], alpha=REGION_ALPHA, lw=0)
    for k in range(3):
        ax.plot(P_GRID, R[:, k], color=COLORS[k], label=LINE_LABELS[k])
    ax.plot(P_GRID, R.min(axis=1), color=INK, lw=1.0, ls="--", label="minimum (Bayes action)")
    ymax = 1.5
    if L.has_reject_band:
        marks = ((th0, rf"$\theta_0={th0:.2f}$", "right"), (th1, rf"$\theta_1={th1:.2f}$", "right"))
    else:
        marks = ((th0, rf"$t^\ast={th0:.3f}$", "right"),)
    for t, text, ha in marks:
        ax.axvline(t, color=INK_MUTED, lw=0.8, ls=":")
        ax.annotate(text, (t, ymax), xytext=(-3, -10), textcoords="offset points",
                    ha=ha, fontsize=8, color=INK_MUTED)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$p = P(y=1 \\mid x)$")
    band = "inspect band" if L.has_reject_band else "no inspect band"
    ax.set_title(rf"$c_{{FP}}={L.c_fp:g}$, $c_{{FN}}={L.c_fn:g}$ ({band}, $c^\ast={L.critical_c_insp:.3f}$)",
                 loc="left")


def main():
    apply_style()
    with_band = LossMatrix(C_FP, C_FN, C_INSP)
    equal = LossMatrix(EQUAL_COST, EQUAL_COST, C_INSP)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    panel(axes[0], with_band)
    panel(axes[1], equal)
    axes[0].set_ylabel("Posterior expected loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(rf"$c_I={C_INSP:g}$",
                 x=0.02, ha="left", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    save(fig, FIGURES_DIR / "t1_expected_loss")
    for L in (with_band, equal):
        print(f"  c_I={L.c_insp:g}: thresholds={tuple(round(t, 4) for t in thresholds(L))}, "
              f"band={L.has_reject_band}")


if __name__ == "__main__":
    main()
