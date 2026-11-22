"""T6 figures and table from the closed-form 1-D Gaussian model (no sampling).

Model: one feature x per email. Real: x ~ N(-mu, 1); spam: x ~ N(+mu, 1); P(spam) = pi1
(mu = 1, pi1 = 0.4). Bayes' rule gives p(x) = sigmoid(2*mu*x + lambda) with
lambda = ln(pi1 / (1 - pi1)), which is increasing in x. So each threshold on p maps to a
cut-point on x, and every action rate, accuracy and expected cost is a difference of
normal CDFs. Nothing is fitted and nothing is sampled, so the numbers are exact and
serve as the reference that the learned models are compared against.

Outputs
  figures/t6_accuracy_vs_cost.{pdf,png}   accuracy and cost of two-action rules vs threshold
  figures/t6_risk_reject_vs_cinsp.{pdf,png} Bayes risk and rejection rate vs c_I
  results/t6_rule_comparison.{csv,tex}    the higher-accuracy / higher-cost exhibit
  results/t6_critical_values.csv          c*, R2 and thresholds per mu
"""
import csv

import numpy as np

from project4.analytic import GaussianModel1D
from project4.config import C_FN, C_FP, C_INSP, FIGURES_DIR, RESULTS_DIR, T6_MU, T6_PI1
from project4.decision import LossMatrix, thresholds
from project4.plotting import COLORS, INK_MUTED, apply_style, plt, save

# ---- configuration --------------------------------------------------------
MUS = (0.5, 1.0, 2.0)               # class separations for the c_I curves
C_INSP_GRID = np.linspace(0, 1.2, 481)
T_GRID = np.linspace(0.01, 0.99, 491)


def rule_table(m: GaussianModel1D):
    L = LossMatrix(C_FP, C_FN, C_INSP)
    rules = [
        ("Accuracy rule ($t=0.5$)", m.accuracy_threshold, m.accuracy_threshold),
        (r"Cost-optimal, no inspection ($t=t^\ast$)",) + (float(m.x_threshold(L.two_action_threshold)),) * 2,
        ("Always deliver", np.inf, np.inf),
        (f"Bayes rule, inspection allowed ($c_I={C_INSP:g}$)",) + m.bayes_x_thresholds(L),
    ]
    rows = []
    for name, t0, t1 in rules:
        Q = m.conditional_rates(t0, t1)
        rows.append({
            "rule": name, "tau0": t0, "tau1": t1,
            "accuracy": m.accuracy(t0, t1), "cost": m.cost(t0, t1, L),
            "reject_rate": m.reject_rate(t0, t1), "fpr": Q[0, 1], "fnr": Q[1, 0],
        })
    return rows


def write_table(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "t6_rule_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Rule & Accuracy & Cost & Inspected & FPR & FNR \\", r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['rule']} & {r['accuracy']:.3f} & {r['cost']:.3f} & {r['reject_rate']:.3f} "
            f"& {r['fpr']:.3f} & {r['fnr']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / "t6_rule_comparison.tex").write_text("\n".join(lines) + "\n")


def fig_accuracy_vs_cost(m: GaussianModel1D):
    L = LossMatrix(C_FP, C_FN, C_INSP)
    tau = m.x_threshold(T_GRID)
    cur = m.curves_vs_threshold(tau, L)
    t_star = L.two_action_threshold
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    # label offsets chosen so text never sits on the curve
    offsets = {("accuracy", 0.5): ((6, 8), "left"), ("accuracy", t_star): ((-8, -24), "right"),
               ("cost", 0.5): ((6, 6), "left"), ("cost", t_star): ((0, 18), "center")}
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"),
                           (axes[1], "cost", "Expected cost per email (log scale)")):
        ax.plot(T_GRID, cur[key], color=COLORS[0])
        for t, name in ((0.5, "$t=0.5$"), (t_star, r"$t^\ast$")):
            val = m.accuracy(m.x_threshold(t)) if key == "accuracy" else m.two_action_cost(m.x_threshold(t), L)
            ax.axvline(t, color=INK_MUTED, lw=0.8, ls="--")
            ax.plot([t], [val], "o", ms=5, color=COLORS[0], mec="white", mew=1.5)
            off, ha = offsets[(key, t)]
            ax.annotate(f"{name}: {val:.3f}", (t, val), xytext=off, textcoords="offset points",
                        ha=ha, fontsize=8, color=INK_MUTED)
        ax.set_xlabel("Probability threshold $t$ (spam iff $p \\geq t$)")
        ax.set_title(label, loc="left")
    axes[1].set_yscale("log")
    axes[1].set_yticks([0.3, 0.5, 1, 2, 5])
    axes[1].set_yticklabels(["0.3", "0.5", "1", "2", "5"])
    axes[1].minorticks_off()
    axes[0].set_ylim(top=0.9)
    fig.suptitle(rf"Two-action rules, $\mu={m.mu:g}$, $\pi_1={m.pi1:g}$, $c_{{FP}}={C_FP:g}$, $c_{{FN}}={C_FN:g}$",
                 x=0.02, ha="left", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "t6_accuracy_vs_cost")


def fig_risk_reject(mus=MUS):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    crit = LossMatrix(C_FP, C_FN, 0.0).critical_c_insp
    rows = []
    for k, mu in enumerate(mus):
        m = GaussianModel1D(mu, T6_PI1)
        cur = m.curves_vs_c_insp(C_INSP_GRID, C_FP, C_FN)
        lab = rf"$\mu={mu:g}$"
        axes[0].plot(C_INSP_GRID, cur["bayes_risk"], color=COLORS[k], label=lab)
        axes[0].axhline(cur["two_action_risk"][0], color=COLORS[k], lw=0.8, ls=":")
        axes[1].plot(C_INSP_GRID, cur["reject_rate"], color=COLORS[k], label=lab)
        th0, th1 = thresholds(LossMatrix(C_FP, C_FN, C_INSP))
        rows.append({"mu": mu, "pi1": T6_PI1, "critical_c_insp": crit,
                     "two_action_risk": cur["two_action_risk"][0],
                     "bayes_risk_at_default_cI": m.bayes_risk(LossMatrix(C_FP, C_FN, C_INSP)),
                     "reject_rate_at_default_cI": m.bayes_reject_rate(LossMatrix(C_FP, C_FN, C_INSP)),
                     "theta0": th0, "theta1": th1})
    for ax, title in ((axes[0], r"Bayes risk $R^\ast(c_I)$"),
                      (axes[1], r"Rejection rate $\rho(c_I) = dR^\ast/dc_I$")):
        ax.axvline(crit, color=INK_MUTED, lw=0.8, ls="--")
        ax.annotate(rf"$c^\ast={crit:.3f}$", (crit, 0.5), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_MUTED)
        ax.set_xlabel("Inspection cost $c_I$")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].plot([], [], color=INK_MUTED, lw=0.8, ls=":", label="no inspection")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "t6_risk_reject_vs_cinsp")
    with open(RESULTS_DIR / "t6_critical_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    apply_style()
    m = GaussianModel1D(T6_MU, T6_PI1)
    rows = rule_table(m)
    write_table(rows)
    fig_accuracy_vs_cost(m)
    fig_risk_reject()
    for r in rows:
        print(f"  {r['rule'][:45]:45s} acc={r['accuracy']:.3f} cost={r['cost']:.3f} reject={r['reject_rate']:.3f}")


if __name__ == "__main__":
    main()
