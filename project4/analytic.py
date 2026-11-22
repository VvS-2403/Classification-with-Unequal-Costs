"""Analytic accuracy-vs-cost model (T6).

1-D model:  y ~ Bernoulli(pi1),  x | y=1 ~ N(+mu, 1),  x | y=0 ~ N(-mu, 1),  mu > 0.

Posterior:  P(y=1 | x) = sigma(2 mu x + lam),  lam = log(pi1 / pi0),
so a probability threshold t maps to the x-threshold
            x_t = (logit(t) - lam) / (2 mu).

A rule with x-thresholds tau0 <= tau1 delivers if x <= tau0, sends to spam
if x >= tau1 and inspects in between.  With Phi the standard normal CDF:

    P(deliver | y=0) = Phi(tau0 + mu)          P(deliver | y=1) = Phi(tau0 - mu)
    P(spam    | y=0) = Phi(-tau1 - mu)         P(spam    | y=1) = Phi(mu - tau1)
    P(inspect | y=k) = 1 - the two above

Every quantity in the report's Section 8 (accuracy, realized cost, rejection
rate, Bayes risk as a function of c_I) is a closed-form combination of these.
These functions are what C6 overlays on the synthetic-data experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from .decision import LossMatrix, thresholds


@dataclass(frozen=True)
class GaussianModel1D:
    mu: float = 1.0
    pi1: float = 0.4

    def __post_init__(self):
        if not self.mu > 0 or not 0 < self.pi1 < 1:
            raise ValueError("need mu > 0 and 0 < pi1 < 1")

    @property
    def pi0(self) -> float:
        return 1.0 - self.pi1

    @property
    def lam(self) -> float:
        """Prior log-odds log(pi1 / pi0)."""
        return float(np.log(self.pi1 / self.pi0))

    # -- probabilities <-> x -------------------------------------------------
    def posterior(self, x):
        return expit(2 * self.mu * np.asarray(x, float) + self.lam)

    def x_threshold(self, t):
        """x-threshold equivalent to the probability threshold t (t<=0 -> -inf, t>=1 -> +inf)."""
        t = np.asarray(t, float)
        with np.errstate(divide="ignore"):
            z = logit(np.clip(t, 0.0, 1.0))
        return (z - self.lam) / (2 * self.mu)

    # -- rates of a rule ------------------------------------------------------
    def conditional_rates(self, tau0, tau1) -> np.ndarray:
        """Q[y, a] = P(action a | y) for the rule (tau0, tau1); columns deliver/spam/inspect."""
        tau0, tau1 = float(tau0), float(tau1)
        if tau0 > tau1:
            raise ValueError("need tau0 <= tau1")
        m = self.mu
        Q = np.empty((2, 3))
        Q[0, 0], Q[0, 1] = norm.cdf(tau0 + m), norm.cdf(-tau1 - m)
        Q[1, 0], Q[1, 1] = norm.cdf(tau0 - m), norm.cdf(m - tau1)
        Q[:, 2] = np.clip(1.0 - Q[:, 0] - Q[:, 1], 0.0, 1.0)
        return Q

    def joint_rates(self, tau0, tau1) -> np.ndarray:
        """J[y, a] = P(y, action a): the population confusion matrix (sums to 1)."""
        return np.array([[self.pi0], [self.pi1]]) * self.conditional_rates(tau0, tau1)

    def reject_rate(self, tau0, tau1) -> float:
        return float(self.joint_rates(tau0, tau1)[:, 2].sum())

    def accuracy(self, tau0, tau1=None, on_reject: str = "exclude") -> float:
        """Accuracy of the rule; ``tau1=None`` means a two-action rule at tau0.

        on_reject has the same meaning as in ``metrics.accuracy``.
        """
        tau1 = tau0 if tau1 is None else tau1
        J = self.joint_rates(tau0, tau1)
        correct, rho = J[0, 0] + J[1, 1], J[:, 2].sum()
        if on_reject == "exclude":
            return float(correct / (1 - rho)) if rho < 1 else float("nan")
        if on_reject == "correct":
            return float(correct + rho)
        if on_reject == "error":
            return float(correct)
        raise ValueError("on_reject must be 'exclude', 'correct' or 'error'")

    def cost(self, tau0, tau1, L: LossMatrix) -> float:
        """Expected loss per email:  c_fp pi0 Phi(-tau1-mu) + c_fn pi1 Phi(tau0-mu) + c_I * rho."""
        J = self.joint_rates(tau0, tau1)
        return float((J * L.matrix.T).sum())

    def two_action_cost(self, tau, L: LossMatrix) -> float:
        return self.cost(tau, tau, L)

    # -- Bayes rule -----------------------------------------------------------
    def bayes_x_thresholds(self, L: LossMatrix) -> tuple[float, float]:
        theta0, theta1 = thresholds(L)
        return float(self.x_threshold(theta0)), float(self.x_threshold(theta1))

    def bayes_risk(self, L: LossMatrix) -> float:
        return self.cost(*self.bayes_x_thresholds(L), L)

    def bayes_reject_rate(self, L: LossMatrix) -> float:
        return self.reject_rate(*self.bayes_x_thresholds(L))

    def best_two_action_risk(self, L: LossMatrix) -> float:
        """Bayes risk when inspection is not allowed (single threshold c_fp/(c_fp+c_fn))."""
        tau = float(self.x_threshold(L.two_action_threshold))
        return self.two_action_cost(tau, L)

    @property
    def accuracy_threshold(self) -> float:
        """x-threshold of the accuracy-maximising rule (probability threshold 1/2)."""
        return -self.lam / (2 * self.mu)

    # -- curves for plotting / overlays --------------------------------------
    def curves_vs_c_insp(self, c_insp_grid, c_fp: float, c_fn: float) -> dict:
        """Bayes risk, rejection rate and accuracy (on non-inspected emails) against c_I."""
        c = np.asarray(c_insp_grid, float)
        risk, rho, acc = (np.empty_like(c) for _ in range(3))
        for i, ci in enumerate(c):
            L = LossMatrix(c_fp, c_fn, ci)
            t0, t1 = self.bayes_x_thresholds(L)
            risk[i], rho[i], acc[i] = self.cost(t0, t1, L), self.reject_rate(t0, t1), self.accuracy(t0, t1)
        L0 = LossMatrix(c_fp, c_fn, 0.0)
        return {
            "c_insp": c,
            "bayes_risk": risk,
            "reject_rate": rho,
            "accuracy": acc,
            "two_action_risk": np.full_like(c, self.best_two_action_risk(L0)),
            "critical_c_insp": L0.critical_c_insp,
        }

    def curves_vs_threshold(self, tau_grid, L: LossMatrix) -> dict:
        """Accuracy and realized cost of two-action rules 'spam iff x >= tau'."""
        tau = np.asarray(tau_grid, float)
        acc = np.array([self.accuracy(t) for t in tau])
        cost = np.array([self.two_action_cost(t, L) for t in tau])
        return {"tau": tau, "accuracy": acc, "cost": cost}
