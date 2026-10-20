"""
BG/NBD and Gamma-Gamma customer models, implemented from the original papers.

BG/NBD  - Fader, Hardie & Lee (2005), "Counting your customers the easy way"
Gamma-Gamma - Fader, Hardie & Lee (2005), "RFM and CLV: Using iso-value curves"

Inputs per customer (time in days, one "purchase" = one distinct purchase day):
  x   : number of REPEAT purchases (purchase days - 1)
  t_x : days between first and last purchase
  T   : days between first purchase and the cutoff
  m   : average value of the repeat purchases (Gamma-Gamma only)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


def rfm_summary(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """orders: CustomerID, date, value (one row per invoice) -> x, t_x, T, m per customer."""
    o = orders[orders["date"] < cutoff].copy()
    o["day"] = o["date"].dt.normalize()
    d = o.groupby(["CustomerID", "day"])["value"].sum().reset_index()   # purchases per day
    g = d.groupby("CustomerID")
    first, last = g["day"].min(), g["day"].max()
    s = pd.DataFrame({
        "x": g.size() - 1,
        "t_x": (last - first).dt.days,
        "T": (cutoff.normalize() - first).dt.days,
    })
    # Gamma-Gamma uses the mean of repeat purchases only (the first purchase is excluded).
    d["rank"] = g.cumcount()
    s["m"] = d[d["rank"] > 0].groupby("CustomerID")["value"].mean()
    s["m"] = s["m"].fillna(0)
    return s


class BGNBD:
    """Each customer buys at their own Poisson rate (lambda ~ Gamma(r, alpha)) while 'alive',
    and after every purchase drops out with their own probability (p ~ Beta(a, b))."""

    def fit(self, x, t_x, T, penalizer=0.001):
        x, t_x, T = map(np.asarray, (x, t_x, T))

        def nll(log_params):
            r, alpha, a, b = np.exp(log_params)      # optimise on log scale: keeps params positive
            A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
            A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
            A3 = -(r + x) * np.log(alpha + T)
            A4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-10))
                          - (r + x) * np.log(alpha + t_x), -np.inf)
            # With under a year of history the dropout parameters (a, b) are barely identifiable:
            # the likelihood is flat along them. A small L2 penalty on a and b only keeps them stable
            # without distorting the purchase-rate parameters (r, alpha).
            return -(A1 + A2 + np.logaddexp(A3, A4)).mean() + penalizer * (a ** 2 + b ** 2)

        # Bounded optimiser from several starting points; keep the best fit.
        fits = [minimize(nll, np.log(x0), method="L-BFGS-B", bounds=[(-8, 6)] * 4)
                for x0 in ([1, 10, 1, 1], [0.5, 20, 0.5, 2], [1, 50, 1, 5])]
        best = min(fits, key=lambda f: f.fun)
        self.r, self.alpha, self.a, self.b = np.exp(best.x)
        return self

    def p_alive(self, x, t_x, T):
        x, t_x, T = map(np.asarray, (x, t_x, T))
        ratio = np.where(x > 0, self.a / np.maximum(self.b + x - 1, 1e-10)
                         * ((self.alpha + T) / (self.alpha + t_x)) ** (self.r + x), 0)
        return 1 / (1 + ratio)

    def p_no_purchase(self, t, x, t_x, T):
        """P(zero purchases in the next t days) = dead, or alive but buys nothing."""
        x, T = np.asarray(x), np.asarray(T)
        pa = self.p_alive(x, t_x, T)
        return (1 - pa) + pa * ((self.alpha + T) / (self.alpha + T + t)) ** (self.r + x)

    def expected_purchases(self, t, x, t_x, T):
        x, t_x, T = map(np.asarray, (x, t_x, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        den = 1 + np.where(x > 0, a / np.maximum(b + x - 1, 1e-10)
                           * ((alpha + T) / (alpha + t_x)) ** (r + x), 0)
        return num / den


class GammaGamma:
    """Spend per purchase varies around each customer's own average, and those averages vary
    across customers. Shrinks a customer's observed average toward the population mean - more
    so when they have few purchases."""

    def fit(self, x, m):
        x, m = np.asarray(x, float), np.asarray(m, float)
        keep = (x > 0) & (m > 0)
        x, m = x[keep], m[keep]

        def nll(log_params):
            p, q, v = np.exp(log_params)
            ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
                  + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(v + m * x))
            return -ll.mean()

        fits = [minimize(nll, np.log(x0), method="L-BFGS-B", bounds=[(-8, 8)] * 3)
                for x0 in ([1, 2, 10], [2, 3, 300], [5, 5, 100])]
        self.p, self.q, self.v = np.exp(min(fits, key=lambda f: f.fun).x)
        return self

    def expected_value(self, x, m):
        x, m = np.asarray(x, float), np.asarray(m, float)
        pop_mean = self.p * self.v / (self.q - 1)
        w = self.p * x / (self.p * x + self.q - 1)      # weight on the customer's own average
        return np.where(x > 0, (1 - w) * pop_mean + w * m, pop_mean)
