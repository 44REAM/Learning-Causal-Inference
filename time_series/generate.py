import numpy as np
import pandas as pd
from scipy.special import expit


def simulate_blackwell_glynn(n_units=1000, t_periods=10, seed=42):
    # ----------------------------------------------------------------
    # 1) Simulate Blackwell & Glynn style panel data
    # ----------------------------------------------------------------
    np.random.seed(seed)
    # true blip‐parameters
    mu_11 = mu_201 = mu_211 = -0.1  
    alpha0, alpha1, alpha2 = -1.3, 1.5, 2.5
    gamma0, gamma1 = 0.5, -0.5
    sigma = np.sqrt(0.12)

    rows = []
    for i in range(n_units):
        U = np.random.normal(0, 0.1)
        Z_prev = np.random.normal(0.4, 0.1)
        Y_prev = 0.0
        X_prev = 0

        for t in range(t_periods):
            # time‐varying covariate Z
            Z1 = gamma0 + gamma1 + 0.7*U + np.random.normal(0, sigma)
            Z0 = gamma0        + 0.1*U + np.random.normal(0, sigma)
            Z  = X_prev*Z1 + (1-X_prev)*Z0

            # treatment assignment
            p = expit(alpha0 + alpha1*Z + alpha2*Y_prev)
            X = np.random.binomial(1, p)

            # structural nested mean model for Y:
            # Y = baseline + ψ0 * X + ψ1 * X_prev + noise,
            # where ψ0 = mu_11 and ψ1 = mu_201 (same as mu_211 here)
            base = 0.8 + 0.9*U + np.random.normal(0, sigma)
            Y = base + mu_11*X + mu_201*X_prev

            rows.append({
                "i": i, "t": t,
                "U": U, "Z": Z, "Y": Y,
                "X": X
            })

            X_prev, Y_prev, Z_prev = X, Y, Z

    return pd.DataFrame(rows)