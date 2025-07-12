
import numpy as np
from sklearn.linear_model import LinearRegression

def create_lagged_features(df, n_lags, unit, treatment, outcome, common_causes):
    """Creates multiple lags for specified columns, grouped by unit."""
    n_lags+=1
    df_out = df.copy()
    for i in range(n_lags+1):
        df_out[f"{treatment}_{i}"] = df_out.groupby(unit)[treatment].shift(i)

    for i in range(0, n_lags+1):
        df_out[f"{outcome}_{i}"] = df_out.groupby(unit)[outcome].shift(i)

    for lag in range(n_lags):
        for col in common_causes:
            df_out[f'{col}_{lag}'] = df_out.groupby(unit)[col].shift(lag)
    # Drop rows with NaNs created by the shift operation
    return df_out.dropna().reset_index(drop=True)


def nsmm_lag1(df, unit, treatment, outcome, common_causes):
    """
    Estimates a Structural Nested Mean Model using sequential g-estimation,
    controlling for all specified confounders (X, Y_lag, and Z),
    but using sklearn.linear_model.LinearRegression.
    Returns the estimated ‘blip’ parameters for X_0 and X_1.
    """
    # 1) create lagged features
    df2 = create_lagged_features(df, n_lags=1, unit=unit, 
                                 treatment=treatment, outcome=outcome, 
                                 common_causes=common_causes)

    # --- Step 1: Fit the full outcome model to Y_0 = b0 + b1 X_0 + b2 X_1 + b3 Y_1 + b4 Z_0 + ...
    y0 = df2[f"{outcome}_0"].values
    X1_cols = [f"{treatment}_0", f"{treatment}_1", f"{outcome}_1"] \
              + [f"{col}_0" for col in common_causes]
    X1 = df2[X1_cols].values

    model1 = LinearRegression(fit_intercept=True)
    model1.fit(X1, y0)
    # coefficient for treatment_0
    beta_0 = model1.coef_[X1_cols.index(f"{treatment}_0")]

    # --- Step 2: "blip down" Y_0 by removing contemporaneous effect of X_0
    Y_tilde = y0 - beta_0 * df2[f"{treatment}_0"].values

    # --- Step 3: Fit second model Y_tilde ~ X_1, X_2, Y_2, Z_1, ...
    X2_cols = [f"{treatment}_1", f"{treatment}_2", f"{outcome}_2"] \
              + [f"{col}_1" for col in common_causes]
    X2 = df2[X2_cols].values

    model2 = LinearRegression(fit_intercept=True)
    model2.fit(X2, Y_tilde)
    beta_1 = model2.coef_[X2_cols.index(f"{treatment}_1")]

    return beta_0, beta_1

def nsmm_lag1_cate(df, unit, treatment, outcome, common_causes):
    """
    Sequential g‐estimation that returns not just average blip parameters
    but their dependence on covariates (CATE function).
    
    Outputs:
     - beta0, gamma0: baseline & covariate‐slopes for X_0 effect
     - beta1, gamma1: baseline & covariate‐slopes for X_1 effect
    
    So that for a unit with covariate‐vector C_0:
       CATE0(C_0) = beta0 + gamma0 @ C_0
    and similarly for CATE1(C_1).
    """
    # 1) create lags
    df2 = create_lagged_features(df, n_lags=1,
                                 unit=unit,
                                 treatment=treatment,
                                 outcome=outcome,
                                 common_causes=common_causes)
    
    # --- STEP 1: model Y0
    y0 = df2[f"{outcome}_0"].values
    
    # basic regressors
    X0_cols = [f"{treatment}_0", f"{treatment}_1", f"{outcome}_1"] \
              + [f"{col}_0" for col in common_causes]
    X0 = df2[X0_cols].values
    
    # build interaction terms: X0 * each C0
    C0 = df2[[f"{col}_0" for col in common_causes]].values
    TX0 = (df2[f"{treatment}_0"].values[:,None] * C0)
    
    # assemble final design for step 1
    # [ intercept automatically added by LinearRegression ]
    # columns = [ X0_basic | treatment0*C0_1 | ... | treatment0*C0_k ]
    X1_design = np.hstack([X0, TX0])
    
    model1 = LinearRegression(fit_intercept=True)
    model1.fit(X1_design, y0)
    
    # extract the effect‐on‐X0 coefficients:
    # the first element of coef_ pertaining to treatment_0
    idx_t0 = X0_cols.index(f"{treatment}_0")
    beta0 = model1.coef_[idx_t0]
    
    # the interaction coefficients are right after the basic block:
    # They correspond to treatment0*C0_j
    gamma0 = model1.coef_[len(X0_cols):len(X0_cols)+len(common_causes)]
    
    # --- STEP 2: blip‐down Y0 by removing conc. effect of X0
    Y_tilde = y0 - (beta0 + C0.dot(gamma0)) * df2[f"{treatment}_0"].values
    
    # --- STEP 3: model on Y_tilde
    # basic regressors for the second step
    X1_cols = [f"{treatment}_1", f"{treatment}_2", f"{outcome}_2"] \
              + [f"{col}_1" for col in common_causes]
    X1 = df2[X1_cols].values
    
    # interactions: treatment_1 * C1
    C1 = df2[[f"{col}_1" for col in common_causes]].values
    TX1 = (df2[f"{treatment}_1"].values[:,None] * C1)
    
    X2_design = np.hstack([X1, TX1])
    
    model2 = LinearRegression(fit_intercept=True)
    model2.fit(X2_design, Y_tilde)
    
    idx_t1 = X1_cols.index(f"{treatment}_1")
    beta1 = model2.coef_[idx_t1]
    gamma1 = model2.coef_[len(X1_cols):len(X1_cols)+len(common_causes)]
    
    return {
        "beta0": beta0,
        "gamma0": gamma0,      # array of length len(common_causes)
        "beta1": beta1,
        "gamma1": gamma1       # same length
    }