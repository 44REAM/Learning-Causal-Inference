
import statsmodels.api as sm


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
    controlling for all specified confounders (X, Y_lag, and Z).
    """
    df = df.copy()
    df = create_lagged_features(df,1, unit ,treatment, outcome, common_causes)

    # --- Step 1: Fit the outcome model, controlling for ALL confounders ---

    col1 = [f"{treatment}_0", f"{treatment}_1", f"{outcome}_1"]
    col1.extend([col+"_0" for col in common_causes])

    model1 = sm.OLS(df[f"{outcome}_0"], sm.add_constant(df[col1]))
    res1 = model1.fit()

    # --- Step 2: Blip down the outcome ---
    # Remove the estimated contemporaneous effect of X from Y
    Y_tilde = df[f"{outcome}_0"] - res1.params[f"{treatment}_0"] * df[f"{treatment}_0"]

    # --- Step 1: Fit the outcome model for lag 1
    col2 = [f"{treatment}_1", f"{treatment}_2", f"{outcome}_2"]
    col2.extend([col+"_1" for col in common_causes])
    model2 = sm.OLS(Y_tilde, sm.add_constant(df[col2]))
    
    res2 = model2.fit()
    
    # We return the coefficients of treatment ()
    return res1.params['X_0'], res2.params['X_1']

