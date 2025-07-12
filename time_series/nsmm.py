
import statsmodels.api as sm
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