import pandas as pd
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

class MarginalStructuralModel:
    def __init__(self, treatment, outcome, common_causes, id_col='patient_id', time_col='time'):
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.id_col = id_col
        self.time_col = time_col
        self.propensity_models = {}
        self.propensity_marginal_models = {}
        self.outcome_model = None

        self.full_features = [f'cum_{a}' for a in self.treatment]
        self.full_features += [f'{col}_lag1' for col in [self.outcome] + self.common_causes]
        self.marg_features = [f'cum_{x}' for x in self.treatment]

        self.model_features = [f'cum_{a}' for a in self.treatment]
        self.model_features += self.common_causes
        self.model_features += [f'{col}_lag1' for col in [self.outcome] + self.common_causes]

    def _prepare_data(self, df):
        df = df.sort_values([self.id_col, self.time_col]).copy()
        for a in self.treatment:
            df[f'cum_{a}'] = df.groupby(self.id_col)[a].cumsum().shift(fill_value=0)
        for col in [self.outcome] + self.common_causes:
            df[f'{col}_lag1'] = df.groupby(self.id_col)[col].shift(1)
        df = df.dropna()
        return df

    def _fit_propensity_models(self, df):
        for a in self.treatment:
            X_full = df[self.full_features]
            y = df[a]
            self.propensity_models[a] = LogisticRegression().fit(X_full, y)

            X_marg = df[self.marg_features]
            self.propensity_marginal_models[a] = LogisticRegression().fit(X_marg, y)
    
    def _compute_stabilized_weights(self, df):
        df = df.copy()
        df['sw'] = 1.0  # Initialize

        for a in self.treatment:
            # Full model includes confounders
            X_full = df[self.full_features]
            X_marg = df[self.marg_features]

            p_full = self.propensity_models[a].predict_proba(X_full)[:, 1]
            p_marg = self.propensity_marginal_models[a].predict_proba(X_marg)[:, 1]

            actual = df[a]
            weight = np.where(actual == 1, p_marg / p_full, (1 - p_marg) / (1 - p_full))

            # Store intermediate weight component
            df[f'w_{a}'] = weight

        # Multiply all treatment components
        df['inst_w'] = df[[f'w_{a}' for a in self.treatment]].prod(axis=1)

        # Compute stabilized weight per patient using cumulative product
        df['sw'] = (
            df.sort_values([self.id_col, self.time_col])
            .groupby(self.id_col)['inst_w']
            .cumprod()
        )

        # Truncate and normalize
        df['sw'] = df['sw'].clip(lower=df['sw'].quantile(0.01), upper=df['sw'].quantile(0.99))
        df['sw'] = df['sw'] / df['sw'].mean()

        return df

    def _fit_outcome_model(self, df):

        X = df[self.model_features]
        y = df[self.outcome]
        w = df['sw']

        self.outcome_model = LinearRegression().fit(X, y, sample_weight=w)

    def fit(self, df):
        df = self._prepare_data(df)
        self._fit_propensity_models(df)
        df = self._compute_stabilized_weights(df)
        self._fit_outcome_model(df)
        self._fitted_df = df  # optionally retain for debugging or prediction

    def predict(self, df):
        df = self._prepare_data(df)

        return self.outcome_model.predict(df[self.model_features])

    def get_model(self):
        return self.outcome_model

    def get_fitted_data(self):
        return self._fitted_df
    
    def summary(self):
        model = self.outcome_model
        coef_names = self.model_features
        print("Model coefficients:")
        for name, coef in zip(coef_names, model.coef_):
            print(f"{name:>20}: {coef: .3f}")
        print(f"{'Intercept':>20}: {model.intercept_: .3f}")
    
class MarginalStructuralModel:
    def __init__(self, treatment, 
                 outcome, common_causes, 
                 effect_modifiers = [], 
                 id_col='patient_id', time_col='time'):
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.id_col = id_col
        self.time_col = time_col
        self.user_effect_modifiers = effect_modifiers
        self.propensity_models = {}
        self.propensity_marginal_models = {}
        self.outcome_model = None

    def _prepare_data(self, df, drop = True):
        df = df.sort_values([self.id_col, self.time_col]).copy()

        for a in self.treatment:
            df[f'{a}_lag1'] = df.groupby(self.id_col)[a].shift(1)

        for col in [self.outcome] + self.common_causes:
            df[f'{col}_lag1'] = df.groupby(self.id_col)[col].shift(1)

        if drop:
            df = df.dropna()

        # Define features dynamically after shift
        self.full_features = self.treatment + [f'{a}_lag1' for a in self.treatment]
        self.full_features += [f'{col}_lag1' for col in [self.outcome] + self.common_causes]

        self.marg_features = self.treatment + [f'{a}_lag1' for a in self.treatment]

        self.model_features = self.treatment + [f'{a}_lag1' for a in self.treatment]
        self.model_features += self.common_causes
        self.model_features += [f'{col}_lag1' for col in [self.outcome] + self.common_causes]

        # Collect all effect modifiers (user + auto-added)
        lagged_treatment = [f'{a}_lag1' for a in self.treatment]
        lagged_outcome = [f'{self.outcome}_lag1']
        self.effect_modifiers = list(set(
            self.user_effect_modifiers + lagged_treatment + lagged_outcome
        ))

        # Add effect modifiers and interaction terms
        self.interaction_terms = []

        for a, em in itertools.product(self.treatment, self.effect_modifiers):
            interaction_name = f'{a}_x_{em}'
            df[interaction_name] = df[a] * df[em]
            self.model_features.append(interaction_name)
            self.interaction_terms.append(interaction_name)

        return df

    def _fit_propensity_models(self, df):
        for a in self.treatment:
            X_full = df[self.full_features]
            y = df[a]
            self.propensity_models[a] = LogisticRegression().fit(X_full, y)

            X_marg = df[self.marg_features]
            self.propensity_marginal_models[a] = LogisticRegression().fit(X_marg, y)

    def _compute_stabilized_weights(self, df):
        df = df.copy()

        for a in self.treatment:
            X_full = df[self.full_features]
            X_marg = df[self.marg_features]

            p_full = self.propensity_models[a].predict_proba(X_full)[:, 1]
            p_marg = self.propensity_marginal_models[a].predict_proba(X_marg)[:, 1]

            actual = df[a]
            weight = np.where(actual == 1, p_marg / p_full, (1 - p_marg) / (1 - p_full))

            df[f'w_{a}'] = weight

        df['inst_w'] = df[[f'w_{a}' for a in self.treatment]].prod(axis=1)

        # Compute stabilized weights with cumulative product
        df['sw'] = (
            df.sort_values([self.id_col, self.time_col])
              .groupby(self.id_col)['inst_w']
              .cumprod()
        )

        df['sw'] = df['sw'].clip(lower=df['sw'].quantile(0.01), upper=df['sw'].quantile(0.99))
        df['sw'] = df['sw'] / df['sw'].mean()
        return df

    def _fit_outcome_model(self, df):
        X = df[self.model_features]
        y = df[self.outcome]
        w = df['sw']
        self.outcome_model = LinearRegression().fit(X, y, sample_weight=w)

    

    def fit(self, df):
        df = self._prepare_data(df)
        self._fit_propensity_models(df)
        df = self._compute_stabilized_weights(df)
        self._fit_outcome_model(df)
        self._fitted_df = df

    def predict(self, df, prepare_data = True):
        if prepare_data:
            df = self._prepare_data(df, drop=False)
        mask = df.isnull().any(axis=1)
        preds = pd.Series(index=df.index, dtype='float64')
        
        preds[~mask] = self.outcome_model.predict(df[~mask][self.model_features])
        preds[mask] = np.nan

        return preds.to_numpy()

    def get_model(self):
        return self.outcome_model

    def get_fitted_data(self):
        return self._fitted_df
    
    def summary(self):
        model = self.outcome_model
        coef_names = self.model_features
        print("Model coefficients:")
        for name, coef in zip(coef_names, model.coef_):
            print(f"{name:>20}: {coef: .3f}")
        print(f"{'Intercept':>20}: {model.intercept_: .3f}")
