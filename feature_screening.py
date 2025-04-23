import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shared_utils import preprocessed_data
from sklearn.ensemble import RandomForestRegressor


def feature_screening_with_OLS():
    """
    Performing the cross-sectional OLS regression for feature screening.
    Returns: DataFrame of betas per date and Series of mean absolute betas.
    """
    results = []
    df, scaled_df, factor_cols = preprocessed_data()

    for date, group in df.groupby(level='Date'):
        X = group[factor_cols]
        y = group['price_return_1M']

        X = sm.add_constant(X)

        if len(X) > len(factor_cols) + 1:
            model = sm.OLS(y, X).fit()
            res = {
                'Date': date,
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj,
            }
            res.update({f'Beta_{factor}': coef for factor, coef in model.params.items()})
            results.append(res)

    factor_analysis_df = pd.DataFrame(results)

    beta_cols = [col for col in factor_analysis_df.columns if col.startswith("Beta_")]
    mean_betas = factor_analysis_df[beta_cols].mean().sort_values(key=abs, ascending=False)

    return factor_analysis_df, mean_betas


def feature_screening_with_tstats():
    """
    Cross-sectional OLS regression per date with t-statistics and betas.
    Returns: DataFrame of R-squared, Betas, and T-stats for each date.
    """
    results = []
    df, scaled_df, factor_cols = preprocessed_data()

    for date, group in df.groupby(level='Date'):
        X = group[factor_cols]
        y = group['price_return_1M']
        X = sm.add_constant(X)

        if len(X) > len(factor_cols) + 1:
            model = sm.OLS(y, X).fit()

            res = {
                'Date': date,
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj,
            }

            # Store Betas
            for factor, coef in model.params.items():
                res[f'Beta_{factor}'] = coef

            # Store t-stats
            for factor, tval in model.tvalues.items():
                res[f'Tstat_{factor}'] = tval

            results.append(res)

    return pd.DataFrame(results)
    

def plot_feature_imp(mean_betas, top_n=15):
    """
    Plot the top N features by average absolute beta value.
    """
    mean_betas.head(top_n).plot(kind='barh', figsize=(10, 6), color='cornflowerblue')
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Features by Avg Absolute OLS Coefficient")
    plt.xlabel("Average Absolute Coefficient")
    plt.tight_layout()
    plt.show()


def plot_rf_feature_importance(top_n=20):
    """
    Training a RandmForestRegressor on the givn features and target.
    Plots the top_n features based on featre importance.
    
    Parameters:
    - df: DataFrame with your data
    - features: list of feature column names
    - target: name of the target column
    - top_n: number of top features to show
    - random_state: for reproducibility
    """
    df, scaled_df, factor_cols = preprocessed_data()
    all_rf_results = []

    for date, group in scaled_df.groupby(level='Date'):
        X = group[factor_cols]
        y = df.loc[date]['price_return_1M'] 

        if len(X) > len(factor_cols) + 1:  
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            result = {
                'Date': date,
                **{f'{feat}': imp for feat, imp in zip(factor_cols, rf.feature_importances_)}
            }
            all_rf_results.append(result)

    rf_df = pd.DataFrame(all_rf_results)
    rf_df.set_index('Date', inplace=True)

    mean_importances = rf_df.abs().mean().sort_values(ascending=False)
    top_features = mean_importances.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette='coolwarm')
    plt.title(f"Top {top_n} Features by Avg Random Forest Importance Over Time")
    plt.xlabel("Avg Absolute Importance")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return mean_importances, rf_df, top_features

# factor_analysis_df, mean_betas = feature_screening_with_OLS()
# factor_analysis_with_tstats = feature_screening_with_tstats()
top_rf_importances = plot_rf_feature_importance(top_n=20)
print(top_rf_importances)
# print(factor_analysis_with_tstats)
# plot_feature_imp(mean_betas, top_n=15)

mean_importances, rf_df, top_features = plot_rf_feature_importance(top_n=20)
print(top_features)


