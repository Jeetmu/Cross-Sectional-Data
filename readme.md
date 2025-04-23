🛠️ Data Pipeline
    A panel dataset was cleaned, pivoted, and a target variable was constructed based on next-period returns (priceM1).

    The data was then scaled and passed through a feature engineering pipeline.

    Top 20 features were selected using Ordinary Least Squares (OLS) and Regression Tree importance scores (see feature_screening.py).

🤖 Modeling
    Two models were trained: LightGBM and XGBoost.

    Hyperparameter tuning was performed using Optuna (see model.py).

    Performance was evaluated using:

        1. R² (R-squared)

        2. RMSE (Root Mean Squared Error)
        (see visualization.py for plots and metrics).

💼 Backtesting Strategy
    A basic long-short strategy was constructed using model predictions:

    Long positions were taken in the top 20% of predicted returns.

    Short positions in the bottom 20%.

    Transaction costs and slippage were accounted for with:

    transaction_cost = 0.001

    slippage = 0.0005

    Cumulative returns (raw and cost-adjusted) were calculated and plotted.

📦 Output
    all_outputs.pkl contains:

        1. The actual and predicted returns.

        2. R² and RMSE evaluation metrics.

        3. Data used for backtesting.