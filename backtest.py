import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def backtest_strategy(preds, actual_returns, transaction_cost=0.001, slippage=0.0005):
    """
    Simulating a long-short backtest strategy using model predictions.
    
    Parameters:
    - preds: Predicted returns from the model
    - actual_returns: Actual future returns for the stocks
    - transaction_cost: Per-trade transaction cost (default 0.1%)
    - slippage: Slippage cost (default 0.05%)

    Returns:
    - DataFrame with raw and cost-adjusted cumulative returns
    """
    preds = np.array(preds)
    actual_returns = np.array(actual_returns)

    threshold_long = np.percentile(preds, 80)
    threshold_short = np.percentile(preds, 20)

    longs = preds >= threshold_long
    shorts = preds <= threshold_short

    strategy_returns = np.zeros_like(actual_returns)
    strategy_returns[longs] = actual_returns[longs]
    strategy_returns[shorts] = -actual_returns[shorts]

    costs = transaction_cost + slippage
    traded = longs | shorts
    cost_penalty = traded * costs
    strategy_returns_after_cost = strategy_returns - cost_penalty

    cum_returns = np.cumsum(strategy_returns)
    cum_returns_after_cost = np.cumsum(strategy_returns_after_cost)

    result_df = pd.DataFrame({
        'Cumulative Return (Raw)': cum_returns,
        'Cumulative Return (After Cost)': cum_returns_after_cost
    })

    return result_df

data = joblib.load("all_outputs.pkl")
results_df = data['results']
backtest_df = data['backtest']

backtest_result = backtest_strategy(backtest_df['Predicted'], backtest_df['Actual'])

backtest_result.plot(title="Backtest Cumulative Returns", grid=True, figsize=(10, 5))
plt.ylabel("Cumulative Return")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

backtest_result.tail()
