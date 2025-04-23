import pandas as pd
import matplotlib.pyplot as plt
from model import results

def visualization(df):
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(df['Date'], df['R2'], marker='o', linestyle='-', color='b')
    ax[0].axhline(0, color='gray', linestyle='--')
    ax[0].set_title('R² Over Time')
    ax[0].set_ylabel('R²')
    ax[0].grid(True)

    ax[1].plot(df['Date'], df['RMSE'], marker='s', linestyle='-', color='r')
    ax[1].set_title('RMSE Over Time')
    ax[1].set_ylabel('RMSE')
    ax[1].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

    worst_r2 = df.nsmallest(5, 'R2')[['Date', 'R2', 'RMSE']]
    worst_rmse = df.nlargest(5, 'RMSE')[['Date', 'R2', 'RMSE']]

    worst_r2, worst_rmse

print(visualization(results))