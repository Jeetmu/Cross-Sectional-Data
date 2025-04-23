import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from shared_utils import preprocessed_data
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
import matplotlib.pyplot as plt
import joblib

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Training LightGBM model with Optuna for hyperparameter tuning.
    """
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        }
        
        model = lgb.LGBMRegressor(**param)
        evals = [(X_val, y_val)]
        model.fit(X_train, y_train,
                  eval_set=evals,
                  eval_metric='rmse',
                  #early_stopping=True,
                  )
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model with Optuna for hyperparameter tuning.
    """
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        }
        model = xgb.XGBRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    return model

def ensemble_predictions(models, X_test):
    """
    Combine predictions from multiple models via averaging.
    """
    preds = [model.predict(X_test) for model in models]
    return sum(preds) / len(models)

def cross_sectional_training(df):
    """
    Train and validate models using a cross-sectional approach, grouped by 'Date'.
    Also return predictions and actuals for backtesting.
    """
    performance_results = []
    backtest_data = []

    for date, group in df.groupby(level='Date'):
        X = group.drop(columns=['price_return_1M'])
        y = group['price_return_1M']

        train_size = int(0.8 * len(group))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        ensemble_preds = ensemble_predictions([lgb_model, xgb_model], X_val)

        r2 = r2_score(y_val, ensemble_preds)
        rmse = mean_squared_error(y_val, ensemble_preds, squared=False)
        performance_results.append({'Date': date, 'R2': r2, 'RMSE': rmse})

        backtest_df = pd.DataFrame({
            'Date': date,
            'Ticker': y_val.index.get_level_values('Company'),
            'Actual': y_val.values,
            'Predicted': ensemble_preds
        })
        backtest_data.append(backtest_df)

    results_df = pd.DataFrame(performance_results)
    backtest_df = pd.concat(backtest_data).reset_index(drop=True)

    return results_df, backtest_df


if __name__ == "__main__":
    df, scale, factor, merge_df = preprocessed_data()
    results_df, backtest_data_df = cross_sectional_training(merge_df)
    joblib.dump({'results': results_df, 'backtest': backtest_data_df}, "all_outputs.pkl")
    print(results_df)
    print(backtest_data_df)


