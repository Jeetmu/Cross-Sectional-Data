import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('Derivative Pricing\Panel_data\Input.csv')
# shape = reshape_data(df)
# shape.head(-1)

# missing_data = shape.isnull().sum()
# print('missing data', missing_data)
# missing_percent = shape.isnull().mean()
# print(f'missing data above 50% ', missing_percent[missing_percent > 0.5].index)
# clean = shape.drop(missing_percent[missing_percent > 0.5].index, axis=1)
# clean.head(-1)

# df_interpolate = clean.groupby(level='Company').apply(lambda group: group.ffill().bfill())
# df_interpolate.head(-1)

# -----------------------X-------------------------

# input_data = pd.read_csv('Derivative Pricing\Panel_data\Input.csv')
# data = reshape_data(input_data)
# data = clean_data(data)
# print(f'Interpolated data',data)
# df_sorted = data.sort_index(level=['Company','Date'])
# df_sorted['price_M1'] = df_sorted.groupby(level='Company')['Price1M'].shift(1)
# df_sorted.head(-1)

# ------------------------X--------------------------

# factor_cols = []
# for col in df_sorted.columns:
#     if col not in ['Date', 'Company', 'Price1M', 'price_M1']:
#         factor_cols.append(col)
# print('Factor Columns:', factor_cols)

# scaler = StandardScaler()
# scaled_array = scaler.fit_transform(df_sorted[factor_cols])
# df_scaled = pd.DataFrame(scaled_array, columns=factor_cols, index=df_sorted.index)
# df_scaled.head(-1)

# print('Scaled Data', df_scaled)

#---------------------------X-----------------------------

def reshape_data(df):
    ''' 
    Reshaping the dataset - Making the features as columns 
    and company and date as multi-index
    '''
    df_melted = df.melt(id_vars=["Date", "factor"], var_name="Company", value_name="Value")
    df_pivoted = df_melted.pivot_table(index=["Date", "Company"], columns="factor", values="Value")
    df_reshaped = df_pivoted.sort_index()
    return df_reshaped

def clean_data(df):
    ''' 
    - Drops columns with >50% missing data
    - Fills remaining missing values with forward/backward interpolation by company
    - Remaining NaNs are filled with median of the column
    '''
    missing_data = df.isnull().sum()
    # print('missing data', missing_data)
    missing_percent = df.isnull().mean()
    # print(f'missing data above 50% ', missing_percent[missing_percent > 0.5].index)
    df_drop = df.drop(missing_percent[missing_percent > 0.5].index, axis=1)
    df_interpolate = df_drop.groupby(level='Company').transform(lambda group: group.ffill().bfill())
    df_interpolate = df_interpolate.fillna(df_interpolate.median())
    return df_interpolate

def target_column(df):
    '''
    - Making a target variable Price1M
    - Computing it's return
    - Dropping the first row for each company
    '''
    df_sorted = df.sort_index(level=['Company','Date'])
    df_sorted['price_M1'] = df_sorted.groupby(level='Company')['Price1M'].shift(1)
    df_sorted['price_return_1M'] = df_sorted['Price1M'] / df_sorted['price_M1'] - 1
    df_sorted = df_sorted.dropna(subset=['price_return_1M'])
    df_sorted = df_sorted.drop(['price_M1', 'Price1M'], axis=1)
    return df_sorted


def scaled_feature(df):
    '''
    scaling the data for feature screening
    '''
    factor_cols = []
    for col in df.columns:
        if col not in ['Date', 'Company', 'Price1M','price_M1','price_return_1M']:
            factor_cols.append(col)
    # print('Factor Columns:', factor_cols)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[factor_cols])
    df_scaled = pd.DataFrame(scaled_array, columns=factor_cols, index=df.index)
    return df_scaled, factor_cols

def engineering_feature(df):
    top_features = [
    "ROC20",
    "net_operate_cashflow_growth_rate",
    "book_to_price_ratio",
    "liquidity",
    "cash_flow_to_price_ratio",
    "growth",
    "PLRC12",
    "PEG",
    "MAC20",
    "MFI14",
    "net_asset_per_share",
    "eps_ttm",
    "ATR14",
    "next_mth_return",
    "momentum",
    "VMACD",
    "monthly_return",
    "boll_down",
    "retained_earnings_per_share",
    "net_profit_growth_rate"
    ]
    print(f"Columns in scaled_df: {df.columns.tolist()}")
    selected_features = df[top_features]
    return selected_features

def merged_data(X, y):
    merged_df = pd.concat([X, y[['price_return_1M']]], axis=1)

    return merged_df

# def preprocessed_data(path='Input.csv'):
#     df = pd.read_csv(path)
#     reshaped = reshape_data(df)
#     cleaned = clean_data(reshaped)
#     target_df = target_column(cleaned)
#     scaled_df, factor_cols = scaled_feature(target_df)
#     selected_df = engineering_feature(scaled_df)
#     merge_df = merged_data(selected_df, target_df)
    
#     return target_df, scaled_df, factor_cols, merge_df




