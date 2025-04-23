from data_pipeline import reshape_data, clean_data, target_column, scaled_feature, engineering_feature
import pandas as pd

def preprocessed_data(path='Input.csv'):
    df = pd.read_csv(path)
    reshaped = reshape_data(df)
    cleaned = clean_data(reshaped)
    target_df = target_column(cleaned)
    scaled_df, factor_cols = scaled_feature(target_df)
    selected_df = engineering_feature(scaled_df)
    
    return target_df, scaled_df, factor_cols, selected_df