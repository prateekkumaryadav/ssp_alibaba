"""
Data Preprocessing for Alibaba Trace
Cleans and prepares data for analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple
import argparse
import os


class TracePreprocessor:
    """Preprocess trace data for analysis"""
    
    def __init__(self):
        self.data = None
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate trace records"""
        print(f"Original records: {len(df)}")
        df_clean = df.drop_duplicates()
        print(f"After removing duplicates: {len(df_clean)}")
        return df_clean
    
    def filter_failed_calls(self, df: pd.DataFrame, 
                           keep_failed: bool = True) -> pd.DataFrame:
        """
        Filter trace records based on call status
        
        Args:
            df: Input DataFrame
            keep_failed: If True, keep failed calls for analysis
            
        Returns:
            Filtered DataFrame
        """
        if 'status' in df.columns:
            if keep_failed:
                return df
            else:
                return df[df['status'] == 'success']
        return df
    
    def handle_outliers(self, df: pd.DataFrame, 
                       column: str = 'latency',
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if column not in df.columns:
            return df
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            outliers = (df[column] < lower) | (df[column] > upper)
            print(f"Found {outliers.sum()} outliers in {column}")
            
            # Option 1: Remove outliers
            # df_clean = df[~outliers]
            
            # Option 2: Cap outliers
            df_clean = df.copy()
            df_clean.loc[df_clean[column] < lower, column] = lower
            df_clean.loc[df_clean[column] > upper, column] = upper
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
            print(f"Found {outliers.sum()} outliers in {column}")
            df_clean = df[~outliers]
        
        return df_clean
    
    def aggregate_time_windows(self, df: pd.DataFrame,
                              window: str = '1min') -> pd.DataFrame:
        """
        Aggregate traces by time windows
        
        Args:
            df: Input DataFrame with timestamp column
            window: Aggregation window (e.g., '1min', '5min', '1H')
            
        Returns:
            Aggregated DataFrame
        """
        if 'timestamp' not in df.columns:
            return df
            
        df = df.set_index('timestamp')
        
        # Aggregate by time window
        agg_df = df.groupby([pd.Grouper(freq=window), 'service']).agg({
            'latency': ['mean', 'median', 'std', 'count'],
            'status': lambda x: (x == 'success').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        
        return agg_df
    
    def normalize_features(self, df: pd.DataFrame, 
                          columns: list = ['latency']) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df_norm[f'{col}_normalized'] = (df[col] - mean) / std
        
        return df_norm


def main():
    parser = argparse.ArgumentParser(description='Preprocess trace data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--remove-outliers', action='store_true', 
                       help='Remove outliers from latency')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Preprocess
    preprocessor = TracePreprocessor()
    df = preprocessor.remove_duplicates(df)
    
    if args.remove_outliers:
        df = preprocessor.handle_outliers(df, column='latency')
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == '__main__':
    main()
