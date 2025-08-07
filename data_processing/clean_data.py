import pandas as pd
import numpy as np

class StockDataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @staticmethod
    def clean_cash_flow_df(df):
        """
        Clean and standardize the cash flow DataFrame.
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing cash flow data with columns:
            'invest_cost', 'from_invest', 'from_financial', 'from_sale'
        
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame with standardized column names and filled NaN values.
        """
        df.rename(columns={'invest_cost': 'capEx',
                        'from_invest': 'investing_cash_flow',
                        'from_financial': 'financing_cash_flow',
                        'from_sale': 'operating_cash_flow'}, inplace=True)
        df.fillna(0.0, inplace=True)
        df['free_cash_flow'] = df['operating_cash_flow'] + df['capEx']
        return df
    
    @staticmethod
    def ttm(df):
        """
        Calculate Trailing Twelve Months (TTM) for financial data.
        
        This method computes the TTM values for all numeric columns in the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial data with numeric columns.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with TTM values calculated for each numeric column.
        -----------
        """    
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Calculate TTM for each numeric column
        for column in numeric_columns:
            df[column] = df[column].rolling(window=4, min_periods=4).sum()
            
        return df
