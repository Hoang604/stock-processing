from data_processing.fetch_raw_data import DataFetcher

fetcher = DataFetcher('HPG')

data = fetcher.fetch_company_balance_sheet()
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns in DataFrame
print(data)