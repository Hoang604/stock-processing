import vnstock
import pandas as pd
from time import sleep
import datetime
import os
import pickle

class DataFetcher:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.company = vnstock.Company(source='tcbs', symbol=stock_code)
        self.finance = vnstock.Finance(source='tcbs', symbol=stock_code)
        self.stock = vnstock.Vnstock(show_log=False).stock(source='vci', symbol=stock_code)

    def fetch_company_overview(self):
        try:
            return self.company.overview()
        except Exception as e:
            print(f"Error fetching company overview: {e}")
            return None

    def fetch_company_income_statement(self, period='quarter'):
        try:
            df = self.finance.income_statement(period=period)
            
            if df is None or df.empty:
                print(f"No income statement data returned for {self.stock_code}")
                return None
                
            df.sort_index(ascending=True, inplace=True)
            
            # Determine frequency based on period parameter
            freq = 'Q' if period == 'quarter' else 'Y' if period == 'annual' else 'Q'
            
            # Convert index to proper time series format
            try:
                df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
            except Exception as idx_error:
                print(f"Error converting index for {self.stock_code}: {idx_error}")
                # Try alternative approach if PeriodIndex fails
                if hasattr(df.index, 'to_timestamp'):
                    df.index = df.index.to_timestamp()
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Check for and handle duplicates with logging
            duplicates_count = df.index.duplicated().sum()
            if duplicates_count > 0:
                print(f"Warning: Found {duplicates_count} duplicate entries, keeping last occurrence")
            
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as e:
            print(f"Error fetching income statement for {self.stock_code}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def fetch_company_balance_sheet(self, period='quarter'):
        try:
            df = self.finance.balance_sheet(period=period)
            
            if df is None or df.empty:
                print(f"No balance sheet data returned for {self.stock_code}")
                return None
                
            df.sort_index(ascending=True, inplace=True)
            
            # Determine frequency based on period parameter
            freq = 'Q' if period == 'quarter' else 'Y' if period == 'annual' else 'Q'
            
            # Convert index to proper time series format
            try:
                df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
            except Exception as idx_error:
                print(f"Error converting index for {self.stock_code}: {idx_error}")
                # Try alternative approach if PeriodIndex fails
                if hasattr(df.index, 'to_timestamp'):
                    df.index = df.index.to_timestamp()
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Check for and handle duplicates with logging
            duplicates_count = df.index.duplicated().sum()
            if duplicates_count > 0:
                print(f"Warning: Found {duplicates_count} duplicate entries, keeping last occurrence")
            
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as e:
            print(f"Error fetching balance sheet for {self.stock_code}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def fetch_company_cash_flow(self, period='quarter'):
        try:
            df = self.finance.cash_flow(period=period)
            
            if df is None or df.empty:
                print(f"No cash flow data returned for {self.stock_code}")
                return None
                
            df.sort_index(ascending=True, inplace=True)

            # Determine frequency based on period parameter
            freq = 'Q' if period == 'quarter' else 'Y' if period == 'annual' else 'Q'
            
            # Convert index to proper time series format
            try:
                df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
            except Exception as idx_error:
                print(f"Error converting index for {self.stock_code}: {idx_error}")
                # Try alternative approach if PeriodIndex fails
                if hasattr(df.index, 'to_timestamp'):
                    df.index = df.index.to_timestamp()
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Check for and handle duplicates with logging
            duplicates_count = df.index.duplicated().sum()
            if duplicates_count > 0:
                print(f"Warning: Found {duplicates_count} duplicate entries, keeping last occurrence")
            
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as e:
            print(f"Error fetching cash flow for {self.stock_code}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def fetch_stock_price(self, start, end=None):
        if end is None:
            end = str(datetime.date.today())
        
        try:
            df = self.stock.quote.history(start=start, end=end)
            return df
        except Exception as e:
            print(f"Error fetching stock price: {e}")
            return pd.DataFrame()
        
    def fetch_company_ratio(self):
        """
        Retrieve quarterly financial ratios for a given stock code.
        
        Parameters:
        -----------
        stock_code : str
            The stock symbol to fetch ratios for
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing quarterly financial ratios with datetime index,
            or empty DataFrame if an error occurs
        """
        try:
            sleep(1)  # Rate limiting to avoid API throttling
            ratio_df = self.finance.ratio()
            
            if ratio_df is None or ratio_df.empty:
                print(f"No ratio data returned for {self.stock_code}")
                return pd.DataFrame()
            
            # Check if required columns exist
            if 'year' not in ratio_df.columns or 'quarter' not in ratio_df.columns:
                print(f"Missing year/quarter columns for {self.stock_code}. Available: {ratio_df.columns.tolist()}")
                return pd.DataFrame()
                        

            if ratio_df.empty:
                print(f"No valid ratio data after filtering for {self.stock_code}")
                return pd.DataFrame()
            
            # Convert year/quarter columns to proper datetime index
            ratio_df.index = self._create_quarter_end_datetime(ratio_df['year'], ratio_df['quarter'])
            
            # Clean up the dataframe
            ratio_df = ratio_df.drop(['quarter', 'year'], axis=1)
            ratio_df = self._clean_financial_dataframe(ratio_df)
            
            return ratio_df
        except Exception as e:
            print(f"Error when fetching company ratio for {self.stock_code}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    @staticmethod
    def _create_quarter_end_datetime(years, quarters):
        """
        Create quarter-end datetime index from year and quarter columns.
        
        Parameters:
        -----------
        years : pandas.Series
            Series containing years
        quarters : pandas.Series  
            Series containing quarters (1-4)
            
        Returns:
        --------
        pandas.DatetimeIndex
            DatetimeIndex with quarter-end dates
        """
        try:
            # Mapping quarters to their end months
            quarter_end_months = {1: '03', 2: '06', 3: '09', 4: '12', 
                                '1': '03', '2': '06', '3': '09', '4': '12'}
            
            # Clean the data first - convert to numeric and handle invalid values
            years_clean = pd.to_numeric(years, errors='coerce')
            quarters_clean = pd.to_numeric(quarters, errors='coerce')
            
            # Filter out invalid values
            valid_mask = (years_clean.notna() & quarters_clean.notna() & 
                         (years_clean > 1900) & (years_clean < 2100) &
                         (quarters_clean >= 1) & (quarters_clean <= 4))
            
            if not valid_mask.any():
                raise ValueError("No valid year/quarter combinations found")
            
            years_clean = years_clean[valid_mask].astype(int)
            quarters_clean = quarters_clean[valid_mask].astype(int)
            
            # Create date strings and convert to datetime
            date_strings = []
            for year, quarter in zip(years_clean, quarters_clean):
                month = quarter_end_months[quarter]
                date_strings.append(f"{year}-{month}-01")
            
            dates = pd.to_datetime(date_strings) + pd.tseries.offsets.QuarterEnd(0)
            
            # If we filtered some data, create a full index with NaT for invalid entries
            if len(dates) != len(years):
                full_dates = pd.Series(index=years.index, dtype='datetime64[ns]')
                full_dates[valid_mask] = dates
                return pd.DatetimeIndex(full_dates)
            
            return pd.DatetimeIndex(dates)
            
        except Exception as e:
            print(f"Error creating quarter end datetime: {e}")
            # Return a minimal valid index if all else fails
            return pd.DatetimeIndex([pd.Timestamp('2000-01-01')])

    @staticmethod
    def _clean_financial_dataframe(df):
        """
        Clean financial dataframe by handling missing values, sorting, and removing duplicates.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to clean
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame
        """
        # Fill NaN values with a very small number instead of 0 to avoid division issues
        df.fillna(1e-15, inplace=True)
        
        # Sort by date and remove duplicates
        df.sort_index(ascending=True, inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        
        return df

    def fetch_ticker_icb_code(self):
        """
        Fetch the ICB code for a given stock symbol.
        
        Parameters:
        -----------
        stock_code : str
            Stock symbol to fetch ICB code for
            
        Returns:
        --------
        int
            ICB code for the stock symbol
        """
        try:
            temp = vnstock.Vnstock(show_log=False).stock(symbol=self.stock_code)
            all_tickers = temp.listing.symbols_by_industries()
        except Exception as e:
            print(f"Error occur when get {self.stock_code} icb code: {e}")
            return 0
        all_tickers['icb_code3'] = all_tickers['icb_code3'].astype(int)
        try:
            ticker_icb_code = all_tickers.loc[all_tickers['symbol'] == self.stock_code, 'icb_code3'].values[0]
            return ticker_icb_code
        except Exception:
            raise ValueError(f"No data for {self.stock_code}")
        
    def fetch_industry_tickers(self):
        """
        Fetch all stock symbols in a specific industry based on ICB code.
        
        Parameters:
        -----------
        icb_code : int
            ICB code for the industry to fetch tickers for
            
        Returns:
        --------
        list
            List of stock symbols in the specified industry
        """
        icb_code = self.fetch_ticker_icb_code()
        try:
            temp = vnstock.Vnstock(show_log=False).stock('ACB', source='TCBS')
            industries_tickers = temp.listing.symbols_by_industries()
        except Exception as e:
            print(f"Error occur when fetch company has icb code {icb_code}: {e}")
            return []
        industries_tickers['icb_code3'] = industries_tickers['icb_code3'].astype(int)
        industry_tickers = industries_tickers.loc[industries_tickers['icb_code3'] == icb_code]
        industry_tickers = industry_tickers['symbol'].tolist()
        return industry_tickers

    def fetch_industry_ratio(self):
        """
        Fetch financial ratios for a list of stock symbols and save them to a pickle file.
        Then return the dictionary of ratios.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols to fetch ratios for
            
        Returns:
        --------
        dict
            Dictionary with stock symbols as keys and their financial ratios as values
        """
        symbols = self.fetch_industry_tickers()
        print(symbols)
        icb_code = self.fetch_ticker_icb_code()
        filename = f'{icb_code}_dict_ratio.pkl'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        # Check if the file already exists and load it
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                try:
                    return pickle.load(f)
                except (EOFError, pickle.UnpicklingError): # Handle potential errors during loading
                    print(f"Error reading from {filename}. Recalculating ratios...")
                    pass # Continue to recalculate if there's an issue reading the file
        # If file doesn't exist or loading failed, fetch the data          
        dic = {}
        for stock_code in symbols:
            print(stock_code)
            temp = DataFetcher(stock_code).fetch_company_ratio()
            if temp.empty:
                continue
            dic[stock_code] = temp
        # Save the dictionary to a pickle file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        with open(file_path, 'wb') as f:
            pickle.dump(dic, f)

        return dic
