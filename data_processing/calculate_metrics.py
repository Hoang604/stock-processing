import numpy as np
import pandas as pd
import datetime
import os
import time
from dateutil.relativedelta import relativedelta
from data_processing.fetch_raw_data import DataFetcher

def last_day_in_quarter(year, quarter):
    if quarter == 1:
        return pd.to_datetime(f'{year}-03-31')
    elif quarter == 2:
        return pd.to_datetime(f'{year}-06-30')
    elif quarter == 3:
        return pd.to_datetime(f'{year}-09-30')
    elif quarter == 4:
        return pd.to_datetime(f'{year}-12-31')

class StockDataProcessor:
    def __init__(self, stock_code):
        self.fetcher = DataFetcher(stock_code)
        self.stock_code = stock_code
        
        # Initialize all cache dictionaries consistently
        self._industry_tickers_cache = None
        self._industry_ratios_cache = None
        self._market_cap_cache = {}  # Initialize as dict instead of None
        self._icb_code_cache = None
        self._company_overviews_cache = {}
        
    def _get_industry_tickers(self):
        """Get industry tickers with caching"""
        if self._industry_tickers_cache is None:
            self._industry_tickers_cache = self.fetcher.fetch_industry_tickers()
        return self._industry_tickers_cache
    
    def _get_icb_code(self):
        """Get ICB code with caching"""
        if self._icb_code_cache is None:
            self._icb_code_cache = self.fetcher.fetch_ticker_icb_code()
        return self._icb_code_cache
    
    def _get_industry_ratios(self):
        """Get industry ratios with caching"""
        if self._industry_ratios_cache is None:
            self._industry_ratios_cache = self.fetcher.fetch_industry_ratio()
        return self._industry_ratios_cache
    
    def _get_company_overview_batch(self, tickers):
        """Get company overviews for multiple tickers with caching"""
        missing_tickers = [t for t in tickers if t not in self._company_overviews_cache]
        
        if missing_tickers:
            print(f"Fetching overviews for {len(missing_tickers)} companies...")
            for ticker in missing_tickers:
                try:
                    temp_fetcher = DataFetcher(ticker)
                    overview = temp_fetcher.fetch_company_overview()
                    self._company_overviews_cache[ticker] = overview
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching overview for {ticker}: {e}")
                    self._company_overviews_cache[ticker] = None
        
        return {t: self._company_overviews_cache.get(t) for t in tickers}

    def quarterly_company_market_capital(self):
        """
        Calculate the quarterly market capitalization of a specific stock symbol.

        Return:
        -------
        pandas.Series
            Series containing the quarterly market capitalization of the stock symbol
        """
        start_day = str(datetime.datetime((datetime.date.today() - relativedelta(years=10)).year,1,1).date())
        time.sleep(0.5)
        stock_price_df = self.fetcher.fetch_stock_price(start_day)
        if stock_price_df.empty:
            return pd.Series(dtype=float)
            
        # Fix index handling - check if 'time' column exists, otherwise use existing index
        if 'time' in stock_price_df.columns:
            stock_price_df = stock_price_df.set_index('time')
        stock_price_df.index = pd.to_datetime(stock_price_df.index)
        stock_price_df = stock_price_df.resample('QE').last()
        
        # Use cached company overview if available
        if self.stock_code in self._company_overviews_cache:
            overview_df = self._company_overviews_cache[self.stock_code]
        else:
            try: 
                overview_df = self.fetcher.fetch_company_overview()
                self._company_overviews_cache[self.stock_code] = overview_df
            except Exception as e:
                print(f"Error fetching overview for {self.stock_code}: {e}")
                overview_df = None
                
        try:
            outstanding_share = overview_df['outstanding_share'].iloc[0] if overview_df is not None else 0
        except (Exception, KeyError, IndexError):
            outstanding_share = 0
            
        stock_price_df[f"{self.stock_code}_capital"] = float(outstanding_share) * stock_price_df['close']
        return stock_price_df[f"{self.stock_code}_capital"]


    def all_company_quarterly_market_capital(self, tickers=None):
        """
        Calculate the quarterly market capitalization of all companies in a given list of stock symbols.
        Uses caching to avoid repeated API calls.

        Parameter:
        ----------
        tickers: list, optional
            List of stock symbols to calculate market capitalization for.
            If None, uses industry tickers from cache.

        Return:
        -------
        pandas.DataFrame
            DataFrame containing the quarterly market capitalization of each stock symbol.
        """
        if tickers is None:
            tickers = self._get_industry_tickers()
            
        if not tickers:
            print("No tickers provided")
            return pd.DataFrame()
        
        # Use cached market cap data if available
        cache_key = tuple(sorted(tickers))
        if cache_key in self._market_cap_cache:
            print("Using cached market cap data")
            return self._market_cap_cache[cache_key]
            
        print(f"Calculating market cap for {len(tickers)} companies...")
        
        # Batch fetch company overviews first to populate cache
        self._get_company_overview_batch(tickers)
        
        df = pd.DataFrame()
        
        for i, stock_code in enumerate(tickers):
            try:
                print(f"Processing {stock_code} ({i+1}/{len(tickers)})")
                
                # Temporarily update the current instance's stock_code and fetcher
                original_stock_code = self.stock_code
                original_fetcher = self.fetcher
                
                # Update for current ticker
                self.stock_code = stock_code
                self.fetcher = DataFetcher(stock_code)
                
                # Reuse the single company method
                market_cap_series = self.quarterly_company_market_capital()
                
                # Restore original values
                self.stock_code = original_stock_code
                self.fetcher = original_fetcher
                
                if market_cap_series.empty:
                    print(f"{stock_code} returned empty data")
                    continue
                
                print(f"Done {stock_code}")
                
                if df.empty:
                    df = market_cap_series.to_frame()
                else:
                    df = pd.concat([df, market_cap_series], axis=1)
                    
            except Exception as e:
                print(f"{stock_code} raised error: {e}")
                # Restore original values in case of error
                self.stock_code = original_stock_code
                self.fetcher = original_fetcher
                continue
                
        # Fill NaN values after processing all tickers
        if not df.empty:
            df.fillna(1e-15, inplace=True)
            # Cache the result
            self._market_cap_cache[cache_key] = df.copy()
        else:
            print("No successful market cap calculations")
            
        return df

    def quarterly_industry_market_capital(self, tickers=None):
        """
        Calculate the quarterly market capitalization of all companies in a specific industry.

        Parameter:
        ----------
        tickers: list, optional
            List of stock symbols in the industry. If None, uses cached industry tickers.

        Return:
        -------
        pandas.DataFrame
            DataFrame containing the quarterly market capitalization of each stock symbol in the industry.
        """
        if tickers is None:
            tickers = self._get_industry_tickers()
            
        companies_capital = self.all_company_quarterly_market_capital(tickers)
        
        if companies_capital.empty:
            print("No market capital data available")
            return pd.DataFrame()
            
        # Calculate total market capital
        market_capital = pd.DataFrame({'market_capital': companies_capital.sum(axis=1)})
        companies_capital = pd.concat([companies_capital, market_capital], axis=1)
        companies_capital.index = pd.to_datetime(companies_capital.index)
        return companies_capital

    def quarterly_all_company_market_cap_weight(self, tickers=None):
        """
        Calculate the market capitalization weight of each company in a specific group of companies.

        Parameter:
        ----------
        tickers: list, optional
            List of stock symbols to calculate weights for. If None, uses cached industry tickers.

        Return:
        pandas.DataFrame
            DataFrame containing the market capitalization weight of each stock symbol.
        """
        if tickers is None:
            tickers = self._get_industry_tickers()
            
        icb_code = self._get_icb_code()
            
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', f'{icb_code}_industry_market_cap_weight.csv')
        if os.path.exists(filepath):
            try:
                print("Using cached market cap weights from file")
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            except Exception as e:
                print(f"Error reading cached weights file: {e}")
                # Continue to recalculate
                
        df = self.quarterly_industry_market_capital(tickers)
        
        if df.empty:
            print("No industry market capital data available")
            return pd.DataFrame()
            
        # Calculate weights
        last = df.iloc[:, -1]  # Total market capital column
        if (last == 0).all():
            print("All market capital values are zero")
            return pd.DataFrame()
            
        df = df.iloc[:, :-1].div(last, axis=0)  # Exclude total column and divide by total
        
        try:
            df.to_csv(filepath)
            print(f"Saved market cap weights to {filepath}")
        except Exception as e:
            print(f"Warning: Could not save weights to file: {e}")
            
        return df

    def calculate_weighted_average_of_a_specific_metric(self, metric, year, quarter, quarter_weights, financial_metrics_dfs, tickers=None):
        """
        Calculate the weighted average of a specific financial metric for a given quarter.
        For 'price_to_earning' metric, only values less than 70 are included in the calculation.

        Parameters:
        -----------
        metric : str
            Name of the financial metric to calculate
        year : int
            Year to calculate for
        quarter: int
            Quarter to calculate for
        quarter_weights : pandas.Series
            Weights for each stock symbol in the quarter
        financial_metrics_dfs : dict
            Dictionary containing financial data DataFrames for each stock symbol
        tickers : list, optional
            List of tickers to process. If None, uses cached industry tickers.

        Returns:
        --------
        float
            Weighted average value of the metric
        """
        if tickers is None:
            tickers = self._get_industry_tickers()
            
        date_index = pd.Timestamp(last_day_in_quarter(year, quarter))
        weighted_sum = 0
        total_weight = 0
        
        for ticker in tickers:
            weight_column = f'{ticker}_capital'
            if weight_column not in quarter_weights:
                continue
                
            weight = quarter_weights[weight_column]
            if weight >= 0:
                try:
                    if ticker not in financial_metrics_dfs:
                        continue
                    
                    # Handle potential date misalignment with nearest date lookup
                    ticker_df = financial_metrics_dfs[ticker]
                    if date_index in ticker_df.index:
                        ticker_ratio_value = ticker_df.loc[date_index, metric]
                    else:
                        # Find nearest date if exact date doesn't exist
                        nearest_date = ticker_df.index[ticker_df.index <= date_index].max()
                        if pd.isna(nearest_date):
                            continue
                        ticker_ratio_value = ticker_df.loc[nearest_date, metric]
                        
                    if pd.notna(ticker_ratio_value) and not np.isinf(ticker_ratio_value):
                        if metric == 'price_to_earning' and ticker_ratio_value >= 70:
                            continue  # Skip values >= 70 for P/E
                        weighted_sum += weight * ticker_ratio_value
                        total_weight += weight
                except (KeyError, ValueError):
                    continue

        return weighted_sum / total_weight if total_weight > 0 else np.nan



    def calculate_quarter_averages(self, year, quarter, capital_weights_df, financial_metrics_dfs, metrics=None, tickers=None):
        """
        Calculate all market average financial metrics for a specific quarter.
        
        Parameters:
        -----------
        year : int
            Year to calculate for
        quarter: int
            Quarter to calculate for
        capital_weights_df : pandas.DataFrame
            DataFrame containing market cap weights
        financial_metrics_dfs : dict
            Dictionary containing financial metrics for each ticker
        metrics : list, optional
            List of metrics to calculate. If None, derives from financial_metrics_dfs
        tickers : list, optional
            List of stock tickers. If None, uses cached industry tickers
            
        Returns:
        --------
        dict
            Dictionary containing calculated metric values for the quarter
        """
        if tickers is None:
            tickers = self._get_industry_tickers()
            
        if metrics is None and financial_metrics_dfs:
            metrics = financial_metrics_dfs[list(financial_metrics_dfs.keys())[0]].columns.tolist()
        elif not metrics:
            return {}
            
        quarter_average = {}
        quarter_end_date = last_day_in_quarter(year, quarter)  # Renamed to avoid shadowing
        
        # Check if the date exists in the weights dataframe
        if capital_weights_df.empty or quarter_end_date < capital_weights_df.index[0]:
            return {}
            
        try:
            quarter_weights = capital_weights_df.loc[quarter_end_date]
        except KeyError:
            return {}
            
        if quarter_weights.empty:
            return quarter_average
            
        for metric in metrics:
            quarter_average[metric] = self.calculate_weighted_average_of_a_specific_metric(
                metric, year, quarter, quarter_weights, financial_metrics_dfs, tickers)

        return quarter_average

    def industry_average(self):
        """
        Calculate industry average financial metrics based on market capitalization weights.
        Uses extensive caching to minimize API calls.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing industry average metrics by quarter
        """
        icb_code_3 = self._get_icb_code()
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        file_name = os.path.join(data_dir, f'{icb_code_3}_industry_average.csv')

        if os.path.isfile(file_name):
            print("Using cached industry average data from file")
            df_read = pd.read_csv(file_name, index_col='Unnamed: 0')
            df_read.replace(0, 1e-15, inplace=True)
            df_read.index = pd.to_datetime(df_read.index)
            return df_read

        print("Calculating industry averages from scratch...")
        
        # Get all required data with caching
        tickers = self._get_industry_tickers()
        print(f"Processing {len(tickers)} tickers in industry")
        
        financial_metrics_dfs = self._get_industry_ratios()
        capital_weight_df = self.quarterly_all_company_market_cap_weight()
        
        if capital_weight_df.empty:
            print("No market cap weights available")
            return pd.DataFrame()
            
        years = capital_weight_df.index.year.unique()
        columns = financial_metrics_dfs[list(financial_metrics_dfs.keys())[0]].columns.tolist()
        market_averages = pd.DataFrame(columns=columns)

        current_date = datetime.datetime.today()
        current_year = current_date.year
        current_quarter = (current_date.month - 1) // 3 + 1
        
        # Check if financial_metrics_dfs is not empty before accessing
        if not financial_metrics_dfs:
            print("No financial metrics data available")
            return pd.DataFrame()
            
        # Get metrics from any available ticker's data
        sample_ticker = list(financial_metrics_dfs.keys())[0]
        metrics = financial_metrics_dfs[sample_ticker].columns.tolist()
        print(f"Calculating {len(metrics)} metrics for {len(years)} years")
        
        # Calculate values for each quarter
        quarters_processed = 0
        for year in years:
            for quarter in range(1, 5):
                if year == current_year and quarter > current_quarter:
                    break
                try:
                    quarter_average = self.calculate_quarter_averages(
                        year, quarter, capital_weight_df, financial_metrics_dfs, metrics, tickers)
                
                    if quarter_average:
                        date_index = last_day_in_quarter(year, quarter)
                        market_averages.loc[date_index] = quarter_average
                        quarters_processed += 1
                        
                        if quarters_processed % 10 == 0:
                            print(f"Processed {quarters_processed} quarters...")
                except Exception as e:
                    print(f"Error processing {year} Q{quarter}: {e}")
                    continue

        if market_averages.empty:
            print("No market averages calculated")
            return pd.DataFrame()
            
        market_averages.fillna(1e-15, inplace=True)
        os.makedirs(data_dir, exist_ok=True)
        market_averages.to_csv(file_name, index=True)
        print(f"Saved industry averages to {file_name}")
        return market_averages

    def get_ratio(self):
        """Trả về ticker ratio và industry ratio tương ứng"""
        print(f"Fetching ratios for {self.stock_code}...")
        ticker_ratio = self.fetcher.fetch_company_ratio()
        print(f"Fetching industry ratios...")
        industry_ratio = self.industry_average()
        return ticker_ratio, industry_ratio
    

