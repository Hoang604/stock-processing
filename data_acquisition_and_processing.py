import pandas as pd
import numpy as np
import datetime
import vnstock
from time import sleep 
import vnstock
import time
import os
import webbrowser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dateutil.relativedelta import relativedelta
from icecream import ic


def get_all_subdirectories(dir):
    """Trả về danh sách tên của tất cả các thư mục con trong thư mục được chỉ định.

    Args:
        dir: Đường dẫn đến thư mục cần lấy danh sách thư mục con.

    Returns:
        Một list chứa tên của tất cả các thư mục con.
    """
    try:
        return [ten for ten in os.listdir(dir) if os.path.isdir(os.path.join(dir, ten))]
    except FileNotFoundError:
        print(f"Error: Not found '{dir}'")
        return []

def run_all_html_files(subdir_path):
    """Chạy tất cả các file .html trong thư mục con được chỉ định."""
    try:
        for filename in os.listdir(subdir_path):
            if filename.endswith(".html"):
                filepath = os.path.abspath(os.path.join(subdir_path, filename)) # Make path absolute
                webbrowser.open_new_tab(f"file://{filepath}")
    except FileNotFoundError:
        print(f"Thư mục '{subdir_path}' không tồn tại.")


def run_specific_html_file(subdir_path, filename="revenue_profit_and_expense.html"):
    """Chạy một file .html cụ thể trong thư mục con được chỉ định.
       Mặc định là file revenue_profit_and_expense.html.
    """
    filepath = os.path.abspath(os.path.join(subdir_path, filename))
    try:
        webbrowser.open_new_tab(f"file://{filepath}")
    except FileNotFoundError:
        print(f"File '{filepath}' không tồn tại.")


def get_company_overview(symbol):
    try:
        df = vnstock.Company(source='tcbs',  symbol=symbol).overview()
        return df
    except:
        return pd.DataFrame()

def get_income_statement(symbol, period='quarter'):
    """
    Retrieve income statement data for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for
    period : str, optional
        Data period - 'quarter' or 'annual' (default: 'quarter')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing income statement data with datetime index,
        or empty DataFrame if an error occurs
    """
    try:
        df = vnstock.Finance(source='tcbs', symbol=symbol).income_statement(period=period)
        if df.empty:
            return pd.DataFrame()
            
        # Determine frequency based on period
        freq = 'Q' if period == 'quarter' else 'Y'
        
        # Convert to proper datetime index and clean
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error occurred when fetching {symbol} income statement: {e}")
        return pd.DataFrame()


def get_balance_sheet(symbol, period='quarter'):
    """
    Retrieve balance sheet data for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for
    period : str, optional
        Data period - 'quarter' or 'annual' (default: 'quarter')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing balance sheet data with datetime index,
        or empty DataFrame if an error occurs
    """
    try:
        df = vnstock.Finance(source='tcbs', symbol=symbol).balance_sheet(period=period)
        if df.empty:
            return pd.DataFrame()
            
        # Determine frequency based on period
        freq = 'Q' if period == 'quarter' else 'Y'
        
        # Convert to proper datetime index and clean
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error occurred when fetching {symbol} balance sheet: {e}")
        return pd.DataFrame()

def get_cash_flow(symbol, period='quarter'):
    """
    Retrieve cash flow statement data for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for
    period : str, optional
        Data period - 'quarter' or 'annual' (default: 'quarter')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing cash flow data with datetime index,
        or empty DataFrame if an error occurs
    """
    try:
        df = vnstock.Finance(source='tcbs', symbol=symbol).cash_flow(period=period)
        if df.empty:
            return pd.DataFrame()
            
        # Determine frequency based on period
        freq = 'Q' if period == 'quarter' else 'Y'
        
        # Convert to proper datetime index and clean
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp(freq)
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error occurred when fetching {symbol} cash flow statement: {e}")
        return pd.DataFrame()

def recalculate_cash_flow(df):
    """Chuẩn hóa tên gọi của các cột trong cash flow"""
    df.rename(columns={'invest_cost': 'capEx',
                       'from_invest': 'investing_cash_flow',
                       'from_financial': 'financing_cash_flow',
                       'from_sale': 'operating_cash_flow'}, inplace=True)
    df.fillna(0.0, inplace=True)
    df['free_cash_flow'] = df['operating_cash_flow'] + df['capEx']
    df['change_in_cash'] = (df['operating_cash_flow']
                            + df['investing_cash_flow']
                            + df['financing_cash_flow'])
    return df

def ttm(df):
    """
    Tính TTM cho tất cả các cột số trong DataFrame
    
    Parameters:
    df (pandas.DataFrame): DataFrame chứa dữ liệu tài chính theo quý
    
    Returns:
    pandas.DataFrame: DataFrame với các cột TTM mới
    """    
    # Lọc ra các cột số
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Tính TTM cho mỗi cột số
    for column in numeric_columns:
        df[column] = df[column].rolling(window=4, min_periods=4).sum()
        
    return df

def get_stock_price(symbol, start, end=str(datetime.date.today())):
    """Trả về giá cổ phiếu của symbol theo ngày"""
    try:
        df = vnstock.Vnstock(source="VCI", show_log=False).stock(symbol=symbol, source='VCI').quote.history(start=start, end=end)
        return df
    except Exception as e:
        print(f"Error when fetch stock price {symbol}: {e}")
        return pd.DataFrame()
    
def last_day_in_quarter(year, quarter):
    if quarter == 1:
        return pd.to_datetime(f'{year}-03-31')
    elif quarter == 2:
        return pd.to_datetime(f'{year}-06-30')
    elif quarter == 3:
        return pd.to_datetime(f'{year}-09-30')
    elif quarter == 4:
        return pd.to_datetime(f'{year}-12-31')


def get_company_ratio(stock_code):
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
        sleep(0.5)  # Rate limiting to avoid API throttling
        ratio_df = vnstock.Finance(source='tcbs', symbol=stock_code).ratio()
        
        if ratio_df.empty:
            return pd.DataFrame()
        
        # Convert year/quarter columns to proper datetime index
        ratio_df.index = _create_quarter_end_datetime(ratio_df['year'], ratio_df['quarter'])
        
        # Clean up the dataframe
        ratio_df = ratio_df.drop(['quarter', 'year'], axis=1)
        ratio_df = _clean_financial_dataframe(ratio_df)
        
        return ratio_df
    except Exception as e:
        print(f"Error when fetching company ratio for {stock_code}: {e}")
        return pd.DataFrame()


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
    # Mapping quarters to their end months
    quarter_end_months = {'1': '03', '2': '06', '3': '09', '4': '12'}
    
    # Create date strings and convert to datetime
    date_strings = (years.astype(str) + '-' + 
                   quarters.astype(str).map(quarter_end_months) + 
                   '-01')
    
    return pd.to_datetime(date_strings) + pd.tseries.offsets.QuarterEnd(0)


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

import pickle

def get_industry_ratio(symbols):
    """
    Tổng hợp danh sách các chỉ số theo từng năm của từng công ty một trong tập
    hợp một nhóm công ty

    Parameter:
    ----------
    symbols: list
        Danh sách các mã cổ phiếu cần tổng hợp chỉ số

    Return:
    -------
    dictionary
        Một từ điển có key là các mã cổ phiếu, và value là DataFrame tương ứng
        chứa thông tin về các chỉ số của mã cổ phiếu đó
    """
    print(symbols)
    icb_code = get_ticker_icb_code(symbols[0])
    filename = f'{icb_code}_dict_ratio.pkl'
    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)
    # Nếu có sẵn file thì chỉ cần tải vào
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            try:
                return pickle.load(f)
            except (EOFError, pickle.UnpicklingError): # Handle potential errors during loading
                print(f"Error reading from {filename}. Recalculating ratios...")
                pass # Continue to recalculate if there's an issue reading the file
    # Nếu không có sẵn file thì lấy dữ liệu từ vnstock, sau đó lưu vào file                
    dic = {}
    for stock_code in symbols:
        print(stock_code)
        temp = get_company_ratio(stock_code)
        if temp.empty:
            continue
        dic[stock_code] = temp
    # Save the dictionary to a pickle file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, 'wb') as f:
        pickle.dump(dic, f)

    return dic

def get_ticker_icb_code(ticker):
    """Trả về icb_code3 của ticker"""
    try:
        temp = vnstock.Vnstock(show_log=False).stock(symbol=ticker)
        all_tickers = temp.listing.symbols_by_industries()
    except Exception as e:
        print(f"Error occur when get {ticker} icb code: {e}")
        return 0
    all_tickers['icb_code3'] = all_tickers['icb_code3'].astype(int)
    try:
        ticker_icb_code = all_tickers.loc[all_tickers['symbol'] == ticker, 'icb_code3'].values[0]
    except Exception:
        raise ValueError(f"Không có dữ liệu cho mã {ticker}")
    return ticker_icb_code

def get_industry_tickers(icb_code):
    """
    Lấy danh sách các mã cổ phiếu trong cùng ngành, xác định bởi icb code

    Return:
    -------
    List chứa danh sách mã cổ phiếu, cùng với mã ngành icb_code3 của cổ phiếu đó
    """
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


def quarterly_company_market_capital(stock_symbol):
    """
    Tính toán vốn hóa thị trường tại thời điểm cuối quý của công ty trong từng
    quý, từ 10 năm trở lại đây

    Parameter:
    ----------
    stock_symbol: string
        Mã cổ phiếu cần tính toán

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa vốn hóa theo từng quý của mã cổ phiếu
    """
    start_day = str(datetime.datetime((datetime.date.today() - relativedelta(years=10)).year,1,1).date())
    time.sleep(0.5)
    stock_price_df = get_stock_price(stock_symbol, start_day)
    if stock_price_df.empty:
        return stock_price_df
    stock_price_df.index = pd.to_datetime(stock_price_df.index)
    stock_price_df = stock_price_df.set_index('time')
    stock_price_df = stock_price_df.resample('QE').last()
    time.sleep(0.5)
    try: 
        df = vnstock.Company(stock_symbol).overview()
        outstanding_share = df['outstanding_share'].iloc[0]
    except Exception:
        outstanding_share = 0
    stock_price_df[f"{stock_symbol}_capital"] = float(outstanding_share) * stock_price_df['close']
    return stock_price_df[f"{stock_symbol}_capital"]


def all_company_quarterly_market_capital(stock_symbols):
    """
    Tính vốn hóa thị trường tại thời điểm cuối quý của tập hợp một nhóm công ty
    nhất định

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa các mã cổ phiếu và vốn hóa thị trường của chúng theo từng
        quý
    """
    df = pd.DataFrame()
    for stock_code in stock_symbols:
        temp = quarterly_company_market_capital(stock_code)
        if temp.empty:
            df[f'{stock_code}_capital'] = 0
            print(f"{stock_code} raise error")
            continue
        print(f"Done {stock_code}")
        if df.empty or len(df) > len(temp):
            df = pd.concat([df, temp], axis=1)
        else:
            df = pd.concat([temp, df], axis=1)
        df.fillna(0.000000000000001, inplace=True)
    return df

def quarterly_industry_market_capital(stock_symbols):
    """
    Tính tổng vốn hóa thị trường của một nhóm công ty theo từng quý

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa tổng vốn hóa thị trường của từng công ty và tổng của chúng theo từng quý
    """
    df = all_company_quarterly_market_capital(stock_symbols)
    temp = pd.DataFrame({'market_capital': df.sum(axis=1)})
    df = pd.concat([df, temp], axis=1)
    df.index = pd.to_datetime(df.index)
    return df

def quarterly_all_company_market_cap_weight(stock_symbols):
    """
    Tính tỉ lệ vốn hóa thị trường của từng công ty trong một nhóm công ty nhất định

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính toán

    Return:
    pandas.DataFrame
        DataFrame chứa tỉ lệ vốn hóa thị trường của từng công ty theo từng quý
    """
    icb_code = get_ticker_icb_code(stock_symbols[0])
    filepath = os.path.join(os.path.dirname(__file__), 'data', f'{icb_code}_industry_market_cap_weight.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = quarterly_industry_market_capital(stock_symbols)
    last = df.iloc[:, -1]
    df = df.iloc[:, :-1].div(last, axis=0)
    df.to_csv(filepath)
    return df

def get_company_market_cap_weight_in_specific_quarter(capital_weights_df, year, quarter):
    """
    Lấy trọng số vốn hóa của các công ty tại thời điểm cuối quý

    Parameters:
    -----------
    capital_weights_df : pandas.DataFrame
        DataFrame chứa tỷ trọng vốn hóa theo thời gian
    year : int
        năm cần lấy dữ liệu
    quarter: int
        quý cần lấy dữ liệu

    Returns:
    --------
    pandas.Series
        Series chứa trọng số vốn hóa của các công ty tại quý chỉ định
    """
    time = last_day_in_quarter(year, quarter)
    if time < capital_weights_df.index[0]:
        return pd.Series()
    return capital_weights_df.loc[time]


def calculate_weighted_average_of_a_specific_metric(metric, year, quarter, tickers, quarter_weights, financial_metrics_dfs):
    """
    Tính giá trị trung bình có trọng số của một chỉ số tài chính cụ thể trong quý chỉ định của một ngành.
    Nếu metric là 'price_to_earning', chỉ tính trung bình có trọng số của các giá trị nhỏ hơn 70.

    Parameters:
    -----------
    year : int
        Năm cần tính.
    quarter: int
        Quý cần tính.
    metric : str
        Tên chỉ số cần tính.
    tickers : list
        Danh sách mã chứng khoán.
    quarter_weights : pandas.Series
        Trọng số của từng mã chứng khoán trong quý.
    financial_metrics_dfs : dict
        Từ điển chứa DataFrame dữ liệu tài chính cho mỗi mã chứng khoán.


    Returns:
    --------
    float
        Giá trị trung bình có trọng số của chỉ số.
    """
    date_index = last_day_in_quarter(year, quarter)
    date_index = pd.Timestamp(last_day_in_quarter(year, quarter))  # Convert to Timestamp
    weighted_sum = 0
    total_weight = 0
    for ticker in tickers:
        weight = quarter_weights[f'{ticker}_capital']
        if weight >= 0:
            try:
                ticker_ratio_value = financial_metrics_dfs[ticker].loc[date_index, metric]
                if pd.notna(ticker_ratio_value) and not np.isinf(ticker_ratio_value):
                    if metric == 'price_to_earning' and ticker_ratio_value >= 70:
                        print(f"{ticker}: pe = {ticker_ratio_value}")
                        continue  # Skip values >= 70 for P/E
                    weighted_sum += weight * ticker_ratio_value
                    total_weight += weight
            except (KeyError, ValueError):
                continue

    return weighted_sum if total_weight > 0 else np.nan



def calculate_quarter_averages(year, quarter, tickers, capital_weights_df, financial_metrics_dfs, metrics):
    """
    Tính tất cả các chỉ số trung bình của thị trường cho một quý trong năm
    
    Parameters:
    -----------
    year : int
        Năm cần tính
    quarter: int
        quý cần tính
    metrics : list
        Danh sách các chỉ số cần tính
    icb_code_3: icb code của ngành cần tính
        
    Returns:
    --------
    dict
        Dictionary chứa giá trị các chỉ số trong năm
    """
    quarter_average = {}
    quarter_weights = get_company_market_cap_weight_in_specific_quarter(capital_weights_df, year, quarter)
    if quarter_weights.empty:
        return quarter_average
    for metric in metrics:
        quarter_average[metric] = calculate_weighted_average_of_a_specific_metric(metric, year, quarter, tickers, quarter_weights, financial_metrics_dfs)
    
    return quarter_average

def calculate_market_averages(icb_code_3):
    """
    Tính các chỉ số trung bình của thị trường dựa trên tỷ trọng vốn hóa của ngành cần tính
    
    Parameters:
    -----------
    icb_code_3: icb code của ngành cần tính

    Returns:
    --------
    pandas.DataFrame
        DataFrame chứa các chỉ số trung bình của thị trường theo từng năm
    """
    tickers = get_industry_tickers(icb_code_3)
    financial_metrics_dfs = get_industry_ratio(tickers)
    capital_weight_df = quarterly_all_company_market_cap_weight(tickers)
    years = capital_weight_df.index.year.unique()
    i = 0
    while(get_company_ratio(tickers[i]).empty):
        i += 1
        if i == len(tickers):
            break
    metrics = get_company_ratio(tickers[i]).select_dtypes(include=np.number).columns.tolist()
    # Khởi tạo DataFrame kết quả
    market_averages = pd.DataFrame(columns=metrics)
    
    # Tính giá trị cho từng quý
    for year in years:
        for quarter in range (1, 5):
            current_date = datetime.datetime.today()
            current_year = current_date.year
            current_quarter = (current_date.month - 1) // 3 + 1
            if year == current_year and quarter > current_quarter:
                break
            quarter_average = calculate_quarter_averages(year, quarter, tickers, capital_weight_df, financial_metrics_dfs, metrics)
            if quarter_average == {}:
                continue
            date_index = last_day_in_quarter(year, quarter)
            market_averages.loc[date_index] = quarter_average
    
    return market_averages

def industry_average(ticker):
    """
    Tính toán chỉ số trung bình ngành của công ty được chỉ định

    Parameter:
    ----------
    ticker: string
        Mã cổ phiếu được chỉ định

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa thông tin về các chỉ số trung bình ngành
    """
    icb_code_3 = get_ticker_icb_code(ticker)
    df = calculate_market_averages(icb_code_3)
    df.fillna(0.000000000000001, inplace=True)
    data_dir = os.path.join(os.path.dirname(__file__), 'data') # Go up two levels
    os.makedirs(data_dir, exist_ok=True)  # Create if doesn't exist
    file_name = os.path.join(data_dir, f'{icb_code_3}_industry_average.csv')
    df.to_csv(file_name, index=True)
    return df

def read_industry_average_ratio(ticker):
    """
    Đọc dữ liệu trung bình ngành từ file CSV nếu file tồn tại
    Nếu file không tồn tại thì tính toán dữ liệu trung bình ngành
    
    Parameters:
    ticker (str): Mã chứng khoán cần tra cứu
    
    Returns:
    pd.DataFrame: DataFrame chứa dữ liệu trung bình ngành
    """
    icb_code = get_ticker_icb_code(ticker)
    print(icb_code)
    data_dir = os.path.join(os.path.dirname(__file__), 'data') # Go up two levels
    file_path = os.path.join(data_dir, f'{icb_code}_industry_average.csv')
    if os.path.isfile(file_path):
        df_read = pd.read_csv(file_path, index_col='Unnamed: 0')
        df_read.replace(0, 0.000000000000001, inplace=True)
        df_read.index = pd.to_datetime(df_read.index)
        return df_read
    else:
        return industry_average(ticker)
    
def get_ratio(symbol):
    """Trả về ticker ratio và industry ratio tương ứng"""
    ticker_ratio = get_company_ratio(symbol)
    industry_ratio = read_industry_average_ratio(symbol)
    return ticker_ratio, industry_ratio

def linear_regression_parameter(income_statement, outlier_threshold=3):  # Add threshold parameter
    """
    Calculates the coefficient and R-squared of a linear regression on 'operation_profit'.

    Args:
        income_statement (pd.DataFrame): DataFrame with 'operation_profit' column.
        outlier_threshold (float, optional): The threshold for outlier removal using Z-score. Defaults to 3.

    Returns:
        tuple: Coefficient and R-squared of the linear regression.
    """
    # Convert index to ordinal values for regression
    X = np.array(range(len(income_statement))).reshape(-1, 1)
    y = income_statement['revenue'].values.reshape(-1, 1)

    # Handle NaN values
    nan_mask = np.isnan(y)
    X = X[~nan_mask.flatten()]
    y = y[~nan_mask.flatten()]

    # Outlier Removal using Z-score
    z = np.abs((y - np.mean(y)) / np.std(y))
    X = X[z.flatten() < outlier_threshold]
    y = y[z.flatten() < outlier_threshold]


    # Rescale y (only if there are data points left after outlier removal)
    if len(y) > 0:
        max_y = np.max(y)
        if max_y != 0:  # Check for max_y = 0 to avoid ZeroDivisionError
            y = (y / max_y) * 1000
        else:
            print("The maximum 'operation_profit' is 0.  Cannot rescale.")

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

         # Get the coefficient and R-squared
        coefficient = model.coef_[0][0]
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        return coefficient, r_squared
    else:
        print("No data points left after outlier removal.")
        return np.nan, np.nan  # Return NaN values if no data left

def find_stock_code():
    industry_tickers = vnstock.Vnstock(show_log=False).stock('ACB', source='TCBS').listing.symbols_by_industries()
    industry_tickers = industry_tickers['symbol']
    bank_tickers = ['ABB', 'ACB', 'BAB', 'BID', 'BVB', 'CTG', 'EIB', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB',
                    'OCB', 'PGB', 'SGB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VBB', 'VCB', 'VIB', 'VPB']
    industry_tickers = [ticker for ticker in industry_tickers if ticker not in bank_tickers]
    tickers = []
    print('lenth:', len(industry_tickers))
    i = 1
    for tic in industry_tickers:
        print(f"{i}:{tic}")
        i+=1
        sleep(0.5)
        df = get_income_statement(tic).tail(20)
        if df.empty or len(df) < 20:
            continue
        try:
            c, r = linear_regression_parameter(df)
        except Exception:
            continue
        if c > 11:
            tickers.append(tic)
    return tickers

def predict_future_yeild(pe, grow_rate):
    table = {'year': [], 'yield': [], 'pe': [], 'grow_rate': []}
    x = grow_rate / 100 + 1
    for year in range(6):
        table['pe'].append(pe)
        table['grow_rate'].append(round(grow_rate, 2))
        table['year'].append(year + datetime.date.today().year)
        table['yield'].append(round(pow(x, year) / pe, 3) * 100)
    df = pd.DataFrame(data=table)
    return df

def predict_future_yeild_all(tickers):
    df = pd.DataFrame(columns=['ticker', 'pe', 'grow_rate', 'year', 'yield'])
    for ticker in tickers:
        grow_rate = get_income_statement(ticker, period='year').year_share_holder_income_growth.tail(3).mean() * 100
        pe = get_company_ratio(ticker).price_to_earning.iloc[-1]
        yeild_df = predict_future_yeild(pe, grow_rate).dropna()
        yeild_df['ticker'] = ticker
        df = pd.concat([df, yeild_df])
    df.set_index(['ticker', 'year'], inplace=True)
    df['2029_yield'] = df.xs(2029, level='year', drop_level=False)['yield']
    df = df.reset_index()  # Chuyển MultiIndex thành cột để dễ thao tác
    df['sort_key'] = df.groupby('ticker')['2029_yield'].transform('max')  # Tìm max yield năm 2029 cho mỗi ticker
    df = df.sort_values(by=['sort_key', 'ticker', 'year'], ascending=[False, True, True])  # Sắp xếp
    df = df.drop(columns=['2029_yield', 'sort_key']).set_index(['ticker', 'pe', 'grow_rate', 'year'])
    return df


if __name__ == '__main__':
    read_industry_average_ratio('FPT')


        