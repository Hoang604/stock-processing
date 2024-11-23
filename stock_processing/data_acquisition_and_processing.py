import pandas as pd
import numpy as np
import datetime
import vnstock3
from time import sleep 
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dateutil.relativedelta import relativedelta


def get_income_statement(symbol, period='quarter'):
    """Trả về income statement của symbol, mặc định là theo quý"""
    try:
        df = vnstock3.Finance(symbol).income_statement(period=period)
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp('Q')
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return pd.DataFrame()


def get_balance_sheet(symbol, period='quarter'):
    """Trả về balance sheet của symbol, mặc định là theo quý"""
    try:
        df = vnstock3.Finance(symbol).balance_sheet(period=period)
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp('Q')
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return pd.DataFrame()

def get_cash_flow(symbol, period='quarter'):
    """Trả về cash flow của symbol, mặc định là theo quý"""
    try:
        df = vnstock3.Finance(symbol).cash_flow(period=period)
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp('Q')
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return pd.DataFrame()

def recalculate_cash_flow(df):
    print(df)
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
        df = vnstock3.Vnstock(source="VCI", show_log=False).stock(symbol=symbol, source='VCI').quote.history(start=start, end=end)
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
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
    Trả về DataFrame chứa dữ liệu về ratio của stock code theo từng quý
    """
    try:
        sleep(1)
        ratio_df = vnstock3.Finance(stock_code).ratio()
        # Chỉnh sửa cột index lại thành kiểu datetime
        ratio_df.index = pd.to_datetime(ratio_df['year'].astype(str) + '-' +
                                        ratio_df['quarter'].astype(str).map({'1':'03','2':'06','3':'09','4':'12'}) + 
                                        '-01') + pd.tseries.offsets.QuarterEnd(0)

        # Loại bỏ các cột không cần thiết
        ratio_df = ratio_df.drop(['quarter', 'year'], axis=1)
        ratio_df.fillna(0.000000000000001, inplace=True)
        ratio_df.sort_index(ascending=True, inplace=True)
        ratio_df = ratio_df[~ratio_df.index.duplicated(keep="last")]
        return ratio_df
    except Exception as e:
        print(f"Error processing {stock_code}: {e}")
        return pd.DataFrame()

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
    # Nếu không có sẵn file thì lấy dữ liệu từ vnstock3, sau đó lưu vào file                
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
        temp = vnstock3.Vnstock(show_log=False).stock(ticker, source='TCBS')
        all_tickers = temp.listing.symbols_by_industries()
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return 0
    all_tickers['icb_code3'] = all_tickers['icb_code3'].astype(int)
    ticker_icb_code = all_tickers.loc[all_tickers['symbol'] == ticker, 'icb_code3'].values[0]
    return ticker_icb_code

def get_industry_tickers(icb_code):
    """
    Lấy danh sách các mã cổ phiếu trong cùng ngành, xác định bởi icb code

    Return:
    -------
    List chứa danh sách mã cổ phiếu, cùng với mã ngành icb_code3 của cổ phiếu đó
    """
    try:
        temp = vnstock3.Vnstock(show_log=False).stock('ACB', source='TCBS')
        industries_tickers = temp.listing.symbols_by_industries()
    except Exception as e:
        print(f"Error processing {icb_code}: {e}")
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
    time.sleep(1)
    stock_price_df = get_stock_price(stock_symbol, start_day)
    if stock_price_df.empty:
        return stock_price_df
    stock_price_df.index = pd.to_datetime(stock_price_df.index)
    stock_price_df = stock_price_df.set_index('time')
    stock_price_df = stock_price_df.resample('QE').last()
    time.sleep(1)
    try: 
        df = vnstock3.Company(stock_symbol).overview()
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

def coefficient_and_r_squared_of_operation_profit(income_statement):
    # Convert index to ordinal values for regression
    X = np.array(range(len(income_statement))).reshape(-1, 1)
    y = income_statement['operation_profit'].values.reshape(-1, 1)

    # Handle NaN values
    nan_mask = np.isnan(y)
    X = X[~nan_mask.flatten()]
    y = y[~nan_mask.flatten()]

    # Rescale y to a maximum of 1000
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
    r_squared = r2_score(y, model.predict(X))
    return coefficient, r_squared

def find_stock_code():
    industry_tickers = vnstock3.Vnstock(show_log=False).stock('ACB', source='TCBS').listing.symbols_by_industries()
    industry_tickers = industry_tickers['symbol']
    tickers = []
    print('lenth:', len(industry_tickers))
    i = 1
    for tic in industry_tickers:
        print(f"{i}:{tic}")
        i+=1
        sleep(1)
        df = get_income_statement(tic)
        if df.empty:
            continue
        try:
            c, r = coefficient_and_r_squared_of_operation_profit(df)
        except Exception:
            continue
        if c > 8 and r > 0.5:
            tickers.append(tic)
    return tickers

if __name__ == '__main__':
    tickers = find_stock_code()
    print(tickers)

        