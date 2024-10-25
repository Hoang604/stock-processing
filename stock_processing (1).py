#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import vnstock3
from matplotlib import pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import time


# In[2]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)
symbol = 'MBB'


# In[3]:


company = vnstock3.Company(symbol)
finance = vnstock3.Finance(symbol)


# In[4]:


balance_sheet = finance.balance_sheet(period='quarter')
balance_sheet = balance_sheet.sort_index(ascending=True)


# In[5]:


income_statement = finance.income_statement(period='quarter')
income_statement = income_statement.sort_index(ascending=True)


# In[6]:


def profit_on_revenue_plot(symbol):
    """
    Vẽ và lưu đồ thị doanh thu theo lợi nhuận

    Parameter:
    ----------
    symbol: string
        Mã cổ phiếu cần vẽ đồ thị
    """
    finance = vnstock3.Finance(symbol)
    income_statement = finance.income_statement(period='quarter')
    income_statement = income_statement.sort_index(ascending=True)

    # Tên gọi của cùng một chỉ số trong bảng income statement của các ngành khác nhau là khác nhau
    # Đoạn này là để chuẩn hóa tên gọi để có thể dùng một hàm cho tất cả 
    if 'gross_profit' in income_statement.columns:
        income_statement.rename(columns={'operation_profit': 'operation_income'}, inplace=True)
    
    if 'provision_expense' in income_statement.columns:
        income_statement.drop('revenue', axis=1, inplace=True)
        income_statement.rename(columns={'operation_profit': 'revenue'}, inplace=True)

    # Bắt đầu vẽ đồ thị
    ax = income_statement['revenue'].plot(kind='bar', figsize=(10, 5),
                                        label='Doanh thu theo quý',
                                        color='lightgreen')
    income_statement['operation_income'].plot(kind='line', ax=ax,
                                            label='Lợi nhuận theo quý từ hoạt'
                                            + ' động kinh doanh',
                                            color='purple', secondary_y=True)
    income_statement['post_tax_profit'].plot(kind='line', ax=ax,
                                            label='Lợi nhuận sau thuế',
                                            color='red',
                                            secondary_y=True)
    ax.set_ylabel('Doanh thu')
    ax.right_ax.set_ylabel('Lợi nhuận')
    ax.legend(loc='upper left')
    ax.right_ax.legend(loc='upper right')
    ax.set_xticks(range(0, len(income_statement['revenue']), 4))
    ax.set_xticklabels(income_statement.index[::4], rotation=30)
    plt.title('Doanh thu và lợi nhuận')
    file_name = 'doanh_thu_va_loi_nhuan_sau_thue.png'
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[7]:


def expense_on_profit_plot(symbol):
   """
   Vẽ và lưu đồ thị biểu diễn mối quan hệ giữa chi phí hoạt động và lợi nhuận hoạt động của doanh nghiệp.
   
   Đồ thị bao gồm:
   - Tỷ lệ chi phí hoạt động trên lợi nhuận hoạt động (trục y bên trái)
   - Chi phí hoạt động (trục y bên phải)
   - Lợi nhuận hoạt động (trục y bên phải)
   - Lợi nhuận hoạt động thuần (trục y bên phải)
   
   Parameters:
   ----------
   symbol: string
       Mã cổ phiếu cần vẽ đồ thị
       
   Returns:
   -------
   None - Đồ thị được lưu vào thư mục /Pictures/{symbol}/tong_chi_tren_tong_thu.png
   """
   # Lấy dữ liệu từ BCTC
   finance = vnstock3.Finance(symbol)
   income_statement = finance.income_statement(period='quarter')
   income_statement = income_statement.sort_index(ascending=True)
   
   # Chuẩn hóa tên cột 
   if 'gross_profit' in income_statement.columns:
       income_statement.rename(columns={
           'operation_profit': 'operation_income',
           'gross_profit': 'operation_profit'
       }, inplace=True)
   
   # Tính tỷ lệ chi phí/lợi nhuận, xử lý các trường hợp đặc biệt
   income_statement['operation_expense_on_operation_profit'] = np.where(
       income_statement['operation_profit'] != 0,
       (-income_statement['operation_expense']) / income_statement['operation_profit'],
       0  # Gán 0 cho các trường hợp operation_profit = 0
   )
   
   # Loại bỏ các giá trị vô cùng
   income_statement['operation_expense_on_operation_profit'] = income_statement['operation_expense_on_operation_profit'].replace(
       [np.inf, -np.inf], np.nan
   )
   
   # Vẽ đồ thị cột cho tỷ lệ chi phí/lợi nhuận
   ax = income_statement['operation_expense_on_operation_profit'].plot(
       kind='bar',
       figsize=(10, 5),
       label='operation expense/operation income ratio',
       color='lightgreen'
   )
   
   # Vẽ các đường cho chi phí và lợi nhuận
   (-income_statement['operation_expense']).plot(
       color='purple',
       secondary_y=True,
       label='operation expense'
   )
   income_statement['operation_profit'].plot(
       label='operation income',
       ax=ax,
       color='red',
       secondary_y=True
   )
   income_statement['operation_income'].plot(
       label='net operation income',
       ax=ax,
       color='black',
       secondary_y=True
   )
   
   # Thiết lập nhãn trục y
   ax.set_ylabel('Tỉ lệ giữa doanh thu và chi tiêu')
   ax.right_ax.set_ylabel('Tỉ đồng')
   
   # Tính và thiết lập giới hạn cho trục y trái
   data_min = income_statement['operation_expense_on_operation_profit'].min()
   data_max = income_statement['operation_expense_on_operation_profit'].max()
   
   # Xử lý trường hợp giá trị không hợp lệ
   if np.isnan(data_min) or np.isinf(data_min):
       data_min = 0
   if np.isnan(data_max) or np.isinf(data_max):
       data_max = 1
       
   # Thêm padding 5% của range dữ liệu
   data_range = data_max - data_min
   padding = data_range * 0.05
   
   # Tính toán và thiết lập giới hạn với padding
   ymin = max(-10, data_min - padding)  # Giới hạn dưới không quá -10
   ymax = min(10, data_max + padding)   # Giới hạn trên không quá 10
   
   # Đảm bảo ymin < ymax
   if ymin >= ymax:
       ymin = 0
       ymax = 1
       
   ax.set_ylim(ymin, ymax)
   
   # Thiết lập trục x
   ax.set_xticks(range(0, len(income_statement['revenue']), 4))
   ax.set_xticklabels(income_statement.index[::4], rotation=30)
   
   # Thiết lập legend và tiêu đề
   ax.legend(loc='upper left')
   ax.right_ax.legend(loc='upper right')
   plt.title("Tổng chi trên tổng thu")
   
   # Lưu đồ thị
   file_name = 'tong_chi_tren_tong_thu.png'
   dir = f"/home/hoang/Pictures/{symbol}"
   if not os.path.exists(dir):
       os.makedirs(dir)
   plt.savefig(os.path.join(dir, file_name))
   plt.show()


# In[8]:


def company_market_capital(stock_symbol):
    """
    Tính toán vốn hóa thị trường tại thời điểm cuối năm của công ty trong từng
    năm

    Parameter:
    ----------
    stock_symbol: string
        Mã cổ phiếu cần tính toán

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa vốn hóa theo từng năm của mã cổ phiếu
    """
    company = vnstock3.Company(stock_symbol)
    instance = vnstock3.Vnstock(source="VCI", show_log=False).stock(symbol=stock_symbol, source='VCI')
    start_day = str(datetime.datetime((datetime.date.today() - relativedelta(years=10)).year,1,1).date())
    time.sleep(1)
    stock = instance.quote.history(start=start_day, end=str(datetime.date.today()))
    stock = stock.set_index('time')
    stock = stock.resample('YE').last()
    stock = stock.drop(['high', 'low', 'open', 'volume'], axis=1)
    time.sleep(1)
    outstanding_share = company.overview()['outstanding_share'].iloc[0]
    stock[f"{stock_symbol}_capital"] = float(outstanding_share) * stock['close']
    stock = stock.drop('close', axis=1)
    return stock


# In[9]:


def yearly_market_capital(stock_symbols):
    """
    Tính vốn hóa thị trường tại thời điểm cuối năm của tập hợp một nhóm công ty
    nhất định

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa các mã cổ phiếu và vốn hóa thị trường của chúng theo từng
        năm
    """
    df = pd.DataFrame()
    for stock_code in stock_symbols:
        temp = company_market_capital(stock_code)
        df = pd.concat([df, temp], axis=1)
        df.fillna(0.000000000000001, inplace=True)
    return df


# In[10]:


def yearly_total_market_capital(stock_symbols):
    """
    Tính tổng vốn hóa thị trường của một nhóm công ty theo từng năm

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính

    Return:
    -------
    pandas.DataFrame
        DataFrame chứa tổng vốn hóa thị trường của nhóm công ty theo từng năm
    """
    df = yearly_market_capital(stock_symbols)
    temp = pd.DataFrame({'market_capital': df.sum(axis=1)})
    df = pd.concat([df, temp], axis=1)
    return df


# In[11]:


def yearly_company_maket_share(stock_symbols):
    """
    Tính tỉ lệ vốn hóa thị trường của từng công ty trong một nhóm công ty nhất định

    Parameter:
    ----------
    stock_symbols: list
        Danh sách các mã cổ phiếu cần tính toán

    Return:
    pandas.DataFrame
        DataFrame chứa tỉ lệ vốn hóa thị trường của từng công ty theo từng năm
    """
    df = yearly_total_market_capital(stock_symbols)
    last = df.iloc[:, -1]
    df = df.iloc[:, :-1].div(last, axis=0)
    return df


# In[12]:


def get_year_end_weights(capital_weights_df, year):
    """
    Lấy trọng số vốn hóa của các công ty tại thời điểm cuối năm

    Parameters:
    -----------
    capital_weights_df : pandas.DataFrame
        DataFrame chứa tỷ trọng vốn hóa theo thời gian
    year : int
        Năm cần lấy dữ liệu

    Returns:
    --------
    pandas.Series
        Series chứa trọng số vốn hóa của các công ty tại thời điểm cuối năm
    """
    year_end = capital_weights_df[capital_weights_df.index.year == year].index[-1]
    return capital_weights_df.loc[year_end]


# In[13]:


def calculate_weighted_metric(year_weights, financial_metrics_dfs, year, metric, tickers):
    """
    Tính giá trị trung bình có trọng số của một chỉ số tài chính cụ thể trong năm chỉ định

    Parameters:
    -----------
    year_weights : pandas.Series
        Series chứa trọng số vốn hóa của các công ty
    financial_metrics_dfs : dict
        Dictionary chứa DataFrame các chỉ số tài chính của từng công ty
    year : int
        Năm cần tính
    metric : str
        Tên chỉ số cần tính    df = pd.concat([df, last], axis=1)

    tickers : list
        Danh sách mã cổ phiếu

    Returns:
    --------
    float
        Giá trị trung bình có trọng số của chỉ số
    """
    weighted_sum = 0
    total_weight = 0

    for ticker in tickers:
        weight = year_weights[f'{ticker}_capital']

        if weight >= 0:
            try:
                ticker_value = financial_metrics_dfs[ticker].loc[str(year), metric]

                if pd.notna(ticker_value) and not np.isinf(ticker_value):
                    weighted_sum += weight * ticker_value
                    total_weight += weight
            except (KeyError, ValueError):
                continue
    
    return weighted_sum / total_weight if total_weight > 0 else np.nan


# In[14]:


def calculate_year_averages(year, capital_weights_df, financial_metrics_dfs, metrics, tickers):
    """
    Tính tất cả các chỉ số trung bình của thị trường cho một năm
    
    Parameters:
    -----------
    year : int
        Năm cần tính
    capital_weights_df : pandas.DataFrame
        DataFrame chứa tỷ trọng vốn hóa theo thời gian
    financial_metrics_dfs : dict
        Dictionary chứa DataFrame các chỉ số tài chính của từng công ty
    metrics : list
        Danh sách các chỉ số cần tính
    tickers : list
        Danh sách mã cổ phiếu
        
    Returns:
    --------
    dict
        Dictionary chứa giá trị các chỉ số trong năm
    """
    year_weights = get_year_end_weights(capital_weights_df, year)
    year_averages = {}
    
    for metric in metrics:
        year_averages[metric] = calculate_weighted_metric(
            year_weights, financial_metrics_dfs, year, metric, tickers
        )
    
    return year_averages


# In[15]:


def calculate_market_averages(symbols, financial_metrics_dfs):
    """
    Tính các chỉ số trung bình của thị trường dựa trên tỷ trọng vốn hóa
    
    Parameters:
    -----------
    symbols : list
        Danh sách các mã cổ phiếu
    financial_metrics_dfs : dict
        Dictionary chứa DataFrame các chỉ số tài chính của từng công ty

    Returns:
    --------
    pandas.DataFrame
        DataFrame chứa các chỉ số trung bình của thị trường theo từng năm
    """
    capital_weights_df = yearly_company_maket_share(symbols)
    years = capital_weights_df.index.year.unique()
    metrics = financial_metrics_dfs[symbols[0]].columns.tolist()
    
    # Khởi tạo DataFrame kết quả
    market_averages = pd.DataFrame(index=years, columns=metrics)
    
    # Tính giá trị cho từng năm
    for year in years:
        year_averages = calculate_year_averages(
            year, capital_weights_df, financial_metrics_dfs, metrics, symbols
        )
        market_averages.loc[year] = year_averages
    
    return market_averages


# In[16]:


def get_ticker_icb_code(ticker):
    """
    Trả về icb_code3 của ticker
    """
    temp = vnstock3.Vnstock(show_log=False).stock(ticker, source='TCBS')
    temp = temp.listing.symbols_by_industries()
    temp['icb_code3'] = temp['icb_code3'].astype(int)
    ticker_icb_code = temp.loc[temp['symbol'] == ticker, 'icb_code3'].values[0]
    return ticker_icb_code


# In[17]:


def get_industry_tickers(ticker):
    """
    Lấy danh sách các mã cổ phiếu cùng nhóm ngành với một công ty được chỉ định

    Parameter:
    ----------
    ticker: string
        Mã cổ phiếu của công ty được chỉ định

    Return:
    -------
    List chứa danh sách mã cổ phiếu, cùng với mã ngành icb_code3 của cổ phiếu đó
    """
    temp = vnstock3.Vnstock(show_log=False).stock(ticker, source='TCBS')
    temp = temp.listing.symbols_by_industries()
    temp['icb_code3'] = temp['icb_code3'].astype(int)
    ticker_icb_code = get_ticker_icb_code(ticker)
    temp = temp[temp['icb_code3'] == ticker_icb_code]
    temp = temp['symbol'].tolist()
    for ticker in ['HPT', 'PMJ', 'BCB', 'AAH', 'KCB', 'D17', 'FRM', 'FRC',
                   'VBG', 'HMG', 'LMC', 'NHV', 'TQN', 'VDB', 'VDT', 'VIM',
                   'TMW', 'VGL', 'MGC', 'DFC', 'YBM', 'BIG', 'CK8', 'DCH',
                   'HU6', 'HD8', 'HRB', 'MA1', 'HD2', 'PWA', 'TBR', 'TAL',
                   'DSE', 'ANT', 'C22', 'CNA', 'BHK', 'BHP', 'CBS', 'BLT',
                   'BHG', 'CMN', 'BSD', 'EPC', 'NSS', 'CAT', 'HLB', 'BBM',
                   'HAV', 'HDS', 'PRO', 'PSL', 'APT', 'SBB', 'SPV', 'SNC',
                   'SPH', 'QHW', 'SKN', 'TAN', 'TJC', 'FCC', 'SB1', 'TCJ',
                   'BIO', 'BCP', 'DPP', 'NTF', 'AMP', 'TW3', 'CNC', 'DHN',
                   'DPH', 'HDP', 'MEF', 'MRF', 'MTP', 'NDP', 'NDC', 'DTH',
                   'YTC', 'AGX', 'CPH', 'HFX', 'DKC', 'FHN', 'PNG', 'TV6',
                   'BTD', 'CIP', 'TS3', 'ACS', 'DCR', 'VMK', 'C12', 'DC1',
                   'TSA', 'HFB', 'BCR', 'DND', 'ICC', 'BTN', 'CDR', 'CCC',
                   'DVW', 'ICN', 'CHC', 'ACE', 'CI5', 'CT3', 'CH5', 'H11',
                   'GH3', 'HCI', 'HEJ', 'HMS', 'TVH', 'HPP', 'ICI', 'ING',
                   'CNN', 'BMN', 'LCC', 'L12', 'HC1', 'LAI', 'LG9', 'NAC',
                   'VCE', 'L45', 'HAM', 'NXT', 'PCC', 'PLE', 'PNT', 'QNT',
                   'RCD', 'SDX', 'PTO', 'PX1', 'SIG', 'XLV', 'TA6', 'TBT',
                   'TEL', 'TNM', 'TRT', 'GND', 'TVA', 'TVG', 'UMC', 'CGV',
                   'USC', 'VIH', 'CCV', 'X77', 'BCO', 'XMD', 'VW3', 'CEG',
                   'BMF', 'BGE', 'BWA', 'HIO', 'DMS', 'TDB', 'DVC', 'HFC',
                   'LKW', 'NLS', 'NNT', 'NTW', 'TOW', 'PJS', 'PND', 'GCB',
                   'KTW', 'POB', 'TAW', 'TBW', 'NSL', 'THW', 'TQW', 'UPC',
                   'VXT', 'VPW', 'NVP', 'VWS', 'ECO', 'AVG', 'DOC', 'GER',
                   'HPH', 'LNC', 'HNP', 'BT1', 'RBC', 'SEP', 'NSG', 'SIV',
                   'PVO', 'VXP', 'ODE', 'IN4', 'IBD', 'NBE', 'VNX', 'VPR',
                   'EPH', 'HNB', 'DNL', 'DOP', 'TAB', 'DDH', 'HHN', 'TUG',
                   'ISG', 'NAS', 'PLO', 'PTX', 'PTT', 'QSP', 'QNP', 'RAT',
                   'TSG', 'STS', 'TBH', 'TNP', 'TR1', 'TRS', 'VMT', 'WTC',
                   'VSE', 'DAS', 'FTI', 'VMA', 'DTB', 'BMD', 'UDL', 'DUS',
                   'MBN', 'BRS', 'MTV', 'BTU', 'DNE', 'CAR', 'CDH', 'CFM',
                   'HEC', 'HEP', 'HSA', 'MDA', 'MLC', 'MPY', 'MQB', 'MTL',
                   'NAU', 'MND', 'MTX', 'NUE', 'MTH', 'MQN', 'SDV', 'QNU',
                   'MTB', 'THU', 'TVM', 'UCT', 'USD', 'VTK', 'VLP', 'VCT',
                   'MND', 'BCV', 'DNT', 'BLN', 'DLX', 'MTC', 'NWT', 'ONW',
                   'HES', 'TPS', 'TSD', 'VIR', 'VTM', 'CMK', 'APL', 'CKA',
                   'CE1', 'FSO', 'FBC', 'FT1', 'IME', 'LMI', 'LQN', 'PEC',
                   'L63', 'SAL', 'SCO', 'A32', 'UEM', 'BAL', 'BBH', 'TKA',
                   'TB8', 'FBA', 'PEQ', 'PTP', 'DCG', 'BMG', 'THM', 'AG1',
                   'MGG', 'BVN', 'VDG', 'HCB', 'HLT', 'LGM', 'NJC', 'PTG',
                   'SPB', 'SSF', 'GTD', 'TLI', 'TTG', 'VDN', 'VTI', 'X26',
                   'HLO', 'EME', 'EMG', 'HLS', 'NEM', 'TGP', 'MFS', 'DXL',
                   'BHI']:
        if ticker in temp:
            temp.remove(ticker)
    print(len(temp))
    return temp, ticker_icb_code


# In[18]:


def get_company_ratio(stock_code):
    """
    Trả về DataFrame chứa dữ liệu về ratio của stock code theo từng năm
    """
    time.sleep(1)
    print(stock_code)
    temp = vnstock3.Finance(stock_code).ratio()
    temp.index = pd.to_datetime(temp['year'].astype(str) + '-' +
                                temp['quarter'].astype(str).map({'1':'03','2':'06','3':'09','4':'12'}) + 
                                '-01')
    temp.index = pd.to_datetime(temp.index)
    temp = temp.resample('YE').last()
    temp.index = temp['year']
    temp = temp.drop(['quarter', 'year'], axis=1)
    temp.fillna(0.000000000000001, inplace=True)
    
    return temp


# In[19]:


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
    dic = {}
    for stock_code in symbols:
        temp = get_company_ratio(stock_code)
        dic[stock_code] = temp
    return dic


# In[20]:


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
    symbols, icb_code = get_industry_tickers(ticker)
    ratio_dict = get_industry_ratio(symbols)
    df = calculate_market_averages(symbols, ratio_dict)
    df.fillna(0.000000000000001, inplace=True)
    file_name = f'{icb_code}_industry_average.csv'
    df.to_csv(file_name, index=True)
    return df


# In[21]:


def read_average_industry(ticker):
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
    file_path = f'{icb_code}_industry_average.csv'
    if os.path.isfile(file_path):
        df_read = pd.read_csv(file_path, index_col='time')
        df_read.replace(0, 0.000000000000001, inplace=True)
        return df_read
    else:
        return industry_average(ticker)


# In[22]:


def get_ratio(symbol):
    """Trả về ticker ratio và industry ratio tương ứng"""
    ticker_ratio = get_company_ratio(symbol)
    ticker_ratio.index = pd.to_datetime(ticker_ratio.index)
    industry_ratio = read_average_industry(symbol)
    industry_ratio.index = pd.to_datetime(industry_ratio.index.astype(str) +
                                          '-01-01')
    return ticker_ratio, industry_ratio


# In[23]:


def company_vs_industry_roe(symbol, ticker_ratio, industry_ratio):
    """
    Vẽ đồ thị hai đường ROE của toàn ngành và của công ty

    Parameters:
    -----------
    ticker_ratio: DataFrame
        Chứa chỉ số của công ty
    industry_ratio: DataFrame
        Chứa chỉ số trung bình ngành
    """
    ticker_ratio_aligned, industry_ratio_aligned = ticker_ratio.align(industry_ratio, join='inner', axis=0)
    # Vẽ đồ thị ROE
    ax = industry_ratio_aligned['roe'].plot(label='Industry ROE',
                                            color='purple', figsize=(10,5))
    ticker_ratio_aligned['roe'].plot(label='Company ROE', ax=ax, color='red')
    ax.set_ylabel('ROE')
    ax.legend(loc='upper right')
    # Tạo thư mục nếu chưa tồn tại
    file_name = "ROE_cua_cong_ty_va_toan_nganh.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Lưu hình ảnh và hiển thị
    plt.title(f"ROE của {symbol} và toàn ngành")
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[24]:


def company_vs_industry_roa(symbol, ticker_ratio, industry_ratio):
    """
    Vẽ đồ thị hai đường ROA của toàn ngành và của công ty

    Parameters:
    -----------
    ticker_ratio: DataFrame
        Chứa chỉ số của công ty
    industry_ratio: DataFrame
        Chứa chỉ số trung bình ngành
    """
    ticker_ratio_aligned, industry_ratio_aligned = ticker_ratio.align(industry_ratio, join='inner', axis=0)
    # Vẽ đồ thị ROA
    ax = industry_ratio_aligned['roa'].plot(label='Industry ROA',
                                            color='purple', figsize=(10, 5))
    ticker_ratio_aligned['roa'].plot(label='Company ROA', ax=ax, color='red')
    ax.set_ylabel('ROE')
    ax.legend(loc='upper right')
    # Tạo thư mục nếu chưa tồn tại
    file_name = "ROA_cua_cong_ty_va_toan_nganh.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Lưu hình ảnh và hiển thị
    plt.title(f"ROA của {symbol} và toàn ngành")
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[25]:


def company_roe_vs_roa(symbol):
    """
    Vẽ đồ thị hai đường ROE và ROA của công ty và tỷ lệ giữa chúng

    Parameters:
    -----------
    ticker_ratio: DataFrame
        Chứa chỉ số của công ty
    industry_ratio: DataFrame
        Chứa chỉ số trung bình ngành
    """
    
    ticker_ratio = vnstock3.Finance(symbol).ratio().sort_index(ascending=True)
    ticker_ratio['ratio'] = ticker_ratio['roe'] / ticker_ratio['roa']
    ax = ticker_ratio['ratio'].plot(kind='bar', label='company ROE/ROA ratio',
                                    color='lightblue', figsize=(10, 5))
    ticker_ratio['roe'].plot(label='company ROE', secondary_y=True,
                             ax=ax, color='purple')
    ticker_ratio['roa'].plot(label='Company ROA', ax=ax, color='red',
                             secondary_y=True)
    ax.set_ylable = "Tỉ số ROE / ROA"
    ax.legend(loc='upper left')
    ax.right_ax.legend(loc='upper right')
    ax.set_xticks(range(0, len(ticker_ratio), 4))
    ax.set_xticklabels(ticker_ratio.index[::4], rotation=30)
    # Tạo thư mục nếu chưa tồn tại
    file_name = "ROE_vs_ROA.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Lưu hình ảnh và hiển thị
    plt.title(f"ROA và ROE của {symbol}")
    plt.savefig(os.path.join(dir, file_name))
    plt.legend(loc='best')
    plt.show()


# In[26]:


def company_vs_industry_pe(symbol, ticker_ratio, industry_ratio):
    # Align data
    ticker_ratio_aligned, industry_ratio_aligned = ticker_ratio.align(industry_ratio, join='inner', axis=0)
    # Tính tỉ số PE công ty/ngành
    pe_ratio = ticker_ratio_aligned['price_to_earning'] / industry_ratio_aligned['price_to_earning']
    # Tạo figure và axis chính
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Vẽ tỉ số PE dưới dạng bar chart
    # Chuyển index datetime thành số để vẽ bar
    x = range(len(pe_ratio))
    bars = ax1.bar(x, pe_ratio.values, width=0.8, alpha=1, color='lightgreen',
                   label='Company/Industry PE Ratio')
    # Đặt lại các mốc thời gian trên trục x
    ax1.set_xticks(x)
    ax1.set_xticklabels(pe_ratio.index.strftime('%Y-%m-%d'), rotation=45)
    # Thêm đường tham chiếu y=1
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Company/Industry PE Ratio')
    ax1.tick_params(axis='y')
    # Tạo trục y phụ cho PE values
    ax2 = ax1.twinx()
    # Vẽ đè PE công ty và ngành lên
    ax2.plot(x, industry_ratio_aligned['price_to_earning'].values,
             label='Industry PE', color='purple', linewidth=2)
    ax2.plot(x, ticker_ratio_aligned['price_to_earning'].values,
             label='Company PE', color='red', linewidth=2)
    ax2.set_ylabel('PE Ratio')
    ax2.tick_params(axis='y')
    data_min = min(ticker_ratio_aligned['price_to_earning'].min(), industry_ratio_aligned['price_to_earning'].min())
    data_max = max(ticker_ratio_aligned['price_to_earning'].max(), industry_ratio_aligned['price_to_earning'].max())
    # Thêm padding 5% của range dữ liệu
    data_range = data_max - data_min
    padding = data_range * 0.05

    # Tính toán giới hạn với padding
    ymin = max(data_min - padding, -10)
    ymax = min(50, data_max + padding)  # Giới hạn trên vẫn là 10
    ax2.set_ylim(ymin, ymax)
    # Kết hợp legends từ cả hai trục
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    # Thêm grid cho dễ đọc
    ax1.grid(True, alpha=0.3)
    # Tạo thư mục nếu chưa tồn tại
    file_name = "PE_cua_cong_ty_va_toan_nganh.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Đặt tiêu đề và lưu hình
    plt.title(f"PE của {symbol} và toàn ngành")
    plt.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[27]:


def company_vs_industry_pb(symbol, ticker_ratio, industry_ratio):
    # Align data
    ticker_ratio_aligned, industry_ratio_aligned = ticker_ratio.align(industry_ratio, join='inner', axis=0)

    # Tính tỉ số PE công ty/ngành
    pe_ratio = ticker_ratio_aligned['price_to_book'] / industry_ratio_aligned['price_to_book']

    # Tạo figure và axis chính
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(pe_ratio))
    bars = ax1.bar(x, pe_ratio.values, width=0.8, alpha=1, color='lightgreen',
                   label='Company/Industry PB Ratio')

    # Đặt lại các mốc thời gian trên trục x
    ax1.set_xticks(x)
    ax1.set_xticklabels(pe_ratio.index.strftime('%Y-%m-%d'), rotation=45)

    # Thêm đường tham chiếu y=1
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax1.set_ylabel('Company/Industry PB Ratio')
    ax1.tick_params(axis='y')

    # Tạo trục y phụ cho PE values
    ax2 = ax1.twinx()

    # Vẽ đè PE công ty và ngành lên
    ax2.plot(x, industry_ratio_aligned['price_to_book'].values,
             label='Industry PB', color='purple', linewidth=2)
    ax2.plot(x, ticker_ratio_aligned['price_to_book'].values,
             label='Company PB', color='red', linewidth=2)
    ax2.set_ylabel('PB Ratio')
    ax2.tick_params(axis='y')

    # Kết hợp legends từ cả hai trục
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Thêm grid cho dễ đọc
    ax1.grid(True, alpha=0.3)

    # Tạo thư mục nếu chưa tồn tại
    file_name = "PB_cua_cong_ty_va_toan_nganh.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Đặt tiêu đề và lưu hình
    plt.title(f"PB của {symbol} và toàn ngành")
    plt.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[28]:


def equity_on_debt(symbol, ticker_ratio, industry_ratio):
        # Align data
    ticker_ratio_aligned, industry_ratio_aligned = ticker_ratio.align(industry_ratio, join='inner', axis=0)

    # Tính tỉ số PE công ty/ngành
    if 'debt_on_equity' in ticker_ratio.columns:
        ticker_ratio_aligned.rename(columns={'debt_on_equity': 'equity_on_loan'}, inplace=True)
        industry_ratio_aligned.rename(columns={'debt_on_equity': 'equity_on_loan'}, inplace=True)
    pe_ratio = ticker_ratio_aligned['equity_on_loan'] / industry_ratio_aligned['equity_on_loan']

    # Tạo figure và axis chính
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Vẽ tỉ số PE dưới dạng bar chart
    # Chuyển index datetime thành số để vẽ bar
    x = range(len(pe_ratio))
    bars = ax1.bar(x, pe_ratio.values, width=0.8, alpha=1, color='lightgreen',
                   label='Company/Industry equity/debt Ratio')

    # Đặt lại các mốc thời gian trên trục x
    ax1.set_xticks(x)
    ax1.set_xticklabels(pe_ratio.index.strftime('%Y-%m-%d'), rotation=45)

    # Thêm đường tham chiếu y=1
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax1.set_ylabel('Company/Industry equity/debt Ratio')
    ax1.tick_params(axis='y')

    # Tạo trục y phụ cho PE values
    ax2 = ax1.twinx()

    # Vẽ đè PE công ty và ngành lên
    ax2.plot(x, industry_ratio_aligned['equity_on_loan'].values,
             label='Industry equity/debt', color='purple', linewidth=2)
    ax2.plot(x, ticker_ratio_aligned['equity_on_loan'].values,
             label='Company equity/debt', color='red', linewidth=2)
    ax2.set_ylabel('equity/debt Ratio')
    ax2.tick_params(axis='y')

    # Kết hợp legends từ cả hai trục
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Thêm grid cho dễ đọc
    ax1.grid(True, alpha=0.3)

    # Tạo thư mục nếu chưa tồn tại
    file_name = "equity_on_debt_cua_cong_ty_va_toan_nganh.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Đặt tiêu đề và lưu hình
    plt.title(f"equity/debt của {symbol} và toàn ngành")
    plt.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[29]:


def balance_sheet_processing_for_normal_company(symbol):
    finance = vnstock3.Finance(symbol)
    balance_sheet = finance.balance_sheet(period='quarter')
    balance_sheet.sort_index(ascending=True, inplace=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    balance_sheet['short_asset_on_debt'] = balance_sheet['short_asset'] / balance_sheet['short_debt']
    balance_sheet['asset_on_debt'] = balance_sheet['asset'] / balance_sheet['debt']

    # Vẽ biểu đồ ngắn hạn
    ax1.bar(balance_sheet.index, balance_sheet['short_asset_on_debt'],
            color='lightgreen', label='Tài sản ngắn hạn trên nợ ngắn hạn')
    ax1.set_ylabel('Tỉ số tài sản/nợ')
    ax1.plot(balance_sheet.index, [1] * len(balance_sheet), linestyle='--',
             color='black', alpha=0.5, label='Đường giới hạn')
    ax12 = ax1.twinx()
    ax12.plot(balance_sheet.index, balance_sheet['short_asset'],
              label='Tài sản ngắn hạn')
    ax12.plot(balance_sheet.index, balance_sheet['short_debt'],
              label='Nợ ngắn hạn')
    ax12.set_ylabel('Tỉ đồng')
    ax1.legend(loc='upper left')
    ax12.legend(loc='upper right')
    ax1.set_xticks(range(0, len(balance_sheet), 4))
    ax1.set_xticklabels(balance_sheet.index[::4], rotation=30)
    ax1.set_title('Ngắn hạn')

    # Vẽ biểu đồ dài hạn
    ax2.bar(balance_sheet.index, balance_sheet['asset_on_debt'],
            color='lightgreen', label='Tổng tài sản trên tổng nợ')
    ax2.set_ylabel('Tỉ số tài sản/nợ')
    ax2.plot(balance_sheet.index, [1] * len(balance_sheet), linestyle='--',
             color='black', alpha=0.5, label='Đường giới hạn')
    ax22 = ax2.twinx()
    ax22.plot(balance_sheet.index, balance_sheet['asset'],
              label='Tổng tài sản')
    ax22.plot(balance_sheet.index, balance_sheet['debt'],
              label='Tổng nợ')
    ax22.plot(balance_sheet.index, balance_sheet['equity'],
              label='Vốn chủ sở hữu', color='red')
    ax22.set_ylabel('Tỉ đồng')
    ax2.legend(loc='upper left')
    ax22.legend(loc='upper right')
    ax2.set_xticks(range(0, len(balance_sheet), 4))
    ax2.set_xticklabels(balance_sheet.index[::4], rotation=30)
    ax2.set_title('Dài hạn')

    file_name = "phan_tich_balance_sheet.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Đặt tiêu đề và lưu hình
    fig.suptitle(f"Balance sheet của {symbol}")
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[30]:


def balance_sheet_for_bank(symbol):
    finance = vnstock3.Finance(symbol)
    balance_sheet = finance.balance_sheet(period='quarter')
    balance_sheet.sort_index(ascending=True, inplace=True)
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    balance_sheet['asset_on_debt'] = balance_sheet['asset'] / balance_sheet['debt']
    ax2.bar(balance_sheet.index, balance_sheet['asset_on_debt'],
            color='lightgreen', label='Tổng tài sản trên tổng nợ')
    ax2.set_ylabel('Tỉ số tài sản/nợ')
    ax2.plot(balance_sheet.index, [1] * len(balance_sheet), linestyle='--',
             color='black', alpha=0.5, label='Đường giới hạn')
    ax22 = ax2.twinx()
    ax22.plot(balance_sheet.index, balance_sheet['asset'],
              label='Tổng tài sản')
    ax22.plot(balance_sheet.index, balance_sheet['debt'],
              label='Tổng nợ')
    ax22.plot(balance_sheet.index, balance_sheet['equity'],
              label='Vốn chủ sở hữu', color='red')
    ax22.set_ylabel('Tỉ đồng')
    ax2.legend(loc='upper left')
    ax22.legend(loc='upper right')
    ax2.set_xticks(range(0, len(balance_sheet), 4))
    ax2.set_xticklabels(balance_sheet.index[::4], rotation=30)

    file_name = "phan_tich_balance_sheet.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Đặt tiêu đề và lưu hình
    fig.suptitle(f"Balance sheet của {symbol}")
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[31]:


def cash_flow_processing(symbol):
    finance = vnstock3.Finance(symbol)
    cash_flow = finance.cash_flow(period='quarter')
    cash_flow.sort_index(ascending=True, inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(cash_flow.index, cash_flow['from_sale'],
            label='Tiền thu được từ hoạt động kinh doanh')
    ax.plot(cash_flow.index, cash_flow['from_financial'],
            label='Dòng tiền từ hoạt động tài chính')
    ax.plot(cash_flow.index, cash_flow['from_invest'],
            label='Dòng tiền từ hoạt động đầu tư')
    ax.plot(cash_flow.index, -cash_flow['invest_cost'],
            label='Tiền được dùng để tái đầu tư')
    ax.plot(cash_flow.index, [1] * len(cash_flow), color='black',
            alpha=0.5, label='Đường bằng 0', linestyle='--')
    ax.legend()
    ax.set_ylabel('Tỷ đồng')
    ax.set_xticks(range(0, len(cash_flow), 4))
    ax.set_xticklabels(cash_flow.index[::4], rotation=30)
    file_name = "phan_tich_cash_flow.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Đặt tiêu đề và lưu hình
    fig.suptitle(f"Cash flow của {symbol}")
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()


# In[32]:


def find_closest_index_linear(sorted_list, target):
    closest_index = -1
    min_diff = float('inf')
    
    for i in range(len(sorted_list)):
        diff = abs(sorted_list[i] - target)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
            
    return closest_index


# In[33]:


def plot_price(symbol):
    finance = vnstock3.Finance(symbol)
    ratio = finance.ratio()
    ratio.sort_index(ascending=True, inplace=True)
    x = ratio['price_to_earning']

    stock = vnstock3.Vnstock(source='TCBS', show_log=False).stock(symbol=symbol, source='TCBS')
    df = stock.quote.history(start='2015-01-01', end=str(datetime.date.today()))
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Step 1: Create 'quarter' column in df
    df['quarter'] = df.index.to_period('Q')

    # Step 2: Convert Series x to DataFrame for merging
    income_df = x.reset_index()
    income_df.columns = ['quarter', 'earning_per_share']
    income_df['quarter'] = pd.PeriodIndex(income_df['quarter'], freq='Q')

    # Step 3: Merge df with income_df based on 'quarter', keep original index
    df = df.merge(income_df, on='quarter', how='left').set_index(df.index)
    df.drop('quarter', axis=1, inplace=True)

    df = df.ffill()
    df['close'] = df['close'] * 1000
    df['earning_per_share'] = df['close'] / df['earning_per_share']
    data = []

    for i in range(1, 30):
        data.append((df['earning_per_share'] * i < df['close']).sum())

    data = [x / len(df) for x in data]
    factor = find_closest_index_linear(data, 0.5) + 1
    df['fair_value'] = df['earning_per_share'] * factor
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(df.index, df['close'], color='green', alpha=0.8, label='Market price')
    ax.plot(df.index, df['fair_value'], color='red', alpha=0.8, label='Fair price')
    ax.legend()
    ax.set_title(f'Market price and fair price of {symbol}')
    file_name = "market_price_vs_fair_price.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Đặt tiêu đề và lưu hình
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()

    plt.show()


# In[34]:


def plot_price2(symbol):

    #  get PE
    finance = vnstock3.Finance(symbol)
    ratio = finance.ratio()
    ratio.sort_index(ascending=True, inplace=True)
    x = ratio['price_to_earning']

    #  get share_holder_grow_rate
    income_statement = finance.income_statement(period='quarter')
    income_statement.sort_index(ascending=True, inplace=True)
    # Tính grow_rate theo phần trăm so với cùng kỳ năm ngoái
    income_statement['grow_rate'] = (
        income_statement['share_holder_income']
        .pct_change(periods=1) + 1
    )
    income_statement['y'] = income_statement['grow_rate'].rolling(window=20).apply(lambda x: np.prod(x))
    income_statement['yy'] = (pow(income_statement['y'], 1/4) - 1) * 100
    y = (pow(income_statement['y'], 1/4) - 1) * 100

    # get stock price
    stock = vnstock3.Vnstock(source='TCBS', show_log=False).stock(symbol=symbol, source='TCBS')
    df = stock.quote.history(start='2015-01-01', end=str(datetime.date.today()))
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Step 1: Create 'quarter' column in df
    df['quarter'] = df.index.to_period('Q')

    # merge x
    # Step 2: Convert Series x to DataFrame for merging
    income_df = x.reset_index()
    income_df.columns = ['quarter', 'price_to_earning']
    income_df['quarter'] = pd.PeriodIndex(income_df['quarter'], freq='Q')

    # Step 3: Merge df with income_df based on 'quarter', keep original index
    df = df.merge(income_df, on='quarter', how='left').set_index(df.index)

    # merge y
    income_df = y.reset_index()
    income_df.columns = ['quarter', 'grow_rate']
    income_df['quarter'] = pd.PeriodIndex(income_df['quarter'], freq='Q')

    # Step 3: Merge df with income_df based on 'quarter', keep original index
    df = df.merge(income_df, on='quarter', how='left').set_index(df.index)

    df.drop('quarter', axis=1, inplace=True)

    df = df.ffill()
    df['close'] = df['close'] * 1000
    df['earning_per_share'] = df['close'] / df['price_to_earning']
    df.loc[df['grow_rate'] > 25, 'grow_rate'] = 25
    df['fair_value'] = df['grow_rate'] * df['earning_per_share']
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(df.index, df['close'], color='green', alpha=0.8, label='Market price')
    ax.plot(df.index, df['fair_value'], color='red', alpha=0.8, label='Fair price')
    ax.legend()
    ax.set_title(f'Market price and fair price of {symbol}')
    file_name = "market_price_vs_fair_price_2.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Đặt tiêu đề và lưu hình
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()

    plt.show()


# In[35]:


def plot_price3(symbol):

    #  get PE
    finance = vnstock3.Finance(symbol)
    ratio = finance.ratio()
    ratio.sort_index(ascending=True, inplace=True)
    x = ratio['price_to_earning']

    #  get share_holder_grow_rate
    income_statement = finance.income_statement(period='quarter')
    income_statement.sort_index(ascending=True, inplace=True)
    # Tính grow_rate theo phần trăm so với cùng kỳ năm ngoái
    income_statement['TTM_year_share_holder_income_growth'] = (
        income_statement['share_holder_income']
        .rolling(window=4).sum()
    )

    # Tính grow_rate theo phần trăm so với cùng kỳ năm ngoái
    # Tạo cột mới cho giá trị trung bình 5 kỳ gần nhất cho cùng quý
    income_statement['TTM_5_year_avg_same_quarter'] = None  # Cột mới để lưu kết quả

    income_statement['grow_rate'] = (
        income_statement['TTM_year_share_holder_income_growth']
        .pct_change(periods=4) * 100  # chuyển thành phần trăm
    )
    # Thực hiện tính toán cho từng quý
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        # Lọc các hàng của từng quý
        quarter_data = income_statement[income_statement.index.str.endswith(quarter)]

        # Tính trung bình 5 kỳ gần nhất
        rolling_mean = quarter_data['grow_rate'].rolling(window=5).mean()

        # Gán kết quả vào DataFrame gốc
        income_statement.loc[quarter_data.index, '5_year_avg_grow'] = rolling_mean

    y = income_statement['5_year_avg_grow']

    # get stock price
    stock = vnstock3.Vnstock(source='TCBS', show_log=False).stock(symbol=symbol, source='TCBS')
    df = stock.quote.history(start='2015-01-01', end=str(datetime.date.today()))
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Step 1: Create 'quarter' column in df
    df['quarter'] = df.index.to_period('Q')

    # merge x
    # Step 2: Convert Series x to DataFrame for merging
    income_df = x.reset_index()
    income_df.columns = ['quarter', 'price_to_earning']
    income_df['quarter'] = pd.PeriodIndex(income_df['quarter'], freq='Q')

    # Step 3: Merge df with income_df based on 'quarter', keep original index
    df = df.merge(income_df, on='quarter', how='left').set_index(df.index)

    # merge y
    income_df = y.reset_index()
    income_df.columns = ['quarter', 'grow_rate']
    income_df['quarter'] = pd.PeriodIndex(income_df['quarter'], freq='Q')

    # Step 3: Merge df with income_df based on 'quarter', keep original index
    df = df.merge(income_df, on='quarter', how='left').set_index(df.index)

    df.drop('quarter', axis=1, inplace=True)

    df = df.ffill()
    df['close'] = df['close'] * 1000
    df['earning_per_share'] = df['close'] / df['price_to_earning']
    df.loc[df['grow_rate'] > 25, 'grow_rate'] = 25
    df['fair_value'] = df['grow_rate'] * df['earning_per_share']
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(df.index, df['close'], color='green', alpha=0.8, label='Market price')
    ax.plot(df.index, df['fair_value'], color='red', alpha=0.8, label='Fair price')
    ax.legend()
    ax.set_title(f'Market price and fair price of {symbol}')
    file_name = "market_price_vs_fair_price_3.png"
    dir = f"/home/hoang/Pictures/{symbol}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Đặt tiêu đề và lưu hình
    fig.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.savefig(os.path.join(dir, file_name))
    plt.show()

    plt.show()


# In[36]:


def plot_all(symbol):
    icb_code = get_ticker_icb_code(symbol)
    if icb_code == get_ticker_icb_code('TCB'):
        balance_sheet_for_bank(symbol)
    else:
        balance_sheet_processing_for_normal_company(symbol)
    cash_flow_processing(symbol)
    ticker_ratio, industry_ratio = get_ratio(symbol)
    profit_on_revenue_plot(symbol)
    expense_on_profit_plot(symbol)
    company_vs_industry_roa(symbol, ticker_ratio, industry_ratio)
    company_vs_industry_roe(symbol, ticker_ratio, industry_ratio)
    company_vs_industry_pe(symbol, ticker_ratio, industry_ratio)
    company_vs_industry_pb(symbol, ticker_ratio, industry_ratio)
    equity_on_debt(symbol, ticker_ratio, industry_ratio)
    company_roe_vs_roa(symbol)
    plot_price(symbol)
    plot_price2(symbol)
    plot_price3(symbol)

