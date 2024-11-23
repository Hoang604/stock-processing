import vnstock3
from data_acquisition_and_processing import get_company_ratio
from data_acquisition_and_processing import get_stock_price
import pandas as pd
from time import sleep

def all_tickers():
    temp = vnstock3.Vnstock(show_log=False).stock('ACB', source='TCBS')
    industries_tickers = temp.listing.symbols_by_industries()
    return industries_tickers['symbol'].tolist()  # Return the list directly

def low_pe_pb():
    tickers = all_tickers()
    print(len(tickers))
    pe_list = []
    pb_list = []
    for ticker in tickers:
        try:
            df = get_company_ratio(ticker)[['price_to_earning', 'price_to_book']]
            print(ticker)
            pe_rows = df[df['price_to_earning'] < 4].copy() # Find PE values below 4
            if not pe_rows.empty:
                pe_rows['ticker'] = ticker  # Add ticker to the DataFrame
                pe_rows = pe_rows.rename_axis('time').reset_index()  # Reset the index to a column named 'time'
                pe_rows = pe_rows[['ticker', 'time', 'price_to_earning']]  # Reorder the columns
                pe_list.extend(pe_rows.to_dict('records')) # Add the selected rows as dictionaries to the list

            pb_rows = df[df['price_to_book'] < 1].copy() # Find PB values below 1
            if not pb_rows.empty:
                pb_rows['ticker'] = ticker   # Add ticker to the DataFrame
                pb_rows = pb_rows.rename_axis('time').reset_index()  # Reset the index to a column named 'time'
                pb_rows = pb_rows[['ticker', 'time', 'price_to_book']]  # Reorder the columns
                pb_list.extend(pb_rows.to_dict('records'))# Add the selected rows as dictionaries to the list


        except (KeyError, TypeError, IndexError) as e:  # Handle potential errors (e.g., no data for a ticker)
            print(f"Error processing {ticker}: {e}")
            continue
        sleep(1)
    
    pe_df = pd.DataFrame(pe_list) # Create the PE DataFrame
    pb_df = pd.DataFrame(pb_list) # Create the PB DataFrame
    return pe_df, pb_df

def fix():
    pb = pd.read_csv('low_pb.csv', index_col='Unnamed: 0')
    pe = pd.read_csv('low_pe.csv', index_col='Unnamed: 0')

    pb_tickers = pb.ticker.drop_duplicates().tolist()
    pe_tickers = pe.ticker.drop_duplicates().tolist()

    pb['median_vol'] = 0.0
    pe['median_vol'] = 0.0


    i = 1

    for ticker in pb_tickers:
        try:
            print(i)
            i += 1
            df = get_stock_price(ticker, start='2023-01-01')
            median_vol = df['volume'].median()
            pb.loc[pb['ticker'] == ticker, 'median_vol'] = median_vol
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    for ticker in pe_tickers:
        try:
            print(i)
            i += 1
            df = get_stock_price(ticker, start= '2023-01-01')
            median_vol = df['volume'].median()
            pe.loc[pe['ticker'] == ticker, 'median_vol'] = median_vol
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    pb = pb[pb['median_vol'] > 500]
    pe = pe[pe['median_vol'] > 500]

    pe = pe.sort_values(by=['ticker', 'time'])
    pb = pb.sort_values(by=['ticker', 'time'])

    pe = pe[['ticker', 'time', 'price_to_earning', 'median_vol']]
    pb = pb[['ticker', 'time', 'price_to_book', 'median_vol']]

    pe.to_csv('low_pe.csv')
    pb.to_csv('low_pb.csv')

    print('done')

import pandas as pd
from data_acquisition_and_processing import get_stock_price
from datetime import timedelta
from dateutil.relativedelta import relativedelta



pd.set_option('display.max_rows', None)

pb = pd.read_csv('/home/hoang/python/stock_processing/data/low_pb.csv', index_col='index')

pb['raise_from_lowest_price'] = 'N'
pb['raise_from_highest_price'] = 'N'

pb.time = pd.to_datetime(pb.time)

pb_tickers = pb.ticker.drop_duplicates().tolist()

def get_lowest_price_near_times(price_df, times):
    # Đảm bảo times là một list các đối tượng datetime
    times = pd.to_datetime(times)
    
    selected_rows = []
    for time in times:
        start_date = time - timedelta(days=3)
        end_date = time + timedelta(days=3)
        mask = (price_df['time'] >= start_date) & (price_df['time'] <= end_date)
        selected_rows.extend(price_df.loc[mask].index.tolist())  # Add row indices

    # Loại bỏ các hàng trùng lặp nếu có
    selected_rows = sorted(list(set(selected_rows)))
    df = price_df.loc[selected_rows]
    return df.loc[df['close'] == df['close'].min()]

def get_highest_price_near_times(price_df, times):
    # Đảm bảo times là một list các đối tượng datetime
    times = pd.to_datetime(times)
    
    selected_rows = []
    for time in times:
        start_date = time - timedelta(days=3)
        end_date = time + timedelta(days=3)
        mask = (price_df['time'] >= start_date) & (price_df['time'] <= end_date)
        selected_rows.extend(price_df.loc[mask].index.tolist())  # Add row indices

    # Loại bỏ các hàng trùng lặp nếu có
    selected_rows = sorted(list(set(selected_rows)))
    df = price_df.loc[selected_rows]
    return df.loc[df['close'] == df['close'].max()]

i = 0
k = 0
loop = 1
for tic in pb_tickers:

    print(f'{loop}: {tic}')
    loop +=1
    try:
        df = get_stock_price(tic, start='2015-01-01')
    except Exception as s:
        print(f"Error processing {tic}:")
        continue

    times = pb.loc[pb['ticker']==tic, 'time'].tolist()

    # Tìm giá thấp nhất trong những lúc pb < 4
    lowest_price_row = get_lowest_price_near_times(df, times).iloc[0]
    highest_price_row = get_highest_price_near_times(df, times).iloc[0]

    # Thời gian và giá thấp nhất tương ứng
    lowest_time = lowest_price_row['time']
    lowest_price = lowest_price_row['close']

    highest_time = highest_price_row['time']
    highest_price = highest_price_row['close']



    three_years_later_from_lowest_time = lowest_time + relativedelta(years=3)

    df_lowest = df[(df['time'] >= lowest_time) & (df['time'] <= three_years_later_from_lowest_time)]
    
    three_years_later_from_highest_time = highest_time + relativedelta(years=3)

    df_highest = df[(df['time'] >= highest_time) & (df['time'] <= three_years_later_from_highest_time)]


    if df_lowest['close'].max() > lowest_price * 2:
        pb.loc[pb['ticker']==tic, 'raise_from_lowest_price'] = 'x2'
        k+=1
        print(2)

    elif df['close'].max() > lowest_price * 1.5:
        pb.loc[pb['ticker']==tic, 'raise_from_lowest_price'] = 'x1.5'
        i+=1
        print(1.5)
    else: 
        pb.loc[pb['ticker']==tic, 'raise_from_lowest_price'] = 'N'

    if df_highest['close'].max() > highest_price * 2:
        pb.loc[pb['ticker']==tic, 'raise_from_highest_price'] = 'x2'
        k+=1
        print(2)
    elif df_highest['close'].max() > highest_price * 1.5:
        pb.loc[pb['ticker']==tic, 'raise_from_highest_price'] = 'x1.5'
        i+=1
        print(1.5)
    else: 
        pb.loc[pb['ticker']==tic, 'raise_from_highest_price'] = 'N'




print('x1.5:', i)
print('x2:', k)
    




pb.to_csv('/home/hoang/python/stock_processing/data/low_pb.csv')


print('done')

