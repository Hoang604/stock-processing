import vnstock3
from data_acquisition_and_processing import get_company_ratio
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

pe = pd.read_csv("low_pe.csv", index_col='Unnamed: 0')
pb = pd.read_csv("low_pb.csv", index_col='Unnamed: 0')

print(pe.ticker.drop_duplicates().tolist())
print(pb.ticker.drop_duplicates().tolist())

pe.to_csv("low_pe.csv")
pb.to_csv("low_pb.csv")

print(pe)
print(pb)