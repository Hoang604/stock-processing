from time import sleep
import multiprocessing
from visualization import *
from data_acquisition_and_processing import *
import pandas as pd

def process_ticker(ticker):
    """Wrapper function to be used in multiprocessing"""
    try:
        temp(ticker)
        return ticker  # Return the ticker if successful
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None  # Return None if there's an error


if __name__ == '__main__':
    temp('SSI')

    # failed_tickers = tickers.copy() # Initially, all tickers are considered potentially failed

    # while failed_tickers:  # Keep retrying until failed_tickers is empty
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    #         results = pool.map(process_ticker, failed_tickers)  # Process only the failed ones

    #     successful_tickers = [r for r in results if r is not None]
    #     # Update failed_tickers for the next iteration, removing successful ones
    #     failed_tickers = [t for t in failed_tickers if t not in successful_tickers]

    #     if failed_tickers:  # Only print if there are still failures
    #          print("Retrying the following tickers:", failed_tickers)
    #          print("wating 10 seconds......")
    #          sleep(10)
    #     else:
    #         print("All tickers processed successfully.")





