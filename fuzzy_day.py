import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import yfinance as yf
from skfuzzy import control as ctrl
import vnstock as vn
import datetime

# step 1: read data(from Yahoo Finance)
def get_stock_data(symbol):
    df = yf.download(symbol, start="2010-01-01", end=str(datetime.date.today()))
    df['Week'] = df.index.to_period('W').start_time
    # week aggression
    df_weekly = df.groupby('Week').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    })
    df_weekly.rename(columns={'Week': 'Date'}, inplace=True)    
    return df, df_weekly

def get_vn_stock_data(symbol, dtype):
    df = vn.stock_historical_data(symbol, start_date="2010-08-08", end_date = str(datetime.date.today()), resolution='1D', type=dtype)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df['week'] = df.index.to_period('W').start_time
    df_weekly = df.groupby('week').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ticker': 'first'
    })
    df_weekly.rename(columns={'week': 'Date'}, inplace=True)    
    df_weekly.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    df.drop('week', axis = 1, inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    print(df.head())

    return df, df_weekly
# step 2: Calculate MACD and EMA200
def calculate_indicators(df):
    # Calculate MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD strategy'] = df['MACD'] - df['Signal Line']
    # Calculate volume indicator
    df['Volume_EMA35'] = df["Volume"].ewm(span=35, adjust = False).mean() 
    df['vol'] = df['Volume'] / df['Volume_EMA35']
    return df

# step 3: Define system fuzzy logic
def create_fuzzy_system():
    # Define input variable for fuzzy system
    macd = ctrl.Antecedent(np.arange(-10, 10, 0.001), 'macd')
    # define fuzzy sets for MACD
    macd['negative'] = fuzz.trapmf(macd.universe, [-10, -10, 0, 0])
    macd['positive'] = fuzz.trapmf(macd.universe, [0, 0, 10, 10])


    vol = ctrl.Antecedent(np.arange(0, 10, 0.001), "vol")
    #define fuzzy sets for vol 
    vol['low'] = fuzz.trimf(vol.universe, [0, 0, 0.8])
    vol['high'] = fuzz.trapmf(vol.universe, [0.8, 1, 10, 10])


    # define  output variable fuzzy system
    decision = ctrl.Consequent(np.arange(0, 1, 0.01), 'decision')
    # define fuzzy sets for decision (0: sell, 1: buy)
    decision['sell'] = fuzz.trimf(decision.universe, [0, 0, 0.5])
    decision['hold'] = fuzz.trimf(decision.universe, [0, 0.5, 1])
    decision['buy'] = fuzz.trimf(decision.universe, [0.75, 1, 1]) 

    # define fuzzy rules
    rule1 = ctrl.Rule(macd['positive'] & vol['high'], decision['buy'])
    rule2 = ctrl.Rule(macd['negative'] & vol['high'], decision['sell']) 

    rule_hold = ctrl.Rule(vol['low'], decision['hold'])


    # Create control system 
    decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule_hold])
    decision_simulation = ctrl.ControlSystemSimulation(decision_ctrl)

    return decision_simulation

# step 4: Apply fuzzy system for data
def apply_fuzzy_logic(df):
    decision_simulation = create_fuzzy_system()

    decisions = []
    position = []
    for i in range(35):
        decisions.append(0.5)
        position.append('hold')
    for i in range(35, len(df)):
        macd_value = df['MACD strategy'].iloc[i]
        vol_value = df['vol'].iloc[i]
        decision_simulation.input['macd'] = macd_value
        decision_simulation.input['vol'] = vol_value

        decision_simulation.compute()
        decisions.append(decision_simulation.output['decision'])

        if decisions[i] >= 0.875:
            position.append('buy')  
        elif decisions[i] <= 0.3:
            position.append('sell')
        else: position.append('hold')
    df['Decision'] = decisions
    df['Position'] = position
    return df


        
# step 5: Run Program   
if __name__ == "__main__":
    print('========== Securities analysis program ==========')
    print('Rule:')
    print('1. Buy at the first buy signal')
    print("2. Wait for the trade to close before considering new buy signals.")
    print('3. Sell at the first sell signal')
    print('4. Do nothing when ecounter hold signal\n\n')
    decide = input('1. For Vietnam stock/index,\n2. For foreign stock/ index\n')
    if decide == '1':
        symbol = input("enter stock/index symbol: ")
        dtype = 'stock'
        decision = input('1. For stock analysis\n2. For index analysis\n')
        if decision == '2': dtype = 'index'
        data = get_vn_stock_data(symbol, dtype)
    elif decide == '2': 
        symbol = input("enter stock/index symbol: ")
        data = get_stock_data(symbol)

    i = int(input("enter: 0. For 1 day time frame analysis\n       1. For 1 week time frame analysis\n       "))
    df = data[i]
    df = calculate_indicators(df)
    df = apply_fuzzy_logic(df)

    pd.set_option('display.max_rows', None)
    print(df['Position'])
    # Define the colors
    colors = {
        'sell': 'red',
        'hold': 'gray',
        'buy': 'green'
    }

    # Create a color column based on the Position column
    df['Color'] = df['Position'].map(colors)

    # Create subplots for the main chart and the histogram
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot the price and EMA200 on ax1
    ax1.plot(df.index, df['Close'], label='Close Price')
    # Scatter plot with custom colors and smaller points on ax1
    scatter = ax1.scatter(df.index, df['Close'], c=df['Color'], s=10, label='Decision')
    ax1.legend()

    # Plot the MACD on ax2
    ax2.plot(df.index, df['MACD'], label='MACD')
    ax2.plot(df.index, df['Signal Line'], label='Signal Line')
    ax2.bar(df.index, df['MACD strategy'], label='MACD Histogram', color='gray')
    ax2.legend()

    # Create histogram on ax3
    ax3.bar(df.index, df['Decision'], color=df['Color'])
    ax3.set_yticks([])  # Remove y-axis ticks for the histogram

    # Show plot
    plt.show()
