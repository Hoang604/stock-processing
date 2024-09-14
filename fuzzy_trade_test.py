import pandas as pd
import numpy as np
import yfinance as yf
from vnstock3 import Vnstock
import datetime
import math
import fuzzylite as fl


# step 1: read data(from Yahoo Finance)
def get_stock_data(symbol):
    df = yf.download(symbol, start="2000-01-01", end=str(datetime.date.today()))
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
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    df = stock.quote.history(start = "2000-08-08", end = str(datetime.date.today()))
    # df = vn.stock_historical_data(symbol, start_date="2000-08-08", end_date = str(datetime.date.today()), resolution='1D', type=dtype)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df['week'] = df.index.to_period('W').start_time
    df_weekly = df.groupby('week').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    df_weekly.rename(columns={'week': 'Date'}, inplace=True)    
    df_weekly.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.drop('week', axis = 1, inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

def calculate_macd(df):
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD hist'] = (df['MACD'] - df['Signal Line']) / df['close']
    df['MACD histogram'] = df['MACD hist']
    return df

def filter_macd(df):
    df['decrease 1'] = 0.0
    df['decrease 2'] = 0.0
    df['increase 1'] = 0.0
    df['increase 2'] = 0.0
    for i in range(37, len(df)):
        
        current_histogram = df.iloc[i]['MACD hist']
        prev_histogram = df.iloc[i-1]['MACD hist']
        prev2_histogram = df.iloc[i-2]['MACD hist']
        close_price = df.iloc[i]['close']
        macd_threshold = 0.1 * close_price  # 3% của giá đóng cửa

        # Chỉ thực thi khi MACD và MACD histogram lớn hơn ngưỡng
        if df.iloc[i]['MACD'] > macd_threshold and (abs(df['MACD hist'].iloc[i-1]) > 0.011 or abs(df['MACD hist'].iloc[i-2]) > 0.011):
            # Check for positive MACD histogram decreasing more than 30%
            if current_histogram > 0 and prev_histogram > 0 and prev2_histogram > 0 and current_histogram < prev_histogram:
                decrease_1 = (prev_histogram - current_histogram) / prev_histogram
                decrease_2 = (prev2_histogram - current_histogram) / prev2_histogram
                df.at[df.index[i], 'decrease 1'] = decrease_1
                df.at[df.index[i], 'decrease 2'] = decrease_2
                if decrease_1 > 0.66 or decrease_2 > 0.66:
                    df.at[df.index[i], 'MACD histogram'] = -0.5
                    # Set following values to -0.5 until MACD histogram increases or turns negative
                    for j in range(i+1, len(df)):
                        next_histogram = df.iloc[j]['MACD hist']
                        if next_histogram > current_histogram:
                            break
                        df.at[df.index[j], 'MACD histogram'] = -0.5
                        current_histogram = next_histogram

            # Check for negative MACD histogram increasing more than 30%
            elif current_histogram < 0 and prev_histogram < 0 and prev2_histogram < 0 and current_histogram > prev_histogram:
                increase_1 = (current_histogram - prev_histogram) / abs(prev_histogram)
                increase_2 = (current_histogram - prev2_histogram) / abs(prev2_histogram)
                df.at[df.index[i], 'increase 1'] = increase_1
                df.at[df.index[i], 'increase 2'] = increase_2
                
                if increase_1 > 0.66 or increase_2 > 0.66:
                    df.at[df.index[i], 'MACD histogram'] = 0.5
                    # Set following values to 0.5 until MACD histogram decreases or turns positive
                    for j in range(i+1, len(df)):
                        next_histogram = df.iloc[j]['MACD hist']
                        if next_histogram < current_histogram:
                            break
                        df.at[df.index[j], 'MACD histogram'] = 0.5
                        current_histogram = next_histogram
    return df

def calculate_rsi(df):
    delta = df['close'].diff() 
    gain = ( delta.where(delta > 0, 0)).rolling(window=69).mean()  
    loss = (-delta.where(delta < 0, 0)).rolling(window=69).mean()  
    
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))  # Công thức tính RSI
    return df

def calculate_golden_cross(df):
    df['EMA47'] = df['close'].ewm(span=47, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

    df['EMA subtract'] = df['EMA47'] - df['EMA200']
    
    # Tạo cột emacross, khởi tạo bằng 0
    df['emacross'] = 0

    # Xác định khi EMA47 cắt qua EMA200 (từ dưới lên hoặc từ trên xuống)
    cross_up = (df['EMA47'].shift(1) < df['EMA200'].shift(1)) & (df['EMA47'] > df['EMA200'])
    cross_down = (df['EMA47'].shift(1) > df['EMA200'].shift(1)) & (df['EMA47'] < df['EMA200'])
    
    # Khi EMA47 cắt EMA200, thiết lập giá trị emacross là 30
    df.loc[cross_up | cross_down, 'emacross'] = 30

    if len(df) < 201:
        raise ValueError("DataFrame có ít hơn 201 hàng, không thể thực hiện phép toán này.")

    # Giảm giá trị emacross đi 1 sau mỗi phiên, đến 0 thì dừng lại
    for i in range(0, 200):
        df.iloc[i, df.columns.get_loc('emacross')] = 1  # Đảm bảo sử dụng iloc cho vị trí cột
    for i in range(200, len(df)):
        if df.iloc[i, df.columns.get_loc('emacross')] == 0:
            continue  # Nếu emacross đã bằng 0, không thay đổi gì nữa
        elif df.iloc[i, df.columns.get_loc('emacross')] > 0:
            df.iloc[i, df.columns.get_loc('emacross')] = df.iloc[i-1, df.columns.get_loc('emacross')] - 1
    return df

def identify_trend(df):
    # Xác định xu hướng (Trend)
    df['Trend'] = None
    df.loc[df['emacross'] > 0, 'Trend'] = 0
    df.loc[df['emacross'] == 0, 'Trend'] = df['EMA subtract'] / df['EMA200']
    df['Trend'] = df['Trend'].astype(float)
    # 1 hoặc -1 nếu hai EMA tách xa nhau ra 
    # 0 nếu không có xu hướng cụ thể
    df['EMA subtract trend'] = df['EMA subtract'].rolling(window=15).apply(lambda x: 1 if all(x.diff()[1:] > 0) 
                                                                      else (-1 if all(x.diff()[1:] < 0) 
                                                                            else 0))

    # Tạo cột để tính phần trăm thay đổi trong 5 kỳ gần nhất
    df['EMA subtract change (%)'] = df['EMA subtract'].pct_change(periods=5) * 100

    # Cập nhật giá trị cho cột 'Trend' dựa trên điều kiện về sự thay đổi của EMA subtract
    trend_active = None
    for i in range(len(df)):
        if df['EMA subtract'].iloc[i] > 0:
            if df['EMA subtract change (%)'].iloc[i] < -10:
                trend_active = -1
            elif trend_active == -1 and df['EMA subtract'].iloc[i] > df['EMA subtract'].iloc[i-1]:
                trend_active = None
        elif df['EMA subtract'].iloc[i] < 0:
            if df['EMA subtract change (%)'].iloc[i] > 10:
                trend_active = 1
            elif trend_active == 1 and df['EMA subtract'].iloc[i] < df['EMA subtract'].iloc[i-1]:
                trend_active = None

        if trend_active:
            df.iloc[i, df.columns.get_loc('Trend')] = trend_active
    return df

def calculate_indicators(df):
    df['close'] = df['Close'].ewm(span = 11, adjust = False).mean()

    df = calculate_macd(df)
    df = filter_macd(df)

    df = calculate_rsi(df)
    
    df = calculate_golden_cross(df)
    df = identify_trend(df)
    
    aboveEMA200 = 0
    for i in range(200, len(df)):
        if df['close'].iloc[i] > df['EMA200'].iloc[i]:
            aboveEMA200 +=1 
        
    return df, (aboveEMA200/(len(df) - 200))


def create_fuzzy_system():
    # Initialize the fuzzy engine
    engine = fl.Engine(
        name="TradingDecisionEngine",
        input_variables=[
            fl.InputVariable(
            name="macd",
            minimum=-1.0,
            maximum=1.0,
            terms=[
                fl.Trapezoid("negative", -1.0, -1.0, -0.002, 0.002),
                fl.Trapezoid("positive", -0.002, 0.002, 1.0, 1.0)
            ]),
            fl.InputVariable(
            name="rsi",
            minimum=0.0,
            maximum=100.0,
            terms=[
                fl.Trapezoid("low", 0.0, 0.0, 40.0, 40.0),
                fl.Trapezoid("medium", 40.0, 40.0, 60.0, 60.0),
                fl.Trapezoid("high", 60.0, 60.0, 100.0, 100.0)
            ]),
            fl.InputVariable(
            name="trend",
            minimum=-10.0,
            maximum=10.0,
            terms=[
                fl.Trapezoid("downtrend", -10.0, -10.0, -0.05, -0.025),
                fl.Trapezoid("sideway", -0.05, -0.025, 0.025, 0.05),
                fl.Trapezoid("uptrend", 0.025, 0.05, 10.0, 10.0)
            ])
        ],
        output_variables=[
            fl.OutputVariable(
            name="decision",
            minimum=0.0,
            maximum=1.0,
            default_value= 0.515,
            defuzzifier=fl.Centroid(100),
            aggregation= fl.DrasticSum(),
            terms=[
                fl.Trapezoid("sell", 0.0, 0.0, 0.2, 0.5),
                fl.Trapezoid("hold", 0.0, 0.4, 0.6, 1.0),
                fl.Trapezoid("buy", 0.5, 0.8, 1.0, 1.0)
            ])
        ],
        rule_blocks = [
            fl.RuleBlock(
                name="mamdani",
                conjunction= fl.BoundedDifference(),
                disjunction=fl.Maximum(),
                implication=fl.AlgebraicProduct(),
                activation=fl.General(),
                rules=[
                    fl.Rule.create("if macd is positive and rsi is low then decision is buy"),
                    fl.Rule.create("if macd is positive and rsi is medium and trend is uptrend then decision is buy"),
                    fl.Rule.create("if macd is positive and rsi is medium and trend is sideway then decision is buy"),
                    fl.Rule.create("if macd is positive and rsi is medium and trend is downtrend then decision is hold"),
                    fl.Rule.create("if macd is positive and rsi is high and trend is uptrend then decision is buy"),
                    fl.Rule.create("if macd is positive and rsi is high and trend is sideway then decision is hold"),
                    fl.Rule.create("if macd is positive and rsi is high and trend is downtrend then decision is sell"),
                    fl.Rule.create("if macd is negative and rsi is high then decision is sell"),
                    fl.Rule.create("if macd is negative and rsi is medium then decision is sell"),
                    fl.Rule.create("if macd is negative and rsi is low and trend is uptrend then decision is hold"),
                    fl.Rule.create("if macd is negative and rsi is low and trend is sideway then decision is sell"),
                    fl.Rule.create("if macd is negative and rsi is low and trend is downtrend then decision is sell")
                ]
            )
        ]
    )

    return engine

def apply_fuzzy_logic(df):
    engine = create_fuzzy_system()
    decisions = []
    position = []
    for i in range(200):
        decisions.append(0.5)
        position.append('hold')
    for i in range(200, len(df)):
        macd_value = df['MACD hist'].iloc[i]
        rsi_value = df['RSI'].iloc[i]
        trend_value = df['Trend'].iloc[i]
        engine.input_variables[0].value = macd_value
        engine.input_variables[1].value = rsi_value
        engine.input_variables[2].value = trend_value

        engine.process()
        np.set_printoptions(suppress=True, precision=4)

        # print(engine.values, engine.output_variable('decision').fuzzy_value())
        decision_value = engine.output_variable('decision').value
        decisions.append(decision_value)
        if decision_value < 0.33:
            position.append('sell')
        elif decision_value < 0.66:
            position.append('hold')
        else:
            position.append('buy')

    df['Decision'] = decisions
    df['Position'] = position
    return df


def testing_for_test_all(df, percent_above_EMA200):
    initial_budget = 1000000
    total_budget = initial_budget
    holding = False  # Cờ để kiểm tra xem đang giữ vị thế hay không
    buy_day = []
    sell_day = []
    sell_price = 0.0
    buy_price = 0.0
    peak_price = 0  # Đỉnh giá bán trước đó
    cooldown_period = 0  # Khoảng thời gian chờ sau khi bán (nếu lợi nhuận > 70%)
    cumulative_holding_period = 0  # Tổng thời gian giữ vị thế
    total_trade = 0
    total_win = 0
    longest_win_streak = 0
    longest_loss_streak = 0
    last_is_win = False
    win_streak = 0
    loss_streak = 0
    biggest_win = 0.0 
    biggest_loss = 0.0
    just_touch_stoploss = False
    stoplossPrice = 0.0
    if percent_above_EMA200 < 0.9:

        for i in range(200, len(df)):
            if cooldown_period > 0:  # Đang trong giai đoạn chờ
                # Nếu giá vượt đỉnh, cho phép giao dịch lại
                if df['Close'].iloc[i] > peak_price and df['Decision'].iloc[i] >= 0.6:
                    cumulative_holding_period += cooldown_period  # Cộng thời gian giữ vị thế trước
                    cooldown_period = 0  # Kết thúc giai đoạn chờ
                else:
                    df.at[df.index[i], 'Position'] = 'hold'  # Cập nhật cột Position
                    df.at[df.index[i], 'Decision'] = 0.51  # Cập nhật cột Decision
                    cooldown_period -= 1
                    continue  # Bỏ qua giao dịch trong giai đoạn chờ
            
            # Nếu hết coolldown_period, thì reset peak price
            if cooldown_period == 0: 
                peak_price = 0.0
            # Nếu vừa chạm stop loss và gặp lệnh không phải lệnh buy thì coi như là trước đó chưa chạm stoploss
            # Nếu vừa chạm stop loss và tăng lại lên trên stop loss thì coi như trước đó chưa chạm stop loss
            if just_touch_stoploss and (df['Decision'].iloc[i] < 0.6 or df['Close'].iloc[i] > stoplossPrice) :
                just_touch_stoploss = 0

            # Mua khi không giữ vị thế và có tín hiệu mua và không ở trong giai đoạn sau khi chạm stop loss
            if not holding and df['Decision'].iloc[i] >= 0.6 and not just_touch_stoploss:
                buy_price = df['Close'].iloc[i]
                stoplossPrice = 0.9 * buy_price
                buy_day.append(i)
                holding = True  # Đặt cờ giữ vị thế 

            # Bán khi đang giữ vị thế và có tín hiệu bán hoặc khi giá giảm dưới đỉnh trước hoặc giá chạm stoploss
            elif holding and (df['Decision'].iloc[i] <= 0.3 or df['Close'].iloc[i] < peak_price or i == len(df) - 1 or df['Close'].iloc[i] < stoplossPrice):
                sell_price = df['Close'].iloc[i]
                sell_day.append(i)

                # Tính toán lợi nhuận
                account_change_rate = sell_price / buy_price
                if df['Close'].iloc[i] < stoplossPrice:
                    account_change_rate = 0.9
                    just_touch_stoploss = True
                    
                profit_percent = (account_change_rate - 1) * 100
                total_budget *= account_change_rate
                total_trade += 1
                if profit_percent > 0:
                    if profit_percent > biggest_win:
                        biggest_win = profit_percent
                    total_win += 1
                    last_is_win = True
                    loss_streak = 0
                else: 
                    if profit_percent < biggest_loss:
                        biggest_loss = profit_percent
                    win_streak = 0
                    last_is_win = False
                if last_is_win:
                    win_streak += 1
                    if longest_win_streak < win_streak:
                        longest_win_streak = win_streak 
                else:
                    loss_streak += 1
                    if longest_loss_streak < loss_streak:
                        longest_loss_streak = loss_streak
                holding = False  # Đặt cờ ngừng giữ vị thế

                # Tính tổng thời gian giữ vị thế (cộng với các vị thế trước nếu giá vượt đỉnh)
                holding_period = sell_day[-1] - buy_day[-1] + cumulative_holding_period
                # Nếu lợi nhuận > 70%, hoặc vừa mua thêm sau khi giá tăng quá 70%, đặt cooldown period
                if profit_percent > 70 or cumulative_holding_period > 0:
                    cooldown_period = min(holding_period, 60) 
                    peak_price = sell_price  # Đỉnh giá bán là giá bán hiện tại

                # Reset thời gian giữ vị thế tích lũy
                cumulative_holding_period = 0

    else: 
        for i in range(200, len(df)):
            # Nếu vừa chạm stop loss và gặp lệnh không phải lệnh buy thì coi như là trước đó chưa chạm stoploss
            # Nếu vừa chạm stop loss và tăng lại lên trên stop loss thì coi như trước đó chưa chạm stop loss
            if just_touch_stoploss and (df['Decision'].iloc[i] < 0.6 or df['Close'].iloc[i] > stoplossPrice) :
                just_touch_stoploss = False
            
            # Mua khi không giữ vị thế và có tín hiệu mua và không ở trong giai đoạn sau khi chạm stop loss
            if not holding and df['Decision'].iloc[i] >= 0.6 and not just_touch_stoploss:  # Buy condition
                buy_price = df['Close'].iloc[i]
                stoplossPrice = 0.9 * buy_price
                buy_day.append(i)
                holding = True  # Set holding flag to True after buying
            
            # Bán khi đang giữ vị thế và có tín hiệu bán hoặc giá chạm stoploss
            elif holding and (df['Decision'].iloc[i] <= 0.3 or i == len(df) - 1 or df['Close'].iloc[i] < stoplossPrice): 
                sell_price = df['Close'].iloc[i]
                sell_day.append(i)
                
                # Tính toán lợi nhuận
                account_change_rate = sell_price / buy_price
                if df['Close'].iloc[i] < stoplossPrice:
                    account_change_rate = 0.9
                    just_touch_stoploss = True
                    
                profit_percent = (account_change_rate - 1) * 100
                total_budget *= account_change_rate
                total_trade += 1
                if profit_percent > 0:
                    if profit_percent > biggest_win:
                        biggest_win = profit_percent
                    total_win += 1
                    last_is_win = True
                    loss_streak = 0
                else: 
                    if profit_percent < biggest_loss:
                        biggest_loss = profit_percent
                    win_streak = 0
                    last_is_win = False
                if last_is_win:
                    win_streak += 1
                    if longest_win_streak < win_streak:
                        longest_win_streak = win_streak 
                else:
                    loss_streak += 1
                    if longest_loss_streak < loss_streak:
                        longest_loss_streak = loss_streak
                holding = False  # Đặt cờ ngừng giữ vị thế
    win_rate = (total_win / total_trade) * 100
    total_profit_percent = ((total_budget - initial_budget) / initial_budget) * 100
    return total_budget, total_profit_percent, win_rate, longest_loss_streak, longest_win_streak, total_trade, biggest_win, biggest_loss
def filter(df):
    # filter by trend
    for i in range (0, len(df)):
        if df['EMA subtract trend'].iloc[i] == 1 and df['EMA subtract'].iloc[i] > 0:
            df.iloc[i, df.columns.get_loc('Decision')] = 0.8
            df.iloc[i, df.columns.get_loc('Position')] = 'buy'
    # filter by MACD
    for i in range(37, len(df)):
        if df.iloc[i]['MACD histogram'] == -0.5 :
            df.iloc[i, df.columns.get_loc('Decision')] = 0.2
            df.iloc[i, df.columns.get_loc('Position')] = 'sell'
        elif df.iloc[i]['MACD histogram'] == 0.5 :
            df.iloc[i, df.columns.get_loc('Decision')] = 0.8
            df.iloc[i, df.columns.get_loc('Position')] = 'buy'
    
    return df

def test_all():
    # stock_codes = [
    # "TCH", "TLG", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB",
    # "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VRE", "VSH",
    # "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC", "TCB",
    # "NLG", "NT2", "NVL", "OCB", "PAN", "PC1", "PDR", "PHR", "PLX", "PNJ",
    # "POW", "PPC", "PTB", "PVD", "PVT", "REE", "SAB", "SBT", "SCS",
    # "HDB", "HDC", "HDG", "HHV", "HPG", "HSG", "HT1", "IMP", "KBC", "KDC",
    # "KDH", "KOS", "LPB", "MBB", "MSB", "MSN", "MWG", "NKG",
    # "DBC", "DCM", "DGC", "DGW", "DIG", "DPM", "DXG", "DXS", "EIB", "EVF",
    # "FPT", "FRT", "FTS", "GAS", "GEX", "GMD", "GVR", "HAG", "HCM"
    # ]
    stock_codes = [
    "TPB", "VCB", "VCI", "VGC", "VHC", "VIB",
    "VND", "VPB", "VPI", "VSH",
    "NLG", "NT2", "PC1", "PHR", "PNJ",
    "PTB", "PVT", "REE", "SCS",
    "HDB", "HDC", "HDG", "HPG", "IMP",
    "KDH", "KOS", "LPB", "MBB", "MWG",
    "DBC", "DGC", "DGW", 
    "FPT", "FRT", "FTS", "GAS", "GEX", "GMD", "HCM"
    ]
    # stock_codes = [ "FTS", "DGW", "DGC", "LPB", "VND", "FRT", "HPG", "PHR", "FPT", "HCM"]
    number_test  = 0
    avg_profit_all = 0.0
    total_trade_times = 0
    win_times = 0
    b_win = 0
    b_loss = 0
    higher_than_30 = {}
    higher_than_20 = {}
    lower_than_0 = {}
    for i in range(0, len(stock_codes)):
        data = get_vn_stock_data(stock_codes[i], 'stock')
        df = data[1]
        if (len(df) < 55 * 5):
            print("less than 5 years: ", stock_codes[i])
            continue
        begin_date = df.index[200]
        stop_date = df.index[len(df) - 1]
        time = stop_date - begin_date
        years = time.days / 365
        df, percent_aboveEMA200 = calculate_indicators(df)
        if percent_aboveEMA200 < 0.5:
            print("Highly recommend not buy this stock: ", stock_codes[i])
            continue
        elif percent_aboveEMA200 < 0.8: 
            print("Not recommend to buy this stock: ", stock_codes[i])
            continue
        number_test += 1
        df = apply_fuzzy_logic(df)
        df = filter(df)
        total_account, profit_percent, win_rate, longest_loss_streak, longest_win_streak, total_trade, biggest_win, biggest_loss = testing_for_test_all(df, percent_aboveEMA200)
        if biggest_loss < b_loss:
            b_loss = biggest_loss
        if biggest_win > b_win:
            b_win = biggest_win
        total_trade_times += total_trade
        win_times += total_trade * win_rate
        avg_profit = (math.pow(profit_percent/100 + 1, 1/years) - 1) * 100
        if avg_profit > 30:
            higher_than_30[stock_codes[i]] = avg_profit
        elif avg_profit > 20:
            higher_than_20[stock_codes[i]] = avg_profit
        elif avg_profit < 0:
            lower_than_0[stock_codes[i]] = avg_profit

        avg_profit_all = (avg_profit_all * (number_test - 1) + avg_profit)/ number_test
        print("initial account = 1000000, total account after testing", stock_codes[i], round(years, 1),  "years is", total_account, "profit percent =", round(profit_percent, 2), "%", 'average profit each year: ', round(avg_profit, 2), "%")
        print ('win rate: ', round(win_rate, 2), 'total trade times: ', total_trade, 'longest win streak: ', longest_win_streak, 'longest loss treak: ', longest_loss_streak, '\n')
    higher_than_30 = dict(sorted(higher_than_30.items(), key=lambda item: item[1], reverse=True))
    higher_than_20 = dict(sorted(higher_than_20.items(), key=lambda item: item[1], reverse=True))
    lower_than_0 = dict(sorted(lower_than_0.items(), key=lambda item: item[1], reverse=True))
    print('average profit per year after testing ', number_test ,' stocks is: ', round(avg_profit_all, 2), ' %')
    print('average win rate is: ', round(win_times / total_trade_times, 2) , ' %,  biggest win is: ', round(b_win, 2), '%, biggest loss is: ', round(b_loss, 2) , ' %')
    
    print("stocks has average profit higher than 30% each year: ")
    for key, value in higher_than_30.items():
        print(f"{key:<10} : {value:.2f}")

    print("stocks has average profit higher than 20% each year: ")
    for key, value in higher_than_20.items():
        print(f"{key:<10} : {value:.2f}")

    print("stocks has average profit lower than 0 % each year: ")
    for key, value in lower_than_0.items():
        print(f"{key:<10} : {value:.2f}")
        

        

# step 5: Run Program   
if __name__ == "__main__":
    print('========== Securities analysis program ==========')
    print('Rule:')
    print('1. Buy at the first buy signal')
    print("2. Wait for the trade to close before considering new buy signals.")
    print('3. Sell at the first sell signal')
    print('4. Do nothing when ecounter hold signal\n\n')
    test_all()
    