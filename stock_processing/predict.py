import data_acquisition_and_processing as dap
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import plotly.graph_objects as go


class Predict:
    def __init__(self, ticker):
        self.ticker = ticker
        self.income_statement = dap.get_income_statement(ticker)[['revenue', 'cost_of_good_sold', 'gross_profit',
                                                                  'operation_profit', 'operation_expense']]
        self.balance_sheet = dap.get_balance_sheet(ticker)[['short_asset', 'asset', 
                                                            'short_debt', 'debt', 'equity']]
        cash_flow = dap.recalculate_cash_flow(dap.get_cash_flow(ticker))
        self.cash_flow = cash_flow[['capEx', 'operating_cash_flow']]
        self.ratio = dap.get_company_ratio(ticker)[['roe', 'roa', 'price_to_earning', 'price_to_book',
                                                    'equity_on_liability', 'equity_on_total_asset',
                                                    'gross_profit_margin', 'operating_profit_margin',]]

    def predict_income_statement(self):
        results = {}
        for col in self.income_statement.columns:
            try:
                model = ARIMA(self.income_statement[col], order=(8, 1, 1))
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(self.income_statement), end=len(self.income_statement) + 11)
                results[col] = predictions
            except:
                results[col] = pd.Series([None] * 4)  # Placeholder for failed predictions


        # Convert the dictionary of predictions to a DataFrame
        predictions_df = pd.DataFrame(results)
        predictions_df.index = pd.to_datetime(predictions_df.index) 
        return predictions_df

    def predict_balance_sheet(self):
        results = {}
        for col in self.balance_sheet.columns:
            try:
                model = ARIMA(self.balance_sheet[col], order=(4,1,1))
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(self.balance_sheet), end=len(self.balance_sheet) + 11)
                results[col] = predictions
            except:
                results[col] = pd.Series([None] * 4)  # Placeholder for failed predictions
        
        predictions_df = pd.DataFrame(results)
        predictions_df.index = pd.to_datetime(predictions_df.index) 
        return predictions_df

    def predict_cash_flow(self):
        results = {}

        for col in self.cash_flow.columns:
            try:
                model = ARIMA(self.cash_flow[col], order=(4,1,1))
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(self.cash_flow), end=len(self.cash_flow) + 11)
                results[col] = predictions
            except:
                results[col] = pd.Series([None] * 4)  # Placeholder for failed predictions

        predictions_df = pd.DataFrame(results)
        predictions_df.index = pd.to_datetime(predictions_df.index) 
        return predictions_df

    def predict_ratio(self):
        results = {}
        for col in self.ratio.columns:
            try:
                model = ARIMA(self.ratio[col], order=(4,1,1))
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(self.ratio), end=len(self.ratio) + 11)
                results[col] = predictions
            except:
                results[col] = pd.Series([None] * 4)  # Placeholder for failed predictions

        predictions_df = pd.DataFrame(results)
        predictions_df.index = pd.to_datetime(predictions_df.index) 
        return predictions_df


#     def plot_predicted_revenue_and_cost_of_goods_sold(self):
#         """Vẽ đồ thị doanh thu và giá vốn hàng bán đã được dự đoán."""

#         predicted_income_statement = self.predict_income_statement()

#         # Lấy dữ liệu lịch sử
#         historical_revenue = self.income_statement['revenue']
#         historical_cost_of_goods_sold = self.income_statement['cost_of_good_sold']

#         # Lấy dữ liệu dự đoán
#         predicted_revenue = predicted_income_statement['revenue']
#         predicted_cost_of_goods_sold = predicted_income_statement['cost_of_good_sold']

#         fig = go.Figure()

#         # Vẽ doanh thu
#         fig.add_trace(go.Bar(x=historical_revenue.index, y=historical_revenue,
#                             name='Doanh thu (lịch sử)', marker_color='blue'))
#         fig.add_trace(go.Bar(x=predicted_revenue.index, y=predicted_revenue,
#                             name='Doanh thu (dự đoán)', marker_color='gold'))

#         # Vẽ giá vốn hàng bán
#         fig.add_trace(go.Bar(x=historical_cost_of_goods_sold.index, y=-historical_cost_of_goods_sold,  # Đảo ngược dấu để vẽ xuống dưới
#                             name='Giá vốn hàng bán (lịch sử)', marker_color='blue'))
#         fig.add_trace(go.Bar(x=predicted_cost_of_goods_sold.index, y=-predicted_cost_of_goods_sold,  # Đảo ngược dấu
#                             name='Giá vốn hàng bán (dự đoán)', marker_color='gold'))



#         title = f'Doanh thu và Giá vốn hàng bán (Dự đoán) - {self.ticker}'
#         from visualization import update_layout_dark # Import directly within the function
#         update_layout_dark(fig, title, yaxis_title='Giá trị')

#         fig_name = "predicted_revenue_and_cost_of_goods_sold.html"
#         from visualization import save_fig
#         save_fig(fig, self.ticker, fig_name)


#         fig.show()
    
# Predict('FPT').plot_predicted_revenue_and_cost_of_goods_sold()