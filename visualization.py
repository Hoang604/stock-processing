import numpy as np
import data_acquisition_and_processing as dap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def update_layout_dark(fig, title, yaxis_title="Value", yaxis2_title=None, yaxis3_title=None, yaxis4_title=None, barmode='group'):
    """Cập nhật layout cho biểu đồ với theme tối, hỗ trợ cả subplot và single plot."""
    layout_updates = dict(
        title=title,
        font_color='white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        hovermode='x unified',
        hoverlabel=dict(bgcolor="rgba(0, 0, 0, 0.8)", font_size=12),
        barmode=barmode,
        # bargroupgap=0.01,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(title='Time', gridcolor='rgba(128, 128, 128, 0.5)', color='white')
    )

    if yaxis_title:
        layout_updates['yaxis'] = dict(title=yaxis_title, gridcolor='rgba(128, 128, 128, 0.5)', color='white')
    if yaxis2_title:
        layout_updates['yaxis2'] = dict(title=yaxis2_title, gridcolor='rgba(128, 128, 128, 0)', color='white', overlaying='y', side='right')
    
    # Conditional updates for subplot axes
    if hasattr(fig.layout, 'xaxis2'):  # Check if xaxis2 exists
        layout_updates['xaxis2'] = dict(title='Time', gridcolor='rgba(128, 128, 128, 0.5)', color='white')
    if hasattr(fig.layout, 'yaxis3'):  # Check if yaxis3 exists
        layout_updates['yaxis3'] = dict(title=yaxis3_title, gridcolor='rgba(128, 128, 128, 0.5)', color='white') if yaxis3_title else None
    if hasattr(fig.layout, 'yaxis4'):  # Check if yaxis4 exists
        layout_updates['yaxis4'] = dict(title=yaxis4_title, gridcolor='rgba(128, 128, 128, 0.5)', color='white', overlaying='y3', side='right') if yaxis4_title else None


    return fig.update_layout(**layout_updates)


def save_fig(fig, ticker, fig_name, base_dir=None):
    """Lưu hình ảnh vào thư mục con ticker trong thư mục /stock_processing."""

    if base_dir is None:
        base_dir = os.path.dirname(__file__)  # Get parent directory
    if dap.get_ticker_icb_code(ticker) == 8350:  # Bank
        save_dir = os.path.join(base_dir,'picture', 'Bank', ticker)
    else:
        save_dir = os.path.join(base_dir,'picture', 'Normal_company', ticker)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    fig.write_html(os.path.join(save_dir, fig_name))

from icecream import ic

class CompanyVisualization:

    def __init__(self, ticker):
        self.ticker = ticker

    def cash_flow(self):
        df = dap.get_cash_flow(self.ticker)
        if df.empty:
            raise ValueError(f"No cash flow data available for {self.ticker}")
        df = dap.recalculate_cash_flow(df)
        # dap.ttm(df)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df['operating_cash_flow'], 
                                    name='Operating cash flow', marker_color='rgba(0, 141, 0, 1)'))
        fig.add_trace(go.Bar(x=df.index, y=-df['capEx'], name='Capital Expenditure', 
                                    marker_color='rgba(208, 21, 21, 0.4)'))
        
        title = f'Cash flow - {self.ticker}'
        update_layout_dark(fig, title, barmode='group')

        fig_name = "cash_flow.html"
        save_fig(fig, self.ticker, fig_name)
        #fig.show()

    def yearly_cash_flow(self):
        df = dap.get_cash_flow(self.ticker, period='year')
        ic(df)
        if df.empty:
            raise ValueError(f"No cash flow data available for {self.ticker}")
        df = dap.recalculate_cash_flow(df)
        dap.ttm(df)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df['operating_cash_flow'], 
                                    name='Operating cash flow', marker_color='rgba(0, 141, 0, 1)'))
        fig.add_trace(go.Bar(x=df.index, y=-df['capEx'], name='Capital Expenditure', 
                                    marker_color='rgba(208, 21, 21, 0.4)'))
        
        title = f'Cash flow - {self.ticker}'
        update_layout_dark(fig, title, barmode='group')

        fig_name = "yearly_cash_flow.html"
        save_fig(fig, self.ticker, fig_name)
        #fig.show()
    

    def balance_sheet(self):
        """
        Phân tích bảng cân đối kế toán với các chỉ số ngắn hạn và dài hạn
        """
        # Balance sheet gốc
        balance_sheet = dap.get_balance_sheet(self.ticker)
        if balance_sheet.empty:
            raise ValueError(f"No balance sheet data available for {self.ticker}")
        
        # Balance sheet dự đoán
        # prediction_balance_sheet = Predict(self.ticker).predict_balance_sheet()

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Short term', 'Long term'), vertical_spacing=0.15,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        
        # Vẽ biểu đồ tài sản và nợ gắn hạn
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['short_asset'], name='Short asset',
                                marker_color='rgba(0, 141, 0, 1)'), row=1, col=1, secondary_y=False)
        
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['short_debt'], name='Short debt',
                                marker_color='rgba(208, 21, 21, 0.4)'), row=1, col=1, secondary_y=False)

        # Vẽ biểu đồ tài sản và nợ tổng thể
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['asset'], name='Asset',
                                marker_color='rgba(0, 141, 0, 1)'), row=2, col=1, secondary_y=False)
        
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['debt'], name='Debt',
                                marker_color='rgba(208, 21, 21, 0.4)'), row=2, col=1, secondary_y=False)
        
        fig.add_trace(go.Scatter(x=balance_sheet.index, y=balance_sheet['equity'], name='Equity',
                                line=dict(color='#ff9800', width=3), mode='lines+markers'), row=2, col=1, secondary_y=True)


        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

        fig.update_layout(
            title=f'Balance sheet - {self.ticker}', font_color='white',
            plot_bgcolor='black', paper_bgcolor='black', hovermode='x unified',
            hoverlabel=dict(bgcolor="rgba(0, 0, 0, 0.8)", font_size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis=dict(tickangle=30, tickmode='array', ticktext=balance_sheet.index[::4],  # Use balance_sheet index
                    tickvals=balance_sheet.index[::4], gridcolor='rgba(128, 128, 128, 0.5)', color='white'),  #Grid 0.5
            xaxis2=dict(tickangle=30, tickmode='array', ticktext=balance_sheet.index[::4],  # Use balance_sheet index
                        tickvals=balance_sheet.index[::4], gridcolor='rgba(128, 128, 128, 0.5)', color='white'),  # Grid 0.5
            yaxis=dict(title="Tỉ đồng", gridcolor='rgba(128, 128, 128, 0.5)', color='white'),  #Grid 0.5
            yaxis3=dict(title="Tỉ đồng", gridcolor='rgba(128, 128, 128, 0.5)', color='white'), #Grid 0.5
            yaxis4=dict(title="Tỉ đồng", gridcolor='rgba(128, 128, 128, 0)', color='white'), #Grid 0
        )
        
        fig_name = 'balance_sheet.html'
        save_fig(fig, self.ticker, fig_name)
        ##fig.show()


    def profit_and_expense_plot(self):
        """
        Vẽ đồ thị doanh thu, lợi nhuận gộp, chi phí hoạt động và lợi nhuận hoạt động trên 2 subplot.
        """
        income_statement = dap.get_income_statement(self.ticker)
        if income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")


        fig = make_subplots(rows=2, cols=1, subplot_titles=('Revenue and profit', 'Gross profit and operation expense'),
                            vertical_spacing=0.15, specs=[[{}], [{"secondary_y": True}]])

        # Subplot 1: Doanh thu & Lợi nhuận
        fig.add_trace(go.Bar(name='Revenue', x=income_statement.index, y=income_statement['revenue'],
                             marker_color='rgba(0, 141, 0, 1)'), row=1, col=1)
        fig.add_trace(go.Bar(name='Cost of goods sold', x=income_statement.index,
                             y=-income_statement['cost_of_good_sold'], marker_color='rgba(208, 21, 21, 0.5)'),
                      row=1, col=1)

        # Subplot 2: Lợi nhuận gộp & Chi phí hoạt động
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['gross_profit'],
                             name="Gross profit", marker_color='rgba(0, 141, 0, 1)'), row=2, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['operation_expense'],
                             name="Operation expense", marker_color='rgba(208, 21, 21, 0.5)'), row=2, col=1)
        fig.add_trace(go.Scatter(name='Operation Profit', x=income_statement.index,
                                 y=income_statement['operation_profit'], line=dict(width=2),
                                 marker_color='yellow', mode='lines+markers'), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(name='Profit after tax', x=income_statement.index,
                                 y=income_statement['share_holder_income'], line=dict(width=2),
                                 marker_color='purple', mode='lines+markers'), row=2, col=1, secondary_y=True)

        title = f'Revenue, profit, expense - {self.ticker}'
        update_layout_dark(fig, title, yaxis4_title='Value', yaxis3_title='Value')

        fig_name = "revenue_profit_and_expense.html"
        save_fig(fig, self.ticker, fig_name)

        #fig.show()
    def profit_and_expense_plot_year(self):
        """
        Vẽ đồ thị doanh thu, lợi nhuận gộp, chi phí hoạt động và lợi nhuận hoạt động theo năm trên 2 subplot.
        """
        try:
            income_statement = dap.get_income_statement(self.ticker, period='year')
        except ValueError as e:
            print(e)
            return
        if income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Revenue and profit', 'Gross profit and operation expense'),
                            vertical_spacing=0.15, specs=[[{}], [{"secondary_y": True}]])

        # Subplot 1: Doanh thu & Lợi nhuận
        fig.add_trace(go.Bar(name='Revenue', x=income_statement.index, y=income_statement['revenue'],
                             marker_color='rgba(0, 141, 0, 1)'), row=1, col=1)
        fig.add_trace(go.Bar(name='Cost of goods sold', x=income_statement.index,
                             y=-income_statement['cost_of_good_sold'], marker_color='rgba(208, 21, 21, 0.5)'),
                      row=1, col=1)

        # Subplot 2: Lợi nhuận gộp & Chi phí hoạt động
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['gross_profit'],
                             name="Gross profit", marker_color='rgba(0, 141, 0, 1)'), row=2, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['operation_expense'],
                             name="Operation expense", marker_color='rgba(208, 21, 21, 0.5)'), row=2, col=1)
        fig.add_trace(go.Scatter(name='Operation Profit', x=income_statement.index,
                                 y=income_statement['operation_profit'], line=dict(width=2),
                                 marker_color='yellow', mode='lines+markers'), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(name='Profit after tax', x=income_statement.index,
                                 y=income_statement['share_holder_income'], line=dict(width=2),
                                 marker_color='purple', mode='lines+markers'), row=2, col=1, secondary_y=True)

        title = f'Revenue, profit, expense - {self.ticker}'
        update_layout_dark(fig, title, yaxis4_title='Value', yaxis3_title='Value')

        fig_name = "revenue_profit_and_expense_yealy.html"
        save_fig(fig, self.ticker, fig_name)

        #fig.show()

    


class Comparision:
    def __init__(self, ticker):
        self.ticker = ticker
        ticker_ratio_df, industry_ratio_df = dap.get_ratio(self.ticker)
        if ticker_ratio_df.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")

        self.ticker_ratio_df, self.industry_ratio_df = ticker_ratio_df.align(industry_ratio_df, join='inner', axis=0)
        self.ticker_ratio_df = self.ticker_ratio_df
        self.industry_ratio_df = self.industry_ratio_df

    def company_roe_roa_comparison(self):
        """
        Vẽ đồ thị so sánh ROE, ROA của công ty và ngành, 
        cùng với tỷ lệ ROE/ROA của công ty trên 3 subplot riêng biệt.
        """
        self.ticker_ratio_df['ratio'] = (self.ticker_ratio_df['roe'] / self.ticker_ratio_df['roa']).replace([np.inf, -np.inf], np.nan)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,  # 3 rows for subplots, shared x-axis
                            subplot_titles=(f"ROE - {self.ticker} vs. industry average", 
                                            f"ROA - {self.ticker} vs. industry average", 
                                            f"ROE/ROA ratio - {self.ticker}"),
                            vertical_spacing=0.1)

        # Subplot 1: ROE comparison
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['roe'],
                                name="Industry's ROE", line=dict(color='#00ffff', width=2), mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['roe'],
                                name="Company's ROE", line=dict(color='#ff6b6b', width=2), mode='lines+markers'), row=1, col=1)

        # Subplot 2: ROA comparison
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['roa'],
                                name="Industry's ROA", line=dict(color='#00ffff', width=2), mode='lines+markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['roa'],
                                name="Company's ROA", line=dict(color='#ff6b6b', width=2), mode='lines+markers'), row=2, col=1)

        # Subplot 3: ROE/ROA ratio
        fig.add_trace(go.Bar(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['ratio'],
                            name='ROE/ROA ratio', marker_color='rgba(1, 147, 178, 0.98)'), row=3, col=1)

        title = f"ROE, ROA comparision - {self.ticker}"
        fig.update_layout(
            title=title,
            font_color='white', plot_bgcolor='black', paper_bgcolor='black', hovermode='x unified',
            hoverlabel=dict(bgcolor="rgba(0, 0, 0, 0.8)", font_size=12),
            yaxis=dict(title="ROE", gridcolor='rgba(128, 128, 128, 0.5)', color='white'),
            yaxis2=dict(title="ROA", gridcolor='rgba(128, 128, 128, 0.5)', color='white'),
            yaxis3=dict(title="ROE/ROA", gridcolor='rgba(128, 128, 128, 0.5)', color='white'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Adjust as needed
        )

        name = "company_roe_roa_comparison.html"
        save_fig(fig, self.ticker, name)
        ##fig.show()


    def company_vs_industry_pe_pb(self):
        """
        Vẽ đồ thị PE, PB của công ty và ngành kèm tỷ lệ PE, PB công ty/ngành 
        trên 2 subplot riêng biệt.
        """
        pe_ratio = (self.ticker_ratio_df['price_to_earning'] / self.industry_ratio_df['price_to_earning']).replace([np.inf, -np.inf], np.nan)
        pb_ratio = (self.ticker_ratio_df['price_to_book'] / self.industry_ratio_df['price_to_book']).replace([np.inf, -np.inf], np.nan)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,  # Shared x-axes
                            subplot_titles=(f"PE - {self.ticker} vs. industry average",
                                            f"PB - {self.ticker} vs. industry average"),
                            vertical_spacing=0.1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

        # Subplot 1: PE Comparison
        fig.add_trace(go.Bar(x=pe_ratio.index, y=pe_ratio, name='Company/Industry PE Ratio',
                             marker_color='rgba(1, 147, 178, 0.98)'), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['price_to_earning'],
                                 name='Industry PE', line=dict(color='#00ffff', width=2), mode='lines+markers'), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['price_to_earning'],
                                 name='Company PE', line=dict(color='#ff6b6b', width=2), mode='lines+markers'), row=1, col=1, secondary_y=True)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1, secondary_y=False)


        # Subplot 2: PB Comparison
        fig.add_trace(go.Bar(x=pb_ratio.index, y=pb_ratio, name='Company/Industry PB Ratio',
                             marker_color='rgba(1, 147, 178, 0.98)'), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['price_to_book'],
                                 name='Industry PB', line=dict(color='#00ffff', width=2), mode='lines+markers'), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['price_to_book'],
                                 name='Company PB', line=dict(color='#ff6b6b', width=2), mode='lines+markers'), row=2, col=1, secondary_y=True)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1, secondary_y=False)

                # Calculate dynamic y-axis range for PE subplot
        min_pe = max(self.ticker_ratio_df['price_to_earning'].min(), self.industry_ratio_df['price_to_earning'].min())
        max_pe = min(self.ticker_ratio_df['price_to_earning'].max(), self.industry_ratio_df['price_to_earning'].max())
        pe_range = [min_pe * 0.95, max_pe * 1.05]  # Add a small buffer (5%)


        # Update layout (common for both subplots)
        title = f"PE, PB comparision- {self.ticker}"
        update_layout_dark(fig, title, yaxis_title="Company/Industry PE Ratio", yaxis2_title="PE Ratio",
                           yaxis3_title="Company/Industry PB Ratio", yaxis4_title="PB Ratio")

        fig.update_layout(yaxis2=dict(range=pe_range)) # Set the y-axis range

        name = "company_vs_industry_pe_pb.html"  # Updated filename
        save_fig(fig, self.ticker, name)
        ##fig.show()


    def equity_ratios_comparison(self):
        """
        Vẽ đồ thị tỷ lệ vốn chủ sở hữu trên nợ và vốn chủ sở hữu trên tổng tài sản 
        của công ty và ngành trên 2 subplot.
        """

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Equity/Debt - {self.ticker} vs Industry average",
                                            f"Equity/Total asset - {self.ticker} vs Industry average"),
                            vertical_spacing=0.1)

        # Subplot 1: Equity/Liability
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['equity_on_liability'], mode='lines+markers',
                                    name='Industry Equity/Debt', line=dict(color='#00ffff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, 
                                    y=self.ticker_ratio_df['equity_on_liability'], mode='lines+markers',
                                    name=f'{self.ticker} Equity/Debt', line=dict(color='#ff6b6b', width=2)), row=1, col=1)

        # Subplot 2: Equity/Total Asset
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['equity_on_total_asset'], mode='lines+markers',
                                    name='Industry Equity/Total asset', line=dict(color='#00ffff', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['equity_on_total_asset'], mode='lines+markers',
                                    name=f'{self.ticker} Equity/Total asset', line=dict(color='#ff6b6b', width=2)), row=2, col=1)

        # Update layout (common for both subplots)
        title = f"Equity ratios comparision - {self.ticker}"
        update_layout_dark(fig, title, yaxis_title="Equity/Debt", yaxis3_title="Industry Equity/Total asset")


        name = "equity_ratios_comparison.html"  # More descriptive file name
        save_fig(fig, self.ticker, name)
        ##fig.show()


class CompanyComparison(Comparision):
    def __init__(self, ticker):
        super().__init__(ticker)


    def profit_margins_comparison(self):
        """
        Vẽ đồ thị so sánh tỷ suất lợi nhuận gộp, tỷ suất lợi nhuận hoạt động,
        và tỷ suất lợi nhuận sau thuế của công ty và ngành.
        """

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Gross profit margin - {self.ticker} vs Industry average",
                                            f"Operation profit margin - {self.ticker} vs Industry average",
                                            f"Post tax margin - {self.ticker} vs Industry average"),
                            vertical_spacing=0.1)

        # Subplot 1: Gross Profit Margin
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['gross_profit_margin'], mode='lines+markers',
                                 name='Industry Gross profit margin', line=dict(color='#00ffff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['gross_profit_margin'], mode='lines+markers',
                                 name=f'Gross profit margin {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=1, col=1)

        # Subplot 2: Operating Profit Margin
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['operating_profit_margin'], mode='lines+markers',
                                 name='Industry Operation profit margin', line=dict(color='#00ffff', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['operating_profit_margin'], mode='lines+markers',
                                 name=f'Operation profit margin {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=2, col=1)

        # Subplot 3: Post Tax Margin
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['post_tax_margin'], mode='lines+markers',
                                 name='Industry Post tax margin', line=dict(color='#00ffff', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['post_tax_margin'], mode='lines+markers',
                                 name=f'Post tax margin {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=3, col=1)

        # Update layout
        title = f"Margin comparision - {self.ticker}"
        update_layout_dark(fig, title, yaxis_title="Gross profit margin", 
                           yaxis3_title="Operation profit margin")
        fig.update_layout(yaxis5=dict(title="Post tax margin", gridcolor='rgba(128, 128, 128, 0)', color='white', overlaying='y', side='right'))

        name = "profit_margins_comparison.html"
        save_fig(fig, self.ticker, name)
        ##fig.show()

    
    def days_inventory_and_days_payable(self):
        """
        Vẽ đồ thị so sánh số ngày tồn kho và số ngày phải trả của công ty và ngành.
        """

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Days inventory - {self.ticker} vs Industry average",
                                            f"Days payable- {self.ticker} vs Industry average"),
                            vertical_spacing=0.1)

        # Subplot 1: Days Inventory
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['days_inventory'], mode='lines+markers',
                                    name='Industry days inventory', line=dict(color='#00ffff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['days_inventory'], mode='lines+markers',
                                    name=f'Days inventory {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=1, col=1)

        # Subplot 2: Days Payable
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['days_payable'], mode='lines+markers',
                                    name='Industry days payable', line=dict(color='#00ffff', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['days_payable'], mode='lines+markers',
                                    name=f'Days payable {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=2, col=1)

        # Update layout
        title = f"Days inventory/payable - {self.ticker}"
        update_layout_dark(fig, title, yaxis_title="Days inventory", yaxis3_title="Days payable")

        name = "days_inventory_and_days_payable.html"
        save_fig(fig, self.ticker, name)
        ##fig.show()
    

class BankVisualization:

    def __init__(self, ticker):
        self.ticker = ticker

    def balance_sheet(self):
        """
        Vẽ biểu đồ bảng cân đối kế toán cho ngân hàng.
        """
        balance_sheet = dap.get_balance_sheet(self.ticker)
        if balance_sheet.empty:
            raise ValueError(f"No balance sheet data available for {self.ticker}")


        fig = make_subplots(rows=2, cols=1, subplot_titles=('Khoản vay & Tiền gửi', 'Tổng quan'),
                            vertical_spacing=0.15, specs=[[{}], [{"secondary_y": True}]])

        # Subplot 1: Customer Loans & Deposits
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['customer_loan'],
                             name='Khoản cho vay khách hàng', marker_color='rgba(0, 141, 0, 1)'), row=1, col=1)
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['deposit'],
                             name='Tiền gửi của khách hàng', marker_color='rgba(208, 21, 21, 0.5)'), row=1, col=1)

        # Subplot 2: Total Assets, Liabilities & Equity
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['asset'],
                             name='Tổng tài sản', marker_color='rgba(0, 141, 0, 1)'), row=2, col=1)
        fig.add_trace(go.Bar(x=balance_sheet.index, y=balance_sheet['debt'],
                             name='Tổng nợ', marker_color='rgba(208, 21, 21, 0.5)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=balance_sheet.index, y=balance_sheet['equity'], mode='lines+markers',
                                 name='Vốn chủ sở hữu', line=dict(width=2), marker_color='yellow'), row=2, col=1, secondary_y=True)

        title = f'Bảng cân đối kế toán - {self.ticker}'
        update_layout_dark(fig, title, yaxis3_title='Value') # Added yaxis3_title

        fig_name = "balance_sheet.html"
        save_fig(fig, self.ticker, fig_name)
        ##fig.show()

    def income_statement(self):
        """
        Vẽ biểu đồ báo cáo kết quả hoạt động kinh doanh cho ngân hàng.
        """
        income_statement = dap.get_income_statement(self.ticker)
        if income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")


        fig = make_subplots(rows=2, cols=1, subplot_titles=('Lợi nhuận & Dự phòng', 'Tổng thu nhập & Chi phí hoạt động'),
                            vertical_spacing=0.15, specs=[[{}], [{"secondary_y": True}]])

        # Subplot 1: Operating Profit & Provision Expense
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['operation_profit'],
                             name='Lợi nhuận hoạt động', marker_color='rgba(0, 141, 0, 1)'), row=1, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['provision_expense'],  # Đảo ngược dấu
                             name='Chi phí dự phòng', marker_color='rgba(208, 21, 21, 0.5)'), row=1, col=1)

        # Subplot 2: Revenue & RALOperating Expenses, Post-tax Profit
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['revenue'],
                             name='Tổng thu nhập', marker_color='rgba(0, 141, 0, 1)'), row=2, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['operation_expense'],  # Đảo ngược dấu
                             name='Chi phí hoạt động', marker_color='rgba(208, 21, 21, 0.5)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=income_statement.index, y=income_statement['post_tax_profit'], mode='lines+markers',
                                 name='Lợi nhuận sau thuế', line=dict(width=2), marker_color='yellow'), row=2, col=1, secondary_y=True)

        title = f'Báo cáo kết quả hoạt động kinh doanh - {self.ticker}'
        update_layout_dark(fig, title,  yaxis3_title='Value') # Added yaxis3_title

        fig_name = "income_statement.html"
        save_fig(fig, self.ticker, fig_name)
        ##fig.show()
    def yearly_income_statement(self):
        """
        Vẽ biểu đồ báo cáo kết quả hoạt động kinh doanh theo năm cho ngân hàng.
        """
        try:
            income_statement = dap.get_income_statement(self.ticker, period='year')
        except ValueError as e:
            print(e)
            return
        if income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Lợi nhuận & Dự phòng', 'Tổng thu nhập & Chi phí hoạt động'),
                            vertical_spacing=0.15, specs=[[{}], [{"secondary_y": True}]])

        # Subplot 1: Operating Profit & Provision Expense
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['operation_profit'],
                             name='Lợi nhuận hoạt động', marker_color='rgba(0, 141, 0, 1)'), row=1, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['provision_expense'],  # Đảo ngược dấu
                             name='Chi phí dự phòng', marker_color='rgba(208, 21, 21, 0.5)'), row=1, col=1)

        # Subplot 2: Revenue & RALOperating Expenses, Post-tax Profit
        fig.add_trace(go.Bar(x=income_statement.index, y=income_statement['revenue'],
                             name='Tổng thu nhập', marker_color='rgba(0, 141, 0, 1)'), row=2, col=1)
        fig.add_trace(go.Bar(x=income_statement.index, y=-income_statement['operation_expense'],  # Đảo ngược dấu
                             name='Chi phí hoạt động', marker_color='rgba(208, 21, 21, 0.5)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=income_statement.index, y=income_statement['post_tax_profit'], mode='lines+markers',
                                 name='Lợi nhuận sau thuế', line=dict(width=2), marker_color='yellow'), row=2,
                        col=1, secondary_y=True)
        title = f'Báo cáo kết quả hoạt động kinh doanh theo năm - {self.ticker}'
        update_layout_dark(fig, title, yaxis3_title='Value')  # Added yaxis3_title
        fig_name = "yearly_income_statement.html"
        save_fig(fig, self.ticker, fig_name)
        #fig.show()

class BankComparision(Comparision):
    def __init__(self, ticker):
        super().__init__(ticker)  # Inherit initialization from parent class

    def interest_margin_and_post_tax_profit_on_toi(self):
        """
        Vẽ biểu đồ so sánh Biên lãi ròng (NIM) và Lợi nhuận sau thuế/Tổng thu nhập hoạt động (TOI)
        của công ty và ngành trên 2 subplot.
        """

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Biên Lãi Thuần (NIM) - {self.ticker} vs Ngành",
                                            f"Lợi Nhuận Sau Thuế/Tổng Thu Nhập Hoạt Động - {self.ticker} vs Ngành"),
                            vertical_spacing=0.1)

        # Subplot 1: Interest Margin (NIM)
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['interest_margin'], mode='lines+markers',
                                 name='Biên Lãi Thuần Ngành', line=dict(color='#00ffff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['interest_margin'], mode='lines+markers',
                                 name=f'Biên Lãi Thuần {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=1, col=1)

        # Subplot 2: Post-tax Profit on TOI
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['post_tax_on_toi'], mode='lines+markers',
                                 name='LNST/Tổng Thu Nhập Hoạt Động Ngành', line=dict(color='#00ffff', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['post_tax_on_toi'], mode='lines+markers',
                                 name=f'LNST/Tổng Thu Nhập Hoạt Động {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=2, col=1)

        # Update layout
        title = f"So sánh Biên Lãi Thuần và LNST/Tổng Thu Nhập Hoạt Động - {self.ticker} vs Ngành"
        update_layout_dark(fig, title, yaxis_title="Biên Lãi Thuần (NIM)", yaxis3_title="LNST/Tổng Thu Nhập Hoạt Động")

        name = "interest_margin_and_post_tax_profit_on_toi.html"
        save_fig(fig, self.ticker, name)
        ##fig.show()

    def bad_debt_and_cancel_debt_ratios(self):
        """
        Vẽ biểu đồ so sánh tỷ lệ nợ xấu và tỷ lệ xóa nợ của công ty và ngành trên 2 subplot.
        """

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Tỷ lệ nợ xấu - {self.ticker} vs Ngành",
                                            f"Tỷ lệ xóa nợ - {self.ticker} vs Ngành"),
                            vertical_spacing=0.1)

        # Subplot 1: Bad Debt Percentage
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['bad_debt_percentage'], mode='lines+markers',
                                 name='Tỷ lệ nợ xấu ngành', line=dict(color='#00ffff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['bad_debt_percentage'], mode='lines+markers',
                                 name=f'Tỷ lệ nợ xấu {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=1, col=1)

        # Subplot 2: Cancel Debt Ratio
        fig.add_trace(go.Scatter(x=self.industry_ratio_df.index, y=self.industry_ratio_df['cancel_debt'], mode='lines+markers',
                                 name='Tỷ lệ xóa nợ ngành', line=dict(color='#00ffff', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.ticker_ratio_df.index, y=self.ticker_ratio_df['cancel_debt'], mode='lines+markers',
                                 name=f'Tỷ lệ xóa nợ {self.ticker}', line=dict(color='#ff6b6b', width=2)), row=2, col=1)


        # Update layout
        title = f"So sánh tỷ lệ nợ xấu và tỷ lệ xóa nợ - {self.ticker} vs Ngành"
        update_layout_dark(fig, title, yaxis_title="Tỷ lệ nợ xấu", yaxis3_title="Tỷ lệ xóa nợ")

        name = "bad_debt_and_cancel_debt_ratios.html"
        save_fig(fig, self.ticker, name)
        ##fig.show()

def plot_all_for_bank(ticker):
    instance = BankVisualization(ticker)
    instance.balance_sheet()
    instance.income_statement()
    instance.yearly_income_statement()

    instance2 = BankComparision(ticker)
    instance2.bad_debt_and_cancel_debt_ratios()
    instance2.interest_margin_and_post_tax_profit_on_toi()
    instance2.company_roe_roa_comparison()
    instance2.company_vs_industry_pe_pb()
    instance2.equity_ratios_comparison()
    print(f'done {ticker}............')


def plot_all_for_company(ticker):
    instance = CompanyVisualization(ticker)
    instance.cash_flow()
    instance.balance_sheet()
    instance.profit_and_expense_plot()
    instance.profit_and_expense_plot_year()

    instance2 = CompanyComparison(ticker)
    instance2.company_roe_roa_comparison()
    instance2.company_vs_industry_pe_pb()
    instance2.equity_ratios_comparison()
    instance2.profit_margins_comparison()
    instance2.days_inventory_and_days_payable()
    print(f'done {ticker}............')

def temp(ticker):
    instance = CompanyVisualization(ticker)
    instance.profit_and_expense_plot_year()
