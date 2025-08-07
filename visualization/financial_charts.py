import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_processing.fetch_raw_data import DataFetcher
from data_processing.calculate_metrics import StockDataProcessor
from data_processing.clean_data import StockDataCleaner
from .base_figure import BaseFigure

class FinancialCharts(BaseFigure):
    """
    Simplified financial charts class that uses the universal chart() method.
    This class only provides data - all chart creation is handled by BaseFigure.
    """
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self._company_data = None
        self._industry_data = None
        self._data_processor = StockDataProcessor(ticker)
        self._data_fetcher = DataFetcher(ticker)
        
    def _load_company_data(self) -> None:
        """Load company financial data if not already loaded."""
        if self._company_data is None:
            try:
                print(f"Loading financial data for {self.ticker}...")
                self._company_data, self._industry_data = self._data_processor.get_ratio()
                if not self._company_data.empty and not self._industry_data.empty:
                    # Align the data
                    self._company_data, self._industry_data = self._company_data.align(
                        self._industry_data, join='inner', axis=0
                    )
            except Exception as e:
                print(f"Error loading data for {self.ticker}: {e}")
                self._company_data = pd.DataFrame()
                self._industry_data = pd.DataFrame()
    
    def _load_only_company_data(self) -> None:
        """Load only company financial data if not already loaded."""
        if self._company_data is None:
            try:
                self._company_data = self._data_fetcher.fetch_company_ratio()
                if self._company_data.empty:
                    raise ValueError(f"No data available for {self.ticker}")
            except Exception as e:
                print(f"Error loading company data for {self.ticker}: {e}")
                self._company_data = pd.DataFrame()
    
    def cash_flow_chart(self, period: str = 'quarter') -> None:
        """Create cash flow chart - just provide data to chart() method."""
        df = self._data_fetcher.fetch_company_cash_flow(period=period)
        if df is None or df.empty:
            raise ValueError(f"No cash flow data available for {self.ticker}")
            
        df = StockDataCleaner.clean_cash_flow_df(df.copy())
        
        # Just call chart() with the data
        self.chart(
            filename=f"{'yearly_' if period == 'year' else ''}cash_flow.html",
            main_title=f'Cash flow - {self.ticker}',
            bar_data=df[['operating_cash_flow', 'capEx']],
            bar_columns=['operating_cash_flow', 'capEx'],
            colors=[self.colors['positive'], self.colors['negative']],
            yaxis_title='Value'
        )
    
    def balance_sheet_chart(self) -> None:
        """Create balance sheet chart with 2 subplots."""
        balance_sheet = self._data_fetcher.fetch_company_balance_sheet()
        if balance_sheet is None or balance_sheet.empty:
            raise ValueError(f"No balance sheet data available for {self.ticker}")
        
        # Create the chart with 2 subplots
        self.chart(
            filename='balance_sheet.html',
            main_title=f'Balance sheet - {self.ticker}',
            # Subplot 1: Short term assets and debt
            bar_data_subplot_1=balance_sheet[['short_asset', 'short_debt']],
            bar_columns_subplot_1=['short_asset', 'short_debt'],
            # Subplot 2: Long term assets, debt, and equity (equity as line)
            bar_data_subplot_2=balance_sheet[['asset', 'debt']],
            bar_columns_subplot_2=['asset', 'debt'],
            line_data_subplot_2=balance_sheet[['equity']],
            line_columns_subplot_2=['equity'],
            # Configuration
            subplot_titles=['Short term', 'Long term'],
            secondary_y_subplots=[2],  # Subplot 2 needs secondary y-axis for equity line
            yaxis_title_subplot_1='Tỉ đồng',
            yaxis_title_subplot_2='Tỉ đồng',
            colors_subplot_1=[self.colors['positive'], self.colors['negative']],
            colors_subplot_2=[self.colors['positive'], self.colors['negative'], self.colors['accent']],
            add_hline_subplot_1=True,
            add_hline_subplot_2=True
        )
    
    def income_statement_chart(self, period: str = 'quarter') -> None:
        """Create income statement chart with 2 subplots."""
        income_statement = self._data_fetcher.fetch_company_income_statement(period=period)
        if income_statement is None or income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")
        
        income_statement['cost_of_good_sold'] = - income_statement['cost_of_good_sold']
        income_statement['operation_expense'] = - income_statement['operation_expense']
        
        # Create the chart with 2 subplots
        self.chart(
            filename=f"revenue_profit_and_expense{'_yearly' if period == 'year' else ''}.html",
            main_title=f'Revenue, profit, expense - {self.ticker}',
            # Subplot 1: Revenue and cost of goods sold
            bar_data_subplot_1=income_statement[['revenue', 'cost_of_good_sold']],
            bar_columns_subplot_1=['revenue', 'cost_of_good_sold'],
            # Subplot 2: Gross profit, operation expense (bars) + operation profit, after-tax profit (lines)
            bar_data_subplot_2=income_statement[['gross_profit', 'operation_expense']],
            bar_columns_subplot_2=['gross_profit', 'operation_expense'],
            line_data_subplot_2=income_statement[['operation_profit', 'share_holder_income']],
            line_columns_subplot_2=['operation_profit', 'share_holder_income'],
            # Configuration
            subplot_titles=['Revenue and profit', 'Gross profit and operation expense'],
            # Remove secondary_y_subplots to put everything on same axis
            yaxis_title_subplot_1='Value',
            yaxis_title_subplot_2='Value',
            # Colors will be automatically separated between bars and lines
            bar_colors_subplot_1=[self.colors['positive'], self.colors['negative']],
            bar_colors_subplot_2=[self.colors['positive'], self.colors['negative']],
            line_colors_subplot_2=[self.colors['warning'], self.colors['info']]
        )
    
    def roe_roa_comparison(self) -> None:
        """Create ROE/ROA comparison with 3 subplots."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Calculate ROE/ROA ratio
        roe_roa_ratio = (self._company_data['roe'] / self._company_data['roa']).replace([np.inf, -np.inf], np.nan)
        
        # Prepare comparison data
        roe_comparison = pd.DataFrame({
            'industry_roe': self._industry_data['roe'],
            'company_roe': self._company_data['roe']
        })
        
        roa_comparison = pd.DataFrame({
            'industry_roa': self._industry_data['roa'],
            'company_roa': self._company_data['roa']
        })
        
        ratio_data = pd.DataFrame({
            'roe_roa_ratio': roe_roa_ratio
        })
        
        # Create the chart with 3 subplots
        self.chart(
            filename="company_roe_roa_comparison.html",
            main_title=f"ROE, ROA comparison - {self.ticker}",
            # Subplot 1: ROE comparison (lines)
            line_data_subplot_1=roe_comparison,
            line_columns_subplot_1=['industry_roe', 'company_roe'],
            # Subplot 2: ROA comparison (lines)
            line_data_subplot_2=roa_comparison,
            line_columns_subplot_2=['industry_roa', 'company_roa'],
            # Subplot 3: ROE/ROA ratio (bar)
            bar_data_subplot_3=ratio_data,
            bar_columns_subplot_3=['roe_roa_ratio'],
            # Configuration
            subplot_titles=[
                f"ROE - {self.ticker} vs. industry average",
                f"ROA - {self.ticker} vs. industry average", 
                f"ROE/ROA ratio - {self.ticker}"
            ],
            rows=3,
            yaxis_title_subplot_1='ROE',
            yaxis_title_subplot_2='ROA',
            yaxis_title_subplot_3='ROE/ROA',
            colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            colors_subplot_2=[self.colors['secondary'], self.colors['primary']],
            colors_subplot_3=[self.colors['tertiary']]
        )
    
    def pe_pb_comparison(self) -> None:
        """Create PE/PB comparison with ratios."""
        self._load_company_data()

        if self._company_data.empty:
            raise ValueError(f"No company ratio data available for {self.ticker}")
        if self._industry_data.empty:
            raise ValueError(f"No industry ratio data available for {self.ticker}")
        
        # Calculate ratios
        pe_ratio = (self._company_data['price_to_earning'] / self._industry_data['price_to_earning']).replace([np.inf, -np.inf], np.nan)
        pb_ratio = (self._company_data['price_to_book'] / self._industry_data['price_to_book']).replace([np.inf, -np.inf], np.nan)
        
        # Prepare data for PE subplot
        pe_data_bar = pd.DataFrame({'pe_ratio': pe_ratio})
        pe_data_line = pd.DataFrame({
            'industry_pe': self._industry_data['price_to_earning'],
            'company_pe': self._company_data['price_to_earning']
        })
        
        # Prepare data for PB subplot
        pb_data_bar = pd.DataFrame({'pb_ratio': pb_ratio})
        pb_data_line = pd.DataFrame({
            'industry_pb': self._industry_data['price_to_book'],
            'company_pb': self._company_data['price_to_book']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="company_vs_industry_pe_pb.html",
            main_title=f"PE, PB comparison - {self.ticker}",
            # Subplot 1: PE comparison (bar for ratio + lines for values)
            bar_data_subplot_1=pe_data_bar,
            bar_columns_subplot_1=['pe_ratio'],
            line_data_subplot_1=pe_data_line,
            line_columns_subplot_1=['industry_pe', 'company_pe'],
            # Subplot 2: PB comparison (bar for ratio + lines for values)  
            bar_data_subplot_2=pb_data_bar,
            bar_columns_subplot_2=['pb_ratio'],
            line_data_subplot_2=pb_data_line,
            line_columns_subplot_2=['industry_pb', 'company_pb'],
            # Configuration
            subplot_titles=[
                f"PE - {self.ticker} vs. industry average",
                f"PB - {self.ticker} vs. industry average"
            ],
            secondary_y_subplots=[1, 2],  # Both subplots need secondary y-axis for lines
            yaxis_title_subplot_1='Company/Industry PE Ratio',
            yaxis_title_subplot_2='Company/Industry PB Ratio',
            colors_subplot_1=[self.colors['tertiary'], self.colors['secondary'], self.colors['primary']],
            colors_subplot_2=[self.colors['tertiary'], self.colors['secondary'], self.colors['primary']],
            add_hline_subplot_1=True,
            add_hline_subplot_2=True
        )
    
    def equity_ratios_comparison(self) -> None:
        """Create equity ratios comparison chart."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare data for subplots
        equity_debt_data = pd.DataFrame({
            'industry_equity_debt': self._industry_data['equity_on_liability'],
            'company_equity_debt': self._company_data['equity_on_liability']
        })
        
        equity_asset_data = pd.DataFrame({
            'industry_equity_asset': self._industry_data['equity_on_total_asset'],
            'company_equity_asset': self._company_data['equity_on_total_asset']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="equity_ratios_comparison.html",
            main_title=f"Equity ratios comparison - {self.ticker}",
            # Subplot 1: Equity/Debt comparison
            line_data_subplot_1=equity_debt_data,
            line_columns_subplot_1=['industry_equity_debt', 'company_equity_debt'],
            # Subplot 2: Equity/Total asset comparison
            line_data_subplot_2=equity_asset_data,
            line_columns_subplot_2=['industry_equity_asset', 'company_equity_asset'],
            # Configuration
            subplot_titles=[
                f"Equity/Debt - {self.ticker} vs Industry average",
                f"Equity/Total asset - {self.ticker} vs Industry average"
            ],
            yaxis_title_subplot_1="Equity/Debt",
            yaxis_title_subplot_2="Equity/Total asset",
            colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            colors_subplot_2=[self.colors['secondary'], self.colors['primary']]
        )
    
    def company_pe_pb_only(self) -> None:
        """Create PE and PB charts for the company only."""
        self._load_only_company_data()
        
        if self._company_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare PE data
        pe_data = pd.DataFrame({
            'pe_ratio': self._company_data['price_to_earning']
        })
        
        # Prepare PB data
        pb_data = pd.DataFrame({
            'pb_ratio': self._company_data['price_to_book']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="company_pe_pb.html",
            main_title=f"PE and PB - {self.ticker}",
            # Subplot 1: PE (line)
            line_data_subplot_1=pe_data,
            line_columns_subplot_1=['pe_ratio'],
            # Subplot 2: PB (line)
            line_data_subplot_2=pb_data,
            line_columns_subplot_2=['pb_ratio'],
            # Configuration
            subplot_titles=[
                f"PE - {self.ticker}",
                f"PB - {self.ticker}"
            ],
            yaxis_title_subplot_1='PE Ratio',
            yaxis_title_subplot_2='PB Ratio',
            line_colors_subplot_1=[self.colors['tertiary']],
            line_colors_subplot_2=[self.colors['secondary']]
        )

    def company_roe_roa_only(self) -> None:
        """Create ROE and ROA charts for the company only."""
        self._load_only_company_data()
        
        if self._company_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare ROE data
        roe_data = pd.DataFrame({
            'roe': self._company_data['roe']
        })
        
        # Prepare ROA data
        roa_data = pd.DataFrame({
            'roa': self._company_data['roa']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="company_roe_roa.html",
            main_title=f"ROE and ROA - {self.ticker}",
            # Subplot 1: ROE (line)
            line_data_subplot_1=roe_data,
            line_columns_subplot_1=['roe'],
            # Subplot 2: ROA (line)
            line_data_subplot_2=roa_data,
            line_columns_subplot_2=['roa'],
            # Configuration
            subplot_titles=[
                f"ROE - {self.ticker}",
                f"ROA - {self.ticker}"
            ],
            yaxis_title_subplot_1='ROE',
            yaxis_title_subplot_2='ROA',
            line_colors_subplot_1=[self.colors['secondary']],
            line_colors_subplot_2=[self.colors['primary']]
        )

    def company_equity_ratios_only(self) -> None:
        """Create equity ratios charts for the company only."""
        self._load_only_company_data()
        
        if self._company_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare equity/debt data
        equity_debt_data = pd.DataFrame({
            'equity_on_liability': self._company_data['equity_on_liability']
        })
        
        # Prepare equity/total asset data
        equity_asset_data = pd.DataFrame({
            'equity_on_total_asset': self._company_data['equity_on_total_asset']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="company_equity_ratios.html",
            main_title=f"Equity ratios - {self.ticker}",
            # Subplot 1: Equity/Debt (line)
            line_data_subplot_1=equity_debt_data,
            line_columns_subplot_1=['equity_on_liability'],
            # Subplot 2: Equity/Total asset (line)
            line_data_subplot_2=equity_asset_data,
            line_columns_subplot_2=['equity_on_total_asset'],
            # Configuration
            subplot_titles=[
                f"Equity/Debt - {self.ticker}",
                f"Equity/Total asset - {self.ticker}"
            ],
            yaxis_title_subplot_1='Equity/Debt',
            yaxis_title_subplot_2='Equity/Total asset',
            line_colors_subplot_1=[self.colors['tertiary']],
            line_colors_subplot_2=[self.colors['secondary']]
        )

    def days_inventory_and_payable_chart(self) -> None:
        """Create days inventory and days payable comparison chart."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare data for subplots
        days_inventory_data = pd.DataFrame({
            'industry_days_inventory': self._industry_data['days_inventory'],
            'company_days_inventory': self._company_data['days_inventory']
        })
        
        days_payable_data = pd.DataFrame({
            'industry_days_payable': self._industry_data['days_payable'],
            'company_days_payable': self._company_data['days_payable']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="days_inventory_and_payable.html",
            main_title=f"Days Inventory and Payable - {self.ticker}",
            # Subplot 1: Days inventory comparison
            line_data_subplot_1=days_inventory_data,
            line_columns_subplot_1=['industry_days_inventory', 'company_days_inventory'],
            # Subplot 2: Days payable comparison
            line_data_subplot_2=days_payable_data,
            line_columns_subplot_2=['industry_days_payable', 'company_days_payable'],
            # Configuration
            subplot_titles=[
                f"Days Inventory - {self.ticker} vs Industry average",
                f"Days Payable - {self.ticker} vs Industry average"
            ],
            yaxis_title_subplot_1="Days",
            yaxis_title_subplot_2="Days",
            line_colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            line_colors_subplot_2=[self.colors['secondary'], self.colors['primary']]
        )

    def profit_margins_comparison(self) -> None:
        """Create profit margins comparison chart with 3 subplots."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare data for subplots
        gross_margin_data = pd.DataFrame({
            'industry_gross_margin': self._industry_data['gross_profit_margin'],
            'company_gross_margin': self._company_data['gross_profit_margin']
        })
        
        operating_margin_data = pd.DataFrame({
            'industry_operating_margin': self._industry_data['operating_profit_margin'],
            'company_operating_margin': self._company_data['operating_profit_margin']
        })
        
        post_tax_margin_data = pd.DataFrame({
            'industry_post_tax_margin': self._industry_data['post_tax_margin'],
            'company_post_tax_margin': self._company_data['post_tax_margin']
        })
        
        # Create the chart with 3 subplots
        self.chart(
            filename="profit_margins_comparison.html",
            main_title=f"Profit Margins Comparison - {self.ticker}",
            # Subplot 1: Gross profit margin comparison
            line_data_subplot_1=gross_margin_data,
            line_columns_subplot_1=['industry_gross_margin', 'company_gross_margin'],
            # Subplot 2: Operating profit margin comparison
            line_data_subplot_2=operating_margin_data,
            line_columns_subplot_2=['industry_operating_margin', 'company_operating_margin'],
            # Subplot 3: Post tax margin comparison
            line_data_subplot_3=post_tax_margin_data,
            line_columns_subplot_3=['industry_post_tax_margin', 'company_post_tax_margin'],
            # Configuration
            subplot_titles=[
                f"Gross Profit Margin - {self.ticker} vs Industry average",
                f"Operating Profit Margin - {self.ticker} vs Industry average",
                f"Post Tax Margin - {self.ticker} vs Industry average"
            ],
            rows=3,
            yaxis_title_subplot_1="Gross Margin %",
            yaxis_title_subplot_2="Operating Margin %", 
            yaxis_title_subplot_3="Post Tax Margin %",
            line_colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            line_colors_subplot_2=[self.colors['secondary'], self.colors['primary']],
            line_colors_subplot_3=[self.colors['secondary'], self.colors['primary']]
        )

    def company_days_inventory_and_payable_only(self) -> None:
        """Create days inventory and days payable charts for the company only."""
        self._load_only_company_data()
        
        if self._company_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare days inventory data
        days_inventory_data = pd.DataFrame({
            'days_inventory': self._company_data['days_inventory']
        })
        
        # Prepare days payable data
        days_payable_data = pd.DataFrame({
            'days_payable': self._company_data['days_payable']
        })
        
        # Create the chart with 2 subplots
        self.chart(
            filename="company_days_inventory_and_payable.html",
            main_title=f"Days Inventory and Payable - {self.ticker}",
            # Subplot 1: Days inventory (line)
            line_data_subplot_1=days_inventory_data,
            line_columns_subplot_1=['days_inventory'],
            # Subplot 2: Days payable (line)
            line_data_subplot_2=days_payable_data,
            line_columns_subplot_2=['days_payable'],
            # Configuration
            subplot_titles=[
                f"Days Inventory - {self.ticker}",
                f"Days Payable - {self.ticker}"
            ],
            yaxis_title_subplot_1='Days',
            yaxis_title_subplot_2='Days',
            line_colors_subplot_1=[self.colors['info']],
            line_colors_subplot_2=[self.colors['warning']]
        )

    def company_profit_margins_only(self) -> None:
        """Create profit margins charts for the company only."""
        self._load_only_company_data()
        
        if self._company_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Prepare profit margin data
        gross_margin_data = pd.DataFrame({
            'gross_profit_margin': self._company_data['gross_profit_margin']
        })
        
        operating_margin_data = pd.DataFrame({
            'operating_profit_margin': self._company_data['operating_profit_margin']
        })
        
        post_tax_margin_data = pd.DataFrame({
            'post_tax_margin': self._company_data['post_tax_margin']
        })
        
        # Create the chart with 3 subplots
        self.chart(
            filename="company_profit_margins.html",
            main_title=f"Profit Margins - {self.ticker}",
            # Subplot 1: Gross profit margin (line)
            line_data_subplot_1=gross_margin_data,
            line_columns_subplot_1=['gross_profit_margin'],
            # Subplot 2: Operating profit margin (line)
            line_data_subplot_2=operating_margin_data,
            line_columns_subplot_2=['operating_profit_margin'],
            # Subplot 3: Post tax margin (line)
            line_data_subplot_3=post_tax_margin_data,
            line_columns_subplot_3=['post_tax_margin'],
            # Configuration
            subplot_titles=[
                f"Gross Profit Margin - {self.ticker}",
                f"Operating Profit Margin - {self.ticker}",
                f"Post Tax Margin - {self.ticker}"
            ],
            rows=3,
            yaxis_title_subplot_1='Gross Margin %',
            yaxis_title_subplot_2='Operating Margin %',
            yaxis_title_subplot_3='Post Tax Margin %',
            line_colors_subplot_1=[self.colors['success']],
            line_colors_subplot_2=[self.colors['info']],
            line_colors_subplot_3=[self.colors['warning']]
        )

    def company_inventory(self) -> None:
        """Create inventory chart for the company only."""
        balance_sheet = self._data_fetcher.fetch_company_balance_sheet()
        if balance_sheet is None or balance_sheet.empty:
            raise ValueError(f"No balance sheet data available for {self.ticker}")
        
        # Prepare inventory data
        inventory_data = pd.DataFrame({
            'inventory': balance_sheet['inventory']
        })
        
        # Create the chart
        self.chart(
            filename="company_inventory.html",
            main_title=f"Inventory - {self.ticker}",
            line_data=inventory_data,
            line_columns=['inventory'],
            yaxis_title='Inventory Value',
            line_colors=[self.colors['primary']]
        )

    def plot_all_charts(self) -> None:
        """Plot all financial charts for the company."""
        self.cash_flow_chart()
        self.balance_sheet_chart()
        self.income_statement_chart()
        self.roe_roa_comparison()
        self.pe_pb_comparison()
        self.equity_ratios_comparison()
        self.company_pe_pb_only()
        self.company_roe_roa_only()
        self.company_equity_ratios_only()
        self.days_inventory_and_payable_chart()
        self.profit_margins_comparison()
        self.company_days_inventory_and_payable_only()
        self.company_profit_margins_only()
        self.company_inventory()
        self.income_statement_chart(period='year')


class BankFinancialCharts(BaseFigure):
    """
    Specialized financial charts class for banks using the universal chart() method.
    Banks have different financial metrics and require different visualizations.
    """
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self._company_data = None
        self._industry_data = None
        self._data_processor = StockDataProcessor(ticker)
        self._data_fetcher = DataFetcher(ticker)
        
    def _load_company_data(self) -> None:
        """Load company financial data if not already loaded."""
        if self._company_data is None:
            try:
                print(f"Loading bank financial data for {self.ticker}...")
                self._company_data, self._industry_data = self._data_processor.get_ratio()
                if not self._company_data.empty and not self._industry_data.empty:
                    # Align the data
                    self._company_data, self._industry_data = self._company_data.align(
                        self._industry_data, join='inner', axis=0
                    )
            except Exception as e:
                print(f"Error loading bank data for {self.ticker}: {e}")
                self._company_data = pd.DataFrame()
                self._industry_data = pd.DataFrame()
    
    def balance_sheet_chart(self) -> None:
        """Create bank balance sheet chart with deposits, loans, and equity."""
        balance_sheet = self._data_fetcher.fetch_company_balance_sheet()
        if balance_sheet is None or balance_sheet.empty:
            raise ValueError(f"No balance sheet data available for {self.ticker}")
        
        # Bank balance sheet focuses on deposits, loans, and equity
        # Check which columns are available
        available_cols = balance_sheet.columns.tolist()
        
        # Map common bank balance sheet items
        asset_cols = [col for col in ['asset', 'total_asset', 'assets'] if col in available_cols]
        debt_cols = [col for col in ['debt', 'total_debt', 'liability', 'liabilities'] if col in available_cols]
        equity_cols = [col for col in ['equity', 'stockholder_equity', 'shareholders_equity'] if col in available_cols]
        
        if not asset_cols or not debt_cols or not equity_cols:
            print(f"Missing required columns. Available: {available_cols}")
            raise ValueError(f"Required balance sheet columns not found for {self.ticker}")
        
        # Create the chart
        self.chart(
            filename='balance_sheet.html',
            main_title=f'Bank Balance Sheet - {self.ticker}',
            # Use available columns
            bar_data=balance_sheet[[asset_cols[0], debt_cols[0]]],
            bar_columns=[asset_cols[0], debt_cols[0]],
            line_data=balance_sheet[[equity_cols[0]]],
            line_columns=[equity_cols[0]],
            # Configuration
            bar_colors=[self.colors['positive'], self.colors['negative']],
            line_colors=[self.colors['accent']],
            yaxis_title='Billion VND',
            add_hline=True
        )
    
    def income_statement_chart(self, period: str = 'quarter') -> None:
        """Create bank income statement chart focusing on interest and non-interest income."""
        income_statement = self._data_fetcher.fetch_company_income_statement(period=period)
        if income_statement is None or income_statement.empty:
            raise ValueError(f"No income statement data available for {self.ticker}")
        
        # Check available columns
        available_cols = income_statement.columns.tolist()
        
        # Map bank-specific income statement items
        # revenue
        revenue_cols = [col for col in ['revenue', 'total_revenue', 'net_interest_income', 'interest_income'] if col in available_cols]
        # operating_expense
        expense_cols = [col for col in ['operation_expense', 'operating_expense', 'total_expense'] if col in available_cols]
        # share_holder_income
        profit_cols = [col for col in ['net_income', 'profit_after_tax', 'share_holder_income'] if col in available_cols]
        
        if not revenue_cols:
            raise ValueError(f"No revenue columns found for {self.ticker}")
        
        if revenue_cols and expense_cols:
            bar_columns = [revenue_cols[0], expense_cols[0]]
        
        income_statement[expense_cols[0]] = - income_statement[expense_cols[0]]  # Make expenses positive for visualization
        
        if profit_cols:
            line_columns = [profit_cols[0]]
        
        # Create the chart
        self.chart(
            filename=f"bank_income_statement{'_yearly' if period == 'year' else ''}.html",
            main_title=f'Bank Income Statement - {self.ticker}',
            yaxis_title='Billion VND',
            bar_colors=[self.colors['positive'], self.colors['negative']],
            line_colors=[self.colors['accent']],
            bar_data=income_statement[bar_columns],
            bar_columns=bar_columns,
            line_data=income_statement[line_columns],
            line_column=line_columns
        )
    
    def interest_margin_and_profit_chart(self) -> None:
        """Create chart showing interest margin and profit ratios for banks."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Look for bank-specific ratios
        available_cols = self._company_data.columns.tolist()
        
        # Common bank ratios
        margin_cols = [col for col in ['net_interest_margin', 'interest_margin', 'nim'] if col in available_cols]
        profit_cols = [col for col in ['roe', 'roa', 'return_on_equity', 'return_on_assets'] if col in available_cols]
        
        if not margin_cols and not profit_cols:
            print("No suitable bank ratios found, using standard financial ratios")
            profit_cols = ['roe', 'roa'] if 'roe' in available_cols and 'roa' in available_cols else []
        
        chart_data = {}
        subplot_titles = []
        
        # Subplot 1: Interest margin comparison (if available)
        if margin_cols:
            margin_col = margin_cols[0]
            margin_comparison = pd.DataFrame({
                f'industry_{margin_col}': self._industry_data.get(margin_col, pd.Series()),
                f'company_{margin_col}': self._company_data.get(margin_col, pd.Series())
            }).dropna()
            
            if not margin_comparison.empty:
                chart_data['line_data_subplot_1'] = margin_comparison
                chart_data['line_columns_subplot_1'] = list(margin_comparison.columns)
                subplot_titles.append(f"Interest Margin - {self.ticker} vs Industry")
        
        # Subplot 2: Profitability ratios
        if profit_cols:
            profit_col = profit_cols[0]
            profit_comparison = pd.DataFrame({
                f'industry_{profit_col}': self._industry_data.get(profit_col, pd.Series()),
                f'company_{profit_col}': self._company_data.get(profit_col, pd.Series())
            }).dropna()
            
            if not profit_comparison.empty:
                subplot_num = 2 if margin_cols else 1
                chart_data[f'line_data_subplot_{subplot_num}'] = profit_comparison
                chart_data[f'line_columns_subplot_{subplot_num}'] = list(profit_comparison.columns)
                subplot_titles.append(f"{profit_col.upper()} - {self.ticker} vs Industry")
        
        if not chart_data:
            raise ValueError(f"No suitable data found for bank charts for {self.ticker}")
        
        # Create the chart
        self.chart(
            filename="bank_interest_margin_and_profit.html",
            main_title=f"Bank Performance Metrics - {self.ticker}",
            subplot_titles=subplot_titles,
            yaxis_title_subplot_1='Margin %' if margin_cols else 'Ratio',
            yaxis_title_subplot_2='Ratio' if len(subplot_titles) > 1 else None,
            line_colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            line_colors_subplot_2=[self.colors['secondary'], self.colors['primary']] if len(subplot_titles) > 1 else None,
            **chart_data
        )
    
    def bad_debt_ratios_chart(self) -> None:
        """Create chart showing bad debt ratios for banks."""
        self._load_company_data()
        
        if self._company_data.empty or self._industry_data.empty:
            raise ValueError(f"No ratio data available for {self.ticker}")
        
        # Look for bad debt related ratios
        available_cols = self._company_data.columns.tolist()
        bad_debt_cols = [col for col in available_cols if 'debt' in col.lower() and any(word in col.lower() for word in ['bad', 'loss', 'impairment'])]
        provision_for_bad_debt_cols = [col for col in available_cols if 'provision' in col.lower() and 'bad' in col.lower()]


        
        if not bad_debt_cols:
            # Fallback to common debt ratios
            debt_ratios = [col for col in ['debt_to_equity', 'debt_ratio', 'leverage'] if col in available_cols]
            if debt_ratios:
                bad_debt_cols = debt_ratios[:2]
        
        if not bad_debt_cols:
            print(f"No bad debt or debt ratio columns found for {self.ticker}")
            print(f"Available columns: {available_cols}")
            raise ValueError(f"No bad debt data available for {self.ticker}")
        
        bad_debt_data = pd.DataFrame()
        bad_debt_data[f'industry_{bad_debt_cols[0]}'] = self._industry_data.get(bad_debt_cols[0], pd.Series())
        bad_debt_data[f'company_{bad_debt_cols[0]}'] = self._company_data.get(bad_debt_cols[0], pd.Series())

        provision_data = pd.DataFrame()
        provision_data[f'industry_{provision_for_bad_debt_cols[0]}'] = self._industry_data.get(provision_for_bad_debt_cols[0], pd.Series())
        provision_data[f'company_{provision_for_bad_debt_cols[0]}'] = self._company_data.get(provision_for_bad_debt_cols[0], pd.Series())

        # Create the chart
        self.chart(
            filename="bank_bad_debt_ratios.html",
            main_title=f"Bank Debt Ratios - {self.ticker}",
            subplot_titles=[
                f"Bad Debt Ratio - {self.ticker} vs Industry",
                f"Provision for Bad Debt - {self.ticker} vs Industry"
            ],
            # Subplot 1: Bad debt ratio (line)
            line_data_subplot_1=bad_debt_data,
            line_columns_subplot_1=list(bad_debt_data.columns),
            # Subplot 2: Provision for bad debt (line)
            line_data_subplot_2=provision_data,
            line_columns_subplot_2=list(provision_data.columns),
            # Configuration
            yaxis_title_subplot_1='Bad Debt Ratio',
            yaxis_title_subplot_2='Provision Ratio',
            colors_subplot_1=[self.colors['secondary'], self.colors['primary']],
            colors_subplot_2=[self.colors['secondary'], self.colors['primary']],
            add_hline_subplot_1=True,
            add_hline_subplot_2=True
        )
