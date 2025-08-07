# 📈 Stock Processing and Visualization

> A comprehensive Python toolkit for analyzing Vietnamese stock market data with professional-grade financial visualizations.

## 🚀 Overview

This project provides powerful tools to examine every stock in the Vietnamese market with:
- **Universal Chart System**: Create any combination of financial charts with a single method
- **Automated Data Processing**: Fetch, clean, and calculate financial metrics automatically  
- **Industry Comparisons**: Compare company performance against industry averages
- **Bank-Specific Analysis**: Specialized visualizations for banking sector
- **Outlier-Resistant Charts**: Smart y-axis scaling that handles extreme values

## 📊 Features

### **Core Capabilities**
- ✅ **Financial Statement Analysis**: Balance sheet, income statement, cash flow
- ✅ **Ratio Analysis**: PE, PB, ROE, ROA, profit margins, equity ratios
- ✅ **Industry Benchmarking**: Compare against sector averages
- ✅ **Bank Analytics**: Specialized charts for banking metrics
- ✅ **Trend Analysis**: Historical performance tracking
- ✅ **Data Quality**: Automatic outlier detection and handling

### **Visualization System**
- 🎨 **Universal Chart Method**: One method creates any chart type
- 🎯 **Smart Scaling**: Automatic y-axis optimization for small values and outliers
- 🌙 **Dark Theme**: Professional dark theme with color separation
- 📱 **Interactive Charts**: Hover details and zoom capabilities
- 🏦 **Dual Systems**: Separate classes for normal companies and banks

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Hoang604/stock-processing.git
cd stock-processing

# Install dependencies
pip install vnstock plotly pandas numpy
```

## 📚 Quick Start

### **Normal Company Analysis**
```python
from visualization.financial_charts import FinancialCharts

# Initialize charts for a company
charts = FinancialCharts('VNM')  # Vinamilk

# Financial statements
charts.cash_flow_chart(period='quarter')
charts.balance_sheet_chart()
charts.income_statement_chart(period='year')

# Performance analysis
charts.roe_roa_comparison()
charts.pe_pb_comparison()
charts.equity_ratios_comparison()

# Additional metrics
charts.profit_margins_comparison()
charts.days_inventory_and_payable_chart()

# Company-only charts (no industry comparison)
charts.company_pe_pb_only()
charts.company_roe_roa_only()
```

### **Bank Analysis**
```python
from visualization.financial_charts import BankFinancialCharts

# Initialize bank-specific charts
bank_charts = BankFinancialCharts('VCB')  # Vietcombank

# Bank financial statements
bank_charts.balance_sheet_chart()
bank_charts.income_statement_chart(period='quarter')
bank_charts.income_statement_chart(period='year')

# Bank-specific metrics
bank_charts.interest_margin_and_profit_chart()
bank_charts.bad_debt_ratios_chart()
```

### **Data Processing**
```python
from data_processing.fetch_raw_data import DataFetcher
from data_processing.calculate_metrics import StockDataProcessor

# Fetch raw financial data
fetcher = DataFetcher('FPT')
balance_sheet = fetcher.fetch_company_balance_sheet()
income_statement = fetcher.fetch_company_income_statement()

# Calculate financial ratios
processor = StockDataProcessor('FPT')
company_ratios, industry_ratios = processor.get_ratio()
```

## 🏗️ Project Structure

```
stock_processing/
├── data_processing/           # Data fetching and processing
│   ├── fetch_raw_data.py     # Data acquisition from vnstock
│   ├── calculate_metrics.py  # Financial ratio calculations
│   └── clean_data.py         # Data cleaning utilities
├── visualization/            # Chart creation system
│   ├── base_figure.py        # Universal chart creator
│   ├── financial_charts.py   # Main chart classes
│   └── __init__.py          # Module exports
├── data/                     # Cached data and results
├── picture/                  # Generated chart outputs
│   ├── Bank/                # Bank company charts
│   └── Normal_company/      # Regular company charts
└── main.py                  # Example usage
```

## 🎯 Universal Chart System

The core innovation is the universal `chart()` method that can create any financial visualization:

```python
# Single chart with mixed data types
charts.chart(
    filename="mixed_chart.html",
    main_title="Revenue and Profit Analysis",
    bar_data=df[['revenue', 'cost']],           # Bar chart data
    line_data=df[['profit', 'margin']],         # Line chart data
    yaxis_title="Value (Billion VND)"
)

# Multiple subplots with different metrics
charts.chart(
    filename="comprehensive_analysis.html",
    main_title="Full Financial Analysis",
    # Subplot 1: Revenue trends
    line_data_subplot_1=revenue_df,
    # Subplot 2: Balance sheet
    bar_data_subplot_2=balance_df,
    line_data_subplot_2=equity_df,
    # Configuration
    subplot_titles=["Revenue Trends", "Balance Sheet"],
    secondary_y_subplots=[2]  # Equity on secondary axis
)
```

## 📈 Smart Features

### **Outlier Handling**
Automatically detects and handles outliers using statistical methods:
- Uses IQR (Interquartile Range) for outlier detection
- Focuses charts on 5th-95th percentile range when outliers present
- Maintains data integrity while improving visualization clarity

### **Automatic Scaling**
- **Small values** (< 1): Enhanced precision for ratios like 0.01
- **Normal ranges**: Optimal padding and scaling
- **With outliers**: Statistical filtering for better insights

### **Color System**
Consistent color coding across all charts:
- 🟢 **Positive/Assets**: Green tones
- 🔴 **Negative/Liabilities**: Red tones  
- 🔵 **Company data**: Blue tones
- 🟡 **Industry data**: Cyan tones

## 🏦 Bank vs Normal Company

### **Normal Companies**
- Revenue, cost, profit analysis
- Asset, liability, equity tracking
- Standard financial ratios (PE, PB, ROE, ROA)
- Inventory and payable cycles
- Profit margin analysis

### **Banks**
- Interest income and margin analysis
- Deposit and loan portfolio tracking
- Bad debt and provision ratios
- Bank-specific performance metrics
- Regulatory ratio monitoring

## 📊 Output Examples

All charts are saved as interactive HTML files in organized directories:
```
picture/
├── Bank/VCB/
│   ├── balance_sheet.html
│   ├── bank_income_statement.html
│   └── bank_bad_debt_ratios.html
└── Normal_company/VNM/
    ├── balance_sheet.html
    ├── revenue_profit_and_expense.html
    └── company_roe_roa_comparison.html
```

## 🔧 Advanced Usage

### **Custom Chart Creation**
```python
# Create custom financial analysis
charts.chart(
    filename="custom_analysis.html",
    main_title="Custom Financial Metrics",
    line_data=custom_df,
    line_columns=['custom_ratio_1', 'custom_ratio_2'],
    bar_data=another_df,
    bar_columns=['metric_a', 'metric_b'],
    line_colors=['#ff6b6b', '#4ecdc4'],
    bar_colors=['#45b7d1', '#96ceb4'],
    yaxis_title="Custom Metrics"
)
```

### **Data Processing Pipeline**
```python
from data_processing import DataFetcher, StockDataProcessor, StockDataCleaner

# Complete data pipeline
fetcher = DataFetcher('ticker')
processor = StockDataProcessor('ticker')
cleaner = StockDataCleaner()

# Fetch -> Process -> Clean -> Visualize
raw_data = fetcher.fetch_company_cash_flow()
cleaned_data = cleaner.clean_cash_flow_df(raw_data)
ratios = processor.calculate_financial_ratios(cleaned_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **vnstock**: Vietnamese stock data API
- **Plotly**: Interactive visualization library
- **pandas**: Data manipulation and analysis
