from visualization.financial_charts import FinancialCharts, BankFinancialCharts

ticker = 'HPG'

charts = FinancialCharts(ticker)
charts.income_statement_chart(period='year')
charts.income_statement_chart()