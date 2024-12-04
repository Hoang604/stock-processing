#  Stock processing and visualization

This project help you examine every stock in Vietnam.


The file data_acquisition_and_processing.py is used for clone data (using [vnstock3 library](https://vnstocks.com/)), calculate industry average metrics, filter stock, and predict future returns.
```python
# Example
from data_acquisition_and_processing import read_industry_average_ratio, predict_future_yeild

# Read the industry average data for the sector that FPT belongs to.
df = read_industry_average_ratio('FPT')
print(df)

# Predict future returns for FPT
pe = get_company_ratio('FPT').price_to_earning.iloc[-1]
grow_rate = get_income_statement('FPT', period='year').year_share_holder_income_growth.tail(3).mean() * 100
print(predict_future_yeild(pe, grow_rate))
```


The file visualization.py is used to visualize company financial statements, including the balance sheet, income statement, and cash flow statement, and to plot company metrics against industry averages.

```python
# Example 
from visualization import plot_all_for_company, CompanyVisualization

x = CopanyVisualization('FPT')
x.profit_and_expense_plot()
x.balance_sheet()

# plot all chart
plot_all_for_company('FPT')
```
