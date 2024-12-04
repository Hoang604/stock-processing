#  Stock processing and visualization

This project help you examine every stock in Vietnam.


The file data_acquisition_and_processing.py is used for clone data (using [vnstock3 library](https://vnstocks.com/)), calculate industry average metrics, filter stock, and predict future returns.
```python
# Example
from data_acquisition_and_processing import read_industry_average_ratio

# Read the industry average data for the sector that FPT belongs to.
df = read_industry_average_ratio('FPT')
print(df)
```


The file visualization.py is used to visualize company financial statements, including the balance sheet, income statement, and cash flow statement, and to plot company metrics against industry averages.

```python
# Example 
from visualization import plot_all_normal_company, CompanyVisualization

x = CopanyVisualization('FPT')
x.profit_and_expense_plot()
x.balance_sheet()

# plot all chart
plot_all_for_company('FPT')
```
