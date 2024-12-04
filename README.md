#  Stock processing and visualization

This project help you examine every stock in Vietnam.


The file data_acquisition_and_processing.py is used for clone data (using [vnstock3 library](https://vnstocks.com/)), calculate industry average metrics, filter stock, and predict future returns.


The file visualization.py is used to visualize company financial statements, including the balance sheet, income statement, and cash flow statement, and to plot company metrics against industry averages.

```python
# Example 
from visualization import plot_all_normal_company
plot_all_for_company('FPT')
