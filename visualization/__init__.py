"""
Stock Processing Visualization Module

This module provides base classes and specialized financial chart creation tools
that eliminate code duplication and provide consistent styling across all charts.

Classes:
    BaseFigure: Base class for creating standardized charts
    FinancialCharts: Specialized class for financial analysis charts
    BankFinancialCharts: Specialized class for bank financial analysis charts
"""

from .base_figure import BaseFigure
from .financial_charts import FinancialCharts, BankFinancialCharts

__all__ = ['BaseFigure', 'FinancialCharts', 'BankFinancialCharts']
