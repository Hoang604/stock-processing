import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_processing.fetch_raw_data import DataFetcher

class BaseFigure:
    """
    Universal chart creator that can handle any combination of line/bar charts 
    across multiple subplots with a single method call.
    """
    
    def __init__(self, ticker: str, theme: str = 'dark'):
        """
        Initialize the base figure with ticker and theme.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        theme : str
            Chart theme ('dark' or 'light')
        """
        self.ticker = ticker
        self.theme = theme
        self.colors = {
            'positive': 'rgba(0, 141, 0, 1)',
            'negative': 'rgba(208, 21, 21, 0.4)',
            'primary': '#ff6b6b',
            'secondary': '#00ffff',
            'tertiary': 'rgba(1, 147, 178, 0.98)',
            'accent': '#ff9800',
            'warning': '#ffeb3b',
            'info': '#9c27b0',
            'success': '#4caf50',
            'error': '#f44336'
        }
        
    def _get_dark_layout(self, title: str, subplot_titles: List[str] = None) -> Dict:
        """Get standardized dark theme layout."""
        layout = {
            'title': title,
            'font_color': 'white',
            'plot_bgcolor': 'black',
            'paper_bgcolor': 'black',
            'hovermode': 'x unified',
            'hoverlabel': dict(bgcolor="rgba(0, 0, 0, 0.8)", font_size=12),
            'legend': dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        return layout
    
    def _get_optimal_y_range(self, data_values):
        """Calculate optimal y-axis range that handles outliers for better visualization."""
        if not data_values or len(data_values) == 0:
            return None
        
        import numpy as np
        
        # Convert to numpy array and remove NaN/inf values
        data_array = np.array([x for x in data_values if not (np.isnan(x) or np.isinf(x))])
        
        if len(data_array) == 0:
            return None
            
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        # Calculate percentiles to detect outliers
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        median = np.median(data_array)
        
        # Define outlier bounds using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if we have significant outliers
        has_outliers = (min_val < lower_bound) or (max_val > upper_bound)
        
        if has_outliers:
            # Use a more conservative range based on percentiles
            # Use 5th and 95th percentiles to exclude extreme outliers
            range_min = np.percentile(data_array, 5)
            range_max = np.percentile(data_array, 95)
            
            # Add some padding
            padding = (range_max - range_min) * 0.1
            final_min = range_min - padding
            final_max = range_max + padding
            
            return [final_min, final_max]
        
        # For small values (< 1), use the original logic
        elif max_val < 1 and max_val > 0:
            padding = (max_val - min_val) * 0.1 if max_val != min_val else max_val * 0.1
            return [max(0, min_val - padding), max_val + padding]
        
        # For normal data without outliers, let plotly auto-scale
        return None
    
    def _format_column_name(self, col_name: str) -> str:
        """Format column name for display."""
        return col_name.replace('_', ' ').title()
    
    def _should_negate_values(self, col_name: str) -> bool:
        """Check if values should be negated for expense-type columns."""
        # Disable automatic negation - user wants to see actual values for comparison
        return False
    
    def _save_figure(self, fig: go.Figure, filename: str) -> None:
        """Save figure to appropriate directory."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        try:
            data_fetcher = DataFetcher(self.ticker)
            icb_code = data_fetcher.fetch_ticker_icb_code()
            folder = 'Bank' if icb_code == 8350 else 'Normal_company'
            save_dir = os.path.join(base_dir, 'picture', folder, self.ticker)
        except:
            save_dir = os.path.join(base_dir, 'picture', 'Normal_company', self.ticker)
            
        os.makedirs(save_dir, exist_ok=True)
        fig.write_html(os.path.join(save_dir, filename))
        print(f"Chart saved: {folder}/{filename}")
    
    def chart(self, filename: str, main_title: str = None, **kwargs) -> go.Figure:
        """
        Universal chart method that creates any combination of line/bar charts 
        across multiple subplots based on provided data.
        
        Parameters:
        -----------
        filename : str
            Filename to save the chart
        main_title : str, optional
            Main chart title (defaults to ticker-based title)
        **kwargs : dict
            Chart data and configuration in the format:
            
            For single plots:
            - line_data : pd.DataFrame - Data for line chart
            - bar_data : pd.DataFrame - Data for bar chart
            - line_columns : List[str] - Columns to plot as lines
            - bar_columns : List[str] - Columns to plot as bars
            - line_colors : List[str] - Custom colors for lines
            - bar_colors : List[str] - Custom colors for bars
            
            For subplots:
            - line_data_subplot_1 : pd.DataFrame - Line data for subplot 1
            - bar_data_subplot_1 : pd.DataFrame - Bar data for subplot 1
            - line_data_subplot_2 : pd.DataFrame - Line data for subplot 2
            - bar_data_subplot_2 : pd.DataFrame - Bar data for subplot 2
            - ... (continue for more subplots)
            - subplot_titles : List[str] - Titles for each subplot
            - rows : int - Number of rows (default: auto-calculated)
            - cols : int - Number of columns (default: 1)
            - secondary_y_subplots : List[int] - Subplots that can have secondary y-axis
            - line_secondary_y_subplot_N : bool - Put lines on secondary y-axis for subplot N
            - bar_secondary_y_subplot_N : bool - Put bars on secondary y-axis for subplot N
            - line_colors_subplot_N : List[str] - Custom line colors for subplot N
            - bar_colors_subplot_N : List[str] - Custom bar colors for subplot N
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Created and saved figure
            
        Examples:
        ---------
        # Single line chart
        chart(filename="simple_line.html", line_data=df, line_columns=['revenue', 'profit'])
        
        # Single bar chart  
        chart(filename="simple_bar.html", bar_data=df, bar_columns=['assets', 'debt'])
        
        # Mixed single chart
        chart(filename="mixed.html", line_data=df1, line_columns=['roe'], 
              bar_data=df2, bar_columns=['revenue'])
        
        # Multiple subplots (all on same axis)
        chart(filename="subplots.html",
              line_data_subplot_1=df1, line_columns_subplot_1=['roe', 'roa'],
              bar_data_subplot_2=df2, bar_columns_subplot_2=['revenue', 'cost'],
              subplot_titles=['ROE vs ROA', 'Revenue vs Cost'])
              
        # Multiple subplots with specific secondary y-axis control
        chart(filename="subplots_secondary.html",
              bar_data_subplot_1=df1, bar_columns_subplot_1=['assets', 'debt'],
              line_data_subplot_1=df2, line_columns_subplot_1=['equity'],
              secondary_y_subplots=[1],
              line_secondary_y_subplot_1=True,  # Only lines on secondary y-axis
              subplot_titles=['Balance Sheet'])
        """
        
        # Set default title
        if main_title is None:
            main_title = f"Financial Analysis - {self.ticker}"
        
        # Detect if this is a subplot chart
        subplot_keys = [k for k in kwargs.keys() if 'subplot_' in k]
        is_subplot = len(subplot_keys) > 0
        
        if is_subplot:
            return self._create_subplot_chart(filename, main_title, **kwargs)
        else:
            return self._create_single_chart(filename, main_title, **kwargs)
    
    def _create_single_chart(self, filename: str, title: str, **kwargs) -> go.Figure:
        """Create a single chart (line and/or bar)."""
        fig = go.Figure()
        
        # Handle line data
        line_data = kwargs.get('line_data')
        line_columns = kwargs.get('line_columns', [])
        if line_data is not None and not line_data.empty:
            if not line_columns:  # Use all columns if none specified
                line_columns = line_data.columns.tolist()
            
            # Use line-specific colors
            line_colors = kwargs.get('line_colors', [self.colors['warning'], self.colors['info'], 
                                   self.colors['accent'], self.colors['tertiary']])
            
            for i, col in enumerate(line_columns):
                if col in line_data.columns:
                    color = line_colors[i % len(line_colors)]
                    fig.add_trace(go.Scatter(
                        x=line_data.index,
                        y=line_data[col],
                        mode='lines+markers',
                        name=self._format_column_name(col),
                        line=dict(color=color, width=3)  # Increased line width for better visibility
                    ))
        
        # Handle bar data
        bar_data = kwargs.get('bar_data')
        bar_columns = kwargs.get('bar_columns', [])
        if bar_data is not None and not bar_data.empty:
            if not bar_columns:  # Use all columns if none specified
                bar_columns = bar_data.columns.tolist()
            
            # Use bar-specific colors
            bar_colors = kwargs.get('bar_colors', [self.colors['positive'], self.colors['negative'],
                                 self.colors['success'], self.colors['error']])
            
            for i, col in enumerate(bar_columns):
                if col in bar_data.columns:
                    color = bar_colors[i % len(bar_colors)]
                    y_values = (-bar_data[col] if self._should_negate_values(col) 
                              else bar_data[col])
                    
                    fig.add_trace(go.Bar(
                        x=bar_data.index,
                        y=y_values,
                        name=self._format_column_name(col),
                        marker_color=color
                    ))
        
        # Add horizontal reference line if requested
        if kwargs.get('add_hline'):
            fig.add_hline(y=kwargs.get('hline_value', 1), 
                         line_dash="dash", line_color="gray", opacity=0.5)
        
        # Collect all y-values to determine optimal range
        all_y_values = []
        if line_data is not None and not line_data.empty:
            for col in line_columns:
                if col in line_data.columns:
                    all_y_values.extend(line_data[col].dropna().tolist())
        
        if bar_data is not None and not bar_data.empty:
            for col in bar_columns:
                if col in bar_data.columns:
                    values = bar_data[col].dropna()
                    if self._should_negate_values(col):
                        values = -values
                    all_y_values.extend(values.tolist())
        
        # Apply layout
        layout = self._get_dark_layout(title)
        
        # Determine optimal y-range
        y_range = self._get_optimal_y_range(all_y_values)
        
        layout.update({
            'xaxis': dict(title='Time', gridcolor='rgba(128, 128, 128, 0.5)', 
                         color='white', tickangle=30),
            'yaxis': dict(title=kwargs.get('yaxis_title', 'Value'), 
                         gridcolor='rgba(128, 128, 128, 0.5)', color='white',
                         range=y_range),  # Add optimal range
            'barmode': kwargs.get('barmode', 'group')
        })
        
        fig.update_layout(**layout)
        self._save_figure(fig, filename)
        return fig
    
    def _create_subplot_chart(self, filename: str, title: str, **kwargs) -> go.Figure:
        """Create a chart with multiple subplots."""
        
        # Detect number of subplots
        subplot_numbers = set()
        for key in kwargs.keys():
            if 'subplot_' in key:
                # Extract subplot number from key like 'line_data_subplot_1'
                parts = key.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    subplot_numbers.add(int(parts[-1]))
        
        max_subplot = max(subplot_numbers) if subplot_numbers else 1
        rows = kwargs.get('rows', max_subplot)
        cols = kwargs.get('cols', 1)
        
        # Get subplot titles
        subplot_titles = kwargs.get('subplot_titles', [f"Subplot {i+1}" for i in range(max_subplot)])
        
        # Create subplot specs (handle secondary y-axes)
        secondary_y_subplots = kwargs.get('secondary_y_subplots', [])
        specs = []
        for i in range(rows):
            row_specs = []
            for j in range(cols):
                subplot_idx = i * cols + j + 1
                if subplot_idx in secondary_y_subplots:
                    row_specs.append({"secondary_y": True})
                else:
                    row_specs.append({})
            specs.append(row_specs)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            specs=specs
        )
        
        # Add traces for each subplot
        for subplot_num in range(1, max_subplot + 1):
            row = ((subplot_num - 1) // cols) + 1
            col = ((subplot_num - 1) % cols) + 1
            
            # Handle line data for this subplot
            line_data_key = f'line_data_subplot_{subplot_num}'
            line_columns_key = f'line_columns_subplot_{subplot_num}'
            
            if line_data_key in kwargs:
                line_data = kwargs[line_data_key]
                line_columns = kwargs.get(line_columns_key, line_data.columns.tolist() if line_data is not None else [])
                
                if line_data is not None and not line_data.empty:
                    # Use line-specific colors (different from bar colors)
                    line_colors = kwargs.get(f'line_colors_subplot_{subplot_num}', 
                                           [self.colors['warning'], self.colors['info'], self.colors['accent'], self.colors['primary']])
                    
                    for i, col_name in enumerate(line_columns):
                        if col_name in line_data.columns:
                            color = line_colors[i % len(line_colors)]
                            # Only use secondary y-axis if explicitly specified for lines
                            use_secondary_for_line = kwargs.get(f'line_secondary_y_subplot_{subplot_num}', False)
                            
                            fig.add_trace(go.Scatter(
                                x=line_data.index,
                                y=line_data[col_name],
                                mode='lines+markers',
                                name=self._format_column_name(col_name),
                                line=dict(color=color, width=3)  # Increased line width for better visibility
                            ), row=row, col=col, secondary_y=use_secondary_for_line if subplot_num in secondary_y_subplots else False)
            
            # Handle bar data for this subplot
            bar_data_key = f'bar_data_subplot_{subplot_num}'
            bar_columns_key = f'bar_columns_subplot_{subplot_num}'
            
            if bar_data_key in kwargs:
                bar_data = kwargs[bar_data_key]
                bar_columns = kwargs.get(bar_columns_key, bar_data.columns.tolist() if bar_data is not None else [])
                
                if bar_data is not None and not bar_data.empty:
                    # Use bar-specific colors (different from line colors)
                    bar_colors = kwargs.get(f'bar_colors_subplot_{subplot_num}',
                                          [self.colors['positive'], self.colors['negative'], self.colors['success'], self.colors['error']])
                    
                    for i, col_name in enumerate(bar_columns):
                        if col_name in bar_data.columns:
                            color = bar_colors[i % len(bar_colors)]
                            y_values = (-bar_data[col_name] if self._should_negate_values(col_name) 
                                      else bar_data[col_name])
                            
                            # Only use secondary y-axis if explicitly specified for bars
                            use_secondary_for_bar = kwargs.get(f'bar_secondary_y_subplot_{subplot_num}', False)
                            
                            fig.add_trace(go.Bar(
                                x=bar_data.index,
                                y=y_values,
                                name=self._format_column_name(col_name),
                                marker_color=color
                            ), row=row, col=col, secondary_y=use_secondary_for_bar if subplot_num in secondary_y_subplots else False)
            
            # Add horizontal reference lines if requested
            if kwargs.get(f'add_hline_subplot_{subplot_num}'):
                hline_value = kwargs.get(f'hline_value_subplot_{subplot_num}', 1)
                fig.add_hline(y=hline_value, line_dash="dash", line_color="gray", 
                            opacity=0.5, row=row, col=col)
        
        # Apply layout
        layout = self._get_dark_layout(title)
        
        # Add axis titles and optimal ranges for each subplot
        for i in range(1, rows * cols + 1):
            if i == 1:
                axis_key = 'yaxis'
            else:
                axis_key = f'yaxis{i}'
            
            # Collect y-values for this subplot to determine optimal range
            subplot_y_values = []
            
            # Check line data for this subplot
            line_data_key = f'line_data_subplot_{i}'
            line_columns_key = f'line_columns_subplot_{i}'
            if line_data_key in kwargs:
                line_data = kwargs[line_data_key]
                line_columns = kwargs.get(line_columns_key, line_data.columns.tolist() if line_data is not None else [])
                if line_data is not None and not line_data.empty:
                    for col in line_columns:
                        if col in line_data.columns:
                            subplot_y_values.extend(line_data[col].dropna().tolist())
            
            # Check bar data for this subplot
            bar_data_key = f'bar_data_subplot_{i}'
            bar_columns_key = f'bar_columns_subplot_{i}'
            if bar_data_key in kwargs:
                bar_data = kwargs[bar_data_key]
                bar_columns = kwargs.get(bar_columns_key, bar_data.columns.tolist() if bar_data is not None else [])
                if bar_data is not None and not bar_data.empty:
                    for col in bar_columns:
                        if col in bar_data.columns:
                            values = bar_data[col].dropna()
                            if self._should_negate_values(col):
                                values = -values
                            subplot_y_values.extend(values.tolist())
            
            # Get optimal range for this subplot
            y_range = self._get_optimal_y_range(subplot_y_values)
            
            layout[axis_key] = dict(
                title=kwargs.get(f'yaxis_title_subplot_{i}', 'Value'),
                gridcolor='rgba(128, 128, 128, 0.5)',
                color='white',
                range=y_range  # Add optimal range
            )
        
        fig.update_layout(**layout)
        self._save_figure(fig, filename)
        return fig
