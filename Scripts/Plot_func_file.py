# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:27:34 2023

@author: s.k
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import plotly.figure_factory as ff
import plotly.io as pio
import os
import re
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

#%%
# Primary secondary line chart
def Pri_sec_plot (df,y2,sec_axis_name):
    figure={}
    for i in df.columns:
        # Create the primary trace
        x=df.index
        y1=df[i]
        trace1 = go.Scatter(x=x, y=y1, name=i, mode='lines')
        # Create the secondary trace
        trace2 = go.Scatter(x=x, y=y2, name=sec_axis_name, mode='lines',yaxis='y2')  # Assign the secondary y-axis
        # Create the layout with secondary y-axis
        sec_axis_name= sec_axis_name
        layout = go.Layout(
                 title=i+"and"+sec_axis_name,
                 xaxis=dict(title='Date'),
                 yaxis=dict(title=i,color='blue'),
                 yaxis2=dict(title=sec_axis_name, overlaying='y', side='right',color='red'),
                 width=1200,  # Set the width of the figure (pixels)
                 height=400) # Set the height of the figure (pixels))
        
        # Combine the traces and layout into a figure
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.update_layout(
          {
        'title': {
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20},}})
        
        fig.show(renderer="browser")
        figure[i]=fig
    return figure
#%%
def create_line_chart(x_data, y_data, x_title, y_title, chart_title):
    trace = go.Scatter(x=x_data, y=y_data, mode='lines+markers',name=chart_title,
                       line=dict(color='blue', width=2),
                       marker=dict(size=8))
    layout = go.Layout(title=chart_title,
                       xaxis=dict(title=x_title),
                       yaxis=dict(title=y_title))
    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(
      {
    'title': {
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20},}})
    fig.show(renderer="browser")
    return fig
#%%    
# function file for box plots
def create_box_plot(data, y_data, title,y_title): 
    if isinstance(data, list):
        # If data is a list, create a box trace directly
        trace = go.Box(y=data, name=title)
    elif isinstance(data, pd.Series):
        # If data is a pandas DataFrame, specify data_frame and column names
        trace = go.Box(y=data, name=title)
    layout = go.Layout(
    title=title,
    yaxis=dict(title=y_title)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(
      {
    'title': {
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20},}})
    fig.show(renderer="browser")
    return fig

#%%
#Create a combined histogram and density plot using Plotly and save it as an HTML file. 
#bin_size=num_bins
def create_histogram_density_plot(data, filename, title, x_title):    
   fig = ff.create_distplot([data], group_labels=['Data'], colors=['blue'], show_hist=True, show_curve=True)
   fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title="Frequency/Density"))
   # Save the plot as an HTML file
   fig.write_html(filename)
   #fig.show(renderer="browser")
    
#%% 
# Box plot of target variable with respect to other process variables using multiple bins of x varibales
def box_plot_xy_color_new(df, y_out, x_out, x_axis_title) :
    df_new = df[[y_out] + [x_out]]
    fig = go.Figure()
    for (i, y) in enumerate(df_new[x_out].unique()) :
        df_new_y = df_new[df_new[x_out] == y]
        fig.add_trace(go.Box(y = df_new_y[y_out].values[:],
                             name= y, marker_color=px.colors.qualitative.Dark2[i]))

    fig.update_xaxes(showgrid=True, linewidth=1, linecolor='black', 
                     mirror=True, gridwidth=1, gridcolor='rgb(220,220,220)')
    fig.update_yaxes(showgrid=True, linewidth=1, linecolor='black', 
                     mirror=True, gridwidth=1, gridcolor='rgb(220,220,220)')

    fig.update_xaxes(zeroline=True, zerolinewidth=1, 
                     zerolinecolor='rgb(220,220,220)')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, 
                     zerolinecolor='rgb(220,220,220)')    
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", 
                      plot_bgcolor="rgba(0,0,0,0)")

    fig.update_layout(font={'family':"Arial",
                                'size':20,
                                'color':'#000000'
                            },
                      yaxis_title= '',
                      xaxis_title =x_axis_title,
                      showlegend = False)
    fig.update_layout(autosize=False, width=900, height=700)    
    return fig    

 
def data_division_in_bins_with_same_amnt_data_plots(df, bin_divided_by, target_tag, number_of_bins):

 

    df = df.dropna(subset = [bin_divided_by])
    df = df.sort_values(by = [bin_divided_by])
    df = df.reset_index(drop = True)
    df[bin_divided_by + '_Bin'] = ''


    number_of_points_in_each_bin = int(len(df)/number_of_bins)

    for i in range(number_of_bins) :
        low_value = round(df[bin_divided_by].values[i*number_of_points_in_each_bin], 2)
        high_value = round(df[bin_divided_by].values[(i+1)*number_of_points_in_each_bin- 1], 2)
        df[bin_divided_by + '_Bin'].values[i*number_of_points_in_each_bin : (i+1)*number_of_points_in_each_bin- 1] = str(low_value) + ' - ' +  str(high_value)

    df = df[df[bin_divided_by + '_Bin'] != '']

    fig     = box_plot_xy_color_new(df               = df, 
                                    y_out            = target_tag, 
                                    x_out            = bin_divided_by + '_Bin', 
                                    x_axis_title     = '')
    fig.update_traces(boxmean           = True)
    fig.update_layout(font              = {'family'     : "Arial",
                                           'size'       : 20,
                                           'color'      :'#000000'
                                           },
                      yaxis_title       = target_tag,
                      xaxis_title       = 'Bin of ' + str(bin_divided_by))

    return fig
     
#%%  
# perason and spearman correlation
def calculate_correlations(df):
    pearson_corr = df.corr()
    # Calculate Spearman correlation
    spearman_corr = df.corr(method='spearman')
    return pearson_corr, spearman_corr  
    
#%% Generates a month-wise box plot with connected median lines and annotated median values.
def plot_monthwise_boxplot_with_mean(df, date_col, value_col,y_axis_name,title):
    df[date_col] = pd.to_datetime(df[date_col])
    df['month_year'] = df[date_col].dt.strftime('%b-%y')
    month_years = df['month_year'].unique()
    mean = df.groupby('month_year')[value_col].mean().reindex(month_years).values
    # Create box plots for each month-year
    fig = go.Figure()
    for month_year in month_years:
        fig.add_trace(go.Box(
            y=df[df['month_year'] == month_year][value_col],
            name=month_year,
            boxmean=True,  # Display mean and standard deviation
            fillcolor='white',            # White fill color for the box
            line=dict(color='black')      # Black border line
        ))
        # Add scatter plot to connect median lines
    fig.add_trace(go.Scatter(
        x=month_years,
        y=mean,
        mode='lines+markers',
        name='Median',
        line=dict(color='Black'),
        marker=dict(symbol='star', color='yellow', size=10)))
    # Add text annotations for the median values
    for i, month_year in enumerate(month_years):
        fig.add_trace(go.Scatter(
            x=[month_year],
            y=[mean[i]],
            mode='text',
            text=[f'{mean[i]:.2f}'],
            textposition='top center',
            showlegend=False))
        # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Month-Year',
        yaxis_title=y_axis_name,
        plot_bgcolor='white',              # Set plotting area background color to white
        paper_bgcolor='white',             # Set paper (outside plotting area) background color to white
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=month_years,
            tickfont=dict(
                color='black',             # Tick label color
                size=12,                   # Tick label size
                family='Arial Black'       # Use a bold font family like Arial Black
            ),
            tickangle=-45                   # Rotate x-axis labels to 45 degrees
        ),
        yaxis=dict(
            showgrid=True,                 # Show grid lines
            gridcolor='lightgrey',         # Grid lines color
            gridwidth=1,                   # Grid lines width
            griddash='dash',               # Dashed grid lines
            tickfont=dict(
                color='black',             # Tick label color
                size=12,                   # Tick label size
                family='Arial Black'             # Font family
            )
        )
    )
    
    fig.show(renderer="browser")
    return fig

#%% Process parameter box plot
def Process_parametrs_boxplot(df, output_dir ):
    output_dir = "..\\Results\\Monthwise Process param box plot"
    def iqr_filter(group, column):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group[(group[column] >= lower) & (group[column] <= upper)]
    
    for col in df.select_dtypes(include='number').columns:
        box_traces = []
    
    # Loop through each month
        for period in sorted(df['month_year'].unique(), key=lambda x: pd.to_datetime(x, format='%b-%Y')):
            month_data = df[df['month_year'] == period]
            filtered = iqr_filter(month_data, col)
    
            if not filtered.empty:
                trace = go.Box(
                    y=filtered[col],
                    name=period,
                    boxmean='sd',
                    marker=dict(color='blue')
                )
                box_traces.append(trace)
    
        if box_traces:
            layout = go.Layout(
                title=f'Month-wise Boxplot for {col}',
                xaxis_title='Month-Year',
                yaxis_title=col,
                title_x=0.5,
                font=dict(size=14),
                plot_bgcolor='white',
                width=600,  
                height=300,
                # Add borders around the plot
                xaxis=dict(
                    showgrid=False,  # Remove gridlines
                    linecolor='black',  # Add border to x-axis
                    linewidth=2,  # Make line thicker
                    mirror=True  # Mirror the axis lines to all sides
                ),
                
                yaxis=dict(
                gridcolor='lightgrey',  # Keep gridlines for clarity
                linecolor='black',  # Add border to y-axis
                linewidth=2,  # Make line thicker
                mirror=True), # Mirror the axis lines to all sides)
                
                # Optional: Box around the entire plot
                showlegend=False,  # Hide legend if not needed
                margin=dict(l=50, r=50, t=50, b=50)  # Add margin for better spacing
            )
    
            fig = go.Figure(data=box_traces, layout=layout)
    
            # Safe filename
            safe_col = col.replace("/", "_")
        
                # Save the figure to an HTML file
            filename = f"{output_dir}/{safe_col}.html"
            fig.write_html(filename)
    return 0
 #%% Multi-Parameter Analysis with Specific Steam Generation (SSG)   
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to sanitize file names by replacing invalid characters
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to generate and save line charts
def Parameters_line_chart (df, Line_chart_dir, width_px=1200, height_px=800):
    """
    Generates line charts for each column in the dataframe and saves them as HTML files.

    Parameters:
    df (DataFrame): The dataframe containing the data.
    Line_chart_dir (str): Directory where the charts will be saved.
    width_px (int): Width of the chart in pixels. Default is 1200.
    height_px (int): Height of the chart in pixels. Default is 800.
    """
    # Ensure the directory exists
    ensure_directory_exists(Line_chart_dir)
    
    # Create a dictionary to hold the figures
    figures = {}

    # Loop through each column in the dataframe
    for column in df.columns:
        fig = go.Figure()

        # Add a trace (line + marker) for the current column
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[column], 
            mode='lines+markers',  # Lines with dots at each data point
            name=column,
            line=dict(color='royalblue', width=2),  # Custom line color
            marker=dict(size=8, color='darkorange', symbol='circle', line=dict(width=2, color='black'))  # Custom marker style
        ))

        # Customize layout for an attractive look
        fig.update_layout(
            title=f'<b>Line Chart - {column}</b>',  # Bold title
            title_font=dict(size=24, color='darkblue', family='Arial'),  # Title font customization
            xaxis_title='Date_time',  # X-axis label
            xaxis_title_font=dict(size=18, color='black'),  # Bold font for X-axis title
            yaxis_title=column,  # Y-axis label
            yaxis_title_font=dict(size=18, color='black'),  # Bold font for Y-axis title
            xaxis=dict(
                showgrid=True, gridcolor='grey',  # Light gridlines for x-axis
                gridwidth=1,  # Gridline thickness for x-axis
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True, gridcolor='grey',  # Light gridlines for y-axis
                gridwidth=1,  # Gridline thickness for y-axis
                zeroline=False
            ),
            plot_bgcolor='lightgrey',  # Plot area background color
            paper_bgcolor='white',  # Chart background color
            font=dict(family='Arial', size=14, color='black'),  # Global font style
            legend=dict(
                title='<b>Legend</b>',  # Bold legend title
                font=dict(family='Arial', size=12, color='black'),
                bgcolor='white',  # Background color of the legend
                bordercolor='lightgrey', borderwidth=2,
                x=1, y=1  # Position at top-right
            ),
            hovermode='x unified',  # Better hover interaction
            width=width_px,  # Chart width
            height=height_px  # Chart height
        )

        # Store the figure in the dictionary
        figures[column] = fig

        # Sanitize the column name for file naming
        sanitized_column = sanitize_filename(column)

        # Create the file path
        file_path = os.path.join(Line_chart_dir, f'{sanitized_column}_line_chart.html')

        # Save the figure as an HTML file
        fig.write_html(file_path)

        # Optionally, open the chart in a browser
        plot(fig, filename=file_path, auto_open=False)

    print("Parameters line charts generated and saved successfully!")
    
#%%
def RB_Parameters_subplots_SSG(df, subplot_chart_dir, width_px=1300, height_px=1500):
    """
    Generates multi-parameter subplots for analysis and saves them as an HTML file.

    Parameters:
    df (DataFrame): The dataframe containing the data.
    subplot_chart_dir (str): Directory where the charts will be saved.
    width_px (int): Width of the chart in pixels. Default is 1300.
    height_px (int): Height of the chart in pixels. Default is 1500.
    """
    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    # Data for the subplots
    x = df.index
    y1 = df['Black Liquor Flow ']
    y2 = df['Dry Solids Measurement A']
    y3 = df['DENSITY']
    y4 = df['TOTAL DRY SOLIDS']
    y5 = df['Primary Air Flow set point']
    y6 = df['Secondary Air Flow Control Set point']
    y7 = df['Gross Steam Flow']
    y8 = df['ESP-I Outlet Flue Gas Temperature']
    y9 = df['Specific Steam Generation']
    y10 = df['Soot Blower Steam Flow']

    # Create subplots: 7 rows, 1 column
    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
               [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # Add traces to each subplot
    fig.add_trace(go.Scatter(x=x, y=y1, name='Black Liquor Flow (m3/hr)', mode='lines+markers', line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=x, y=y2, name='%Dry Solid', mode='lines+markers', line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=y4, name='TOTAL DRY SOLIDS (TPD)', mode='lines+markers', line=dict(color="red")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=3, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=y5, name='Primary Air Flow set point (NM3/hr)', mode='lines+markers', line=dict(color="red")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=y6, name='Secondary Air Flow set point (NM3/hr)', mode='lines+markers', line=dict(color="orange")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=4, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=y7, name='Gross Steam Flow (TPH)', mode='lines+markers', line=dict(color="red")), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=5, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=y8, name='Flue Gas Temperature', mode='lines+markers', line=dict(color="red")), row=6, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=6, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=y10, name='Soot Blowing steam flow(TPH)', mode='lines+markers', line=dict(color="red")), row=7, col=1)
    fig.add_trace(go.Scatter(x=x, y=y9, name='Specific Steam Generation', mode='lines+markers', line=dict(color='green')), row=7, col=1, secondary_y=True)

    # Update layout for title, dimensions, and styling
    fig.update_layout(
        title="Recovery boiler parameters subplots with SSG",
        height=height_px, width=width_px,
        title_x=0.5,
        showlegend=False,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        font=dict(color='black')
    )

    # Customize x and y axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=2, zerolinecolor='Gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=2, zerolinecolor='Gray')
    
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=7, col=1)  # Shared x-axis for all subplots
    
    # Add labels for each subplot
    fig.update_yaxes(title_text="Black liquor flow (m3/hr)", row=1, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=1, col=1)

    fig.update_yaxes(title_text="%Dry Solid", row=2, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=2, col=1)

    fig.update_yaxes(title_text="Total dry Solid (TPD)", row=3, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=3, col=1)

    fig.update_yaxes(title_text="Air Flow (NM3/hr)", row=4, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=4, col=1)

    fig.update_yaxes(title_text="Gross steam flow (TPH)", row=5, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=5, col=1)

    fig.update_yaxes(title_text="Flue gas stack temp (degC)", row=6, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=6, col=1)

    fig.update_yaxes(title_text="Soot Blowing steam Flow(TPH)", row=7, col=1)
    fig.update_yaxes(title_text="SSG", secondary_y=True, row=7, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5
    for i in range(7):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 7), y1=(1 - i / 7), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'Recovery boiler parameters subplots with SSG_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)

    print("Recovery boiler parameters subplots with SSG_line_chart plotted successfully!")   
    
#%% Steam distribution and it's parameters subplots
def Steam_distribution_and_parameters_subplots(df, subplot_chart_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    x = df.index
    y1 = df['Gross Steam Flow']
    y2 = df['Main Steam Temperature']
    y3 = df['Mainsteam Pressure']
    y4 = df['Soot Blower Steam Flow']
    y5 = df['RB Plus IJT steam flow(TPH)']
    y6 = df['Steam flow to turbine(TPH)']
    y7 = df['IJT STEAM FLOW']

    # Create subplots: 4 rows, 1 column
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: Gross steam flow
    fig.add_trace(go.Scatter(x=x, y=y1, name='Gross Steam Flow (TPH)', mode='lines+markers', line=dict(color="red")), row=1, col=1)

    # Second subplot: main steam teamp & pressure
    fig.add_trace(go.Scatter(x=x, y=y2, name='Steam Temperature (degC)', mode='lines+markers', line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y3, name='Steam Pressure (barG)', mode='lines+markers', line=dict(color='green')), row=2, col=1, secondary_y=True)

    # Third subplot: Soot blower steam flow
    fig.add_trace(go.Scatter(x=x, y=y4, name='Soot Blower Steam Flow (TPH)', mode='lines+markers', line=dict(color="red")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=y7, name='IJT STEAM FLOW', mode='lines+markers', line=dict(color="green")), row=3, col=1)

    # Fourth subplot: RB Plus IJT steam flow(TPH) & Steam flow to turbine
    fig.add_trace(go.Scatter(x=x, y=y5, name='RB Plus IJT steam flow(TPH)', mode='lines+markers', line=dict(color="red")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=y6, name='Steam flow to turbine(TPH)', mode='lines+markers', line=dict(color="green")), row=4, col=1)

    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="Steam distribution and it's parameters subplots",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=4, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='Gross Steam Flow (TPH)', row=1, col=1)

    fig.update_yaxes(title_text='Steam Temperature (degC)', row=2, col=1)
    fig.update_yaxes(title_text='Steam Pressure (barG)', secondary_y=True, row=2, col=1)

    fig.update_yaxes(title_text='Soot Blower & IJT Steam Flow (TPH)', row=3, col=1)

    fig.update_yaxes(title_text='RB Plus IJT steam flow & Steam flow to turbine (TPH)', row=4, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5

    # Define the box around each subplot
    for i in range(4):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 4), y1=(1 - i / 4), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'Steam distribution and parameters subplots_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("Steam distribution and parameters subplots_line_chart plotted successfully!")   
    

#%% Evaporator section parameters subplots
def Evaporator_section_parameters_subplots(df, subplot_chart_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    x = df.index
    y1 = df['Street-1 Feed WBL Flow (M3/h)']
    y2 = df['Street-2 Feed WBL flow (M3/h)']
    y3 = df['Evap-4F Main LP Steam flow (TPH)']
    y4 = df['Evap-4F Product liquor outlet Temperature']
    y5 = df['WBL % SOLIDS']
    y6 = df['WBL flow to evaporator WBL density']

    # Create subplots: 4 rows, 1 column
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: Feed WBL Flow to evaporator (M3/h)
    fig.add_trace(go.Scatter(x=x, y=y1, name='Street-1 Feed WBL Flow (M3/h)', mode='lines+markers', line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, name='Street-2 Feed WBL flow (M3/h)', mode='lines+markers', line=dict(color='green')), row=1, col=1)

    # Second subplot: WBL % SOLIDS & density
    fig.add_trace(go.Scatter(x=x, y=y5, name='WBL % SOLIDS', mode='lines+markers', line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y6, name='WBL density', mode='lines+markers', line=dict(color="green")), row=2, col=1,secondary_y=True)

    # Third subplot: Evap-4F Main LP Steam flow (TPH) & liquor outlet Temperature
    fig.add_trace(go.Scatter(x=x, y=y3, name='Evap-4F LP Steam flow (TPH)', mode='lines+markers', line=dict(color="red")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=y4, name='Evap-4F liquor outlet Temperature', mode='lines+markers', line=dict(color='green')), row=3, col=1, secondary_y=True)

    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="Evaporator section parameters subplots",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=3, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='Feed WBL Flow (M3/h)', row=1, col=1)

    fig.update_yaxes(title_text='WBL % SOLIDS', row=2, col=1)
    fig.update_yaxes(title_text='WBL density', secondary_y=True, row=2, col=1)

    fig.update_yaxes(title_text='Evap-4F LP Steam flow (TPH)', row=3, col=1)
    fig.update_yaxes(title_text='Evap-4F liquor temperature', secondary_y=True, row=3, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5

    # Define the box around each subplot
    for i in range(3):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 3), y1=(1 - i / 3), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'Evaporator section parameters subplots_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("Evaporator section parameters subplots_line_chart plotted successfully!")   
    
#%% White liquor clarifier overflow sample analysis
def White_liquor_clarifier_overflow_sample_subplots(df, subplot_chart_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    x = df.index
    y1 = df['WL clarifier O/F TTA']
    y2 = df['WL clarifier O/FAA']
    y3 = df['WL clarifier O/F NaOH']
    y4 = df['WL clarifier O/F Na2S']
    y5 = df['WL clarifier O/F Na2CO3']
    y6 = df['WL clarifier O/F sulphidity']
    y7 = df['WL clarifier O/F causticity']

    # Create subplots: 7 rows, 1 column
    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}],
               [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: WL clarifier O/F TTA
    fig.add_trace(go.Scatter(x=x, y=y1, name='TTA (gpl)', mode='lines+markers', line=dict(color="black")), row=1, col=1)

    # Second subplot: WL clarifier O/F AA
    fig.add_trace(go.Scatter(x=x, y=y2, name='AA (gpl)', mode='lines+markers', line=dict(color="black")), row=2, col=1)

    # Third subplot: WL clarifier O/F NaOH
    fig.add_trace(go.Scatter(x=x, y=y3, name='NaOH (gpl)', mode='lines+markers', line=dict(color="black")), row=3, col=1)

    # Fourth subplot: WL clarifier O/F Na2S
    fig.add_trace(go.Scatter(x=x, y=y4, name='Na2S (gpl)', mode='lines+markers', line=dict(color="black")), row=4, col=1)

    # Fifth subplot: WL clarifier O/F Na2CO3
    fig.add_trace(go.Scatter(x=x, y=y5, name='Na2CO3 (gpl)', mode='lines+markers', line=dict(color="black")), row=5, col=1)

    # Sixth subplot: WL clarifier O/F sulphidity
    fig.add_trace(go.Scatter(x=x, y=y6, name='Sulphidity', mode='lines+markers', line=dict(color="black")), row=6, col=1)

    # Seventh subplot: WL clarifier O/F causticity
    fig.add_trace(go.Scatter(x=x, y=y7, name='Causticity', mode='lines+markers', line=dict(color="black")), row=7, col=1)

    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="White liquor clarifier overflow sample analysis",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=7, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='WL TTA (gpl)', row=1, col=1)
    fig.update_yaxes(title_text='WL AA (gpl)', row=2, col=1)
    fig.update_yaxes(title_text='WL NaOH (gpl)', row=3, col=1)
    fig.update_yaxes(title_text='WL Na2S (gpl)', row=4, col=1)
    fig.update_yaxes(title_text='WL Na2CO3 (gpl)', row=5, col=1)
    fig.update_yaxes(title_text='WL Sulphidity', row=6, col=1)
    fig.update_yaxes(title_text='WL Causticity', row=7, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5

    # Define the box around each subplot
    for i in range(7):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 7), y1=(1 - i / 7), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'White liquor clarifier overflow sample analysis_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("White liquor clarifier overflow sample analysis_line_chart plotted successfully!")   

#%% Green liquor storage sample analysis
def Green_liquor_storage_sample_subplots(df, subplot_chart_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    x = df.index
    y1 = df['Green liquor storage TTA']
    y2 = df['Green liquor storage AA']
    y3 = df['Green liquor storage NaOH']
    y4 = df['Green liquor storage Na2S']
    y5 = df['Green liquor storage Na2CO3']


    # Create subplots: 5 rows, 1 column
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}],
               [{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: Green liquor storage TTA
    fig.add_trace(go.Scatter(x=x, y=y1, name='TTA (gpl)', mode='lines+markers', line=dict(color="black")), row=1, col=1)

    # Second subplot: Green liquor storage AA
    fig.add_trace(go.Scatter(x=x, y=y2, name='AA (gpl)', mode='lines+markers', line=dict(color="black")), row=2, col=1)

    # Third subplot: Green liquor storage NaOH
    fig.add_trace(go.Scatter(x=x, y=y3, name='NaOH (gpl)', mode='lines+markers', line=dict(color="black")), row=3, col=1)

    # Fourth subplot: Green liquor storage Na2S
    fig.add_trace(go.Scatter(x=x, y=y4, name='Na2S (gpl)', mode='lines+markers', line=dict(color="black")), row=4, col=1)

    # Fifth subplot: Green liquor storage Na2CO3
    fig.add_trace(go.Scatter(x=x, y=y5, name='Na2CO3 (gpl)', mode='lines+markers', line=dict(color="black")), row=5, col=1)

    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="Green liquor storage sample analysis",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=7, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='GL TTA (gpl)', row=1, col=1)
    fig.update_yaxes(title_text='GL AA (gpl)', row=2, col=1)
    fig.update_yaxes(title_text='GL NaOH (gpl)', row=3, col=1)
    fig.update_yaxes(title_text='GL Na2S (gpl)', row=4, col=1)
    fig.update_yaxes(title_text='GL Na2CO3 (gpl)', row=5, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5

    # Define the box around each subplot
    for i in range(5):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 5), y1=(1 - i / 5), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'Green liquor storage sample analysis_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("Green liquor storage sample analysis_line_chart plotted successfully!")  
    
#%% Pulp mill white liquor sample analysis
def Pulp_mill_white_liquor_sample_subplots(df, subplot_chart_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(subplot_chart_dir)

    x = df.index
    y1 = df['WL tank No1 from pulp mill TTA']
    y2 = df['WL tank No1 from pulp mill AA']
    y3 = df['WL tank No1 from pulp mill NaOH']
    y4 = df['WL tank No1 from pulp mill Na2S']
    y5 = df['WL tank No1 from pulp mill Na2CO3']
    y6 = df['WL tank No1from pulp mill Sulphidity']
    y7 = df['WL tank No1 from pulp mill Causticity']

    # Create subplots: 7 rows, 1 column
    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}],
               [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: WL tank No1 from pulp mill TTA
    fig.add_trace(go.Scatter(x=x, y=y1, name='TTA (gpl)', mode='lines+markers', line=dict(color="black")), row=1, col=1)

    # Second subplot: WL tank No1 from pulp mill AA
    fig.add_trace(go.Scatter(x=x, y=y2, name='AA (gpl)', mode='lines+markers', line=dict(color="black")), row=2, col=1)

    # Third subplot: WL tank No1 from pulp mill NaOH
    fig.add_trace(go.Scatter(x=x, y=y3, name='NaOH (gpl)', mode='lines+markers', line=dict(color="black")), row=3, col=1)

    # Fourth subplot: WL tank No1 from pulp mill Na2S
    fig.add_trace(go.Scatter(x=x, y=y4, name='Na2S (gpl)', mode='lines+markers', line=dict(color="black")), row=4, col=1)

    # Fifth subplot: WL tank No1 from pulp mill Na2CO3
    fig.add_trace(go.Scatter(x=x, y=y5, name='Na2CO3 (gpl)', mode='lines+markers', line=dict(color="black")), row=5, col=1)

    # Sixth subplot: WL tank No1 from pulp mill sulphidity
    fig.add_trace(go.Scatter(x=x, y=y6, name='Sulphidity', mode='lines+markers', line=dict(color="black")), row=6, col=1)

    # Seventh subplot: WL tank No1 from pulp mill causticity
    fig.add_trace(go.Scatter(x=x, y=y7, name='Causticity', mode='lines+markers', line=dict(color="black")), row=7, col=1)

    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="Pulp mill white liquor sample analysis",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=7, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='WL TTA (gpl)', row=1, col=1)
    fig.update_yaxes(title_text='WL AA (gpl)', row=2, col=1)
    fig.update_yaxes(title_text='WL NaOH (gpl)', row=3, col=1)
    fig.update_yaxes(title_text='WL Na2S (gpl)', row=4, col=1)
    fig.update_yaxes(title_text='WL Na2CO3 (gpl)', row=5, col=1)
    fig.update_yaxes(title_text='WL Sulphidity', row=6, col=1)
    fig.update_yaxes(title_text='WL Causticity', row=7, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5

    # Define the box around each subplot
    for i in range(7):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / 7), y1=(1 - i / 7), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join(subplot_chart_dir, 'Pulp mill white liquor sample analysis_line_chart.html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("Pulp mill white liquor sample analysis_line_chart plotted successfully!")
    
#%% Pulp mill white liquor sample analysis
def Figure_plotting_for_reporting (df, reporting_plot_dir, width_px=1300, height_px=1500):

    # Ensure the directory exists
    ensure_directory_exists(reporting_plot_dir)

    x = df.index
    y1 = df['Current SSG (DCS calculated value)']
    y2 = df['Recommended SSG (Model predicted value)']
    y3 = df['Current reduction efficiency (model)']
    y4 = df['Recommended reduction efficiency (model)']
    y5 = df['Primary Air Flow set point']
    #y6 = df['Recommended primary air flow']
    y7 = df['Secondary Air Flow set point']
    # y8 = df['Recommended secondary air flow']
    
    
    # Create subplots: 4 rows, 1 column
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}], [{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]]
    )

    # First subplot: SSG current vs recommended
    fig.add_trace(go.Scatter(x=x, y=y1, name='Current SSG (DCS calculated value)', mode='lines+markers', line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, name='Recommended SSG (Model predicted value)', mode='lines+markers', line=dict(color='green')), row=1, col=1, secondary_y=False)
    
    # Second subplot: Reduction efficiency current vs recommended
    fig.add_trace(go.Scatter(x=x, y=y3, name='Current reduction efficiency (model)', mode='lines+markers', line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y4, name='Recommended reduction efficiency (model)', mode='lines+markers', line=dict(color='green')), row=2, col=1, secondary_y=False)
      
    # Third subplot : Primary air flow SP vs recommended
    fig.add_trace(go.Scatter(x=x, y=y5, name='Primary Air Flow set point (NM3/hr)', mode='lines+markers', line=dict(color="red")), row=3, col=1)
    #fig.add_trace(go.Scatter(x=x, y=y6, name='Recommended primary air flow (NM3/hr)', mode='lines+markers', line=dict(color='green')), row=3, col=1, secondary_y=False)
    
    # Fourth subplot: Secondary air flow SP vs recommended
    fig.add_trace(go.Scatter(x=x, y=y7, name='Secondary Air Flow set point (NM3/hr)', mode='lines+markers', line=dict(color="red")), row=4, col=1)
    #fig.add_trace(go.Scatter(x=x, y=y8, name='Recommended secondary air flow (NM3/hr)', mode='lines+markers', line=dict(color='green')), row=4, col=1, secondary_y=False)
    # Update layout for better title positioning and axis labels
    fig.update_layout(
        title="SSG, Reduction efficiency, Air flow parameters (Actual & Recommended)",
        height=1000, width=1300, 
        title_x=0.5,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        plot_bgcolor='white',  # Set plot area background color to white
        paper_bgcolor='white',  # Set overall background color to white
        hovermode="x unified",  # Unified hover mode to show all traces at once
        font=dict(color='black')
    )

    # Customize gridlines for all x-axes and y-axes
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the x-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',  # Adds gridlines to the y-axis
        zeroline=True, zerolinewidth=2, zerolinecolor='Gray'  # Zero line (if applicable)
    )
    # Set x-axis title and y-axis titles
    fig.update_xaxes(title_text="Date/Time", row=4, col=1)  # Shared x-axis for all subplots

    # First subplot axis labels
    fig.update_yaxes(title_text='SSG', row=1, col=1)
    fig.update_yaxes(title_text='Reduction efficiency', row=2, col=1)
    fig.update_yaxes(title_text='Primary Air Flow', row=3, col=1)
    fig.update_yaxes(title_text='Secondary air flow', row=4, col=1)

    # Adding shapes (boxes) around each subplot
    box_color = 'black'
    box_line_width = 1.5
    
    number_of_plots = 4
    # Define the box around each subplot
    for i in range(number_of_plots):
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=(1 - (i + 1) / number_of_plots), y1=(1 - i / number_of_plots), xref="paper", yref="paper",
            line=dict(color=box_color, width=box_line_width)
        )
    # Save the figure as an HTML file
    file_path = os.path.join( reporting_plot_dir, 'SSG, Reduction efficiency, Air flow parameters (Actual & Recommended).html')
    fig.write_html(file_path)
    plot(fig, filename=file_path, auto_open=False)
    print("SSG, Reduction efficiency, Air flow parameters (Actual & Recommended)_line_chart plotted successfully!")