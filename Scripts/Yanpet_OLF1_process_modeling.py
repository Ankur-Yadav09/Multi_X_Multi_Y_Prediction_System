# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:55:23 2026

@author: 30793167 : Sumit Kumar
"""
# Import libraries

import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import os
from Plot_func_file import create_histogram_density_plot,data_division_in_bins_with_same_amnt_data_plots,Parameters_line_chart, Process_parametrs_boxplot
from plotly.offline import plot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from ML_Function_file import random_forest, Random_searchCV_rf,linear_regression,Ridge_regression,Lasso_regression,remove_outliers_by_boxplot,Grid_searchCV_rf, Xgboost  
import pickle
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.signal import cont2discrete
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace
from nfoursid.kalman import Kalman
import joblib
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize_scalar
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.io as pio
 

#%% data loading
# For reproducibility.
 
file_path = "..\\Data"

df = pd.read_excel(
    os.path.join(file_path, "DMC_Screen_tags_data.xlsx"),
    sheet_name="PI data"
)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df.set_index("Timestamp",inplace =True)

df["Furnace_feed_flow_1_5"] = df[[
    "Total_ET1NE_Feed_F1",
    "Total_ET1NE_Feed_F2",
    "Total_ET1NE_Feed_F3",
    "Total_ET1NE_Feed_F4",
    "Total_ET1NE_Feed_F5"
]].sum(axis=1, skipna=True)

df["Furnace_feed_flow_6_12"] = df[[
    "Total_ET1NE_Feed_F6",
    "Total_ET1NE_Feed_F7",
    "Total_ET1NE_Feed_F8",
    "Total_ET1NE_Feed_F9",
    "Total_ET1NE_Feed_F10",
    "Total_ET1NE_Feed_F11",
    "Total_ET1NE_Feed_F12"
]].sum(axis=1, skipna=True)

# Only include terms where both temp and flow are valid
valid_f1_5 = df["Quench_tower_feed_temp_F1_5"].notna() & df["Furnace_feed_flow_1_5"].notna()
valid_f6_12 = df["Quench_tower_feed_temp_F6_12"].notna() & df["Furnace_feed_flow_6_12"].notna()

numerator = (
    df["Quench_tower_feed_temp_F1_5"] * df["Furnace_feed_flow_1_5"] * valid_f1_5 +
    df["Quench_tower_feed_temp_F6_12"] * df["Furnace_feed_flow_6_12"] * valid_f6_12
)

denominator = (
    df["Furnace_feed_flow_1_5"] * valid_f1_5 +
    df["Furnace_feed_flow_6_12"] * valid_f6_12
)

df["Quench_tower_inlet_temp"] = np.where(denominator != 0, numerator / denominator, np.nan)

# df["PRC_1ST_STAGE_Suction_FLOW "] = df["PRC_3RD_STAGE_Suction_FLOW"] + df["K1601 2ND STAGE Discharge FLOW"]- df['PRC_2nd_stage_drum_Overhead_Flow']

df.drop(columns = ['Furnace_feed_flow_1_5','Furnace_feed_flow_6_12'],inplace =True)

df.dropna(inplace =True)

df = df.apply(pd.to_numeric, errors='coerce')
df_stats = df.describe(include='all')

# def remove_outliers_by_boxplot (df,col_list):
#     for col_name in col_list:
#         Q1=df[col_name].quantile(0.25)
#         Q3=df[col_name].quantile(0.75)
#         IQR=Q3-Q1
#         df=df[(df[col_name]>=Q1-1.5*IQR) & (df[col_name]<=Q3+1.5*IQR)]
#     return df

# col_list = df.columns
# df = remove_outliers_by_boxplot (df,col_list)

#%% Furnace average COT calculation

# 1. Load and Clean Data
df_furnace = pd.read_excel(
    os.path.join(file_path, "DMC_Screen_tags_data.xlsx"),
    sheet_name="Furnace data"
)

# Set first row as header and clean
df_furnace.columns = df_furnace.iloc[0]
df_furnace = df_furnace[1:].reset_index(drop=True)
df_furnace.columns = df_furnace.columns.str.strip()

# Ensure Timestamp is index and data is numeric
if "Timestamp" in df_furnace.columns:
    df_furnace.set_index("Timestamp", inplace=True)

df_furnace = df_furnace.apply(pd.to_numeric, errors='coerce')

# 2. Define Global Accumulators
# We will sum (Value * Feed) and Sum(Feed) across ALL furnaces and coils
metrics_config = [
    ('Global_WA_COT', 'Furnace_Coil_Outlet_Temperature_Coil', 'Furnace_Feed_Rate_Coil'),
    ('Global_WA_CIP', 'Coil', 'Furnace_Feed_Rate_Coil', '_CIP_Corrected_atma'), # Special pattern
    ('Global_WA_SHC', 'Furnace_Coil', 'Furnace_Feed_Rate_Coil', '_SHC_Ratio'),  # Special pattern
    ('Global_WA_Feed_CV', 'Furnace_Feed_Coil', 'Furnace_Feed_Rate_Coil', '_CV_Opening'),
    ('Global_WA_Steam_CV', 'Furnace_Dilution_Steam_Coil', 'Furnace_Feed_Rate_Coil', '_CV_Opening'),
    ('Global_WA_Mixed_Feed_Temp', 'Coil', 'Furnace_Feed_Rate_Coil', '_Mixed_Feed_Inlet_Temperature')
]

# Initialize columns to 0 for accumulation
total_valid_feed_sum = pd.Series(0, index=df_furnace.index, dtype=float)
weighted_sums = {name: pd.Series(0, index=df_furnace.index, dtype=float) for name, *_ in metrics_config}

# 3. Iterate through every Furnace (1-12) and Coil (1-4)
for furnace in range(1, 13):
    status_col = f'F{furnace}_online_status'
    
    # Base Status Mask
    if status_col in df_furnace.columns:
        status_mask = (df_furnace[status_col] == 1)
    else:
        status_mask = pd.Series(True, index=df_furnace.index)

    for coil in range(1, 5):
        feed_col = f'F{furnace}_Furnace_Feed_Rate_Coil{coil}'
        if feed_col not in df_furnace.columns:
            continue

        feed_series = df_furnace[feed_col] # Keep NaNs for now to check validity
        
        # Basic Mask: Online AND Feed exists and > 0
        # We fillna(0) ONLY for the feed check to treat missing feed as 0
        feed_valid = feed_series.fillna(0) > 0
        base_mask = status_mask & feed_valid

        # Add valid feed to total denominator immediately (if base conditions met)
        # Note: We will adjust this later per metric if the VALUE is missing
        current_feed = feed_series.fillna(0).where(base_mask, 0)
        
        for config in metrics_config:
            metric_name = config[0]
            
            # Construct Value Column
            if len(config) == 3:
                val_col = f'F{furnace}_{config[1]}{coil}'
            else:
                val_col = f'F{furnace}_{config[1]}{coil}{config[3]}'
            
            if val_col not in df_furnace.columns:
                continue

            val_series = df_furnace[val_col] # Keep NaNs
            
            # CRITICAL FIX: Specific Mask for this Metric
            # Data is only valid if: Base Mask is True AND Value is NOT NaN
            valid_data_mask = base_mask & val_series.notna()
            
            # Extract feed and value ONLY where data is fully valid
            valid_feed = current_feed.where(valid_data_mask, 0)
            valid_value = val_series.where(valid_data_mask, 0) # NaNs become 0 here safely
            
            # Accumulate
            weighted_sums[metric_name] += (valid_value * valid_feed)
            
            # Add to metric-specific total feed denominator
            # We create a temporary total feed tracker per metric to be precise, 
            # OR we can just accumulate to a global dict of feed sums per metric.
            # To keep it simple and accurate, let's accumulate valid feed per metric inside the loop 
            # by using a separate dictionary for denominators.
            if f"{metric_name}_feed" not in weighted_sums:
                weighted_sums[f"{metric_name}_feed"] = pd.Series(0, index=df_furnace.index, dtype=float)
            
            weighted_sums[f"{metric_name}_feed"] += valid_feed

# --- 4. Final Calculation ---
for metric_name in [c[0] for c in metrics_config]:
    result_col = f"{metric_name}_AllCoils"
    feed_col = f"{metric_name}_feed"
    
    denominator = weighted_sums[feed_col]
    numerator = weighted_sums[metric_name]
    
    df_furnace[result_col] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )

# Cleanup helper columns if needed (optional, as they were dict keys not DF cols)
# The logic above used the dict to store intermediate feed sums per metric.

print("Calculation complete with NaN handling.")
print(df_furnace.filter(like="Global_WA").head())

# Initialize accumulators for COP calculation
global_weighted_sum_cop = pd.Series(0, index=df_furnace.index, dtype=float)
global_valid_feed_cop = pd.Series(0, index=df_furnace.index, dtype=float)

for furnace in range(1, 13):
    # Define Columns
    feed_1 = f'F{furnace}_Furnace_Feed_Rate_Coil1'
    feed_2 = f'F{furnace}_Furnace_Feed_Rate_Coil2'
    feed_3 = f'F{furnace}_Furnace_Feed_Rate_Coil3'
    feed_4 = f'F{furnace}_Furnace_Feed_Rate_Coil4'
    
    cop_tlea = f'F{furnace}_Corrected_COP_Furnace_TLEA'
    cop_tleb = f'F{furnace}_Corrected_COP_Furnace_TLEB'
    
    status_col = f'F{furnace}_online_status'
    
    required_cols = [feed_1, feed_2, feed_3, feed_4, cop_tlea, cop_tleb, status_col]
    
    if all(col in df_furnace.columns for col in required_cols):
        # Get Series (keep NaNs for COP, fill NaNs with 0 for Feed checks)
        f1 = df_furnace[feed_1].fillna(0)
        f2 = df_furnace[feed_2].fillna(0)
        f3 = df_furnace[feed_3].fillna(0)
        f4 = df_furnace[feed_4].fillna(0)
        
        cop_a = df_furnace[cop_tlea] # Keep NaNs
        cop_b = df_furnace[cop_tleb] # Keep NaNs
        
        status = df_furnace[status_col]
        
        # Base Mask: Furnace Online
        base_mask = (status == 1)
        
        # --- TLEA Calculation (Coils 1 & 2) ---
        feed_tlea = f1 + f2
        # Valid only if Online, Feed > 0, AND COP is not NaN
        mask_tlea = base_mask & (feed_tlea > 0) & cop_a.notna()
        
        # --- TLEB Calculation (Coils 3 & 4) ---
        feed_tleb = f3 + f4
        # Valid only if Online, Feed > 0, AND COP is not NaN
        mask_tleb = base_mask & (feed_tleb > 0) & cop_b.notna()
        
        # Accumulate Weighted Sum (NaNs in COP become 0 when multiplied, but mask prevents bad data addition)
        # We use .where(mask, 0) to ensure only valid rows contribute
        term_a = (cop_a * feed_tlea).where(mask_tlea, 0)
        term_b = (cop_b * feed_tleb).where(mask_tleb, 0)
        
        global_weighted_sum_cop += term_a + term_b
        
        # Accumulate VALID Feed Denominator
        # Only add feed_tlea if mask_tlea is True, same for TLEB
        global_valid_feed_cop += feed_tlea.where(mask_tlea, 0) + feed_tleb.where(mask_tleb, 0)

# Final Global Calculation
df_furnace['Global_Weighted_Avg_COP_AllCoils'] = np.where(
    global_valid_feed_cop > 0,
    global_weighted_sum_cop / global_valid_feed_cop,
    np.nan
)

col_rename = {'Global_WA_COT_AllCoils': 'Coil_Avg_COT',
              'Global_WA_CIP_AllCoils': 'Coil_Avg_CIP',
              'Global_WA_SHC_AllCoils': 'Coil_Avg_SHC_Ratio',
              'Global_Weighted_Avg_COP_AllCoils': 'Coil_Weighted_Avg_COP',
              'Global_WA_Feed_CV_AllCoils': 'Coil_Weighted_Avg_Feed_CV_opening',
              'Global_WA_Steam_CV_AllCoils': 'Coil_Weighted_Avg_Steam_CV_opening',
              'Global_WA_Mixed_Feed_Temp_AllCoils': 'Coil_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'
              }

df_furnace.rename(columns=col_rename, inplace=True, errors='ignore')

#Average molecular weight calculation
df_furnace["Furnace_Sum_of_Feed_Components"] = (df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_CH4"] + 
                                          df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_C2H6"] + 
                                          df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_C3H8"])

df_furnace['Furnace_Normalised_Feed_CH4'] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_CH4']*100/df_furnace["Furnace_Sum_of_Feed_Components"]
df_furnace["Furnace_Normalised_Feed_C2H6"] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_C2H6']*100/df_furnace["Furnace_Sum_of_Feed_Components"]
df_furnace["Furnace_Normalised_Feed_C3H8"] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_C3H8']*100/df_furnace["Furnace_Sum_of_Feed_Components"]

df_furnace['Furnace_Normalised_Feed_CH4_Wt'] = df_furnace['Furnace_Normalised_Feed_CH4']*16*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)
df_furnace['Furnace_Normalised_Feed_C2H6_Wt'] = df_furnace['Furnace_Normalised_Feed_C2H6']*30*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)
df_furnace['Furnace_Normalised_Feed_C3H8_Wt'] = df_furnace['Furnace_Normalised_Feed_C3H8']*44*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)

df_furnace["Furnace_Feed_Average_Molecular_Wt"] = df_furnace['Furnace_Normalised_Feed_CH4']/100*16+df_furnace["Furnace_Normalised_Feed_C2H6"]/100*30+df_furnace["Furnace_Normalised_Feed_C3H8"]/100*44

df_furnace['Furnace_conversion'] = ((df_furnace['DMCTF_feed']/1000)*(df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100) - 
                            (df_furnace['DMCTF_feed']/1000-df_furnace['Fresh ethane feed']))/((df_furnace['DMCTF_feed']/1000)*df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100)

df_furnace['Furnace_Effluent_C2H6'] = (df_furnace['DMCTF_feed']/1000)*(df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100) * (1 - df_furnace['Furnace_conversion'])

df_furnace['Furnace_Effluent_C2H6_wt%'] =(df_furnace['Furnace_Effluent_C2H6']/(df_furnace['DMCTF_feed']/1000))*100

df_furnace['Plant_average_feed_rate_Coil'] = df_furnace["DMCTF_feed"]/(df_furnace["Number_Of_Furnaces_Online"]*4)

df_furnace['Coil_CIP_Calculated'] = (-131.3081 +
        (0.0755 * df_furnace['Plant_average_feed_rate_Coil']) +
        (0.1463 * df_furnace['Ethane_Feed_Preheater_Ethane_Feed_Outlet_Pressure']) +
        (0.6819 * df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_Outlet_Temperature']) +
        (0.4853 * df_furnace['Coil_Weighted_Avg_Feed_CV_opening']) +
        (0.8766 * df_furnace['Coil_Weighted_Avg_Steam_CV_opening']))

df_furnace['Coil_Steam_Flow'] = (df_furnace['Coil_Avg_SHC_Ratio']*df_furnace['Plant_average_feed_rate_Coil'])
       
df_furnace['Coil_Mixed_Feed_flow'] = df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil']

df_furnace['Coil_Mixed_Feed_Cp'] = (
    (df_furnace['Coil_Steam_Flow'] * 2.067) + 
    (df_furnace['Plant_average_feed_rate_Coil'] * 1.909)
) / (df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil'])   

df_furnace['Coil_Mixed_Feed_Mol_wt'] = (
    df_furnace['Coil_Mixed_Feed_flow'] / 
    (
        (df_furnace['Plant_average_feed_rate_Coil'] / df_furnace["Furnace_Feed_Average_Molecular_Wt"]) + 
        (df_furnace['Coil_Steam_Flow'] / 18.0)
    )
)

df_furnace['Coil_Volumetric_Flow'] = (
    df_furnace['Coil_Mixed_Feed_flow']
) / (
    ((df_furnace['Coil_CIP_Calculated'] + 101.325) * 
      0.00982963 * df_furnace['Coil_Mixed_Feed_Mol_wt'] ) / 
    (0.08206 * (df_furnace['Coil_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] + 273.15))
)

df_furnace['Coil_CIP_Corrected_atma'] = np.where(
    (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) < 5,
    (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
    (df_furnace['Coil_Volumetric_Flow'] * 144 / 1309.83) * 0.00986923,
    (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
    (df_furnace['Coil_Volumetric_Flow'] * 131 / 1209.52) * 0.00986923
)

selected_col = ['Coil_Avg_COT','Ethane_Feed_Preheater_Ethane_Feed_Outlet_Pressure', 
                'Furnace_Ethane_Feed_Preheater_Ethane_Feed_Outlet_Temperature','Coil_Weighted_Avg_Feed_CV_opening', 
                'Coil_Weighted_Avg_Steam_CV_opening', 'Coil_Avg_SHC_Ratio','Coil_Weighted_Avg_COP',
                'Coil_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature',"Furnace_Feed_Average_Molecular_Wt",
                'Coil_CIP_Calculated','Coil_Steam_Flow', 'Coil_Mixed_Feed_flow','Coil_Mixed_Feed_Cp',
                'Coil_Mixed_Feed_Mol_wt', 'Coil_Volumetric_Flow', 'Coil_CIP_Corrected_atma',
                'Furnace_Normalised_Feed_C2H6_Wt','Furnace_Normalised_Feed_C3H8_Wt',"Number_Of_Furnaces_Online",
                'Furnace_conversion','Furnace_Effluent_C2H6','Furnace_Effluent_C2H6_wt%']


df_furnace_truncate = df_furnace[selected_col]

df_updated = df.merge(df_furnace_truncate, how="inner", left_index=True, right_index=True)   

df = df_updated.copy()


# df_furnace = pd.read_excel(
#     os.path.join(file_path, "DMC_Screen_tags_data.xlsx"),
#     sheet_name="Furnace data"
# )

# df_furnace.columns = df_furnace.iloc[0]  # Set first row as header
# df_furnace = df_furnace[1:].reset_index(drop=True)
# df_furnace.columns = df_furnace.columns.str.strip()  # Clean column names
# df_furnace.set_index("Timestamp", inplace=True)   

# feed_cols = [f'F{furnace}_Furnace_Feed_Rate_Coil{i}' 
#              for furnace in range(1, 13) 
#              for i in range(1, 5)]

# # Keep only rows where at least one feed column is non-zero
# df_furnace = df_furnace[(df_furnace[feed_cols] != 0).any(axis=1)]
# df_furnace = df_furnace.apply(pd.to_numeric, errors='coerce', axis=0)


# # Total furnaces
# num_furnaces = 12
# num_coils = 4

# for furnace in range(1, 13):
#     cot_cols = [f'F{furnace}_Furnace_Coil_Outlet_Temperature_Coil{i}' for i in range(1, 5)]
#     feed_cols = [f'F{furnace}_Furnace_Feed_Rate_Coil{i}' for i in range(1, 5)]
#     cip_cols = [f'F{furnace}_Coil{i}_CIP_Corrected_atma' for i in range(1, 5)]
#     shc_cols = [f'F{furnace}_Furnace_Coil{i}_SHC_Ratio' for i in range(1, 5)]
#     Feed_CV_opening_cols =  [f'F{furnace}_Furnace_Feed_Coil{i}_CV_Opening' for i in range(1, 5)]
#     Steam_CV_opening_cols =  [f'F{furnace}_Furnace_Dilution_Steam_Coil{i}_CV_Opening' for i in range(1, 5)]
#     Coil_Mixed_Feed_Inlet_Temperature_cols = [f'F{furnace}_Coil{i}_Mixed_Feed_Inlet_Temperature' for i in range(1, 5)]
#     status_col = f'F{furnace}_online_status'

#     if all(col in df_furnace.columns for col in cot_cols + feed_cols + cip_cols + shc_cols + Feed_CV_opening_cols + Steam_CV_opening_cols + Coil_Mixed_Feed_Inlet_Temperature_cols + [status_col]):
#         feed_total = df_furnace[feed_cols].sum(axis=1)

#         # Use df_furnace consistently
#         df_furnace[f'F{furnace}_Weighted_Avg_COT'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[cot] * df_furnace[feed] for cot, feed in zip(cot_cols, feed_cols)) / feed_total,
#             np.nan
#         )

#         df_furnace[f'F{furnace}_Weighted_Avg_CIP'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[cip] * df_furnace[feed] for cip, feed in zip(cip_cols, feed_cols)) / feed_total,
#             np.nan
#         )

#         df_furnace[f'F{furnace}_Weighted_Avg_SHC_Ratio'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[shc] * df_furnace[feed] for shc, feed in zip(shc_cols, feed_cols)) / feed_total,
#             np.nan
#         )
        
#         df_furnace[f'F{furnace}_Weighted_Avg_Feed_CV_opening'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[Feed_CV_opening] * df_furnace[feed] for Feed_CV_opening, feed in zip(Feed_CV_opening_cols, feed_cols)) / feed_total,
#             np.nan
#         )
        
#         df_furnace[f'F{furnace}_Weighted_Avg_Steam_CV_opening'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[Steam_CV_opening] * df_furnace[feed] for Steam_CV_opening, feed in zip(Steam_CV_opening_cols, feed_cols)) / feed_total,
#             np.nan
#         )
        
#         df_furnace[f'F{furnace}_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] = np.where(
#             (df_furnace[status_col] == 1) & (feed_total > 0),
#             sum(df_furnace[Coil_Mixed_Feed_Inlet_Temperature] * df_furnace[feed] for Coil_Mixed_Feed_Inlet_Temperature, feed in zip(Coil_Mixed_Feed_Inlet_Temperature_cols, feed_cols)) / feed_total,
#             np.nan
#         )
        
#     else:
#         df_furnace[f'F{furnace}_Weighted_Avg_COT'] = np.nan
#         df_furnace[f'F{furnace}_Weighted_Avg_CIP'] = np.nan
#         df_furnace[f'F{furnace}_Weighted_Avg_SHC_Ratio'] = np.nan 
#         df_furnace[f'F{furnace}_Weighted_Avg_Feed_CV_opening'] = np.nan 
#         df_furnace[f'F{furnace}_Weighted_Avg_Steam_CV_opening'] = np.nan 
#         df_furnace[f'F{furnace}_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] = np.nan 


# #weighted average COP calculations
# for furnace in range(1, 13):
#     # Define Feed Columns
#     feed_1 = f'F{furnace}_Furnace_Feed_Rate_Coil1'
#     feed_2 = f'F{furnace}_Furnace_Feed_Rate_Coil2'
#     feed_3 = f'F{furnace}_Furnace_Feed_Rate_Coil3'
#     feed_4 = f'F{furnace}_Furnace_Feed_Rate_Coil4'
    
#     # Define COP Columns
#     cop_tlea = f'F{furnace}_Corrected_COP_Furnace_TLEA'
#     cop_tleb = f'F{furnace}_Corrected_COP_Furnace_TLEB'
    
#     status_col = f'F{furnace}_online_status'
    
#     # Check if columns exist
#     required_cols = [feed_1, feed_2, feed_3, feed_4, cop_tlea, cop_tleb, status_col]
    
#     if all(col in df_furnace.columns for col in required_cols):
#         # Calculate specific feed sums
#         feed_tlea = df_furnace[feed_1] + df_furnace[feed_2]  # Coils 1 & 2
#         feed_tleb = df_furnace[feed_3] + df_furnace[feed_4]  # Coils 3 & 4
#         total_feed = feed_tlea + feed_tleb
        
#         # Condition: Online and Total Feed > 0
#         condition = (df_furnace[status_col] == 1) & (total_feed > 0)
        
#         # Weighted Average Calculation
#         df_furnace[f'F{furnace}_Weighted_Avg_COP'] = np.where(
#             condition,
#             (
#                 (df_furnace[cop_tlea] * feed_tlea) + 
#                 (df_furnace[cop_tleb] * feed_tleb)
#             ) / total_feed,
#             np.nan
#         )
#     else:
#         df_furnace[f'F{furnace}_Weighted_Avg_COP'] = np.nan

# # Optional: Calculate Plant Average COP (Simple Mean of online furnaces)
# # Define column lists for the 12 furnaces
# cot_cols = [f'F{i}_Weighted_Avg_COT' for i in range(1, 13)]
# cip_cols = [f'F{i}_Weighted_Avg_CIP' for i in range(1, 13)]
# shc_cols = [f'F{i}_Weighted_Avg_SHC_Ratio' for i in range(1, 13)]
# Feed_CV_opening_cols = [f'F{i}_Weighted_Avg_Feed_CV_opening' for i in range(1, 13)]
# Steam_CV_opening_cols = [f'F{i}_Weighted_Avg_Steam_CV_opening' for i in range(1, 13)]
# Coil_Mixed_Feed_Inlet_Temperature_cols = [f'F{i}_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature' for i in range(1, 13)]
# cop_cols = [f'F{i}_Weighted_Avg_COP' for i in range(1, 13)]

# # Calculate simple average across all furnaces (ignores NaN values automatically)
# df_furnace['Plant_Avg_COT'] = df_furnace[cot_cols].mean(axis=1, skipna=True)
# df_furnace['Plant_Avg_CIP'] = df_furnace[cip_cols].mean(axis=1,skipna=True)
# df_furnace['Plant_Avg_SHC_Ratio'] = df_furnace[shc_cols].mean(axis=1,skipna=True)
# df_furnace['Plant_Weighted_Avg_COP'] = df_furnace[cop_cols].mean(axis=1,skipna=True)   

# df_furnace['Plant_Weighted_Avg_Feed_CV_opening'] = df_furnace[Feed_CV_opening_cols].mean(axis=1,skipna=True)  
# df_furnace['Plant_Weighted_Avg_Steam_CV_opening'] = df_furnace[Steam_CV_opening_cols].mean(axis=1,skipna=True)  
# df_furnace['Plant_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] = df_furnace[Coil_Mixed_Feed_Inlet_Temperature_cols].mean(axis=1,skipna=True)  

# #Average molecular weight calculation
# df_furnace["Furnace_Sum_of_Feed_Components"] = (df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_CH4"] + 
#                                           df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_C2H6"] + 
#                                           df_furnace["Furnace_Ethane_Feed_Preheater_Ethane_Feed_C3H8"])

# df_furnace['Furnace_Normalised_Feed_CH4'] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_CH4']*100/df_furnace["Furnace_Sum_of_Feed_Components"]
# df_furnace["Furnace_Normalised_Feed_C2H6"] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_C2H6']*100/df_furnace["Furnace_Sum_of_Feed_Components"]
# df_furnace["Furnace_Normalised_Feed_C3H8"] = df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_C3H8']*100/df_furnace["Furnace_Sum_of_Feed_Components"]

# df_furnace['Furnace_Normalised_Feed_CH4_Wt'] = df_furnace['Furnace_Normalised_Feed_CH4']*16*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)
# df_furnace['Furnace_Normalised_Feed_C2H6_Wt'] = df_furnace['Furnace_Normalised_Feed_C2H6']*30*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)
# df_furnace['Furnace_Normalised_Feed_C3H8_Wt'] = df_furnace['Furnace_Normalised_Feed_C3H8']*44*100/(df_furnace['Furnace_Normalised_Feed_CH4']*16+df_furnace["Furnace_Normalised_Feed_C2H6"]*30+df_furnace["Furnace_Normalised_Feed_C3H8"]*44)

# df_furnace["Furnace_Feed_Average_Molecular_Wt"] = df_furnace['Furnace_Normalised_Feed_CH4']/100*16+df_furnace["Furnace_Normalised_Feed_C2H6"]/100*30+df_furnace["Furnace_Normalised_Feed_C3H8"]/100*44

# df_furnace['Furnace_conversion'] = ((df_furnace['DMCTF_feed']/1000)*(df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100) - 
#                             (df_furnace['DMCTF_feed']/1000-df_furnace['Fresh ethane feed']))/((df_furnace['DMCTF_feed']/1000)*df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100)

# df_furnace['Furnace_Effluent_C2H6'] = (df_furnace['DMCTF_feed']/1000)*(df_furnace['Furnace_Normalised_Feed_C2H6_Wt']/100) * (1 - df_furnace['Furnace_conversion'])

# df_furnace['Furnace_Effluent_C2H6_wt%'] =(df_furnace['Furnace_Effluent_C2H6']/(df_furnace['DMCTF_feed']/1000))*100

# df_furnace['Plant_average_feed_rate_Coil'] = df_furnace["DMCTF_feed"]/(df_furnace["Number_Of_Furnaces_Online"]*4)

# df_furnace['Coil_CIP_Calculated'] = (-131.3081 +
#         (0.0755 * df_furnace['Plant_average_feed_rate_Coil']) +
#         (0.1463 * df_furnace['Ethane_Feed_Preheater_Ethane_Feed_Outlet_Pressure']) +
#         (0.6819 * df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_Outlet_Temperature']) +
#         (0.4853 * df_furnace['Plant_Weighted_Avg_Feed_CV_opening']) +
#         (0.8766 * df_furnace['Plant_Weighted_Avg_Steam_CV_opening']))

# df_furnace['Coil_Steam_Flow'] = (df_furnace['Plant_Avg_SHC_Ratio']*df_furnace['Plant_average_feed_rate_Coil'])
       
# df_furnace['Coil_Mixed_Feed_flow'] = df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil']

# df_furnace['Coil_Mixed_Feed_Cp'] = (
#     (df_furnace['Coil_Steam_Flow'] * 2.067) + 
#     (df_furnace['Plant_average_feed_rate_Coil'] * 1.909)
# ) / (df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil'])   

# df_furnace['Coil_Mixed_Feed_Mol_wt'] = (
#     df_furnace['Coil_Mixed_Feed_flow'] / 
#     (
#         (df_furnace['Plant_average_feed_rate_Coil'] / df_furnace["Furnace_Feed_Average_Molecular_Wt"]) + 
#         (df_furnace['Coil_Steam_Flow'] / 18.0)
#     )
# )

# df_furnace['Coil_Volumetric_Flow'] = (
#     df_furnace['Coil_Mixed_Feed_flow']
# ) / (
#     ((df_furnace['Coil_CIP_Calculated'] + 101.325) * 
#       0.00982963 * df_furnace['Coil_Mixed_Feed_Mol_wt'] ) / 
#     (0.08206 * (df_furnace['Plant_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] + 273.15))
# )

# df_furnace['Coil_CIP_Corrected_atma'] = np.where(
#     (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) < 5,
#     (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
#     (df_furnace['Coil_Volumetric_Flow'] * 144 / 1309.83) * 0.00986923,
#     (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
#     (df_furnace['Coil_Volumetric_Flow'] * 131 / 1209.52) * 0.00986923
# )

# selected_col = ['Plant_Avg_COT','Ethane_Feed_Preheater_Ethane_Feed_Outlet_Pressure', 
#                 'Furnace_Ethane_Feed_Preheater_Ethane_Feed_Outlet_Temperature','Plant_Weighted_Avg_Feed_CV_opening', 
#                 'Plant_Weighted_Avg_Steam_CV_opening', 'Plant_Avg_SHC_Ratio','Plant_Weighted_Avg_COP',
#                 'Plant_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature',"Furnace_Feed_Average_Molecular_Wt",
#                 'Coil_CIP_Calculated','Coil_Steam_Flow', 'Coil_Mixed_Feed_flow','Coil_Mixed_Feed_Cp',
#                 'Coil_Mixed_Feed_Mol_wt', 'Coil_Volumetric_Flow', 'Coil_CIP_Corrected_atma',
#                 'Furnace_Normalised_Feed_C2H6_Wt','Furnace_Normalised_Feed_C3H8_Wt',"Number_Of_Furnaces_Online",
#                 'Furnace_conversion','Furnace_Effluent_C2H6','Furnace_Effluent_C2H6_wt%']


# df_furnace_truncate = df_furnace[selected_col]

# df_updated = df.merge(df_furnace_truncate, how="inner", left_index=True, right_index=True)   

# df = df_updated.copy()

#%% Config file loading 
config_df_model_details = pd.read_excel(file_path +"/" + "Config_file.xlsx", sheet_name= 'Model details')

#%% Perform the state space model (kalman filter) for quench tower overhead prediction

y_col = "Quench_tower_overhead_temp"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X = df[u_cols].values
y = df[y_col].values

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

# System identification
nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)   

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for Quench tower OH temp')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (no input)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted (Kalman)', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Overhead Temp (°C)')
plt.title('Quench_tower_overhead_temperature Prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f} °C")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Quench_OH_temp_pred.pkl', 'wb') as f:
    pickle.dump(kalman, f)

# Load the model later
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Quench_OH_temp_pred.pkl', 'rb') as f:
    Quench_OH_temp_kalman = pickle.load(f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_Quench_tower_OH_temp.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_Quench_tower_OH_temp.pkl')


#%% CGC_STAGE_1_SUCTION_PRESSURE prediction

"""For a continuous-time LTI system:
ẋ(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)   
"""

"""
For a discrete-time linear time-invariant (LTI) system:
x[k+1] = A x[k] + B u[k]   (State equation)
y[k]   = C x[k] + D u[k]   (Output equation)
"""    

"""
Where:
x = state vector
u = input vector
y = output vector
A = State matrix: governs internal dynamics (how states evolve over time)
B = Input matrix: determines how inputs affect states
C = Output matrix: maps states to outputs
D = Feedthrough (or direct transmission) matrix: represents direct influence of input on output (without delay) 
"""

"""
Interpretation:
A: System dynamics (eigenvalues determine stability and response modes)
B: Controllability — which states can be influenced by inputs
C: Observability — which states are visible in the outputs
D: Immediate feedthrough; if zero, output depends only on states (not directly on input)
"""

# u_cols = ['DMCTF_feed','Cooling water supply temp', 'Quench_tower_overhead_temp',
#           'CGC_TURBINE_1_SPEED_(RPM)','CGC_TURBINE_2_SPEED_(RPM)',
#           'Average interstage differential pressure'
#         ]

# #'CGC_5TH_STG_DISCH_PRES'
# y_col = 'CGC_STAGE_1_SUCTION_PRESSURE'

y_col = "CGC_STAGE_1_SUCTION_PRESSURE"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()


# df[u_cols] = df[u_cols].apply(pd.to_numeric, errors='coerce')
# df[y_col] = pd.to_numeric(df[y_col], errors='coerce')  
 
# df = df[df['CGC_TURBINE_1_SPEED_(RPM)']>4400]
# df = df[(df['DMCTF_feed'] > 195000) & (df['DMCTF_feed'] < 196000)]

# split_idx = int(0.9 * len(df))
# train_data = df.iloc[:split_idx]
# test_data = df.iloc[split_idx:]

# Define max threshold (e.g., 95% of max operational limit)
# threshold_1301 = 0.993 * df['CGC_TURBINE_1_SPEED_(RPM)'].max()
# threshold_1302 = 0.993 * 8050
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# X_train = train_data[u_cols]
# y_train = train_data[y_col]
# X_test = test_data[u_cols]
# y_test = test_data[y_col]   

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy() 

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)   

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for CGC_STAGE_1_SUCTION_PRESSURE')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

# for i in range(len(test_data)):
#     u_step = X_scaled_test[i].reshape(-1, 1)
#     y_step = y_scaled_test[i].reshape(-1, 1)
#     kalman.step(y=y_step, u=u_step)

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)


results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted (Kalman)', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('CGC_STAGE_1_SUCTION_PRESSURE (KPA)')
plt.title('CGC_STAGE_1_SUCTION_PRESSURE Prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_Suction_press_pred.pkl', 'wb') as f:
    pickle.dump(kalman, f)

# Load the model later
with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_Suction_press_pred.pkl', 'rb') as f:
    CGC_Suction_pressure_kalman = pickle.load(f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_CGC_Suction_press_pred.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_CGC_Suction_press_pred.pkl')


#%% CGC stage 5 pressure prediction with DMCTF_feed
# u_cols = ['DMCTF_feed']  # One input
# #'CGC_TURBINE_2_SPEED_(RPM)', 'CGC_STAGE_1_SUCTION_PRESSURE'
# y_col = 'CGC_5TH_STG_DISCH_PRES'  # One output

y_col = "CGC_5TH_STG_DISCH_PRES"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()


X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for 5th stg discharge press')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

# for i in range(len(test_data)):
#     u_step = X_scaled_test[i].reshape(-1, 1)
#     y_step = y_scaled_test[i].reshape(-1, 1)
#     kalman.step(y=y_step, u=u_step)
    
for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('CGC_5TH_STG_DISCH_PRES')
plt.title('CGC_5TH_STG_DISCH_PRES  prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_5th_stg_discharge_press.pkl', 'wb') as f:
    pickle.dump(kalman, f)

# Load the model later
with open("..\\Results\\Model"+"/"+'kalman_filter_model_5th_stg_discharge_press.pkl', 'rb') as f:
    Fifth_stg_discharge_press_kalman = pickle.load(f)
    
joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_5th_stg_discharge_press.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_5th_stg_discharge_press.pkl')
        
#%% CGC system power calculation

df["St1_water_mass_fr"] = (
    -0.00287525285953703
    + (-0.00224084161646362 * df['CGC_1ST_STG_SUCT_TEMP'])
    + (0.00033152809652647 * (df['CGC_STAGE_1_SUCTION_PRESSURE'] + 101.325))
    + (0.0000390475915032036 * (df['CGC_1ST_STG_SUCT_TEMP'] ** 2))
    + (-6.8226252627273e-07 * (df['CGC_STAGE_1_SUCTION_PRESSURE'] + 101.325) ** 2)
    + (0.0479208338577593 * (df['CGC_1ST_STG_SUCT_TEMP'] / (df['CGC_STAGE_1_SUCTION_PRESSURE'] + 101.325)))
    + (0.449625943237655 * ((df['CGC_1ST_STG_SUCT_TEMP'] / (df['CGC_STAGE_1_SUCTION_PRESSURE'] + 101.325)) ** 2))
)

df["St2_water_mass_fr"] = (
    0.0169186229823141
    + (-0.000694968091247861 * df['CGC_2ND_STG_SUCT_TEMP'])
    + (0.0000587301670470908 * (df['CGC_2ND_STG_SUCT_PRESS'] + 101.325))
    + (0.0000200596650894789 * (df['CGC_2ND_STG_SUCT_TEMP'] ** 2))
    + (-1.30323839196608e-07 * (df['CGC_2ND_STG_SUCT_PRESS'] + 101.325) ** 2)
    + (-0.178428373248992 * (df['CGC_2ND_STG_SUCT_TEMP'] / (df['CGC_2ND_STG_SUCT_PRESS'] + 101.325)))
    + (1.15771970789725 * ((df['CGC_2ND_STG_SUCT_TEMP'] / (df['CGC_2ND_STG_SUCT_PRESS'] + 101.325)) ** 2))
)

df["St3_water_mass_fr"] = (
    0.0130685542704302
    + (-0.000270271056650772 * df['CGC_3RD_STG_SUCT_TEMP'])
    + (0.0000111540866084984 * (df['CGC_3RD_STG_SUCT_PRESS'] + 101.325))
    + (0.0000107241662139917 * (df['CGC_3RD_STG_SUCT_TEMP'] ** 2))
    + (-2.09094703957513e-08 * (df['CGC_3RD_STG_SUCT_PRESS'] + 101.325) ** 2)
    + (-0.250942260213592 * (df['CGC_3RD_STG_SUCT_TEMP'] / (df['CGC_3RD_STG_SUCT_PRESS'] + 101.325)))
    + (2.3090125662945 * ((df['CGC_3RD_STG_SUCT_TEMP'] / (df['CGC_3RD_STG_SUCT_PRESS'] + 101.325)) ** 2))
)

df["St4_water_mass_fr"] = (
    0.00725076090179286
    + (-0.000175773863726547 * df['CGC_4TH_STG_SUCT_TEMP'])
    + (3.96408122385643e-06 * (df['CGC_4TH_STG_SUCT_PRESS'] + 101.325))
    + (6.38762797715282e-06 * (df['CGC_4TH_STG_SUCT_TEMP'] ** 2))
    + (-3.90720050249725e-09 * (df['CGC_4TH_STG_SUCT_PRESS'] + 101.325) ** 2)
    + (-0.233018771286096 * (df['CGC_4TH_STG_SUCT_TEMP'] / (df['CGC_4TH_STG_SUCT_PRESS'] + 101.325)))
    + (3.90383544123553 * ((df['CGC_4TH_STG_SUCT_TEMP'] / (df['CGC_4TH_STG_SUCT_PRESS'] + 101.325)) ** 2))
)

df["St5_water_mass_fr"] = (
    0.00202308272864611
    + (-0.000116155805193373 * df['CGC_5TH_STG_SUCT_TEMP'])
    + (1.25431682713172e-06 * (df['CGC_5TH_STG_SUCT_PRESS'] + 101.325))
    + (3.06029257093502e-06 * (df['CGC_5TH_STG_SUCT_TEMP'] ** 2))
    + (-3.24761985409324e-10 * (df['CGC_5TH_STG_SUCT_PRESS'] + 101.325) ** 2)
    + (-0.139560061723164 * (df['CGC_5TH_STG_SUCT_TEMP'] / (df['CGC_5TH_STG_SUCT_PRESS'] + 101.325)))
    + (8.03052832216614 * ((df['CGC_5TH_STG_SUCT_TEMP'] / (df['CGC_5TH_STG_SUCT_PRESS'] + 101.325)) ** 2))
)

df["St1_Mass_flow"] = (df['CHG_GAS_FLOW_TO_DRYER']*(1+df['St1_water_mass_fr']))
df["St2_Mass_flow"] = (df['CHG_GAS_FLOW_TO_DRYER']*(1+df['St2_water_mass_fr']))
df["St3_Mass_flow"] = (df['CHG_GAS_FLOW_TO_DRYER']*(1+df['St3_water_mass_fr']))
df["St4_Mass_flow"] = (df['CHG_GAS_FLOW_TO_DRYER']*(1+df['St4_water_mass_fr']))
df["St5_Mass_flow"] = (df['CHG_GAS_FLOW_TO_DRYER']*(1+df['St5_water_mass_fr']))

df['St1_volm_flow'] = (
    (0.996 * 8.314 * df['St1_Mass_flow'] * 1000) * 
    (df['CGC_1ST_STG_SUCT_TEMP'] + 273.15)) / ((df['CGC_STAGE_1_SUCTION_PRESSURE'] + 101.325) * df['Avg_MW'])
          
df['St2_volm_flow'] = (
    (0.993 * 8.314 * df['St2_Mass_flow'] * 1000) * 
    (df['CGC_2ND_STG_SUCT_TEMP'] + 273.15)) / ((df['CGC_2ND_STG_SUCT_PRESS'] + 101.325) * df['Avg_MW'])     
     
df['St3_volm_flow'] = (
    (0.988 * 8.314 * df['St3_Mass_flow'] * 1000) * 
    (df['CGC_3RD_STG_SUCT_TEMP'] + 273.15)) / ((df['CGC_3RD_STG_SUCT_PRESS'] + 101.325) * df['Avg_MW'])     
     
df['St4_volm_flow'] = (
    (0.978 * 8.314 * df['St4_Mass_flow'] * 1000) * 
    (df['CGC_4TH_STG_SUCT_TEMP'] + 273.15)) / ((df['CGC_4TH_STG_SUCT_PRESS'] + 101.325) * df['Avg_MW'])     
     
df['St5_volm_flow'] = (
    (0.96 * 8.314 * df['St5_Mass_flow'] * 1000) * 
    (df['CGC_5TH_STG_SUCT_TEMP'] + 273.15)) / ((df['CGC_5TH_STG_SUCT_PRESS'] + 101.325) * df['Avg_MW'])
     
df["St1_power_KW_Perf_curve"] =  (-8.49131 * df['CGC_TURBINE_1_SPEED_(RPM)']
                                  + 0.001262* df['CGC_TURBINE_1_SPEED_(RPM)']**2
                                  + 0.049509 * df['St1_volm_flow']
                                  + 1.91501e-07 * df['St1_volm_flow']**2
                                  + -13.4744 * (df['St1_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])
                                  + -3.2998 * (df['St1_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])**2
                                  + -7.7908e-19 * df['St1_volm_flow']**4 + 14106.21 )

df["St2_power_KW_Perf_curve"] =  (-6.88124 * df['CGC_TURBINE_1_SPEED_(RPM)']
                                  + 0.001063* df['CGC_TURBINE_1_SPEED_(RPM)']**2
                                  + 0.06122 * df['St2_volm_flow']
                                  + 9.78027e-07 * df['St2_volm_flow']**2
                                  + -8.15085 * (df['St2_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])
                                  + -12.0822 * (df['St2_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])**2
                                  + -2.02379e-17 * df['St2_volm_flow']**4 + 11532.79 )

df["St3_power_KW_Perf_curve"] =  (-8.64535 * df['CGC_TURBINE_1_SPEED_(RPM)']
                                  + 0.001236* df['CGC_TURBINE_1_SPEED_(RPM)']**2
                                  + 0.281358 * df['St3_volm_flow']
                                  + 1.48066e-06 * df['St3_volm_flow']**2
                                  + -417.304 * (df['St3_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])
                                  + -31.9242 * (df['St3_volm_flow']/df['CGC_TURBINE_1_SPEED_(RPM)'])**2
                                  + -1.45881e-16 * df['St3_volm_flow']**4 + 15013.35 )

df["St4_power_KW_Perf_curve"] =  (-6.76353 * df['CGC_TURBINE_2_SPEED_(RPM)']
                                  + 0.000531* df['CGC_TURBINE_2_SPEED_(RPM)']**2
                                  + 0.798089 * df['St4_volm_flow']
                                  + 6.50734e-07 * df['St4_volm_flow']**2
                                  + -1890.13 * (df['St4_volm_flow']/df['CGC_TURBINE_2_SPEED_(RPM)'])
                                  + -395.304 * (df['St4_volm_flow']/df['CGC_TURBINE_2_SPEED_(RPM)'])**2
                                  + -5.33865e-16 * df['St4_volm_flow']**4 + 20444.53)

df["St5_power_KW_Perf_curve"] =  (-5.46075 * df['CGC_TURBINE_2_SPEED_(RPM)']
                                  + 0.00043* df['CGC_TURBINE_2_SPEED_(RPM)']**2
                                  + 1.15707 * df['St5_volm_flow']
                                  + 2.18463e-05 * df['St5_volm_flow']**2
                                  + -2291.25 * (df['St5_volm_flow']/df['CGC_TURBINE_2_SPEED_(RPM)'])
                                  + -1986.65 * (df['St5_volm_flow']/df['CGC_TURBINE_2_SPEED_(RPM)'])**2
                                  + -3.07159e-14 * df['St5_volm_flow']**4 + 16708.1)

df["St1_delta_power_dashboard_Perf_curve"] = df["CGC_Stage_1_power"] - df["St1_power_KW_Perf_curve"]
df["St2_delta_power_dashboard_Perf_curve"] = df["CGC_Stage_2_power"] - df["St2_power_KW_Perf_curve"]
df["St3_delta_power_dashboard_Perf_curve"] = df["CGC_Stage_3_power"] - df["St3_power_KW_Perf_curve"]
df["St4_delta_power_dashboard_Perf_curve"] = df["CGC_Stage_4_power"] - df["St4_power_KW_Perf_curve"]
df["St5_delta_power_dashboard_Perf_curve"] = df["CGC_Stage_5_power"] - df["St5_power_KW_Perf_curve"]    
 
df["CGC_Power_KW_Perf_curve"] = (df["St1_power_KW_Perf_curve"] + df["St2_power_KW_Perf_curve"] + df["St3_power_KW_Perf_curve"]
                                + df["St4_power_KW_Perf_curve"] + df["St5_power_KW_Perf_curve"])

df["CGC_Power_KW"] = (df["CGC_Stage_1_power"] + df["CGC_Stage_2_power"] + df["CGC_Stage_3_power"]
                                + df["CGC_Stage_4_power"] + df["CGC_Stage_5_power"])

df["CGC_delta_Power_KW_dashboard_Perf_curve"] = df["CGC_Power_KW"] - df["CGC_Power_KW_Perf_curve"] 

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(df["CGC_Power_KW"], label='Dashboard calculated power (KW)', alpha=0.7)
plt.plot(df["CGC_Power_KW_Perf_curve"], label='performace curve calculated power (KW)', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('CGC Power')
plt.title('CGC power estimation and comparison')
plt.grid(True)
plt.show()

#%% CGC power prediction
# u_cols = ['DMCTF_feed','CGC_STAGE_1_SUCTION_PRESSURE','CGC_5TH_STG_DISCH_PRES',
#           'CGC_TURBINE_1_SPEED_(RPM)','Cooling water supply temp']
# #
# y_col = 'CGC_Power_KW'

y_col = "CGC_Power_KW"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance forCGC power prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('CGC power KW')
plt.title('CGC power KW prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_power_pred.pkl', 'wb') as f:
    pickle.dump(kalman, f)

# Load the model later
with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_power_pred.pkl', 'rb') as f:
    CGC_power_pred_kalman = pickle.load(f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_CGC_power_pred.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_CGC_power_pred.pkl')


#%% CGC turbine 1 steam flow prediction
#config_df_model_details = pd.read_excel(file_path +"/" + "Config_file.xlsx", sheet_name= 'Model details')

y_col = "CGC_Turbine_HP_Steam_flow"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for CGC Turbine HP Steam flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('CGC Turbine HP Steam flow')
plt.title('CGC Turbine HP Steam flow prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_Turbine_HP_Steam_flow_pred.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_CGC_Turbine_HP_Steam_flow_pred.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_CGC_Turbine_HP_Steam_flow_pred.pkl')

#%% --------------------------Charge gas dryer flow prediction ------------------------------------

# u_cols = ['DMCTF_feed']
# y_col = 'CHG_GAS_FLOW_TO_DRYER'

y_col = "CHG_GAS_FLOW_TO_DRYER"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2, shuffle=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

common_index = train_data.index.intersection(test_data.index)
# Number of common rows
num_common = len(common_index)
print(f"Number of common rows based on timestamp index: {num_common}")

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Charge gas dryer flow')
plt.title('Charge gas dryer flow prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Charge_gas_dryer_flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_Charge_gas_dryer_flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_Charge_gas_dryer_flow.pkl')


#%% Compressor k1601 flow and power calculation

# Get density of propylene (C3H6) at T and P
Density_1st_stage =[]
rho_1st_stage_flow = []
for i in range(len(df)):
    T_K = df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15
    P_Pa = df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
    rho_1st_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000 / rho_1st_stage
    rho_1st_stage_flow.append(volumetric_flow)
    Density_1st_stage.append(rho_1st_stage)
       
df["PRC_Density_1st_stage"] = Density_1st_stage
df["PRC VOL FLOW 1ST STAGE"] = rho_1st_stage_flow


# import CoolProp.CoolProp as CP
# import numpy as np

# K1601_2ND_STAGE_Saturation_TEMP = []

# for i in range(len(df)):
#     pressure_kPa = df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i]
    
#     # Check for NaN
#     if pd.isna(pressure_kPa):
#         K1601_2ND_STAGE_Saturation_TEMP.append(np.nan)
#         continue
    
#     P_Pa = pressure_kPa * 1000   # Convert to absolute pressure

#     try:
#         T_sat_K = CP.PropsSI('Tsat', 'P', P_Pa, 'Propylene')
#         T_sat_C = T_sat_K - 273.15
#     except (ValueError, KeyError, AttributeError) as e:
#         print(f"Error at index {i}, pressure = {P_Pa:.0f} Pa: {e}")
#         T_sat_C = np.nan

#     K1601_2ND_STAGE_Saturation_TEMP.append(T_sat_C)   


Density_2nd_stage =[]
rho_2nd_stage_flow = []
for i in range(len(df)):
    T_K = df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15
    P_Pa = df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5
    rho_2nd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = (df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["PRC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000 / rho_2nd_stage
    rho_2nd_stage_flow.append(volumetric_flow)
    Density_2nd_stage.append(rho_2nd_stage)
    
df["PRC_Density_2nd_stage"] = Density_2nd_stage   
df["PRC VOL FLOW 2ND STAGE"] = rho_2nd_stage_flow

Density_3rd_stage =[]
rho_3rd_stage_flow = []

for i in range(len(df)):
    T_K = df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15
    P_Pa = df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
    rho_3rd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = df["PRC_3RD_STAGE_Suction_FLOW"].iloc[i]  * 1000 / rho_3rd_stage
    rho_3rd_stage_flow.append(volumetric_flow)
    Density_3rd_stage.append(rho_3rd_stage)

df["PRC_Density_3rd_stage"] = Density_3rd_stage
df["PRC VOL FLOW 3RD STAGE"] = rho_3rd_stage_flow

#Compressor power calculation based on isentropic enthalpy at outlet pressure, keeping entropy constant (i.e., isentropic compression).
eta = 0.70  # Assume Compressor efficiency
PRC_1st_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["PRC_1ST_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["PRC_1ST_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
    PRC_1st_stage_comp_estimated_power.append(power)
    
df["PRC_1st_stage_comp_estimated_power_MW"] = PRC_1st_stage_comp_estimated_power

# 2nd stage compressor power
PRC_2nd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = (((df["PRC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i]) * 1000)/3600) * (h2 - h1)/1e6  # MW
    PRC_2nd_stage_comp_estimated_power.append(power)
    
df["PRC_2nd_stage_comp_estimated_power_MW"] = PRC_2nd_stage_comp_estimated_power

# 3rd stage compressor power
PRC_3rd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["PRC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["PRC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df["PRC_3RD_STAGE_Suction_FLOW"].iloc[i]* 1000)/3600) * (h2 - h1)/1e6  # MW
    PRC_3rd_stage_comp_estimated_power.append(power)
    
df["PRC_3rd_stage_comp_estimated_power_MW"] = PRC_3rd_stage_comp_estimated_power

df["PRC_Total_estimated_power_MW"] = df["PRC_1st_stage_comp_estimated_power_MW"]+df["PRC_2nd_stage_comp_estimated_power_MW"]+df["PRC_3rd_stage_comp_estimated_power_MW"]


#%% KT-1601 dynamic simulation

PRC_turbine_current_steam_enthalpy_KJ_Kg = []
PRC_turbine_current_steam_entropy_KJ_KgK =[]
PRC_turbine_current_outlet_ethalpy_KJ_Kg =[]
PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg =[]
PRC_turbine_current_power_gen_extraction_MW = []
PRC_turbine_current_power_gen_exhaust_MW =[]
PRC_turbine_current_Turbine_power_MW_based_on_EE = []
PRC_turbine_current_Turbine_power_MW_based_on_steam_flow = []
PRC_turbine_current_Specific_steam_consumption_MT_MW =[]
#turbine_efficiency =[]

for i in range(len(df)):
    Steam_flow_TPH = df['PRC_turbine_steam_flow'].iloc[i]
    Condensate_flow_TPH = df['PRC_turbine_condensate_flow'].iloc[i]
    Extraction_flow_TPH = df['PRC_turbine_Extraction_flow'].iloc[i]
    
    Steam_flow_Kg_hr       = Steam_flow_TPH *1000
    Extraction_flow_Kg_hr  = Extraction_flow_TPH *1000
    Condensate_flow_Kg_hr  = Condensate_flow_TPH *1000

    # Steam Inlet conditions
    P_steam = df['PRC_turbine_Steam_pressure'].iloc[i]*1000   # Pa
    T_steam = df['PRC_turbine_Steam_Temp'].iloc[i] +273.15   # K
    T_sat_steam = PropsSI('T', 'P', P_steam, 'Q', 1, 'Water') - 273.15 # degC

    # Extraction pressure
    Pe = df['PRC_turbine_Extraction_Pressure'].iloc[i]*1000      # Pa 
    T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K

    # Condenser pressure
    Pc = df['PRC_turbine_Condensate_Pressure'].iloc[i]*1000
    Tc_actual = df['PRC_turbine_Condensate_Temperature'].iloc[i] +273.15  # K (actual exhaust steam temp)
    T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
    
    # Steam (inlet) Enthalpy calculation 
    h_steam = PropsSI('H', 'P', P_steam, 'T', T_steam,'Water')/1000   # kJ/kg
    s_steam = PropsSI('S', 'P', P_steam, 'T', T_steam, 'Water')/1000  # KJ/kg·K
    
    # Actual outlet enthalpies
    # Actual outlet enthalpies
    # he = PropsSI('H', 'P', Pe, 'T', Te_actual, 'Water')/1000  # Extracted steam (actual)
    he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
    #hc = PropsSI('H', 'P', Pc, 'T', Tc_actual, 'Water')/1000  # Condenser outlet (actual)
    hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
    dryness_fraction = 0.92
    hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)

    hc = hc_liquid + hc_vapor*dryness_fraction

    Net_heat_release = Steam_flow_Kg_hr*h_steam -Extraction_flow_Kg_hr*he - Condensate_flow_Kg_hr*hc
    power_gen_extraction = Extraction_flow_Kg_hr*(h_steam-he)/3600/1000
    power_gen_exhaust = Condensate_flow_Kg_hr*(h_steam-hc)/3600/1000
    Turbine_power_MW_EE = power_gen_extraction + power_gen_exhaust
    Specific_steam_consumption = Steam_flow_TPH/(Turbine_power_MW_EE) # MT/MW

    def isentropic_enthalpy(P_target, s_in):
        # Keep everything in J/kg·K
        s_f = PropsSI("S", "P", P_target, "Q", 0, "Water")/1000
        s_g = PropsSI("S", "P", P_target, "Q", 1, "Water")/1000
    
        if s_f < s_in < s_g:
            x = (s_in - s_f) / (s_g - s_f)
            h_f = PropsSI("H", "P", P_target, "Q", 0, "Water")
            h_g = PropsSI("H", "P", P_target, "Q", 1, "Water")
            h_iso = h_f + x * (h_g - h_f)
        else:
            h_iso = PropsSI("H", "P", P_target, "S", s_in, "Water")
    
        return h_iso / 1000  # Convert to kJ/kg


    # Isentropic outlet enthalpies
    he_s = isentropic_enthalpy(Pe, s_steam)
    hc_s = isentropic_enthalpy(Pc, s_steam)
    
    # Mass-weighted outlet enthalpies
    h2_actual = (Extraction_flow_Kg_hr * he + Condensate_flow_Kg_hr * hc) / Steam_flow_Kg_hr
    h2s_ideal = (Extraction_flow_Kg_hr  * he_s + Condensate_flow_Kg_hr * hc_s) / Steam_flow_Kg_hr
    
    # Turbine isentropic efficiency
    efficiency = (h_steam - h2_actual) / (h_steam - h2s_ideal)*100
    
    turbine_power_MW_SF = (h_steam - h2_actual) * (Steam_flow_Kg_hr / (3600*1000))  # Convert kg/hr to kg/s
    
    PRC_turbine_current_steam_enthalpy_KJ_Kg.append(h_steam)
    PRC_turbine_current_steam_entropy_KJ_KgK.append(s_steam)
    PRC_turbine_current_outlet_ethalpy_KJ_Kg.append(h2_actual)
    PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg.append(h2s_ideal)
    PRC_turbine_current_power_gen_extraction_MW.append(power_gen_extraction)
    PRC_turbine_current_power_gen_exhaust_MW.append(power_gen_exhaust)
    PRC_turbine_current_Turbine_power_MW_based_on_EE.append(Turbine_power_MW_EE)
    PRC_turbine_current_Turbine_power_MW_based_on_steam_flow.append(turbine_power_MW_SF)
    PRC_turbine_current_Specific_steam_consumption_MT_MW.append(Specific_steam_consumption)
    #turbine_efficiency.append(efficiency)
    
df["PRC_turbine_current_steam_enthalpy_KJ_Kg"] = PRC_turbine_current_steam_enthalpy_KJ_Kg
df["PRC_turbine_current_steam_entropy_KJ_KgK"] = PRC_turbine_current_steam_entropy_KJ_KgK
df["PRC_turbine_current_outlet_ethalpy_KJ_Kg"] = PRC_turbine_current_outlet_ethalpy_KJ_Kg
df["PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg"] = PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg
df["PRC_turbine_current_power_gen_extraction_MW"] = PRC_turbine_current_power_gen_extraction_MW
df["PRC_turbine_current_power_gen_exhaust_MW"] = PRC_turbine_current_power_gen_exhaust_MW
df["PRC_turbine_current_Turbine_power_MW_based_on_EE"] = PRC_turbine_current_Turbine_power_MW_based_on_EE
df["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"] = PRC_turbine_current_Turbine_power_MW_based_on_steam_flow
df["PRC_turbine_current_Specific_steam_consumption_MT_MW"] = PRC_turbine_current_Specific_steam_consumption_MT_MW
#df["turbine_efficiency"] = turbine_efficiency

#%% optimization of steam and extraction flow for PRC_turbine

PRC_turbine_optimized_extraction = []
PRC_turbine_calculated_steam_flow = []
PRC_turbine_matched_power_EE = []
PRC_turbine_matched_power_SF = []
PRC_turbine_matched_h2_actual = []

def match_actual_power(row):
    try:
        # Steam inlet enthalpy
        P_steam = row["PRC_turbine_Steam_pressure"] * 1000  # Pa
        T_steam = row["PRC_turbine_Steam_Temp"] + 273.15    # K
        h_steam = PropsSI("H", "P", P_steam, "T", T_steam, "Water") / 1000  # kJ/kg
     
        # Actual extracted steam enthalpy (he)
        Pe = row["PRC_turbine_Extraction_Pressure"] * 1000  # Pa
        # Te_actual = row["Extraction Temperature"] + 273.15  # K
        T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K
        he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
        
        # Actual condenser steam enthalpy (hc)
        Pc = row["PRC_turbine_Condensate_Pressure"] * 1000  # Pa
        # Tc_actual = row["Condensate Temperature"] + 273.15  # K
        T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
        hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
        dryness_fraction = 0.92
        hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)
    
        hc = hc_liquid + hc_vapor*dryness_fraction
    
        condensate_flow_TPH = row["PRC_turbine_condensate_flow"]
        condensate_flow_Kg_hr = condensate_flow_TPH * 1000
        actual_power = row["PRC_Total_estimated_power_MW"]
    
        def objective(extraction_flow_TPH):
            extraction_flow_Kg_hr = extraction_flow_TPH * 1000
            steam_flow_Kg_hr = extraction_flow_Kg_hr + condensate_flow_Kg_hr
    
            power = ((extraction_flow_Kg_hr * (h_steam - he)) + 
                     (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000
            return (power - actual_power) ** 2
    
        result = minimize_scalar(objective, bounds=(10, 350), method='bounded')
    
        if result.success:
            ef_opt = result.x
            ef_Kg_hr = ef_opt * 1000
            sf_opt = ef_opt + condensate_flow_TPH
            sf_Kg_hr = sf_opt * 1000
            # Matched power via energy balance
            power_matched_EE = ((ef_Kg_hr * (h_steam - he)) + (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000

            # Mass-weighted outlet enthalpy
            h2_actual = (ef_Kg_hr * he + condensate_flow_Kg_hr * hc) / sf_Kg_hr

            # Power via inlet/outlet enthalpy and total flow
            power_matched_SF = (h_steam - h2_actual) * (sf_Kg_hr / 3600 / 1000)

            return ef_opt, sf_opt, power_matched_EE, power_matched_SF, h2_actual
        else:
            return (row["PRC_turbine_Extraction_flow"], row["PRC_turbine_steam_flow"],
                    row["PRC_turbine_current_Turbine_power_MW_based_on_EE"], row["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"],
                    None)
    except Exception as e:
        print(f"Error at index {row.name}: {e}")
        return (row["PRC_turbine_Extraction_flow"], row["PRC_turbine_steam_flow"],
                row["PRC_turbine_current_Turbine_power_MW_based_on_EE"], row["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"],
                None)
    
# Loop and apply
for _, row in df.iterrows():
    ef, sf, power_EE, power_SF, h2 = match_actual_power(row)
    PRC_turbine_optimized_extraction.append(ef)
    PRC_turbine_calculated_steam_flow.append(sf)
    PRC_turbine_matched_power_EE.append(power_EE)
    PRC_turbine_matched_power_SF.append(power_SF)
    PRC_turbine_matched_h2_actual.append(h2)

# Save to DataFrame
df["PRC_turbine_Optimized_Extraction_flow_TPH"] = PRC_turbine_optimized_extraction
df["PRC_turbine_Calculated_Steam_flow_TPH"] = PRC_turbine_calculated_steam_flow
df["PRC_turbine_Matched_Turbine_power_MW_EE"] = PRC_turbine_matched_power_EE
df["PRC_turbine_Matched_Turbine_power_MW_SF"] = PRC_turbine_matched_power_SF
df["PRC_turbine_Matched_h2_actual_KJ_Kg"] = PRC_turbine_matched_h2_actual
df["Power_Error"] = (
    df["PRC_turbine_Matched_Turbine_power_MW_EE"] -
    df["PRC_Total_estimated_power_MW"]
)
df["Power_EE_vs_SF_Diff"] = (
    df["PRC_turbine_Matched_Turbine_power_MW_EE"] -
    df["PRC_turbine_Matched_Turbine_power_MW_SF"]
)
df["Devaiation in steam flow (Simulated-actual)"] = df["PRC_turbine_Calculated_Steam_flow_TPH"]- df["PRC_turbine_steam_flow"]
df["Devaiation in extraction (Simulated-actual)"] = df["PRC_turbine_Optimized_Extraction_flow_TPH"]- df["PRC_turbine_Extraction_flow"]
df["Specific_steam_consumption_MT_MW_updated"] = df["PRC_turbine_Calculated_Steam_flow_TPH"] / df["PRC_Total_estimated_power_MW"]

# df.to_excel("..\\Results"+"/"+"Yanpet_OLF1_MultiX_multiY_results.xlsx")

# df_corr = df.corr (method = "pearson").round(2)

# def color_columns(val):
#     if val > 0.4:
#         color = 'background-color: green'
#     elif val < -0.4:
#         color = 'background-color: red'
#     else:
#         color = ''
#     return color

# with pd.ExcelWriter("..\\Results" +'/'+'correlation_with_color_conditioning.xlsx', engine='openpyxl') as writer:
#     # Write the DataFrame to the Excel file with styling
#     df_corr.style.applymap(color_columns).to_excel(writer, index=True)

#%% -------------- PRC 1st stage suction flow prediction
# u_cols = ['DMCTF_feed', 'CGC_5TH_STG_DISCH_PRES', 'CGC_STAGE_1_SUCTION_PRESSURE','PRC_turbine Turbine RPM']

# y_col = 'PRC_1ST_STAGE_Suction_FLOW'

y_col = "PRC_1ST_STAGE_Suction_FLOW"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for PRC 1st stage suction flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Measured', alpha=0.7)
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('PRC 1st stage suction flow')
plt.title('PRC 1st stage suction flow prediction')
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_1st_stage_suction_flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

# Load the model later
with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_1st_stage_suction_flow.pkl', 'rb') as f:
    CGC_power_pred_kalman = pickle.load(f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_PRC_1st_stage_suction_flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_PRC_1st_stage_suction_flow.pkl')

#%% -------------- PRC 1st stage suction pressure prediction
# u_cols = ['DMCTF_feed', 'CGC_5TH_STG_DISCH_PRES', 'CGC_STAGE_1_SUCTION_PRESSURE','PRC_turbine Turbine RPM']

# y_col = 'PRC_1ST_STAGE_Suction_PRESSURE'

y_col = "PRC_1ST_STAGE_Suction_PRESSURE"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()


X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for PRC 1st stage suction pressure prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='PRC 1st stage suction pressure Prediction',
    xaxis_title='Time Step',
    yaxis_title='PRC 1st stage suction pressure',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_1st_stage_suction_pressure.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_PRC_1st_stage_suction_pressure.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_PRC_1st_stage_suction_pressure.pkl')

#%% -------------- PRC_2nd_stage_drum_Overhead_Flow prediction
# u_cols = ['DMCTF_feed','Avg_MW','PRC_1ST_STAGE_Suction_FLOW', 'PRC_1ST_STAGE_Suction_PRESSURE',
#           'PRC_turbine Turbine RPM','PRC_1ST_STAGE_Suction_TEMP']

# y_col = 'PRC_2nd_stage_drum_Overhead_Flow'

y_col = "PRC_2nd_stage_drum_Overhead_Flow"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for PRC PRC_2nd_stage_drum_Overhead_Flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='PRC_2nd_stage_drum_Overhead_Flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='PRC_2nd_stage_drum_Overhead_Flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_2nd_stage_drum_Overhead_Flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_PRC_2nd_stage_drum_Overhead_Flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_PRC_2nd_stage_drum_Overhead_Flow.pkl')


#%% Compressor k1701 flow and power calculation

# Get density of propylene (C3H6) at T and P
Density_1st_stage =[]
rho_1st_stage_flow = []
for i in range(len(df)):
    T_K = df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15
    P_Pa = df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
    rho_1st_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
    volumetric_flow = df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000 / rho_1st_stage
    rho_1st_stage_flow.append(volumetric_flow)
    Density_1st_stage.append(rho_1st_stage)
       
df["ERC_Density_1st_stage"] = Density_1st_stage
df["ERC VOL FLOW 1ST STAGE"] = rho_1st_stage_flow

Density_2nd_stage =[]
rho_2nd_stage_flow = []
for i in range(len(df)):
    T_K = df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15
    P_Pa = df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5
    rho_2nd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
    volumetric_flow = (df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000 / rho_2nd_stage
    rho_2nd_stage_flow.append(volumetric_flow)
    Density_2nd_stage.append(rho_2nd_stage)
    
df["ERC_Density_2nd_stage"] = Density_2nd_stage   
df["ERC VOL FLOW 2ND STAGE"] = rho_2nd_stage_flow

Density_3rd_stage =[]
rho_3rd_stage_flow = []
for i in range(len(df)):
    T_K = df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15
    P_Pa = df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
    rho_3rd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
    volumetric_flow = (df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["ERC_3rd_stage_drum_overhead_FLOW"].iloc[i])  * 1000 / rho_3rd_stage
    rho_3rd_stage_flow.append(volumetric_flow)
    Density_3rd_stage.append(rho_3rd_stage)

df["ERC_Density_3rd_stage"] = Density_3rd_stage
df["ERC VOL FLOW 3RD STAGE"] = rho_3rd_stage_flow

#Compressor power calculation based on isentropic enthalpy at outlet pressure, keeping entropy constant (i.e., isentropic compression).
eta = 0.80  # Assume Compressor efficiency
ERC_1st_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
    s1 = PropsSI('S', 'T',df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Ethylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Ethylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
    ERC_1st_stage_comp_estimated_power.append(power)
    
df["ERC_1st_stage_comp_estimated_power_MW"] = ERC_1st_stage_comp_estimated_power

# 2nd stage compressor power
ERC_2nd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
    s1 = PropsSI('S', 'T',df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = (((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000)/3600) * (h2 - h1)/1e6  # MW
    ERC_2nd_stage_comp_estimated_power.append(power)
    
df["ERC_2nd_stage_comp_estimated_power_MW"] = ERC_2nd_stage_comp_estimated_power

# 3rd stage compressor power
ERC_3rd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df)):
    h1 = PropsSI('H', 'T',df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
    s1 = PropsSI('S', 'T',df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df["ERC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Ethylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df["ERC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Ethylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["ERC_3rd_stage_drum_overhead_FLOW"].iloc[i]* 1000)/3600) * (h2 - h1)/1e6  # MW
    ERC_3rd_stage_comp_estimated_power.append(power)
    
df["ERC_3rd_stage_comp_estimated_power_MW"] = ERC_3rd_stage_comp_estimated_power

df["ERC_Total_estimated_power_MW"] = df["ERC_1st_stage_comp_estimated_power_MW"]+df["ERC_2nd_stage_comp_estimated_power_MW"]+df["ERC_3rd_stage_comp_estimated_power_MW"]

df['diff_ERC_power_calculated-simulated'] = df['ERC_power']/1000 - df["ERC_Total_estimated_power_MW"]


#%% ERC ERC_2nd_stage_drum_Overhead_Flow prediction
# u_cols = ['DMCTF_feed', 'CGC_5TH_STG_DISCH_PRES', 'CGC_STAGE_1_SUCTION_PRESSURE','ERC_turbine_Speed','ERC_turbine_steam_flow']

# y_col = 'ERC_2nd_stage_drum_Overhead_Flow'

# config_df_model_details = pd.read_excel(file_path +"/" + "Config_file.xlsx", sheet_name= 'Model details')

y_col = "ERC_2nd_stage_drum_Overhead_Flow"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()


# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for ERC_2nd_stage_drum_Overhead_Flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='ERC_2nd_stage_drum_Overhead_Flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='ERC_2nd_stage_drum_Overhead_Flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_2nd_stage_drum_Overhead_Flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)
    
joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_ERC_2nd_stage_drum_Overhead_Flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_ERC_2nd_stage_drum_Overhead_Flow.pkl')


#%% KT1701 (ERC) steam flow prediction
# u_cols = ['DMCTF_feed','CGC_5TH_STG_DISCH_PRES','ERC_1ST_STAGE_Suction_FLOW','ERC_2nd_stage_drum_Overhead_Flow','ERC_turbine_Speed']
# y_col = 'ERC_turbine_steam_flow'

y_col = "ERC_turbine_steam_flow"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for ERC turbine steam flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='ERC turbine steam flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='ERC turbine steam flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_turbine_steam_flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_ERC_turbine_steam_flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_ERC_turbine_steam_flow.pkl')


#%% ERC power prediction through Steam flow

# u_cols = ['DMCTF_feed','ERC_turbine_steam_flow']
# y_col = 'ERC power'

y_col = "ERC_power"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for ERC power')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='ERC power Prediction',
    xaxis_title='Time Step',
    yaxis_title='ERC power',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_power.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_ERC_power.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_ERC_power.pkl')


#%% ERC first stage suction flow predixtion

# u_cols = ['DMCTF_feed', 'CGC_5TH_STG_DISCH_PRES', 'CGC_STAGE_1_SUCTION_PRESSURE','ERC_turbine_Speed','ERC_turbine_steam_flow']

# y_col = 'ERC_1ST_STAGE_Suction_FLOW'

y_col = "ERC_1ST_STAGE_Suction_FLOW"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

#'ERC_1ST_STAGE_Suction_FLOW','Avg_MW','ERC_2nd_stage_drum_Overhead_Flow'
# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for ERC first stage suction flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='ERC first stage suction flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='ERC first stage suction flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_first_stage_suction_flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_ERC_first_stage_suction_flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_ERC_first_stage_suction_flow.pkl')


#%% ERC first stage suction pressure prediction

# u_cols = ['DMCTF_feed', 'CGC_5TH_STG_DISCH_PRES', 'CGC_STAGE_1_SUCTION_PRESSURE','ERC_turbine_Speed',
#           'ERC_turbine_steam_flow','Cooling water supply temp']

# y_col = 'ERC_1ST_STAGE_Suction_PRESSURE'

y_col = "ERC_1ST_STAGE_Suction_PRESSURE"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for ERC first stage suction pressure prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='ERC first stage suction pressure Prediction',
    xaxis_title='Time Step',
    yaxis_title='ERC first stage suction pressure',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_first_stage_suction_pressure.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_ERC_first_stage_suction_pressure.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_ERC_first_stage_suction_pressure.pkl')

#%% -------------- Ethylene loss to fuel prediction -----------------------

# u_cols = ['DMCTF_feed','COLD TRAIN SEPARATOR FEED TEMP (°C)','ETHYL REFRG ACMLTR D1704',
#           'CGC_5TH_STG_DISCH_PRES','PRC_turbine CHEST PRESSURE',
#           'Demeth OH to condenser','DeltaP Demeth coldbox',
#           'Demeth tower pressure control OP', 'Demeth Reflux drum level','E1413 ER LVL control OP',
#           'Demeth feed separator OH valve opening', 'Methane exp 1402 inlet pressure opening' 
#           ]

#'Ethylene in Ethy fract bottom'
# u_cols = ['DMCTF_feed','Demeth feed separator OH valve opening','Demeth OH to condenser',
#           'Demeth tower pressure control OP','COLD TRAIN SEPARATOR FEED TEMP (°C)',
#           'PRC_1ST_STAGE_Suction_PRESSURE','Demeth No2 feed flow','Demeth No3 feed flow','DeltaP Demeth coldbox',
#          'Off gas exp 1401 inlet pressure']

# u_cols = ['Demeth OH to condenser','Demeth No2 feed flow','Demeth tower pressure control OP',
#           'PRC_1ST_STAGE_Suction_PRESSURE','DMCTF_feed','Demeth tower reflux temp',
#           'Demeth feed separator OH valve opening', 'Fresh ethane feed'
#           ]

# #'Demeth reboiler PR flow ratio','Demeth No3 feed flow','Demeth tower pressure reflux control'
# y_col = 'TOTAL ETHYLENE LOSS'

y_col = "TOTAL_ETHYLENE_LOSS_to_fuel"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

# df1 = df[(df['DMCTF_feed'] > 170000) & (df['DMCTF_feed'] < 180000)]
X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for Ethylene loss to fuel prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='PRC PRC_2nd_stage_drum_Overhead_Flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='PRC PRC_2nd_stage_drum_Overhead_Flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Ethylene_loss_to_fuel.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_Ethylene_loss_to_fuel.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_Ethylene_loss_to_fuel.pkl')


#%% Ethylene prodcut flow prediction
# u_cols = ['DMCTF_feed','CGC_5TH_STG_DISCH_PRES','CGC_STAGE_1_SUCTION_PRESSURE',
#           'PRC_1ST_STAGE_Suction_PRESSURE',
#           'ERC_2nd_stage_drum_Overhead_Flow','Fresh ethane feed']
# #
# y_col = 'Ethylene product flow'

y_col = "Ethylene_product_flow"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# After scaling
X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# Now concat with proper column names
train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

X_scaled_train = X_scaled_train.to_numpy()
X_scaled_test = X_scaled_test.to_numpy()
y_scaled_train = y_scaled_train.to_numpy()
y_scaled_test = y_scaled_test.to_numpy()

nfoursid = NFourSID(
    train_data,
    output_columns=[y_col],
    input_columns=u_cols,  # Now only one input
    num_block_rows=10
)
nfoursid.subspace_identification()
state_space, _ = nfoursid.system_identification(rank=2)

"""
The D matrix in the state-space model shows how inputs directly affect the output. 
A larger absolute value indicates higher direct impact
"""

# After obtaining state_space
D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
print("D Matrix (Input to Output):")
D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

"""
The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
"""
B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# If rank=2, B_matrix has 2 rows
print("B Matrix (Input to State):")
B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

"""You can compute the total influence of each input as the Frobenius norm across states:"""
input_influence_B = np.linalg.norm(B_matrix, axis=0)
input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
print("Input Importance (via B matrix):")
print(input_importance_B)   

"""Combine B and D for Overall Importance"""
total_importance = pd.Series(
    input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
    index=u_cols
).sort_values(ascending=False)

print("Overall Variable Importance:")
print(total_importance)

total_importance.plot(kind='barh', title='Variable Importance for Ethylene product flow prediction')
plt.xlabel('Composite Importance Score (B + D)')
plt.show()

kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

for i in range(len(X_scaled_test)):
    u_step = X_scaled_test[i].reshape(-1, 1)
    # Pass y=None to indicate missing measurement
    kalman.step(y=None, u=u_step)

results_test = kalman.to_dataframe()
y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
y_true = y_test.values.ravel()   

# Create the Plotly figure
fig = go.Figure()

# Add measured (true) values
fig.add_trace(go.Scatter(
    x=list(range(len(y_true))),
    y=y_true,
    mode='lines',
    name='Measured',
    opacity=0.7,
    line=dict(width=2)
))

# Add predicted values
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(dash='dash', width=2)
))

# Update layout
fig.update_layout(
    title='Ethylene product flow Prediction',
    xaxis_title='Time Step',
    yaxis_title='Ethylene product flow',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified',
    template='plotly_white'
)

# Display the interactive plot
pio.renderers.default = "browser"
fig.show()

print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Save the trained Kalman filter model
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Ethylene_product_flow.pkl', 'wb') as f:
    pickle.dump(kalman, f)

joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_Ethylene_product_flow.pkl')
joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_Ethylene_product_flow.pkl')


#%% Overall COT predicton through data model
# u_cols = ['DMCTF_feed','CGC_STAGE_1_SUCTION_PRESSURE'
#           ,'Furnace_conversion']
# #'Fresh ethane feed','Total feed C2H6 mass content'
# y_col = 'Overall_COT'

# y_col = "Overall_COT"
# u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
# u_cols = u_cols.dropna(axis =1)
# u_cols = u_cols.iloc[:, 1:].values
# u_cols = u_cols.ravel().tolist()

# X_train, X_test, y_train, y_test = train_test_split(df[u_cols], df[y_col], test_size=0.2,random_state=42)

# # Scale data
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_scaled_train = scaler_X.fit_transform(X_train)
# y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# X_scaled_test = scaler_X.transform(X_test)  # Use same scaler
# y_scaled_test = scaler_y.transform(y_test.values.reshape(-1, 1))   

# # After scaling
# X_scaled_train = pd.DataFrame(X_scaled_train, columns=u_cols, index=X_train.index)
# X_scaled_test = pd.DataFrame(X_scaled_test, columns=u_cols, index=X_test.index)
# y_scaled_train = pd.DataFrame(y_scaled_train, columns=[y_col], index=y_train.index)
# y_scaled_test = pd.DataFrame(y_scaled_test, columns=[y_col], index=y_test.index)

# # Now concat with proper column names
# train_data = pd.concat([X_scaled_train, y_scaled_train], axis=1)
# test_data = pd.concat([X_scaled_test, y_scaled_test], axis=1)

# X_scaled_train = X_scaled_train.to_numpy()
# X_scaled_test = X_scaled_test.to_numpy()
# y_scaled_train = y_scaled_train.to_numpy()
# y_scaled_test = y_scaled_test.to_numpy()

# nfoursid = NFourSID(
#     train_data,
#     output_columns=[y_col],
#     input_columns=u_cols,  # Now only one input
#     num_block_rows=10
# )
# nfoursid.subspace_identification()
# state_space, _ = nfoursid.system_identification(rank=2)

# """
# The D matrix in the state-space model shows how inputs directly affect the output. 
# A larger absolute value indicates higher direct impact
# """

# # After obtaining state_space
# D_matrix = state_space.d  # Shape: (n_outputs, n_inputs)
# print("D Matrix (Input to Output):")
# D_matrix= pd.DataFrame(D_matrix, columns=u_cols, index=[y_col]).round(2)

# """
# The B matrix determines how inputs affect the internal states. Combined with C, it affects long-term behavior.
# """
# B_matrix = state_space.b  # Shape: (n_states, n_inputs)
# # If rank=2, B_matrix has 2 rows
# print("B Matrix (Input to State):")
# B_matrix = pd.DataFrame(B_matrix, columns=u_cols, index=[f'State_{i}' for i in range(B_matrix.shape[0])]).round(4)

# """You can compute the total influence of each input as the Frobenius norm across states:"""
# input_influence_B = np.linalg.norm(B_matrix, axis=0)
# input_importance_B = pd.Series(input_influence_B, index=u_cols).sort_values(ascending=False)
# print("Input Importance (via B matrix):")
# print(input_importance_B)   

# """Combine B and D for Overall Importance"""
# total_importance = pd.Series(
#     input_influence_B + np.abs(D_matrix.to_numpy()).flatten(),
#     index=u_cols
# ).sort_values(ascending=False)

# print("Overall Variable Importance:")
# print(total_importance)

# total_importance.plot(kind='barh', title='Variable Importance for Overall COT prediction')
# plt.xlabel('Composite Importance Score (B + D)')
# plt.show()

# kalman = Kalman(state_space=state_space, noise_covariance=np.eye(3))

# for i in range(len(X_scaled_test)):
#     u_step = X_scaled_test[i].reshape(-1, 1)
#     # Pass y=None to indicate missing measurement
#     kalman.step(y=None, u=u_step)

# results_test = kalman.to_dataframe()
# y_pred_scaled = results_test[('$y_0$', 'filtered', 'output')].values
# # y_pred_scaled = results_test[('$y_0$', 'next predicted (input corrected)', 'output')].values
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()   
# y_true = y_test.values.ravel()   

# # Create the Plotly figure
# fig = go.Figure()

# # Add measured (true) values
# fig.add_trace(go.Scatter(
#     x=list(range(len(y_true))),
#     y=y_true,
#     mode='lines',
#     name='Measured',
#     opacity=0.7,
#     line=dict(width=2)
# ))

# # Add predicted values
# fig.add_trace(go.Scatter(
#     x=list(range(len(y_pred))),
#     y=y_pred,
#     mode='lines',
#     name='Predicted',
#     line=dict(dash='dash', width=2)
# ))

# # Update layout
# fig.update_layout(
#     title='Overall COT Prediction',
#     xaxis_title='Time Step',
#     yaxis_title='Overall COT',
#     legend=dict(x=0.02, y=0.98),
#     hovermode='x unified',
#     template='plotly_white'
# )

# # Display the interactive plot
# pio.renderers.default = "browser"
# fig.show()

# print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}")    
# r2 = r2_score(y_true, y_pred)
# print(f"R²: {r2:.3f}")

# # Save the trained Kalman filter model
# with open("..\\Results\\Model"+"/"+'kalman_filter_model_overall_COT.pkl', 'wb') as f:
#     pickle.dump(kalman, f)

# joblib.dump(scaler_X, "..\\Results\\Model"+"/"+'scaler_X_overall_COT.pkl')
# joblib.dump(scaler_y, "..\\Results\\Model"+"/"+'scaler_y_overall_COT.pkl')

df["Total_Power_(KW)"]= df["CGC_Power_KW"] + df["PRC_Total_estimated_power_MW"]*1000 + df["ERC_power"]
df["Total_required_steam_flow_(TPH)"] = df["CGC_Turbine_HP_Steam_flow"] + df["PRC_turbine_Calculated_Steam_flow_TPH"] + df["ERC_turbine_steam_flow"]

df.to_excel('..\\Results' + '/'+'Yanpet_OLF1_MultiX_multiY_results.xlsx')

df_corr = df.corr (method = "pearson").round(2)

def color_columns(val):
    if val > 0.4:
        color = 'background-color: green'
    elif val < -0.4:
        color = 'background-color: red'
    else:
        color = ''
    return color

with pd.ExcelWriter("..\\Results" +'/'+'correlation_with_color_conditioning.xlsx', engine='openpyxl') as writer:
    # Write the DataFrame to the Excel file with styling
    df_corr.style.applymap(color_columns).to_excel(writer, index=True)

#%% LBT creation

# ============================== 
# CONFIGURATION SECTION
# ==============================

# EXCEL_FILE = "LBT_Input.xlsx"            # 📥 Input Excel file with test, clean, and config data
# OUTPUT_FILE = "LBT_Result.xlsx"      # 📤 Output Excel file with results
# MAX_BENCHMARKS = 10                           # 🧮 Max benchmark rows to pick per match

# SHEET_T = "test_data"                         # 🧪 Sheet with actual test data
# SHEET_C = "clean_data"                        # ✅ Sheet with clean benchmark/reference data
# SHEET_L = "lbt_input"                         # ⚙️ Sheet with matching rules and tolerances

# ==============================
# FUNCTION DEFINITIONS
# ==============================

T = df.copy()
C = df.copy()
L = pd.read_excel(file_path + "/" + 'DMC_Screen_tags_data.xlsx',sheet_name= "lbt_input" )
# 🕒 Convert Timestamp to datetime
T = T.reset_index(drop=False)
C = C.reset_index(drop=False)

T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")

# 📏 Dictionary: Match tag → decimal precision
decimal_dict = L.set_index("Match_tags")["Match_Tag_Decimal"].dropna().to_dict()
MAX_BENCHMARKS = 10   

# def load_data(excel_file):
#     # 🔄 Load all input sheets
#     T = pd.read_excel(excel_file, sheet_name=SHEET_T)
#     C = pd.read_excel(excel_file, sheet_name=SHEET_C)
#     L = pd.read_excel(excel_file, sheet_name=SHEET_L)

#     # 🕒 Convert Timestamp to datetime
#     T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
#     C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")

#     # 📏 Dictionary: Match tag → decimal precision
#     decimal_dict = L.set_index("Match_tags")["Match_Tag_Decimal"].dropna().to_dict()
#     return T, C, L, decimal_dict

def build_result(row_t, matched_row, T, C, L, match_tags, iteration_unused, decimal_dict):
    result = {}

    # ✅ Add actual and benchmark values for T columns
    for col in T.columns:
        val_actual = row_t[col]
        val_benchmark = matched_row.get(col, np.nan)

        if col in decimal_dict:
            decimals = int(decimal_dict[col])
            val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
            val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark

        result[f"{col}_Actual"] = val_actual
        result[f"{col}_benchmark"] = val_benchmark

    # ➕ Add extra benchmark-only columns from C
    for col in C.columns:
        if col not in T.columns:
            val_benchmark = matched_row.get(col, np.nan)
            if col in decimal_dict:
                decimals = int(decimal_dict[col])
                val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark
            result[f"{col}_benchmark"] = val_benchmark

    matched_iterations = []

    # 🔁 For each matching tag, calculate delta and find which iteration matched
    for tag in match_tags:
        delta_col = f"{tag}_delta"
        iter_col = f"matched_iteration_{tag}"

        actual_val = row_t.get(tag, np.nan)
        benchmark_val = matched_row.get(tag, np.nan)

        if pd.notnull(actual_val) and pd.notnull(benchmark_val):
            decimals = int(decimal_dict.get(tag, 4))
            actual_val = round(actual_val, decimals)
            benchmark_val = round(benchmark_val, decimals)
            delta = round(benchmark_val - actual_val, decimals)
            result[delta_col] = delta

            found_iter = None
            for _, tol_row in L[L["Match_tags"] == tag].iterrows():
                it_no = tol_row["Iteration No"]
                min_tol = tol_row["Tolerance_minimum"]
                max_tol = tol_row["Tolerance_maximum"]
                if -min_tol <= delta <= max_tol:
                    found_iter = it_no
                    break

            result[iter_col] = found_iter
            if found_iter is not None:
                matched_iterations.append(found_iter)
        else:
            result[delta_col] = np.nan
            result[iter_col] = None

    result["iteration_matched"] = max(matched_iterations) if matched_iterations else None
    return result

def build_no_match_result(row_t, T, C, match_tags, decimal_dict):
    result = {}

    # 🚫 If no match is found, store actuals and fill rest with "BENCHMARK NOT FOUND".
    for col in T.columns:
        val_actual = row_t[col]
        if col in decimal_dict:
            decimals = int(decimal_dict[col])
            val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
        result[f"{col}_Actual"] = val_actual
        result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

    for col in C.columns:
        if col not in T.columns:
            result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

    for tag in match_tags:
        result[f"{tag}_delta"] = "BENCHMARK NOT FOUND"
        result[f"matched_iteration_{tag}"] = "BENCHMARK NOT FOUND"

    result["iteration_matched"] = "BENCHMARK NOT FOUND"
    return result

def match_records(T, C, L, max_benchmarks, decimal_dict):
    results = []
    match_tags = L["Match_tags"].dropna().unique().tolist()
    max_iterations = int(pd.to_numeric(L["Iteration No"], errors="coerce").dropna().max())

    for _, row_t in T.iterrows():
        match_found = False

        for iteration in range(1, max_iterations + 1):
            current_tol = L[L["Iteration No"] == iteration]
            filtered_C = C.copy()

            for _, tol_row in current_tol.iterrows():
                col = tol_row["Match_tags"]
                min_tol = tol_row["Tolerance_minimum"]
                max_tol = tol_row["Tolerance_maximum"]

                if col not in T.columns or col not in C.columns or col == "Timestamp":
                    continue

                val_t = row_t[col]
                if pd.isnull(val_t):
                    continue

                if col in decimal_dict:
                    decimals = int(decimal_dict[col])
                    val_t = round(val_t, decimals)
                    filtered_C[col] = filtered_C[col].round(decimals)

                filtered_C = filtered_C[
                    (filtered_C[col] >= val_t - min_tol) & (filtered_C[col] <= val_t + max_tol)
                ]

            if not filtered_C.empty:
                match_found = True
                for _, matched_row in filtered_C.head(max_benchmarks).iterrows():
                    results.append(build_result(row_t, matched_row, T, C, L, match_tags, iteration, decimal_dict))
                break

        if not match_found:
            results.append(build_no_match_result(row_t, T, C, match_tags, decimal_dict))

    return results

def filter_optimal_rows(df, L):
    perf_config = L[["performance_tag", "direction"]].dropna().drop_duplicates()

    if perf_config.empty:
        print("⚠️ No performance tag found in config - skipping filtering.")
        return df

    perf_tag = perf_config.iloc[0]["performance_tag"]
    direction = perf_config.iloc[0]["direction"].strip().lower()
    perf_col = f"{perf_tag}_benchmark"

    if perf_col not in df.columns:
        print(f"⚠️ Performance column '{perf_col}' not found - skipping filtering.")
        return df

    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")
    matched_df = df.dropna(subset=[perf_col])
    unmatched_df = df[df[perf_col].isna()]

    if direction == "minimize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmin()]
    elif direction == "maximize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmax()]
    else:
        raise ValueError(f"Unknown direction '{direction}' in performance config.")

    unmatched_latest = unmatched_df.sort_values("Timestamp_Actual").drop_duplicates(
        subset=["Timestamp_Actual"], keep="last"
    )

    return pd.concat(
        [optimal_df, unmatched_latest[~unmatched_latest["Timestamp_Actual"].isin(optimal_df["Timestamp_Actual"])]],
        ignore_index=True,
    ).sort_values("Timestamp_Actual").reset_index(drop=True)

def process_results(results, T, C, L):
    df = pd.DataFrame(results)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    for col in [c for c in df.columns if c.endswith("Timestamp_benchmark")]:
        df[col] = df[col].fillna("BENCHMARK NOT FOUND")  # Consistent fill

    actual_bench_cols = []
    for col in T.columns:
        if f"{col}_Actual" in df.columns: actual_bench_cols.append(f"{col}_Actual")
        if f"{col}_benchmark" in df.columns: actual_bench_cols.append(f"{col}_benchmark")

    for col in C.columns:
        if col not in T.columns and f"{col}_benchmark" in df.columns:
            actual_bench_cols.append(f"{col}_benchmark")

    delta_cols = sorted([c for c in df.columns if c.endswith("_delta")])
    iter_cols = sorted([c for c in df.columns if c.startswith("matched_iteration_")])

    final_cols = actual_bench_cols + delta_cols + iter_cols
    if "iteration_matched" in df.columns:
        final_cols.append("iteration_matched")

    remaining = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + remaining]

    return filter_optimal_rows(df, L)

# def save_results(df, output_file):
#     df.to_excel(output_file, index=False)
#     print(f"✅ Results saved to: {output_file}")

# ==============================
# MAIN EXECUTION BLOCK
# ==============================

# 📦 Load input data
# T, C, L, decimal_dict = load_data(EXCEL_FILE)

# 🔍 Perform record matching
results = match_records(T, C, L, MAX_BENCHMARKS, decimal_dict)

# 🧹 Clean and organize results
final_df = process_results(results, T, C, L)

# 💾 Save to Excel
# save_results(final_df, OUTPUT_FILE)

final_df.to_excel("..\\Results"+"/"+"YP_OlF1_LBT_result.xlsx")


#%% what if preparation for YANPET OLF1
config_df_model_details_model_details = pd.read_excel(file_path +"/" + "Config_file.xlsx", sheet_name= 'Model details')
user_input_df = pd.read_excel(file_path + "/" + 'Config_file.xlsx',sheet_name= "user inputs")
constraints_df = pd.read_excel(file_path + "/" + 'Config_file.xlsx',sheet_name= "Constraints")
user_time = '2025-02-11 07:00:00'
row = df.loc[user_time]

# user_input = input("Enter timestamp (YYYY-MM-DD HH:MM:SS):")
user_input = user_time
selected_row = df.loc[user_input]
selected_row = pd.DataFrame(selected_row).T  # Convert Series to DataFrame
selected_row.index.name = "Timestamp"

selected_row_updated = selected_row.copy()

DMCTF_feed_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == 'DMCTF_feed', 'Value'].iloc[0]   
# Convert to float, handle NaN
try:
    user_value = float(DMCTF_feed_input) if pd.notna(DMCTF_feed_input) else np.nan
except (ValueError, TypeError):
    user_value = np.nan

DMCTF_value = selected_row_updated['DMCTF_feed'].iloc[0] if isinstance(selected_row_updated['DMCTF_feed'], pd.Series) else selected_row_updated['DMCTF_feed']
selected_row_updated['DMCTF_feed'] = user_value if not np.isnan(user_value) else DMCTF_value

# Quench section prediction updation
with open("..\\Results\\Model"+"/"+'kalman_filter_model_Quench_OH_temp_pred.pkl', 'rb') as f:
    Quench_OH_temp_kalman = pickle.load(f)

# Load scalers later for predictions
scaler_X_Quench_tower_OH_temp = joblib.load("..\\Results\\Model"+"/"+'scaler_X_Quench_tower_OH_temp.pkl')
scaler_y_Quench_tower_OH_temp = joblib.load("..\\Results\\Model"+"/"+'scaler_y_Quench_tower_OH_temp.pkl')

y_col = "Quench_tower_overhead_temp"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

Quench_Tower_df = selected_row_updated[u_cols]

Quench_Tower_scaled_test = scaler_X_Quench_tower_OH_temp.transform(Quench_Tower_df) 
Quench_OH_temp_kalman.step(y=None, u=Quench_Tower_scaled_test.reshape(-1,1))

Quench_OH_temp_results = Quench_OH_temp_kalman.to_dataframe()
Quench_OH_temp_pred_scaled = Quench_OH_temp_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
Quench_OH_temp_pred = scaler_y_Quench_tower_OH_temp.inverse_transform([[Quench_OH_temp_pred_scaled]]).ravel()

selected_row_updated["Quench_tower_overhead_temp"] = Quench_OH_temp_pred[0]

#Quench_tower_overhead_temperature user input
Quench_OH_temp_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == "Quench_tower_overhead_temp", 'Value'].iloc[0]   
# Convert to float, handle NaN
try:
    user_value = float(Quench_OH_temp_input) if pd.notna(Quench_OH_temp_input) else np.nan
except (ValueError, TypeError):
    user_value = np.nan

Quench_OH_temp_value = selected_row_updated["Quench_tower_overhead_temp"].iloc[0] if isinstance(selected_row_updated["Quench_tower_overhead_temp"], pd.Series) else selected_row_updated["Quench_tower_overhead_temp"]
selected_row_updated["Quench_tower_overhead_temp"] = user_value if not np.isnan(user_value) else Quench_OH_temp_value

# CGC suction pressure prediction updation
mask = (selected_row_updated['CGC_Turbine_HP_Steam_flow'] < constraints_df[constraints_df["Parameter"]=="CGC_Turbine_HP_Steam_flow"]["user input value"].values[0]) & (selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)'] < constraints_df[constraints_df["Parameter"]=="CGC_TURBINE_1_SPEED_(RPM)"]["user input value"].values[0])
selected_row_updated.loc[mask, 'CGC_TURBINE_1_SPEED_(RPM)'] = constraints_df[constraints_df["Parameter"]=="CGC_TURBINE_1_SPEED_(RPM)"]["Max vlaue"].values[0] 

# selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)'] = 4405    

#KT1301 Turbine RPM user based input
CGC_Turbine_RPM_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == 'CGC_TURBINE_1_SPEED_(RPM)', 'Value'].iloc[0]   
# Convert to float, handle NaN
try:
    user_value = float(CGC_Turbine_RPM_input) if pd.notna(CGC_Turbine_RPM_input) else np.nan
except (ValueError, TypeError):
    user_value = np.nan

CGC_Turbine_RPM_value = selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)'].iloc[0] if isinstance(selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)'], pd.Series) else selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)']
selected_row_updated['CGC_TURBINE_1_SPEED_(RPM)'] = user_value if not np.isnan(user_value) else CGC_Turbine_RPM_value

with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_Suction_press_pred.pkl', 'rb') as f:
    CGC_Suction_pressure_kalman = pickle.load(f)

scaler_X_CGC_Suction_press = joblib.load("..\\Results\\Model"+"/"+'scaler_X_CGC_Suction_press_pred.pkl')
scaler_y_CGC_Suction_press = joblib.load("..\\Results\\Model"+"/"+'scaler_y_CGC_Suction_press_pred.pkl')

y_col = "CGC_STAGE_1_SUCTION_PRESSURE"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()

CGC_Suction_press_df = selected_row_updated[u_cols]

CGC_Suction_press_scaled_test = scaler_X_CGC_Suction_press.transform(CGC_Suction_press_df)

CGC_Suction_pressure_kalman.step(y=None, u=CGC_Suction_press_scaled_test.reshape(-1,1))

CGC_Suction_pressure_results = CGC_Suction_pressure_kalman.to_dataframe()
CGC_Suction_pressure_pred_scaled = CGC_Suction_pressure_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
CGC_Suction_pressure_pred = scaler_y_CGC_Suction_press.inverse_transform([[CGC_Suction_pressure_pred_scaled]]).ravel()

selected_row_updated['CGC_STAGE_1_SUCTION_PRESSURE'] = CGC_Suction_pressure_pred[0]

# CGC Suction pressure user based value 
CGC_Suction_pressure_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == "CGC_STAGE_1_SUCTION_PRESSURE", 'Value'].iloc[0]   
# Convert to float, handle NaN
try:
    user_value = float(CGC_Suction_pressure_input) if pd.notna(CGC_Suction_pressure_input) else np.nan
except (ValueError, TypeError):
    user_value = np.nan

CGC_Suction_pressure_value = selected_row_updated["CGC_STAGE_1_SUCTION_PRESSURE"].iloc[0] if isinstance(selected_row_updated["CGC_STAGE_1_SUCTION_PRESSURE"], pd.Series) else selected_row_updated["CGC_STAGE_1_SUCTION_PRESSURE"]
selected_row_updated["CGC_STAGE_1_SUCTION_PRESSURE"] = user_value if not np.isnan(user_value) else CGC_Suction_pressure_value


#Overall COT analysis
def COT_calculation(df_furnace):
    df_furnace['Plant_average_feed_rate_Coil'] = df_furnace["DMCTF_feed"]/(df_furnace["Number_Of_Furnaces_Online"]*4)
    
    df_furnace['Coil_CIP_Calculated'] = (-131.3081 +
            (0.0755 * df_furnace['Plant_average_feed_rate_Coil']) +
            (0.1463 * df_furnace['Ethane_Feed_Preheater_Ethane_Feed_Outlet_Pressure']) +
            (0.6819 * df_furnace['Furnace_Ethane_Feed_Preheater_Ethane_Feed_Outlet_Temperature']) +
            (0.4853 * df_furnace['Coil_Weighted_Avg_Feed_CV_opening']) +
            (0.8766 * df_furnace['Coil_Weighted_Avg_Steam_CV_opening']))
    
    df_furnace['Coil_Steam_Flow'] = (df_furnace['Coil_Avg_SHC_Ratio']*df_furnace['Plant_average_feed_rate_Coil'])
           
    df_furnace['Coil_Mixed_Feed_flow'] = df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil']
    
    df_furnace['Coil_Mixed_Feed_Cp'] = (
        (df_furnace['Coil_Steam_Flow'] * 2.067) + 
        (df_furnace['Plant_average_feed_rate_Coil'] * 1.909)
    ) / (df_furnace['Coil_Steam_Flow'] + df_furnace['Plant_average_feed_rate_Coil'])   
    
    df_furnace['Coil_Mixed_Feed_Mol_wt'] = (
        df_furnace['Coil_Mixed_Feed_flow'] / 
        (
            (df_furnace['Plant_average_feed_rate_Coil'] / df_furnace["Furnace_Feed_Average_Molecular_Wt"]) + 
            (df_furnace['Coil_Steam_Flow'] / 18.0)
        )
    )
    
    df_furnace['Coil_Volumetric_Flow'] = (
        df_furnace['Coil_Mixed_Feed_flow']
    ) / (
        ((df_furnace['Coil_CIP_Calculated'] + 101.325) * 
          0.00982963 * df_furnace['Coil_Mixed_Feed_Mol_wt'] ) / 
        (0.08206 * (df_furnace['Coil_Weighted_Avg_Coil_Mixed_Feed_Inlet_Temperature'] + 273.15))
    )
    
    df_furnace['Coil_CIP_Corrected_atma'] = np.where(
        (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) < 5,
        (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
        (df_furnace['Coil_Volumetric_Flow'] * 144 / 1309.83) * 0.00986923,
        (df_furnace['Coil_CIP_Calculated'] / 101.325 + 1) - 
        (df_furnace['Coil_Volumetric_Flow'] * 131 / 1209.52) * 0.00986923
    )
    return df_furnace

selected_row = COT_calculation(selected_row)

Delta_CGC_Suction_Pressure_old = 0

selected_row['Corrected_COP_Furnace'] = selected_row['Coil_Weighted_Avg_COP'] + Delta_CGC_Suction_Pressure_old

selected_row['Furnace_Effluent_C2H6'] = (selected_row['DMCTF_feed']/1000)*(selected_row['Furnace_Normalised_Feed_C2H6_Wt']/100) * (1 - selected_row['Furnace_conversion'])

selected_row['Furnace_Effluent_C2H6_wt%'] =(selected_row['Furnace_Effluent_C2H6']/(selected_row['DMCTF_feed']/1000))*100

selected_row['Coil_Avg_COT_actual'] = ((
    0.937913371 * (selected_row['Coil_CIP_Corrected_atma'] - 0.3) -
    2.413045433 * (selected_row['Corrected_COP_Furnace'] / 101.325 + 1 - 0.2) +
    2.774285758 * selected_row['Coil_Avg_SHC_Ratio'] +
    0.002253435 * selected_row['Plant_average_feed_rate_Coil'] -
    0.463411867 * selected_row['Furnace_Normalised_Feed_C3H8_Wt'] +
    0.674941411 * selected_row['Furnace_Normalised_Feed_C2H6_Wt'] +
    337.8851416 - selected_row['Furnace_Effluent_C2H6_wt%']
    ))/0.451606991


selected_row_updated = COT_calculation(selected_row_updated)

Delta_CGC_Suction_Pressure = selected_row_updated["CGC_STAGE_1_SUCTION_PRESSURE"] - selected_row["CGC_STAGE_1_SUCTION_PRESSURE"]

selected_row_updated['Corrected_COP_Furnace'] = selected_row_updated['Coil_Weighted_Avg_COP'] + Delta_CGC_Suction_Pressure

if ((selected_row_updated['DMCTF_feed'].iloc[0] / 1000) - selected_row_updated['Fresh_ethane_feed'].iloc[0])>70:
    selected_row_updated['Fresh_ethane_feed'] = (selected_row_updated['DMCTF_feed']/1000) - 70

selected_row_updated['Furnace_conversion'] = ((selected_row_updated['DMCTF_feed']/1000)*(selected_row_updated['Furnace_Normalised_Feed_C2H6_Wt']/100) - 
                            (selected_row_updated['DMCTF_feed']/1000-selected_row_updated['Fresh_ethane_feed']))/((selected_row_updated['DMCTF_feed']/1000)*selected_row_updated['Furnace_Normalised_Feed_C2H6_Wt']/100)

selected_row_updated['Furnace_Effluent_C2H6'] = (selected_row_updated['DMCTF_feed']/1000)*(selected_row_updated['Furnace_Normalised_Feed_C2H6_Wt']/100) * (1 - selected_row_updated['Furnace_conversion'])

selected_row_updated['Furnace_Effluent_C2H6_wt%'] =(selected_row_updated['Furnace_Effluent_C2H6']/(selected_row_updated['DMCTF_feed']/1000))*100

selected_row_updated['Coil_Avg_COT'] = ((
    0.937913371 * (selected_row_updated['Coil_CIP_Corrected_atma'] - 0.3) -
    2.413045433 * (selected_row_updated['Corrected_COP_Furnace'] / 101.325 + 1 - 0.2) +
    2.774285758 * selected_row_updated['Coil_Avg_SHC_Ratio'] +
    0.002253435 * selected_row_updated['Plant_average_feed_rate_Coil'] -
    0.463411867 * selected_row_updated['Furnace_Normalised_Feed_C3H8_Wt'] +
    0.674941411 * selected_row_updated['Furnace_Normalised_Feed_C2H6_Wt'] +
    337.8851416 - selected_row_updated['Furnace_Effluent_C2H6_wt%']
    ))/0.451606991

delta_COT = selected_row_updated['Coil_Avg_COT'] -selected_row['Coil_Avg_COT_actual']

selected_row_updated['Coil_Avg_COT'] = selected_row['Coil_Avg_COT'] + delta_COT

# with open("..\\Results\\Model"+"/"+'kalman_filter_model_overall_COT.pkl', 'rb') as f:
#     overall_COT_kalman = pickle.load(f)

# # Load scalers later for predictions
# scaler_X_overall_COT = joblib.load("..\\Results\\Model"+"/"+'scaler_X_overall_COT.pkl')
# scaler_y_overall_COT = joblib.load("..\\Results\\Model"+"/"+'scaler_y_overall_COT.pkl')

# u_cols = ['DMCTF_feed','CGC_STAGE_1_SUCTION_PRESSURE'
#           ,'Furnace_conversion']
# y_col = 'Overall_COT'

# overall_COT_df = selected_row_updated[u_cols]

# overall_COT_scaled_test = scaler_X_overall_COT.transform(overall_COT_df) 
# overall_COT_kalman.step(y=None, u=overall_COT_scaled_test.reshape(-1,1))

# overall_COT_results = overall_COT_kalman.to_dataframe()
# overall_COT_pred_scaled = overall_COT_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
# overall_COT_pred = scaler_y_overall_COT.inverse_transform([[overall_COT_pred_scaled]]).ravel()

# selected_row_updated['Overall_COT'] = overall_COT_pred[0]

# CGC stage 5 pressure prediction updation
with open("..\\Results\\Model"+"/"+'kalman_filter_model_5th_stg_discharge_press.pkl', 'rb') as f:
    Fifth_stg_discharge_press_kalman = pickle.load(f)

scaler_X_5th_stg_discharge_press = joblib.load( "..\\Results\\Model"+"/"+'scaler_X_5th_stg_discharge_press.pkl')
scaler_y_5th_stg_discharge_press = joblib.load("..\\Results\\Model"+"/"+'scaler_y_5th_stg_discharge_press.pkl')

y_col = "CGC_5TH_STG_DISCH_PRES"
u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
u_cols = u_cols.dropna(axis =1)
u_cols = u_cols.iloc[:, 1:].values
u_cols = u_cols.ravel().tolist()


Fifth_stg_discharge_press_df = selected_row_updated[u_cols]

Fifth_stg_discharge_press_scaled_test = scaler_X_5th_stg_discharge_press.transform(Fifth_stg_discharge_press_df) 
Fifth_stg_discharge_press_kalman.step(y=None, u=Fifth_stg_discharge_press_scaled_test.reshape(-1,1))

Fifth_stg_discharge_press_results = Fifth_stg_discharge_press_kalman.to_dataframe()
Fifth_stg_discharge_press_scaled = Fifth_stg_discharge_press_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
Fifth_stg_discharge_press_pred = scaler_y_5th_stg_discharge_press.inverse_transform([[Fifth_stg_discharge_press_scaled]]).ravel()

selected_row_updated['CGC_5TH_STG_DISCH_PRES'] = Fifth_stg_discharge_press_pred[0]

#Fifth stage discharge pressure user based value
Fifth_stg_discharge_press_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == "CGC_5TH_STG_DISCH_PRES", 'Value'].iloc[0]   
# Convert to float, handle NaN
try:
    user_value = float(Fifth_stg_discharge_press_input) if pd.notna(Fifth_stg_discharge_press_input) else np.nan
except (ValueError, TypeError):
    user_value = np.nan

Fifth_stg_discharge_press_value = selected_row_updated["CGC_5TH_STG_DISCH_PRES"].iloc[0] if isinstance(selected_row_updated["CGC_5TH_STG_DISCH_PRES"], pd.Series) else selected_row_updated["CGC_5TH_STG_DISCH_PRES"]
selected_row_updated["CGC_5TH_STG_DISCH_PRES"] = user_value if not np.isnan(user_value) else Fifth_stg_discharge_press_value

# 1. Safe Constraint Check
constraint_hit = False
param_name = "CGC_5TH_STG_DISCH_PRES"

# Check if parameter exists in constraints_df before accessing
mask = constraints_df["Parameter"] == param_name
if mask.any():
    limit = constraints_df.loc[mask, "user input value"].values[0]
    if selected_row_updated[param_name].values[0] > limit:
        selected_row_updated[param_name] = "constraints hits: Reduce the DMCTF"
        constraint_hit = True
        Actual_vs_estimated = pd.concat([selected_row, selected_row_updated], axis=0)
        Actual_vs_estimated.index = ["actual", "estimated"]
        styled = Actual_vs_estimated
        styled.to_excel("..\\Results\\Actual_vs_estimated what if.xlsx", engine='openpyxl')
else:
    print(f"⚠️ Warning: Parameter '{param_name}' not found in constraints_df")

if not constraint_hit:
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_power_pred.pkl', 'rb') as f:
        CGC_power_pred_kalman = pickle.load(f)
    
    scaler_X_CGC_power = joblib.load( "..\\Results\\Model"+"/"+'scaler_X_CGC_power_pred.pkl')
    scaler_y_CGC_power = joblib.load("..\\Results\\Model"+"/"+'scaler_y_CGC_power_pred.pkl')
     
    y_col = "CGC_Power_KW"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    CGC_power_df = selected_row_updated[u_cols]
    
    CGC_power_scaled_test = scaler_X_CGC_power.transform(CGC_power_df) 
    CGC_power_pred_kalman.step(y=None, u=CGC_power_scaled_test.reshape(-1,1))
    
    CGC_power_results = CGC_power_pred_kalman.to_dataframe()
    CGC_power_pred_scaled = CGC_power_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    CGC_power_pred = scaler_y_CGC_power.inverse_transform([[CGC_power_pred_scaled]]).ravel()
    
    selected_row_updated['CGC_Power_KW'] = CGC_power_pred[0]
    
    # CGC Turbine HP steam flow
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_CGC_Turbine_HP_Steam_flow_pred.pkl', 'rb') as f:
        CGC_Turbine_HP_Steam_flow_pred_kalman = pickle.load(f)
    
    scaler_X_CGC_Turbine_HP_Steam_flow = joblib.load( "..\\Results\\Model"+"/"+'scaler_X_CGC_Turbine_HP_Steam_flow_pred.pkl')
    scaler_y_CGC_Turbine_HP_Steam_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_CGC_Turbine_HP_Steam_flow_pred.pkl')
    
    y_col = "CGC_Turbine_HP_Steam_flow"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    CGC_Turbine_HP_Steam_flow_df = selected_row_updated[u_cols]
    
    CGC_Turbine_HP_Steam_flow_scaled_test = scaler_X_CGC_Turbine_HP_Steam_flow.transform(CGC_Turbine_HP_Steam_flow_df)
    CGC_Turbine_HP_Steam_flow_pred_kalman.step(y=None, u = CGC_Turbine_HP_Steam_flow_scaled_test.reshape(-1,1))
    
    CGC_Turbine_HP_Steam_flow_results = CGC_Turbine_HP_Steam_flow_pred_kalman.to_dataframe()
    CGC_Turbine_HP_Steam_flow_pred_scaled = CGC_Turbine_HP_Steam_flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    CGC_Turbine_HP_Steam_flow_pred = scaler_y_CGC_Turbine_HP_Steam_flow.inverse_transform([[CGC_Turbine_HP_Steam_flow_pred_scaled]]).ravel()
    
    selected_row_updated['CGC_Turbine_HP_Steam_flow'] = CGC_Turbine_HP_Steam_flow_pred[0]
    
    
    # PRC 1st stage suction flow prediction
    mask = (selected_row_updated['PRC_turbine_steam_flow'] < constraints_df[constraints_df["Parameter"]=="PRC_turbine_steam_flow"]["user input value"].values[0]) & (selected_row_updated['PRC_turbine_RPM'] < constraints_df[constraints_df["Parameter"]=="PRC_turbine_RPM"]["user input value"].values[0])
    selected_row_updated.loc[mask, 'PRC_turbine_RPM'] = constraints_df[constraints_df["Parameter"]=="PRC_turbine_RPM"]["Max vlaue"].values[0] 
    
    #PRC_turbine Turbine speed user based input
    PRC_turbine_Turbine_RPM_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == 'PRC_turbine_RPM', 'Value'].iloc[0]   
    # Convert to float, handle NaN
    try:
        user_value = float(PRC_turbine_Turbine_RPM_input) if pd.notna(PRC_turbine_Turbine_RPM_input) else np.nan
    except (ValueError, TypeError):
        user_value = np.nan
    
    PRC_turbine_Turbine_RPM_value = selected_row_updated["PRC_turbine_RPM"].iloc[0] if isinstance(selected_row_updated["PRC_turbine_RPM"], pd.Series) else selected_row_updated["PRC_turbine_RPM"]
    selected_row_updated["PRC_turbine_RPM"] = user_value if not np.isnan(user_value) else PRC_turbine_Turbine_RPM_value
    
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_1st_stage_suction_flow.pkl', 'rb') as f:
        PRC_1st_stage_suction_flow_kalman = pickle.load(f)
    
    scaler_X_PRC_1st_stage_suction_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_X_PRC_1st_stage_suction_flow.pkl')
    scaler_y_PRC_1st_stage_suction_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_PRC_1st_stage_suction_flow.pkl')
    
    y_col = "PRC_1ST_STAGE_Suction_FLOW"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    PRC_1st_stage_suction_flow_df = selected_row_updated[u_cols]
    
    PRC_1st_stage_suction_flow_scaled_test = scaler_X_PRC_1st_stage_suction_flow.transform(PRC_1st_stage_suction_flow_df) 
    PRC_1st_stage_suction_flow_kalman.step(y=None, u=PRC_1st_stage_suction_flow_scaled_test.reshape(-1,1))
    
    PRC_1st_stage_suction_flow_results = PRC_1st_stage_suction_flow_kalman.to_dataframe()
    PRC_1st_stage_suction_flow_pred_scaled = PRC_1st_stage_suction_flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    PRC_1st_stage_suction_flow_pred = scaler_y_PRC_1st_stage_suction_flow.inverse_transform([[PRC_1st_stage_suction_flow_pred_scaled]]).ravel()
    
    selected_row_updated['PRC_1ST_STAGE_Suction_FLOW'] = PRC_1st_stage_suction_flow_pred[0]
    
    # PRC 1st stage suction Pressure prediction 
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_1st_stage_suction_pressure.pkl', 'rb') as f:
        PRC_1st_stage_suction_pressure_kalman = pickle.load(f)
    
    scaler_X_PRC_1st_stage_suction_pressure = joblib.load("..\\Results\\Model"+"/"+'scaler_X_PRC_1st_stage_suction_pressure.pkl')
    scaler_y_PRC_1st_stage_suction_pressure = joblib.load("..\\Results\\Model"+"/"+'scaler_y_PRC_1st_stage_suction_pressure.pkl')
    
    y_col = "PRC_1ST_STAGE_Suction_PRESSURE"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    PRC_1st_stage_suction_pressure_df = selected_row_updated[u_cols]
    
    PRC_1st_stage_suction_pressure_scaled_test = scaler_X_PRC_1st_stage_suction_pressure.transform(PRC_1st_stage_suction_pressure_df) 
    PRC_1st_stage_suction_pressure_kalman.step(y=None, u=PRC_1st_stage_suction_pressure_scaled_test.reshape(-1,1))
    
    PRC_1st_stage_suction_pressure_results = PRC_1st_stage_suction_pressure_kalman.to_dataframe()
    PRC_1st_stage_suction_pressure_pred_scaled = PRC_1st_stage_suction_pressure_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    PRC_1st_stage_suction_pressure_pred = scaler_y_PRC_1st_stage_suction_pressure.inverse_transform([[PRC_1st_stage_suction_pressure_pred_scaled]]).ravel()
        
    selected_row_updated['PRC_1ST_STAGE_Suction_PRESSURE'] = PRC_1st_stage_suction_pressure_pred[0]
    
    #PRC_1ST_STAGE_Suction_PRESSURE user based input
    K1601_1st_stage_Suction_pressure_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == "PRC_1ST_STAGE_Suction_PRESSURE", 'Value'].iloc[0]   
    # Convert to float, handle NaN
    try:
        user_value = float(K1601_1st_stage_Suction_pressure_input) if pd.notna(K1601_1st_stage_Suction_pressure_input) else np.nan
    except (ValueError, TypeError):
        user_value = np.nan
    
    K1601_1st_stage_Suction_pressure_value = selected_row_updated["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[0] if isinstance(selected_row_updated["PRC_1ST_STAGE_Suction_PRESSURE"], pd.Series) else selected_row_updated["PRC_1ST_STAGE_Suction_PRESSURE"]
    selected_row_updated["PRC_1ST_STAGE_Suction_PRESSURE"] = user_value if not np.isnan(user_value) else K1601_1st_stage_Suction_pressure_value
    
    
    # PRC PRC_2nd_stage_drum_Overhead_Flow prediction
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_PRC_2nd_stage_drum_Overhead_Flow.pkl', 'rb') as f:
        PRC_2nd_stage_drum_Overhead_Flow_kalman = pickle.load(f)
        
    scaler_X_PRC_2nd_stage_drum_Overhead_Flow = joblib.load("..\\Results\\Model"+"/"+'scaler_X_PRC_2nd_stage_drum_Overhead_Flow.pkl')
    scaler_y_PRC_2nd_stage_drum_Overhead_Flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_PRC_2nd_stage_drum_Overhead_Flow.pkl')
    
    y_col = "PRC_2nd_stage_drum_Overhead_Flow"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    PRC_2nd_stage_drum_Overhead_Flow_df = selected_row_updated[u_cols]
    
    PRC_2nd_stage_drum_Overhead_Flow_scaled_test = scaler_X_PRC_2nd_stage_drum_Overhead_Flow.transform(PRC_2nd_stage_drum_Overhead_Flow_df) 
    PRC_2nd_stage_drum_Overhead_Flow_kalman.step(y=None, u=PRC_2nd_stage_drum_Overhead_Flow_scaled_test.reshape(-1,1))
    
    PRC_2nd_stage_drum_Overhead_Flow_results = PRC_2nd_stage_drum_Overhead_Flow_kalman.to_dataframe()
    PRC_2nd_stage_drum_Overhead_Flow_pred_scaled = PRC_2nd_stage_drum_Overhead_Flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    PRC_2nd_stage_drum_Overhead_Flow_pred = scaler_y_PRC_2nd_stage_drum_Overhead_Flow.inverse_transform([[PRC_2nd_stage_drum_Overhead_Flow_pred_scaled]]).ravel()
    
    selected_row_updated['PRC_2nd_stage_drum_Overhead_Flow'] = PRC_2nd_stage_drum_Overhead_Flow_pred[0]
    
    
    def PRC_section_power(df):
        Density_1st_stage =[]
        rho_1st_stage_flow = []
        for i in range(len(df)):
            T_K = df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15
            P_Pa = df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
            rho_1st_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
            volumetric_flow = df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000 / rho_1st_stage
            rho_1st_stage_flow.append(volumetric_flow)
            Density_1st_stage.append(rho_1st_stage)
               
        df["PRC_Density_1st_stage"] = Density_1st_stage
        df["PRC VOL FLOW 1ST STAGE"] = rho_1st_stage_flow

        Density_2nd_stage =[]
        rho_2nd_stage_flow = []
        for i in range(len(df)):
            T_K = df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15
            P_Pa = df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5
            rho_2nd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
            volumetric_flow = (df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["PRC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000 / rho_2nd_stage
            rho_2nd_stage_flow.append(volumetric_flow)
            Density_2nd_stage.append(rho_2nd_stage)
            
        df["PRC_Density_2nd_stage"] = Density_2nd_stage   
        df["PRC VOL FLOW 2ND STAGE"] = rho_2nd_stage_flow

        Density_3rd_stage =[]
        rho_3rd_stage_flow = []

        for i in range(len(df)):
            T_K = df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15
            P_Pa = df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
            rho_3rd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
            volumetric_flow = df["PRC_3RD_STAGE_Suction_FLOW"].iloc[i]  * 1000 / rho_3rd_stage
            rho_3rd_stage_flow.append(volumetric_flow)
            Density_3rd_stage.append(rho_3rd_stage)

        df["PRC_Density_3rd_stage"] = Density_3rd_stage
        df["PRC VOL FLOW 3RD STAGE"] = rho_3rd_stage_flow

        #Compressor power calculation based on isentropic enthalpy at outlet pressure, keeping entropy constant (i.e., isentropic compression).
        eta = 0.70  # Assume Compressor efficiency
        PRC_1st_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
            s1 = PropsSI('S', 'T',df["PRC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["PRC_1ST_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["PRC_1ST_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = ((df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
            PRC_1st_stage_comp_estimated_power.append(power)
            
        df["PRC_1st_stage_comp_estimated_power_MW"] = PRC_1st_stage_comp_estimated_power

        # 2nd stage compressor power
        PRC_2nd_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
            s1 = PropsSI('S', 'T',df["PRC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["PRC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = (((df["PRC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["PRC_1ST_STAGE_Suction_FLOW"].iloc[i]) * 1000)/3600) * (h2 - h1)/1e6  # MW
            PRC_2nd_stage_comp_estimated_power.append(power)
            
        df["PRC_2nd_stage_comp_estimated_power_MW"] = PRC_2nd_stage_comp_estimated_power

        # 3rd stage compressor power
        PRC_3rd_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
            s1 = PropsSI('S', 'T',df["PRC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["PRC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["PRC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["PRC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = ((df["PRC_3RD_STAGE_Suction_FLOW"].iloc[i]* 1000)/3600) * (h2 - h1)/1e6  # MW
            PRC_3rd_stage_comp_estimated_power.append(power)
            
        df["PRC_3rd_stage_comp_estimated_power_MW"] = PRC_3rd_stage_comp_estimated_power

        df["PRC_Total_estimated_power_MW"] = df["PRC_1st_stage_comp_estimated_power_MW"]+df["PRC_2nd_stage_comp_estimated_power_MW"]+df["PRC_3rd_stage_comp_estimated_power_MW"]

        return df
    
    selected_row_updated = PRC_section_power(selected_row_updated)
    
    def PRC_turbine_extraction_steam_flow_prediction (df):
        PRC_turbine_current_steam_enthalpy_KJ_Kg = []
        PRC_turbine_current_steam_entropy_KJ_KgK =[]
        PRC_turbine_current_outlet_ethalpy_KJ_Kg =[]
        PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg =[]
        PRC_turbine_current_power_gen_extraction_MW = []
        PRC_turbine_current_power_gen_exhaust_MW =[]
        PRC_turbine_current_Turbine_power_MW_based_on_EE = []
        PRC_turbine_current_Turbine_power_MW_based_on_steam_flow = []
        PRC_turbine_current_Specific_steam_consumption_MT_MW =[]
        #turbine_efficiency =[]
        
        for i in range(len(df)):
            Steam_flow_TPH = df['PRC_turbine_steam_flow'].iloc[i]
            Condensate_flow_TPH = df['PRC_turbine_condensate_flow'].iloc[i]
            Extraction_flow_TPH = df['PRC_turbine_Extraction_flow'].iloc[i]
            
            Steam_flow_Kg_hr       = Steam_flow_TPH *1000
            Extraction_flow_Kg_hr  = Extraction_flow_TPH *1000
            Condensate_flow_Kg_hr  = Condensate_flow_TPH *1000
        
            # Steam Inlet conditions
            P_steam = df['PRC_turbine_Steam_pressure'].iloc[i]*1000   # Pa
            T_steam = df['PRC_turbine_Steam_Temp'].iloc[i] +273.15   # K
            T_sat_steam = PropsSI('T', 'P', P_steam, 'Q', 1, 'Water') - 273.15 # degC
        
            # Extraction pressure
            Pe = df['PRC_turbine_Extraction_Pressure'].iloc[i]*1000      # Pa 
            T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K
        
            # Condenser pressure
            Pc = df['PRC_turbine_Condensate_Pressure'].iloc[i]*1000
            Tc_actual = df['PRC_turbine_Condensate_Temperature'].iloc[i] +273.15  # K (actual exhaust steam temp)
            T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
            
            # Steam (inlet) Enthalpy calculation 
            h_steam = PropsSI('H', 'P', P_steam, 'T', T_steam,'Water')/1000   # kJ/kg
            s_steam = PropsSI('S', 'P', P_steam, 'T', T_steam, 'Water')/1000  # KJ/kg·K
            
            # Actual outlet enthalpies
            # Actual outlet enthalpies
            # he = PropsSI('H', 'P', Pe, 'T', Te_actual, 'Water')/1000  # Extracted steam (actual)
            he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
            #hc = PropsSI('H', 'P', Pc, 'T', Tc_actual, 'Water')/1000  # Condenser outlet (actual)
            hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
            dryness_fraction = 0.92
            hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)
        
            hc = hc_liquid + hc_vapor*dryness_fraction
        
            Net_heat_release = Steam_flow_Kg_hr*h_steam -Extraction_flow_Kg_hr*he - Condensate_flow_Kg_hr*hc
            power_gen_extraction = Extraction_flow_Kg_hr*(h_steam-he)/3600/1000
            power_gen_exhaust = Condensate_flow_Kg_hr*(h_steam-hc)/3600/1000
            Turbine_power_MW_EE = power_gen_extraction + power_gen_exhaust
            Specific_steam_consumption = Steam_flow_TPH/(Turbine_power_MW_EE) # MT/MW
        
            def isentropic_enthalpy(P_target, s_in):
                # Keep everything in J/kg·K
                s_f = PropsSI("S", "P", P_target, "Q", 0, "Water")/1000
                s_g = PropsSI("S", "P", P_target, "Q", 1, "Water")/1000
            
                if s_f < s_in < s_g:
                    x = (s_in - s_f) / (s_g - s_f)
                    h_f = PropsSI("H", "P", P_target, "Q", 0, "Water")
                    h_g = PropsSI("H", "P", P_target, "Q", 1, "Water")
                    h_iso = h_f + x * (h_g - h_f)
                else:
                    h_iso = PropsSI("H", "P", P_target, "S", s_in, "Water")
            
                return h_iso / 1000  # Convert to kJ/kg
        
        
            # Isentropic outlet enthalpies
            he_s = isentropic_enthalpy(Pe, s_steam)
            hc_s = isentropic_enthalpy(Pc, s_steam)
            
            # Mass-weighted outlet enthalpies
            h2_actual = (Extraction_flow_Kg_hr * he + Condensate_flow_Kg_hr * hc) / Steam_flow_Kg_hr
            h2s_ideal = (Extraction_flow_Kg_hr  * he_s + Condensate_flow_Kg_hr * hc_s) / Steam_flow_Kg_hr
            
            # Turbine isentropic efficiency
            efficiency = (h_steam - h2_actual) / (h_steam - h2s_ideal)*100
            
            turbine_power_MW_SF = (h_steam - h2_actual) * (Steam_flow_Kg_hr / (3600*1000))  # Convert kg/hr to kg/s
            
            PRC_turbine_current_steam_enthalpy_KJ_Kg.append(h_steam)
            PRC_turbine_current_steam_entropy_KJ_KgK.append(s_steam)
            PRC_turbine_current_outlet_ethalpy_KJ_Kg.append(h2_actual)
            PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg.append(h2s_ideal)
            PRC_turbine_current_power_gen_extraction_MW.append(power_gen_extraction)
            PRC_turbine_current_power_gen_exhaust_MW.append(power_gen_exhaust)
            PRC_turbine_current_Turbine_power_MW_based_on_EE.append(Turbine_power_MW_EE)
            PRC_turbine_current_Turbine_power_MW_based_on_steam_flow.append(turbine_power_MW_SF)
            PRC_turbine_current_Specific_steam_consumption_MT_MW.append(Specific_steam_consumption)
            #turbine_efficiency.append(efficiency)
            
        df["PRC_turbine_current_steam_enthalpy_KJ_Kg"] = PRC_turbine_current_steam_enthalpy_KJ_Kg
        df["PRC_turbine_current_steam_entropy_KJ_KgK"] = PRC_turbine_current_steam_entropy_KJ_KgK
        df["PRC_turbine_current_outlet_ethalpy_KJ_Kg"] = PRC_turbine_current_outlet_ethalpy_KJ_Kg
        df["PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg"] = PRC_turbine_current_outlet_isentropic_ethalpy_KJ_Kg
        df["PRC_turbine_current_power_gen_extraction_MW"] = PRC_turbine_current_power_gen_extraction_MW
        df["PRC_turbine_current_power_gen_exhaust_MW"] = PRC_turbine_current_power_gen_exhaust_MW
        df["PRC_turbine_current_Turbine_power_MW_based_on_EE"] = PRC_turbine_current_Turbine_power_MW_based_on_EE
        df["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"] = PRC_turbine_current_Turbine_power_MW_based_on_steam_flow
        df["PRC_turbine_current_Specific_steam_consumption_MT_MW"] = PRC_turbine_current_Specific_steam_consumption_MT_MW
#df["turbine_efficiency"] = turbine_efficiency
    
        PRC_turbine_optimized_extraction = []
        PRC_turbine_calculated_steam_flow = []
        PRC_turbine_matched_power_EE = []
        PRC_turbine_matched_power_SF = []
        PRC_turbine_matched_h2_actual = []

        def match_actual_power(row):
            try:
                # Steam inlet enthalpy
                P_steam = row["PRC_turbine_Steam_pressure"] * 1000  # Pa
                T_steam = row["PRC_turbine_Steam_Temp"] + 273.15    # K
                h_steam = PropsSI("H", "P", P_steam, "T", T_steam, "Water") / 1000  # kJ/kg
             
                # Actual extracted steam enthalpy (he)
                Pe = row["PRC_turbine_Extraction_Pressure"] * 1000  # Pa
                # Te_actual = row["Extraction Temperature"] + 273.15  # K
                T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K
                he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
                
                # Actual condenser steam enthalpy (hc)
                Pc = row["PRC_turbine_Condensate_Pressure"] * 1000  # Pa
                # Tc_actual = row["Condensate Temperature"] + 273.15  # K
                T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
                hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
                dryness_fraction = 0.92
                hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)
            
                hc = hc_liquid + hc_vapor*dryness_fraction
            
                condensate_flow_TPH = row["PRC_turbine_condensate_flow"]
                condensate_flow_Kg_hr = condensate_flow_TPH * 1000
                actual_power = row["PRC_Total_estimated_power_MW"]
            
                def objective(extraction_flow_TPH):
                    extraction_flow_Kg_hr = extraction_flow_TPH * 1000
                    steam_flow_Kg_hr = extraction_flow_Kg_hr + condensate_flow_Kg_hr
            
                    power = ((extraction_flow_Kg_hr * (h_steam - he)) + 
                             (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000
                    return (power - actual_power) ** 2
            
                result = minimize_scalar(objective, bounds=(10, 350), method='bounded')
            
                if result.success:
                    ef_opt = result.x
                    ef_Kg_hr = ef_opt * 1000
                    sf_opt = ef_opt + condensate_flow_TPH
                    sf_Kg_hr = sf_opt * 1000
                    # Matched power via energy balance
                    power_matched_EE = ((ef_Kg_hr * (h_steam - he)) + (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000
    
                    # Mass-weighted outlet enthalpy
                    h2_actual = (ef_Kg_hr * he + condensate_flow_Kg_hr * hc) / sf_Kg_hr
    
                    # Power via inlet/outlet enthalpy and total flow
                    power_matched_SF = (h_steam - h2_actual) * (sf_Kg_hr / 3600 / 1000)
    
                    return ef_opt, sf_opt, power_matched_EE, power_matched_SF, h2_actual
                else:
                    return (row["PRC_turbine_Extraction_flow"], row["PRC_turbine_steam_flow"],
                            row["PRC_turbine_current_Turbine_power_MW_based_on_EE"], row["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"],
                            None)
            except Exception as e:
                print(f"Error at index {row.name}: {e}")
                return (row["PRC_turbine_Extraction_flow"], row["PRC_turbine_steam_flow"],
                        row["PRC_turbine_current_Turbine_power_MW_based_on_EE"], row["PRC_turbine_current_Turbine_power_MW_based_on_steam_flow"],
                        None)
            
        # Loop and apply
        for _, row in df.iterrows():
            ef, sf, power_EE, power_SF, h2 = match_actual_power(row)
            PRC_turbine_optimized_extraction.append(ef)
            PRC_turbine_calculated_steam_flow.append(sf)
            PRC_turbine_matched_power_EE.append(power_EE)
            PRC_turbine_matched_power_SF.append(power_SF)
            PRC_turbine_matched_h2_actual.append(h2)
    
        # Save to DataFrame
        df["PRC_turbine_Optimized_Extraction_flow_TPH"] = PRC_turbine_optimized_extraction
        df["PRC_turbine_Calculated_Steam_flow_TPH"] = PRC_turbine_calculated_steam_flow
        df["PRC_turbine_Matched_Turbine_power_MW_EE"] = PRC_turbine_matched_power_EE
        df["PRC_turbine_Matched_Turbine_power_MW_SF"] = PRC_turbine_matched_power_SF
        df["PRC_turbine_Matched_h2_actual_KJ_Kg"] = PRC_turbine_matched_h2_actual
        df["Power_Error"] = (
            df["PRC_turbine_Matched_Turbine_power_MW_EE"] -
            df["PRC_Total_estimated_power_MW"]
        )
        df["Power_EE_vs_SF_Diff"] = (
            df["PRC_turbine_Matched_Turbine_power_MW_EE"] -
            df["PRC_turbine_Matched_Turbine_power_MW_SF"]
        )
        df["Devaiation in steam flow (Simulated-actual)"] = df["PRC_turbine_Calculated_Steam_flow_TPH"]- df["PRC_turbine_steam_flow"]
        df["Devaiation in extraction (Simulated-actual)"] = df["PRC_turbine_Optimized_Extraction_flow_TPH"]- df["PRC_turbine_Extraction_flow"]
        df["Specific_steam_consumption_MT_MW_updated"] = df["PRC_turbine_Calculated_Steam_flow_TPH"] / df["PRC_Total_estimated_power_MW"]

        return df
    
    selected_row_updated = PRC_turbine_extraction_steam_flow_prediction (selected_row_updated)
    
    # ERC ERC_2nd_stage_drum_Overhead_Flow prediction
    mask = (selected_row_updated['ERC_turbine_steam_flow'] < constraints_df[constraints_df["Parameter"]=='ERC_turbine_steam_flow']["user input value"].values[0]) & (selected_row_updated['ERC_turbine_Speed'] < constraints_df[constraints_df["Parameter"]=="ERC_turbine_Speed"]["user input value"].values[0])
    selected_row_updated.loc[mask, 'ERC_turbine_Speed'] =constraints_df[constraints_df["Parameter"]=="ERC_turbine_Speed"]["Max vlaue"].values[0] 
    
    #ERC_turbine_Speed user based input
    ERC_Turbine_RPM_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == 'ERC_turbine_Speed', 'Value'].iloc[0]   
    # Convert to float, handle NaN
    try:
        user_value = float(ERC_Turbine_RPM_input) if pd.notna(ERC_Turbine_RPM_input) else np.nan
    except (ValueError, TypeError):
        user_value = np.nan
    
    ERC_Turbine_RPM_value = selected_row_updated["ERC_turbine_Speed"].iloc[0] if isinstance(selected_row_updated["ERC_turbine_Speed"], pd.Series) else selected_row_updated["ERC_turbine_Speed"]
    selected_row_updated["ERC_turbine_Speed"] = user_value if not np.isnan(user_value) else ERC_Turbine_RPM_value
    
    
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_2nd_stage_drum_Overhead_Flow.pkl', 'rb') as f:
        ERC_2nd_stage_drum_Overhead_Flow_kalman = pickle.load(f)
        
    scaler_X_ERC_2nd_stage_drum_Overhead_Flow = joblib.load("..\\Results\\Model"+"/"+'scaler_X_ERC_2nd_stage_drum_Overhead_Flow.pkl')
    scaler_y_ERC_2nd_stage_drum_Overhead_Flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_ERC_2nd_stage_drum_Overhead_Flow.pkl')
    
    y_col = "ERC_2nd_stage_drum_Overhead_Flow"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    ERC_2nd_stage_drum_Overhead_Flow_df = selected_row_updated[u_cols]
    
    ERC_2nd_stage_drum_Overhead_Flow_scaled_test = scaler_X_ERC_2nd_stage_drum_Overhead_Flow.transform(ERC_2nd_stage_drum_Overhead_Flow_df) 
    ERC_2nd_stage_drum_Overhead_Flow_kalman.step(y=None, u=ERC_2nd_stage_drum_Overhead_Flow_scaled_test.reshape(-1,1))
    
    ERC_2nd_stage_drum_Overhead_Flow_results = ERC_2nd_stage_drum_Overhead_Flow_kalman.to_dataframe()
    ERC_2nd_stage_drum_Overhead_Flow_pred_scaled = ERC_2nd_stage_drum_Overhead_Flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    ERC_2nd_stage_drum_Overhead_Flow_pred = scaler_y_ERC_2nd_stage_drum_Overhead_Flow.inverse_transform([[ERC_2nd_stage_drum_Overhead_Flow_pred_scaled]]).ravel()
    
    selected_row_updated['ERC_2nd_stage_drum_Overhead_Flow'] = ERC_2nd_stage_drum_Overhead_Flow_pred[0]
    
    # KT1701 steam flow prediction
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_turbine_steam_flow.pkl', 'rb') as f:
        ERC_turbine_steam_flow_kalman = pickle.load(f)
    
    scaler_X_ERC_turbine_steam_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_X_ERC_turbine_steam_flow.pkl')
    scaler_y_ERC_turbine_steam_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_ERC_turbine_steam_flow.pkl')
    
    y_col = "ERC_turbine_steam_flow"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    ERC_turbine_steam_flow_df = selected_row_updated[u_cols]
    
    ERC_turbine_steam_flow_scaled_test = scaler_X_ERC_turbine_steam_flow.transform(ERC_turbine_steam_flow_df) 
    ERC_turbine_steam_flow_kalman.step(y=None, u=ERC_turbine_steam_flow_scaled_test.reshape(-1,1))
    
    ERC_turbine_steam_flow_results = ERC_turbine_steam_flow_kalman.to_dataframe()
    ERC_turbine_steam_flow_pred_scaled = ERC_turbine_steam_flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    ERC_turbine_steam_flow_pred = scaler_y_ERC_turbine_steam_flow.inverse_transform([[ERC_turbine_steam_flow_pred_scaled]]).ravel()
    
    selected_row_updated['ERC_turbine_steam_flow'] = ERC_turbine_steam_flow_pred[0]
    
    # ERC power prediction through Steam flow
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_power.pkl', 'rb') as f:
        ERC_power_kalman = pickle.load(f)
    
    scaler_X_ERC_power = joblib.load("..\\Results\\Model"+"/"+'scaler_X_ERC_power.pkl')
    scaler_y_ERC_power = joblib.load("..\\Results\\Model"+"/"+'scaler_y_ERC_power.pkl')
    
    y_col = "ERC_power"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    ERC_power_df = selected_row_updated[u_cols]
    
    ERC_power_scaled_test = scaler_X_ERC_power.transform(ERC_power_df) 
    ERC_power_kalman.step(y=None, u=ERC_power_scaled_test.reshape(-1,1))
    
    ERC_power_results = ERC_power_kalman.to_dataframe()
    ERC_power_pred_scaled = ERC_power_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    ERC_power_pred = scaler_y_ERC_power.inverse_transform([[ERC_power_pred_scaled]]).ravel()
    
    selected_row_updated['ERC_power'] = ERC_power_pred[0]
    
    #ERC first stage suction flow prediction
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_first_stage_suction_flow.pkl', 'rb') as f:
        ERC_1st_stage_suction_flow_kalman = pickle.load(f)
    
    scaler_X_ERC_1st_stage_suction_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_X_ERC_first_stage_suction_flow.pkl')
    scaler_y_ERC_1st_stage_suction_flow = joblib.load("..\\Results\\Model"+"/"+'scaler_y_ERC_first_stage_suction_flow.pkl')
    
    y_col = "ERC_1ST_STAGE_Suction_FLOW"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    ERC_1st_stage_suction_flow_df = selected_row_updated[u_cols]
    
    ERC_1st_stage_suction_flow_scaled_test = scaler_X_ERC_1st_stage_suction_flow.transform(ERC_1st_stage_suction_flow_df) 
    ERC_1st_stage_suction_flow_kalman.step(y=None, u=ERC_1st_stage_suction_flow_scaled_test.reshape(-1,1))
    
    ERC_1st_stage_suction_flow_results = ERC_1st_stage_suction_flow_kalman.to_dataframe()
    ERC_1st_stage_suction_flow_pred_scaled = ERC_1st_stage_suction_flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    ERC_1st_stage_suction_flow_pred = scaler_y_ERC_1st_stage_suction_flow.inverse_transform([[ERC_1st_stage_suction_flow_pred_scaled]]).ravel()
    
    selected_row_updated['ERC_1ST_STAGE_Suction_FLOW'] =ERC_1st_stage_suction_flow_pred[0]
    
    
    # ERC first stage suction pressure prediction
    
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_ERC_first_stage_suction_pressure.pkl', 'rb') as f:
        ERC_1st_stage_suction_pressure_kalman = pickle.load(f)
    
    scaler_X_ERC_1st_stage_suction_pressure = joblib.load("..\\Results\\Model"+"/"+'scaler_X_ERC_first_stage_suction_pressure.pkl')
    scaler_y_ERC_1st_stage_suction_pressure = joblib.load("..\\Results\\Model"+"/"+'scaler_y_ERC_first_stage_suction_pressure.pkl')
    
    y_col = "ERC_1ST_STAGE_Suction_PRESSURE"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    ERC_1st_stage_suction_pressure_df = selected_row_updated[u_cols]
    
    ERC_1st_stage_suction_pressure_scaled_test = scaler_X_ERC_1st_stage_suction_pressure.transform(ERC_1st_stage_suction_pressure_df) 
    ERC_1st_stage_suction_pressure_kalman.step(y=None, u=ERC_1st_stage_suction_pressure_scaled_test.reshape(-1,1))
    
    ERC_1st_stage_suction_pressure_results = ERC_1st_stage_suction_pressure_kalman.to_dataframe()
    ERC_1st_stage_suction_pressure_pred_scaled = ERC_1st_stage_suction_pressure_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    ERC_1st_stage_suction_pressure_pred = scaler_y_ERC_1st_stage_suction_pressure.inverse_transform([[ERC_1st_stage_suction_pressure_pred_scaled]]).ravel()
    
    selected_row_updated['ERC_1ST_STAGE_Suction_PRESSURE'] = ERC_1st_stage_suction_pressure_pred[0]
    
    #ERC_1ST_STAGE_Suction_PRESSURE user based input
    ERC_1st_stage_suction_pressure_input = user_input_df.loc[user_input_df["Parameter"].str.strip() == "ERC_1ST_STAGE_Suction_PRESSURE", 'Value'].iloc[0]   
    # Convert to float, handle NaN
    try:
        user_value = float(ERC_1st_stage_suction_pressure_input) if pd.notna(ERC_1st_stage_suction_pressure_input) else np.nan
    except (ValueError, TypeError):
        user_value = np.nan
    
    ERC_1st_stage_suction_pressure_value = selected_row_updated["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[0] if isinstance(selected_row_updated["ERC_1ST_STAGE_Suction_PRESSURE"], pd.Series) else selected_row_updated["ERC_1ST_STAGE_Suction_PRESSURE"]
    selected_row_updated["ERC_1ST_STAGE_Suction_PRESSURE"] = user_value if not np.isnan(user_value) else ERC_1st_stage_suction_pressure_value
    
    
    def ERC_section_power (df):
        Density_1st_stage =[]
        rho_1st_stage_flow = []
        for i in range(len(df)):
            T_K = df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15
            P_Pa = df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
            rho_1st_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
            volumetric_flow = df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000 / rho_1st_stage
            rho_1st_stage_flow.append(volumetric_flow)
            Density_1st_stage.append(rho_1st_stage)
               
        df["ERC_Density_1st_stage"] = Density_1st_stage
        df["ERC VOL FLOW 1ST STAGE"] = rho_1st_stage_flow

        Density_2nd_stage =[]
        rho_2nd_stage_flow = []
        for i in range(len(df)):
            T_K = df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15
            P_Pa = df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5
            rho_2nd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
            volumetric_flow = (df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000 / rho_2nd_stage
            rho_2nd_stage_flow.append(volumetric_flow)
            Density_2nd_stage.append(rho_2nd_stage)
            
        df["ERC_Density_2nd_stage"] = Density_2nd_stage   
        df["ERC VOL FLOW 2ND STAGE"] = rho_2nd_stage_flow

        Density_3rd_stage =[]
        rho_3rd_stage_flow = []
        for i in range(len(df)):
            T_K = df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15
            P_Pa = df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000 + 1e5
            rho_3rd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Ethylene')
            volumetric_flow = (df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["ERC_3rd_stage_drum_overhead_FLOW"].iloc[i])  * 1000 / rho_3rd_stage
            rho_3rd_stage_flow.append(volumetric_flow)
            Density_3rd_stage.append(rho_3rd_stage)

        df["ERC_Density_3rd_stage"] = Density_3rd_stage
        df["ERC VOL FLOW 3RD STAGE"] = rho_3rd_stage_flow

        #Compressor power calculation based on isentropic enthalpy at outlet pressure, keeping entropy constant (i.e., isentropic compression).
        eta = 0.80  # Assume Compressor efficiency
        ERC_1st_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
            s1 = PropsSI('S', 'T',df["ERC_1ST_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_1ST_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Ethylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Ethylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = ((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
            ERC_1st_stage_comp_estimated_power.append(power)
            
        df["ERC_1st_stage_comp_estimated_power_MW"] = ERC_1st_stage_comp_estimated_power

        # 2nd stage compressor power
        ERC_2nd_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
            s1 = PropsSI('S', 'T',df["ERC_2nd_stage_drum_Overhead_Temp"].iloc[i] + 273.15 , 'P', df["ERC_2ND_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = (((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i]) * 1000)/3600) * (h2 - h1)/1e6  # MW
            ERC_2nd_stage_comp_estimated_power.append(power)
            
        df["ERC_2nd_stage_comp_estimated_power_MW"] = ERC_2nd_stage_comp_estimated_power

        # 3rd stage compressor power
        ERC_3rd_stage_comp_estimated_power =[]
        # Step 2: Get inlet enthalpy and entropy
        for i in range(len(df)):
            h1 = PropsSI('H', 'T',df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg
            s1 = PropsSI('S', 'T',df["ERC_3RD_STAGE_Suction_TEMP"].iloc[i] + 273.15 , 'P', df["ERC_3RD_STAGE_Suction_PRESSURE"].iloc[i] * 1000+ 1e5, 'Ethylene')  # J/kg.K
            #Get isentropic outlet enthalpy (h2s at P2, s1)
            T2s = PropsSI('T', 'P', df["ERC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Ethylene')  # isentropic outlet temp
            h2s = PropsSI('H', 'P', df["ERC_3RD_STAGE_Discharge_PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Ethylene')  # J/kg
            h2 = h1 + (h2s - h1) / eta
            power = ((df["ERC_1ST_STAGE_Suction_FLOW"].iloc[i] + df["ERC_2nd_stage_drum_Overhead_Flow"].iloc[i] + df["ERC_3rd_stage_drum_overhead_FLOW"].iloc[i]* 1000)/3600) * (h2 - h1)/1e6  # MW
            ERC_3rd_stage_comp_estimated_power.append(power)
            
        df["ERC_3rd_stage_comp_estimated_power_MW"] = ERC_3rd_stage_comp_estimated_power

        df["ERC_Total_estimated_power_MW"] = df["ERC_1st_stage_comp_estimated_power_MW"]+df["ERC_2nd_stage_comp_estimated_power_MW"]+df["ERC_3rd_stage_comp_estimated_power_MW"]

        df['diff_ERC_power_calculated-simulated'] = df['ERC_power']/1000 - df["ERC_Total_estimated_power_MW"]
        return df
    
    selected_row_updated = ERC_section_power(selected_row_updated)
    
    selected_row_updated["Total_Power_(KW)"]= selected_row_updated["CGC_Power_KW"] + selected_row_updated["PRC_Total_estimated_power_MW"]*1000 + selected_row_updated["ERC_power"]
    selected_row_updated["Total_required_steam_flow_(TPH)"] = selected_row_updated["CGC_Turbine_HP_Steam_flow"] + selected_row_updated["PRC_turbine_Calculated_Steam_flow_TPH"] + selected_row_updated["ERC_turbine_steam_flow"]
    
    # Ethylene loss to fuel prediction
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_Ethylene_loss_to_fuel.pkl', 'rb') as f:
        Ethylene_loss_to_fuel_kalman = pickle.load(f)
        
    scaler_X_Ethylene_loss_to_fuel = joblib.load("..\\Results\\Model"+"/"+"scaler_X_Ethylene_loss_to_fuel.pkl")
    scaler_y_Ethylene_loss_to_fuel= joblib.load( "..\\Results\\Model"+"/"+'scaler_y_Ethylene_loss_to_fuel.pkl')
    
    y_col = "TOTAL_ETHYLENE_LOSS_to_fuel"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    Ethylene_loss_to_fuel_df = selected_row_updated[u_cols]
    
    Ethylene_loss_to_fuel_scaled_test = scaler_X_Ethylene_loss_to_fuel.transform(Ethylene_loss_to_fuel_df) 
    Ethylene_loss_to_fuel_kalman.step(y=None, u=Ethylene_loss_to_fuel_scaled_test.reshape(-1,1))
    
    Ethylene_loss_to_fuel_results = Ethylene_loss_to_fuel_kalman.to_dataframe()
    Ethylene_loss_to_fuel_pred_scaled = Ethylene_loss_to_fuel_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    Ethylene_loss_to_fuel_pred = scaler_y_Ethylene_loss_to_fuel.inverse_transform([[Ethylene_loss_to_fuel_pred_scaled]]).ravel()
    
    selected_row_updated['TOTAL_ETHYLENE_LOSS_to_fuel'] = Ethylene_loss_to_fuel_pred[0]
    
    # Ethylene product flow prediction
    with open("..\\Results\\Model"+"/"+'kalman_filter_model_Ethylene_product_flow.pkl', 'rb') as f:
        Ethylene_product_flow_kalman = pickle.load(f)
        
    scaler_X_Ethylene_product_flow = joblib.load("..\\Results\\Model"+"/"+"scaler_X_Ethylene_product_flow.pkl")
    scaler_y_Ethylene_product_flow= joblib.load( "..\\Results\\Model"+"/"+'scaler_y_Ethylene_product_flow.pkl')
    
    y_col = "Ethylene_product_flow"
    u_cols = config_df_model_details[config_df_model_details["Predicted parameter"] == y_col]
    u_cols = u_cols.dropna(axis =1)
    u_cols = u_cols.iloc[:, 1:].values
    u_cols = u_cols.ravel().tolist()
    
    Ethylene_product_flow_df = selected_row_updated[u_cols]
    
    Ethylene_product_flow_scaled_test = scaler_X_Ethylene_product_flow.transform(Ethylene_product_flow_df) 
    Ethylene_product_flow_kalman.step(y=None, u=Ethylene_product_flow_scaled_test.reshape(-1,1))
    
    Ethylene_product_flow_results = Ethylene_product_flow_kalman.to_dataframe()
    Ethylene_product_flow_pred_scaled = Ethylene_product_flow_results[('$y_0$', 'filtered', 'output')].iloc[-1]  # Only last value
    Ethylene_product_flow_pred = scaler_y_Ethylene_product_flow.inverse_transform([[Ethylene_product_flow_pred_scaled]]).ravel()
    
    selected_row_updated['Ethylene_product_flow'] = Ethylene_product_flow_pred[0]
    
    Actual_vs_estimated = pd.concat([selected_row, selected_row_updated], axis=0)
    Actual_vs_estimated.index = ["actual", "estimated"]
    
    def color_diff(val):
        color = 'green' if val > 0 else 'red' if val < 0 else ''
        return f'background-color: {color}' if color else ''
    
    diff = Actual_vs_estimated.loc['estimated'] - Actual_vs_estimated.loc['actual']
    Actual_vs_estimated["Timestamp"] = user_time
    styled = Actual_vs_estimated.style.apply(lambda _: diff.map(color_diff), axis=1)
    styled.to_excel("..\\Results\\Actual_vs_estimated what if.xlsx", engine='openpyxl')

#%% cold box correlation matrix formation

# df_coldbox = pd.read_excel(file_path + "/" + 'DMC_Screen_tags_data.xlsx',sheet_name= "Cold section data")
# df_coldbox.columns = df_coldbox.iloc[0]
# df_coldbox = df_coldbox[1:].reset_index(drop=True)
# df_coldbox.set_index("Timestamp",inplace =True)
# df_coldbox_corr = df_coldbox.corr (method = "pearson").round(2)

# def color_columns(val):
#     if val > 0.4:
#         color = 'background-color: green'
#     elif val < -0.4:
#         color = 'background-color: red'
#     else:
#         color = ''
#     return color

# with pd.ExcelWriter("..\\Results" +'/'+'correlation_with_color_conditioning_coldbox.xlsx', engine='openpyxl') as writer:
#     # Write the DataFrame to the Excel file with styling
#     df_coldbox_corr.style.applymap(color_columns).to_excel(writer, index=True)


