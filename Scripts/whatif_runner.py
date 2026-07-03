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

file_path= "..\\Data"

def load_process_data() -> pd.DataFrame:
    """Load and preprocess DMC_Screen_tags_data.xlsx PI data sheet."""
    df = pd.read_excel(
        os.path.join("..\\Results", 'Yanpet_OLF1_MultiX_multiY_results.xlsx'),
    )
    df.set_index('Timestamp', inplace=True)
    return df

# df= load_process_data()

#%% LBT creation

# # ============================== 
# # CONFIGURATION SECTION
# # ==============================

# # EXCEL_FILE = "LBT_Input.xlsx"            # 📥 Input Excel file with test, clean, and config data
# # OUTPUT_FILE = "LBT_Result.xlsx"      # 📤 Output Excel file with results
# # MAX_BENCHMARKS = 10                           # 🧮 Max benchmark rows to pick per match

# # SHEET_T = "test_data"                         # 🧪 Sheet with actual test data
# # SHEET_C = "clean_data"                        # ✅ Sheet with clean benchmark/reference data
# # SHEET_L = "lbt_input"                         # ⚙️ Sheet with matching rules and tolerances

# # ==============================
# # FUNCTION DEFINITIONS
# # ==============================

# T = df.copy()
# C = df.copy()
# L = pd.read_excel(file_path + "/" + 'DMC_Screen_tags_data.xlsx',sheet_name= "lbt_input" )
# # 🕒 Convert Timestamp to datetime
# T = T.reset_index(drop=False)
# C = C.reset_index(drop=False)

# T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
# C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")

# # 📏 Dictionary: Match tag → decimal precision
# decimal_dict = L.set_index("Match_tags")["Match_Tag_Decimal"].dropna().to_dict()
# MAX_BENCHMARKS = 10   

# # def load_data(excel_file):
# #     # 🔄 Load all input sheets
# #     T = pd.read_excel(excel_file, sheet_name=SHEET_T)
# #     C = pd.read_excel(excel_file, sheet_name=SHEET_C)
# #     L = pd.read_excel(excel_file, sheet_name=SHEET_L)

# #     # 🕒 Convert Timestamp to datetime
# #     T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
# #     C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")

# #     # 📏 Dictionary: Match tag → decimal precision
# #     decimal_dict = L.set_index("Match_tags")["Match_Tag_Decimal"].dropna().to_dict()
# #     return T, C, L, decimal_dict

# def build_result(row_t, matched_row, T, C, L, match_tags, iteration_unused, decimal_dict):
#     result = {}

#     # ✅ Add actual and benchmark values for T columns
#     for col in T.columns:
#         val_actual = row_t[col]
#         val_benchmark = matched_row.get(col, np.nan)

#         if col in decimal_dict:
#             decimals = int(decimal_dict[col])
#             val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
#             val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark

#         result[f"{col}_Actual"] = val_actual
#         result[f"{col}_benchmark"] = val_benchmark

#     # ➕ Add extra benchmark-only columns from C
#     for col in C.columns:
#         if col not in T.columns:
#             val_benchmark = matched_row.get(col, np.nan)
#             if col in decimal_dict:
#                 decimals = int(decimal_dict[col])
#                 val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark
#             result[f"{col}_benchmark"] = val_benchmark

#     matched_iterations = []

#     # 🔁 For each matching tag, calculate delta and find which iteration matched
#     for tag in match_tags:
#         delta_col = f"{tag}_delta"
#         iter_col = f"matched_iteration_{tag}"

#         actual_val = row_t.get(tag, np.nan)
#         benchmark_val = matched_row.get(tag, np.nan)

#         if pd.notnull(actual_val) and pd.notnull(benchmark_val):
#             decimals = int(decimal_dict.get(tag, 4))
#             actual_val = round(actual_val, decimals)
#             benchmark_val = round(benchmark_val, decimals)
#             delta = round(benchmark_val - actual_val, decimals)
#             result[delta_col] = delta

#             found_iter = None
#             for _, tol_row in L[L["Match_tags"] == tag].iterrows():
#                 it_no = tol_row["Iteration No"]
#                 min_tol = tol_row["Tolerance_minimum"]
#                 max_tol = tol_row["Tolerance_maximum"]
#                 if -min_tol <= delta <= max_tol:
#                     found_iter = it_no
#                     break

#             result[iter_col] = found_iter
#             if found_iter is not None:
#                 matched_iterations.append(found_iter)
#         else:
#             result[delta_col] = np.nan
#             result[iter_col] = None

#     result["iteration_matched"] = max(matched_iterations) if matched_iterations else None
#     return result

# def build_no_match_result(row_t, T, C, match_tags, decimal_dict):
#     result = {}

#     # 🚫 If no match is found, store actuals and fill rest with "BENCHMARK NOT FOUND".
#     for col in T.columns:
#         val_actual = row_t[col]
#         if col in decimal_dict:
#             decimals = int(decimal_dict[col])
#             val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
#         result[f"{col}_Actual"] = val_actual
#         result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

#     for col in C.columns:
#         if col not in T.columns:
#             result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

#     for tag in match_tags:
#         result[f"{tag}_delta"] = "BENCHMARK NOT FOUND"
#         result[f"matched_iteration_{tag}"] = "BENCHMARK NOT FOUND"

#     result["iteration_matched"] = "BENCHMARK NOT FOUND"
#     return result

# def match_records(T, C, L, max_benchmarks, decimal_dict):
#     results = []
#     match_tags = L["Match_tags"].dropna().unique().tolist()
#     max_iterations = int(pd.to_numeric(L["Iteration No"], errors="coerce").dropna().max())

#     for _, row_t in T.iterrows():
#         match_found = False

#         for iteration in range(1, max_iterations + 1):
#             current_tol = L[L["Iteration No"] == iteration]
#             filtered_C = C.copy()

#             for _, tol_row in current_tol.iterrows():
#                 col = tol_row["Match_tags"]
#                 min_tol = tol_row["Tolerance_minimum"]
#                 max_tol = tol_row["Tolerance_maximum"]

#                 if col not in T.columns or col not in C.columns or col == "Timestamp":
#                     continue

#                 val_t = row_t[col]
#                 if pd.isnull(val_t):
#                     continue

#                 if col in decimal_dict:
#                     decimals = int(decimal_dict[col])
#                     val_t = round(val_t, decimals)
#                     filtered_C[col] = filtered_C[col].round(decimals)

#                 filtered_C = filtered_C[
#                     (filtered_C[col] >= val_t - min_tol) & (filtered_C[col] <= val_t + max_tol)
#                 ]

#             if not filtered_C.empty:
#                 match_found = True
#                 for _, matched_row in filtered_C.head(max_benchmarks).iterrows():
#                     results.append(build_result(row_t, matched_row, T, C, L, match_tags, iteration, decimal_dict))
#                 break

#         if not match_found:
#             results.append(build_no_match_result(row_t, T, C, match_tags, decimal_dict))

#     return results

# def filter_optimal_rows(df, L):
#     perf_config = L[["performance_tag", "direction"]].dropna().drop_duplicates()

#     if perf_config.empty:
#         print("⚠️ No performance tag found in config - skipping filtering.")
#         return df

#     perf_tag = perf_config.iloc[0]["performance_tag"]
#     direction = perf_config.iloc[0]["direction"].strip().lower()
#     perf_col = f"{perf_tag}_benchmark"

#     if perf_col not in df.columns:
#         print(f"⚠️ Performance column '{perf_col}' not found - skipping filtering.")
#         return df

#     df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")
#     matched_df = df.dropna(subset=[perf_col])
#     unmatched_df = df[df[perf_col].isna()]

#     if direction == "minimize":
#         optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmin()]
#     elif direction == "maximize":
#         optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmax()]
#     else:
#         raise ValueError(f"Unknown direction '{direction}' in performance config.")

#     unmatched_latest = unmatched_df.sort_values("Timestamp_Actual").drop_duplicates(
#         subset=["Timestamp_Actual"], keep="last"
#     )

#     return pd.concat(
#         [optimal_df, unmatched_latest[~unmatched_latest["Timestamp_Actual"].isin(optimal_df["Timestamp_Actual"])]],
#         ignore_index=True,
#     ).sort_values("Timestamp_Actual").reset_index(drop=True)

# def process_results(results, T, C, L):
#     df = pd.DataFrame(results)
#     df.dropna(axis=1, how="all", inplace=True)
#     df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

#     for col in [c for c in df.columns if c.endswith("Timestamp_benchmark")]:
#         df[col] = df[col].fillna("BENCHMARK NOT FOUND")  # Consistent fill

#     actual_bench_cols = []
#     for col in T.columns:
#         if f"{col}_Actual" in df.columns: actual_bench_cols.append(f"{col}_Actual")
#         if f"{col}_benchmark" in df.columns: actual_bench_cols.append(f"{col}_benchmark")

#     for col in C.columns:
#         if col not in T.columns and f"{col}_benchmark" in df.columns:
#             actual_bench_cols.append(f"{col}_benchmark")

#     delta_cols = sorted([c for c in df.columns if c.endswith("_delta")])
#     iter_cols = sorted([c for c in df.columns if c.startswith("matched_iteration_")])

#     final_cols = actual_bench_cols + delta_cols + iter_cols
#     if "iteration_matched" in df.columns:
#         final_cols.append("iteration_matched")

#     remaining = [c for c in df.columns if c not in final_cols]
#     df = df[final_cols + remaining]

#     return filter_optimal_rows(df, L)

# # def save_results(df, output_file):
# #     df.to_excel(output_file, index=False)
# #     print(f"✅ Results saved to: {output_file}")

# # ==============================
# # MAIN EXECUTION BLOCK
# # ==============================

# # 📦 Load input data
# # T, C, L, decimal_dict = load_data(EXCEL_FILE)

# # 🔍 Perform record matching
# results = match_records(T, C, L, MAX_BENCHMARKS, decimal_dict)

# # 🧹 Clean and organize results
# final_df = process_results(results, T, C, L)

# # 💾 Save to Excel
# # save_results(final_df, OUTPUT_FILE)

# final_df.to_excel("..\\Results"+"/"+"YP_OlF1_LBT_result.xlsx")


#%% what if preparation for YANPET OLF1
def whatif_analysis(df, user_time, user_input_df):
    config_df_model_details = pd.read_excel(file_path +"/" + "Config_file.xlsx", sheet_name= 'Model details')
    constraints_df = pd.read_excel(file_path + "/" + 'Config_file.xlsx',sheet_name= "Constraints")
    user_time = user_time
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
        
        return styled  


























