# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:20:03 2025

@author: sukumar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import os
from Plot_func_file import create_histogram_density_plot,data_division_in_bins_with_same_amnt_data_plots,Parameters_line_chart, Process_parametrs_boxplot
from plotly.offline import plot
import re
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from ML_Function_file import random_forest, Random_searchCV_rf,linear_regression,Ridge_regression,Lasso_regression,remove_outliers_by_boxplot,Grid_searchCV_rf, Xgboost  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
TF_ENABLE_ONEDNN_OPTS=0
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.keras import regularizers
import joblib
#Auto ML libraries
#from lazypredict.Supervised import LazyRegressor
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
import time
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
import networkx as nx
from collections import OrderedDict
import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill
from Hample_filter import hampel_filter_dataframe, plotly_plot_all_columns
from Hunting_identification_algorithm import detect_hunting_tags
from sklearn.inspection import partial_dependence

#%% 
def lasso_selection(x_train_scaled, y_train, x_test_scaled, y_test, df, target_column,iteration):
    # Run Lasso regression
    Lasso_model_result, Lasso = Lasso_regression(x_train_scaled, y_train, x_test_scaled, y_test)

    # Extract coefficients and compute absolute values
    feat_df = Lasso_model_result["Lasso_CV_reg coef"].copy()
    feat_df["abs_coef"] = np.abs(feat_df["coefficents"])
    feat_df = feat_df[feat_df["abs_coef"] > 0].sort_values(by="abs_coef", ascending=False)

    # Normalize scores (0–10)
    max_coef = feat_df["abs_coef"].max()
    feat_df["score_norm"] = feat_df["abs_coef"] / max_coef if max_coef > 0 else 0.0
    feat_df["score_0_10"] = feat_df["score_norm"] * 10

    # Normalize scores (0–100)
    min_score, max_score = feat_df["score_0_10"].min(), feat_df["score_0_10"].max()
    feat_df["score_0_100"] = ((feat_df["score_0_10"] - min_score) / (max_score - min_score)) * 100 if max_score > min_score else 0.0

    # Select top features
    selected_features = list(feat_df.index)
    final_df = df[selected_features + [target_column]].copy()

    # Prepare score DataFrame for export
    score_df = (
        feat_df[["score_0_10", "score_0_100"]]
        .reset_index()
        .rename(columns={"index": "tag_name"})
        .sort_values(by="score_0_10", ascending=False)
    )
    
    if iteration == 1:
        result_folder_path ="..//Results_iter1"
    elif iteration == 2:
        result_folder_path ="..//Results_iter2"        
    elif iteration == 3:
        result_folder_path ="..//Results_iter3"
    else:
        raise ValueError("Iteration must be 1, 2, or 3")
        
    lasso_score_folder = os.path.join(result_folder_path,"Feature_Scores")
    os.makedirs(lasso_score_folder, exist_ok=True)
    output_path = os.path.join(lasso_score_folder, "feature_scores.xlsx")
    
    # ✅ Write data to Sheet1, columns A–C
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        score_df.to_excel(writer, index=False, sheet_name="Sheet1", startcol=0, startrow=0)

    print(f"✅ Lasso feature scores saved to: {output_path}")

    return final_df

def pdp_selection(x_train, y_train, x_test, y_test, df, target_column, config_data,iteration):
    # Separate target and feature columns
    x_target_columns = df.drop(columns=target_column)
    y_target_column = df[[target_column]]

    x_train = x_train[x_target_columns.columns]
    x_test = x_test[x_target_columns.columns]

    # 🔍 Random search for best hyperparameters
    Random_search_result = Random_searchCV_rf(x_train, y_train)
    rf_result, rf = random_forest(
        x_train, y_train, x_test, y_test,
        n_estimators=Random_search_result['n_estimators'],
        max_depth=Random_search_result['max_depth'],
        max_features=Random_search_result['max_features'],
        criterion="squared_error",
        min_samples_split=Random_search_result['min_samples_split'],
        min_samples_leaf=Random_search_result['min_samples_leaf'],
        n_jobs=-1, random_state=42
    )

    # 📊 PDP-based feature scoring
    pdp_variations = []
    for feature in x_target_columns.columns:
        pdp = partial_dependence(rf, X=x_test, features=[feature], grid_resolution=100)
        pd_values = pdp['average'][0]
        pd_range = np.ptp(pd_values)   # range of PDP
        pd_std = np.std(pd_values)     # standard deviation
        pd_score = (pd_range + pd_std) / 2  # combined score
        pdp_variations.append((feature, pd_range, pd_std, pd_score))

    # 🧾 Create PDP DataFrame
    pdp_df = pd.DataFrame(pdp_variations, columns=['Feature', 'PDP Range', 'PDP StdDev', 'PDP Score'])
    pdp_df = pdp_df.sort_values(by='PDP Score', ascending=False).reset_index(drop=True)

    # 🎯 Normalize scores between 0–100
    max_val = pdp_df["PDP Score"].max()
    min_val = pdp_df["PDP Score"].min()
    if max_val > min_val:
        pdp_df["PDP_100"] = ((pdp_df["PDP Score"] - min_val) / (max_val - min_val)) * 100
    else:
        pdp_df["PDP_100"] = 0.0

    # ⚙️ Get number of features to select
    num_features_to_select = config_data.loc[
        config_data["input"] == "Number_of_feature_selected_through_PDP", "description"
    ].iloc[0]
    num_features_to_select = int(num_features_to_select)
    num_features_in_df = x_target_columns.shape[1]

    num_features_final = min(num_features_to_select, num_features_in_df)
    selected_features = list(pdp_df["Feature"].head(num_features_final))

    # ✅ Final filtered DataFrame
    final_df = df[selected_features + [target_column]].copy()

    # ✅ Prepare feature score DataFrame
    score_df = pdp_df[["Feature", "PDP Score", "PDP_100"]].copy()
    
    if iteration == 1:
        result_folder_path ="..//Results_iter1"
    elif iteration == 2:
        result_folder_path ="..//Results_iter2"
    elif iteration == 3:
        result_folder_path ="..//Results_iter3"
    else:
        raise ValueError("Iteration must be 1, 2, or 3")
        
    lasso_score_path = os.path.join(result_folder_path, "Feature_Scores","feature_scores.xlsx")
    
    # If Lasso file exists, append columns D–E
    if os.path.exists(lasso_score_path):
        existing_df = pd.read_excel(lasso_score_path)
        existing_df = existing_df.rename(columns={"tag_name": "Feature"})
        merged = pd.merge(existing_df, score_df, on="Feature", how="left")

        # Reorder columns neatly
        merged = merged.rename(columns={
            "score_0_10": "lasso_score",
            "score_0_100": "lasso_100",
            "PDP Score": "pdp_score",
            "PDP_100": "pdp_100"
        })[["Feature", "lasso_score", "lasso_100", "pdp_score", "pdp_100"]]

        merged.to_excel(lasso_score_path, index=False)
    else:
        os.makedirs(os.path.dirname(lasso_score_path), exist_ok=True)
        
        merged = score_df.rename(columns={"Feature": "tag_name", "PDP Score": "pdp_score", "PDP_100": "pdp_100"})
        merged.to_excel(lasso_score_path, index=False)

    print(f"✅ PDP scores added to existing Excel: {lasso_score_path}")

    return final_df
def shap_selection(x_train, y_train, x_test, df, target_column, config_data,iteration):
    # Separate target and features
    x_features = df.drop(columns=target_column)
    y_target = df[[target_column]]

    x_train = x_train[x_features.columns]
    x_test = x_test[x_features.columns]

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train.values.ravel())

    # Compute SHAP values
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_train)

    # Calculate mean absolute SHAP value per feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = x_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': shap_importance
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    # 🎯 Normalize scores between 0–100
    max_val = feature_importance_df["Mean_Abs_SHAP"].max()
    min_val = feature_importance_df["Mean_Abs_SHAP"].min()
    if max_val > min_val:
        feature_importance_df["SHAP_100"] = ((feature_importance_df["Mean_Abs_SHAP"] - min_val) / (max_val - min_val)) * 100
    else:
        feature_importance_df["SHAP_100"] = 0.0

    # Get top features from config
    num_features_to_select = config_data.loc[
        config_data["input"] == "Number_of_feature_selected_through_SHAP", "description"
    ].iloc[0]
    num_features_to_select = int(num_features_to_select)
    num_features_available = x_features.shape[1]
    num_features_final = min(num_features_to_select, num_features_available)

    # Select top features
    selected_features = feature_importance_df["Feature"].head(num_features_final).tolist()

    # Filter dataframe for selected features + target
    final_df = df[selected_features + [target_column]].copy()

    # Prepare SHAP score DataFrame
    shap_score_df = feature_importance_df[["Feature", "Mean_Abs_SHAP", "SHAP_100"]].copy()
    if iteration == 1:
        result_folder_path ="..//Results_iter1"
    elif iteration == 2:
        result_folder_path ="..//Results_iter2"
    elif iteration == 3:
        result_folder_path ="..//Results_iter3"
    else:
        raise ValueError("Iteration must be 1, 2, or 3")
    lasso_score_path = os.path.join(result_folder_path, "Feature_Scores","feature_scores.xlsx")
    os.makedirs(os.path.dirname(lasso_score_path), exist_ok=True)

    # ✅ Merge into the same Excel (with Lasso + PDP)
    if os.path.exists(lasso_score_path):
        existing_df = pd.read_excel(lasso_score_path)
        # Handle column name variations
        if "tag_name" in existing_df.columns:
            existing_df = existing_df.rename(columns={"tag_name": "Feature"})

        merged = pd.merge(existing_df, shap_score_df, on="Feature", how="left")

        merged = merged.rename(columns={
            "score_0_10": "lasso_score",
            "score_0_100": "lasso_100",
            "PDP Score": "pdp_score",
            "PDP_100": "pdp_100",
            "Mean_Abs_SHAP": "shap_score",
            "SHAP_100": "shap_100"
        })

        # ✅ Ensure all expected columns are present (fill missing)
        for col in ["lasso_score", "lasso_100", "pdp_score", "pdp_100", "shap_score", "shap_100"]:
            if col not in merged.columns:
                merged[col] = np.nan

        merged = merged[["Feature", "lasso_score", "lasso_100", "pdp_score", "pdp_100", "shap_score", "shap_100"]]
        merged.to_excel(lasso_score_path, index=False)
    else:
        # If file doesn’t exist yet, create it fresh
        merged = shap_score_df.rename(columns={
            "Feature": "tag_name",
            "Mean_Abs_SHAP": "shap_score",
            "SHAP_100": "shap_100"
        })
        merged.to_excel(lasso_score_path, index=False)

    print(f"✅ SHAP scores added to Excel: {lasso_score_path}")

    return final_df


# ========================
# Controller Function
# ========================
def feature_selection_pipeline(iteration, x_train, y_train, x_test,y_test, x_train_scaled, x_test_scaled, df, Target_column, config_data):
    if iteration == 1:
        df = lasso_selection(x_train_scaled, y_train, x_test_scaled, y_test, df, Target_column,iteration)
        df = pdp_selection(x_train, y_train, x_test, y_test, df, Target_column, config_data,iteration)
    elif iteration == 2:
        df = pdp_selection(x_train, y_train, x_test, y_test, df, Target_column, config_data,iteration)
        df = shap_selection(x_train, y_train, x_test, df, Target_column, config_data,iteration)
    elif iteration == 3:
        df = shap_selection(x_train, y_train, x_test, df, Target_column, config_data,iteration)
    else:
        raise ValueError("Iteration must be 1, 2, or 3")
    return df

#%% Run ML pieline
def run_ml_pipeline(df, Target_column, Result_folder_path, 
                    x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test):
    """
    Run ML models on given dataframe and save results in Result_folder_path.
    """

    # ----------------------
    # Preprocess X, y
    # ----------------------
    x_Target_column = df.drop(columns=Target_column)
    y_Target_column = df[[Target_column]]

    x_train = x_train[x_Target_column.columns]
    x_test = x_test[x_Target_column.columns]
    x_train_scaled = x_train_scaled[x_Target_column.columns]
    x_test_scaled = x_test_scaled[x_Target_column.columns]

    # ----------------------
    # Linear Regression
    # ----------------------
    LR_model_result, Lr = linear_regression(x_train_scaled, y_train, x_test_scaled, y_test)
    LR_result_folder_path = os.path.join(Result_folder_path, "Linear regression model")
    os.makedirs(LR_result_folder_path, exist_ok=True)

    with open(LR_result_folder_path + "/Linear regression parameters & accuracy matrix.txt", "w") as file:
        file.write("Linear Regression Results\n")
        file.write(f"columns: {list(x_Target_column.columns)}\n")
        file.write(f"Intercept: {LR_model_result['intercept'][0]}\n")
        file.write(f"Coefficient: {np.round(LR_model_result['linear reg coef'][0], 2)}\n")
        file.write(f"MAE train set: {np.round(LR_model_result['MAE train set '], 2)}\n")
        file.write(f"RMSE train set: {np.round(LR_model_result['RMSE train set '], 2)}\n")
        file.write(f"MAPE train set: {np.round(LR_model_result['MAPE train set '], 3)}\n")
        file.write(f"R-square of train set: {np.round(LR_model_result['R-square of train set '], 2)}\n")
        file.write(f"MAE test: {np.round(LR_model_result['MAE test '], 2)}\n")
        file.write(f"RMSE test: {np.round(LR_model_result['RMSE test '], 2)}\n")
        file.write(f"MAPE test: {np.round(LR_model_result['MAPE test '], 3)}\n")
        file.write(f"R-square of test set: {np.round(LR_model_result['R-square of test set '], 2)}\n")

    LR_ytrain_hat = Lr.predict(x_train_scaled)
    plt.scatter(y_train, LR_ytrain_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (train set)")
    plt.savefig(LR_result_folder_path +'/'+"LR train set prediction.jpeg", dpi=1080)
    plt.close()

    LR_ytest_hat = Lr.predict(x_test_scaled)
    plt.scatter(y_test, LR_ytest_hat)
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (test set)")
    plt.savefig(LR_result_folder_path +'/'+"LR test set prediction.jpeg", dpi=1080)
    plt.close()

    with open (LR_result_folder_path+"/"+"Linear_regression.pkl","wb") as file:
        pickle.dump(Lr,file)

    # ----------------------
    # Lasso Regression
    # ----------------------
    Lasso_model_result, Lasso = Lasso_regression(x_train_scaled,y_train,x_test_scaled,y_test)
    Lasso_result_folder_path = Result_folder_path + "/"+"Lasso regression model"
    os.makedirs(Lasso_result_folder_path,exist_ok=True)

    with open(Lasso_result_folder_path + "/Lasso regression parameters & accuracy matrix.txt", "w") as file:
        file.write("Lasso Regression Results\n")
        file.write(f"columns: {list(x_Target_column.columns)}\n")
        file.write(f"Intercept: {Lasso_model_result['intercept']}\n")
        file.write(f"Coefficient: {np.round(Lasso_model_result['Lasso_CV_reg coef'], 2)}\n")
        file.write(f"MAE train set: {np.round(Lasso_model_result['MAE train set '], 2)}\n")
        file.write(f"RMSE train set: {np.round(Lasso_model_result['RMSE train set '], 2)}\n")
        file.write(f"MAPE train set: {np.round(Lasso_model_result['MAPE train set '], 3)}\n")
        file.write(f"R-square of train set: {np.round(Lasso_model_result['R-square of train set '], 2)}\n")
        file.write(f"MAE test: {np.round(Lasso_model_result['MAE test '], 2)}\n")
        file.write(f"RMSE test: {np.round(Lasso_model_result['RMSE test '], 2)}\n")
        file.write(f"MAPE test: {np.round(Lasso_model_result['MAPE test '], 3)}\n")
        file.write(f"R-square of test set: {np.round(Lasso_model_result['R-square of test set '], 2)}\n")

    Lasso_ytrain_hat = Lasso.predict(x_train_scaled)
    plt.scatter(y_train, Lasso_ytrain_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (train set)")
    plt.savefig(Lasso_result_folder_path +'/'+"Lasso train set prediction.jpeg", dpi=1080)
    plt.close()

    Lasso_ytest_hat = Lasso.predict(x_test_scaled)
    plt.scatter(y_test, Lasso_ytest_hat)
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (test set)")
    plt.savefig(Lasso_result_folder_path +'/'+"Lasso test set prediction.jpeg", dpi=1080)
    plt.close()

    with open (Lasso_result_folder_path+"/"+"Lasso_regression.pkl","wb") as file:
        pickle.dump(Lasso,file)

    Feature_extraction_through_Lasso = Lasso_model_result["Lasso_CV_reg coef"]
    Feature_extraction_through_Lasso["coefficents"] = np.abs(Feature_extraction_through_Lasso["coefficents"])
    Feature_extraction_through_Lasso = Feature_extraction_through_Lasso[Feature_extraction_through_Lasso["coefficents"]!=0]
    Feature_extraction_through_Lasso = Feature_extraction_through_Lasso[Feature_extraction_through_Lasso["coefficents"]>0.01]

    # ----------------------
    # Random Forest
    # ----------------------
    Random_search_result=Random_searchCV_rf(x_train,y_train)
    # random forest implementation using random search CV
    n_estimators=Random_search_result['n_estimators']
    max_depth=Random_search_result['max_depth']
    criterion="squared_error"
    min_samples_split=Random_search_result['min_samples_split']
    min_samples_leaf=Random_search_result['min_samples_leaf']
    n_jobs=-1
    max_features=Random_search_result['max_features']
    random_state=42 

    rf_result,rf=random_forest(x_train, y_train, x_test, y_test,n_estimators=n_estimators,max_depth=max_depth,
                    max_features=max_features,criterion=criterion,min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf, n_jobs=n_jobs,random_state=random_state)

    RF_result_folder_path = Result_folder_path + "/"+"Random forest model"
    os.makedirs(RF_result_folder_path,exist_ok=True)
    with open(RF_result_folder_path+"/"+"Random forest parameters & accuracy matrix.txt", "w") as file:
        file.write("Random forest Results\n")
        file.write(f"columns: {list(x_Target_column.columns)}\n")
        file.write(f"MAE train set: {np.round(rf_result['MAE train set '],2)}\n")
        file.write(f"RMSE train set: {np.round(rf_result['RMSE train set '], 2)}\n")
        file.write(f"MAPE train set: {np.round(rf_result['MAPE train set '], 3)}\n")
        file.write(f"R-square of train set: {np.round(rf_result['R-square of train set '],2)}\n")
        file.write(f"MAE test: {np.round(rf_result[ 'MAE test '], 2)}\n")
        file.write(f"RMSE test: {np.round(rf_result['RMSE test '],2)}\n")
        file.write(f"MAPE test: {np.round(rf_result[ 'MAPE test '],3)}\n")
        file.write(f"R-square of test set: {np.round(rf_result[ 'R-square of test set '],2)}\n\n")
        file.write(f"Random forest hyper parameters: {rf_result['random_forest_param']}\n")

    rf_ytrain_hat = rf.predict(x_train)
    plt.scatter(y_train, rf_ytrain_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (train set)")
    plt.savefig(RF_result_folder_path +'/'+"RF train set prediction.jpeg", dpi=1080)
    plt.close()
 
    rf_ytest_hat = rf.predict(x_test)
    plt.scatter(y_test, rf_ytest_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (test set)")
    plt.savefig(RF_result_folder_path +'/'+"RF test set prediction.jpeg", dpi=1080)
    plt.close()
 
    # Residual plot for test set
    Residuals = rf_ytest_hat-y_test[Target_column]
    Residual_point_count_below_0 = np.count_nonzero(Residuals<0)
    Residual_point_count_above_0 = np.count_nonzero(Residuals>0)
    Residual_point_count_equals_0 = len(Residuals)- Residual_point_count_below_0 - Residual_point_count_above_0
 
    plt.scatter(rf_ytest_hat, Residuals)
    plt.xlabel("Predicted " +Target_column +" test set")
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(RF_result_folder_path +'/'+"Random forest residual plot.jpeg", dpi=1080)
 
    # Create a partial dependency plot
    from sklearn.inspection import PartialDependenceDisplay
    features = list(x_Target_column.columns)
    pdp_folder_path = RF_result_folder_path +"/" + "pdp folder path"
    os.makedirs(pdp_folder_path,exist_ok=True)
 
    for i, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=(6, 4))
        display = PartialDependenceDisplay.from_estimator(
            rf,
            X=x_test,
            features=[feature],         # Only one feature at a time
            grid_resolution=100,
            ax=ax
        )
        filename = f"{pdp_folder_path}/PDP_{feature}.jpeg"
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close(fig)  # Close figure to avoid memory leaks
 
    from sklearn.inspection import partial_dependence
    pdp_variations = []
 
    for feature in x_Target_column.columns:
        pdp = partial_dependence(rf, X=x_test, features=[feature], grid_resolution=100)
        pd_values = pdp['average'][0]
        
        # Measure how much the PDP changes across feature values
        pd_range = np.ptp(pd_values)  # Peak-to-peak range
        pd_std = np.std(pd_values)    # Standard deviation
 
        pdp_variations.append((feature, pd_range, pd_std))
 
    # Create a DataFrame for easy sorting
    pdp_df = pd.DataFrame(pdp_variations, columns=['Feature', 'PDP Range', 'PDP StdDev'])
    pdp_df = pdp_df.sort_values(by='PDP Range', ascending=False)
    pdp_df.to_csv(RF_result_folder_path +"/" + "Pdp variation index.csv")
 
    imp_feature_rf = list(pdp_df["Feature"])
    # df= df[imp_feature_rf + [Target_column]]
           
    with open (RF_result_folder_path+"/"+"Random_Forest_regression.pkl","wb") as file:
        pickle.dump(rf,file)


    # ----------------------
    # XGBoost
    # ----------------------
    n_estimators_Xgboost = 150
    max_depth_Xgboost = 4
    eta=0.1
    Xgboost_result, Xgboost_model = Xgboost(x_train,y_train,x_test,y_test,n_estimators_Xgboost,max_depth_Xgboost,eta)

    XGB_result_folder_path = Result_folder_path + "/"+"XgBoost model"
    os.makedirs(XGB_result_folder_path,exist_ok=True)
    with open(XGB_result_folder_path+"/"+"XgBoost parameters & accuracy matrix.txt", "w") as file:
        file.write("XgBoost Results\n")
        file.write(f"columns: {list(x_Target_column.columns)}\n")
        file.write(f"MAE train set: {np.round(Xgboost_result['MAE train set '], 2)}\n")
        file.write(f"RMSE train set: {np.round(Xgboost_result['RMSE train set '], 2)}\n")
        file.write(f"MAPE train set: {np.round(Xgboost_result['MAPE train set '], 3)}\n")
        file.write(f"R-square of train set: {np.round(Xgboost_result['R-square of train set '], 2)}\n")
        file.write(f"MAE test: {np.round(Xgboost_result[ 'MAE test '], 2)}\n")
        file.write(f"RMSE test: {np.round(Xgboost_result['RMSE test '], 2)}\n")
        file.write(f"MAPE test: {np.round(Xgboost_result[ 'MAPE test '],3)}\n")
        file.write(f"R-square of test set: {np.round(Xgboost_result[ 'R-square of test set '],2)}\n\n")
        file.write(f"XgBoost hyper parameters: {Xgboost_result['XgBoost_param']}\n")

    XgBoost_ytrain_hat = Xgboost_model.predict(x_train)
    plt.scatter(y_train, XgBoost_ytrain_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (train set)")
    plt.savefig(XGB_result_folder_path +'/'+"XgBoost train set prediction.jpeg", dpi=1080)
    plt.close()

    XgBoost_ytest_hat = Xgboost_model.predict(x_test)
    plt.scatter(y_test, XgBoost_ytest_hat)    
    plt.xlabel("Actual "+ Target_column)
    plt.ylabel("Predicted "+ Target_column)
    plt.title("Actual "+ Target_column+  " vs Predicted "+ Target_column +" (test set)")
    plt.savefig(XGB_result_folder_path +'/'+"XgBoost test set prediction.jpeg", dpi=1080)
    plt.close()

    with open (XGB_result_folder_path +"/"+"XgBoost_regression.pkl","wb") as file:
        pickle.dump(Xgboost_model,file)  

    # ----------------------
    # ANN
    # ----------------------
    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Dense(x_Target_column.shape[1],
                                # input_shape=[x_Target_column.shape[1]],
                                #     activation= 'selu',
                                #     # activation= 'linear'
                                # )) 
    # model.add(tf.keras.layers.Dense(12, activation= 'relu')) # relu
    # model.add(tf.keras.layers.Dense(10, activation= 'tanh'))  # tanh
    # model.add(tf.keras.layers.Dense(8, activation= 'relu' ))# relu
    # model.add(tf.keras.layers.Dense(6, activation= 'elu'))  # elu
    # model.add(tf.keras.layers.Dense(4, activation= 'relu')) #relu
    # model.add(tf.keras.layers.Dense(2, activation= 'selu')) #selu
    # model.add(tf.keras.layers.Dense(1, activation= 'relu')) #relu
    # model.add(tf.keras.layers.Dense(1))

#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=1e-06)
#     optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-04)
#     loss = tf.keras.losses.MSE
#     model.compile(loss=loss,
#                 optimizer=optimizer,
#                 metrics=['mean_absolute_percentage_error', 'mean_squared_error'])

#     history = model.fit(x_train_scaled, y_train, batch_size= 64, epochs= 500, callbacks=[callback],
#             validation_data=(x_test_scaled, y_test))

#     def plot_history(history):
#         hist = pd.DataFrame(history.history)
#         hist['epoch'] = history.epoch

#         plt.figure(figsize=(6, 6))
#         plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train MSE')
#         plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val MSE')
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Squared Error')
#         plt.legend()
#         plt.grid(True)
#         plt.title('Training vs Validation MSE')
#         plt.show()

#         plt.figure(figsize=(6, 6))
#         plt.plot(hist['epoch'], hist['mean_absolute_percentage_error'], label='Train MAPE')
#         plt.plot(hist['epoch'], hist['val_mean_absolute_percentage_error'], label='Val MAPE')
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Absolute Percentage Error')
#         plt.legend()
#         plt.grid(True)
#         plt.title('Training vs Validation MAPE')
#         plt.show()

#     plot_history(history)

    # Predict on train and test sets
#     y_train_pred = model.predict(x_train_scaled)
#     y_test_pred = model.predict(x_test_scaled)

#     # Evaluate
#     print("TRAIN")
#     print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
#     print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
#     print(f"R²: {r2_score(y_train, y_train_pred):.4f}")

#     print("\nTEST")
#     print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
#     print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
#     print(f"R²: {r2_score(y_test, y_test_pred):.4f}")

#     ANN_result_dir = os.path.join(Result_folder_path, "ANN model")
#     os.makedirs(ANN_result_dir, exist_ok=True)
#     model_path = os.path.join(ANN_result_dir, "model.keras")
#     model.save(model_path)

#     results_df = pd.DataFrame({
#         'Actual': np.squeeze(y_test),
#         'Predicted': np.squeeze(y_test_pred)
#     })
#     results_df.to_csv(os.path.join(ANN_result_dir, "predictions.csv"), index=False)

#     hist_df = pd.DataFrame(history.history)
#     hist_df.to_csv(os.path.join(ANN_result_dir, "training_history.csv"), index=False)

#     metrics = {
#         "Train_MSE": float(mean_squared_error(y_train, y_train_pred)),
#         "Train_MAE": float(mean_absolute_error(y_train, y_train_pred)),
#         "Train_R2": float(r2_score(y_train, y_train_pred)),
#         "Test_MSE": float(mean_squared_error(y_test, y_test_pred)),
#         "Test_MAE": float(mean_absolute_error(y_test, y_test_pred)),
#         "Test_R2": float(r2_score(y_test, y_test_pred)),
#     }
#     with open(os.path.join(ANN_result_dir, "evaluation_metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=4)

#     # Extract model architecture config and weights
#     model_data = {
#         "config": model.to_json(),  # model architecture as JSON string
#         "weights": model.get_weights()
#     }

#     # Save to pickle
#     with open(ANN_result_dir+"/"+"ANN_model.pkl", "wb") as f:
#         pickle.dump(model_data, f)
        
#%% data drift function execution
def data_drift_check(df, Target_column, config_data):
    X = df.drop(columns=Target_column)
    y = df[[Target_column]]

    # First split into train+test and validation
    split_index = int(len(df) * config_data.loc[config_data["input"] == "Split_index", "description"].iloc[0])    
    # split_index = int(len(df) * 0.1)
    df_train_test = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    # Train-test split
    x_train_ED, x_test_ED, y_train_ED, y_test_ED = train_test_split(
        df_train_test.drop(columns=Target_column),
        df_train_test[[Target_column]],
        random_state=42,
        test_size=0.1
    )
    # Validation set
    x_val_ED = df_val.drop(columns=Target_column)
    y_val_ED = df_val[[Target_column]]

    # Scale using only training data
    scaler = StandardScaler()
    x_train_ED = pd.DataFrame(scaler.fit_transform(x_train_ED), columns=x_train_ED.columns)
    x_test_ED = pd.DataFrame(scaler.transform(x_test_ED), columns=x_test_ED.columns)
    x_val_ED = pd.DataFrame(scaler.transform(x_val_ED), columns=x_val_ED.columns)

    ann = tf.keras.Sequential()
    # Input layer
    ann.add(tf.keras.layers.Dense(
        X.shape[1],
        input_shape=[X.shape[1]],
        kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    # Hidden layers
    ann.add(tf.keras.layers.Dense(30, kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    ann.add(tf.keras.layers.Dense(20, kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    ann.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    ann.add(tf.keras.layers.Dense(20, kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    ann.add(tf.keras.layers.Dense(30, kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.LeakyReLU())

    # Output layer
    ann.add(tf.keras.layers.Dense(X.shape[1], kernel_regularizer=regularizers.L2(0.01)))
    ann.add(tf.keras.layers.LeakyReLU())  # Optional: only if outputs are expected to be positive

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-08)
    optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-03)
    loss = tf.keras.losses.MSE 
    ann.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mean_squared_error'])

    ann.fit(x_train_ED, x_train_ED, batch_size= 128, epochs= 8000, callbacks=[callback])


    # n_features = X.shape[1]
    # ann = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=[n_features]),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(n_features, activation='linear')  # ✅ linear for reconstruction
    # ])

    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-08)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # ann.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    # ann.fit(
    #     x_train_ED, x_train_ED,
    #     batch_size=64, epochs=8000,
    #     callbacks=[callback],
    #     verbose=1
    # )

    # ---------- 1. Predict on train set ----------
    x_train_predict_ED = ann.predict(x_train_ED)
    x_predict_org = scaler.inverse_transform(x_train_predict_ED)
    x_train_org =   scaler.inverse_transform(x_train_ED)

    # ---------- 2. Overall train error ----------
    train_error = np.zeros(4)
    train_error[0] = r2_score(x_train_ED, x_train_predict_ED)
    train_error[1] = mean_squared_error(x_train_ED,x_train_predict_ED)
    train_error[2] = mean_absolute_percentage_error(x_train_ED, x_train_predict_ED)
    train_error[3] = mean_absolute_error(x_train_ED, x_train_predict_ED)

    Autoencoder_decoder_dir = "..\\Results\\Autoencoder_decoder_results"
    os.makedirs(Autoencoder_decoder_dir,exist_ok=True)

    # ---------- 3. Save the model file ----------
    model_path = os.path.join(Autoencoder_decoder_dir, "Autoencoder_model.keras")
    ann.save(model_path)

    # ---------- 4. Save the standard scaler model file ----------
    scaler_path = os.path.join(Autoencoder_decoder_dir, 'scaler_ED.pkl')
    joblib.dump(scaler, scaler_path)

    # ---------- 5. Save overall test error ----------
    df_metrics = pd.DataFrame({
        'R2 Score': [train_error[0]],
        'MSE': [train_error[1]],
        'MAPE': [train_error[2]],
        'MAE': [train_error[3]]
    })
    print(f"Autoencoders training accuracy matrix: {df_metrics}")

    Autoencoder_decoder_train_dir = "..\\Results\\Autoencoder_decoder_results\\Train_results"
    os.makedirs(Autoencoder_decoder_train_dir,exist_ok=True)

    metrics_path_train = os.path.join(Autoencoder_decoder_train_dir, 'training_metrics.csv')
    df_metrics.to_csv(metrics_path_train, index=False)

    # ---------- 6. Save reconstructed train output ----------
    df_reconstructed_train = pd.DataFrame(
        x_predict_org ,
        columns=x_train_ED.columns
    )
    reconstructed_train_path = os.path.join(Autoencoder_decoder_train_dir, 'reconstructed_train_output.csv')
    df_reconstructed_train.to_csv(reconstructed_train_path, index=False)

    # ---------- 7. Save original train output ----------
    df_train_org = pd.DataFrame(
        x_train_org,
        columns=x_train_ED.columns
    )
    df_train_org[Target_column] = y_train_ED[Target_column].values
    train_org_path = os.path.join(Autoencoder_decoder_train_dir, 'Original_train_data.csv')
    df_train_org.to_csv(train_org_path, index=False)

    # ---------- 8. Calculate MAE per column (feature-wise) ----------
    mae_per_column = {}
    for col in x_train_ED.columns:
        mae = mean_absolute_error(x_train_ED[col], x_train_predict_ED[:, x_train_ED.columns.get_loc(col)])
        mae_per_column[col] = mae
        
    df_mae = pd.DataFrame.from_dict(mae_per_column, orient='index', columns=['MAE'])
    df_mae.index.name = 'Feature'

    mae_train_path = os.path.join(Autoencoder_decoder_train_dir, 'mae_per_feature.csv')
    df_mae.to_csv(mae_train_path)

    # ---------- 1. Predict on test set ----------
    x_test_predict_ED = ann.predict(x_test_ED)

    # ---------- 2. Overall test error ----------
    test_error = np.zeros(4)
    test_error[0] = r2_score(x_test_ED, x_test_predict_ED)
    test_error[1] = mean_squared_error(x_test_ED, x_test_predict_ED)
    test_error[2] = mean_absolute_percentage_error(x_test_ED, x_test_predict_ED)
    test_error[3] = mean_absolute_error(x_test_ED, x_test_predict_ED)

    # Save overall test error
    df_test_metrics = pd.DataFrame({
        'R2 Score': [test_error[0]],
        'MSE': [test_error[1]],
        'MAPE': [test_error[2]],
        'MAE': [test_error[3]]
    })
    print(f"Autoencoders testing accuracy matrix: {df_test_metrics}")

    Autoencoder_decoder_test_dir = "..\\Results\\Autoencoder_decoder_results\\Test_results"
    os.makedirs(Autoencoder_decoder_test_dir,exist_ok=True)

    df_test_metrics.to_csv(os.path.join(Autoencoder_decoder_test_dir, 'test_metrics.csv'), index=False)

    # ---------- 3. MAE per feature on test ----------
    mae_test_per_column = {
        col: mean_absolute_error(x_test_ED[col], x_test_predict_ED[:, i])
        for i, col in enumerate(x_test_ED.columns)
    }
    df_mae_test = pd.DataFrame.from_dict(mae_test_per_column, orient='index', columns=['MAE'])
    df_mae_test.index.name = 'Feature'
    df_mae_test.to_csv(os.path.join(Autoencoder_decoder_test_dir, 'test_mae_per_feature.csv'))

    # ---------- 4. Save reconstructed test output ----------
    y_test_reconstructed = scaler.inverse_transform(x_test_predict_ED)
    df_test_reconstructed = pd.DataFrame(y_test_reconstructed, columns=x_test_ED.columns)
    df_test_reconstructed.to_csv(os.path.join(Autoencoder_decoder_test_dir, 'reconstructed_test_output.csv'), index=False)

    # ---------- 5. Save original test output ----------
    y_test_original = scaler.inverse_transform(x_test_ED)
    df_test_org = pd.DataFrame(y_test_original, columns=x_test_ED.columns)
    df_test_org[Target_column] = y_test_ED[Target_column].values
    df_test_org.to_csv(os.path.join(Autoencoder_decoder_test_dir, 'Original_test_data.csv'), index=False)


    # Model health index identification on validation

    # ---------- 1. Predict on validation set ----------
    x_val_predict_ED = ann.predict(x_val_ED)
    x_val_predict_org = scaler.inverse_transform(x_val_predict_ED)
    x_val_org =   scaler.inverse_transform(x_val_ED)

    # ---------- 2. Overall validation error ----------
    val_error = np.zeros(4)
    val_error[0] = r2_score(x_val_ED, x_val_predict_ED)
    val_error[1] = mean_squared_error(x_val_ED, x_val_predict_ED)
    val_error[2] = mean_absolute_percentage_error(x_val_ED, x_val_predict_ED)
    val_error[3] = mean_absolute_error(x_val_ED, x_val_predict_ED)

    # Save overall validation error
    df_val_metrics = pd.DataFrame({
        'R2 Score': [val_error[0]],
        'MSE': [val_error[1]],
        'MAPE': [val_error[2]],
        'MAE': [val_error[3]]
    })
    print(f"Autoencoders validation accuracy matrix: {df_val_metrics}")

    Autoencoder_decoder_val_dir = "..\\Results\\Autoencoder_decoder_results\\Validation_results"
    os.makedirs(Autoencoder_decoder_val_dir,exist_ok=True)

    df_val_metrics.to_csv(os.path.join(Autoencoder_decoder_val_dir, 'validation_metrics.csv'), index=False)

    # ---------- 3. MAE per feature on validation ----------
    mae_val_per_column = {}
    for col in x_val_ED.columns:
        mae = mean_absolute_error(x_val_ED[col], x_val_predict_ED[:, x_val_ED.columns.get_loc(col)])
        mae_val_per_column[col] = mae
        
    df_val_mae = pd.DataFrame.from_dict(mae_val_per_column, orient='index', columns=['MAE'])
    df_val_mae.index.name = 'Feature'
    df_val_mae.to_csv(os.path.join(Autoencoder_decoder_val_dir, 'val_mae_per_feature.csv'))

    # ---------- 4. Save reconstructed validation output ----------
    x_val_reconstructed = scaler.inverse_transform(x_val_ED)
    df_val_reconstructed = pd.DataFrame(x_val_reconstructed, columns=x_val_ED.columns)
    df_val_reconstructed.to_csv(os.path.join(Autoencoder_decoder_val_dir, 'reconstructed_validation_output.csv'), index=False)

    # ---------- 5. Save original validation data ----------
    df_val_org = pd.DataFrame(x_val_org, columns=x_train_ED.columns)
    df_val_org[Target_column] = y_val_ED[Target_column].values
    df_val_org.to_csv(os.path.join(Autoencoder_decoder_val_dir, 'Original_validation_data.csv'), index=False)


    # compare the columns where data drift occurs from train data
    df_diff_train_val_MAE = pd.DataFrame()
    df_diff_train_val_MAE["Train MAE"] = df_mae["MAE"]
    df_diff_train_val_MAE["Validation MAE"] = df_val_mae["MAE"]
    df_diff_train_val_MAE["diff_MAE"] = np.abs(df_diff_train_val_MAE["Validation MAE"] - df_diff_train_val_MAE["Train MAE"])
    df_diff_train_val_MAE_sorted = df_diff_train_val_MAE.sort_values(by="diff_MAE", ascending=False)

    df_diff_train_val_MAE_sorted.to_csv(os.path.join('..\\Results\\Autoencoder_decoder_results', 'Train_validation_MAE_diff.csv'), index=True)

    #𝗝𝗲𝗻𝘀𝗲𝗻-𝗦𝗵𝗮𝗻𝗻𝗼𝗻 𝗗𝗶𝘀𝘁𝗮𝗻𝗰𝗲 for data drift checking in historical avs current data
    # Normalize all columns to be valid probability distributions
    Jensen_Shannon_distance_dir = "..//Results//Data drift matrices//Jensen_Shannon_distance"
    os.makedirs(Jensen_Shannon_distance_dir,exist_ok=True)
    df_div = df.div(df.sum(axis=0), axis=1)
    # Calculate JSD between 'output' and each input column
    output = df[Target_column]
    js_distances = {}

    for col in df_div.columns:
        if col!=Target_column:
            js = jensenshannon(output, df[col])
            js_distances[col]=js

    js_df = pd.DataFrame.from_dict(js_distances, orient='index', columns=['JSDistance'])
    js_df = js_df.sort_values(by='JSDistance')
    js_df.to_csv(os.path.join(Jensen_Shannon_distance_dir, 'JS distance Input to output.csv'))

    """
    This shows how similar each input distribution is to the output. Lower JSDistance = more similar.
    JSD is ideal because:

    A) It’s bounded between 0 (no drift) and 1 (max drift).
    B) It handles probability distributions, so it's perfect for comparing historical vs current distributions.
    C) It’s symmetric and interpretable.
        
    🟢 𝗔𝗱𝘃𝗮𝗻𝘁𝗮𝗴𝗲𝘀 𝗼𝗳 𝗝𝗦 𝗗𝗶𝘀𝘁𝗮𝗻𝗰𝗲:
    𝗦𝘆𝗺𝗺𝗲𝘁𝗿𝘆 𝗮𝗻𝗱 𝗜𝗻𝘁𝗲𝗿𝗽𝗿𝗲𝘁𝗮𝗯𝗶𝗹𝗶𝘁𝘆:
    Unlike many other metrics, JS Distance is symmetric and bounded between 0 and 1, making it easier to interpret and compare across features.

    𝗥𝗲𝗹𝗮𝘁𝗶𝘃𝗲 𝗦𝘁𝗮𝗯𝗶𝗹𝗶𝘁𝘆:
    Because it’s less sensitive to minor outliers and noise, JS Distance tends to reduce false alarms compared to more sensitive metrics, e.g. KL Divergence.

    𝗜𝗱𝗲𝗮𝗹 𝗳𝗼𝗿 𝗗𝗶𝘀𝗰𝗿𝗲𝘁𝗲 𝗮𝗻𝗱 𝗖𝗼𝗻𝘁𝗶𝗻𝘂𝗼𝘂𝘀 𝗗𝗮𝘁𝗮:
    It can be used for different data types, making it suitable for a variety of ML applications.

    𝗦𝗰𝗮𝗹𝗲𝘀 𝗘𝗮𝘀𝗶𝗹𝘆 𝗔𝗰𝗿𝗼𝘀𝘀 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀:
    Since it’s capped at 1, JS Distance is easy to compare across features and models, supporting more consistent monitoring.


    🔴 𝗗𝗶𝘀𝗮𝗱𝘃𝗮𝗻𝘁𝗮𝗴𝗲𝘀 𝗼𝗳 𝗝𝗦 𝗗𝗶𝘀𝘁𝗮𝗻𝗰𝗲:
    𝗖𝗼𝗺𝗽𝘂𝘁𝗮𝘁𝗶𝗼𝗻𝗮𝗹𝗹𝘆 𝗘𝘅𝗽𝗲𝗻𝘀𝗶𝘃𝗲:
    Calculating JS Distance over many features or high-dimensional data can be resource-intensive, which could impact real-time monitoring efficiency.

    𝗗𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝗰𝗲 𝗼𝗻 𝗦𝗮𝗺𝗽𝗹𝗲 𝗦𝗶𝘇𝗲:
    JS Distance can be sensitive to the sample size.

    With small sample sizes, minor fluctuations can artificially increase or decrease the measured distance, potentially leading to misleading results.    
    """
    # Assume df_reference and df_current are historical and live DataFrames
    split_ratio = 0.6
    # === SPLIT DATAFRAME ===
    split_index = int(len(df) * split_ratio)
    df_train_test = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    # === DRIFT CALCULATION ===
    js_drift = {}

    for col in df.columns:
        try:
            # Get normalized value counts (probability distribution)
            p = df_train_test[col].value_counts(normalize=True, dropna=False).sort_index()
            q = df_val[col].value_counts(normalize=True, dropna=False).sort_index()

            # Align index (include all possible values in both sets)
            p, q = p.align(q, fill_value=0)

            # Compute JSD and store
            js_drift[col] = jensenshannon(p, q)
            
        except Exception as e:
            print(f"⚠️ Skipped column '{col}' due to error: {e}")
            continue

    # === CONVERT TO DATAFRAME ===
    drift_df = pd.DataFrame.from_dict(js_drift, orient='index', columns=['JSDistance'])
    drift_df.sort_values(by='JSDistance', inplace=True)

    drift_df.to_csv(os.path.join(Jensen_Shannon_distance_dir, 'Data drift detection through JS.csv'))

    #Kolmogorov–Smirnov (KS) test for data drift detection feature-wise between training and validation datasets.
    # === KS TEST CALCULATION ===
    Kolmogorov_Smirnov_test_dir = "..//Results//Data drift matrices//Kolmogorov_Smirnov_test"
    os.makedirs(Kolmogorov_Smirnov_test_dir,exist_ok=True)

    ks_results = {}

    for col in df.columns:
        try:
            # Drop NaNs (KS test doesn't handle them)
            train_values = df_train_test[col].dropna()
            val_values = df_val[col].dropna()

            # Skip if not enough data
            if len(train_values) < 10 or len(val_values) < 10:
                print(f"Skipped '{col}' — not enough samples.")
                continue

            # Apply KS test
            ks_stat, p_value = ks_2samp(train_values, val_values)

            ks_results[col] = {
                'KS_Statistic': ks_stat,
                'P_Value': p_value
            }

        except Exception as e:
            print(f"Skipped column '{col}' due to error: {e}")
            continue
        
    ks_df = pd.DataFrame.from_dict(ks_results, orient='index')
    ks_df.sort_values(by='KS_Statistic', ascending=False, inplace=True)

    ks_df.to_csv(os.path.join(Kolmogorov_Smirnov_test_dir, 'Data drift detection through KS test.csv'))

    """
    KS_Statistic → max distance between the CDFs of train and val
    P_Value → if p < 0.05, the distributions are likely statistically different

    KS Value	Interpretation
    ~0.0	Distributions are very similar
    > 0.2	Potential drift
    > 0.3	Likely significant distribution shift
    """
    # Wasserstein Distance (also known as Earth Mover's Distance) for data drift detection feature-wise between training and validation datasets.
    Wasserstein_Distance_dir = "..//Results//Data drift matrices//Wasserstein_Distance"
    os.makedirs(Wasserstein_Distance_dir,exist_ok=True)
    # === WASSERSTEIN DISTANCE CALCULATION ===
    wd_results = {}

    for col in df.columns:
        try:
            # Drop NaNs
            train_values = df_train_test[col].dropna()
            val_values = df_val[col].dropna()

            # Ensure numeric data
            if pd.api.types.is_numeric_dtype(train_values) and pd.api.types.is_numeric_dtype(val_values):
                if len(train_values) >= 10 and len(val_values) >= 10:
                    wd = wasserstein_distance(train_values, val_values)
                    wd_results[col] = wd
                else:
                    print(f"Skipped '{col}' — not enough samples.")
            else:
                print(f"Skipped '{col}' — non-numeric data.")

        except Exception as e:
            print(f"Error in column '{col}': {e}")
            continue

    # === CONVERT TO DATAFRAME ===
    wd_df = pd.DataFrame.from_dict(wd_results, orient='index', columns=['WassersteinDistance'])
    wd_df.sort_values(by='WassersteinDistance', ascending=False, inplace=True)

    wd_df.to_csv(os.path.join(Wasserstein_Distance_dir, 'Data drift detection through Wasserstein_Distance.csv'))

    """
    The Wasserstein Distance tells you:

    “On average, how much would I have to move a unit of probability mass to transform one distribution into another?”

    Think of it as how different two distributions are, in terms of both:

    Location (mean/center shift),

    Shape (spread, skew, etc.)

    Interpretation by Value:
    Wasserstein Distance	             Interpretation
    0	                                 Perfect match between distributions
    Small (e.g., < 0.1)	                 Very similar distributions
    Moderate (e.g., 0.1–1)	             Some difference — could indicate minor drift
    Large (e.g., > 1)	                 Strong difference — likely data drift or change in distribution
    """

    df_data_drift_merge = pd.concat([df_diff_train_val_MAE_sorted,drift_df,ks_df,wd_df],axis=1)
    df_data_drift_merge.drop(columns="P_Value",inplace=True)
    df_data_drift_merge= df_data_drift_merge[df_data_drift_merge["KS_Statistic"]>=0.3]
    df_data_drift_merge= df_data_drift_merge[df_data_drift_merge["WassersteinDistance"]>=0.1]
    
    df_data_drift_merge.to_csv(os.path.join('..//Results//Data drift matrices', 'Final_data_drift_varibales_table.csv'), index=True)
    
    data_drift_features = list(df_data_drift_merge.index)

    return data_drift_features







