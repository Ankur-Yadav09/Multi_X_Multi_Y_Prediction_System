# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:55:23 2026

@author: 30793167
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
from lazypredict.Supervised import LazyRegressor
import os
from Plot_func_file import create_histogram_density_plot,data_division_in_bins_with_same_amnt_data_plots,Parameters_line_chart, Process_parametrs_boxplot
from plotly.offline import plot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from ML_Function_file import random_forest, Random_searchCV_rf,linear_regression,Ridge_regression,Lasso_regression,remove_outliers_by_boxplot,Grid_searchCV_rf, Xgboost  
import pickle
import openpyxl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% random genration fixation
# For reproducibility.
 
file_path = "..\\Data"
 
# def seed_everything(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     print(f"Seeds set to {seed} for reproducibility.") 
# seed_everything(42)
 

df = pd.read_excel(file_path + "/" + 'DMC_Screen_tags_data.xlsx',sheet_name= "Cleaned data")
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df.set_index("Timestamp",inplace =True)

# df = df.apply(pd.to_numeric)

# df_rolling = df.rolling(window=6).mean()
# df_rolling.dropna(inplace=True)
# df= df_rolling

#%% remove the outleir
# lower_percentile = 0.95
# upper_percentile = 0.05

# for col in df.select_dtypes(include=['float', 'int']).columns:
#     low = df[col].quantile(lower_percentile)
#     high = df[col].quantile(upper_percentile)
#     df[col] = df[col].clip(lower=low, upper=high)

# threshold = 0.5  # drop rows with more than 50% missing
# df.dropna(thresh=int(df.shape[1] * threshold), inplace=True)      

#%% Correlation matrix
Result_folder_path = "..\\Results"
pearson_corr = df.corr(method="pearson").round(2)
spearman_corr = df.corr(method="spearman").round(2)
# Extract the lower triangular part of the correlation matrices
lower_triangular_pearson = np.tril(pearson_corr, k=-1)
lower_triangular_spearman = np.tril(spearman_corr, k=-1)
# Create DataFrames from the lower triangular parts

lower_triangular_df_pearson = pd.DataFrame(lower_triangular_pearson, columns=pearson_corr.columns, index=pearson_corr.index)
lower_triangular_df_spearman = pd.DataFrame(lower_triangular_spearman, columns=spearman_corr.columns, index=spearman_corr.index)
#Define a function to apply color based on column values

def color_columns(val):
    if val > 0.5:
        color = 'background-color: green'
    elif val < -0.5:
        color = 'background-color: red'
    else:
        color = ''
    return color

with pd.ExcelWriter(Result_folder_path +'/'+'correlation_with_color_conditioning_MA1.xlsx', engine='openpyxl') as writer:
    # Write the DataFrame to the Excel file with styling
    # lower_triangular_df_spearman.style.applymap(color_columns).to_excel(writer, index=True)
    pearson_corr.style.applymap(color_columns).to_excel(writer, index=True)

#%% Compute VIF to Detect and Remove Multicollinear Features
target_cols = ["Ethylene product flow"]
Y = df[target_cols]
X = df.drop(columns=target_cols)
X_const = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)

"""
VIF = 1: No multicollinearity
1 < VIF ≤ 5: Moderate correlation
VIF > 5: High multicollinearity
VIF > 10: Severe multicollinearity (remove or combine feature)
"""
# Iteratively drop highest VIF feature until all < threshold
def remove_high_vif(X, threshold=10):
    X = X.copy()
    while True:
        X_const = sm.add_constant(X)
        vif = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])]
        max_vif = max(vif)
        if max_vif < threshold:
            break
        max_vif_idx = vif.index(max_vif)
        col_to_drop = X.columns[max_vif_idx]
        X = X.drop(columns=col_to_drop)
        print(f"Dropped {col_to_drop} (VIF = {max_vif:.2f})")
    return X

df_copy = remove_high_vif(X, threshold=10)

df = pd.concat([df_copy,Y],axis =1)

X = df.drop(columns=target_cols)

#%% correlation matrix

Result_folder_path = "..\\Results"
pearson_corr = df.corr(method="pearson").round(2)
spearman_corr = df.corr(method="spearman").round(2)
# Extract the lower triangular part of the correlation matrices
lower_triangular_pearson = np.tril(pearson_corr, k=-1)
lower_triangular_spearman = np.tril(spearman_corr, k=-1)
# Create DataFrames from the lower triangular parts

lower_triangular_df_pearson = pd.DataFrame(lower_triangular_pearson, columns=pearson_corr.columns, index=pearson_corr.index)
lower_triangular_df_spearman = pd.DataFrame(lower_triangular_spearman, columns=spearman_corr.columns, index=spearman_corr.index)
# Define a function to apply color based on column values

def color_columns(val):
    if val > 0.4:
        color = 'background-color: green'
    elif val < -0.4:
        color = 'background-color: red'
    else:
        color = ''
    return color

with pd.ExcelWriter(Result_folder_path +'/'+'correlation_with_color_conditioning.xlsx', engine='openpyxl') as writer:
    # Write the DataFrame to the Excel file with styling
    lower_triangular_df_spearman.style.applymap(color_columns).to_excel(writer, index=True)

X = df.drop(columns=target_cols)
Y = df[target_cols]
 
#%% Histogram with density plot
#%% histogram along with density plot
Histogram_density_folder_path = Result_folder_path +'/'+"Histogram and density plot"
os.makedirs(Histogram_density_folder_path, exist_ok=True)
figure = {}
for i in df.columns:
    data = df[i]
    filename = i+"_"+"hist_density_plot.html"
    title = i + " "+"histogram and density plot"
    x_title = i
    fig = create_histogram_density_plot(data,Histogram_density_folder_path+'/'+filename, title, x_title)
    figure[i] = fig

#%% Box plot of output process parameters with respect to input process parameters

Box_plot_folder_path = Result_folder_path +'/'+"Box plot of process parameters"
os.makedirs(Box_plot_folder_path, exist_ok=True)
bin_divided_by = df.columns.values
target_tag='Ethylene_Product_Flow_Compensated'
number_of_bins = 5
figures= {}    
for bin_name in bin_divided_by:
    fig = data_division_in_bins_with_same_amnt_data_plots(df, bin_name, target_tag, number_of_bins)
    figures["bin_name"] = fig
    plot(figures["bin_name"], filename=Box_plot_folder_path + "\Box plot of Ethylene_Product_Flow_Compensated with " + bin_name +".html", auto_open=False)
    #fig.write_html(html_file_path, auto_open=True)

#%% Process parameter line chart

Line_chart_dir = r"..\\Results\\Parameters line chart"
Parameters_line_chart(df=df, Line_chart_dir=Line_chart_dir, width_px=1200, height_px=800)
 
#%% Process parameters box plot

df_copy = df.copy()
df_copy.reset_index(inplace=True)
df_copy['Timestamp'] = pd.to_datetime(df_copy['Timestamp'])  # Replace 'date' with your actual datetime column
df_copy['month_year'] = df_copy['Timestamp'].dt.strftime('%b-%Y')  # e.g., Jan-2024
Boxplot_output_dir = "..\\Results\\Monthwise Process param box plot"
os.makedirs(Boxplot_output_dir, exist_ok=True)
Process_parametrs_boxplot(df_copy, Boxplot_output_dir)
 
#%% model building activity

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#%% DAE algorithm




#%% AutoML implementation for best model selection

# all_results = []
# for target in target_cols:
#     print(f"\n🔹 Running LazyPredict for {target}")
#     reg = LazyRegressor(verbose=0, ignore_warnings=True)
#     models, _ = reg.fit(
#         x_train, x_test,
#         y_train[target],
#         y_test[target]
#     )
#     # Add target column
#     models['Target'] = target
#     # Reset index so model names become a column
#     models = models.reset_index().rename(columns={'index': 'Model'})
#     all_results.append(models) 
# # Combine all results

# final_df = pd.concat(all_results, axis=0).reset_index(drop=True)
 
# In[59]:

# final_df[final_df['Target'] == 'Average_Furnace_Conversion'].sort_values(by='R-Squared', ascending=False)
# final_df[final_df['Target'] == 'Overall_Plant_Ultimate_Yield'].sort_values(by='R-Squared', ascending=False)
# final_df[final_df['Target'] == 'Ethylene_Product_Flow_Compensated'].sort_values(by='R-Squared', ascending=False)
 
#%% Sample splitting and train test data preparation

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)   # fit on training data
x_test_scaled = scaler.transform(x_test)         # transform test data
x_train_scaled = pd.DataFrame(x_train_scaled, columns=X.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=X.columns)
 
#%%Linear regression model

LR_model_result, Lr    = linear_regression(x_train_scaled,y_train,x_test_scaled,y_test)
LR_result_folder_path = Result_folder_path + "/"+"Linear regression model"
os.makedirs(LR_result_folder_path,exist_ok=True)
with open(LR_result_folder_path + "/Linear regression parameters & accuracy matrix.txt", "w") as file:

    file.write("Linear Regression Results\n")
    file.write(f"columns: {list(cols)}\n")
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
plt.xlabel('Actual #%% Hyper-parameter selection for random forest and model prediction')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (train set)')
plt.savefig(LR_result_folder_path +'/'+"LR train set prediction.jpeg", dpi=1080)
plt.close()

LR_ytest_hat = Lr.predict(x_test_scaled)
plt.scatter(y_test, LR_ytest_hat)
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (test set)')
plt.savefig(LR_result_folder_path +'/'+"LR test set prediction.jpeg", dpi=1080)
plt.close()
with open (LR_result_folder_path+"/"+"Linear_regression.pkl","wb") as file:

    pickle.dump(Lr,file)
  
#%% Lasso regression model -- second order penalty

Lasso_model_result, Lasso = Lasso_regression(x_train_scaled,y_train,x_test_scaled,y_test)
Lasso_result_folder_path = Result_folder_path + "/"+"Lasso regression model"
os.makedirs(Lasso_result_folder_path,exist_ok=True)
with open(Lasso_result_folder_path + "/Lasso regression parameters & accuracy matrix.txt", "w") as file:
    file.write("Lasso Regression Results\n")
    file.write(f"columns: {list(cols)}\n")
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
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (train set)')
plt.savefig(Lasso_result_folder_path +'/'+"Lasso train set prediction.jpeg", dpi=1080)
plt.close()
 
Lasso_ytest_hat = Lasso.predict(x_test_scaled)

plt.scatter(y_test, Lasso_ytest_hat)
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (test set)')
plt.savefig(Lasso_result_folder_path +'/'+"Lasso test set prediction.jpeg", dpi=1080)
plt.close()
 
with open (Lasso_result_folder_path+"/"+"Lasso_regression.pkl","wb") as file:

    pickle.dump(Lasso,file)
 
 
Feature_extraction_through_Lasso = Lasso_model_result["Lasso_CV_reg coef"]
Feature_extraction_through_Lasso["coefficents"] = np.abs(Feature_extraction_through_Lasso["coefficents"])
Feature_extraction_through_Lasso = Feature_extraction_through_Lasso[Feature_extraction_through_Lasso["coefficents"]!=0]
# df= df[list(Feature_extraction_through_Lasso.index) +target_cols]
 
#%% Hyper-parameter selection for random forest and model prediction
 
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
    file.write(f"columns: {list(cols)}\n")
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
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (train set)')
plt.savefig(RF_result_folder_path +'/'+"RF train set prediction.jpeg", dpi=1080)
plt.close()

rf_ytest_hat = rf.predict(x_test)

plt.scatter(y_test, rf_ytest_hat)    
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (test set)')
plt.savefig(RF_result_folder_path +'/'+"RF test set prediction.jpeg", dpi=1080)
plt.close()
 
# Create a partial dependency plot
from sklearn.inspection import PartialDependenceDisplay
features = cols
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
for feature in cols:
    pdp = partial_dependence(rf, X=x_test, features=[feature], grid_resolution=100)
    pd_values = pdp['average'][0]
    # Measure how much the PDP changes across feature values
    pd_range = np.ptp(pd_values)  # Peak-to-peak range
    pd_std = np.std(pd_values)    # Standard deviation 
    pdp_variations.append((feature, pd_range, pd_std))
 
#%% Xgboost model implementation

n_estimators_Xgboost = 150
max_depth_Xgboost = 4
eta=0.1
Xgboost_result, Xgboost_model = Xgboost(x_train,y_train,x_test,y_test,n_estimators_Xgboost,max_depth_Xgboost,eta)
XGB_result_folder_path = Result_folder_path + "/"+"XgBoost model"
os.makedirs(XGB_result_folder_path,exist_ok=True)

with open(XGB_result_folder_path+"/"+"XgBoost parameters & accuracy matrix.txt", "w") as file:
    file.write("XgBoost Results\n")
    file.write(f"columns: {list(cols)}\n")
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
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (train set)')
plt.savefig(XGB_result_folder_path +'/'+"XgBoost train set prediction.jpeg", dpi=1080)
plt.close()
 
XgBoost_ytest_hat = Xgboost_model.predict(x_test)

plt.scatter(y_test,XgBoost_ytest_hat)
plt.xlabel('Actual output')
plt.ylabel('Predicted output')
plt.title('Actual output vs Predicted output (test set)')
plt.savefig(XGB_result_folder_path +'/'+"XgBoost test set prediction.jpeg", dpi=1080)
plt.close()
 
with open (XGB_result_folder_path +"/"+"XgBoost_regression.pkl","wb") as file:
    pickle.dump(Xgboost_model,file)

 
#%% Grid search implementation
 
with open (Lasso_result_folder_path+"/"+"Lasso_regression.pkl","rb") as file:
    Laaso_model = pickle.load(file)
 
 # Step 1: Select base row
base = df[cols].mean().copy()  
# base = df[cols].iloc[-1].copy()
 # Step 2: Define grid range

cot_range = np.linspace(df['Overall_COT'].min(),
                        df['Overall_COT'].max(), 20)
 
feed_range = np.linspace(
    df['Ethane_Feed_Preheater_Ethane_Feed_Flow_DMCTF'].min(),
    df['Ethane_Feed_Preheater_Ethane_Feed_Flow_DMCTF'].max(),
    20
)
 
# feed_range = np.linspace(
#     df['C2_Splitter_DP'].min(),
#     df['C2_Splitter_DP'].max(),
#     20
# )
 
 
# Step 3: Grid search
results = []
for cot in cot_range:
    for feed in feed_range:
        temp = base.copy()
        temp['Overall_COT'] = cot
        temp['Ethane_Feed_Preheater_Ethane_Feed_Flow_DMCTF'] = feed
        # temp['C2_Splitter_DP'] = feed
        X = pd.DataFrame([temp])[cols]
        X_scaled = scaler.transform(X)
        pred = Laaso_model.predict(X_scaled)[0]
        row = temp.copy()
        row['Predicted_Ethylene'] = pred
        results.append(row)
 
# Convert to DataFrame

res_df = pd.DataFrame(results)
GridSearch_result_folder_path = Result_folder_path + "/"+"GridSearch optimization"
os.makedirs(GridSearch_result_folder_path,exist_ok=True)
res_df.to_excel(GridSearch_result_folder_path +"/"+ "GridSearch results.xlsx")
 
 
# Step 4: Find optimum

best = res_df.loc[res_df['Predicted_Ethylene'].idxmax()]
print("✅ Optimal Condition:")
print(best)
 
pivot = res_df.pivot(index='Overall_COT',
                     columns='Ethane_Feed_Preheater_Ethane_Feed_Flow_DMCTF',
                     # columns='C2_Splitter_DP',
                     values='Predicted_Ethylene')
plt.figure()
im = plt.imshow(pivot.values, aspect='auto')
plt.colorbar(im, label='Ethylene Production')
 
# ✅ Set real axis values

plt.xticks(
    ticks=np.arange(len(pivot.columns)),
    labels=np.round(pivot.columns, 0),
    rotation=90
)
 
plt.yticks(
    ticks=np.arange(len(pivot.index)),
    labels=np.round(pivot.index, 1)
)
 
# plt.xlabel('Feed Flow')

plt.xlabel('DMCTF')
plt.ylabel('Overall COT')
plt.title('Grid Search Optimization')
plt.tight_layout()
plt.savefig(GridSearch_result_folder_path +"/"+"optimization_heatmap.jpeg", dpi=300)
plt.show()

 