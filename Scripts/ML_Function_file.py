# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:38:16 2022

@author: sumit
"""
# %%

#Custom plo# numerical
import numpy as np
from scipy import stats

# %%
# random search CV for random forest hyper tunning
def Random_searchCV_rf(x,y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    import pandas as pd
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
    min_samples_split = [2, 5,7,10,12]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10,
                                   cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(x,y)
    return rf_random.best_params_

# %%
# grid search CV for random forest hyper tunning
def Grid_searchCV_rf(x,y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
    min_samples_split = [2, 5,7,10,12,14]
    min_samples_leaf = [1, 2, 4,6,8]
    bootstrap = [True, False]
    # Create the Grid search
    param_grid= {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    rf_Grid = GridSearchCV(estimator = rf, param_grid = param_grid,
                                   cv = 3, verbose=2, n_jobs = -1)
    rf_Grid.fit(x,y)
    return rf_Grid.best_params_


# %%    
# random forest regression
def random_forest(x_train,y_train,x_test,y_test,n_estimators,max_depth,criterion,
                  min_samples_split,min_samples_leaf,n_jobs,random_state,max_features):
    from sklearn.ensemble import RandomForestRegressor
    rf=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,
                             max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                             n_jobs=n_jobs,random_state=random_state)
    rf_fit=rf.fit(x_train,y_train)
    rf_pred_train=rf_fit.predict(x_train)
    rf_pred_test=rf_fit.predict(x_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
    accuracy_MAE_train=mean_absolute_error(y_true=y_train, y_pred=rf_pred_train)
    accuracy_RMSE_train=np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_pred_train))
    accuracy_MAPE_train=mean_absolute_percentage_error(y_true=y_train, y_pred=rf_pred_train)
    r2_score_train=r2_score(y_true=y_train, y_pred=rf_pred_train)
    
    accuracy_MAE=mean_absolute_error(y_true=y_test, y_pred=rf_pred_test)
    accuracy_RMSE=np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_pred_test))
    accuracy_MAPE=mean_absolute_percentage_error(y_true=y_test, y_pred=rf_pred_test)
    r2_score_test=r2_score(y_true=y_test, y_pred=rf_pred_test)
    random_forest_result={"MAE train set ":accuracy_MAE_train,
                          "RMSE train set ":accuracy_RMSE_train,
                          "MAPE train set ":accuracy_MAPE_train,
                          "R-square of train set ":r2_score_train, 
                           "MAE test ":accuracy_MAE,
                           "RMSE test ":accuracy_RMSE,
                           "MAPE test ":accuracy_MAPE,
                          "R-square of test set ":r2_score_test,
                          'random_forest_param':rf.get_params()}
    return random_forest_result, rf
#%%
# Xgboost regression
def Xgboost(x_train,y_train,x_test,y_test,n_estimators_Xgboost,max_depth_Xgboost,eta):
    import xgboost as xgb
    Xgboost = xgb.XGBRegressor(n_estimators=n_estimators_Xgboost, 
                   max_depth=max_depth_Xgboost,
                   objective ="reg:squarederror",
                   eta=eta, 
                   n_jobs=-1)
    Xgboost_fit=Xgboost.fit(x_train,y_train)
    Xgboost_pred_train=Xgboost_fit.predict(x_train)
    Xgboost_pred_test=Xgboost_fit.predict(x_test) 
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
    accuracy_MAE_train=mean_absolute_error(y_true=y_train, y_pred=Xgboost_pred_train)
    accuracy_RMSE_train=np.sqrt(mean_squared_error(y_true=y_train, y_pred=Xgboost_pred_train))
    accuracy_MAPE_train=mean_absolute_percentage_error(y_true=y_train, y_pred=Xgboost_pred_train)
    r2_score_train=r2_score(y_true=y_train, y_pred=Xgboost_pred_train)
    
    accuracy_MAE=mean_absolute_error(y_true=y_test, y_pred=Xgboost_pred_test)
    accuracy_RMSE=mean_squared_error(y_true=y_test, y_pred=Xgboost_pred_test)
    accuracy_MAPE=mean_absolute_percentage_error(y_true=y_test, y_pred=Xgboost_pred_test)
    r2_score_train=r2_score(y_true=y_train, y_pred=Xgboost_pred_train)
    r2_score_test=r2_score(y_true=y_test, y_pred=Xgboost_pred_test)
    Xgboost_result={"MAE train set ":accuracy_MAE_train,
                          "RMSE train set ":accuracy_RMSE_train,
                          "MAPE train set ":accuracy_MAPE_train,
                          "R-square of train set ":r2_score_train, 
                           "MAE test ":accuracy_MAE,
                           "RMSE test ":accuracy_RMSE,
                           "MAPE test ":accuracy_MAPE,
                          "R-square of test set ":r2_score_test,
                          'XgBoost_param':Xgboost.get_params()}
    return Xgboost_result, Xgboost


# %%    
#linear regression
def linear_regression(x_train,y_train,x_test,y_test):
    from sklearn.linear_model import LinearRegression
    LR=LinearRegression()
    LR_fit=LR.fit(x_train,y_train)
    LR_pred_train=LR_fit.predict(x_train)
    LR_pred_test=LR_fit.predict(x_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
    accuracy_MAE_train=mean_absolute_error(y_true=y_train, y_pred=LR_pred_train)
    accuracy_RMSE_train=np.sqrt(mean_squared_error(y_true=y_train, y_pred=LR_pred_train))
    accuracy_MAPE_train=mean_absolute_percentage_error(y_true=y_train, y_pred=LR_pred_train)
    r2_score_train=r2_score(y_true=y_train, y_pred=LR_pred_train)
    
    accuracy_MAE=mean_absolute_error(y_true=y_test, y_pred=LR_pred_test)
    accuracy_RMSE=np.sqrt(mean_squared_error(y_true=y_test, y_pred=LR_pred_test))
    accuracy_MAPE=mean_absolute_percentage_error(y_true=y_test, y_pred=LR_pred_test)
    r2_score_test=r2_score(y_true=y_test, y_pred=LR_pred_test)
    Linear_result={"MAE train set ":accuracy_MAE_train,
                    "RMSE train set ":accuracy_RMSE_train,
                    "MAPE train set ":accuracy_MAPE_train,
                    "R-square of train set ":r2_score_train, 
                     "MAE test ":accuracy_MAE,
                     "RMSE test ":accuracy_RMSE,
                     "MAPE test ":accuracy_MAPE,
                    "R-square of test set ":r2_score_test,
                  "intercept":LR.intercept_,
                  "linear reg coef":LR.coef_}
    return Linear_result, LR
    
 # %% 
#logistic regression   
def logistic_regression(x_train,y_train,x_test,y_test):
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression()
    log_fit=log.fit(x_train,y_train)
    log_pred_test=log_fit.predict(x_test)
    from sklearn.metrics import classification_report, confusion_matrix_score,accuracy_score
    classification_report=classification_report(y_true=y_test, y_pred=log_pred_test)
    accuracy_log=accuracy_score(y_true=y_test, y_pred=log_pred_test)
    confusion_log=confusion_matrix_score(y_true=y_test, y_pred=log_pred_test)
    return {"accuracyr":accuracy_log,
            "confusion matrix":confusion_log,
            "classification_report":classification_report,
                  "intercept":log.intercept_,
                  "log reg coef":log.coef_,
                  'log_prob':log.predict_proba(x_test)}
        
#%%    
# varinace inflation factor
def VIF_test(data,col_list,Output_col):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd
    Input_columns=list()
    for col_name in col_list:
        if ((data[col_name].dtypes!="object") & (col_name!=Output_col)):
            Input_columns.append(col_name)
    Input_columns        
    x=data[Input_columns]
    vif_data=pd.DataFrame()
    vif_data["features"]=x.columns
    vif_data["VIF"]=[variance_inflation_factor(exog=x.values, exog_idx=i) for i in range(len(x.columns))]
    vif_data
    return vif_data    
        
#%%
def standard_scalar(x):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    std=StandardScaler()
    x_std=std.fit_transform(x)
    x_std=pd.DataFrame(x_std,columns=x.columns)
    return x_std, std    
    
#%%
#Ridge regression 
def Ridge_regression(x_train,y_train,x_test,y_test):  
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold
    import pandas as pd
    import numpy as np
    cv=RepeatedKFold(n_splits=10, n_repeats=3,random_state=1)
    Ridge_CV=RidgeCV(cv=cv,alphas=np.arange(0.01, 1, 0.01))
    Ridge_CV_fit=Ridge_CV.fit(x_train,y_train)
    Ridge_CV_pred_train=Ridge_CV_fit.predict(x_train)
    Ridge_CV_pred_test=Ridge_CV_fit.predict(x_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
    accuracy_MAE=mean_absolute_error(y_true=y_test, y_pred=Ridge_CV_pred_test)
    accuracy_RMSE=mean_squared_error(y_true=y_test, y_pred=Ridge_CV_pred_test)
    accuracy_MAPE=mean_absolute_percentage_error(y_true=y_test, y_pred=Ridge_CV_pred_test)
    r2_score_train=r2_score(y_true=y_train, y_pred=Ridge_CV_pred_train)
    r2_score_test=r2_score(y_true=y_test, y_pred=Ridge_CV_pred_test)
    Ridge_result={"mean absolute error":accuracy_MAE,
            "Root mean square error":accuracy_RMSE,
            "Mean absolute percentage error":accuracy_MAPE,
                  "R-squared of train set":r2_score_train,
                  "R-squared of test set":r2_score_test,
                  "intercept":Ridge_CV.intercept_,
                  "alpha":Ridge_CV.alpha_,
                  "Ridge_CV_reg coef":pd.DataFrame(np.transpose(Ridge_CV.coef_),index=x_train.columns,columns=["coefficents"])}
    return Ridge_result, Ridge_CV   
   
   
#%%
#Lasso regression 
def Lasso_regression(x_train,y_train,x_test,y_test):  
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import RepeatedKFold
    import pandas as pd
    import numpy as np
    Lasso_CV=LassoCV(cv=10,eps=0.001,n_alphas=100,fit_intercept=True,random_state=2,n_jobs=-1,tol=0.00001,max_iter=20000)
    Lasso_CV_fit=Lasso_CV.fit(x_train,y_train)
    Lasso_CV_pred_train=Lasso_CV_fit.predict(x_train)
    Lasso_CV_pred_test=Lasso_CV_fit.predict(x_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
    accuracy_MAE_train=mean_absolute_error(y_true=y_train, y_pred=Lasso_CV_pred_train)
    accuracy_RMSE_train=np.sqrt(mean_squared_error(y_true=y_train, y_pred=Lasso_CV_pred_train))
    accuracy_MAPE_train=mean_absolute_percentage_error(y_true=y_train, y_pred=Lasso_CV_pred_train)
    r2_score_train=r2_score(y_true=y_train, y_pred=Lasso_CV_pred_train)
    accuracy_MAE=mean_absolute_error(y_true=y_test, y_pred=Lasso_CV_pred_test)
    accuracy_RMSE=mean_squared_error(y_true=y_test, y_pred=Lasso_CV_pred_test)
    accuracy_MAPE=mean_absolute_percentage_error(y_true=y_test, y_pred=Lasso_CV_pred_test)
    r2_score_train=r2_score(y_true=y_train, y_pred=Lasso_CV_pred_train)
    r2_score_test=r2_score(y_true=y_test, y_pred=Lasso_CV_pred_test)
    Lasso_result={"MAE train set ":accuracy_MAE_train,
                    "RMSE train set ":accuracy_RMSE_train,
                    "MAPE train set ":accuracy_MAPE_train,
                    "R-square of train set ":r2_score_train, 
                     "MAE test ":accuracy_MAE,
                     "RMSE test ":accuracy_RMSE,
                     "MAPE test ":accuracy_MAPE,
                    "R-square of test set ":r2_score_test,
                  "intercept":Lasso_CV.intercept_,
                  "alpha":Lasso_CV.alpha_,
                  "Lasso_CV_reg coef":pd.DataFrame(np.transpose(Lasso_CV.coef_),index=x_train.columns,columns=["coefficents"])}
    return Lasso_result, Lasso_CV 

#%%
#Outlier removal through box plot
def remove_outliers_by_boxplot (df,col_list):
    for col_name in col_list:
        Q1=df[col_name].quantile(0.25)
        Q3=df[col_name].quantile(0.75)
        IQR=Q3-Q1
        df=df[(df[col_name]>=Q1-1.5*IQR) & (df[col_name]<=Q3+1.5*IQR)]
    return df




