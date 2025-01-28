import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau
import random
from scipy.stats import kendalltau, spearmanr
import time 
import pandas as pd  
import pickle
import openpyxl
from openpyxl import load_workbook
from pathlib import Path
import matplotlib.pyplot as plt 
from tabular_datasets import * 
from HSICNet.HSICFeatureNet import *
from HSICNet.HSICNet import *
from HSICNet.util import *

from explainers.L2x_reg import *
from invase import INVASE
import shap 
from explainers.bishapley_kernel import Bivariate_KernelExplainer
from shapreg import removal, games, shapley
from explainers.MAPLE import MAPLE
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer


import warnings
warnings.filterwarnings("ignore")



    
def model_performance_metrics(y_test, y_pred):
    
        # Calculate MAE
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.2f}")

        # Calculate RMSE
        rmse = np.sqrt(mse)  # Or directly use mean_squared_error with squared=False
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Calculate R-squared
        r2 = r2_score(y_test, y_pred)
        print(f"R-squared: {r2:.2f}")
        metrics = [mae, mse, rmse, r2]

        return metrics
       
def call_LimeToExplain(lime_explainer , X_tbx, exp_func, d):
    lime_values = []
    for x in X_tbx:
        explanation = lime_explainer.explain_instance(data_row = x, predict_fn = exp_func, num_features = d)
        exp = [0] * d
        for feature_idx, contribution in explanation.local_exp[0]:
            exp[feature_idx] = contribution
        lime_values.append(exp)
    
    return lime_values




def call_UnbiasedShap(imputer, X_tbx , X_sample_no):         
    ## Unbiased kernel shap 
    # imputer = removal.MarginalExtension(X_tbx, exp_func)
    ushap_values = np.empty_like(X_tbx)
   
    for i in range(X_tbx.shape[0]):
        x = X_tbx[i, ]      
        game = games.PredictionGame(imputer, x)
        values = shapley.ShapleyRegression(game, n_samples=X_sample_no, paired_sampling=False)
        ushap_values[i,:] = values.values.squeeze()

    return ushap_values

      
def call_Maple(maple_explainer, X_tbx ):
    maple_values = np.empty_like(X_tbx)
   
    for i in range(X_tbx.shape[0]):
        x = X_tbx[i, ]
        mpl_exp = maple_explainer.explain(x)
        maple_values[i,] = (mpl_exp['coefs'][1:]).squeeze()

    return maple_values



def call_HSIC_methods(method_name, X_tensor, y_tensor,  feature_imp):
    #HSIC_anova
   
    sigma_init_X = 0.5 *torch.ones(X_tensor.size(1))
    sigma_init_Y = 0.5 * torch.ones(1)
    # sigma_init_X = initialize_sigma_median_heuristic(X_tensor)
    # sigma_init_Y = initialize_sigma_y_median_heuristic(y_tensor)
    num_sampling = feature_imp

    

    #HSICNetGumbelSparsemax
    if method_name =='HSIC_GumbelSparsemax':
        model = HSICNetGumbelSparsemax(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_sampling)
    elif method_name== 'HSICFeatureNet_GumbelSparsemax':
        model = HSICFeatureNetGumbelSparsemax(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_sampling)

    elif method_name == 'HSIC_GumbelSoftmax':
        model = HSICNetGumbelSoftmax(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_sampling)
    elif method_name == 'HSICFeatureNet_GumbelSoftmax':
        model = HSICFeatureNetGumbelSoftmax(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_sampling)

    elif method_name == 'HSIC_Sparsemax':
        model = HSICNetSparsemax(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y)
    elif method_name == 'HSICFeatureNet_Sparsemax':
        model = HSICFeatureNetSparsemax(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y)


    model.train_model(X_tensor, y_tensor, num_epochs=epoch, BATCH_SIZE = BATCH_SIZE)
    sigmas = model.sigmas
    sigma_y = model.sigma_y
    weights = model(X_tensor)[0]
    

    return model , sigmas, sigma_y, weights

def compare_methods(model, d, exp_func, X_tensor_bg, y_tensor_bg, X_bg, y_bg, X_df_bg, y_series_bg, X_tensor_tbx,y_tensor_tbx, X_sample_no ):
      # instantiate all methods
        gumbelsparsemax_model, gsp_sigmas, gsp_sigma_y, gsp_weights = call_HSIC_methods('HSIC_GumbelSparsemax', X_tensor_bg, y_tensor_bg, feature_imp=1)
        gumbelsparsemax_model2, gsp_sigmas2, gsp_sigma_y2, gsp_weights2 = call_HSIC_methods('HSIC_GumbelSparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d)
        featureNet_gumbelsparsemax_model, featureNet_gsp_sigmas, featureNet_gsp_sigma_y, featureNet_gsp_weights = call_HSIC_methods('HSICFeatureNet_GumbelSparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d)

        
        gumbelsoftmax_model, gso_sigmas, gso_sigma_y, gso_weights = call_HSIC_methods('HSIC_GumbelSoftmax', X_tensor_bg, y_tensor_bg, feature_imp=d)
        featureNet_gumbelsoftmax_model, featureNet_gso_sigmas, featureNet_gso_sigma_y, featureNet_gso_weights = call_HSIC_methods('HSICFeatureNet_GumbelSoftmax', X_tensor_bg, y_tensor_bg, feature_imp=d)

        
        sparsemax_model, sp_sigmas, sp_sigma_y, sp_weights = call_HSIC_methods('HSIC_Sparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d) # d here does not have any effect
        featureNet_sparsemax_model, featureNet_sp_sigmas, featureNet_sp_sigma_y, featureNet_sp_weights = call_HSIC_methods('HSICFeatureNet_Sparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d) # d here does not have any effect



        L2X_explainer, _ = train_L2X(X_bg, y_bg, d, epochs= epoch , batch_size = BATCH_SIZE) 
        Invase_explainer = INVASE (model, X_df_bg, y_series_bg, n_epoch=epoch, prefit=False) #prefit = False to train the model
        
        # X_tbx = X[:200,:]
        X_bg_shap = shap.sample(X, 100)
        shap_explainer = shap.KernelExplainer(exp_func, X_bg_shap)
        # imputer_ushap = removal.MarginalExtension(X_bg, exp_func)  ## Unbiased kernel shap 
        bshap_explainer = Bivariate_KernelExplainer(exp_func, X_bg_shap)
        lime_explainer = LimeTabularExplainer(training_data = X_bg_shap, discretize_continuous=False, mode = 'regression') 
        # maple_explainer = MAPLE(X_bg, y_bg, X_bg, y_bg)

    ##----------------------------------------------------------------------------------------------------------   
        
        # Perform experiment
       
        gumbelsparsemax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_gsp_weights, _ ,_ = gumbelsparsemax_model(X_tensor_tbx)  
        # hsic_gsp_weights = l_gsp_weights.detach().cpu().numpy()
        gumbelsparsemax_selected_features = (hsic_gsp_weights > 1e-3).to(torch.int32)
        
        l_gsp_shap_values, _ = gumbelsparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , gsp_sigmas, gsp_sigma_y, gsp_weights)
        hsic_gsp_shap_values = l_gsp_shap_values.detach().cpu().numpy()
        ##---------------
        gumbelsparsemax_model2.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_gsp_weights2, _ ,_ = gumbelsparsemax_model2(X_tensor_tbx)  
        # hsic_gsp_weights2 = l_gsp_weights2.detach().cpu().numpy()
        gumbelsparsemax2_selected_features = (hsic_gsp_weights2 > 1e-3).to(torch.int32)

        l_gsp_shap_values2, _ = gumbelsparsemax_model2.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , gsp_sigmas2, gsp_sigma_y2, gsp_weights2)
        hsic_gsp_shap_values2 = l_gsp_shap_values2.detach().cpu().numpy()

        ##--------------------
        featureNet_gumbelsparsemax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_fNET_gsp_weights, _ ,_ = featureNet_gumbelsparsemax_model(X_tensor_tbx)  
        # hsic_fNET_gsp_weights = l_fNET_gsp_weights.detach().cpu().numpy()
        HISCFGSP_selected_features = (hsic_fNET_gsp_weights > 1e-3).to(torch.int32)

        l_fNET_gsp_shap_values, _ = featureNet_gumbelsparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , featureNet_gsp_sigmas, featureNet_gsp_sigma_y, featureNet_gsp_weights)
        hsic_fNET_gsp_shap_values = l_fNET_gsp_shap_values.detach().cpu().numpy()
        ##---------------------
        gumbelsoftmax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_gso_weights, _ ,_ = gumbelsoftmax_model(X_tensor_tbx)  
        # hsic_gso_weights = l_gso_weights.detach().cpu().numpy()
        gumbelsoftmax_selected_features = (hsic_gso_weights > 1e-3).to(torch.int32)

        l_gso_shap_values, _ = gumbelsoftmax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx,   gso_sigmas, gso_sigma_y, gso_weights)
        hsic_gso_shap_values = l_gso_shap_values.detach().cpu().numpy()
        ##------------------
        featureNet_gumbelsoftmax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_fNET_gso_weights, _ ,_ = featureNet_gumbelsparsemax_model(X_tensor_tbx)  
        # hsic_fNET_gso_weights = l_fNET_gso_weights.detach().cpu().numpy()
        HISCFGSO_selected_features = (hsic_fNET_gso_weights > 1e-3).to(torch.int32)

        l_fNET_gso_shap_values, _ = featureNet_gumbelsoftmax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx,   featureNet_gso_sigmas, featureNet_gso_sigma_y, featureNet_gso_weights)
        hsic_fNET_gso_shap_values = l_fNET_gso_shap_values.detach().cpu().numpy()
        ##------------------
        sparsemax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_sp_weights, _ ,_ = sparsemax_model(X_tensor_tbx)  
        # hsic_sp_weights = l_sp_weights.detach().cpu().numpy()
        sparsemax_selected_features = (hsic_sp_weights > 1e-3).to(torch.int32)

        l_sp_shap_values, _ = sparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , sp_sigmas, sp_sigma_y, sp_weights)
        hsic_sp_shap_values = l_sp_shap_values.detach().cpu().numpy()
        ##----------------------
        featureNet_sparsemax_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            hsic_fNET_sp_weights, _ ,_ = featureNet_sparsemax_model(X_tensor_tbx)  
        # hsic_fNET_sp_weights = l_fNET_sp_weights.detach().cpu().numpy()
        HISCFSP_selected_features = (hsic_fNET_sp_weights > 1e-3).to(torch.int32)

        l_fNET_sp_shap_values, _ = featureNet_sparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , featureNet_sp_sigmas, featureNet_sp_sigma_y, featureNet_sp_weights)
        hsic_fNET_sp_shap_values = l_fNET_sp_shap_values.detach().cpu().numpy()
        ##---------------------------
        
        L2X_explainer.eval()
        with torch.no_grad():
                    _, l_L2X_weights= L2X_explainer(X_tensor_tbx, training=False)
        L2X_weights= l_L2X_weights.cpu().numpy()
        L2X_selected_features = (L2X_weights > 1e-3).astype(int)
        
        invase_scores =(Invase_explainer.explain(X_df_tbx)).to_numpy()                      
        invase_selected_features = (invase_scores > 0.5).astype(int)
    
        shap_values = shap_explainer.shap_values(X_tbx, nsamples=X_sample_no, l1_reg=True)  
        shap_selected_featuers = (np.abs(shap_values) > 1e-3).astype(int) 
       
        # ushap_values = call_UnbiasedShap(imputer_ushap, X_tbx , num_samples)     
       
        bshap_values = bshap_explainer.shap_values(X_tbx, nsamples=X_sample_no, l1_reg=True)
        bishap_selected_featuers = (np.abs(bshap_values) > 1e-3).astype(int)              
        
        lime_values = call_LimeToExplain(lime_explainer, X_tbx, exp_func, d)
        lime_selected_featuers = (np.abs(lime_values) > 1e-3).astype(int)

        feature_importances = [hsic_gsp_weights, 
            hsic_gsp_weights2,
            hsic_fNET_gsp_weights,
            hsic_gso_weights,
            hsic_fNET_gso_weights,
            hsic_sp_weights,
            hsic_fNET_sp_weights,
            L2X_weights,
            invase_scores,
            shap_values,
            bshap_values,
            lime_values]
        
        selected_features = [ gumbelsparsemax_selected_features,
            gumbelsparsemax2_selected_features,
            HISCFGSP_selected_features,
            gumbelsoftmax_selected_features,
            HISCFGSO_selected_features,
            sparsemax_selected_features,
            HISCFSP_selected_features,
            L2X_selected_features,
            invase_selected_features,
            shap_selected_featuers,
            bishap_selected_featuers,
            lime_selected_featuers]
        return feature_importances , selected_features

# def select_top_features(feature_importances, feature_names, k=10):
#     top_features = []
#     for fi in feature_importances:
#         # Flatten the feature importance array if necessary
#         if isinstance(fi, list):
#             fi = np.array(fi)
#         if isinstance(fi, np.ndarray):
#             if len(fi.shape) > 1:  # If it's a 2D array (instance-wise feature importance)
#                 fi = np.mean(fi, axis=0)  # Average over instances
#             sorted_indices = np.argsort(np.abs(fi))[-k:]  # Select top k features
#             top_features.append([feature_names[i] for i in sorted_indices])
#         else:
#             fi = fi.detach().numpy() if isinstance(fi, torch.Tensor) else fi
#             sorted_indices = np.argsort(np.abs(fi))[-k:]  # Select top k features
#             top_features.append([feature_names[i] for i in sorted_indices])
#     return top_features

# def train_and_evaluate_downstream_model(X_train, X_test, y_train, y_test, selected_features, feature_names, model_type='regression'):
#     # Get the indices of the selected features
#     feature_indices = [feature_names.index(feature) for feature in selected_features]
    
#     # Subset the training and test data using the selected features
#     X_train_selected = X_train[:, feature_indices]
#     X_test_selected = X_test[:, feature_indices]
    
#     # Train a model
#     if model_type == 'regression':
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     elif model_type == 'classification':
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train_selected, y_train)
    
#     # Evaluate the model
#     y_pred = model.predict(X_test_selected)
#     metrics = model_performance_metrics(y_test, y_pred)
#     return metrics

# def compare_downstream_task_performance(X_train, X_test, y_train, y_test, feature_importances, feature_names, method_names, k=10):
#     # Select top features for each method
#     top_features_list = select_top_features(feature_importances, feature_names, k)
    
#     # Train and evaluate downstream models
#     performance_metrics = {}
#     for method, top_features in zip(method_names, top_features_list):
#         print(f"Training model with features selected by {method}...")
#         metrics = train_and_evaluate_downstream_model(X_train, X_test, y_train, y_test, top_features, feature_names)
#         performance_metrics[method] = metrics
    
#     # Print performance comparison
#     print("\nDownstream Task Performance Comparison:")
#     for method, metrics in performance_metrics.items():
#         print(f"{method}: MAE={metrics[0]:.2f}, MSE={metrics[1]:.2f}, RMSE={metrics[2]:.2f}, R2={metrics[3]:.2f}")
    
#     return performance_metrics

if __name__=='__main__':
   
    #miles_per_gallon(), stackloos()
    # datasets = [diabetes(), california_housing(), extramarital_affairs(), mode_choice(),  statlog_heart(), credit_approval(), heart_mortality()]  
    # datasets = [diabetes(), california_housing(), extramarital_affairs(), mode_choice(),  statlog_heart(), credit_approval()]  

    datasets = [diabetes()]

    
    epoch = 100
    BATCH_SIZE =1000

    for data in datasets:
        
        # Loading data
        X, y, db_name, mode = data
        print(db_name)
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n_train, d = X_train.shape
        input_dim = d 
        # hidden_dim1 = 100
        # hidden_dim2 = 100
        layers = [200, 300, 400, 500, 400, 300, 200]
        feature_layers = [20, 50, 100, 200, 100, 50, 20]
        act_fun_layer = torch.nn.Sigmoid
        act_fun_featlayer = torch.nn.Sigmoid
        
        
    ##----------------------------------------------------------------------------------------------------------
        # train a glass box model 
        model = RandomForestRegressor(n_estimators=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        exp_func = model.predict
        metrics = model_performance_metrics(y_test, y_pred)

    ##----------------------------------------------------------------------------------------------------------   
        ## Determine number of samples from train data to be replaced by removed feature in explainers
        # X_bg = X_train if X_train.shape[0] < 100 else shap.sample(X_train, 100)

        X_bg = X_train 
        sampleNo_tbx = 200
        indices = np.random.choice(X_test.shape[0], size=min(sampleNo_tbx, X_test.shape[0]), replace=False)
        y_bg = model.predict(X_bg)
        X_tbx = X_test[indices,:]
        y_tbx = model.predict(X_tbx)

        X_tensor_bg = torch.from_numpy(X_bg).float()  # Convert to float tensor
        y_tensor_bg = torch.from_numpy(y_bg).float()
        X_tensor_tbx = torch.from_numpy(X_tbx).float()  # Convert to float tensor
        y_tensor_tbx = torch.from_numpy(y_tbx).float()

        feature_names = [f"Feature_{i}" for i in range(X.shape[1])] #Convert to dataframe for invase method
        X_df_bg = pd.DataFrame(X_bg, columns=feature_names)
        y_series_bg = pd.Series(y_bg, name="Target")
        X_df_tbx = pd.DataFrame(X_tbx, columns=feature_names)
        X_sample_no = 500  # number of sampels for generating explanation


    ##-----------------------------------------------------------------------------------------------------------
      
        # maple_values = call_Maple(maple_explainer, X_tbx)

        feature_importances, selected_features =compare_methods(model, d, exp_func, X_tensor_bg, y_tensor_bg, X_bg, y_bg, X_df_bg, y_series_bg, X_tensor_tbx,y_tensor_tbx, X_sample_no )
  
        # Save feature_importances to a .pkl file
        with open('feature_importances.pkl', 'wb') as f:
             pickle.dump(feature_importances, f)

        with open('selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
    ##---------------------------------------------------------------------------------------------------------
      
    
        method_names = [
            'Hsic_GumbelSparsemax', 'Hsic_GumbelSparsemax2', 
            'HSICFeatureNet_GumbelSparsemax', 
            'Hsic_GumbelSoftmax', 
            'HsicFeatureNet_GumbelSoftmax', 
            'Hsic_Sparsemax', 
            'HsicFeatureNet_Sparsemax', 
            'L2X', 'INVASE', 'Kernel SHAP',  'Bivariate SHAP',
            'LIME'
        ]




        # ... (rest of the code remains the same)

        # Compare feature importance weights
        # feature_importances = {
        #     'Hsic_GumbelSparsemax': hsic_gsp_weights,
        #     'Hsic_GumbelSparsemax2': hsic_gsp_weights2,
        #     'HSICFeatureNet_GumbelSparsemax': hsic_fNET_gsp_weights,
        #     'Hsic_GumbelSoftmax': hsic_gso_weights,
        #     'HsicFeatureNet_GumbelSoftmax': hsic_fNET_gso_weights,
        #     'Hsic_Sparsemax': hsic_sp_weights,
        #     'HsicFeatureNet_Sparsemax': hsic_fNET_sp_weights,
        #     'L2X': L2X_weights,
        #     'INVASE': invase_scores,
        #     'Kernel SHAP': shap_values,
        #     'Bivariate SHAP': bshap_values,
        #     'LIME': lime_values
        # }

        # methods = list(feature_importances.keys())
        # instances, features = feature_importances[methods[0]].shape

        # # Create a figure with a grid of subplots
        # fig, axs = plt.subplots(instances, 1, figsize=(10, instances*2))

        # for i in range(5):
        #     for method in methods:
        #         importance = feature_importances[method][i]
        #         axs[i].plot(importance, label=method)
        #     axs[i].set_xlabel('Feature Index')
        #     axs[i].set_ylabel('Importance')
        #     axs[i].set_title(f'Instance {i+1}')
        #     axs[i].legend()

        # plt.tight_layout()
        # plt.show()


    # k = 5
    # feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # # Compare downstream task performance
    # performance_metrics = compare_downstream_task_performance(
    #     X_train, X_test, y_train, y_test, feature_importances, feature_names, method_names, k
    # )







        # # Plot feature importance weight        plt.figure(figsize=(10, 6))
        # for method, importance in feature_importances.items():
        #     plt.plot(importance, label=method)
        # plt.legend()
        # plt.xlabel('Feature Index')
        # plt.ylabel('Feature Importance')
        # plt.show()

        # # Compare selected features
        # selected_features = {
        #     'Hsic_GumbelSparsemax': gumbelsparsemax_selected_features,
        #     'Hsic_GumbelSparsemax2': gumbelsparsemax2_selected_features,
        #     'HSICFeatureNet_GumbelSparsemax': HISCFGSP_selected_features,
        #     'Hsic_GumbelSoftmax': gumbelsoftmax_selected_features,
        #     'HsicFeatureNet_GumbelSoftmax': HISCFGSO_selected_features,
        #     'Hsic_Sparsemax': sparsemax_selected_features,
        #     'HsicFeatureNet_Sparsemax': HISCFSP_selected_features,
        #     'L2X': L2X_selected_features,
        #     'INVASE': invase_selected_features,
        #     'Kernel SHAP': shap_selected_featuers,
        #     'Bivariate SHAP': bishap_selected_featuers,
        #     'LIME': lime_selected_featuers
        # }

        # # Print selected features
        # for method, features in selected_features.items():
        #     print(f"{method}: {features}")

        # # Calculate correlation between feature importance weights
        # correlation_matrix = np.zeros((len(feature_importances), len(feature_importances)))
        # methods = list(feature_importances.keys())
        # for i, method1 in enumerate(methods):
        #     for j, method2 in enumerate(methods):
        #         importance1 = feature_importances[method1]
        #         importance2 = feature_importances[method2]
        #         correlation_matrix[i, j] = np.corrcoef(importance1, importance2)[0, 1]

        # # Plot correlation matrix
        # plt.figure(figsize=(10, 8))
        # plt.imshow(correlation_matrix, interpolation='nearest')
        # plt.colorbar()
        # plt.xticks(range(len(methods)), methods, rotation=90)
        # plt.yticks(range(len(methods)), methods)
        # plt.show()

       


