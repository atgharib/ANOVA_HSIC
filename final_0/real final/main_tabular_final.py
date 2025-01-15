import numpy as np
import pandas as pd
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
from hsic_gumbelsparsemax import *
from hsic_gumbelsoftmax import *
from hsic_sparsemax import *
from explainer.L2x_reg2 import *
from invase import INVASE
import shap 
from explainer.bishapley_kernel import Bivariate_KernelExplainer
from shapreg import removal, games, shapley
from explainer.MAPLE import MAPLE
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings("ignore")


def feature_removing_effect(feature_importance, X_tbx, X_bg, exp_func, remove_feature):
    
    sorted_features = np.argsort(np.abs(feature_importance), axis=1)
    all_predic_diff = []
    all_predic_diff_normalized = []
    all_y_real=[]
    y_x = exp_func(X_tbx)
    for i, x in enumerate(X_tbx):
        X_inverted = np.tile(x, (X_bg.shape[0],1))
        y_x_i = y_x[i]
        predic_diff = []
        predic_diff_normalized = []

        
        for j in range(remove_feature):
            X_inverted[:, sorted_features[i,:j]] = X_bg[:,sorted_features[i,:j]]
            y_hat = np.mean(exp_func(X_inverted))             

            predic_diff.append(np.abs(y_x_i - y_hat))
            predic_diff_normalized.append(np.abs(y_x_i - y_hat)/np.abs(y_x_i)) 
        
        all_predic_diff.append(predic_diff)
        all_predic_diff_normalized.append(predic_diff_normalized)
        all_y_real.append(y_x_i)


    return np.array(all_predic_diff), np.array(all_predic_diff_normalized), np.array(all_y_real)

# def compare_methods(X, y, X_test, X_sample_no, exp_func, feature_imp, input_dim, hidden_dim1, hidden_dim2):

    
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
   
    sigma_init_X = 0.1 *torch.ones(X_tensor.size(1))
    sigma_init_Y = 0.1 * torch.ones(1)
    # sigma_init_X = initialize_sigma_median_heuristic(X_tensor)
    # sigma_init_Y = initialize_sigma_y_median_heuristic(y_tensor)
    num_sampling = feature_imp

    #HSICNetGumbelSparsemax
    if method_name =='HSIC_GumbelSparsemax':
        model = HSICNetGumbelSparsemax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, num_sampling)
        
    elif method_name == 'HSIC_GumbelSoftmax':
        model = HSICNetGumbelSoftmax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, num_sampling)
    elif method_name == 'HSIC_Sparsemax':
        model = HSICNetSparseMax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y)

    model.train_model(X_tensor, y_tensor)
    sigmas = model.sigmas
    sigma_y = model.sigma_y
    weights = model.importance_weights
    

    return model , sigmas, sigma_y, weights


if __name__=='__main__':
   
    #miles_per_gallon(), stackloos()
    # datasets = [diabetes(), california_housing(), extramarital_affairs(), mode_choice(),  statlog_heart(), credit_approval(), heart_mortality()]  
    datasets = [diabetes(), california_housing(), extramarital_affairs(), mode_choice(),  statlog_heart(), credit_approval()]  

    # datasets = [diabetes()]

    sampleNo_tbx = 50

    for data in datasets:
        
        # Loading data
        X, y, db_name, mode = data
        print(db_name)
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n_train, d = X_train.shape
        input_dim = d 
        hidden_dim1 = 100
        hidden_dim2 = 100
        BATCH_SIZE =1000
        
    ##----------------------------------------------------------------------------------------------------------
        # train a glass box model 
        model = RandomForestRegressor(n_estimators=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        exp_func = model.predict
        metrics = model_performance_metrics(y_test, y_pred)

    ##----------------------------------------------------------------------------------------------------------   
        ## Determine number of samples from train data to be replaced by removed feature in explainers
        X_bg = X_train if X_train.shape[0] < 100 else shap.sample(X_train, 100)
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


    ##-----------------------------------------------------------------------------------------------------------
        # instantiate all methods
        gumbelsparsemax_model, gsp_sigmas, gsp_sigma_y, gsp_weights = call_HSIC_methods('HSIC_GumbelSparsemax', X_tensor_bg, y_tensor_bg, feature_imp=1)
        gumbelsparsemax_model2, gsp_sigmas2, gsp_sigma_y2, gsp_weights2 = call_HSIC_methods('HSIC_GumbelSparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d)
        gumbelsoftmax_model, gso_sigmas, gso_sigma_y, gso_weights = call_HSIC_methods('HSIC_GumbelSoftmax', X_tensor_bg, y_tensor_bg, feature_imp=d)
        sparsemax_model, sp_sigmas, sp_sigma_y, sp_weights = call_HSIC_methods('HSIC_Sparsemax', X_tensor_bg, y_tensor_bg, feature_imp=d) # d here does not have any effect
        L2X_explainer = L2X(X_bg, y_bg, d, d)
        # Invase_explainer = INVASE (model, X_df_bg, y_series_bg, n_epoch=100, prefit=True) #prefit = False to train the model
        shap_explainer = shap.KernelExplainer(exp_func, X_bg)
        imputer_ushap = removal.MarginalExtension(X_bg, exp_func)  ## Unbiased kernel shap 
        bshap_explainer = Bivariate_KernelExplainer(exp_func, X_bg)
        lime_explainer = LimeTabularExplainer(training_data = X_bg, mode = 'regression') 
        maple_explainer = MAPLE(X_bg, y_bg, X_bg, y_bg)


        ##*************************
        # gsp_sigmas = 0.1 *torch.ones(d)
        # gsp_sigmas2 = 0.1 *torch.ones(d)
        # gso_sigmas = 0.1 *torch.ones(d)
        # sp_sigmas = 0.1 *torch.ones(d)
        ##*************************

    ##------------------------------------------------------------------------------------------------------------

        # Different numbers of samples for explanation
        max_samplesize = 5000
        sample_no = d #if d < 14 else d*2
        sample_size = random.sample(range(2*d, np.min((2**d, max_samplesize))), sample_no)
        sample_sizes = np.sort(sample_size)   
    ##----------------------------------------------------------------------------------------------------------   
        
        # Perform experiment
        for num_samples in sample_sizes:

            ## Get the effect of feature removal
            n_sampleNo = len(sample_sizes)
            mid_index = n_sampleNo // 2
            
            if n_sampleNo % 2 == 1:
                mid_value = sample_sizes[mid_index]  # Odd length, return middle item
            else:
                mid_value = sample_sizes[mid_index - 1]
                
            if num_samples == int(mid_value):
                remove_feature = int(np.ceil(d * 0.6))

                l_gsp_shap_values, _ = gumbelsparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , gsp_sigmas, gsp_sigma_y, gsp_weights)
                hsic_gsp_shap_values = l_gsp_shap_values.detach().cpu().numpy()

                l_gsp_scores, _, _ = gumbelsparsemax_model.predict(X_tensor_tbx)
                hsic_gsp_scores = l_gsp_scores.detach().cpu().numpy()
                ##---------------
                l_gsp_shap_values2, _ = gumbelsparsemax_model2.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , gsp_sigmas2, gsp_sigma_y2, gsp_weights2)
                hsic_gsp_shap_values2 = l_gsp_shap_values2.detach().cpu().numpy()

                l_gsp_scores2, _, _ = gumbelsparsemax_model2.predict(X_tensor_tbx)
                hsic_gsp_scores2 = l_gsp_scores2.detach().cpu().numpy()
                ##--------------------
                l_gso_shap_values, _ = gumbelsoftmax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx,   gso_sigmas, gso_sigma_y, gso_weights)
                hsic_gso_shap_values = l_gso_shap_values.detach().cpu().numpy()

                l_gso_scores, _, _ = gumbelsoftmax_model.predict(X_tensor_tbx)
                hsic_gso_scores = l_gso_scores.detach().cpu().numpy()
                ##------------------
                l_sp_shap_values, _ = sparsemax_model.instancewise_shapley_value(X_tensor_bg, y_tensor_bg, X_tensor_tbx, y_tensor_tbx , sp_sigmas, sp_sigma_y, sp_weights)
                hsic_sp_shap_values = l_sp_shap_values.detach().cpu().numpy()

                l_sp_scores, _, _ = sparsemax_model.predict(X_tensor_tbx)
                hsic_sp_scores = l_sp_scores.detach().cpu().numpy()
                ##----------------------
                L2X_scores = L2X_explainer.predict(X_tbx, verbose=1, batch_size=BATCH_SIZE)
                
                # invase_scores = (Invase_explainer.explain(X_df_tbx)).to_numpy()     
                shap_values = shap_explainer.shap_values(X_tbx, nsamples=num_samples)   
                ushap_values = call_UnbiasedShap(imputer_ushap, X_tbx , num_samples)     
                bshap_values = bshap_explainer.shap_values(X_tbx, nsamples=num_samples)               
                lime_values = call_LimeToExplain(lime_explainer, X_tbx, exp_func, d)
                maple_values = call_Maple(maple_explainer, X_tbx)


                gsp_removal_effect, gsp_removal_effect_normalized, gsp_y_gt = feature_removing_effect(hsic_gsp_shap_values, X_tbx, X_bg, exp_func, remove_feature)
                gsp_removal_effect2, gsp_removal_effect_normalized2, gsp_y_gt2 = feature_removing_effect(hsic_gsp_shap_values2, X_tbx, X_bg, exp_func, remove_feature)
                gso_removal_effect, gso_removal_effect_normalized, gso_y_gt = feature_removing_effect(hsic_gso_shap_values, X_tbx, X_bg, exp_func, remove_feature)
                sp_removal_effect, sp_removal_effect_normalized, sp_y_gt = feature_removing_effect(hsic_sp_shap_values, X_tbx, X_bg, exp_func, remove_feature)

                gsp_score_removal_effect, gsp_score_removal_effect_normalized, _ = feature_removing_effect(hsic_gsp_scores, X_tbx, X_bg, exp_func, remove_feature)
                gsp_score_removal_effect2, gsp_score_removal_effect_normalized2, _ = feature_removing_effect(hsic_gsp_scores2, X_tbx, X_bg, exp_func, remove_feature)
                gso_score_removal_effect, gso_score_removal_effect_normalized, _ = feature_removing_effect(hsic_gso_scores, X_tbx, X_bg, exp_func, remove_feature)
                sp_score_removal_effect, sp_score_removal_effect_normalized, _ = feature_removing_effect(hsic_sp_scores, X_tbx, X_bg, exp_func, remove_feature)

                L2X_removal_effect, L2X_removal_effect_normalized, L2X_y_gt = feature_removing_effect(L2X_scores, X_tbx, X_bg, exp_func, remove_feature)
                # invase_removal_effect, invase_removal_effect_normalized, invase_y_gt = feature_removing_effect(invase_scores, X_tbx, X_bg, exp_func, remove_feature)
                shap_removal_effect, shap_removal_effect_normalized, shap_y_gt = feature_removing_effect(shap_values, X_tbx, X_bg, exp_func, remove_feature)
                ushap_removal_effect,  ushap_removal_effect_normalized, ushap_y_gt = feature_removing_effect(ushap_values, X_tbx, X_bg, exp_func, remove_feature)
                bshap_removal_effect, bshap_removal_effect_normalized, bshap_y_gt = feature_removing_effect(bshap_values, X_tbx, X_bg, exp_func, remove_feature)
                lime_removal_effect, lime_removal_effect_normalized, lime_y_gt = feature_removing_effect(lime_values, X_tbx, X_bg, exp_func, remove_feature)
                maple_removal_effect, maple_removal_effect_normalized, maple_y_gt = feature_removing_effect(maple_values, X_tbx, X_bg, exp_func, remove_feature)
                

    ##---------------------------------------------------------------------------------------------------------
     
        ## Saving the feature removal effect
        excel_path_feature_removal = Path(f'tabular_feature_removal.xlsx')
        excel_path_scores_feature_removal = Path(f'tabular_scores_feature_removal.xlsx')
    

        method_names = [
            'Hsic_GumbelSparsemax', 'Hsic_GumbelSparsemax_scores','Hsic_GumbelSparsemax2', 'Hsic_GumbelSparsemax2_scores',
            'Hsic_GumbelSoftmax', 'Hsic_GumbelSoftmax_scores', 'Hsic_Sparsemax', 'Hsic_Sparsemax_scores',
            'L2X', 'Kernel SHAP', 'Unbiased SHAP', 'Bivariate SHAP',
            'LIME', 'MAPLE'
        ]


        # Column headers with alternating mean and std columns
        column_header = []
        for i in range(1, remove_feature + 1):
            column_header.append(str(i))       # Mean column
            column_header.append(f'{i} std')   # Std column

        # Collect the mean and std arrays separately
        mean_results = [
            np.mean(gsp_removal_effect, axis=0),
            np.mean(gsp_score_removal_effect, axis=0),
            np.mean(gsp_removal_effect2, axis=0),
            np.mean(gsp_score_removal_effect2, axis=0),
            np.mean(gso_removal_effect, axis=0),
            np.mean(gso_score_removal_effect, axis=0),
            np.mean(sp_removal_effect, axis=0),
            np.mean(sp_score_removal_effect, axis=0),
            np.mean(L2X_removal_effect, axis=0),
            np.mean(shap_removal_effect, axis=0),
            np.mean(ushap_removal_effect, axis=0),
            np.mean(bshap_removal_effect, axis=0),
            np.mean(lime_removal_effect, axis=0),
            np.mean(maple_removal_effect, axis=0),
        ]

        std_results = [
            np.std(gsp_removal_effect, axis=0),
            np.std(gsp_score_removal_effect, axis=0),
            np.std(gsp_removal_effect2, axis=0),
            np.std(gsp_score_removal_effect2, axis=0),
            np.std(gso_removal_effect, axis=0),
            np.std(gso_score_removal_effect, axis=0),
            np.std(sp_removal_effect, axis=0),
            np.std(sp_score_removal_effect, axis=0),
            np.std(L2X_removal_effect, axis=0),
            np.std(shap_removal_effect, axis=0),
            np.std(ushap_removal_effect, axis=0),
            np.std(bshap_removal_effect, axis=0),
            np.std(lime_removal_effect, axis=0),
            np.std(maple_removal_effect, axis=0),
        ]

        # Create an interleaved DataFrame
        all_results = []
        for mean, std in zip(mean_results, std_results):
            interleaved = np.empty((mean.size + std.size,), dtype=mean.dtype)
            interleaved[0::2] = mean   # Mean values go in even indices
            interleaved[1::2] = std    # Std values go in odd indices
            all_results.append(interleaved)

   
        df = pd.DataFrame(all_results, index=method_names, columns=column_header)

        # Collect the normalized mean and std arrays separately (normalized results)
        mean_results_normalized = [
            np.mean(gsp_removal_effect_normalized, axis=0),
            np.mean(gsp_score_removal_effect_normalized, axis=0),
            np.mean(gsp_removal_effect_normalized2, axis=0),
            np.mean(gsp_score_removal_effect_normalized2, axis=0),
            np.mean(gso_removal_effect_normalized, axis=0),
            np.mean(gso_score_removal_effect_normalized, axis=0),
            np.mean(sp_removal_effect_normalized, axis=0),
            np.mean(sp_score_removal_effect_normalized, axis=0),
            np.mean(L2X_removal_effect_normalized, axis=0),
            np.mean(shap_removal_effect_normalized, axis=0),
            np.mean(ushap_removal_effect_normalized, axis=0),
            np.mean(bshap_removal_effect_normalized, axis=0),
            np.mean(lime_removal_effect_normalized, axis=0),
            np.mean(maple_removal_effect_normalized, axis=0),
        ]

        std_results_normalized = [
            np.std(gsp_removal_effect_normalized, axis=0),
            np.std(gsp_score_removal_effect_normalized, axis=0),
            np.std(gsp_removal_effect_normalized2, axis=0),
            np.std(gsp_score_removal_effect_normalized2, axis=0),
            np.std(gso_removal_effect_normalized, axis=0),
            np.std(gso_score_removal_effect_normalized, axis=0),
            np.std(sp_removal_effect_normalized, axis=0),
             np.std(sp_score_removal_effect_normalized, axis=0),
            np.std(L2X_removal_effect_normalized, axis=0),
            np.std(shap_removal_effect_normalized, axis=0),
            np.std(ushap_removal_effect_normalized, axis=0),
            np.std(bshap_removal_effect_normalized, axis=0),
            np.std(lime_removal_effect_normalized, axis=0),
            np.std(maple_removal_effect_normalized, axis=0),
        ]


        # Interleave mean and std for normalized results
        normalized_all_results = []
        for mean, std in zip(mean_results_normalized, std_results_normalized):
            interleaved = np.empty((mean.size + std.size,), dtype=mean.dtype)
            interleaved[0::2] = mean   # Mean values go in even indices
            interleaved[1::2] = std    # Std values go in odd indices
            normalized_all_results.append(interleaved)

  
        df_normalized = pd.DataFrame(normalized_all_results, index=method_names, columns=column_header)

        ### Now save both DataFrames to Excel with a gap of 2 rows ###

        # Saving to Excel
        mode = 'a' if excel_path_feature_removal.exists() else 'w'
        with pd.ExcelWriter(excel_path_feature_removal, engine='openpyxl', mode=mode) as writer:
            if mode == 'a':
                writer_book = load_workbook(excel_path_feature_removal)
                writer_sheets = dict((ws.title, ws) for ws in writer.book.worksheets)

            sheet_name = db_name
            counter = 1
            while sheet_name in writer.sheets:
                sheet_name = f"{sheet_name}_{counter}"
                counter += 1

            
            df_y_diff = pd.DataFrame(['y diff'], index=[0], columns=[None])
            
            # Write the "y diff" label
            df_y_diff.to_excel(writer, sheet_name=sheet_name, startrow=0, header=False, index=False)

            # Write the original results below "y diff"
            df.to_excel(writer, sheet_name=sheet_name, startrow=1, index_label='Method')

            # Add 2 blank rows and a "normalized diff" label before writing normalized results
            startrow = df.shape[0] + 4  # Shape[0] gives the number of rows, add 3 for 2 blank rows and 1 for the label
            
            df_normalized_diff = pd.DataFrame(['normalized diff'], index=[0], columns=[None])
            
            # Write "normalized diff" before the normalized results
            df_normalized_diff.to_excel(writer, sheet_name=sheet_name, startrow=startrow - 1, header=False, index=False)

            # Write the normalized results
            df_normalized.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index_label='Method')

        print("done!")


        ##------------------------------------------------------------------------------------------------------------------------------------------------------
        ##save Y-real
        methods_data = {
            'Hsic_GumbelSparsemax': gsp_y_gt,
            'Hsic_GumbelSparsemax2': gsp_y_gt2,
            'Hsic_GumbelSoftmax': gso_y_gt,
            'Hsic_Sparsemax': sp_y_gt,
            'L2X': L2X_y_gt,
            'Kernel SHAP': shap_y_gt,
            'Unbiased SHAP': ushap_y_gt,
            'Bivariate SHAP': bshap_y_gt,
            'LIME': lime_y_gt,
            'MAPLE': maple_y_gt
        }

        y_results = 'y_real'

       
        save_folder = os.path.join(y_results, db_name)  
        os.makedirs(save_folder, exist_ok=True)

        # Choose between 'npy' or 'pickle' as the file format
        save_format = 'npy'  # Set to 'pickle' for .pickle files

       
        for method_name, array_data in methods_data.items():
            save_path = os.path.join(save_folder, f"{method_name}.{save_format}")
            
            if save_format == 'npy':
                # Save as .npy file
                np.save(save_path, array_data)
            elif save_format == 'pickle':
                # Save as .pickle file
                with open(save_path, 'wb') as f:
                    pickle.dump(array_data, f)

        print(f"All arrays have been saved in '{save_folder}' as '{save_format}' files.")
