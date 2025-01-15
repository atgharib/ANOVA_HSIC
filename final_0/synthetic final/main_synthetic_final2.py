import numpy as np
import matplotlib.pyplot as plt 

import shap 
from explainer.bishapley_kernel import Bivariate_KernelExplainer
from shapreg import removal, games, shapley
from explainer.MAPLE import MAPLE
from lime import lime_tabular
from pathlib import Path
import pandas as pd  
from openpyxl import load_workbook
from synthesized_data import *
from hsic_gumbelsparsemax3 import *
from hsic_gumbelsoftmax3 import *
from hsic_sparsemax3 import *
from explainer.L2x_reg2 import *
from invase import INVASE
from sklearn.linear_model import LinearRegression


# Create a wrapper class for 'fn' 
class CustomModel:
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        # Since this is a deterministic function, there's nothing to fit
        # This is just a placeholder to prevent errors in INVASE
        return self

    def predict(self, X):
        # Call the function 'fn' to get the target values 'y'
        return self.fn(X)


def create_rank(scores): 
	"""
	Compute rank of each feature based on weight.
	
	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d) 
		permutated_weights = score[idx]  
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks)

# def performance_tp_fp(ranks, g_truth):

#     exists_features = np.zeros_like(ranks)
#     exists_features[np.argsort(ranks)[:np.sum(g_truth)]] = 1  # Predict top k ranked items as 1 (positive)

#     # True Positives (TP): Where both prediction and ground truth are 1
#     TP = np.sum((g_truth == 1) & (exists_features == 1))

#     # False Positives (FP): Where prediction is 1 but ground truth is 0
#     FP = np.sum((g_truth == 0) & (exists_features == 1))
#     return TP, FP


# def sort_shap_values(scores, k):
#     scores = np.abs(scores)
#     # Sort the array by absolute values in descending order and take the top two
#     top_k = scores[np.argsort(-scores)[:k]]
#     return 
def create_important_features_existence(ranks, g_truth):
    ''' ranks is the rank of each feature'''
    ''' This function finds the indices of the top k ranked features and 
        sets the corresponding positions in an important_features array to 1. '''

    important_features = np.zeros_like(ranks)
    for i in range(ranks.shape[0]):
        index_imp = np.argsort(ranks[i])[:int(np.sum(g_truth[i,:]))]
        important_features[i, index_imp] = 1 
    
    return important_features
    
     
def convert_Scores_to_impfExistence(score_init, Threshold):
    score_abs=abs(score_init)
    score = 1.*(score_abs > Threshold)    
    return score

def performance_metric(score, g_truth):

    n = len(score)
    TPR = np.zeros([n,])
    FDR = np.zeros([n,])
    
    for i in range(n):

        # TPR    
        TP_vals = np.sum(score[i,:] * g_truth[i,:])
        TPR_den = np.sum(g_truth[i,:])
        TPR[i] = 100 * float(TP_vals)/float(TPR_den+1e-8)
    
        # FDR
        FD_vals = np.sum(score[i,:] * (1-g_truth[i,:]))
        FDR_den = np.sum(score[i,:])
        FDR[i] = 100 * float(FD_vals)/float(FDR_den+1e-8)

    return np.mean(TPR), np.mean(FDR), np.std(TPR), np.std(FDR), TP_vals, FD_vals
    
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

    return model , model.sigmas, model.sigma_y, model.importance_weights

def call_ushap_maple_lime(X, y, fn, X_sample_no):
       ## LIME, Unbiased SHAP, and MAPLE 
    lime_exp = lime_tabular.LimeTabularExplainer(X, discretize_continuous=False, mode="regression")
    imputer = removal.MarginalExtension(X, fn)
    exp_maple = MAPLE(X, y, X, y)

    ushap_values = np.empty_like(X)
    lime_values = np.empty_like(X)
    maple_values = np.empty_like(X)
    for i in range(X.shape[0]):
        x = X[i, ]
    
        ## Unbiased kernel shap 
        game = games.PredictionGame(imputer, x)
        values = shapley.ShapleyRegression(game, n_samples=X_sample_no, paired_sampling=False)
        ushap_values[i,:] = values.values.squeeze()

        ## LIME 
        exp = lime_exp.explain_instance(x, fn, num_samples = X_sample_no)
            
        for tpl in exp.as_list():
            lime_values[i, int(tpl[0])] = tpl[1]

        ## MAPLE
        mpl_exp = exp_maple.explain(x)
        maple_values[i,] = (mpl_exp['coefs'][1:]).squeeze()
    
    return ushap_values, lime_values, maple_values

def calculate_Hsic_shap_scores_impfeatures(method_name, X_tensor, y_tensor, num_sampling, threshold):
     
    if method_name =='HSIC_GumbelSparsemax':
        model, sigmas, sigma_y, weights = call_HSIC_methods('HSIC_GumbelSparsemax', X_tensor, y_tensor, num_sampling)
    elif method_name == 'HSIC_GumbelSoftmax':
        model, sigmas, sigma_y, weights = call_HSIC_methods('HSIC_GumbelSoftmax', X_tensor, y_tensor, num_sampling)
    elif method_name == 'HSIC_Sparsemax':
        model, sigmas, sigma_y, weights = call_HSIC_methods('HSIC_Sparsemax', X_tensor, y_tensor, num_sampling)
  
    l_shap_values, _ = model.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor, sigmas, sigma_y, weights)
    hsic_shap_values = l_shap_values.detach().cpu().numpy()
    normalized_hsic_gsp_shap_values = abs(hsic_shap_values) / abs(hsic_shap_values).sum(axis=1, keepdims=True)

    hsic_shap_ranks = create_rank(hsic_shap_values.squeeze())
    hsic_shap_avg_ranks = np.mean(hsic_shap_ranks[:,feature_imp], axis=1)

    l_scores, _, _ = model.predict(X_tensor)
    hsic_scores = l_scores.detach().cpu().numpy()
    normalized_hsic_gsp_scores = abs(hsic_scores) / abs(hsic_scores).sum(axis=1, keepdims=True)
    hsic_score_ranks = create_rank(hsic_scores.squeeze())

    # l2_norm = np.linalg.norm(abs(hsic_gsp_shap_values), ord=2, axis=1, keepdims=True)
   
    impfeatures_existence_fromShap = convert_Scores_to_impfExistence(hsic_shap_values, threshold)
    impfeatures_existence_fromShap_normalized = convert_Scores_to_impfExistence(normalized_hsic_gsp_shap_values, threshold)

    impfeatures_existence_fromScores = convert_Scores_to_impfExistence(hsic_scores, threshold)
    impfeatures_existence_fromScores_normalized = convert_Scores_to_impfExistence(normalized_hsic_gsp_scores, threshold)


    return impfeatures_existence_fromShap, impfeatures_existence_fromShap_normalized, impfeatures_existence_fromScores, impfeatures_existence_fromScores_normalized, hsic_shap_ranks, hsic_shap_avg_ranks, hsic_score_ranks

def Compare_methods(X, y, X_test, X_sample_no, fn, feature_imp, threshold):

    #HSIC_anova
    X_tensor = torch.from_numpy(X).float()  # Convert to float tensor
    y_tensor = torch.from_numpy(y).float()
    X_tensor_test = torch.from_numpy(X_test).float() 
    num_sampling = len(feature_imp)

    gsp_impfeatures_fromShap, gsp_impfeatures_fromShap_normalized, gsp_impfeatures_fromScores, gsp_impfeatures_fromScores_normalized, hsic_gsp_shap_ranks, hsic_gsp_shap_avg_ranks, hsic_gsp_score_ranks =calculate_Hsic_shap_scores_impfeatures('HSIC_GumbelSparsemax', X_tensor, y_tensor, num_sampling, threshold)
    hsic_gsp_shap_impfeatures_No = create_important_features_existence(hsic_gsp_shap_ranks, g_train)
    hsic_gsp_score_impfeatures_No = create_important_features_existence(hsic_gsp_score_ranks, g_train)
    hsic_gsp_fromShap_TPR_mean, hsic_gsp_fromShap_FDR_mean, hsic_gsp_fromShap_TPR_std, hsic_gsp_fromShap_FDR_std, hsic_gsp_fromShap_TP, hsic_gsp_fromShap_FD = performance_metric(gsp_impfeatures_fromShap, g_train)
    hsic_gsp_fromShap_normalized_TPR_mean, hsic_gsp_fromShap_normalized_FDR_mean, hsic_gsp_fromShap_normalized_TPR_std, hsic_gsp_fromShap_normalized_FDR_std, hsic_gsp_fromShap_normalized_TP, hsic_gsp_fromShap_normalized_FD = performance_metric(gsp_impfeatures_fromShap_normalized, g_train)
    hsic_gsp_fromScores_TPR_mean, hsic_gsp_fromScores_FDR_mean, hsic_gsp_fromScores_TPR_std, hsic_gsp_fromScores_FDR_std, hsic_gsp_fromScores_TP, hsic_gsp_fromScores_FD = performance_metric(gsp_impfeatures_fromScores, g_train)
    hsic_gsp_fromScores_normalized_TPR_mean, hsic_gsp_fromScores_normalized_FDR_mean, hsic_gsp_fromScores_normalized_TPR_std, hsic_gsp_fromScores_normalized_FDR_std, hsic_gsp_fromScores_normalized_TP, hsic_gsp_fromScores_normalized_FD = performance_metric(gsp_impfeatures_fromScores_normalized, g_train)
    hsic_gsp_shap_impfeatures_No_TPR_mean, hsic_gsp_shap_impfeatures_No_FDR_mean, hsic_gsp_shap_impfeatures_No_TPR_std, hsic_gsp_shap_impfeatures_No_FDR_std, hsic_gsp_shap_impfeatures_No_TP, hsic_gsp_shap_impfeatures_No_FD = performance_metric(hsic_gsp_shap_impfeatures_No, g_train)
    hsic_gsp_score_impfeatures_No_TPR_mean, hsic_gsp_score_impfeatures_No_FDR_mean, hsic_gsp_score_impfeatures_No_TPR_std, hsic_gsp_score_impfeatures_No_FDR_std, hsic_gsp_score_impfeatures_No_TP, hsic_gsp_score_impfeatures_No_FD = performance_metric(hsic_gsp_score_impfeatures_No, g_train)

    #HSICNetGumbelSparsemax another run with different # of sampling
    gsp_impfeatures_fromShap2, gsp_impfeatures_fromShap_normalized2, gsp_impfeatures_fromScores2, gsp_impfeatures_fromScores_normalized2, hsic_gsp_shap_ranks2, hsic_gsp_shap_avg_ranks2, hsic_gsp_score_ranks2 =calculate_Hsic_shap_scores_impfeatures('HSIC_GumbelSparsemax', X_tensor, y_tensor, num_sampling+3, threshold)
    hsic_gsp_shap_impfeatures_No2 = create_important_features_existence(hsic_gsp_shap_ranks2, g_train)
    hsic_gsp_score_impfeatures_No2 = create_important_features_existence(hsic_gsp_score_ranks2, g_train)
    hsic_gsp_fromShap_TPR_mean2, hsic_gsp_fromShap_FDR_mean2, hsic_gsp_fromShap_TPR_std2, hsic_gsp_fromShap_FDR_std2, hsic_gsp_fromShap_TP2, hsic_gsp_fromShap_FD2 = performance_metric(gsp_impfeatures_fromShap2, g_train)
    hsic_gsp_fromShap_normalized_TPR_mean2, hsic_gsp_fromShap_normalized_FDR_mean2, hsic_gsp_fromShap_normalized_TPR_std2, hsic_gsp_fromShap_normalized_FDR_std2, hsic_gsp_fromShap_normalized_TP2, hsic_gsp_fromShap_normalized_FD2 = performance_metric(gsp_impfeatures_fromShap_normalized2, g_train)
    hsic_gsp_fromScores_TPR_mean2, hsic_gsp_fromScores_FDR_mean2, hsic_gsp_fromScores_TPR_std2, hsic_gsp_fromScores_FDR_std2, hsic_gsp_fromScores_TP2, hsic_gsp_fromScores_FD2 = performance_metric(gsp_impfeatures_fromScores2, g_train)
    hsic_gsp_fromScores_normalized_TPR_mean2, hsic_gsp_fromScores_normalized_FDR_mean2, hsic_gsp_fromScores_normalized_TPR_std2, hsic_gsp_fromScores_normalized_FDR_std2, hsic_gsp_fromScores_normalized_TP2, hsic_gsp_fromScores_normalized_FD2 = performance_metric(gsp_impfeatures_fromScores_normalized2, g_train)
    hsic_gsp_shap_impfeatures_No_TPR_mean2, hsic_gsp_shap_impfeatures_No_FDR_mean2, hsic_gsp_shap_impfeatures_No_TPR_std2, hsic_gsp_shap_impfeatures_No_FDR_std2, hsic_gsp_shap_impfeatures_No_TP, hsic_gsp_shap_impfeatures_No_FD2 = performance_metric(hsic_gsp_shap_impfeatures_No2, g_train)
    hsic_gsp_score_impfeatures_No_TPR_mean2, hsic_gsp_score_impfeatures_No_FDR_mean2, hsic_gsp_score_impfeatures_No_TPR_std2, hsic_gsp_score_impfeatures_No_FDR_std2, hsic_gsp_score_impfeatures_No_TP2, hsic_gsp_score_impfeatures_No_FD2 = performance_metric(hsic_gsp_score_impfeatures_No2, g_train)



    #HSICNetGumbelSoftmax
    gso_impfeatures_fromShap, gso_impfeatures_fromShap_normalized, gso_impfeatures_fromScores, gso_impfeatures_fromScores_normalized, hsic_gso_shap_ranks, hsic_gso_shap_avg_ranks, hsic_gso_score_ranks =calculate_Hsic_shap_scores_impfeatures('HSIC_GumbelSoftmax', X_tensor, y_tensor, num_sampling, threshold)
    hsic_gso_shap_impfeatures_No = create_important_features_existence(hsic_gso_shap_ranks, g_train)
    hsic_gso_score_impfeatures_No = create_important_features_existence(hsic_gso_score_ranks, g_train)
    hsic_gso_fromShap_TPR_mean, hsic_gso_fromShap_FDR_mean, hsic_gso_fromShap_TPR_std, hsic_gso_fromShap_FDR_std, hsic_gso_fromShap_TP, hsic_gso_fromShap_FD = performance_metric(gso_impfeatures_fromShap, g_train)
    hsic_gso_fromShap_normalized_TPR_mean, hsic_gso_fromShap_normalized_FDR_mean, hsic_gso_fromShap_normalized_TPR_std, hsic_gso_fromShap_normalized_FDR_std, hsic_gso_fromShap_normalized_TP, hsic_gso_fromShap_normalized_FD = performance_metric(gso_impfeatures_fromShap_normalized, g_train)
    hsic_gso_fromScores_TPR_mean, hsic_gso_fromScores_FDR_mean, hsic_gso_fromScores_TPR_std, hsic_gso_fromScores_FDR_std, hsic_gso_fromScores_TP, hsic_gso_fromScores_FD = performance_metric(gso_impfeatures_fromScores, g_train)
    hsic_gso_fromScores_normalized_TPR_mean, hsic_gso_fromScores_normalized_FDR_mean, hsic_gso_fromScores_normalized_TPR_std, hsic_gso_fromScores_normalized_FDR_std, hsic_gso_fromScores_normalized_TP, hsic_gso_fromScores_normalized_FD = performance_metric(gso_impfeatures_fromScores_normalized, g_train)
    hsic_gso_shap_impfeatures_No_TPR_mean, hsic_gso_shap_impfeatures_No_FDR_mean, hsic_gso_shap_impfeatures_No_TPR_std, hsic_gso_shap_impfeatures_No_FDR_std, hsic_gso_shap_impfeatures_No_TP, hsic_gso_shap_impfeatures_No_FD = performance_metric(hsic_gso_shap_impfeatures_No, g_train)
    hsic_gso_score_impfeatures_No_TPR_mean, hsic_gso_score_impfeatures_No_FDR_mean, hsic_gso_score_impfeatures_No_TPR_std, hsic_gso_score_impfeatures_No_FDR_std, hsic_gso_score_impfeatures_No_TP, hsic_gso_score_impfeatures_No_FD = performance_metric(hsic_gso_score_impfeatures_No, g_train)


    #HSICNetGumbelSoftmax with 2*num_feature_imp
    gso_impfeatures_fromShap2, gso_impfeatures_fromShap_normalized2, gso_impfeatures_fromScores2, gso_impfeatures_fromScores_normalized2, hsic_gso_shap_ranks2, hsic_gso_shap_avg_ranks2, hsic_gso_score_ranks2 =calculate_Hsic_shap_scores_impfeatures('HSIC_GumbelSoftmax', X_tensor, y_tensor, 2*num_sampling, threshold)
    hsic_gso_shap_impfeatures_No2 = create_important_features_existence(hsic_gso_shap_ranks2, g_train)
    hsic_gso_score_impfeatures_No2 = create_important_features_existence(hsic_gso_score_ranks2, g_train)
    hsic_gso_fromShap_TPR_mean2, hsic_gso_fromShap_FDR_mean2, hsic_gso_fromShap_TPR_std2, hsic_gso_fromShap_FDR_std2, hsic_gso_fromShap_TP2, hsic_gso_fromShap_FD2 = performance_metric(gso_impfeatures_fromShap2, g_train)
    hsic_gso_fromShap_normalized_TPR_mean2, hsic_gso_fromShap_normalized_FDR_mean2, hsic_gso_fromShap_normalized_TPR_std2, hsic_gso_fromShap_normalized_FDR_std2, hsic_gso_fromShap_normalized_TP2, hsic_gso_fromShap_normalized_FD2 = performance_metric(gso_impfeatures_fromShap_normalized2, g_train)
    hsic_gso_fromScores_TPR_mean2, hsic_gso_fromScores_FDR_mean2, hsic_gso_fromScores_TPR_std2, hsic_gso_fromScores_FDR_std2, hsic_gso_fromScores_TP2, hsic_gso_fromScores_FD2 = performance_metric(gso_impfeatures_fromScores2, g_train)
    hsic_gso_fromScores_normalized_TPR_mean2, hsic_gso_fromScores_normalized_FDR_mean2, hsic_gso_fromScores_normalized_TPR_std2, hsic_gso_fromScores_normalized_FDR_std2, hsic_gso_fromScores_normalized_TP2, hsic_gso_fromScores_normalized_FD2 = performance_metric(gso_impfeatures_fromScores_normalized2, g_train)
    hsic_gso_shap_impfeatures_No_TPR_mean2, hsic_gso_shap_impfeatures_No_FDR_mean2, hsic_gso_shap_impfeatures_No_TPR_std2, hsic_gso_shap_impfeatures_No_FDR_std2, hsic_gso_shap_impfeatures_No_TP, hsic_gso_shap_impfeatures_No_FD2 = performance_metric(hsic_gso_shap_impfeatures_No2, g_train)
    hsic_gso_score_impfeatures_No_TPR_mean2, hsic_gso_score_impfeatures_No_FDR_mean2, hsic_gso_score_impfeatures_No_TPR_std2, hsic_gso_score_impfeatures_No_FDR_std2, hsic_gso_score_impfeatures_No_TP2, hsic_gso_score_impfeatures_No_FD2 = performance_metric(hsic_gso_score_impfeatures_No2, g_train)

   

    #HSICNetSparseMax
    sp_impfeatures_fromShap, sp_impfeatures_fromShap_normalized, sp_impfeatures_fromScores, sp_impfeatures_fromScores_normalized, hsic_sp_shap_ranks, hsic_sp_shap_avg_ranks, hsic_sp_score_ranks =calculate_Hsic_shap_scores_impfeatures('HSIC_Sparsemax', X_tensor, y_tensor, num_sampling, threshold)
    hsic_sp_shap_impfeatures_No = create_important_features_existence(hsic_sp_shap_ranks, g_train)
    hsic_sp_score_impfeatures_No = create_important_features_existence(hsic_sp_score_ranks, g_train)
    hsic_sp_fromShap_TPR_mean, hsic_sp_fromShap_FDR_mean, hsic_sp_fromShap_TPR_std, hsic_sp_fromShap_FDR_std, hsic_sp_fromShap_TP, hsic_sp_fromShap_FD = performance_metric(sp_impfeatures_fromShap, g_train)
    hsic_sp_fromShap_normalized_TPR_mean, hsic_sp_fromShap_normalized_FDR_mean, hsic_sp_fromShap_normalized_TPR_std, hsic_sp_fromShap_normalized_FDR_std, hsic_sp_fromShap_normalized_TP, hsic_sp_fromShap_normalized_FD = performance_metric(sp_impfeatures_fromShap_normalized, g_train)
    hsic_sp_fromScores_TPR_mean, hsic_sp_fromScores_FDR_mean, hsic_sp_fromScores_TPR_std, hsic_sp_fromScores_FDR_std, hsic_sp_fromScores_TP, hsic_sp_fromScores_FD = performance_metric(sp_impfeatures_fromScores, g_train)
    hsic_sp_fromScores_normalized_TPR_mean, hsic_sp_fromScores_normalized_FDR_mean, hsic_sp_fromScores_normalized_TPR_std, hsic_sp_fromScores_normalized_FDR_std, hsic_sp_fromScores_normalized_TP, hsic_sp_fromScores_normalized_FD = performance_metric(sp_impfeatures_fromScores_normalized, g_train)
    hsic_sp_shap_impfeatures_No_TPR_mean, hsic_sp_shap_impfeatures_No_FDR_mean, hsic_sp_shap_impfeatures_No_TPR_std, hsic_sp_shap_impfeatures_No_FDR_std, hsic_sp_shap_impfeatures_No_TP, hsic_sp_shap_impfeatures_No_FD = performance_metric(hsic_sp_shap_impfeatures_No, g_train)
    hsic_sp_score_impfeatures_No_TPR_mean, hsic_sp_score_impfeatures_No_FDR_mean, hsic_sp_score_impfeatures_No_TPR_std, hsic_sp_score_impfeatures_No_FDR_std, hsic_sp_score_impfeatures_No_TP, hsic_sp_score_impfeatures_No_FD = performance_metric(hsic_sp_score_impfeatures_No, g_train)

    

    #L2X #retrun feature importance
    L2X_explainer = L2X(X, y, input_dim, num_sampling)
    L2X_scores = L2X_explainer.predict(X, verbose=1, batch_size=BATCH_SIZE)
    L2X_ranks = create_rank(L2X_scores)
    L2x_avg_ranks = np.mean(L2X_ranks[:,feature_imp], axis=1)
    normalized_L2X_scores= abs(L2X_scores) / abs(L2X_scores).sum(axis=1, keepdims=True)
    L2x_impfeatures_existence = convert_Scores_to_impfExistence(L2X_scores, threshold)
    L2x_impfeatures_existence_normalized = convert_Scores_to_impfExistence(normalized_L2X_scores, threshold)
    L2x_TPR_mean, L2x_FDR_mean, L2x_TPR_std, L2x_FDR_std, L2x_TP, L2x_FD = performance_metric(L2x_impfeatures_existence, g_train)
    L2x_normalized_TPR_mean, L2x_normalized_FDR_mean, L2x_normalized_TPR_std, L2x_normalized_FDR_std, L2x_normalized_TP, L2x_normalized_FD = performance_metric(L2x_impfeatures_existence_normalized, g_train)




    #L2X with 2*num_feature_imp
    L2X_explainer2 = L2X(X, y, input_dim, 2*num_sampling)
    L2X_scores2 = L2X_explainer2.predict(X, verbose=1, batch_size=BATCH_SIZE)
    L2X_ranks2 = create_rank(L2X_scores2)
    L2x_avg_ranks2 = np.mean(L2X_ranks2[:,feature_imp], axis=1)
    normalized_L2X_scores2= abs(L2X_scores2) / abs(L2X_scores2).sum(axis=1, keepdims=True)
    L2x_impfeatures_existence2 = convert_Scores_to_impfExistence(L2X_scores2, threshold)
    L2x_impfeatures_existence_normalized2 = convert_Scores_to_impfExistence(normalized_L2X_scores2, threshold)
    L2x_TPR_mean2, L2x_FDR_mean2, L2x_TPR_std2, L2x_FDR_std2, L2x_TP2, L2x_FD2 = performance_metric(L2x_impfeatures_existence2, g_train)
    L2x_normalized_TPR_mean2, L2x_normalized_FDR_mean2, L2x_normalized_TPR_std2, L2x_normalized_FDR_std2, L2x_normalized_TP2, L2x_normalized_FD2 = performance_metric(L2x_impfeatures_existence_normalized2, g_train)


    #Invasive #return feature importance
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")

    # Initialize the custom model
    model = CustomModel(fn)
    Invase_explainer = INVASE (model, X_df, y_series, n_epoch=1000, prefit=False  # to train the model
                               )
    invase_scores =(Invase_explainer.explain(X_df)).to_numpy()                      
    invase_rank = create_rank(invase_scores)
    invase_avg_ranks = np.mean(invase_rank[:,feature_imp], axis=1)
    normalized_invase_scores= abs(invase_scores) / abs(invase_scores).sum(axis=1, keepdims=True)
    invase_impfeatures_existence = convert_Scores_to_impfExistence(invase_scores, threshold)
    invase_impfeatures_existence_normalized = convert_Scores_to_impfExistence(normalized_invase_scores, threshold)
    invase_TPR_mean, invase_FDR_mean, invase_TPR_std, invase_FDR_std, invase_TP, invase_FD = performance_metric(invase_impfeatures_existence, g_train)
    invase_normalized_TPR_mean, invase_normalized_FDR_mean, invase_normalized_TPR_std, invase_normalized_FDR_std, invase_normalized_TP, invase_normalized_FD = performance_metric(invase_impfeatures_existence_normalized, g_train)


    ## SHAP
    explainer = shap.KernelExplainer(fn, X, l1_reg=False)
    shap_values = explainer.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    shap_ranks = create_rank(shap_values.squeeze())
    shap_avg_ranks = np.mean(shap_ranks[:,feature_imp], axis=1)
    normalized_shap_values= abs(shap_values) / abs(shap_values).sum(axis=1, keepdims=True)
    shap_impfeatures_existence = create_important_features_existence(shap_ranks, g_train)
    shap_TPR_mean, shap_FDR_mean, shap_TPR_std, shap_FDR_std, shap_TP, shap_FD = performance_metric(shap_impfeatures_existence, g_train)


    # plt.boxplot([gem_avg_ranks, shap_avg_ranks, sshap_avg_ranks])
    ## Bivariate SHAP
    bishap = Bivariate_KernelExplainer(fn, X)
    bishap_values = bishap.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    bishap_ranks = create_rank(np.array(bishap_values).squeeze())
    bishap_avg_ranks = np.mean(bishap_ranks[:,feature_imp], axis=1)
    normalized_bishap_values= abs(bishap_values) / abs(bishap_values).sum(axis=1, keepdims=True)
    bishap_impfeatures_existence = create_important_features_existence(bishap_ranks, g_train)
    bishap_TPR_mean, bishap_FDR_mean, bishap_TPR_std, bishap_FDR_std, bishap_TP, bishap_FD = performance_metric(bishap_impfeatures_existence, g_train)

    ushap_values, lime_values, maple_values= call_ushap_maple_lime(X, y, fn, X_sample_no)
  
    lime_ranks = create_rank(lime_values)
    lime_avg_ranks = np.mean(lime_ranks[:,feature_imp], axis=1)
    normalized_lime_values= abs(lime_values) / abs(lime_values).sum(axis=1, keepdims=True)
    lime_impfeatures_existence = create_important_features_existence(lime_ranks, g_train)
    lime_TPR_mean, lime_FDR_mean, lime_TPR_std, lime_FDR_std, lime_TP, lime_FD = performance_metric(lime_impfeatures_existence, g_train)

    maple_ranks = create_rank(maple_values)
    maple_avg_ranks = np.mean(maple_ranks[:,feature_imp], axis=1)
    normalized_maple_values= abs(maple_values) / abs(maple_values).sum(axis=1, keepdims=True)
    maple_impfeatures_existence = create_important_features_existence(maple_ranks, g_train)
    maple_TPR_mean, maple_FDR_mean, maple_TPR_std, maple_FDR_std, maple_TP, maple_FD = performance_metric(maple_impfeatures_existence, g_train)

    ushap_ranks = create_rank(ushap_values)
    ushap_avg_ranks = np.mean(ushap_ranks[:,feature_imp], axis=1)
    normalized_ushap_values= abs(ushap_values) / abs(ushap_values).sum(axis=1, keepdims=True)
    ushap_impfeatures_existence = create_important_features_existence(ushap_ranks, g_train)
    ushap_TPR_mean, ushap_FDR_mean, ushap_TPR_std, ushap_FDR_std, ushap_TP, ushap_FD = performance_metric(ushap_impfeatures_existence, g_train)


    results = [hsic_gsp_shap_avg_ranks, hsic_gsp_shap_avg_ranks2, hsic_gso_shap_avg_ranks, hsic_gso_shap_avg_ranks2, hsic_sp_shap_avg_ranks, 
               L2x_avg_ranks, L2x_avg_ranks2, invase_avg_ranks,  shap_avg_ranks, ushap_avg_ranks, bishap_avg_ranks, lime_avg_ranks, maple_avg_ranks]
    
    tpr = [hsic_gsp_fromShap_TPR_mean, hsic_gsp_fromScores_TPR_mean, hsic_gsp_shap_impfeatures_No_TPR_mean, hsic_gsp_score_impfeatures_No_TPR_mean, 
           hsic_gsp_fromShap_TPR_mean2, hsic_gsp_fromScores_TPR_mean2, hsic_gsp_shap_impfeatures_No_TPR_mean2, hsic_gsp_score_impfeatures_No_TPR_mean2,
           hsic_gso_fromShap_TPR_mean, hsic_gso_fromScores_TPR_mean, hsic_gso_shap_impfeatures_No_TPR_mean, hsic_gso_score_impfeatures_No_TPR_mean,
           hsic_gso_fromShap_TPR_mean2, hsic_gso_fromScores_TPR_mean2, hsic_gso_shap_impfeatures_No_TPR_mean2, hsic_gso_score_impfeatures_No_TPR_mean2,
           hsic_sp_fromShap_TPR_mean, hsic_sp_fromScores_TPR_mean, hsic_sp_shap_impfeatures_No_TPR_mean, hsic_sp_score_impfeatures_No_TPR_mean,
           L2x_TPR_mean, L2x_TPR_mean2, invase_TPR_mean, shap_TPR_mean, ushap_TPR_mean, bishap_TPR_mean, lime_TPR_mean, maple_TPR_mean ]
    
    fdr = [hsic_gsp_fromShap_FDR_mean, hsic_gsp_fromScores_FDR_mean, hsic_gsp_shap_impfeatures_No_FDR_mean, hsic_gsp_score_impfeatures_No_FDR_mean,  
           hsic_gsp_fromShap_FDR_mean2, hsic_gsp_fromScores_FDR_mean2, hsic_gsp_shap_impfeatures_No_FDR_mean2, hsic_gsp_score_impfeatures_No_FDR_mean2,
           hsic_gso_fromShap_FDR_mean, hsic_gso_fromScores_FDR_mean, hsic_gso_shap_impfeatures_No_FDR_mean, hsic_gso_score_impfeatures_No_FDR_mean,
           hsic_gso_fromShap_FDR_mean2, hsic_gso_fromScores_FDR_mean2, hsic_gso_shap_impfeatures_No_FDR_mean2, hsic_gso_score_impfeatures_No_FDR_mean2,
           hsic_sp_fromShap_FDR_mean, hsic_sp_fromScores_FDR_mean, hsic_sp_shap_impfeatures_No_FDR_mean, hsic_sp_score_impfeatures_No_FDR_mean,
           L2x_FDR_mean, L2x_FDR_mean2,invase_FDR_mean, shap_FDR_mean, ushap_FDR_mean, bishap_FDR_mean, lime_FDR_mean, maple_FDR_mean ]

    tpr_std = [hsic_gsp_fromShap_TPR_std, hsic_gsp_fromScores_TPR_std, hsic_gsp_shap_impfeatures_No_TPR_std, hsic_gsp_score_impfeatures_No_TPR_std, 
               hsic_gsp_fromShap_TPR_std2, hsic_gsp_fromScores_TPR_std2, hsic_gsp_shap_impfeatures_No_TPR_std2, hsic_gsp_score_impfeatures_No_TPR_std2,
               hsic_gso_fromShap_TPR_std, hsic_gso_fromScores_TPR_std, hsic_gso_shap_impfeatures_No_TPR_std, hsic_gso_score_impfeatures_No_TPR_std,
               hsic_gso_fromShap_TPR_std2, hsic_gso_fromScores_TPR_std2, hsic_gso_shap_impfeatures_No_TPR_std2, hsic_gso_score_impfeatures_No_TPR_std2,
               hsic_sp_fromShap_TPR_std, hsic_sp_fromScores_TPR_std, hsic_sp_shap_impfeatures_No_TPR_std, hsic_sp_score_impfeatures_No_TPR_std,
               L2x_TPR_std, L2x_TPR_std2, invase_TPR_std, shap_TPR_std,  ushap_TPR_std, bishap_TPR_std, lime_TPR_std, maple_TPR_std ]
    
    fdr_std = [hsic_gsp_fromShap_FDR_std, hsic_gsp_fromScores_FDR_std, hsic_gsp_shap_impfeatures_No_FDR_std, hsic_gsp_score_impfeatures_No_FDR_std,
               hsic_gsp_fromShap_FDR_std2, hsic_gsp_fromScores_FDR_std2, hsic_gsp_shap_impfeatures_No_FDR_std2, hsic_gsp_score_impfeatures_No_FDR_std2,
               hsic_gso_fromShap_FDR_std, hsic_gso_fromScores_FDR_std, hsic_gso_shap_impfeatures_No_FDR_std, hsic_gso_score_impfeatures_No_FDR_std,
               hsic_gso_fromShap_FDR_std2, hsic_gso_fromScores_FDR_std2, hsic_gso_shap_impfeatures_No_FDR_std2, hsic_gso_score_impfeatures_No_FDR_std2,
               hsic_sp_fromShap_FDR_std, hsic_sp_fromScores_FDR_std, hsic_sp_shap_impfeatures_No_FDR_std, hsic_sp_score_impfeatures_No_FDR_std,
               L2x_FDR_std, L2x_FDR_std2, invase_FDR_std, shap_FDR_std,  ushap_FDR_std, bishap_FDR_std, lime_FDR_std, maple_FDR_std ]


    tpr_normalized_res = [hsic_gsp_fromShap_normalized_TPR_mean, hsic_gsp_fromScores_normalized_TPR_mean, 
                          hsic_gsp_fromShap_normalized_TPR_mean2, hsic_gsp_fromScores_normalized_TPR_mean2, 
                          hsic_gso_fromShap_normalized_TPR_mean, hsic_gso_fromScores_normalized_TPR_mean, 
                          hsic_gso_fromShap_normalized_TPR_mean2, hsic_gso_fromScores_normalized_TPR_mean2, 
                          hsic_sp_fromShap_normalized_TPR_mean, hsic_sp_fromScores_normalized_TPR_mean, 
                          L2x_normalized_TPR_mean, L2x_normalized_TPR_mean2, invase_normalized_TPR_mean]

    fdr_normalized_res = [hsic_gsp_fromShap_normalized_FDR_mean, hsic_gsp_fromScores_normalized_FDR_mean,
                          hsic_gsp_fromShap_normalized_FDR_mean2, hsic_gsp_fromScores_normalized_FDR_mean2, 
                          hsic_gso_fromShap_normalized_FDR_mean, hsic_gso_fromScores_normalized_FDR_mean, 
                          hsic_gso_fromShap_normalized_FDR_mean2, hsic_gso_fromScores_normalized_FDR_mean2, 
                          hsic_sp_fromShap_normalized_FDR_mean, hsic_sp_fromScores_normalized_FDR_mean, 
                          L2x_normalized_FDR_mean, L2x_normalized_FDR_mean2, invase_normalized_FDR_mean]
    

    tpr_normalized_res_std =[hsic_gsp_fromShap_normalized_TPR_std, hsic_gsp_fromScores_normalized_TPR_std, 
                          hsic_gsp_fromShap_normalized_TPR_std2, hsic_gsp_fromScores_normalized_TPR_std2, 
                          hsic_gso_fromShap_normalized_TPR_std, hsic_gso_fromScores_normalized_TPR_std, 
                          hsic_gso_fromShap_normalized_TPR_std2, hsic_gso_fromScores_normalized_TPR_std2, 
                          hsic_sp_fromShap_normalized_TPR_std, hsic_sp_fromScores_normalized_TPR_std, 
                          L2x_normalized_TPR_std, L2x_normalized_TPR_std2, invase_normalized_TPR_std]
    fdr_normalized_res_std=[hsic_gsp_fromShap_normalized_FDR_std, hsic_gsp_fromScores_normalized_FDR_std, 
                          hsic_gsp_fromShap_normalized_FDR_std2, hsic_gsp_fromScores_normalized_FDR_std2, 
                          hsic_gso_fromShap_normalized_FDR_std, hsic_gso_fromScores_normalized_FDR_std, 
                          hsic_gso_fromShap_normalized_FDR_std2, hsic_gso_fromScores_normalized_FDR_std2, 
                          hsic_sp_fromShap_normalized_FDR_std, hsic_sp_fromScores_normalized_FDR_std, 
                          L2x_normalized_FDR_std, L2x_normalized_FDR_std2, invase_normalized_FDR_std]



   

    # tp = [hsic_gsp_TP, hsic_gsp_TP2, hsic_gso_TP, hsic_gso_TP2, hsic_sp_TP, L2x_TP, L2x_TP2, invase_TP, shap_TP,  ushap_TP, bishap_TP, lime_TP, maple_TP]
    # fp = [hsic_gsp_FD, hsic_gsp_FD2, hsic_gso_FD, hsic_gso_FD2, hsic_sp_FD, L2x_FD, L2x_FD2, invase_FD, shap_FD,  ushap_FD, bishap_FD, lime_FD, maple_FD]

    

    # print('TPR mean: ' + str(np.round(hsic_gsp_TPR_mean,1)) + '\%, ' + 'TPR std: ' + str(np.round(hsic_gsp_TPR_std,1)) + '\%, '  )
    # print('FDR mean: ' + str(np.round(hsic_gsp_FDR_mean,1)) + '\%, ' + 'FDR std: ' + str(np.round(hsic_gsp_FDR_std,1)) + '\%, '  )
    return results , tpr, fdr , tpr_std, fdr_std, tpr_normalized_res, fdr_normalized_res, tpr_normalized_res_std, fdr_normalized_res_std



if __name__=='__main__':

    
    num_samples = 500 # number of generated synthesized instances 
    input_dim = 10 # number of features for the synthesized instances
    hidden_dim1 = 100
    hidden_dim2 = 100
    hidden_dim3 = 100
    X_sample_no = 200  # number of sampels for generating explanation
    train_seed = 42
    test_seed = 1
    threshold = 0.5

   
    data_sets=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', 
             'Trigonometric Exponential', 'Exponential Hyperbolic', 'XOR', 'Syn4']
    # data_sets=['Sine Log']
    # ds_name = data_sets[0]
    # data_sets= ['Syn4']

    for ds_name in data_sets:

        # Generate synthetic data
        X_train, y_train, fn, feature_imp, g_train = generate_dataset(ds_name, num_samples, input_dim, train_seed)
        # X_test, y_test, fn, feature_imp, g_test = generate_dataset(ds_name, num_samples, input_dim, test_seed)
        
        all_results, tpr, fpr, tpr_std, fpr_std, tpr_normalized_res, fdr_normalized_res, tpr_normalized_res_std, fdr_normalized_res_std = Compare_methods(X_train, y_train, X_train, X_sample_no,  fn, feature_imp, threshold)
        method_names1 = ['Hsic_GumbelSparsemax', 'Hsic_GumbelSparsemax2', 'Hsic_GumbelSoftmax', 'Hsic_GumbelSoftmax2','Hsic_Sparsemax', 'L2X', 'L2X2', 'invasive', 'Kernel SHAP', 'Unbiased SHAP', 'Bivariate SHAP', 'LIME', 'MAPLE']


        method_names2 = ['Hsic_GumbelSparsemax_fromshapVals', 'Hsic_GumbelSparsemax_fromScores', 'Hsic_GumbelSparsemax_shap_impfeaturesNo', 'Hsic_GumbelSparsemax_score_impfeaturesNo',
                        'Hsic_GumbelSparsemax_fromshapVals2', 'Hsic_GumbelSparsemax_fromScores2', 'Hsic_GumbelSparsemax_shap_impfeaturesNo2', 'Hsic_GumbelSparsemax_score_impfeaturesNo2',
                        'Hsic_GumbelSoftmax_fromshapVals', 'Hsic_GumbelSoftmax_fromScores', 'Hsic_GumbelSoftmax_shap_impfeaturesNo', 'Hsic_GumbelSoftmax_score_impfeaturesNo',
                        'Hsic_GumbelSoftmax_fromshapVals2', 'Hsic_GumbelSoftmax_fromScores2', 'Hsic_GumbelSoftmax_shap_impfeaturesNo2', 'Hsic_GumbelSoftmax_score_impfeaturesNo2',
                        'Hsic_Sparsemax_fromshapVals', 'Hsic_Sparsemax_fromScores', 'Hsic_Sparsemax_shap_impfeaturesNo', 'Hsic_Sparsemax_score_impfeaturesNo',
                         'L2X', 'L2X2', 'invasive', 'Kernel SHAP', 'Unbiased SHAP', 'Bivariate SHAP', 'LIME', 'MAPLE']
        
        
        
        
        method_names3 = ['Hsic_GumbelSparsemax_fromshapVals_normalized', 'Hsic_GumbelSparsemax_fromScores_normalized',
                        'Hsic_GumbelSparsemax_fromshapVals_normalized2', 'Hsic_GumbelSparsemax_fromScores2_normalized2', 
                        'Hsic_GumbelSoftmax_fromshapVals_normalized', 'Hsic_GumbelSoftmax_fromScores_normalized', 
                        'Hsic_GumbelSoftmax_fromshapVals_normalized2', 'Hsic_GumbelSoftmax_fromScores_normalized2',
                        'Hsic_Sparsemax_fromshapVals_normalized', 'Hsic_Sparsemax_fromScores_normalized', 
                        'L2X_normalized', 'L2X_normalized2', 'invasive_normalized']

       
        folder_path = "results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Combine folder path and filename
        results_xsl = os.path.join(folder_path, f"results_{ds_name}_tr={threshold}.xlsx")
        results_tpr_fpr = os.path.join(folder_path, f"tpr_fpr_{ds_name}_tr={threshold}.xlsx")
        
        # write ranks to a file
        df = pd.DataFrame(all_results, index=method_names1)   
        if os.path.exists(results_xsl):
            os.remove(results_xsl)
        with pd.ExcelWriter(results_xsl, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=ds_name, index_label='Method')


        #write tpr and fpr to a file
        results_tpr_fpr_df = pd.DataFrame({
            
            'TPR': tpr,
            'TPR std': tpr_std,
            'FPR': fpr,
            'FPR std' : fpr_std,
        } , index=method_names2
        )

        results_tpr_fpr_normalized_res_df = pd.DataFrame({
            
            'TPR': tpr_normalized_res,
            'TPR std': tpr_normalized_res_std,
            'FPR': fdr_normalized_res,
            'FPR std' : fdr_normalized_res_std,
        } , index=method_names3
        )

        if os.path.exists(results_tpr_fpr):
            os.remove(results_tpr_fpr)
        with pd.ExcelWriter(results_tpr_fpr, mode='w', engine='openpyxl') as writer:
            network_spec = f"num_samples = {num_samples}, input dim = {input_dim}, hidden_dim1 = {hidden_dim1}, hidden_dim2 = {hidden_dim2}, hidden_dim3 = {hidden_dim3}, X_sample_no = {X_sample_no}, threshold = {threshold}"
    
            # Create a Pandas dataframe with a single row for network specification
            pd.DataFrame([network_spec]).to_excel(writer, sheet_name=ds_name, index=False, header=False, startrow=0)
            # Write the first dataframe
            results_tpr_fpr_df.to_excel(writer, sheet_name=ds_name, index_label='Method', startrow=2)
            
            # Leave two blank rows, then write "normalized results" in the third blank row
            start_row = len(results_tpr_fpr_df) + 5  # 3 to account for two blank rows and one for "normalized results"
            writer.sheets[ds_name].cell(row=start_row, column=1, value="normalized results")
            
            # Write the second dataframe starting after the blank rows and "normalized results"
            results_tpr_fpr_normalized_res_df.to_excel(writer, sheet_name=ds_name, index_label='Method', startrow=start_row + 1)

        
        print("done!")

    # plt.boxplot([shap_avg_ranks, bishap_avg_ranks, sshap_avg_ranks, ushap_avg_ranks, lime_avg_ranks, maple_avg_ranks])
    # plt.show()
    

