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
from hsic_gumbelsparsemax import *
from hsic_gumbelsoftmax import *
from hsic_sparsemax import *
from explainer.L2x_reg import *
from explainer.invase_reg import InvaseFeatureImportance
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
    

def Compare_methods(X, y, X_test, X_sample_no, fn, feature_imp):

    #HSIC_anova
    X_tensor = torch.from_numpy(X).float()  # Convert to float tensor
    y_tensor = torch.from_numpy(y).float()
    X_tensor_test = torch.from_numpy(X_test).float() 
    sigma_init_X = initialize_sigma_median_heuristic(X_tensor)
    sigma_init_Y = initialize_sigma_y_median_heuristic(y_tensor)
    num_sampling = len(feature_imp)


    #HSICNetGumbelSparsemax
    gumbelsparsemax_model = HSICNetGumbelSparsemax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, num_sampling)
    gumbelsparsemax_model.train_model(X_tensor, y_tensor)
    sigmas = gumbelsparsemax_model.sigmas
    sigma_y = gumbelsparsemax_model.sigma_y
    weights = gumbelsparsemax_model.importance_weights
    l_gsp_shap_values, _ = gumbelsparsemax_model.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_tensor.shape[0], sigmas, sigma_y, weights)
    hsic_gsp_shap_values = l_gsp_shap_values.detach().cpu().numpy()
    hsic_gsp_shap_ranks = create_rank(hsic_gsp_shap_values.squeeze())
    hsic_gsp_shap_avg_ranks = np.mean(hsic_gsp_shap_ranks[:,feature_imp], axis=1)
    hsic_gsp_shap_mean_rank = np.mean(hsic_gsp_shap_avg_ranks)
    # hsic_TPR_mean, hsic_FDR_mean, hsic_TPR_std, hsic_FDR_std = performance_metric(hsic_shap_values, g_test)
    # l2_norm = np.linalg.norm(abs(hsic_gsp_shap_values), ord=2, axis=1, keepdims=True)
    normalized_hsic_gsp_shap_values = abs(hsic_gsp_shap_values) / abs(hsic_gsp_shap_values).sum(axis=1, keepdims=True)
    hsic_gsp_impfeatures_existence = convert_Scores_to_impfExistence(normalized_hsic_gsp_shap_values, Threshold)
    hsic_gsp_TPR_mean, hsic_gsp_FDR_mean, hsic_gsp_TPR_std, hsic_gsp_FDR_std, hsic_gsp_TP, hsic_gsp_FD = performance_metric(hsic_gsp_impfeatures_existence, g_train)

    #HSICNetGumbelSparsemax another run with different # of sampling
    gumbelsparsemax_model2 = HSICNetGumbelSparsemax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, num_sampling+3)
    gumbelsparsemax_model2.train_model(X_tensor, y_tensor)
    sigmas2 = gumbelsparsemax_model2.sigmas
    sigma_y2 = gumbelsparsemax_model2.sigma_y
    weights2 = gumbelsparsemax_model2.importance_weights
    l_gsp_shap_values2, _ = gumbelsparsemax_model2.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_tensor.shape[0], sigmas2, sigma_y2, weights2)
    hsic_gsp_shap_values2 = l_gsp_shap_values2.detach().cpu().numpy()
    hsic_gsp_shap_ranks2 = create_rank(hsic_gsp_shap_values2.squeeze())
    hsic_gsp_shap_avg_ranks2 = np.mean(hsic_gsp_shap_ranks2[:,feature_imp], axis=1)
    # hsic_TPR_mean, hsic_FDR_mean, hsic_TPR_std, hsic_FDR_std = performance_metric(hsic_shap_values, g_test)
    normalized_hsic_gsp_shap_values2= abs(hsic_gsp_shap_values2) / abs(hsic_gsp_shap_values2).sum(axis=1, keepdims=True)
    hsic_gsp_impfeatures_existence2 = convert_Scores_to_impfExistence(normalized_hsic_gsp_shap_values2, Threshold)
    hsic_gsp_TPR_mean2, hsic_gsp_FDR_mean2, hsic_gsp_TPR_std2, hsic_gsp_FDR_std2, hsic_gsp_TP2, hsic_gsp_FD2 = performance_metric(hsic_gsp_impfeatures_existence2, g_train)


    #HSICNetGumbelSoftmax
    gumbelsoftmax_model = HSICNetGumbelSoftmax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, num_sampling)
    gumbelsoftmax_model.train_model(X_tensor, y_tensor)
    sigmas = gumbelsoftmax_model.sigmas
    sigma_y = gumbelsoftmax_model.sigma_y
    weights = gumbelsoftmax_model.importance_weights
    l_gso_shap_values, _ = gumbelsoftmax_model.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_tensor.shape[0], sigmas, sigma_y, weights)
    hsic_gso_shap_values = l_gso_shap_values.detach().cpu().numpy()
    hsic_gso_shap_ranks = create_rank(hsic_gso_shap_values.squeeze())
    hsic_gso_shap_avg_ranks = np.mean(hsic_gso_shap_ranks[:,feature_imp], axis=1)
    normalized_hsic_gso_shap_values= abs(hsic_gso_shap_values) / abs(hsic_gso_shap_values).sum(axis=1, keepdims=True)
    hsic_gso_impfeatures_existence = convert_Scores_to_impfExistence(normalized_hsic_gso_shap_values, Threshold)
    hsic_gso_TPR_mean, hsic_gso_FDR_mean, hsic_gso_TPR_std, hsic_gso_FDR_std, hsic_gso_TP, hsic_gso_FD = performance_metric(hsic_gso_impfeatures_existence, g_train)

    #HSICNetGumbelSoftmax with 2*num_feature_imp
    gumbelsoftmax_model2 = HSICNetGumbelSoftmax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y, 2*num_sampling)
    gumbelsoftmax_model2.train_model(X_tensor, y_tensor)
    sigmas2 = gumbelsoftmax_model2.sigmas
    sigma_y2 = gumbelsoftmax_model2.sigma_y
    weights2 = gumbelsoftmax_model2.importance_weights
    l_gso_shap_values2, _ = gumbelsoftmax_model2.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_tensor.shape[0], sigmas2, sigma_y2, weights2)
    hsic_gso_shap_values2 = l_gso_shap_values2.detach().cpu().numpy()
    hsic_gso_shap_ranks2 = create_rank(hsic_gso_shap_values2.squeeze())
    hsic_gso_shap_avg_ranks2 = np.mean(hsic_gso_shap_ranks2[:,feature_imp], axis=1)
    normalized_hsic_gso_shap_values2= abs(hsic_gso_shap_values2) / abs(hsic_gso_shap_values2).sum(axis=1, keepdims=True)
    hsic_gso_impfeatures_existence2 = convert_Scores_to_impfExistence(normalized_hsic_gso_shap_values2, Threshold)
    hsic_gso_TPR_mean2, hsic_gso_FDR_mean2, hsic_gso_TPR_std2, hsic_gso_FDR_std2, hsic_gso_TP2, hsic_gso_FD2 = performance_metric(hsic_gso_impfeatures_existence2, g_train)

    #HSICNetSparseMax
    sparsemax_model = HSICNetSparseMax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y)
    sparsemax_model.train_model(X_tensor , y_tensor)
    sigmas = sparsemax_model.sigmas
    sigma_y = sparsemax_model.sigma_y
    weights = sparsemax_model.importance_weights
    l_sp_shap_values, _ = sparsemax_model.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_tensor.shape[0], sigmas, sigma_y, weights)
    hsic_sp_shap_values = l_sp_shap_values.detach().cpu().numpy()
    hsic_sp_shap_ranks = create_rank(hsic_sp_shap_values.squeeze())
    hsic_sp_shap_avg_ranks = np.mean(hsic_sp_shap_ranks[:,feature_imp], axis=1)
    normalized_hsic_sp_shap_values= abs(hsic_sp_shap_values) / abs(hsic_sp_shap_values).sum(axis=1, keepdims=True)
    hsic_sp_impfeatures_existence = convert_Scores_to_impfExistence(normalized_hsic_sp_shap_values, Threshold)
    hsic_sp_TPR_mean, hsic_sp_FDR_mean, hsic_sp_TPR_std, hsic_sp_FDR_std, hsic_sp_TP, hsic_sp_FD  = performance_metric(hsic_sp_impfeatures_existence, g_train)


    #L2X #retrun feature importance
    L2X_scores = L2X(X, y, X, input_dim, num_sampling)
    L2X_ranks = create_rank(L2X_scores)
    L2x_avg_ranks = np.mean(L2X_ranks[:,feature_imp], axis=1)
    L2x_mean_rank = np.mean(L2x_avg_ranks)
    normalized_L2X_scores= abs(L2X_scores) / abs(L2X_scores).sum(axis=1, keepdims=True)
    L2x_impfeatures_existence = convert_Scores_to_impfExistence(normalized_L2X_scores, Threshold)
    L2x_TPR_mean, L2x_FDR_mean, L2x_TPR_std, L2x_FDR_std, L2x_TP, L2x_FD = performance_metric(L2x_impfeatures_existence, g_train)


    #L2X with 2*num_feature_imp
    L2X_scores2 = L2X(X, y, X, input_dim, 2*num_sampling)
    L2X_ranks2 = create_rank(L2X_scores2)
    L2x_avg_ranks2 = np.mean(L2X_ranks2[:,feature_imp], axis=1)
    normalized_L2X_scores2= abs(L2X_scores2) / abs(L2X_scores2).sum(axis=1, keepdims=True)
    L2x_impfeatures_existence2 = convert_Scores_to_impfExistence(normalized_L2X_scores2, Threshold)
    L2x_TPR_mean2, L2x_FDR_mean2, L2x_TPR_std2, L2x_FDR_std2, L2x_TP2, L2x_FD2 = performance_metric(L2x_impfeatures_existence2, g_train)

    #Invasive #retrun feature importance
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")

    # Initialize the custom model
    model = CustomModel(fn)
    Invase_explainer = INVASE (model, X_df, y_series, n_epoch=1000, prefit=False  # to train the model
                               )
    invase_scores =(Invase_explainer.explain(X_df)).to_numpy()                      
    # invase_model = InvaseFeatureImportance(n_epoch=1000)
    # invase_model.fit_model(X_df, y_series)
    # invase_scores = invase_model.compute_feature_importance(X_df)
    invase_rank = create_rank(invase_scores)
    invase_avg_ranks = np.mean(invase_rank[:,feature_imp], axis=1)
    invase_mean_rank = np.mean(invase_avg_ranks)
    normalized_invase_scores= abs(invase_scores) / abs(invase_scores).sum(axis=1, keepdims=True)
    invase_impfeatures_existence = convert_Scores_to_impfExistence(normalized_invase_scores, Threshold)
    invase_TPR_mean, invase_FDR_mean, invase_TPR_std, invase_FDR_std, invase_TP, invase_FD = performance_metric(invase_impfeatures_existence, g_train)

    ## SHAP
    explainer = shap.KernelExplainer(fn, X, l1_reg=False)
    shap_values = explainer.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    shap_ranks = create_rank(shap_values.squeeze())
    shap_avg_ranks = np.mean(shap_ranks[:,feature_imp], axis=1)
    shap_mean_rank = np.mean(shap_avg_ranks)
    normalized_shap_values= abs(shap_values) / abs(shap_values).sum(axis=1, keepdims=True)
    shap_impfeatures_existence = create_important_features_existence(shap_ranks, g_train)
    shap_TPR_mean, shap_FDR_mean, shap_TPR_std, shap_FDR_std, shap_TP, shap_FD = performance_metric(shap_impfeatures_existence, g_train)


    ## Sampling SHAP
    # sexplainer = shap.SamplingExplainer(fn, X, l1_reg=False)
    # sshap_values = sexplainer.shap_values(X, nsamples=X_sample_no, l1_reg=False, min_samples_per_feature=1)
    # sshap_ranks = create_rank(sshap_values.squeeze())
    # sshap_avg_ranks = np.mean(sshap_ranks[:,feature_imp], axis=1)
    # sshap_mean_rank = np.mean(sshap_avg_ranks)
    # sshap_TPR_mean, sshap_FDR_mean, sshap_TPR_std, sshap_FDR_std = performance_metric(sshap_values, g_train)



    # plt.boxplot([gem_avg_ranks, shap_avg_ranks, sshap_avg_ranks])
    ## Bivariate SHAP
    bishap = Bivariate_KernelExplainer(fn, X)
    bishap_values = bishap.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    bishap_ranks = create_rank(np.array(bishap_values).squeeze())
    bishap_avg_ranks = np.mean(bishap_ranks[:,feature_imp], axis=1)
    bishap_mean_rank = np.mean(bishap_avg_ranks)
    normalized_bishap_values= abs(bishap_values) / abs(bishap_values).sum(axis=1, keepdims=True)
    bishap_impfeatures_existence = create_important_features_existence(bishap_ranks, g_train)
    bishap_TPR_mean, bishap_FDR_mean, bishap_TPR_std, bishap_FDR_std, bishap_TP, bishap_FD = performance_metric(bishap_impfeatures_existence, g_train)



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


    lime_ranks = create_rank(lime_values)
    lime_avg_ranks = np.mean(lime_ranks[:,feature_imp], axis=1)
    lime_mean_rank = np.mean(lime_avg_ranks)
    normalized_lime_values= abs(lime_values) / abs(lime_values).sum(axis=1, keepdims=True)
    lime_impfeatures_existence = create_important_features_existence(lime_ranks, g_train)
    lime_TPR_mean, lime_FDR_mean, lime_TPR_std, lime_FDR_std, lime_TP, lime_FD = performance_metric(lime_impfeatures_existence, g_train)


    maple_ranks = create_rank(maple_values)
    maple_avg_ranks = np.mean(maple_ranks[:,feature_imp], axis=1)
    maple_mean_rank = np.mean(maple_avg_ranks)
    normalized_maple_values= abs(maple_values) / abs(maple_values).sum(axis=1, keepdims=True)
    maple_impfeatures_existence = create_important_features_existence(maple_ranks, g_train)
    maple_TPR_mean, maple_FDR_mean, maple_TPR_std, maple_FDR_std, maple_TP, maple_FD = performance_metric(maple_impfeatures_existence, g_train)

    ushap_ranks = create_rank(ushap_values)
    ushap_avg_ranks = np.mean(ushap_ranks[:,feature_imp], axis=1)
    ushap_mean_rank = np.mean(ushap_avg_ranks)
    normalized_ushap_values= abs(ushap_values) / abs(ushap_values).sum(axis=1, keepdims=True)
    ushap_impfeatures_existence = create_important_features_existence(ushap_ranks, g_train)
    ushap_TPR_mean, ushap_FDR_mean, ushap_TPR_std, ushap_FDR_std, ushap_TP, ushap_FD = performance_metric(ushap_impfeatures_existence, g_train)


    results = [hsic_gsp_shap_avg_ranks, hsic_gsp_shap_avg_ranks2, hsic_gso_shap_avg_ranks, hsic_gso_shap_avg_ranks2, hsic_sp_shap_avg_ranks, L2x_avg_ranks, L2x_avg_ranks2, invase_avg_ranks,  shap_avg_ranks, ushap_avg_ranks, bishap_avg_ranks, lime_avg_ranks, maple_avg_ranks]
    tpr = [hsic_gsp_TPR_mean, hsic_gsp_TPR_mean2, hsic_gso_TPR_mean, hsic_gso_TPR_mean2, hsic_sp_TPR_mean, L2x_TPR_mean, L2x_TPR_mean2, invase_TPR_mean, shap_TPR_mean,  ushap_TPR_mean, bishap_TPR_mean, lime_TPR_mean, maple_TPR_mean ]
    fdr = [hsic_gsp_FDR_mean, hsic_gsp_FDR_mean2, hsic_gso_FDR_mean, hsic_gso_FDR_mean2, hsic_sp_FDR_mean, L2x_FDR_mean, L2x_FDR_mean2,invase_FDR_mean, shap_FDR_mean,  ushap_FDR_mean, bishap_FDR_mean, lime_FDR_mean, maple_FDR_mean ]

    tpr_std = [hsic_gsp_TPR_std, hsic_gsp_TPR_std2, hsic_gso_TPR_std, hsic_gso_TPR_std2, hsic_sp_TPR_std, L2x_TPR_std, L2x_TPR_std2, invase_TPR_std, shap_TPR_std,  ushap_TPR_std, bishap_TPR_std, lime_TPR_std, maple_TPR_std ]
    fdr_std = [hsic_gsp_FDR_std, hsic_gsp_FDR_std2, hsic_gso_FDR_std, hsic_gso_FDR_std2, hsic_sp_FDR_std, L2x_FDR_std, L2x_FDR_std2, invase_FDR_std, shap_FDR_std,  ushap_FDR_std, bishap_FDR_std, lime_FDR_std, maple_FDR_std ]

    tp = [hsic_gsp_TP, hsic_gsp_TP2, hsic_gso_TP, hsic_gso_TP2, hsic_sp_TP, L2x_TP, L2x_TP2, invase_TP, shap_TP,  ushap_TP, bishap_TP, lime_TP, maple_TP]
    fp = [hsic_gsp_FD, hsic_gsp_FD2, hsic_gso_FD, hsic_gso_FD2, hsic_sp_FD, L2x_FD, L2x_FD2, invase_FD, shap_FD,  ushap_FD, bishap_FD, lime_FD, maple_FD]

    

    print('TPR mean: ' + str(np.round(hsic_gsp_TPR_mean,1)) + '\%, ' + 'TPR std: ' + str(np.round(hsic_gsp_TPR_std,1)) + '\%, '  )
    print('FDR mean: ' + str(np.round(hsic_gsp_FDR_mean,1)) + '\%, ' + 'FDR std: ' + str(np.round(hsic_gsp_FDR_std,1)) + '\%, '  )
    return results , tpr, fdr , tpr_std, fdr_std, tp, fp



if __name__=='__main__':

    
    num_samples = 400 # number of generated synthesized instances 
    input_dim = 10 # number of features for the synthesized instances
    hidden_dim1 = 100
    hidden_dim2 = 100
    X_sample_no = 200  # number of sampels for generating explanation
    train_seed = 42
    test_seed = 1
    Threshold = 0.001

   
    data_sets=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', 
            'Trigonometric Exponential', 'Exponential Hyperbolic', 'XOR', 'Syn4']
    ds_name = data_sets[1]
    # data_sets= ['Syn4']

    for ds_name in data_sets:

        # Generate synthetic data
        X_train, y_train, fn, feature_imp, g_train = generate_dataset(ds_name, num_samples, input_dim, train_seed)
        # X_test, y_test, fn, feature_imp, g_test = generate_dataset(ds_name, num_samples, input_dim, test_seed)
        
        all_results, tpr, fpr, tpr_std, fpr_std, TP, FP = Compare_methods(X_train, y_train, X_train, X_sample_no,  fn, feature_imp)


        method_names = ['Hsic_GumbelSparsemax', 'Hsic_GumbelSparsemax2', 'Hsic_GumbelSoftmax', 'Hsic_GumbelSoftmax2','Hsic_Sparsemax', 'L2X', 'L2X2', 'invasive', 'Kernel SHAP', 'Unbiased SHAP', 'Bivariate SHAP', 'LIME', 'MAPLE']
        folder_path = "results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Combine folder path and filename
        results_xsl = os.path.join(folder_path, f"results_{ds_name}_tr={Threshold}.xlsx")
        results_tpr_fpr = os.path.join(folder_path, f"tpr_fpr_{ds_name}_tr={Threshold}.xlsx")
        

        df = pd.DataFrame(all_results, index=method_names)

        # if not os.path.exists(results_xsl):
        #     with pd.ExcelWriter(results_xsl, mode='w', engine='openpyxl') as writer:
        #         df.to_excel(writer, sheet_name=ds_name, index_label='Method')
        # else:
        #     with pd.ExcelWriter(results_xsl, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
        #         df.to_excel(writer, sheet_name=ds_name, index_label='Method')
    
        if os.path.exists(results_xsl):
            os.remove(results_xsl)
        with pd.ExcelWriter(results_xsl, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=ds_name, index_label='Method')
    
        results_tpr_fpr_df = pd.DataFrame({
            
            'TPR': tpr,
            'TPR std': tpr_std,
            'FPR': fpr,
            'FPR std' : fpr_std,
            'TP' : TP,
            'FP': FP
        } , index=method_names
        )


        if os.path.exists(results_tpr_fpr):
            os.remove(results_tpr_fpr)
        with pd.ExcelWriter(results_tpr_fpr, mode='w', engine='openpyxl') as writer:
                results_tpr_fpr_df.to_excel(writer, sheet_name = ds_name, index_label='Method')
        
        print("done!")

    # plt.boxplot([shap_avg_ranks, bishap_avg_ranks, sshap_avg_ranks, ushap_avg_ranks, lime_avg_ranks, maple_avg_ranks])
    # plt.show()
    

