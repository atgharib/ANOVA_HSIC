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
from hsic_anova import *
from explainer.L2x_reg import *
from explainer.invase_reg import InvaseFeatureImportance

results_xsl = Path('synthesized_results.xlsx')
results_tpr_fpr = Path('tpr_fpr_results.xlsx')

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

def performance_metric(score, g_truth):

    n = len(score)
    TPR = np.zeros([n,])
    FDR = np.zeros([n,])
    
    for i in range(n):

        # TPR    
        TP_vals = np.sum(score[i,:] * g_truth[i,:])
        TPR_den = np.sum(g_truth[i,:])
        TPR[i] = 100 * float(TP_vals)/float(TPR_den)
    
        # FDR
        FD_vals = np.sum(score[i,:] * (1-g_truth[i,:]))
        FDR_den = np.sum(score[i,:])
        FDR[i] = 100 * float(FD_vals)/float(FDR_den)

    return np.mean(TPR), np.mean(FDR), np.std(TPR), np.std(FDR)
    

def Compare_methods(X, y, X_test, X_sample_no, fn, feature_imp):

    #HSIC_anova
    X_tensor = torch.from_numpy(X).float()  # Convert to float tensor
    y_tensor = torch.from_numpy(y).float()
    X_tensor_test = torch.from_numpy(X_test).float() 
    sigma_init_X = initialize_sigma_median_heuristic(X_tensor)
    sigma_init_Y = initialize_sigma_y_median_heuristic(y_tensor)

    # gumbelsoftmax_model = HSICNetGumbelSoftmax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y)
    #train_model(gumbelsoftmax_model, X, y)
    gumbelsparsemax_model = HSICNetGumbelSparsemax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y)
    gumbelsparsemax_model.train_model(X_tensor, y_tensor)
    # sparsemax_model = HSICNetSparseMax(input_dim, hidden_dim1, hidden_dim2, sigma_init_X, sigma_init_Y)
    #train_model(sparsemax_model, X, y)

    sigmas = gumbelsparsemax_model.sigmas
    sigma_y = gumbelsparsemax_model.sigma_y
    weights = gumbelsparsemax_model.importance_weights
    l_shap_values, _ = gumbelsparsemax_model.instancewise_shapley_value(X_tensor, y_tensor, X_tensor, y_tensor,X_sample_no, sigmas, sigma_y, weights)
    hsic_shap_values = l_shap_values.detach().cpu().numpy()
    hsic_shap_ranks = create_rank(hsic_shap_values.squeeze())
    hsic_shap_avg_ranks = np.mean(hsic_shap_ranks[:,feature_imp], axis=1)
    hsic_shap_mean_rank = np.mean(hsic_shap_avg_ranks)
    # hsic_TPR_mean, hsic_FDR_mean, hsic_TPR_std, hsic_FDR_std = performance_metric(hsic_shap_values, g_test)
    hsic_TPR_mean, hsic_FDR_mean, hsic_TPR_std, hsic_FDR_std = performance_metric(hsic_shap_values, g_train)

    #L2X
    L2X_scores = L2X(X, y, X, input_dim, len(feature_imp))
    L2X_ranks = create_rank(L2X_scores)
    L2x_avg_ranks = np.mean(L2X_ranks[:,feature_imp], axis=1)
    L2x_mean_rank = np.mean(L2x_avg_ranks)
    L2x_TPR_mean, L2x_FDR_mean, L2x_TPR_std, L2x_FDR_std = performance_metric(L2X_scores, g_train)

    #Invasive
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")
    invase_model = InvaseFeatureImportance(n_epoch=1000)
    invase_model.fit_model(X_df, y_series)
    invase_scores = invase_model.compute_feature_importance(X_df)
    invase_rank = create_rank(invase_scores)
    invase_avg_ranks = np.mean(invase_rank[:,feature_imp], axis=1)
    invase_mean_rank = np.mean(invase_avg_ranks)
    invase_TPR_mean, invase_FDR_mean, invase_TPR_std, invase_FDR_std = performance_metric(invase_scores, g_train)

    ## SHAP
    explainer = shap.KernelExplainer(fn, X, l1_reg=False)
    shap_values = explainer.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    shap_ranks = create_rank(shap_values.squeeze())
    shap_avg_ranks = np.mean(shap_ranks[:,feature_imp], axis=1)
    shap_mean_rank = np.mean(shap_avg_ranks)
    shap_TPR_mean, shap_FDR_mean, shap_TPR_std, shap_FDR_std = performance_metric(shap_values, g_train)


    ## Sampling SHAP
    sexplainer = shap.SamplingExplainer(fn, X, l1_reg=False)
    sshap_values = sexplainer.shap_values(X, nsamples=X_sample_no, l1_reg=False, min_samples_per_feature=1)
    sshap_ranks = create_rank(sshap_values.squeeze())
    sshap_avg_ranks = np.mean(sshap_ranks[:,feature_imp], axis=1)
    sshap_mean_rank = np.mean(sshap_avg_ranks)
    sshap_TPR_mean, sshap_FDR_mean, sshap_TPR_std, sshap_FDR_std = performance_metric(sshap_values, g_train)



    # plt.boxplot([gem_avg_ranks, shap_avg_ranks, sshap_avg_ranks])
    ## Bivariate SHAP
    bishap = Bivariate_KernelExplainer(fn, X)
    bishap_values = bishap.shap_values(X, nsamples=X_sample_no, l1_reg=False)
    bishap_ranks = create_rank(np.array(bishap_values).squeeze())
    bishap_avg_ranks = np.mean(bishap_ranks[:,feature_imp], axis=1)
    bishap_mean_rank = np.mean(bishap_avg_ranks)
    bishap_TPR_mean, bishap_FDR_mean, bishap_TPR_std, bishap_FDR_std = performance_metric(bishap_values, g_train)



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
    lime_TPR_mean, lime_FDR_mean, lime_TPR_std, lime_FDR_std = performance_metric(lime_values, g_train)


    maple_ranks = create_rank(maple_values)
    maple_avg_ranks = np.mean(maple_ranks[:,feature_imp], axis=1)
    maple_mean_rank = np.mean(maple_avg_ranks)
    maple_TPR_mean, maple_FDR_mean, maple_TPR_std, maple_FDR_std = performance_metric(maple_values, g_train)

    ushap_ranks = create_rank(ushap_values)
    ushap_avg_ranks = np.mean(ushap_ranks[:,feature_imp], axis=1)
    ushap_mean_rank = np.mean(ushap_avg_ranks)
    ushap_TPR_mean, ushap_FDR_mean, ushap_TPR_std, ushap_FDR_std = performance_metric(ushap_values, g_train)


    results = [hsic_shap_avg_ranks, L2x_avg_ranks, invase_avg_ranks,  shap_avg_ranks, sshap_avg_ranks, ushap_avg_ranks, bishap_avg_ranks, lime_avg_ranks, maple_avg_ranks]
    tpr = [hsic_TPR_mean, L2x_TPR_mean, invase_TPR_mean, shap_TPR_mean, sshap_TPR_mean, ushap_TPR_mean, bishap_TPR_mean, lime_TPR_mean, maple_TPR_mean ]
    fdr = [hsic_FDR_mean, L2x_FDR_mean, invase_FDR_mean, shap_FDR_mean, sshap_FDR_mean, ushap_FDR_mean, bishap_FDR_mean, lime_FDR_mean, maple_FDR_mean ]

    print('TPR mean: ' + str(np.round(hsic_TPR_mean,1)) + '\%, ' + 'TPR std: ' + str(np.round(hsic_TPR_std,1)) + '\%, '  )
    print('FDR mean: ' + str(np.round(hsic_FDR_mean,1)) + '\%, ' + 'FDR std: ' + str(np.round(hsic_FDR_std,1)) + '\%, '  )
    return results , tpr, fdr



if __name__=='__main__':

    X_sample_no = 100  # number of sampels for generating explanation
    smaple_tbX = 100   # number of samples to be explained
    num_samples = 100 # number of generated synthesized instances 
    input_dim = 4 # number of features for the synthesized instances
    hidden_dim1 = 100
    hidden_dim2 = 100
    train_seed = 42
    test_seed = 1

    data_sets=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', 
            'Trigonometric Exponential', 'Exponential Hyperbolic', 'XOR']
    ds_name = data_sets[1]

    # Generate synthetic data
    X_train, y_train, fn, feature_imp, g_train = generate_dataset(ds_name, num_samples, input_dim, train_seed)
    # X_test, y_test, fn, feature_imp, g_test = generate_dataset(ds_name, num_samples, input_dim, test_seed)
    
    all_results, tpr, fpr = Compare_methods(X_train, y_train, X_train, X_sample_no,  fn, feature_imp)


    method_names = ['Hsic', 'L2X', 'invasive', 'Kernel SHAP', 'Sampling SHAP', 'Unbiased SHAP', 'Bivariate SHAP', 'LIME', 'MAPLE']

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
        'FPR': fpr,
        'TPR': tpr,
    } , index=method_names
    )


    if os.path.exists(results_tpr_fpr):
        os.remove(results_tpr_fpr)
    with pd.ExcelWriter(results_tpr_fpr, mode='w', engine='openpyxl') as writer:
            results_tpr_fpr_df.to_excel(writer, sheet_name = ds_name, index_label='Method')
    
    print("done!")

    # plt.boxplot([shap_avg_ranks, bishap_avg_ranks, sshap_avg_ranks, ushap_avg_ranks, lime_avg_ranks, maple_avg_ranks])
    # plt.show()
    

