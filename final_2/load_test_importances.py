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
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch



def plot_feature_importances_matrix(feature_importances, method_names, num_instances=10):
    """
    Plots feature importance matrices for each method for the selected instances.

    Parameters:
        feature_importances (list of tensors): A list of feature importance tensors 
                                               (one for each method, shape: [n_instances, n_features]).
        num_instances (int): Number of instances to randomly select for visualization.

    Returns:
        None (plots the heatmaps for visual comparison).
    """
    num_methods = len(feature_importances)  # Number of methods (should be 12)
    num_features = feature_importances[0].shape[1]  # Number of features (columns in each tensor)

    # Randomly select `num_instances` from the dataset
    random_indices = random.sample(range(feature_importances[0].shape[0]), k=num_instances)

    # Create one heatmap per method
    for method_idx, method_feature_importance in enumerate(feature_importances):
       
        method_feature_importance = np.array(method_feature_importance)

        # Extract specific rows (instances) for the selected indices
        selected_importances = method_feature_importance[random_indices, :]
    

        # Plot the heatmap
        plt.figure(figsize=(8, 6)) 
        plt.imshow(selected_importances, aspect='auto', cmap='coolwarm', interpolation='none')
        plt.title(f"Feature Importances | {method_names[method_idx]}", fontsize=12)
        plt.xlabel("Features", fontsize=10)
        plt.ylabel("Instances", fontsize=10)
        plt.colorbar(label="Importance")
        plt.xticks(ticks=np.arange(num_features), labels=[f"F{i+1}" for i in range(num_features)])
        plt.yticks(ticks=np.arange(num_instances), labels=[f"#{i+1}" for i in range(num_instances)])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch


def consistency_across_methods(feature_matrix, feature_importances_list, method_names, n_clusters=5):
    """
    Evaluate the consistency of feature importance values across similar (clustered) instances
    for multiple attribution methods.

    Parameters:
        feature_matrix (numpy.ndarray): Test data feature matrix of shape [n_instances, n_features].
                                         This represents input instances in feature space.
        feature_importances_list (list of numpy.ndarray): A list of feature importance matrices.
                                                          Each matrix corresponds to one method and has shape [n_instances, n_features].
        method_names (list of str): Names of the methods corresponding to the feature importance matrices.
        n_clusters (int): Number of clusters to group similar instances in feature space.

    Returns:
        consistency_scores (dict): A dictionary with method names as keys and their consistency scores
                                    (list of float scores for each cluster) as values.
    """
    feature_matrix = np.array(feature_matrix)  # Ensure feature_matrix is a NumPy array
    consistency_scores = {}

    for method_idx, (method_name, feature_importances) in enumerate(zip(method_names, feature_importances_list)):
        print(f"Evaluating consistency for method: {method_name}...")
        feature_importances = np.array(feature_importances)  # Ensure importance matrix is a NumPy array

        # Perform clustering using K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Measure consistency within each cluster
        cluster_consistency = []

        for cluster_idx in range(n_clusters):
            # Indices of instances in the current cluster
            cluster_indices = np.where(cluster_labels == cluster_idx)[0]
            cluster_features = feature_matrix[cluster_indices]  # Features of instances in the cluster
            cluster_importances = feature_importances[cluster_indices]  # Importances of instances in the cluster

            # If the cluster has just one element, skip it
            if len(cluster_indices) <= 1:
                cluster_consistency.append(0)  # Small cluster, by definition consistent
                continue

            # 1. Compute pairwise distances between instances in the cluster (feature space)
            pairwise_feature_distances = pairwise_distances(cluster_features, metric='euclidean')

            # 2. Compute pairwise distances between feature importance vectors (importance space)
            pairwise_importance_distances = pairwise_distances(cluster_importances, metric='euclidean')

            # 3. Normalize importance distances by feature space distances (consistency measure)
            # Avoid division by zero in distances
            normalized_distances = np.divide(
                pairwise_importance_distances,
                pairwise_feature_distances + 1e-8  # Small epsilon to prevent division by zero
            )

            # Use mean normalized pairwise distance as the consistency score for this cluster
            cluster_consistency_score = np.mean(normalized_distances)
            cluster_consistency.append(cluster_consistency_score)

        # Store the consistency results for the current method
        consistency_scores[method_name] = cluster_consistency

    return consistency_scores



   
    

if __name__ == "__main__":
   
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

    with open('feature_importances.pkl', 'rb') as f:
        feature_importances = pickle.load(f)

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
    datasets = [diabetes()]
    for data in datasets:
        
        # Loading data
        X, y, db_name, mode = data
        print(db_name)
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sampleNo_tbx = 200
        indices = np.random.choice(X_test.shape[0], size=min(sampleNo_tbx, X_test.shape[0]), replace=False)
        X_tbx = X_test[indices,:]
  
    # Plot feature importances for 10 randomly selected instances
    plot_feature_importances_matrix(feature_importances, method_names, num_instances=10)
   ##-------------------
   #2nd experiment
    # Evaluate consistency across methods
    n_clusters = 6  # Number of clusters to group similar instances
    consistency_scores = consistency_across_methods(X_test, feature_importances, method_names, n_clusters)

    # Output consistency scores for each method
    print("\nConsistency Scores (Lower is better):")
    for method_name, scores in consistency_scores.items():
        print(f"Method: {method_name}")
        for cluster_idx, score in enumerate(scores):
            print(f"  Cluster {cluster_idx + 1}: Consistency Score = {score:.4f}")
