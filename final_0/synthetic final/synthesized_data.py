import numpy as np
import random
import matplotlib.pyplot as plt



data_names=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', 
            'Trigonometric Exponential', 'Exponential Hyperbolic', 'XOR', 'Syn4']

def generate_X(n_samples=100, n_features=10):
    # Generate samples with a standard normal distribution
    return np.random.randn(n_samples, n_features)

def generate_dataset_sinlog(n_samples=100, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Main effects
        main_effect_1 = np.sin(f1)  # Main effect from feature 1
        main_effect_2 = np.log1p(np.abs(f2))  # Main effect from feature 2

        # Complex interaction effect between feature 1 and feature 2
        interaction_effect = np.sin(f1 * f2) + np.exp(-((f1 - f2) ** 2))

        # Combine main effects and interaction effect
        y = main_effect_1 + main_effect_2 + interaction_effect
        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 2), 'Sine Log'

def generate_dataset_sin(n_samples=100, n_features=10, noise=0.1, seed=42):
    """
    Args:
        noise (float): Standard deviation of Gaussian noise to add to the output. 
    """
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Main effects: functions of each individual feature
        main_effects = np.sin(f1) + 0.5 * np.cos(f2)
        
        # Interaction term: product of two features (interaction between features 1 and 2)
        interaction = f1 * f2
        
        # Combine main effects and interaction to compute the true target values
        y_true = main_effects + interaction
        
        # Add Gaussian noise to the target values
        noise_array = noise * np.random.randn(X.shape[0])
        y = y_true + noise_array
        
        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 2), 'Sine Cosine'

def generate_dataset_poly_sine(n_samples=100, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Define the function using polynomial and sine terms
        y = f1**2 - 0.5 * f2**2 + np.sin(2 * np.pi * f1)
        
        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 2), 'Poly Sine'

def generate_dataset_squared_exponentials(n_samples=100, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        # Compute a function based on squared exponentials of the first 2 features
        y = np.exp(np.sum(X[:, :3]**2, axis=1) - 4.0)
        
        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 3), 'Squared Exponentials'

# These functions are for more than 3 features

def generate_dataset_complex_tanhsin(n_samples=1000, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2, f3 = X[:, 0], X[:, 1], X[:, 2]

        # Main effects
        main_effect_1 = np.tanh(f1)  # Hyperbolic tangent effect
        main_effect_2 = np.abs(f2)  # Absolute value effect

        # Interaction effects
        interaction_effect_1 = f1 * f2  # Multiplicative interaction
        interaction_effect_2 = np.sin(f1 + f3)  # Nonlinear interaction

        # Combine effects
        y = main_effect_1 + main_effect_2 + interaction_effect_1 + interaction_effect_2
        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 3), 'Tanh Sine'

def generate_dataset_complex_trig_exp(n_samples=100, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2, f3, f4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Complex non-linear interactions
        y = np.sin(f1) * np.exp(f2) + np.cos(f3 * f4) * np.tanh(f1 * f2)
        y += np.exp(-(f1**2 + f2**2)) * np.sin(f3 + f4)

        return y

    y = fn(X)

    return X, y, fn, np.arange(0, 4), 'Trigonometric Exponential'

def generate_dataset_complex_exponential_hyperbolic(n_samples=100, n_features=10, seed=42):
    
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2, f3, f4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Nested exponential and hyperbolic functions
        y = np.exp(f1) * np.tanh(f2 * f3) + np.exp(-np.abs(f4)) * np.tanh(f1 * f2)
        y += np.exp(f1 * f2) * np.sin(f3 * f4)

        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 4), 'Exponential Hyperbolic'

def generate_dataset_XOR(n_samples=100, n_features=10, seed=42):
   
    X = generate_X(n_samples, n_features)

    def fn(X):
        f1, f2, f3, f4, f5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        # Compute the target using an XOR-like interaction of features
        y = 0.5 * (np.exp(f1 * f2 * f3) + np.exp(f4 * f5))

        return y

    y = fn(X)
    
    return X, y, fn, np.arange(0, 5), 'XOR'

def generate_dataset_Syn4(n_samples=100, n_features=10, seed=42):
    X = generate_X(n_samples, n_features)
    def fn(X):
        f1, f2, f3, f4, f5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        logit1 = np.exp(X[:,0]*X[:,1])
        logit2 = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 

        # Based on X[:,10], combine two logits        
        idx1 = (X[:,9]< 0)*1
        idx2 = (X[:,9]>=0)*1

        y = logit1 * idx1 + logit2 * idx2
        return y
    y = fn(X)

    return X, y, fn, np.arange(0, 8), 'XOR'

    

def generate_dataset(data_name, n_samples=100, n_features=10, seed = 0):
    np.random.seed(seed)
    if data_name == data_names[0]:
         X, y, fn, feature_imp, _ = generate_dataset_sinlog(n_samples, n_features)
    if data_name == data_names[1]:
         X, y, fn, feature_imp, _ = generate_dataset_sin(n_samples, n_features)
    if data_name == data_names[2]:
         X, y, fn, feature_imp, _ = generate_dataset_poly_sine(n_samples, n_features)
    if data_name == data_names[3]:
         X, y, fn, feature_imp, _ = generate_dataset_squared_exponentials(n_samples, n_features)
    if data_name == data_names[4]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_tanhsin(n_samples, n_features)
    if data_name == data_names[5]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_trig_exp(n_samples, n_features)
    if data_name == data_names[6]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_exponential_hyperbolic(n_samples, n_features)
    if data_name == data_names[7]:
        X, y, fn, feature_imp, _ = generate_dataset_XOR(n_samples, n_features)
    if data_name == data_names[8]:
        X, y, fn, feature_imp, _ = generate_dataset_Syn4(n_samples, n_features)
    
    Ground_Truth = Ground_Truth_Generation(X, data_name)
    return X, y, fn, feature_imp, Ground_Truth


def Ground_Truth_Generation(X, data_name):

    # Number of samples and features
    n = len(X[:,0])
    d = len(X[0,:])

    # Output initialization
    out = np.zeros([n,d])
   
    # Index
    if (data_name in data_names[0:3]):        
        out[:,:2] = 1
    
    elif(data_name in data_names[3:5]):        
        out[:,:3] = 1
    
    elif(data_name in data_names[5:7]):        
        out[:,:4] = 1
    elif(data_name in data_names[7]):        
        out[:,:5] = 1
    if (data_name in ['Syn4','Syn5','Syn6']):        
        idx1 = np.where(X[:,9]< 0)[0]
        idx2 = np.where(X[:,9]>=0)[0]
        out[:,9] = 1      
        out[idx1,:2] = 1
        out[idx2,2:6] = 1
          
    return out