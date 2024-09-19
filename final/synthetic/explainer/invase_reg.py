import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from invase import INVASE

# Assuming INVASE is correctly imported or defined
# class INVASE: 
#    # INVASE implementation or import here

class InvaseFeatureImportance:
    def __init__(self, n_epoch=1000):
        self.n_epoch = n_epoch
        self.model = None
        self.explainer = None

    def train_model(self, X_df, y_series) -> None:
        # Convert NumPy arrays to Pandas DataFrame and Series
        # feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        # X_df = pd.DataFrame(X, columns=feature_names)
        # y_series = pd.Series(y, name="Target")

        # Fit a regression model (Linear Regression in this case)
        self.model = LinearRegression()
        self.model.fit(X_df, y_series)

        # Store the training data for later use in the INVASE explainer
        self.X_df = X_df
        self.y_series = y_series

       
    def compute_feature_importance(self, Xt_df):
        """
        Returns:
        - feature_importance: np.ndarray : Feature importance scores as a NumPy array.
        """
        # Instantiate and fit INVASE explainer
        self.explainer = INVASE(
            self.model, 
            self.X_df, 
            self.y_series, 
            n_epoch=self.n_epoch, 
            prefit=True  # The model is already trained
        )

        # Explain the feature importance for the entire dataset
        explanation = self.explainer.explain(Xt_df)

        return explanation.to_numpy()
