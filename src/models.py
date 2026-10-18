"""Model definitions and ensemble wrapper for Electricity Consumption Forecasting."""

import numpy as np


class WeightedEnsembleRegressor:
    """
    Weighted ensemble combining CatBoost, LightGBM, and XGBoost predictions.
    
    Provides standard scikit-learn compatible .predict() interface.
    """
    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights
        # Normalize weights
        total_w = sum(weights.values())
        self.weights = {k: v / total_w for k, v in weights.items()}
        
    def predict(self, X):
        final_preds = np.zeros(len(X), dtype=float)
        for name, model in self.models.items():
            w = self.weights[name]
            final_preds += w * np.asarray(model.predict(X), dtype=float)
        return final_preds
