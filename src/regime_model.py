"""Two-Stage Regime-Switching Forecasting Module.

Stage 1: Multi-class regime classifier (Normal, Surge, Plunge).
Stage 2: Regime-specific regressors (Normal Model, Surge Model, Plunge Model).
Inference: Probability-weighted mixture and Hard-routing predictions.
"""

import numpy as np
import pandas as pd


class TwoStageRegimeForecaster:
    """
    Two-stage forecasting pipeline.
    
    Predicts probability of regime:
      Class 0: Normal  (|delta| <= 30)
      Class 1: Surge   (delta > 30)
      Class 2: Plunge  (delta < -30)
      
    Routes to specialized regressors.
    """
    def __init__(self, classifier, reg_normal, reg_surge, reg_plunge, min_val: float = 2.45):
        self.classifier = classifier
        self.reg_normal = reg_normal
        self.reg_surge = reg_surge
        self.reg_plunge = reg_plunge
        self.min_val = min_val
        
    def predict_components(self, X):
        """Return regime probabilities and individual regime predictions."""
        probs = self.classifier.predict_proba(X)
        pred_norm = self.reg_normal.predict(X)
        pred_surge = self.reg_surge.predict(X)
        pred_plunge = self.reg_plunge.predict(X)
        return probs, pred_norm, pred_surge, pred_plunge
        
    def predict(self, X, mode: str = "probability_weighted"):
        """
        Generate predictions using probability-weighted mixture or hard routing.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix at time t.
        mode : str
            'probability_weighted' or 'hard_routing'
        """
        probs, pred_n, pred_s, pred_p = self.predict_components(X)
        
        if mode == "probability_weighted":
            preds = probs[:, 0] * pred_n + probs[:, 1] * pred_s + probs[:, 2] * pred_p
        elif mode == "hard_routing":
            preds = np.zeros(len(X), dtype=float)
            for i in range(len(X)):
                best_cls = np.argmax(probs[i])
                if best_cls == 1:
                    preds[i] = pred_s[i]
                elif best_cls == 2:
                    preds[i] = pred_p[i]
                else:
                    preds[i] = pred_n[i]
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return np.clip(preds, self.min_val, None)
