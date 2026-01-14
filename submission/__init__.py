"""
TCR Immune State Prediction - Ensemble Model

This package implements a hybrid stacking ensemble model for predicting
immune states from T-cell receptor (TCR) repertoire data.

6 Specialists:
- Physicist: Physics-based features (charge, size, ring, flexibility)
- Sniper: K-mer sequence features (k=4,5,6)
- Ecologist: V/J gene features
- attTCR: Chi-squared based reactive TCR selection
- XGB-Stat: Statistical features (entropy, moments)
- XGB-Freq: Frequency features (V/J/AA distributions)

1 Head: LogisticRegressionCV with L1 regularization
"""

from .predictor import ImmuneStatePredictor

__version__ = "1.0.0"
__all__ = ["ImmuneStatePredictor"]