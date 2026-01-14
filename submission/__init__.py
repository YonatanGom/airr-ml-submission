"""
TCR Immune State Prediction - Ensemble Model

This package implements a hybrid stacking ensemble model for predicting
immune states from T-cell receptor (TCR) repertoire data.

6 Specialists:
- Physicochemical: Physics-based features (charge, size, ring, flexibility)
- Kmer: K-mer sequence features (k=4,5,6)
- VJGene: V/J gene features
- ReactiveTCR: Chi-squared based reactive TCR selection
- Statistical: Statistical features (entropy, moments)
- Frequency: Frequency features (V/J/AA distributions)

1 Head: LogisticRegressionCV with L1 regularization
"""

from .predictor import ImmuneStatePredictor

__version__ = "1.0.0"
__all__ = ["ImmuneStatePredictor"]