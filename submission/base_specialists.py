#!/usr/bin/env python3
"""
Base Specialists: Physicochemical, Kmer, VJGene

These are simple LogisticRegression classifiers with L1 regularization,
each trained on different feature types:
- Physicochemical: Physics features (charge, size, ring, flexibility patterns)
- Kmer: Sequence features (k-mers)
- VJGene: Gene features (V/J gene usage)

The only difference between them is the feature input and C value.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import sparse


# Default hyperparameters (from original model)
PHYSICOCHEMICAL_C = 0.15
KMER_C = 0.20
VJGENE_C = 0.50


def train_physicochemical(X_train, y_train, X_val, seed=42):
    """
    Train Physicochemical specialist on physics features.
    
    Args:
        X_train: Training physics features (sparse or dense)
        y_train: Training labels
        X_val: Validation physics features
        seed: Random seed
    
    Returns:
        val_preds: Predictions on validation set
        model: Tuple of (classifier, scaler) for later use
    """
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=PHYSICOCHEMICAL_C,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)
    
    X_val_scaled = scaler.transform(X_val)
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler)


def train_kmer(X_train, y_train, X_val, seed=42):
    """
    Train Kmer specialist on sequence k-mer features.
    
    Args:
        X_train: Training sequence features (sparse or dense)
        y_train: Training labels
        X_val: Validation sequence features
        seed: Random seed
    
    Returns:
        val_preds: Predictions on validation set
        model: Tuple of (classifier, scaler) for later use
    """
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=KMER_C,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)
    
    X_val_scaled = scaler.transform(X_val)
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler)


def train_vjgene(X_train, y_train, X_val, seed=42):
    """
    Train VJGene specialist on gene features.
    
    Args:
        X_train: Training gene features (sparse or dense)
        y_train: Training labels
        X_val: Validation gene features
        seed: Random seed
    
    Returns:
        val_preds: Predictions on validation set
        model: Tuple of (classifier, scaler) for later use
    """
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=VJGENE_C,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)
    
    X_val_scaled = scaler.transform(X_val)
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler)


def predict_with_model(model, X):
    """
    Predict using a trained model.
    
    Args:
        model: Tuple of (classifier, scaler)
        X: Features to predict on
    
    Returns:
        Probability predictions
    """
    clf, scaler = model
    X_scaled = scaler.transform(X)
    return clf.predict_proba(X_scaled)[:, 1]


def get_model_weights(model):
    """
    Get feature weights from a trained model.
    
    Args:
        model: Tuple of (classifier, scaler)
    
    Returns:
        Normalized weights (coefficients / scale)
    """
    clf, scaler = model
    return clf.coef_[0] / scaler.scale_


# ============================================================================
# Generic specialist function (alternative interface)
# ============================================================================

def train_logreg_specialist(X_train, y_train, X_val, C, seed=42):
    """
    Generic LogisticRegression specialist.
    
    Can be used as alternative to specific functions:
        train_logreg_specialist(X_phys, y, X_phys_val, C=0.15)  # = Physicochemical
        train_logreg_specialist(X_seq, y, X_seq_val, C=0.20)   # = Kmer
        train_logreg_specialist(X_gene, y, X_gene_val, C=0.50) # = VJGene
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        C: Regularization strength
        seed: Random seed
    
    Returns:
        val_preds: Predictions on validation set
        model: Tuple of (classifier, scaler)
    """
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)
    
    X_val_scaled = scaler.transform(X_val)
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler)