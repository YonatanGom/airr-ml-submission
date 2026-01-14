#!/usr/bin/env python3
"""
attTCR Specialist: Chi-squared based TCR selection.

CRITICAL: Chi-squared scoring and reactive TCR selection happen
INSIDE each CV fold, using ONLY the training fold data (no data leakage).

This specialist:
1. Extracts TCR combinations (CDR3_VGENE) from repertoire files
2. Calculates chi-squared scores on training data only
3. Selects top reactive TCRs
4. Builds binary feature matrix
5. Trains LogisticRegression classifier
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc
import random

# --- CONFIG ---
NUM_REACTIVE_TCRS = 500
SEED = 42
N_CORES = min(8, cpu_count())


# ============================================================================
# TCR EXTRACTION FROM RAW FILES
# ============================================================================

def process_file_for_combinations(args):
    """
    Extract all TCR combinations from a single file.
    
    Args:
        args: Tuple of (filepath, patient_idx)
    
    Returns:
        Tuple of (patient_idx, dict of TCR combinations with counts)
    """
    filepath, patient_idx = args
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        combinations = defaultdict(int)
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            
            junction = str(junction) if pd.notna(junction) else ''
            v_call = str(v_call) if pd.notna(v_call) else ''
            
            if len(junction) < 4:
                continue
            
            # Create combination key (CDR3_VGENE)
            combo_key = f"{junction}_{v_call}"
            combinations[combo_key] += 1
        
        return patient_idx, dict(combinations)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return patient_idx, {}


def extract_tcrs_from_directory(data_dir: str, labels_dict: dict = None, n_cores: int = None):
    """
    Extract TCR combinations from all files in a directory.
    
    Args:
        data_dir: Path to directory with TSV files
        labels_dict: Optional dict mapping patient_id to label (for training)
        n_cores: Number of CPU cores
    
    Returns:
        Dictionary with:
            'patient_tcrs': dict mapping patient_idx -> sorted list of TCR keys
            'labels': numpy array of labels (or None for test)
            'patient_ids': list of patient IDs
            'n_patients': number of patients
    """
    if n_cores is None:
        n_cores = N_CORES
    
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.tsv")))
    
    n_patients = len(files)
    patient_ids = []
    labels = np.zeros(n_patients, dtype=np.int32) if labels_dict else None
    
    # Build tasks
    tasks = []
    for idx, f in enumerate(files):
        patient_id = f.stem
        patient_ids.append(patient_id)
        
        if labels_dict and patient_id in labels_dict:
            labels[idx] = 1 if labels_dict[patient_id] else 0
        
        tasks.append((f, idx))
    
    # Process files in parallel
    patient_tcrs = {}
    
    with Pool(n_cores) as p:
        results = list(tqdm(p.imap(process_file_for_combinations, tasks),
                           total=len(tasks), desc="   Extracting TCRs"))
    
    for patient_idx, combos in results:
        # Store as sorted list for deterministic iteration
        patient_tcrs[patient_idx] = sorted(combos.keys())
    
    del results
    gc.collect()
    
    return {
        'patient_tcrs': patient_tcrs,
        'labels': labels,
        'patient_ids': patient_ids,
        'n_patients': n_patients
    }


# ============================================================================
# CHI-SQUARED SCORING (NO DATA LEAKAGE)
# ============================================================================

def calculate_chi_squared_on_fold(patient_tcrs, labels, train_indices):
    """
    Calculate chi-squared scores using ONLY the training fold.
    This is the key to avoiding data leakage!
    
    Args:
        patient_tcrs: dict mapping patient_idx -> list of TCR keys
        labels: array of all labels
        train_indices: indices of patients in training fold
    
    Returns:
        scores: dict mapping TCR -> chi-squared score
    """
    # Get labels for train fold only
    train_labels = labels[train_indices]
    n_pos = np.sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_percent = n_pos / (n_pos + n_neg)
    
    # Count TCR occurrences in train fold only
    tcr_counts = defaultdict(lambda: {'total': 0, 'pos': 0})
    
    for local_idx, global_idx in enumerate(train_indices):
        tcrs = patient_tcrs.get(global_idx, [])
        is_pos = train_labels[local_idx] == 1
        
        for tcr in tcrs:
            tcr_counts[tcr]['total'] += 1
            if is_pos:
                tcr_counts[tcr]['pos'] += 1
    
    # Calculate chi-squared
    scores = {}
    for tcr in sorted(tcr_counts.keys()):  # Sorted for determinism
        counts = tcr_counts[tcr]
        n_with_tcr = counts['total']
        n_pos_with_tcr = counts['pos']
        
        if n_with_tcr < 2:
            continue
        
        expected = pos_percent * n_with_tcr
        if expected == 0:
            continue
        
        score = (n_pos_with_tcr - expected) ** 2 / expected
        if n_pos_with_tcr < expected:
            score = -score
        
        if abs(score) > 50:
            continue
        
        scores[tcr] = score
    
    return scores


def select_reactive_tcrs_from_scores(scores, n_reactive=NUM_REACTIVE_TCRS):
    """
    Select top N reactive TCRs by absolute chi-squared score.
    Uses controlled randomness for deterministic tie-breaking.
    
    Args:
        scores: dict mapping TCR -> chi-squared score
        n_reactive: number of TCRs to select
    
    Returns:
        dict of selected TCRs with their scores
    """
    # Controlled randomness for deterministic tie-breaking
    random.seed(43)
    
    items = list(scores.items())
    random.shuffle(items)  # Shuffle first for random tie-breaking
    sorted_scores = sorted(items, key=lambda x: -abs(x[1]))
    
    return dict(sorted_scores[:n_reactive])


# ============================================================================
# FEATURE MATRIX BUILDING
# ============================================================================

def build_feature_matrix_for_indices(patient_tcrs, reactive_tcrs, indices):
    """
    Build binary feature matrix for given patient indices.
    
    Args:
        patient_tcrs: dict mapping patient_idx -> list of TCR keys
        reactive_tcrs: dict of selected reactive TCRs
        indices: patient indices to include
    
    Returns:
        X: sparse matrix (len(indices), len(reactive_tcrs))
        tcr_list: list of TCR keys (column order)
    """
    # Sorted for deterministic column order
    tcr_list = sorted(reactive_tcrs.keys())
    tcr_to_idx = {tcr: i for i, tcr in enumerate(tcr_list)}
    n_features = len(tcr_list)
    n_samples = len(indices)
    
    data, row, col = [], [], []
    
    for local_idx, global_idx in enumerate(indices):
        tcrs = patient_tcrs.get(global_idx, [])
        for tcr in tcrs:
            if tcr in tcr_to_idx:
                data.append(1)
                row.append(local_idx)
                col.append(tcr_to_idx[tcr])
    
    X = sparse.csr_matrix((data, (row, col)), shape=(n_samples, n_features), dtype=np.float32)
    return X, tcr_list


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_atttcr_fold(patient_tcrs, labels, train_indices, val_indices,
                      n_reactive=NUM_REACTIVE_TCRS, seed=SEED):
    """
    Train attTCR on a single fold with NO data leakage.
    
    1. Calculate chi-squared on train_indices ONLY
    2. Select reactive TCRs from train fold ONLY
    3. Build features for train and val using those TCRs
    4. Train classifier on train, predict on val
    
    Args:
        patient_tcrs: dict mapping patient_idx -> list of TCR keys
        labels: array of all labels
        train_indices: indices for training
        val_indices: indices for validation
        n_reactive: number of reactive TCRs to select
        seed: random seed
    
    Returns:
        val_preds: predictions for validation set
        model: tuple of (clf, scaler, reactive_tcrs, tcr_list)
        scores: chi-squared scores dict
    """
    # Step 1: Chi-squared on TRAIN ONLY
    scores = calculate_chi_squared_on_fold(patient_tcrs, labels, train_indices)
    
    # Step 2: Select reactive TCRs from TRAIN ONLY
    reactive_tcrs = select_reactive_tcrs_from_scores(scores, n_reactive)
    
    if len(reactive_tcrs) == 0:
        return None, None, None
    
    # Step 3: Build feature matrices
    X_train, tcr_list = build_feature_matrix_for_indices(patient_tcrs, reactive_tcrs, train_indices)
    X_val, _ = build_feature_matrix_for_indices(patient_tcrs, reactive_tcrs, val_indices)
    
    y_train = labels[train_indices]
    
    # Step 4: Train classifier
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=0.30,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)
    
    # Predict on validation
    X_val_scaled = scaler.transform(X_val)
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler, reactive_tcrs, tcr_list), scores


def train_atttcr_full(patient_tcrs, labels, n_reactive=NUM_REACTIVE_TCRS, seed=SEED):
    """
    Train attTCR on full dataset (for test prediction).
    
    Args:
        patient_tcrs: dict mapping patient_idx -> list of TCR keys
        labels: array of labels
        n_reactive: number of reactive TCRs
        seed: random seed
    
    Returns:
        model: tuple of (clf, scaler, reactive_tcrs, tcr_list)
        scores: chi-squared scores dict
    """
    all_indices = np.arange(len(labels))
    
    # Chi-squared on ALL training data
    scores = calculate_chi_squared_on_fold(patient_tcrs, labels, all_indices)
    reactive_tcrs = select_reactive_tcrs_from_scores(scores, n_reactive)
    
    if len(reactive_tcrs) == 0:
        return None, None
    
    # Build features
    X_train, tcr_list = build_feature_matrix_for_indices(patient_tcrs, reactive_tcrs, all_indices)
    
    # Train
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(
        penalty='l1',
        C=0.30,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed,
        max_iter=1000
    )
    
    clf.fit(scaler.fit_transform(X_train), labels)
    
    return (clf, scaler, reactive_tcrs, tcr_list), scores


def predict_atttcr_test(model, test_patient_tcrs, test_indices=None):
    """
    Predict on test data using trained model.
    
    Args:
        model: tuple of (clf, scaler, reactive_tcrs, tcr_list)
        test_patient_tcrs: dict mapping patient_idx -> list of TCR keys
        test_indices: optional indices to predict on (default: all)
    
    Returns:
        predictions: probability predictions
    """
    clf, scaler, reactive_tcrs, tcr_list = model
    
    if test_indices is None:
        test_indices = np.arange(len(test_patient_tcrs))
    
    X_test, _ = build_feature_matrix_for_indices(test_patient_tcrs, reactive_tcrs, test_indices)
    X_test_scaled = scaler.transform(X_test)
    
    return clf.predict_proba(X_test_scaled)[:, 1]


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

def extract_and_train_atttcr(train_dir: str, labels_dict: dict, n_cores: int = None,
                              n_reactive: int = NUM_REACTIVE_TCRS, seed: int = SEED):
    """
    Extract TCRs and train attTCR model on full training data.
    
    Args:
        train_dir: Path to training directory
        labels_dict: dict mapping patient_id to label
        n_cores: number of CPU cores
        n_reactive: number of reactive TCRs to select
        seed: random seed
    
    Returns:
        model: trained model tuple
        train_data: extracted TCR data dict
        scores: chi-squared scores
    """
    # Extract TCRs
    train_data = extract_tcrs_from_directory(train_dir, labels_dict, n_cores)
    
    if train_data['labels'] is None or len(train_data['labels']) == 0:
        return None, None, None
    
    # Train model
    model, scores = train_atttcr_full(
        train_data['patient_tcrs'],
        train_data['labels'],
        n_reactive=n_reactive,
        seed=seed
    )
    
    return model, train_data, scores


def predict_atttcr_on_test_dir(model, test_dir: str, n_cores: int = None):
    """
    Predict on test directory using trained model.
    
    Args:
        model: trained model tuple
        test_dir: path to test directory
        n_cores: number of CPU cores
    
    Returns:
        predictions: array of predictions
        patient_ids: list of patient IDs
    """
    if model is None:
        return None, None
    
    # Extract test TCRs
    test_data = extract_tcrs_from_directory(test_dir, labels_dict=None, n_cores=n_cores)
    
    # Predict
    predictions = predict_atttcr_test(model, test_data['patient_tcrs'])
    
    return predictions, test_data['patient_ids']