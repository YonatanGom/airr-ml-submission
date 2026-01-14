#!/usr/bin/env python3
"""
XGBoost Specialist: Statistical & Frequency Features

Two specialists in one:
- Statistical: Statistical features (entropy, gini, moments)
- Frequency: Frequency features (V/J/AA/length distributions)

CRITICAL: Feature selection happens INSIDE each CV fold using only
the training fold data (no data leakage).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import gc
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
SEED = 42
N_CORES = min(8, cpu_count())

# CDR3 position and length constraints
CDR3_POS_START = 3
CDR3_POS_END = 20
CDR3_LEN_MIN = 8
CDR3_LEN_MAX = 29

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_SET = set(AMINO_ACIDS)

# Dataset-specific hyperparameters
XGB_STAT_PARAMS = {
    1: {'min_effect': 0.2, 'max_features': 25, 'C': 0.1},
    2: {'min_effect': 0.1, 'max_features': 25, 'C': 0.1},
    3: {'min_effect': 0.2, 'max_features': 25, 'C': 0.1},
    4: {'min_effect': 0.1, 'max_features': 25, 'C': 0.1},
    5: {'min_effect': 0.3, 'max_features': 25, 'C': 0.1},
    6: {'min_effect': 0.3, 'max_features': 25, 'C': 0.1},
    7: {'min_effect': 0.2, 'max_features': 25, 'C': 0.01},
    8: {'min_effect': 0.1, 'max_features': 100, 'C': 0.01},
}

XGB_FREQ_PARAMS = {
    1: {'min_effect': 0.2, 'max_features': 25, 'C': 0.1},
    2: {'min_effect': 0.1, 'max_features': 50, 'C': 0.1},
    3: {'min_effect': 0.2, 'max_features': 25, 'C': 0.1},
    4: {'min_effect': 0.1, 'max_features': 100, 'C': 0.1},
    5: {'min_effect': 0.3, 'max_features': 25, 'C': 0.001},
    6: {'min_effect': 0.3, 'max_features': 25, 'C': 0.1},
    7: {'min_effect': 0.2, 'max_features': 25, 'C': 0.1},
    8: {'min_effect': 0.1, 'max_features': 100, 'C': 0.01},
}

# Default params for unknown datasets
DEFAULT_PARAMS = {'min_effect': 0.2, 'max_features': 25, 'C': 0.1}


# ============================================================================
# GENE HANDLING
# ============================================================================

def get_v_gene(v_call):
    """Get V-gene, removing only allele info."""
    if not v_call or pd.isna(v_call) or str(v_call).lower() in ['nan', '', 'none']:
        return None
    v = str(v_call).strip()
    v = re.sub(r'\*\d+', '', v)
    return v if v else None


def get_j_gene(j_call):
    """Get J-gene, removing only allele info."""
    if not j_call or pd.isna(j_call) or str(j_call).lower() in ['nan', '', 'none']:
        return None
    j = str(j_call).strip()
    j = re.sub(r'\*\d+', '', j)
    return j if j else None


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def compute_entropy(counts):
    """Shannon entropy."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def compute_gini_simpson(counts):
    """Gini-Simpson diversity."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((c/total)**2 for c in counts.values())


# ============================================================================
# RAW DATA EXTRACTION (per patient file)
# ============================================================================

def extract_raw_from_file(filepath):
    """
    Extract raw counts from a single patient file.
    
    Returns dict with v_counts, j_counts, length_counts, aa_counts, etc.
    """
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        v_counts = Counter()
        j_counts = Counter()
        length_counts = Counter()
        global_aa_counts = Counter()
        aa_pos_counts = defaultdict(Counter)
        cdr3_lengths = []
        n_valid = 0
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            junction = str(junction) if pd.notna(junction) else ''
            
            if len(junction) < CDR3_LEN_MIN or len(junction) > CDR3_LEN_MAX:
                continue
            if not all(c in AA_SET for c in junction):
                continue
            
            n_valid += 1
            length_counts[len(junction)] += 1
            cdr3_lengths.append(len(junction))
            
            v_gene = get_v_gene(v_call)
            if v_gene:
                v_counts[v_gene] += 1
            
            j_gene = get_j_gene(j_call)
            if j_gene:
                j_counts[j_gene] += 1
            
            for aa in junction:
                global_aa_counts[aa] += 1
            
            for pos in range(CDR3_POS_START, min(len(junction), CDR3_POS_END)):
                aa_pos_counts[pos][junction[pos]] += 1
        
        return {
            'v_counts': v_counts,
            'j_counts': j_counts,
            'length_counts': length_counts,
            'global_aa_counts': global_aa_counts,
            'aa_pos_counts': dict(aa_pos_counts),
            'cdr3_lengths': cdr3_lengths,
            'n_valid': n_valid
        }
    except Exception as e:
        print(f"Error in extract_raw_from_file: {e}")
        return None


def extract_raw_wrapper(args):
    """Wrapper for parallel processing."""
    filepath, idx = args
    raw = extract_raw_from_file(filepath)
    return idx, raw


# ============================================================================
# GENE DISCOVERY
# ============================================================================

def discover_genes_from_file(filepath):
    """Discover V and J genes in a file."""
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        v_genes = set()
        j_genes = set()
        
        for row in df.itertuples(index=False):
            v_gene = get_v_gene(getattr(row, 'v_call', None))
            j_gene = get_j_gene(getattr(row, 'j_call', None))
            if v_gene:
                v_genes.add(v_gene)
            if j_gene:
                j_genes.add(j_gene)
        return v_genes, j_genes
    except:
        return set(), set()


def discover_genes_from_files(files, n_cores=None):
    """Discover all V and J genes from multiple files."""
    if n_cores is None:
        n_cores = N_CORES
    
    all_v_genes = set()
    all_j_genes = set()
    
    with Pool(n_cores) as p:
        results = list(tqdm(p.imap(discover_genes_from_file, files),
                           total=len(files), desc="   Discovering genes"))
    
    for v_genes, j_genes in results:
        all_v_genes.update(v_genes)
        all_j_genes.update(j_genes)
    
    return sorted(all_v_genes), sorted(all_j_genes)


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_statistical_features(raw):
    """
    Compute statistical features (entropy, moments).
    
    Returns dict of feature_name -> value
    """
    features = {}
    n_total = raw['n_valid']
    if n_total == 0:
        return None
    
    # Length moments
    lengths = raw['cdr3_lengths']
    if len(lengths) >= 10:
        arr = np.array(lengths, dtype=np.float64)
        features['stat_len_mean'] = np.mean(arr)
        features['stat_len_std'] = np.std(arr)
        skew_val = scipy_stats.skew(arr)
        features['stat_len_skew'] = skew_val if not np.isnan(skew_val) else 0
        kurt_val = scipy_stats.kurtosis(arr)
        features['stat_len_kurt'] = kurt_val if not np.isnan(kurt_val) else 0
        features['stat_len_median'] = np.median(arr)
        features['stat_len_iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
    else:
        for k in ['stat_len_mean', 'stat_len_std', 'stat_len_skew', 'stat_len_kurt', 'stat_len_median', 'stat_len_iqr']:
            features[k] = 0
    
    # V/J entropy
    features['stat_v_entropy'] = compute_entropy(raw['v_counts'])
    features['stat_v_gini'] = compute_gini_simpson(raw['v_counts'])
    features['stat_v_nunique'] = len(raw['v_counts'])
    features['stat_j_entropy'] = compute_entropy(raw['j_counts'])
    features['stat_j_gini'] = compute_gini_simpson(raw['j_counts'])
    features['stat_j_nunique'] = len(raw['j_counts'])
    
    # Global AA entropy
    features['stat_aa_global_entropy'] = compute_entropy(raw['global_aa_counts'])
    features['stat_aa_global_gini'] = compute_gini_simpson(raw['global_aa_counts'])
    
    # Position-specific AA entropy
    for pos in range(CDR3_POS_START, CDR3_POS_END):
        if pos in raw['aa_pos_counts']:
            features[f'stat_aa_ent_pos{pos}'] = compute_entropy(raw['aa_pos_counts'][pos])
            features[f'stat_aa_gini_pos{pos}'] = compute_gini_simpson(raw['aa_pos_counts'][pos])
        else:
            features[f'stat_aa_ent_pos{pos}'] = 0
            features[f'stat_aa_gini_pos{pos}'] = 0
    
    features['stat_log_size'] = np.log1p(n_total)
    
    return features


def compute_frequency_features(raw, all_v_genes, all_j_genes):
    """
    Compute frequency features.
    
    Returns dict of feature_name -> value
    """
    features = {}
    n_total = raw['n_valid']
    if n_total == 0:
        return None
    
    # Length frequencies
    for length in range(CDR3_LEN_MIN, CDR3_LEN_MAX + 1):
        features[f'freq_len_{length}'] = raw['length_counts'].get(length, 0) / n_total
    
    # V-gene frequencies
    v_total = sum(raw['v_counts'].values())
    for v_gene in all_v_genes:
        freq = raw['v_counts'].get(v_gene, 0) / v_total if v_total > 0 else 0
        safe_name = v_gene.replace('-', '_').replace('/', '_')
        features[f'freq_v_{safe_name}'] = freq
    
    # J-gene frequencies
    j_total = sum(raw['j_counts'].values())
    for j_gene in all_j_genes:
        freq = raw['j_counts'].get(j_gene, 0) / j_total if j_total > 0 else 0
        safe_name = j_gene.replace('-', '_').replace('/', '_')
        features[f'freq_j_{safe_name}'] = freq
    
    # Global AA frequencies
    aa_total = sum(raw['global_aa_counts'].values())
    for aa in AMINO_ACIDS:
        freq = raw['global_aa_counts'].get(aa, 0) / aa_total if aa_total > 0 else 0
        features[f'freq_aa_global_{aa}'] = freq
    
    # Position-specific AA frequencies
    for pos in range(CDR3_POS_START, CDR3_POS_END):
        pos_counts = raw['aa_pos_counts'].get(pos, {})
        pos_total = sum(pos_counts.values())
        for aa in AMINO_ACIDS:
            freq = pos_counts.get(aa, 0) / pos_total if pos_total > 0 else 0
            features[f'freq_aa_{aa}_pos{pos}'] = freq
    
    return features


# ============================================================================
# FEATURE EXTRACTION FROM DIRECTORY
# ============================================================================

def extract_xgb_features_from_directory(data_dir: str, labels_dict: dict = None,
                                         all_v_genes: list = None, all_j_genes: list = None,
                                         n_cores: int = None):
    """
    Extract XGB features from all files in a directory.
    
    Args:
        data_dir: Path to directory with TSV files
        labels_dict: Optional dict mapping patient_id to label
        all_v_genes: List of all V genes (for consistent feature columns)
        all_j_genes: List of all J genes (for consistent feature columns)
        n_cores: Number of CPU cores
    
    Returns:
        Dictionary with:
            'X_stat': statistical features matrix
            'X_freq': frequency features matrix
            'y': labels (or None for test)
            'patient_ids': list of patient IDs
            'stat_feature_names': list of stat feature names
            'freq_feature_names': list of freq feature names
            'all_v_genes': discovered/used V genes
            'all_j_genes': discovered/used J genes
    """
    if n_cores is None:
        n_cores = N_CORES
    
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.tsv")))
    
    # Discover genes if not provided
    if all_v_genes is None or all_j_genes is None:
        all_v_genes, all_j_genes = discover_genes_from_files(files, n_cores)
    
    # Extract raw data
    tasks = [(f, i) for i, f in enumerate(files)]
    patient_ids = [f.stem for f in files]
    
    with Pool(n_cores) as p:
        results = list(tqdm(p.imap(extract_raw_wrapper, tasks),
                           total=len(tasks), desc="   Extracting XGB features"))
    
    # Compute features
    stat_rows = []
    freq_rows = []
    valid_ids = []
    labels = []
    
    for idx, raw in results:
        if raw is None or raw['n_valid'] == 0:
            continue
        
        stat_features = compute_statistical_features(raw)
        freq_features = compute_frequency_features(raw, all_v_genes, all_j_genes)
        
        if stat_features is None or freq_features is None:
            continue
        
        pid = patient_ids[idx]
        
        if labels_dict is not None:
            label = labels_dict.get(pid)
            if label is None:
                continue
            labels.append(1 if label else 0)
        
        stat_rows.append(stat_features)
        freq_rows.append(freq_features)
        valid_ids.append(pid)
    
    del results
    gc.collect()
    
    # Convert to arrays
    stat_df = pd.DataFrame(stat_rows)
    freq_df = pd.DataFrame(freq_rows)
    
    stat_feature_names = list(stat_df.columns)
    freq_feature_names = list(freq_df.columns)
    
    X_stat = stat_df.values.astype(np.float32)
    X_freq = freq_df.values.astype(np.float32)
    
    # Handle NaN
    X_stat = np.nan_to_num(X_stat, nan=0.0, posinf=0.0, neginf=0.0)
    X_freq = np.nan_to_num(X_freq, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        'X_stat': X_stat,
        'X_freq': X_freq,
        'y': np.array(labels) if labels else None,
        'patient_ids': np.array(valid_ids),
        'stat_feature_names': stat_feature_names,
        'freq_feature_names': freq_feature_names,
        'all_v_genes': all_v_genes,
        'all_j_genes': all_j_genes
    }


# ============================================================================
# FEATURE SELECTION (OPTIMIZED)
# ============================================================================

def compute_effect_sizes_fast(X, y):
    """
    Compute Cohen's d effect size for all features - VECTORIZED.
    """
    pos_mask = y == 1
    neg_mask = y == 0
    
    pos_mean = np.mean(X[pos_mask], axis=0)
    neg_mean = np.mean(X[neg_mask], axis=0)
    pos_var = np.var(X[pos_mask], axis=0)
    neg_var = np.var(X[neg_mask], axis=0)
    
    diff = np.abs(pos_mean - neg_mean)
    pooled_std = np.sqrt((pos_var + neg_var) / 2)
    
    # Avoid division by zero
    pooled_std[pooled_std == 0] = 1e-10
    
    return diff / pooled_std


def select_features_by_effect(X, y, feature_names, min_effect=0.15, max_features=50):
    """
    Select features by effect size.
    
    Returns:
        selected_idx: list of selected feature indices
        selected_names: list of selected feature names
    """
    effects = compute_effect_sizes_fast(X, y)
    
    # Get indices sorted by effect size (descending)
    sorted_indices = np.argsort(effects)[::-1]
    
    selected_idx = []
    selected_names = []
    
    for idx in sorted_indices:
        if effects[idx] >= min_effect and len(selected_idx) < max_features:
            selected_idx.append(idx)
            selected_names.append(feature_names[idx])
        elif len(selected_idx) >= max_features:
            break
    
    # Ensure minimum features
    if len(selected_idx) < 10:
        for idx in sorted_indices[:10]:
            if idx not in selected_idx:
                selected_idx.append(idx)
                selected_names.append(feature_names[idx])
    
    return selected_idx, selected_names


# ============================================================================
# FOLD-BASED TRAINING (NO DATA LEAKAGE)
# ============================================================================

def train_xgb_fold(X_all, y_all, train_idx, val_idx, feature_names,
                   min_effect=0.15, max_features=50, C=0.1, seed=42):
    """
    Train XGB specialist on a fold - feature selection on train_idx only.
    
    Args:
        X_all: Full feature matrix
        y_all: Full labels
        train_idx: Indices for training
        val_idx: Indices for validation
        feature_names: List of feature names
        min_effect: Minimum effect size for feature selection
        max_features: Maximum features to select
        C: Regularization strength
        seed: Random seed
    
    Returns:
        val_preds: Predictions for val_idx
        model: Trained model tuple (clf, scaler, selected_idx)
        selected_features: Names of selected features
    """
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx]
    
    # Feature selection on TRAINING DATA ONLY (no leakage!)
    selected_idx, selected_names = select_features_by_effect(
        X_train, y_train, feature_names, min_effect, max_features
    )
    
    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    
    clf = LogisticRegression(
        penalty='l2',
        C=C,
        class_weight='balanced',
        max_iter=1000,
        random_state=seed
    )
    clf.fit(X_train_scaled, y_train)
    
    val_preds = clf.predict_proba(X_val_scaled)[:, 1]
    
    return val_preds, (clf, scaler, selected_idx), selected_names


def train_xgb_full(X_all, y_all, feature_names,
                   min_effect=0.15, max_features=50, C=0.1, seed=42):
    """
    Train XGB specialist on full data for test prediction.
    
    Returns:
        model: Trained model tuple (clf, scaler, selected_idx)
        selected_features: Names of selected features
    """
    selected_idx, selected_names = select_features_by_effect(
        X_all, y_all, feature_names, min_effect, max_features
    )
    
    X_sel = X_all[:, selected_idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    
    clf = LogisticRegression(
        penalty='l2',
        C=C,
        class_weight='balanced',
        max_iter=1000,
        random_state=seed
    )
    clf.fit(X_scaled, y_all)
    
    return (clf, scaler, selected_idx), selected_names


def predict_xgb_test(model, X_test):
    """
    Predict on test data using trained model.
    
    Args:
        model: Tuple (clf, scaler, selected_idx)
        X_test: Test feature matrix (full features)
    
    Returns:
        predictions: Probability predictions
    """
    clf, scaler, selected_idx = model
    
    X_test_sel = X_test[:, selected_idx]
    X_test_scaled = scaler.transform(X_test_sel)
    
    return clf.predict_proba(X_test_scaled)[:, 1]


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

def get_xgb_params(dataset_num: int, feature_type: str):
    """
    Get hyperparameters for a dataset.
    
    Args:
        dataset_num: Dataset number (1-8)
        feature_type: 'statistical' or 'frequency'
    
    Returns:
        dict with min_effect, max_features, C
    """
    if feature_type == 'statistical':
        return XGB_STAT_PARAMS.get(dataset_num, DEFAULT_PARAMS)
    else:
        return XGB_FREQ_PARAMS.get(dataset_num, DEFAULT_PARAMS)


def extract_and_prepare_xgb_data(train_dir: str, test_dirs: list, labels_dict: dict,
                                  n_cores: int = None):
    """
    Extract XGB features from training and test directories.
    
    Args:
        train_dir: Path to training directory
        test_dirs: List of test directory paths
        labels_dict: dict mapping patient_id to label
        n_cores: Number of CPU cores
    
    Returns:
        Dictionary with train and test data for both stat and freq features
    """
    # Get all files for gene discovery (train + test)
    train_dir = Path(train_dir)
    all_files = list(train_dir.glob("*.tsv"))
    for test_dir in test_dirs:
        test_dir = Path(test_dir)
        if test_dir.exists():
            all_files.extend(list(test_dir.glob("*.tsv")))
    
    # Discover genes
    print("   Discovering genes...")
    all_v_genes, all_j_genes = discover_genes_from_files(all_files, n_cores)
    print(f"   Found {len(all_v_genes)} V-genes, {len(all_j_genes)} J-genes")
    
    # Extract training features
    print("   Extracting training features...")
    train_data = extract_xgb_features_from_directory(
        str(train_dir), labels_dict, all_v_genes, all_j_genes, n_cores
    )
    
    # Extract test features
    test_data = {}
    for test_dir in test_dirs:
        test_dir = Path(test_dir)
        if not test_dir.exists():
            continue
        test_name = test_dir.name
        print(f"   Extracting {test_name} features...")
        test_data[test_name] = extract_xgb_features_from_directory(
            str(test_dir), None, all_v_genes, all_j_genes, n_cores
        )
    
    return {
        'train': train_data,
        'test': test_data,
        'all_v_genes': all_v_genes,
        'all_j_genes': all_j_genes
    }