#!/usr/bin/env python3
"""
ImmuneStatePredictor: Hybrid Stacking Ensemble Model

6 Specialists:
- Physicochemical: Physics features (charge, size, ring, flexibility)
- Kmer: K-mer sequence features
- VJGene: V/J gene features
- ReactiveTCR: Chi-squared based reactive TCR selection
- Statistical: Statistical features (entropy, moments)
- Frequency: Frequency features (V/J/AA distributions)

1 Head: LogisticRegressionCV with L1 regularization

KEY FEATURE: max(HEAD, best_specialist) - never worse than best individual
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore')

# Import from our modules
from .utils import load_metadata, get_repertoire_ids
from .feature_extraction import (
    extract_features_from_directory,
    TRANS_CHARGE, TRANS_SIZE, get_kmers
)
from .base_specialists import (
    train_physicochemical, train_kmer, train_vjgene,
    predict_with_model, get_model_weights
)
from .atttcr_specialist import (
    extract_tcrs_from_directory,
    train_atttcr_fold, train_atttcr_full,
    predict_atttcr_test
)
from .xgboost_specialist import (
    extract_xgb_features_from_directory,
    discover_genes_from_files,
    train_xgb_fold, train_xgb_full, predict_xgb_test,
    get_xgb_params
)

# --- CONFIG ---
SEED = 42
TOP_K_SEQUENCES = 50000
NUM_REACTIVE_TCRS = 500
BATCH_SIZE = 50000  # Batch size for parallel scoring

# --- PARALLEL SCORING GLOBALS ---
# Used by multiprocessing workers - arrays shared via initializer
_SCORE_PARAMS = {}

def _init_scorer_arrays(junctions, v_calls, j_calls, v_seq, v_phys, v_gene, seq_w, phys_w, gene_w, head_w):
    """Initialize worker process with shared arrays and weights."""
    _SCORE_PARAMS['junctions'] = junctions
    _SCORE_PARAMS['v_calls'] = v_calls
    _SCORE_PARAMS['j_calls'] = j_calls
    _SCORE_PARAMS['v_seq'] = v_seq
    _SCORE_PARAMS['v_phys'] = v_phys
    _SCORE_PARAMS['v_gene'] = v_gene
    _SCORE_PARAMS['seq_w'] = seq_w
    _SCORE_PARAMS['phys_w'] = phys_w
    _SCORE_PARAMS['gene_w'] = gene_w
    _SCORE_PARAMS['head_w'] = head_w


def _score_range(index_range):
    """Score a range of sequences. Returns (list of indices, list of scores)."""
    start_idx, end_idx = index_range
    
    # Get shared data from globals
    junctions = _SCORE_PARAMS['junctions']
    v_calls = _SCORE_PARAMS['v_calls']
    j_calls = _SCORE_PARAMS['j_calls']
    v_seq = _SCORE_PARAMS['v_seq']
    v_phys = _SCORE_PARAMS['v_phys']
    v_gene = _SCORE_PARAMS['v_gene']
    seq_w = _SCORE_PARAMS['seq_w']
    phys_w = _SCORE_PARAMS['phys_w']
    gene_w = _SCORE_PARAMS['gene_w']
    head_w = _SCORE_PARAMS['head_w']
    
    indices = []
    scores = []
    
    for idx in range(start_idx, end_idx):
        junction_aa = junctions[idx]
        v_call = v_calls[idx]
        j_call = j_calls[idx]
        
        seq = str(junction_aa) if pd.notna(junction_aa) else ""
        if len(seq) < 4:
            indices.append(idx)
            scores.append(0.0)
            continue
        
        # Sequence score (k-mers 4,5,6)
        seq_score = 0.0
        for k in (4, 5, 6):
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                idx_k = v_seq.get(kmer)
                if idx_k is not None and idx_k < len(seq_w):
                    seq_score += seq_w[idx_k]
        
        # Physics score
        phys_score = 0.0
        
        # Charge
        s_charge = seq.translate(TRANS_CHARGE)
        for k in (3, 4):
            for i in range(len(s_charge) - k + 1):
                idx_k = v_phys.get(f"C:{s_charge[i:i+k]}")
                if idx_k is not None and idx_k < len(phys_w):
                    phys_score += phys_w[idx_k]
        
        # Size
        s_size = seq.translate(TRANS_SIZE)
        for k in (3, 4):
            for i in range(len(s_size) - k + 1):
                idx_k = v_phys.get(f"S:{s_size[i:i+k]}")
                if idx_k is not None and idx_k < len(phys_w):
                    phys_score += phys_w[idx_k]
        
        # Ring
        s_ring = ''.join('R' if c in "FWYH" else 'N' for c in seq)
        for k in (3, 4):
            for i in range(len(s_ring) - k + 1):
                idx_k = v_phys.get(f"R:{s_ring[i:i+k]}")
                if idx_k is not None and idx_k < len(phys_w):
                    phys_score += phys_w[idx_k]
        
        # Flexibility
        s_flex = ''.join('R' if c == 'P' else ('F' if c in 'GS' else 'N') for c in seq)
        for k in (3, 4):
            for i in range(len(s_flex) - k + 1):
                idx_k = v_phys.get(f"F:{s_flex[i:i+k]}")
                if idx_k is not None and idx_k < len(phys_w):
                    phys_score += phys_w[idx_k]
        
        # Gene score
        gene_score = 0.0
        if v_call and str(v_call) != '' and str(v_call) != 'nan':
            idx_k = v_gene.get(f"V:{v_call}")
            if idx_k is not None and idx_k < len(gene_w):
                gene_score += gene_w[idx_k]
        if j_call and str(j_call) != '' and str(j_call) != 'nan':
            idx_k = v_gene.get(f"J:{j_call}")
            if idx_k is not None and idx_k < len(gene_w):
                gene_score += gene_w[idx_k]
        
        # Combined score
        final_score = (
            head_w[0] * phys_score +
            head_w[1] * seq_score +
            head_w[2] * gene_score
        )
        indices.append(idx)
        scores.append(final_score)
    
    return (indices, scores)


# ============================================================================
# UNIFIED TEST EXTRACTION (Single-pass for all specialists)
# ============================================================================

def _process_test_file_unified(args):
    """
    Process a single test file to extract ALL features in one pass:
    1. Base features (seq, gene, phys) for vectorization
    2. TCR combinations for ReactiveTCR
    3. Raw counts for XGB features
    
    Args:
        args: Tuple of (filepath, v_seq, v_gene, v_phys, all_v_genes, all_j_genes)
    
    Returns:
        Dictionary with all extracted data or None on error
    """
    from collections import Counter, defaultdict
    import re
    
    filepath, v_seq, v_gene, v_phys, all_v_genes, all_j_genes = args
    
    # XGB constants
    CDR3_POS_START = 3
    CDR3_POS_END = 20
    CDR3_LEN_MIN = 8
    CDR3_LEN_MAX = 29
    AA_SET = set('ACDEFGHIKLMNPQRSTVWY')
    
    def get_v_gene_clean(v_call):
        if not v_call or pd.isna(v_call) or str(v_call).lower() in ['nan', '', 'none']:
            return None
        v = str(v_call).strip()
        v = re.sub(r'\*\d+', '', v)
        return v if v else None
    
    def get_j_gene_clean(j_call):
        if not j_call or pd.isna(j_call) or str(j_call).lower() in ['nan', '', 'none']:
            return None
        j = str(j_call).strip()
        j = re.sub(r'\*\d+', '', j)
        return j if j else None
    
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        sample_id = Path(filepath).stem
        
        # === Base features (for vectorization) ===
        d_seq, d_gene, d_phys = {}, {}, {}
        
        # === ReactiveTCR (TCR combinations) ===
        tcr_combinations = []
        
        # === XGB raw counts ===
        xgb_v_counts = Counter()
        xgb_j_counts = Counter()
        xgb_length_counts = Counter()
        xgb_global_aa_counts = Counter()
        xgb_aa_pos_counts = defaultdict(Counter)
        xgb_cdr3_lengths = []
        xgb_n_valid = 0
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            junction = str(junction) if pd.notna(junction) else ''
            v_call_str = str(v_call) if pd.notna(v_call) else ''
            j_call_str = str(j_call) if pd.notna(j_call) else ''
            
            # Skip short sequences for base features
            if len(junction) < 4:
                continue
            
            # === 1. BASE FEATURES ===
            # Genes
            if v_call_str:
                key = f"V:{v_call_str}"
                idx = v_gene.get(key)
                if idx is not None:
                    d_gene[idx] = d_gene.get(idx, 0) + 1
            if j_call_str:
                key = f"J:{j_call_str}"
                idx = v_gene.get(key)
                if idx is not None:
                    d_gene[idx] = d_gene.get(idx, 0) + 1
            
            # Sequence k-mers (4,5,6)
            for k in [4, 5, 6]:
                for i in range(len(junction) - k + 1):
                    kmer = junction[i:i+k]
                    idx = v_seq.get(kmer)
                    if idx is not None:
                        d_seq[idx] = d_seq.get(idx, 0) + 1
            
            # Physics features
            s_c = junction.translate(TRANS_CHARGE)
            for k in [3, 4]:
                for i in range(len(s_c) - k + 1):
                    idx = v_phys.get(f"C:{s_c[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_s = junction.translate(TRANS_SIZE)
            for k in [3, 4]:
                for i in range(len(s_s) - k + 1):
                    idx = v_phys.get(f"S:{s_s[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_r = ''.join('R' if aa in "FWYH" else 'N' for aa in junction)
            for k in [3, 4]:
                for i in range(len(s_r) - k + 1):
                    idx = v_phys.get(f"R:{s_r[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_f = ''.join('R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in junction)
            for k in [3, 4]:
                for i in range(len(s_f) - k + 1):
                    idx = v_phys.get(f"F:{s_f[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            # === 2. ReactiveTCR ===
            combo_key = f"{junction}_{v_call_str}"
            tcr_combinations.append(combo_key)
            
            # === 3. XGB features ===
            if CDR3_LEN_MIN <= len(junction) <= CDR3_LEN_MAX and all(c in AA_SET for c in junction):
                xgb_n_valid += 1
                xgb_length_counts[len(junction)] += 1
                xgb_cdr3_lengths.append(len(junction))
                
                v_gene_clean = get_v_gene_clean(v_call)
                if v_gene_clean:
                    xgb_v_counts[v_gene_clean] += 1
                
                j_gene_clean = get_j_gene_clean(j_call)
                if j_gene_clean:
                    xgb_j_counts[j_gene_clean] += 1
                
                for aa in junction:
                    xgb_global_aa_counts[aa] += 1
                
                for pos in range(CDR3_POS_START, min(len(junction), CDR3_POS_END)):
                    xgb_aa_pos_counts[pos][junction[pos]] += 1
        
        del df
        
        # Use seeded shuffle for deterministic ordering (faster than sort)
        tcr_unique = list(set(tcr_combinations))
        import random
        random.Random(SEED).shuffle(tcr_unique)
        
        return {
            'sample_id': sample_id,
            # Base features
            'd_seq': d_seq,
            'd_gene': d_gene,
            'd_phys': d_phys,
            # ReactiveTCR
            'tcr_combinations': tcr_unique,  # Unique, deterministic via seeded shuffle
            # XGB raw
            'xgb_raw': {
                'v_counts': xgb_v_counts,
                'j_counts': xgb_j_counts,
                'length_counts': xgb_length_counts,
                'global_aa_counts': xgb_global_aa_counts,
                'aa_pos_counts': dict(xgb_aa_pos_counts),
                'cdr3_lengths': xgb_cdr3_lengths,
                'n_valid': xgb_n_valid
            },
            'all_v_genes': all_v_genes,
            'all_j_genes': all_j_genes
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def _compute_xgb_features_from_raw(raw, all_v_genes, all_j_genes):
    """Compute XGB stat and freq features from raw counts."""
    from scipy import stats as scipy_stats
    
    AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
    CDR3_POS_START = 3
    CDR3_POS_END = 20
    CDR3_LEN_MIN = 8
    CDR3_LEN_MAX = 29
    
    def compute_entropy(counts):
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
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return 1.0 - sum((c/total)**2 for c in counts.values())
    
    n_total = raw['n_valid']
    if n_total == 0:
        return None, None
    
    # === Statistical features ===
    stat_features = {}
    
    lengths = raw['cdr3_lengths']
    if len(lengths) >= 10:
        arr = np.array(lengths, dtype=np.float64)
        stat_features['stat_len_mean'] = np.mean(arr)
        stat_features['stat_len_std'] = np.std(arr)
        skew_val = scipy_stats.skew(arr)
        stat_features['stat_len_skew'] = skew_val if not np.isnan(skew_val) else 0
        kurt_val = scipy_stats.kurtosis(arr)
        stat_features['stat_len_kurt'] = kurt_val if not np.isnan(kurt_val) else 0
        stat_features['stat_len_median'] = np.median(arr)
        stat_features['stat_len_iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
    else:
        for k in ['stat_len_mean', 'stat_len_std', 'stat_len_skew', 'stat_len_kurt', 'stat_len_median', 'stat_len_iqr']:
            stat_features[k] = 0
    
    stat_features['stat_v_entropy'] = compute_entropy(raw['v_counts'])
    stat_features['stat_v_gini'] = compute_gini_simpson(raw['v_counts'])
    stat_features['stat_v_nunique'] = len(raw['v_counts'])
    stat_features['stat_j_entropy'] = compute_entropy(raw['j_counts'])
    stat_features['stat_j_gini'] = compute_gini_simpson(raw['j_counts'])
    stat_features['stat_j_nunique'] = len(raw['j_counts'])
    stat_features['stat_aa_global_entropy'] = compute_entropy(raw['global_aa_counts'])
    stat_features['stat_aa_global_gini'] = compute_gini_simpson(raw['global_aa_counts'])
    
    for pos in range(CDR3_POS_START, CDR3_POS_END):
        if pos in raw['aa_pos_counts']:
            stat_features[f'stat_aa_ent_pos{pos}'] = compute_entropy(raw['aa_pos_counts'][pos])
            stat_features[f'stat_aa_gini_pos{pos}'] = compute_gini_simpson(raw['aa_pos_counts'][pos])
        else:
            stat_features[f'stat_aa_ent_pos{pos}'] = 0
            stat_features[f'stat_aa_gini_pos{pos}'] = 0
    
    stat_features['stat_log_size'] = np.log1p(n_total)
    
    # === Frequency features ===
    freq_features = {}
    
    for length in range(CDR3_LEN_MIN, CDR3_LEN_MAX + 1):
        freq_features[f'freq_len_{length}'] = raw['length_counts'].get(length, 0) / n_total
    
    v_total = sum(raw['v_counts'].values())
    for v_gene in all_v_genes:
        freq = raw['v_counts'].get(v_gene, 0) / v_total if v_total > 0 else 0
        safe_name = v_gene.replace('-', '_').replace('/', '_')
        freq_features[f'freq_v_{safe_name}'] = freq
    
    j_total = sum(raw['j_counts'].values())
    for j_gene in all_j_genes:
        freq = raw['j_counts'].get(j_gene, 0) / j_total if j_total > 0 else 0
        safe_name = j_gene.replace('-', '_').replace('/', '_')
        freq_features[f'freq_j_{safe_name}'] = freq
    
    aa_total = sum(raw['global_aa_counts'].values())
    for aa in AMINO_ACIDS:
        freq = raw['global_aa_counts'].get(aa, 0) / aa_total if aa_total > 0 else 0
        freq_features[f'freq_aa_global_{aa}'] = freq
    
    for pos in range(CDR3_POS_START, CDR3_POS_END):
        pos_counts = raw['aa_pos_counts'].get(pos, {})
        pos_total = sum(pos_counts.values())
        for aa in AMINO_ACIDS:
            freq = pos_counts.get(aa, 0) / pos_total if pos_total > 0 else 0
            freq_features[f'freq_aa_{aa}_pos{pos}'] = freq
    
    return stat_features, freq_features


class ImmuneStatePredictor:
    """
    A hybrid stacking ensemble model for predicting immune states from TCR repertoire data.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """
        Initialize the predictor.

        Args:
            n_jobs: Number of CPU cores to use (-1 for all)
            device: Device for computation ('cpu' or 'cuda')
            **kwargs: Additional hyperparameters
        """
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        
        self.device = device
        self.seed = kwargs.get('seed', SEED)
        self.top_k = kwargs.get('top_k', TOP_K_SEQUENCES)
        self.num_reactive_tcrs = kwargs.get('num_reactive_tcrs', NUM_REACTIVE_TCRS)
        
        # Models (set after fit)
        self.physicochemical_model = None
        self.kmer_model = None
        self.vjgene_model = None
        self.reactive_tcr_model = None
        self.statistical_model = None
        self.frequency_model = None
        self.head_model = None
        
        # Data (set after fit)
        self.vocab = None
        self.train_data = None
        self.reactive_tcr_data = None
        self.xgb_data = None
        self.reactive_tcr_scores = None
        self.unique_sequences_df = None
        self.head_weights = None
        self.dataset_num = None
        
        # Important sequences (set after fit)
        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        """
        Train the model on the provided training data.

        Args:
            train_dir_path: Path to the directory with training TSV files

        Returns:
            self: The fitted predictor instance
        """
        print(f"🧬 Fitting model on {train_dir_path}...")
        
        # Try to extract dataset number from path
        self.dataset_num = self._extract_dataset_num(train_dir_path)
        
        # Load metadata
        metadata_path = os.path.join(train_dir_path, "metadata.csv")
        labels_dict = load_metadata(metadata_path)
        
        # --- STEP 1: Extract base features (k-mers, physics, genes) ---
        print("   📊 Extracting base features...")
        feature_data = extract_features_from_directory(
            train_dir_path, 
            test_dirs=None,
            n_cores=self.n_jobs,
            collect_sequences=True
        )
        
        self.train_data = feature_data['train']
        self.vocab = feature_data['vocab']
        self.unique_sequences_df = feature_data.get('unique_sequences')
        
        ids = self.train_data['ids']
        y = self.train_data['y']
        X_seq = self.train_data['X_seq']
        X_gene = self.train_data['X_gene']
        X_phys = self.train_data['X_phys']
        
        print(f"   Samples: {len(y)} (pos={sum(y)}, neg={len(y)-sum(y)})")
        print(f"   Features: Seq={X_seq.shape[1]}, Gene={X_gene.shape[1]}, Phys={X_phys.shape[1]}")
        
        # --- STEP 2: Extract ReactiveTCR data ---
        print("   🧬 Extracting ReactiveTCR data...")
        self.reactive_tcr_data = extract_tcrs_from_directory(
            train_dir_path, labels_dict, self.n_jobs
        )
        has_reactive_tcr = self.reactive_tcr_data is not None and self.reactive_tcr_data['labels'] is not None
        
        # --- STEP 3: Extract XGBoost data ---
        print("   📈 Extracting XGBoost features...")
        self.xgb_data = extract_xgb_features_from_directory(
            train_dir_path, labels_dict, n_cores=self.n_jobs
        )
        has_xgb = self.xgb_data is not None and self.xgb_data['y'] is not None
        
        # --- STEP 4: Build meta-features via internal CV (FULLY PARALLEL) ---
        print("   🔄 Building meta-features (all folds parallel)...")
        n_specialists = 3  # base: physicochemical, kmer, vjgene
        if has_reactive_tcr:
            n_specialists += 1
        if has_xgb:
            n_specialists += 2  # statistical + frequency
        
        meta_train = np.zeros((len(y), n_specialists))
        internal_cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        # Get XGB params
        xgb_stat_params = get_xgb_params(self.dataset_num, 'statistical')
        xgb_freq_params = get_xgb_params(self.dataset_num, 'frequency')
        
        # Build ALL jobs for ALL folds at once
        all_jobs = []
        fold_info = []  # Track (fold_idx, specialist_idx, holdout_indices)
        
        for fold_idx, (i_tr, i_ho) in enumerate(internal_cv.split(y)):
            # Base specialists
            all_jobs.append(delayed(train_physicochemical)(X_phys[i_tr], y[i_tr], X_phys[i_ho], self.seed))
            fold_info.append((fold_idx, 0, i_ho))
            
            all_jobs.append(delayed(train_kmer)(X_seq[i_tr], y[i_tr], X_seq[i_ho], self.seed))
            fold_info.append((fold_idx, 1, i_ho))
            
            all_jobs.append(delayed(train_vjgene)(X_gene[i_tr], y[i_tr], X_gene[i_ho], self.seed))
            fold_info.append((fold_idx, 2, i_ho))
            
            spec_idx = 3
            if has_reactive_tcr:
                all_jobs.append(delayed(train_atttcr_fold)(
                    self.reactive_tcr_data['patient_tcrs'],
                    self.reactive_tcr_data['labels'],
                    i_tr, i_ho,
                    n_reactive=self.num_reactive_tcrs,
                    seed=self.seed
                ))
                fold_info.append((fold_idx, spec_idx, i_ho))
                spec_idx += 1
            
            if has_xgb:
                all_jobs.append(delayed(train_xgb_fold)(
                    self.xgb_data['X_stat'], self.xgb_data['y'],
                    i_tr, i_ho,
                    self.xgb_data['stat_feature_names'],
                    **xgb_stat_params, seed=self.seed
                ))
                fold_info.append((fold_idx, spec_idx, i_ho))
                
                all_jobs.append(delayed(train_xgb_fold)(
                    self.xgb_data['X_freq'], self.xgb_data['y'],
                    i_tr, i_ho,
                    self.xgb_data['freq_feature_names'],
                    **xgb_freq_params, seed=self.seed
                ))
                fold_info.append((fold_idx, spec_idx + 1, i_ho))
        
        # Run ALL jobs in parallel (uses all available cores)
        print(f"      Running {len(all_jobs)} jobs across {self.n_jobs} cores...")
        all_results = Parallel(n_jobs=self.n_jobs)(all_jobs)
        
        # Reassemble results into meta_train
        for job_idx, (fold_idx, spec_idx, i_ho) in enumerate(fold_info):
            preds = all_results[job_idx][0]
            # Handle ReactiveTCR None case
            if preds is None:
                preds = np.full(len(i_ho), 0.5)
            meta_train[i_ho, spec_idx] = preds
        
        # --- STEP 5: Train HEAD ---
        print("   🎯 Training HEAD ensemble...")
        self.head_model = LogisticRegressionCV(
            Cs=np.logspace(-2, 3, 20),
            penalty='l1',
            solver='saga',
            cv=3,
            class_weight='balanced',
            max_iter=5000,
            random_state=self.seed
        )
        self.head_model.fit(meta_train, y)
        self.head_weights = self.head_model.coef_[0]
        
        # --- STEP 6: Train final specialists on full data (PARALLEL) ---
        print("   🏋️ Training final specialists...")
        
        # Build jobs for parallel execution
        final_jobs = [
            delayed(train_physicochemical)(X_phys, y, X_phys, self.seed),
            delayed(train_kmer)(X_seq, y, X_seq, self.seed),
            delayed(train_vjgene)(X_gene, y, X_gene, self.seed),
        ]
        
        if has_reactive_tcr:
            final_jobs.append(delayed(train_atttcr_full)(
                self.reactive_tcr_data['patient_tcrs'],
                self.reactive_tcr_data['labels'],
                n_reactive=self.num_reactive_tcrs,
                seed=self.seed
            ))
        
        if has_xgb:
            final_jobs.append(delayed(train_xgb_full)(
                self.xgb_data['X_stat'], self.xgb_data['y'],
                self.xgb_data['stat_feature_names'],
                **xgb_stat_params, seed=self.seed
            ))
            final_jobs.append(delayed(train_xgb_full)(
                self.xgb_data['X_freq'], self.xgb_data['y'],
                self.xgb_data['freq_feature_names'],
                **xgb_freq_params, seed=self.seed
            ))
        
        # Run all in parallel (loky backend for determinism)
        final_results = Parallel(n_jobs=min(len(final_jobs), self.n_jobs))(final_jobs)
        
        # Extract models
        _, self.physicochemical_model = final_results[0]
        _, self.kmer_model = final_results[1]
        _, self.vjgene_model = final_results[2]
        
        idx = 3
        if has_reactive_tcr:
            self.reactive_tcr_model, self.reactive_tcr_scores = final_results[idx]
            idx += 1
        
        if has_xgb:
            self.statistical_model, _ = final_results[idx]
            self.frequency_model, _ = final_results[idx + 1]
        
        # --- STEP 7: Identify important sequences ---
        print("   🔬 Identifying important sequences...")
        dataset_name = os.path.basename(train_dir_path)
        self.important_sequences_ = self.identify_associated_sequences(
            dataset_name=dataset_name,
            top_k=self.top_k
        )
        
        print("   ✅ Training complete!")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        OPTIMIZED: Predict probabilities for test data using single-pass extraction.

        Args:
            test_dir_path: Path to directory with test TSV files

        Returns:
            DataFrame with predictions
        """
        print(f"   🎯 Predicting on {test_dir_path}...")
        
        if self.head_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        test_name = os.path.basename(test_dir_path)
        
        # --- SINGLE PASS: Extract ALL test features at once ---
        import glob
        from scipy import sparse
        
        test_files = sorted(glob.glob(os.path.join(test_dir_path, "*.tsv")))
        
        # Get gene lists for XGB
        all_v_genes = self.xgb_data['all_v_genes'] if self.xgb_data else []
        all_j_genes = self.xgb_data['all_j_genes'] if self.xgb_data else []
        
        # Prepare tasks
        v_seq = self.vocab['seq']
        v_gene = self.vocab['gene']
        v_phys = self.vocab['phys']
        
        tasks = [(f, v_seq, v_gene, v_phys, all_v_genes, all_j_genes) for f in test_files]
        
        print(f"      Extracting features (single pass)...")
        with Pool(self.n_jobs) as p:
            results = list(tqdm(p.imap(_process_test_file_unified, tasks),
                               total=len(tasks), desc="      Processing test files"))
        
        # --- Build all data structures from unified results ---
        sample_ids = []
        
        # Base features
        seq_data, seq_row, seq_col = [], [], []
        gene_data, gene_row, gene_col = [], [], []
        phys_data, phys_row, phys_col = [], [], []
        
        # ReactiveTCR
        patient_tcrs = {}
        
        # XGB
        xgb_stat_rows = []
        xgb_freq_rows = []
        xgb_valid_indices = []
        
        row_idx = 0
        for res in results:
            if res is None:
                continue
            
            sample_id = res['sample_id']
            sample_ids.append(sample_id)
            
            # === Base features ===
            for col, val in res['d_seq'].items():
                seq_data.append(val)
                seq_row.append(row_idx)
                seq_col.append(col)
            
            for col, val in res['d_gene'].items():
                gene_data.append(val)
                gene_row.append(row_idx)
                gene_col.append(col)
            
            for col, val in res['d_phys'].items():
                phys_data.append(val)
                phys_row.append(row_idx)
                phys_col.append(col)
            
            # === ReactiveTCR ===
            patient_tcrs[row_idx] = res['tcr_combinations']
            
            # === XGB ===
            if res['xgb_raw']['n_valid'] > 0:
                stat_feats, freq_feats = _compute_xgb_features_from_raw(
                    res['xgb_raw'], all_v_genes, all_j_genes
                )
                if stat_feats is not None:
                    xgb_stat_rows.append(stat_feats)
                    xgb_freq_rows.append(freq_feats)
                    xgb_valid_indices.append(row_idx)
            
            row_idx += 1
        
        del results
        gc.collect()
        
        n_samples = len(sample_ids)
        
        # Build sparse matrices
        def build_sparse(data, row, col, n_cols):
            if data:
                return sparse.csr_matrix(
                    (data, (row, col)),
                    shape=(n_samples, n_cols),
                    dtype=np.float32
                )
            return sparse.csr_matrix((n_samples, n_cols), dtype=np.float32)
        
        X_seq_test = build_sparse(seq_data, seq_row, seq_col, len(v_seq))
        X_gene_test = build_sparse(gene_data, gene_row, gene_col, len(v_gene))
        X_phys_test = build_sparse(phys_data, phys_row, phys_col, len(v_phys))
        
        del seq_data, seq_row, seq_col
        del gene_data, gene_row, gene_col
        del phys_data, phys_row, phys_col
        gc.collect()
        
        # --- Get specialist predictions ---
        final_preds = []
        
        # Base specialists
        f1 = predict_with_model(self.physicochemical_model, X_phys_test)
        f2 = predict_with_model(self.kmer_model, X_seq_test)
        f3 = predict_with_model(self.vjgene_model, X_gene_test)
        final_preds.extend([f1, f2, f3])
        
        # ReactiveTCR
        if self.reactive_tcr_model is not None:
            f_att = predict_atttcr_test(self.reactive_tcr_model, patient_tcrs)
            final_preds.append(f_att)
        
        # XGB
        if self.statistical_model is not None and xgb_stat_rows:
            # Build XGB feature matrices
            stat_df = pd.DataFrame(xgb_stat_rows)
            freq_df = pd.DataFrame(xgb_freq_rows)
            
            # Align columns with training
            for col in self.xgb_data['stat_feature_names']:
                if col not in stat_df.columns:
                    stat_df[col] = 0.0
            stat_df = stat_df[self.xgb_data['stat_feature_names']]
            
            for col in self.xgb_data['freq_feature_names']:
                if col not in freq_df.columns:
                    freq_df[col] = 0.0
            freq_df = freq_df[self.xgb_data['freq_feature_names']]
            
            X_stat_test = np.nan_to_num(stat_df.values.astype(np.float32))
            X_freq_test = np.nan_to_num(freq_df.values.astype(np.float32))
            
            # Map predictions back to full sample order
            f_xgb_stat_partial = predict_xgb_test(self.statistical_model, X_stat_test)
            f_xgb_freq_partial = predict_xgb_test(self.frequency_model, X_freq_test)
            
            # Expand to full sample size
            f_xgb_stat = np.full(n_samples, 0.5)
            f_xgb_freq = np.full(n_samples, 0.5)
            for i, idx in enumerate(xgb_valid_indices):
                f_xgb_stat[idx] = f_xgb_stat_partial[i]
                f_xgb_freq[idx] = f_xgb_freq_partial[i]
            
            final_preds.append(f_xgb_stat)
            final_preds.append(f_xgb_freq)
        
        # --- Combine with HEAD ---
        meta_test = np.column_stack(final_preds)
        
        # max(HEAD, best_specialist) logic
        if np.sum(np.abs(self.head_model.coef_[0])) > 0.01:
            preds = self.head_model.predict_proba(meta_test)[:, 1]
        else:
            # HEAD weights are near zero, use fallback
            best_weight_idx = np.argmax(np.abs(self.head_weights))
            if np.abs(self.head_weights[best_weight_idx]) > 0.01:
                preds = final_preds[best_weight_idx]
                spec_names = ['Physicochemical', 'Kmer', 'VJGene', 'ReactiveTCR', 'Statistical', 'Frequency']
                print(f"      ⚠️ Using {spec_names[best_weight_idx]} (highest weight)")
            elif len(final_preds) > 3:
                # Use ReactiveTCR as fallback
                preds = final_preds[3]
                print(f"      ⚠️ HEAD weights all zero, using ReactiveTCR")
            else:
                # No ReactiveTCR, use average
                preds = np.mean(meta_test, axis=1)
                print(f"      ⚠️ HEAD weights all zero, using average")
        
        # Build output DataFrame
        predictions_df = pd.DataFrame({
            'ID': sample_ids,
            'dataset': test_name,
            'label_positive_probability': preds,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })
        
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 
                                          'junction_aa', 'v_call', 'j_call']]
        
        print(f"   ✅ Predicted {len(predictions_df)} samples")
        return predictions_df

    def identify_associated_sequences(self, dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identify the top k important sequences.

        Args:
            dataset_name: Name of the dataset
            top_k: Number of top sequences to return

        Returns:
            DataFrame with top sequences
        """
        if self.unique_sequences_df is None or len(self.unique_sequences_df) == 0:
            print("      ⚠️ No unique sequences available")
            return self._generate_empty_sequences_df(dataset_name)
        
        if self.vocab is None:
            print("      ⚠️ No vocabulary available")
            return self._generate_empty_sequences_df(dataset_name)
        
        # Get model weights
        phys_weights = get_model_weights(self.physicochemical_model)
        seq_weights = get_model_weights(self.kmer_model)
        gene_weights = get_model_weights(self.vjgene_model)
        
        # Use first 3 head weights for base specialists
        head_w = self.head_weights[:3] if self.head_weights is not None else np.array([1.0, 1.0, 1.0])
        
        # Score sequences
        print(f"      🔬 Scoring {len(self.unique_sequences_df)} sequences...")
        scores = self._score_all_sequences(
            self.unique_sequences_df,
            seq_weights, phys_weights, gene_weights, head_w
        )
        
        self.unique_sequences_df['importance_score'] = scores
        
        # Add ReactiveTCR chi-squared scores if available
        if self.reactive_tcr_scores and len(self.head_weights) > 3:
            att_weight = self.head_weights[3]
            print(f"      Adding ReactiveTCR scores (weight={att_weight:.3f})...")
            keys = (self.unique_sequences_df['junction_aa'].astype(str) + '_' + 
                   self.unique_sequences_df['v_call'].astype(str))
            chi_additions = keys.map(lambda k: self.reactive_tcr_scores.get(k, 0.0) * att_weight)
            self.unique_sequences_df['importance_score'] += chi_additions
        
        # Select top sequences
        top_seqs = self.unique_sequences_df.nlargest(top_k, 'importance_score')
        
        # Format output
        important_seqs_df = pd.DataFrame({
            'ID': [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_seqs))],
            'dataset': dataset_name,
            'label_positive_probability': -999.0,
            'junction_aa': top_seqs['junction_aa'].values,
            'v_call': top_seqs['v_call'].values,
            'j_call': top_seqs['j_call'].values
        })
        
        important_seqs_df = important_seqs_df[['ID', 'dataset', 'label_positive_probability',
                                                'junction_aa', 'v_call', 'j_call']]
        
        print(f"      ✅ Selected top {len(important_seqs_df)} sequences")
        return important_seqs_df

    def _score_all_sequences(self, sequences_df, seq_weights, phys_weights, gene_weights, head_weights):
        """Score all sequences using model weights (parallelized, no tuple creation)."""
        n_seqs = len(sequences_df)
        scores = np.zeros(n_seqs, dtype=np.float32)
        
        if n_seqs == 0:
            return scores
        
        # Extract arrays from DataFrame ONCE (no tuple creation!)
        junctions = sequences_df['junction_aa'].values
        v_calls = sequences_df['v_call'].values
        j_calls = sequences_df['j_call'].values
        
        # Create index ranges for workers (just integers, no data copying)
        ranges = [
            (i, min(i + BATCH_SIZE, n_seqs))
            for i in range(0, n_seqs, BATCH_SIZE)
        ]
        
        n_workers = max(1, self.n_jobs)
        
        # Use parallel processing with Pool - arrays passed via initializer
        with Pool(
            n_workers,
            initializer=_init_scorer_arrays,
            initargs=(
                junctions,
                v_calls,
                j_calls,
                self.vocab['seq'],
                self.vocab['phys'],
                self.vocab['gene'],
                seq_weights,
                phys_weights,
                gene_weights,
                head_weights
            )
        ) as pool:
            # imap_unordered for efficiency - order preserved via indices
            for batch_indices, batch_scores in tqdm(
                pool.imap_unordered(_score_range, ranges),
                total=len(ranges),
                desc="      Scoring sequences",
                leave=False
            ):
                # Assign scores back to correct positions
                for idx, score in zip(batch_indices, batch_scores):
                    scores[idx] = score
        
        return scores

    def _generate_empty_sequences_df(self, dataset_name: str) -> pd.DataFrame:
        """Generate empty sequences DataFrame."""
        return pd.DataFrame({
            'ID': [],
            'dataset': [],
            'label_positive_probability': [],
            'junction_aa': [],
            'v_call': [],
            'j_call': []
        })

    def _extract_dataset_num(self, train_dir_path: str) -> int:
        """Try to extract dataset number from path."""
        import re
        match = re.search(r'dataset[_\s]*(\d+)', train_dir_path)
        if match:
            return int(match.group(1))
        return 1  # Default