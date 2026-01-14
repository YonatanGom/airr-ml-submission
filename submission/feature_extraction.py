#!/usr/bin/env python3
"""
Feature extraction for TCR repertoire data.

Extracts three types of features:
1. Sequence features: K-mers (k=4,5,6) from junction_aa
2. Physics features: Charge, Size, Ring, Flexibility patterns (k=3,4)
3. Gene features: V and J gene usage

Based on the original preprocessor.py logic.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from scipy import sparse
from tqdm import tqdm
import gc
import os

# --- TRANSLATION TABLES (Physics of TCRs) ---
TRANS_CHARGE = str.maketrans("LIVMCAFWSTNQYGPDEKRH", "HHHHHHHHPPPPPPPCCCCC")
TRANS_SIZE = str.maketrans("FWYRKLIMHVPCTNQDEGAS", "BBBBBBBBBSSSSSSSSTTT")
SEED = 42

# Default number of cores
N_CORES = min(30, cpu_count())


def get_kmers(seq: str, k: int) -> list:
    """Extract all k-mers from a sequence."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def extract_features_from_row(junction_aa, v_call, j_call):
    """
    Extract all features from a single TCR row.
    
    Args:
        junction_aa: CDR3 amino acid sequence
        v_call: V gene call
        j_call: J gene call
    
    Returns:
        Tuple of (seq_counter, gene_counter, phys_counter) or (None, None, None) if invalid
    """
    seq = str(junction_aa) if pd.notna(junction_aa) else ""
    if len(seq) < 4:
        return None, None, None
    
    c_seq, c_gene, c_phys = Counter(), Counter(), Counter()
    
    # 1. GENES
    if pd.notna(v_call) and str(v_call) != "":
        c_gene[f"V:{v_call}"] += 1
    if pd.notna(j_call) and str(j_call) != "":
        c_gene[f"J:{j_call}"] += 1
    
    # 2. SEQUENCE K-mers (k=4,5,6)
    for k in [4, 5, 6]:
        for km in get_kmers(seq, k):
            c_seq[km] += 1
    
    # 3. PHYSICS FEATURES
    # Charge patterns
    s_c = seq.translate(TRANS_CHARGE)
    for k in [3, 4]:
        for km in get_kmers(s_c, k):
            c_phys[f"C:{km}"] += 1
    
    # Size patterns
    s_s = seq.translate(TRANS_SIZE)
    for k in [3, 4]:
        for km in get_kmers(s_s, k):
            c_phys[f"S:{km}"] += 1
    
    # Ring patterns (aromatic amino acids)
    s_r = "".join(['R' if aa in "FWYH" else 'N' for aa in seq])
    for k in [3, 4]:
        for km in get_kmers(s_r, k):
            c_phys[f"R:{km}"] += 1
    
    # Flexibility patterns
    s_f = "".join(['R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in seq])
    for k in [3, 4]:
        for km in get_kmers(s_f, k):
            c_phys[f"F:{km}"] += 1
    
    return c_seq, c_gene, c_phys


def process_file_for_vocab(filepath):
    """
    Process a single file to extract vocabulary counts.
    OPTIMIZED: Inlined feature extraction to avoid Counter creation per row.
    
    Returns:
        Tuple of (seq_counter, gene_counter, phys_counter) or None on error
    """
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        c_seq, c_gene, c_phys = Counter(), Counter(), Counter()
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            # Inline feature extraction - no Counter creation per row
            seq = str(junction) if pd.notna(junction) else ""
            if len(seq) < 4:
                continue
            
            # 1. GENES
            if pd.notna(v_call) and str(v_call) != "":
                c_gene[f"V:{v_call}"] += 1
            if pd.notna(j_call) and str(j_call) != "":
                c_gene[f"J:{j_call}"] += 1
            
            # 2. SEQUENCE K-mers (k=4,5,6)
            for k in (4, 5, 6):
                for i in range(len(seq) - k + 1):
                    c_seq[seq[i:i+k]] += 1
            
            # 3. PHYSICS FEATURES
            s_c = seq.translate(TRANS_CHARGE)
            for k in (3, 4):
                for i in range(len(s_c) - k + 1):
                    c_phys[f"C:{s_c[i:i+k]}"] += 1
            
            s_s = seq.translate(TRANS_SIZE)
            for k in (3, 4):
                for i in range(len(s_s) - k + 1):
                    c_phys[f"S:{s_s[i:i+k]}"] += 1
            
            s_r = "".join(['R' if aa in "FWYH" else 'N' for aa in seq])
            for k in (3, 4):
                for i in range(len(s_r) - k + 1):
                    c_phys[f"R:{s_r[i:i+k]}"] += 1
            
            s_f = "".join(['R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in seq])
            for k in (3, 4):
                for i in range(len(s_f) - k + 1):
                    c_phys[f"F:{s_f[i:i+k]}"] += 1
        
        del df
        return c_seq, c_gene, c_phys
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_file_for_vocab_and_sequences(filepath):
    """
    Process a single file to extract vocabulary counts AND unique sequences.
    OPTIMIZED: Inlined feature extraction to avoid Counter creation per row.
    
    Returns:
        Tuple of (seq_counter, gene_counter, phys_counter, unique_sequences_list) or None
    """
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        c_seq, c_gene, c_phys = Counter(), Counter(), Counter()
        unique_seqs = []
        seen = set()
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            # Inline feature extraction - no Counter creation per row
            seq = str(junction) if pd.notna(junction) else ""
            if len(seq) < 4:
                continue
            
            # 1. GENES
            if pd.notna(v_call) and str(v_call) != "":
                c_gene[f"V:{v_call}"] += 1
            if pd.notna(j_call) and str(j_call) != "":
                c_gene[f"J:{j_call}"] += 1
            
            # 2. SEQUENCE K-mers (k=4,5,6)
            for k in (4, 5, 6):
                for i in range(len(seq) - k + 1):
                    c_seq[seq[i:i+k]] += 1
            
            # 3. PHYSICS FEATURES
            s_c = seq.translate(TRANS_CHARGE)
            for k in (3, 4):
                for i in range(len(s_c) - k + 1):
                    c_phys[f"C:{s_c[i:i+k]}"] += 1
            
            s_s = seq.translate(TRANS_SIZE)
            for k in (3, 4):
                for i in range(len(s_s) - k + 1):
                    c_phys[f"S:{s_s[i:i+k]}"] += 1
            
            s_r = "".join(['R' if aa in "FWYH" else 'N' for aa in seq])
            for k in (3, 4):
                for i in range(len(s_r) - k + 1):
                    c_phys[f"R:{s_r[i:i+k]}"] += 1
            
            s_f = "".join(['R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in seq])
            for k in (3, 4):
                for i in range(len(s_f) - k + 1):
                    c_phys[f"F:{s_f[i:i+k]}"] += 1
            
            # Collect unique sequence
            key = (seq, 
                   str(v_call) if pd.notna(v_call) else '', 
                   str(j_call) if pd.notna(j_call) else '')
            if key not in seen:
                seen.add(key)
                unique_seqs.append(key)
        
        del df, seen
        return c_seq, c_gene, c_phys, unique_seqs
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_file_complete(filepath):
    """
    OPTIMIZED: Process a single file to extract:
    1. Per-file feature counts (for vectorization)
    2. Unique sequences (for sequence scoring)
    Inlined feature extraction to avoid Counter creation per row.
    
    Returns:
        Tuple of (sample_id, seq_counter, gene_counter, phys_counter, unique_sequences_list) or None
    """
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        sample_id = Path(filepath).stem
        
        c_seq, c_gene, c_phys = Counter(), Counter(), Counter()
        unique_seqs = []
        seen = set()
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            # Inline feature extraction - no Counter creation per row
            seq = str(junction) if pd.notna(junction) else ""
            if len(seq) < 4:
                continue
            
            # 1. GENES
            if pd.notna(v_call) and str(v_call) != "":
                c_gene[f"V:{v_call}"] += 1
            if pd.notna(j_call) and str(j_call) != "":
                c_gene[f"J:{j_call}"] += 1
            
            # 2. SEQUENCE K-mers (k=4,5,6)
            for k in (4, 5, 6):
                for i in range(len(seq) - k + 1):
                    c_seq[seq[i:i+k]] += 1
            
            # 3. PHYSICS FEATURES
            s_c = seq.translate(TRANS_CHARGE)
            for k in (3, 4):
                for i in range(len(s_c) - k + 1):
                    c_phys[f"C:{s_c[i:i+k]}"] += 1
            
            s_s = seq.translate(TRANS_SIZE)
            for k in (3, 4):
                for i in range(len(s_s) - k + 1):
                    c_phys[f"S:{s_s[i:i+k]}"] += 1
            
            s_r = "".join(['R' if aa in "FWYH" else 'N' for aa in seq])
            for k in (3, 4):
                for i in range(len(s_r) - k + 1):
                    c_phys[f"R:{s_r[i:i+k]}"] += 1
            
            s_f = "".join(['R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in seq])
            for k in (3, 4):
                for i in range(len(s_f) - k + 1):
                    c_phys[f"F:{s_f[i:i+k]}"] += 1
            
            # Collect unique sequence
            key = (seq, 
                   str(v_call) if pd.notna(v_call) else '', 
                   str(j_call) if pd.notna(j_call) else '')
            if key not in seen:
                seen.add(key)
                unique_seqs.append(key)
        
        del df, seen
        return sample_id, c_seq, c_gene, c_phys, unique_seqs
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def vectorize_file(args):
    """
    Convert a single file to sparse vector representation using vocabulary.
    OPTIMIZED: Inlined feature extraction with direct vocab lookup.
    
    Args:
        args: Tuple of (filepath, v_seq, v_gene, v_phys)
    
    Returns:
        Tuple of (sample_id, seq_dict, gene_dict, phys_dict) or None on error
    """
    filepath, v_seq, v_gene, v_phys = args
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        # Get sample ID (filename without extension)
        sample_id = Path(filepath).stem
        
        d_seq, d_gene, d_phys = {}, {}, {}
        
        for row in df.itertuples(index=False):
            junction = getattr(row, 'junction_aa', None)
            v_call = getattr(row, 'v_call', None)
            j_call = getattr(row, 'j_call', None)
            
            # Inline feature extraction with direct vocab lookup
            seq = str(junction) if pd.notna(junction) else ""
            if len(seq) < 4:
                continue
            
            # 1. GENES - direct vocab lookup
            if pd.notna(v_call) and str(v_call) != "":
                idx = v_gene.get(f"V:{v_call}")
                if idx is not None:
                    d_gene[idx] = d_gene.get(idx, 0) + 1
            if pd.notna(j_call) and str(j_call) != "":
                idx = v_gene.get(f"J:{j_call}")
                if idx is not None:
                    d_gene[idx] = d_gene.get(idx, 0) + 1
            
            # 2. SEQUENCE K-mers (k=4,5,6)
            for k in (4, 5, 6):
                for i in range(len(seq) - k + 1):
                    idx = v_seq.get(seq[i:i+k])
                    if idx is not None:
                        d_seq[idx] = d_seq.get(idx, 0) + 1
            
            # 3. PHYSICS FEATURES
            s_c = seq.translate(TRANS_CHARGE)
            for k in (3, 4):
                for i in range(len(s_c) - k + 1):
                    idx = v_phys.get(f"C:{s_c[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_s = seq.translate(TRANS_SIZE)
            for k in (3, 4):
                for i in range(len(s_s) - k + 1):
                    idx = v_phys.get(f"S:{s_s[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_r = "".join(['R' if aa in "FWYH" else 'N' for aa in seq])
            for k in (3, 4):
                for i in range(len(s_r) - k + 1):
                    idx = v_phys.get(f"R:{s_r[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
            
            s_f = "".join(['R' if aa == 'P' else ('F' if aa in 'GS' else 'N') for aa in seq])
            for k in (3, 4):
                for i in range(len(s_f) - k + 1):
                    idx = v_phys.get(f"F:{s_f[i:i+k]}")
                    if idx is not None:
                        d_phys[idx] = d_phys.get(idx, 0) + 1
        
        del df
        return sample_id, d_seq, d_gene, d_phys
    except Exception as e:
        print(f"Error vectorizing {filepath}: {e}")
        return None


def build_vocabulary(train_files: list, n_cores: int = None, min_count_seq: int = 5,
                     min_count_gene: int = 1, min_count_phys: int = 5,
                     collect_sequences: bool = False):
    """
    Build vocabulary from training files.
    
    Args:
        train_files: List of file paths
        n_cores: Number of CPU cores to use
        min_count_seq: Minimum count for sequence features
        min_count_gene: Minimum count for gene features
        min_count_phys: Minimum count for physics features
        collect_sequences: Whether to also collect unique sequences
    
    Returns:
        Dictionary with 'seq', 'gene', 'phys' vocabularies
        If collect_sequences=True, also returns list of unique sequences
    """
    if n_cores is None:
        n_cores = N_CORES
    
    gc_seq, gc_gene, gc_phys = Counter(), Counter(), Counter()
    all_unique_seqs = set() if collect_sequences else None
    
    # Choose processing function
    process_func = process_file_for_vocab_and_sequences if collect_sequences else process_file_for_vocab
    
    # Process in chunks
    chunk_size = max(1, len(train_files) // 4)
    
    with Pool(n_cores) as p:
        for i in range(0, len(train_files), chunk_size):
            chunk = train_files[i:i+chunk_size]
            results = list(tqdm(p.imap(process_func, chunk),
                               total=len(chunk),
                               desc=f"   Building vocab {i+1}-{min(i+chunk_size, len(train_files))}"))
            
            for res in results:
                if res:
                    if collect_sequences:
                        gc_seq.update(res[0])
                        gc_gene.update(res[1])
                        gc_phys.update(res[2])
                        for seq_tuple in res[3]:
                            all_unique_seqs.add(seq_tuple)
                    else:
                        gc_seq.update(res[0])
                        gc_gene.update(res[1])
                        gc_phys.update(res[2])
            
            del results
            gc.collect()
    
    # Build vocabularies with minimum count filtering
    def make_vocab(counter, min_count):
        return {k: i for i, k in enumerate([k for k, v in counter.items() if v >= min_count])}
    
    vocab = {
        'seq': make_vocab(gc_seq, min_count_seq),
        'gene': make_vocab(gc_gene, min_count_gene),
        'phys': make_vocab(gc_phys, min_count_phys)
    }
    
    del gc_seq, gc_gene, gc_phys
    gc.collect()
    
    if collect_sequences:
        return vocab, list(all_unique_seqs)
    return vocab


def vectorize_files(files: list, vocab: dict, labels_dict: dict = None, 
                    n_cores: int = None, is_test: bool = False):
    """
    Vectorize files into sparse matrices.
    
    Args:
        files: List of file paths
        vocab: Dictionary with 'seq', 'gene', 'phys' vocabularies
        labels_dict: Dictionary mapping patient_id to label (for training data)
        n_cores: Number of CPU cores to use
        is_test: Whether this is test data (no labels)
    
    Returns:
        Dictionary with:
            'ids': sample IDs
            'y': labels (or -1 for test)
            'X_seq': sparse sequence feature matrix
            'X_gene': sparse gene feature matrix
            'X_phys': sparse physics feature matrix
    """
    if n_cores is None:
        n_cores = N_CORES
    
    v_seq = vocab['seq']
    v_gene = vocab['gene']
    v_phys = vocab['phys']
    
    tasks = [(f, v_seq, v_gene, v_phys) for f in files]
    
    # Process in chunks
    chunk_size = max(1, len(tasks) // 4)
    all_results = []
    
    with Pool(n_cores) as p:
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            results = list(tqdm(p.imap(vectorize_file, chunk),
                               total=len(chunk),
                               desc=f"   Vectorizing {i+1}-{min(i+chunk_size, len(tasks))}"))
            all_results.extend(results)
            gc.collect()
    
    # Build sparse matrices
    sample_ids = []
    y_list = []
    
    seq_data, seq_row, seq_col = [], [], []
    gene_data, gene_row, gene_col = [], [], []
    phys_data, phys_row, phys_col = [], [], []
    
    row_idx = 0
    for res in all_results:
        if res is None:
            continue
        sample_id, ds, dg, dp = res
        
        # Get label
        if is_test:
            y_val = -1
        else:
            label = labels_dict.get(sample_id, None)
            if label is None:
                continue
            y_val = 1 if label else 0
        
        sample_ids.append(sample_id)
        y_list.append(y_val)
        
        # Add to COO data
        for col, val in ds.items():
            seq_data.append(val)
            seq_row.append(row_idx)
            seq_col.append(col)
        
        for col, val in dg.items():
            gene_data.append(val)
            gene_row.append(row_idx)
            gene_col.append(col)
        
        for col, val in dp.items():
            phys_data.append(val)
            phys_row.append(row_idx)
            phys_col.append(col)
        
        row_idx += 1
    
    del all_results
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
    
    X_seq = build_sparse(seq_data, seq_row, seq_col, len(v_seq))
    X_gene = build_sparse(gene_data, gene_row, gene_col, len(v_gene))
    X_phys = build_sparse(phys_data, phys_row, phys_col, len(v_phys))
    
    return {
        'ids': np.array(sample_ids),
        'y': np.array(y_list),
        'X_seq': X_seq,
        'X_gene': X_gene,
        'X_phys': X_phys
    }


def extract_features_from_directory(train_dir: str, test_dirs: list = None,
                                    n_cores: int = None, collect_sequences: bool = True):
    """
    OPTIMIZED: Extract features from training and test directories in SINGLE PASS.
    
    This version reads each file only ONCE, caching feature counts for vectorization.
    
    Args:
        train_dir: Path to training directory
        test_dirs: List of paths to test directories (optional)
        n_cores: Number of CPU cores to use
        collect_sequences: Whether to collect unique sequences for top-k selection
    
    Returns:
        Dictionary with:
            'train': training data dict
            'test': dict of test_name -> test data dict
            'vocab': vocabulary dict
            'unique_sequences': DataFrame of unique sequences (if collect_sequences=True)
    """
    import os
    import glob
    
    if n_cores is None:
        n_cores = N_CORES
    
    # Load metadata
    metadata_path = os.path.join(train_dir, "metadata.csv")
    labels_dict = {}
    df = pd.read_csv(metadata_path)
    for row in df.itertuples(index=False):
        rep_id = str(row.repertoire_id)
        label = row.label_positive
        if isinstance(label, str):
            label = label.lower() == 'true'
        labels_dict[rep_id] = label
    
    # Get training files
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.tsv")))
    print(f"   Found {len(train_files)} training files")
    
    # ========== SINGLE PASS: Read files once, cache counts ==========
    print(f"   Extracting features (single pass)...")
    
    gc_seq, gc_gene, gc_phys = Counter(), Counter(), Counter()
    all_unique_seqs = set() if collect_sequences else None
    file_counts = {}  # Cache: sample_id -> (c_seq, c_gene, c_phys)
    
    # Process in chunks
    chunk_size = max(1, len(train_files) // 4)
    
    with Pool(n_cores) as p:
        for i in range(0, len(train_files), chunk_size):
            chunk = train_files[i:i+chunk_size]
            results = list(tqdm(p.imap(process_file_complete, chunk),
                               total=len(chunk),
                               desc=f"   Processing {i+1}-{min(i+chunk_size, len(train_files))}"))
            
            for res in results:
                if res:
                    sample_id, c_seq, c_gene, c_phys, unique_seqs = res
                    # Aggregate for vocab building
                    gc_seq.update(c_seq)
                    gc_gene.update(c_gene)
                    gc_phys.update(c_phys)
                    # Cache per-file counts for vectorization
                    file_counts[sample_id] = (c_seq, c_gene, c_phys)
                    # Collect unique sequences
                    if collect_sequences:
                        for seq_tuple in unique_seqs:
                            all_unique_seqs.add(seq_tuple)
            
            del results
            gc.collect()
    
    # Build vocabularies with minimum count filtering
    def make_vocab(counter, min_count):
        return {k: i for i, k in enumerate([k for k, v in counter.items() if v >= min_count])}
    
    vocab = {
        'seq': make_vocab(gc_seq, 5),
        'gene': make_vocab(gc_gene, 1),
        'phys': make_vocab(gc_phys, 5)
    }
    
    del gc_seq, gc_gene, gc_phys
    gc.collect()
    
    print(f"   Vocab sizes: Seq={len(vocab['seq'])}, Gene={len(vocab['gene'])}, Phys={len(vocab['phys'])}")
    
    # ========== VECTORIZE FROM CACHE (no file re-reading!) ==========
    print(f"   Vectorizing from cache...")
    
    v_seq = vocab['seq']
    v_gene = vocab['gene']
    v_phys = vocab['phys']
    
    sample_ids = []
    y_list = []
    seq_data, seq_row, seq_col = [], [], []
    gene_data, gene_row, gene_col = [], [], []
    phys_data, phys_row, phys_col = [], [], []
    
    row_idx = 0
    for f in train_files:
        sample_id = Path(f).stem
        if sample_id not in file_counts:
            continue
        
        # Get label
        label = labels_dict.get(sample_id, None)
        if label is None:
            continue
        
        c_seq, c_gene, c_phys = file_counts[sample_id]
        
        sample_ids.append(sample_id)
        y_list.append(1 if label else 0)
        
        # Map counts to vocab indices
        for key, cnt in c_gene.items():
            idx = v_gene.get(key)
            if idx is not None:
                gene_data.append(cnt)
                gene_row.append(row_idx)
                gene_col.append(idx)
        
        for key, cnt in c_seq.items():
            idx = v_seq.get(key)
            if idx is not None:
                seq_data.append(cnt)
                seq_row.append(row_idx)
                seq_col.append(idx)
        
        for key, cnt in c_phys.items():
            idx = v_phys.get(key)
            if idx is not None:
                phys_data.append(cnt)
                phys_row.append(row_idx)
                phys_col.append(idx)
        
        row_idx += 1
    
    # Clear cache
    del file_counts
    gc.collect()
    
    n_samples = len(sample_ids)
    
    def build_sparse(data, row, col, n_cols):
        if data:
            return sparse.csr_matrix(
                (data, (row, col)),
                shape=(n_samples, n_cols),
                dtype=np.float32
            )
        return sparse.csr_matrix((n_samples, n_cols), dtype=np.float32)
    
    X_seq = build_sparse(seq_data, seq_row, seq_col, len(v_seq))
    X_gene = build_sparse(gene_data, gene_row, gene_col, len(v_gene))
    X_phys = build_sparse(phys_data, phys_row, phys_col, len(v_phys))
    
    del seq_data, seq_row, seq_col
    del gene_data, gene_row, gene_col
    del phys_data, phys_row, phys_col
    gc.collect()
    
    train_data = {
        'ids': np.array(sample_ids),
        'y': np.array(y_list),
        'X_seq': X_seq,
        'X_gene': X_gene,
        'X_phys': X_phys
    }
    
    print(f"   Training samples: {len(train_data['ids'])}")
    
    # Vectorize test data (still needs to read test files, but only once each)
    test_data = {}
    if test_dirs:
        for test_dir in test_dirs:
            if not os.path.exists(test_dir):
                continue
            test_name = os.path.basename(test_dir)
            test_files = sorted(glob.glob(os.path.join(test_dir, "*.tsv")))
            if not test_files:
                continue
            print(f"   Vectorizing {test_name}...")
            test_data[test_name] = vectorize_files(test_files, vocab, n_cores=n_cores, is_test=True)
    
    result = {
        'train': train_data,
        'test': test_data,
        'vocab': vocab
    }
    
    if collect_sequences:
        # Use seeded shuffle for deterministic ordering (O(n) vs O(n log n) for sort)
        import random
        unique_seqs_list = list(all_unique_seqs)
        random.Random(SEED).shuffle(unique_seqs_list)
        unique_sequences_df = pd.DataFrame(unique_seqs_list, columns=['junction_aa', 'v_call', 'j_call'])
        result['unique_sequences'] = unique_sequences_df
    
    return result