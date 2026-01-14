# Hybrid Stacking Ensemble for TCR Immune State Prediction

A hybrid stacking ensemble model for predicting immune states from T-cell receptor (TCR) repertoire data, developed for the [Adaptive Immune Profiling Challenge 2025](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025).

## Approach Overview

This solution uses a **6-specialist stacking ensemble** where each specialist extracts different aspects of TCR repertoire data, and a meta-learner (HEAD) combines their predictions.

### Specialists

| Specialist | Features | Description |
|------------|----------|-------------|
| **Physicochemical** | Physics-based k-mers (k=3,4) | Charge, size, ring (aromatic), and flexibility patterns |
| **Kmer** | Sequence k-mers (k=4,5,6) | Direct amino acid sequence patterns |
| **VJGene** | V/J gene usage | Gene segment frequencies |
| **ReactiveTCR** | Chi-squared selection | Identifies reactive TCR combinations (CDR3 + V-gene) |
| **Statistical** | Statistical features | Entropy, Gini-Simpson diversity, length moments |
| **Frequency** | Frequency features | V/J/AA/length distributions |

### Meta-Learner (HEAD)

- L1-regularized Logistic Regression with cross-validated regularization strength
- Combines specialist predictions using learned weights
- Fallback mechanism: `max(HEAD, best_specialist)` ensures performance is never worse than the best individual specialist

### Key Design Decisions

- **No data leakage**: Feature selection (ReactiveTCR chi-squared, effect sizes) happens inside CV folds using only training data
- **Single-pass file reading**: Each file is read once and cached for all feature extractors
- **Parallel processing**: All specialists trained in parallel across CV folds

## Installation
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- tqdm >= 4.62.0
- joblib >= 1.1.0

## Usage

### Command Line Interface
```bash
# Single dataset
python3 -m submission.main \
    --train_dir /path/to/train_dataset_1 \
    --test_dirs /path/to/test_dataset_1 \
    --out_dir /path/to/output \
    --n_jobs 4 \
    --device cpu

# Multiple test directories (e.g., dataset 7 with test_7_1 and test_7_2)
python3 -m submission.main \
    --train_dir /path/to/train_dataset_7 \
    --test_dirs /path/to/test_dataset_7_1 /path/to/test_dataset_7_2 \
    --out_dir /path/to/output \
    --n_jobs 4 \
    --device cpu
```

### Python API
```python
from submission.predictor import ImmuneStatePredictor
from submission.utils import get_dataset_pairs, concatenate_output_files

# Single dataset
predictor = ImmuneStatePredictor(n_jobs=4, device='cpu')
predictor.fit('/path/to/train_dataset_1')
predictions = predictor.predict_proba('/path/to/test_dataset_1')

# All datasets
train_datasets_dir = "/path/to/train_datasets"
test_datasets_dir = "/path/to/test_datasets"
results_dir = "/path/to/results"

train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)

for train_dir, test_dirs in train_test_dataset_pairs:
    predictor = ImmuneStatePredictor(n_jobs=4, device='cpu')
    predictor.fit(train_dir)
    for test_dir in test_dirs:
        predictions = predictor.predict_proba(test_dir)
        # Save predictions...

# Generate final submission file
concatenate_output_files(out_dir=results_dir)
```

## Output Format

The model produces two types of output files per dataset:

### 1. Test Predictions (`{dataset}_test_predictions.tsv`)

| Column | Description |
|--------|-------------|
| ID | Repertoire ID |
| dataset | Test dataset name |
| label_positive_probability | Predicted probability (0-1) |
| junction_aa | -999.0 (placeholder) |
| v_call | -999.0 (placeholder) |
| j_call | -999.0 (placeholder) |

### 2. Important Sequences (`{dataset}_important_sequences.tsv`)

| Column | Description |
|--------|-------------|
| ID | Sequence ID (e.g., train_dataset_1_seq_top_1) |
| dataset | Training dataset name |
| label_positive_probability | -999.0 (placeholder) |
| junction_aa | CDR3 amino acid sequence |
| v_call | V gene call |
| j_call | J gene call |

## Project Structure
```
submission/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point
├── predictor.py             # ImmuneStatePredictor class
├── feature_extraction.py    # Base feature extraction (k-mers, physics, genes)
├── base_specialists.py      # Physicochemical, Kmer, VJGene specialists
├── atttcr_specialist.py     # ReactiveTCR chi-squared specialist
├── xgboost_specialist.py    # Statistical and Frequency specialists
└── utils.py                 # Utility functions (data loading, saving)
```

## Docker

Build and run with Docker:
```bash
# Build
docker build -t immune-predictor .

# Run
docker run -v /path/to/data:/data -v /path/to/output:/output immune-predictor \
    --train_dir /data/train_dataset_1 \
    --test_dirs /data/test_dataset_1 \
    --out_dir /output \
    --n_jobs 4
```

## License

This project is open-source and available for research purposes.

## Acknowledgments

Developed for the [Adaptive Immune Profiling Challenge 2025](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025) organized by the University of Oslo.