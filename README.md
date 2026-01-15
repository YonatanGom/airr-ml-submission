# Hybrid Stacking Ensemble for TCR Immune State Prediction

A hybrid stacking ensemble model for predicting immune states from T-cell receptor (TCR) repertoire data, developed for the [Adaptive Immune Profiling Challenge 2025](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025).

## Approach

This solution uses a **6-specialist stacking ensemble** where each specialist extracts different aspects of TCR repertoire data, and a meta-learner combines their predictions.

### Specialists

| Specialist | Features | Description |
|------------|----------|-------------|
| **Physicochemical** | Physics-based k-mers (k=3,4) | Charge, size, ring, and flexibility patterns |
| **Kmer** | Sequence k-mers (k=4,5,6) | Amino acid sequence patterns |
| **VJGene** | V/J gene usage | Gene segment frequencies |
| **ReactiveTCR** | Chi-squared selection | Reactive TCR combinations (CDR3 + V-gene) |
| **Statistical** | Statistical features | Entropy, Gini-Simpson diversity, length moments |
| **Frequency** | Frequency features | V/J/AA/length distributions |

### Meta-Learner

L1-regularized Logistic Regression with cross-validated regularization strength.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
python3 -m submission.main \
    --train_dir /path/to/train_dataset_1 \
    --test_dirs /path/to/test_dataset_1 \
    --out_dir /path/to/output \
    --n_jobs 10 \
    --device cpu
```

### Multiple Test Sets
```bash
python3 -m submission.main \
    --train_dir /path/to/train_dataset_7 \
    --test_dirs /path/to/test_dataset_7_1 /path/to/test_dataset_7_2 \
    --out_dir /path/to/output \
    --n_jobs 10 \
    --device cpu
```

## Docker
```bash
docker build -t immune-predictor .

docker run -v /path/to/data:/data -v /path/to/output:/output immune-predictor \
    --train_dir /data/train_dataset_1 \
    --test_dirs /data/test_dataset_1 \
    --out_dir /output \
    --n_jobs 10
```

## Project Structure
```
submission/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point
├── predictor.py             # ImmuneStatePredictor class
├── feature_extraction.py    # Feature extraction
├── base_specialists.py      # Physicochemical, Kmer, VJGene
├── atttcr_specialist.py     # ReactiveTCR specialist
├── xgboost_specialist.py    # Statistical and Frequency specialists
└── utils.py                 # Utility functions
```

## License

Open-source for research purposes.