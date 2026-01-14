#!/usr/bin/env python3
"""
Main entry point for the Immune State Predictor.

Usage:
    python3 -m submission.main --train_dir /path/to/train_dir --test_dirs /path/to/test_dir_1 /path/to/test_dir_2 --out_dir /path/to/output_dir --n_jobs 4 --device cpu

Or with multiple datasets:
    train_datasets_dir = "/path/to/train_datasets"
    test_datasets_dir = "/path/to/test_datasets"
    results_dir = "/path/to/results"
    
    train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)
    
    for train_dir, test_dirs in train_test_dataset_pairs:
        main(train_dir=train_dir, test_dirs=test_dirs, out_dir=results_dir, n_jobs=4, device="cpu")
"""
import os
import sys
import argparse
from typing import List

from .predictor import ImmuneStatePredictor
from .utils import save_tsv, validate_dirs_and_files, concatenate_output_files, get_dataset_pairs


def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Train the predictor on the training data."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {os.path.basename(train_dir)}")
    print(f"{'='*60}")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]):
    """Generate predictions for all test directories."""
    import pandas as pd
    
    all_preds = []
    for test_dir in test_dirs:
        print(f"\n   Predicting on {os.path.basename(test_dir)}...")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"   ⚠️ No predictions returned for {test_dir}")
    
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return None


def _save_predictions(predictions, out_dir: str, train_dir: str) -> None:
    """Save predictions to a TSV file."""
    if predictions is None or predictions.empty:
        print("   ⚠️ No predictions to save")
        return
    
    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"   💾 Predictions saved: {preds_path}")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Save important sequences to a TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        print("   ⚠️ No important sequences to save")
        return
    
    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"   💾 Important sequences saved: {seqs_path}")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    """
    Main workflow: train model, predict on test sets, save results.
    
    Args:
        train_dir: Path to training data directory
        test_dirs: List of paths to test data directories
        out_dir: Path to output directory
        n_jobs: Number of CPU cores to use
        device: Device for computation ('cpu' or 'cuda')
    """
    # Validate directories
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    
    # Initialize predictor
    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)
    
    # Train
    _train_predictor(predictor, train_dir)
    
    # Predict
    predictions = _generate_predictions(predictor, test_dirs)
    
    # Save outputs
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)
    
    print(f"\n   ✅ Complete: {os.path.basename(train_dir)}")


def run():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Immune State Predictor - TCR Repertoire Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python3 -m submission.main --train_dir /data/train_dataset_1 --test_dirs /data/test_dataset_1 --out_dir /results --n_jobs 4
  
  # Multiple test sets
  python3 -m submission.main --train_dir /data/train_dataset_7 --test_dirs /data/test_dataset_7_1 /data/test_dataset_7_2 --out_dir /results --n_jobs 4
"""
    )
    
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Path to training data directory containing TSV files and metadata.csv"
    )
    parser.add_argument(
        "--test_dirs",
        required=True,
        nargs="+",
        help="Path(s) to test data directory(ies) containing TSV files"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Path to output directory for predictions and important sequences"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of CPU cores to use. Use -1 for all available cores. (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device to use for computation (default: cpu)"
    )
    
    args = parser.parse_args()
    
    print("🧬 IMMUNE STATE PREDICTOR")
    print("=" * 60)
    print(f"   Train dir:  {args.train_dir}")
    print(f"   Test dirs:  {args.test_dirs}")
    print(f"   Output dir: {args.out_dir}")
    print(f"   N jobs:     {args.n_jobs}")
    print(f"   Device:     {args.device}")
    print("=" * 60)
    
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)
    
    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)


def run_all_datasets(train_datasets_dir: str, test_datasets_dir: str, out_dir: str,
                     n_jobs: int = 4, device: str = 'cpu'):
    """
    Run on all dataset pairs (convenience function for batch processing).
    
    Args:
        train_datasets_dir: Directory containing train_dataset_* folders
        test_datasets_dir: Directory containing test_dataset_* folders
        out_dir: Output directory for results
        n_jobs: Number of CPU cores
        device: Computation device
    """
    print("🧬 IMMUNE STATE PREDICTOR - BATCH MODE")
    print("=" * 60)
    print(f"   Train datasets dir: {train_datasets_dir}")
    print(f"   Test datasets dir:  {test_datasets_dir}")
    print(f"   Output dir:         {out_dir}")
    print(f"   N jobs:             {n_jobs}")
    print(f"   Device:             {device}")
    print("=" * 60)
    
    # Get dataset pairs
    train_test_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)
    
    print(f"\nFound {len(train_test_pairs)} dataset pairs:")
    for train_dir, test_dirs in train_test_pairs:
        print(f"   {os.path.basename(train_dir)} -> {[os.path.basename(t) for t in test_dirs]}")
    
    # Process each pair
    for train_dir, test_dirs in train_test_pairs:
        if not test_dirs:
            print(f"\n⚠️ Skipping {train_dir} - no test directories found")
            continue
        
        main(train_dir=train_dir, test_dirs=test_dirs, out_dir=out_dir,
             n_jobs=n_jobs, device=device)
    
    # Concatenate all outputs
    print("\n" + "=" * 60)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 60)
    concatenate_output_files(out_dir)
    
    print("\n" + "=" * 60)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run()
