#!/usr/bin/env python3
"""
Sanity check script for split-ix functionality.

Runs bnci classification with lda model:
1. All 5 splits together (no split-ix)
2. Each split separately (split-ix 0-4)

Then verifies that the combined separate splits match the all-splits run.
"""
import subprocess
from fnirsatlas.results import Results
import numpy as np


def run_classification(dataset="bnci", model="lda", seed=42, split_ix=None, override=False):
    """Run classification and return True if successful."""
    cmd = [
        "python", "scripts/classify_dataset.py",
        "-d", dataset,
        "-m", model,
        "--seed", str(seed),
    ]
    if split_ix is not None:
        cmd.extend(["--split-ix", str(split_ix)])
    if override:
        cmd.append("--override")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running classification: {result.stderr}")
        return False
    return True


def load_all_splits_results(dataset="bnci", model="lda", seed=42):
    """Load results from all-splits run."""
    results = Results(dataset=dataset, model=model, seeds=[seed])
    results.load()
    return results


def load_split_files(dataset="bnci", model="lda", seed=42):
    """Load and combine split files - automatically tries split files if main file not found."""
    # First delete the main file to force loading of split files
    import os
    from fnirsatlas.results import Results
    main_path = Results.get_results_file_path(dataset, "cross", model, [seed])
    if main_path.exists():
        os.remove(main_path)
    
    results = Results(dataset=dataset, model=model, seeds=[seed])
    results.load()
    return results


def compare_results(all_splits, combined_splits):
    """Compare all-splits vs combined split files."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    match = True
    for task in all_splits.task_dict.keys():
        print(f"\nTask: {task}")
        
        for metric in ["cross_acc", "cross_kappa"]:
            if metric in all_splits.task_dict[task]:
                all_values = np.array(all_splits.task_dict[task][metric])
                combined_values = np.array(combined_splits.task_dict[task][metric])
                
                print(f"  {metric}:")
                print(f"    All splits:    {all_values}")
                print(f"    From splits:   {combined_values}")
                
                if len(all_values) != len(combined_values):
                    print(f"    ERROR: Length mismatch!")
                    match = False
                elif not np.allclose(all_values, combined_values):
                    print(f"    ERROR: Values don't match!")
                    match = False
                else:
                    print(f"    OK: Values match!")
    
    print("\n" + "="*60)
    if match:
        print("SUCCESS: All splits match combined results!")
    else:
        print("FAILURE: Results don't match!")
    print("="*60)
    
    return match


def main():
    dataset = "bnci"
    model = "lda"
    seed = 42

    print("="*60)
    print("SANITY CHECK: split-ix functionality")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Seed: {seed}")
    print()

    # Step 1: Run all splits together
    print("Step 1: Running all splits together...")
    if not run_classification(dataset, model, seed, split_ix=None, override=True):
        print("Failed to run all splits!")
        return 1

    # Step 2: Load results from all-splits run
    print("\nStep 2: Loading all-splits results...")
    all_splits = load_all_splits_results(dataset, model, seed)

    # Step 3: Delete main file and run each split separately
    print("\nStep 3: Running each split separately...")
    main_path = Results.get_results_file_path(dataset, "cross", model, [seed])
    if main_path.exists():
        main_path.unlink()
        print(f"Deleted main file: {main_path}")

    for split_ix in range(5):
        if not run_classification(dataset, model, seed, split_ix=split_ix, override=True):
            print(f"Failed to run split {split_ix}!")
            return 1

    # Step 4: Load combined split files (automatic fallback since main file deleted)
    print("\nStep 4: Loading combined split files...")
    combined_splits = Results(dataset=dataset, model=model, seeds=[seed])
    combined_splits.load()

    # Step 5: Compare results
    print("\nStep 5: Comparing results...")
    match = compare_results(all_splits, combined_splits)

    return 0 if match else 1


if __name__ == "__main__":
    exit(main())
