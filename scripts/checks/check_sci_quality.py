"""
Check data quality using Scalp Coupling Index (SCI) for a given dataset.

Loads all sessions for a dataset, converts to optical density (if not already
in HbO/HbR format), computes SCI, and plots histogram of all SCI ratios.
"""

import argparse
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from fnirsatlas.load_parser import load_parser
from fnirsatlas.preprocessing import inverse_mbll
from fnirsatlas.logging import setup_logging
import mne

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def get_raw_for_subject_session(parser, subject_id, session):
    """Load raw data for a specific subject and session."""
    data = parser.load_data(subject_id, session=session)
    return parser._raw_from_session(data)


def check_and_convert_to_od(raw, subject_id, session):
    """
    Check data format and convert to optical density if needed.
    
    Returns:
        tuple: (raw_od, was_converted, skip_reason)
            - raw_od: Raw object in optical density format
            - was_converted: True if conversion was performed
            - skip_reason: None if OK, otherwise reason to skip
    """
    current_channel_types = set(raw.get_channel_types())
    
    # Check if already in hemoglobin concentration
    if any(ch_type in current_channel_types for ch_type in ("fnirs_hbo", "fnirs_hbr", "hbo", "hbr")):
        logger.debug(
            f"Subject {subject_id}, session {session}: Converting from Hb to OD"
        )
        raw = inverse_mbll(raw)
        return raw, True, None
    
    # Check if in raw amplitude format (needs OD conversion)
    if "fnirs_cw_amplitude" in current_channel_types:
        logger.debug(f"Subject {subject_id}, session {session}: Converting from amplitude to OD")
        raw = mne.preprocessing.nirs.optical_density(raw)
        return raw, True, None
    
    # Check if already in OD format
    if "fnirs_od" in current_channel_types:
        logger.debug(f"Subject {subject_id}, session {session}: Already in OD format")
        return raw, False, None
    
    # Unknown format
    return None, False, f"unknown_format: {current_channel_types}"


def compute_sci_for_raw(raw_od):
    """Compute SCI for all channels in raw OD data."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose=False)
    return sci


def analyze_dataset_sci(dataset_name, plot=True, save_dir=None):
    """
    Analyze SCI quality for all subjects/sessions in a dataset.
    
    Parameters:
        dataset_name: Name of the dataset to analyze
        plot: Whether to display the histogram plot
        save_dir: Override directory for both cached data and plots
    
    Returns:
        DataFrame with SCI values per channel
    """
    if save_dir is None:
        from fnirsatlas.config import get_results_cache_dir, get_results_dir
        save_dir = get_results_cache_dir() / "quality"
        plot_dir = get_results_dir() / "quality"
    else:
        save_dir = Path(save_dir)
        plot_dir = save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset: {dataset_name}")
    parser = load_parser(dataset_name)
    subject_ids = parser.get_subject_ids()
    
    all_sci_data = []
    skipped_subjects = []
    converted_count = 0
    
    for subject_id in subject_ids:
        sessions = parser.get_sessions(subject_id)
        
        for session in sessions:
            try:
                # Load raw data
                raw = get_raw_for_subject_session(parser, subject_id, session)
                
                # Check and convert to OD
                raw_od, was_converted, skip_reason = check_and_convert_to_od(
                    raw, subject_id, session
                )
                
                if skip_reason and skip_reason.startswith("unknown_format"):
                    logger.warning(
                        f"Subject {subject_id}, session {session}: "
                        f"Unknown format ({skip_reason}), skipping"
                    )
                    skipped_subjects.append({
                        "subject": subject_id,
                        "session": session,
                        "reason": skip_reason
                    })
                    continue
                
                if was_converted:
                    converted_count += 1
                
                # Compute SCI
                sci = compute_sci_for_raw(raw_od)
                
                # Collect results per channel
                for ch_name, sci_value in zip(raw_od.ch_names, sci):
                    # Skip Stim channel if present
                    if ch_name == "Stim":
                        continue
                    
                    all_sci_data.append({
                        "dataset": dataset_name,
                        "subject": subject_id,
                        "session": session,
                        "channel": ch_name,
                        "sci": sci_value
                    })
                
            except Exception as e:
                logger.error(
                    f"Subject {subject_id}, session {session}: Error processing - {e}"
                )
                skipped_subjects.append({
                    "subject": subject_id,
                    "session": session,
                    "reason": f"error: {e}"
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_sci_data)
    
    # Summary statistics
    n_subjects = len(subject_ids)
    n_sessions_total = sum(len(parser.get_sessions(sid)) for sid in subject_ids)
    n_processed = len(set(d["subject"] for d in all_sci_data))
    n_skipped = len(skipped_subjects)
    
    print(f"\n{'='*60}")
    print(f"SCI Quality Analysis for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total subjects: {n_subjects}")
    print(f"Total sessions: {n_sessions_total}")
    print(f"Successfully processed: {n_processed} subjects")
    print(f"Skipped: {n_skipped} sessions")
    print(f"OD conversions performed: {converted_count}")
    
    if not df.empty:
        print(f"\nSCI Statistics:")
        print(f"  Total channels analyzed: {len(df)}")
        print(f"  Mean SCI: {df['sci'].mean():.4f}")
        print(f"  Median SCI: {df['sci'].median():.4f}")
        print(f"  Std SCI: {df['sci'].std():.4f}")
        print(f"  Min SCI: {df['sci'].min():.4f}")
        print(f"  Max SCI: {df['sci'].max():.4f}")
        
        # Count channels below common thresholds
        for threshold in [0.0, 0.2, 0.5, 0.7]:
            below = (df['sci'] <= threshold).sum()
            pct = 100 * below / len(df)
            print(f"  Channels with SCI <= {threshold}: {below} ({pct:.1f}%)")
    
    if skipped_subjects:
        print(f"\nSkipped sessions:")
        for skip in skipped_subjects[:10]:  # Show first 10
            print(f"  Subject {skip['subject']}, session {skip['session']}: {skip['reason']}")
        if len(skipped_subjects) > 10:
            print(f"  ... and {len(skipped_subjects) - 10} more")
    
    # Save results
    if not df.empty:
        csv_path = save_dir / f"{dataset_name}_sci_values.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved SCI values to {csv_path}")
        
        # Plot histogram
        fig_path = plot_dir / f"{dataset_name}_sci_histogram.png"
        plot_sci_histogram(df, dataset_name, fig_path, show=plot)
    
    return df


def plot_sci_histogram(df, dataset_name, save_path=None, show=True):
    """Plot histogram of SCI values."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sci_values = df['sci'].values
    
    # Plot histogram
    ax.hist(sci_values, bins=50, range=(-1, 1), alpha=0.7, edgecolor='black')
    ax.axvline(x=0.2, color='r', linestyle='--', linewidth=2, label='Threshold (0.2)')
    ax.axvline(x=np.median(sci_values), color='g', linestyle='-', linewidth=2, 
               label=f'Median ({np.median(sci_values):.3f})')
    
    ax.set_xlabel('Scalp Coupling Index (SCI)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'SCI Distribution for {dataset_name}\n(n={len(sci_values)} channels)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved histogram to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check data quality using Scalp Coupling Index (SCI)"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True,
        help="Dataset name (e.g., luke2024b, shin2019__ma)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Don't display the plot (still saves to file)"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Override directory for cached data and plots"
    )
    
    args = parser.parse_args()
    
    analyze_dataset_sci(
        dataset_name=args.dataset,
        plot=not args.no_plot,
        save_dir=args.save_dir
    )
