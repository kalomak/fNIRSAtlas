#!/usr/bin/env python3
"""
Main script to run the complete pipeline for all datasets.

This script orchestrates the three separate steps:
1. Download datasets (download_dataset.py)
2. Process datasets (process_dataset.py)
3. Classify datasets (classify_dataset.py)
"""

import os
import concurrent.futures
import subprocess
import sys
import logging
import argparse
from collections import defaultdict
import queue
import re
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
)
from fnirsatlas.utils import get_datasets_from_args

logger = logging.getLogger(__name__)

PIPELINE_PHASES = ["Download", "Process", "Classify"]
progress_re = re.compile(r"(\d+)%")
error_re = re.compile(r"^Traceback", re.IGNORECASE)
exception_lastline_re = re.compile(r"^[A-Za-z_][\w\.]*: .+")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete pipeline for specified datasets or suites"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Specific datasets to process. If not provided, all available datasets will be processed.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        help="Suite name to process (e.g., motor, MA, MI, nback, Emotion, MISC)",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=8,
        help="Number of jobs for parallel execution (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation (default: 42)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lda",
        help="Model to use for classification (e.g. lda, logreg, svm, rf, knn)",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip the classification step (only download and process datasets)",
    )
    parser.add_argument(
        "--no_table",
        action="store_true",
        help="Skip displaying the final results table",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-process datasets and overwrite saved outputs even if they already exist.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        help="Log level. Use ERROR for clean output (default: ERROR)",
    )
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        default="cross",
        help="Classification setting: cross or within",
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        default="none",
        choices=["none", "nested", "single", "block"],
        help="Parameter search strategy",
    )
    return parser.parse_args()


def run_command(cmd, description, env=None, on_progress=None, show_stdout=False):
    """Run a command (always streaming) and return (success, stderr_text).

    Streams stderr to parse percent updates if they appear; on_progress can be None.
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    subprocess_env = env or os.environ.copy()

    proc = subprocess.Popen(
        cmd,
        env=subprocess_env,
        stdout=None if show_stdout else subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    traceback_seen = False
    last_exception_line = ""
    collected_stderr = []
    for line in iter(proc.stderr.readline, ""):
        collected_stderr.append(line)
        if not traceback_seen and error_re.search(line):
            traceback_seen = True
        if traceback_seen and exception_lastline_re.match(line):
            last_exception_line = line.strip()
        if on_progress is not None:
            m = progress_re.search(line)
            if m:
                try:
                    on_progress(int(m.group(1)))
                except Exception:
                    pass
    proc.stderr.close()
    proc.wait()
    success = (proc.returncode == 0) and (not traceback_seen)
    if success:
        logger.info(f"{description} completed successfully")
    else:
        logger.debug(f"{description} failed with exit code {proc.returncode}")
    return success, "".join(collected_stderr)


def download_dataset(dataset_name, on_progress=None, **kwargs):
    download_cmd = [
        sys.executable,
        "scripts/download_dataset.py",
        "-d",
        dataset_name,
    ]
    return run_command(
        download_cmd,
        f"Download datasets ({dataset_name})",
        on_progress=on_progress,
        **kwargs,
    )


def process_dataset(dataset_name, args, on_progress=None, **kwargs):
    process_cmd = [
        sys.executable,
        "scripts/process_dataset.py",
        "-d",
        dataset_name,
    ]
    if args.override:
        process_cmd.append("--override")
    return run_command(
        process_cmd,
        f"Process datasets ({dataset_name})",
        on_progress=on_progress,
        **kwargs,
    )


def classify_dataset(dataset_name, args, on_progress=None, **kwargs):
    classify_cmd = [
        sys.executable,
        "scripts/classify_dataset.py",
        "-d",
        dataset_name,
        "--seed",
        str(args.seed),
        "--model",
        args.model,
        "--n_jobs",
        "1",
        "--no_table",
        "--classification_setting",
        args.classification_setting,
        "--search-strategy",
        args.search_strategy,
    ]
    return run_command(
        classify_cmd,
        f"Classify datasets ({dataset_name})",
        on_progress=on_progress,
        **kwargs,
    )


def enqueue_error(error_queue, dataset, step, stderr_text):
    error_queue.put(
        {
            "dataset": dataset,
            "step": step,
            "stderr": (stderr_text or "").strip(),
        }
    )


def results_exist(dataset_name, args):
    """Check if classification results exist for a dataset."""
    from fnirsatlas.utils import build_result_path

    return build_result_path(
        args.classification_setting,
        args.model,
        dataset_name,
        args.seed,
        search_strategy=args.search_strategy,
    ).exists()


def _short_error(stderr_text: str) -> str:
    if not stderr_text:
        return ""
    lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Manual download required" in ln:
            return ln.replace("RuntimeError: ", "").strip()
        return ln
    return ""


def run_single_dataset(base_name, dataset_parts, args, progress, task_id, error_queue):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = str(1)

    def advance(phase_name):
        progress.update(
            task_id, advance=1, description=f"[cyan]{base_name}[/cyan] [{phase_name}]"
        )

    # Download
    ok, stderr_text = download_dataset(base_name, on_progress=None, env=env)
    if not ok:
        progress.update(task_id, description=f"[red]{base_name}[/red] [FAILED]")
        for dt in dataset_parts:
            enqueue_error(error_queue, dt, "download", stderr_text)
        return 1
    advance("Process")

    # Process
    for dt in dataset_parts:
        ok, stderr_text = process_dataset(dt, args, on_progress=None, env=env)
        if not ok:
            progress.update(task_id, description=f"[red]{base_name}[/red] [FAILED]")
            enqueue_error(error_queue, dt, "process", stderr_text)
            return 1
    advance("Classify")

    # Classify
    if args.skip_classification:
        advance("Done")
    else:
        all_results_exist = all(results_exist(dt, args) for dt in dataset_parts)
        if all_results_exist and not args.override:
            advance("Done")
        else:
            for dt in dataset_parts:
                ok, stderr_text = classify_dataset(dt, args, on_progress=None, env=env)
                if not ok:
                    progress.update(
                        task_id, description=f"[red]{base_name}[/red] [FAILED]"
                    )
                    enqueue_error(error_queue, dt, "classify", stderr_text)
                    return 1
            advance("Done")

    return 0


def main():
    """Run the complete pipeline for all datasets."""
    args = parse_arguments()

    # Set log level as environment variable to be used by all scripts
    os.environ["FNIRSATLAS_LOG_LEVEL"] = args.log_level
    datasets = get_datasets_from_args(args.datasets, args.suite)

    # Points base_name to list of dataset_names with task, e.g. shin2019 --> [shin2019__ma, shin2019__mi] to avoid re-downloading the same base dataset multiple times
    dataset_parts = defaultdict(list)
    for dataset in datasets:
        base_name = dataset.split("__")[0]
        dataset_parts[base_name].append(dataset)

    error_queue = queue.Queue()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description:<40}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_ids = {
            base_name: progress.add_task(
                f"[cyan]{base_name}[/cyan] [Download]", total=3
            )
            for base_name in dataset_parts.keys()
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [
                executor.submit(
                    run_single_dataset,
                    base_name,
                    dataset_parts[base_name],
                    args,
                    progress,
                    task_ids[base_name],
                    error_queue,
                )
                for base_name in dataset_parts.keys()
            ]
            concurrent.futures.wait(futures)

    if not args.no_table:
        display_cmd = [
            sys.executable,
            "scripts/display_results.py",
            "--seeds",
            str(args.seed),
            "-m",
            args.model,
            "-c",
            args.classification_setting,
            "--search-strategy",
            args.search_strategy,
        ]
        if args.datasets:
            display_cmd.extend(["-d"] + args.datasets)
        if args.suite:
            display_cmd.extend(["-s", args.suite])
        run_command(display_cmd, "Display results", show_stdout=True)

    # Print collected error summary at the end
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    if errors:
        print("\nError summary:")
        for err in errors:
            dataset = err.get("dataset", "unknown")
            step = err.get("step", "unknown")
            msg = _short_error(err.get("stderr", ""))
            suffix = msg if msg else "(no stderr captured)"
            print(f"- {dataset} [{step}]: {suffix}")


if __name__ == "__main__":
    main()
