#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParAcousti pre-processing routine.

This module processes parAcousti files, which can be resource intensive depending on
grid resolution and weights selected. Processing may take several hours to complete.

Required packages:
    - xarray with netCDF4 (pip install "xarray[io]")
    - scipy (included with xarray[io])
    - tqdm (pip install tqdm)
"""

import argparse
from pathlib import Path
from seat.utils import paracousti_fxns


def process_paracousti_files(
    input_dir: str, output_dir: str, weights: str = "TU"
) -> bool:
    """
    Process parAcousti files and calculate metrics.

    Args:
        input_dir: Directory containing parAcousti files
        output_dir: Directory to save processed files
        weights: Weight type to use ("TU" or "All")

    Returns:
        bool: True if processing successful, False otherwise

    Raises:
        FileNotFoundError: If input directory doesn't exist
        PermissionError: If unable to create output directory
        ValueError: If weight type is invalid
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    return paracousti_fxns.calc_paracousti_metrics(
        str(input_path), str(output_path), weights=weights
    )


def main():
    """Run the main processing routine."""
    parser = argparse.ArgumentParser(description="Process parAcousti files.")
    parser.add_argument("input_dir", help="Directory containing parAcousti files")
    parser.add_argument("output_dir", help="Directory to save processed files")
    parser.add_argument(
        "--weights",
        default="TU",
        choices=["TU", "All"],
        help="Weight type to use (default: TU)",
    )

    args = parser.parse_args()

    try:
        process_success = process_paracousti_files(
            args.input_dir, args.output_dir, weights=args.weights
        )
        print(f"Processing {'successful' if process_success else 'failed'}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"File system error: {e}")
    except ValueError as e:
        print(f"Invalid parameter: {e}")
    except OSError as e:
        print(f"Operating system error: {e}")


if __name__ == "__main__":
    main()
