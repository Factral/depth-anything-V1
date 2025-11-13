#!/usr/bin/env python3
"""
Convert .mat files to .npy, recursively.

- Accepts a DIRECTORY only (recursively processes all .mat files)
- Requires an OUTPUT directory and mirrors structure under it
- Skips MATLAB metadata keys (those starting with '__'); saves ONLY the first variable's value
"""

import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm


def convert_mat_to_clean_dict(mat_data: Dict) -> Dict:
    """Return a dict stripped of MATLAB metadata keys."""
    return {key: value for key, value in mat_data.items() if not key.startswith('__')}


def find_mat_files(input_root: Path) -> List[Path]:
    """Return a list of .mat files to process under a directory path."""
    return [p for p in input_root.rglob("*.mat") if p.is_file()]


def compute_output_path(mat_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Compute the destination .npy path for a given .mat file.
    - Mirror directory structure from input_root under output_root.
    """
    base_name = mat_path.with_suffix(".npy").name
    rel_dir = mat_path.parent.relative_to(input_root)
    return output_root.joinpath(rel_dir, base_name)


def convert_one_file(mat_path: Path, npy_path: Path):
    """
    Convert a single .mat file to .npy.
    - Loads .mat, strips metadata keys, selects the first remaining key,
      converts its value to a numpy array, and saves it without keys.
    """

    # Ensure parent directory exists
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    # Load .mat and strip metadata
    mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    clean = convert_mat_to_clean_dict(mat_data)
    if not clean:
        raise ValueError(f"No usable keys found in MAT file: {mat_path}")

    # Select first key deterministically (insertion order preserved in Python 3.7+)
    first_key = next(iter(clean.keys()))
    value = clean[first_key]

    # Convert to ndarray (may be object dtype for complex MATLAB types)
    array_to_save = np.asarray(value)

    # Save plain array so later usage is simply: np.load(path)
    np.save(str(npy_path), array_to_save)



def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively convert .mat files to .npy")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory containing .mat files (directory-only; no single-file).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root directory where .npy files will be written (structure mirrored).")
    args = parser.parse_args()

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)

    output_root.mkdir(parents=True, exist_ok=True)
    mat_files = find_mat_files(input_root)


    errors: List[Tuple[str, str]] = []
    skipped_count = 0
    with tqdm(total=len(mat_files), unit="file", desc="Converting .mat -> .npy") as pbar:
        for mat_path in mat_files:
            npy_path = compute_output_path(mat_path, input_root, output_root)
            convert_one_file(mat_path, npy_path)
            pbar.update(1)

    converted_count = len(mat_files) - skipped_count - len(errors)
    print(f"Converted: {converted_count}, Skipped: {skipped_count}, Errors: {len(errors)}")


if __name__ == "__main__":
    main()

# python convert_mat_to_npy.py --input_path data/HADAR_database --output_path data/hadar_npy