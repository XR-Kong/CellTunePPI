"""Preprocess gene-by-cell count tables and save cell-by-gene AnnData files.

Default usage:
    python preprocess_data.py

Process one file:
    python preprocess_data.py --input data_original/HCT116.count_mtx.tsv.txt

Dependencies: scanpy, anndata, pandas, numpy, scipy, h5py.
The expression matrix uses float32 during preprocessing and in the H5AD file.
The output X matrix contains log1p-normalized expression. QC metrics describe
raw counts before gene filtering. Gene and cell identifiers are preserved.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = (".tsv.txt", ".tsv", ".txt")


def find_input_files(input_path):
    """Find supported tab-separated tables in a file or directory."""
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(path for path in input_path.iterdir() if path.is_file())
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [
        path for path in candidates
        if path.name.lower().endswith(SUPPORTED_SUFFIXES)
    ]
    if not files:
        raise ValueError(f"No supported TSV or TXT files found: {input_path}")
    return files


def output_filename(input_path):
    """Remove the table extension, including compound .tsv.txt extensions."""
    for suffix in SUPPORTED_SUFFIXES:
        if input_path.name.lower().endswith(suffix):
            return input_path.name[:-len(suffix)] + ".h5ad"
    raise ValueError(f"Unsupported input file: {input_path}")


def preprocess_file(input_path, output_path, min_counts=1000, min_genes=200,
                    min_cells=3, target_sum=10000):
    """Apply count QC, cell/gene filtering, normalization, and log1p."""
    print(f"\nProcessing: {input_path.name}")
    raw_data = pd.read_csv(
        input_path, sep="\t", index_col=0, keep_default_na=False
    )
    if raw_data.empty:
        raise ValueError(f"The input matrix is empty: {input_path}")

    raw_data.index = raw_data.index.astype(str)
    raw_data.columns = raw_data.columns.astype(str)
    for label, names in (("Gene", raw_data.index), ("Cell", raw_data.columns)):
        if not names.is_unique:
            raise ValueError(f"{label} identifiers must be unique: {input_path}")
        if any(not name.strip() for name in names):
            raise ValueError(f"{label} identifiers must not be empty: {input_path}")

    try:
        counts = raw_data.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expression values must be numeric: {input_path}") from exc
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError(
            f"Expression values must be finite and nonnegative: {input_path}"
        )

    original_genes, original_cells = raw_data.shape
    print(f"Original shape (genes, cells): {raw_data.shape}")
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.T),
        obs=pd.DataFrame(index=raw_data.columns.copy()),
        var=pd.DataFrame(index=raw_data.index.copy()),
    )
    # Axis names are fixed so source header labels cannot conflict with QC fields.
    adata.obs.index.name = "cell_id"
    adata.var.index.name = "gene_id"
    del raw_data, counts

    # Disable top-gene percentages so QC also supports small gene panels.
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    adata = adata[adata.obs["total_counts"] >= min_counts, :].copy()
    print(f"Cells after total-count filtering: {adata.n_obs}")
    if adata.n_obs == 0:
        raise ValueError(f"No cells remain after total-count filtering: {input_path}")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"Cells after detected-gene filtering: {adata.n_obs}")
    if adata.n_obs == 0:
        raise ValueError(f"No cells remain after detected-gene filtering: {input_path}")

    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Genes after expressing-cell filtering: {adata.n_vars}")
    if adata.n_vars == 0:
        raise ValueError(f"No genes remain after gene filtering: {input_path}")
    if np.any(np.asarray(adata.X.sum(axis=1)).ravel() == 0):
        raise ValueError(
            f"Some cells have zero counts after gene filtering: {input_path}. "
            "Use a lower --min-cells threshold."
        )

    sc.pp.normalize_total(
        adata, target_sum=target_sum, exclude_highly_expressed=False
    )
    sc.pp.log1p(adata)

    adata.uns["preprocessing"] = {
        "source_file": input_path.name,
        "original_n_cells": original_cells,
        "original_n_genes": original_genes,
        "min_counts": min_counts,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "target_sum": target_sum,
        "qc_metrics_stage": "raw_counts_before_gene_filtering",
        "expression_transform": "normalize_total_then_log1p",
        "expression_dtype": str(adata.X.dtype),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Final shape (cells, genes): {adata.shape}")
    print(f"Expression dtype: {adata.X.dtype}")
    print(f"Saved H5AD: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=SCRIPT_DIR / "data_original",
        help="Input TSV/TXT file or directory (default: ./data_original).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "data_processed",
        help="Output directory (default: ./data_processed).",
    )
    parser.add_argument("--min-counts", type=int, default=1000)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=10000)
    args = parser.parse_args()
    if min(args.min_counts, args.min_genes, args.min_cells) < 1:
        parser.error("Filtering thresholds must be positive integers.")
    if not np.isfinite(args.target_sum) or args.target_sum <= 0:
        parser.error("--target-sum must be a positive finite number.")

    try:
        input_files = find_input_files(args.input.resolve())
        output_dir = args.output_dir.resolve()
        output_names = [output_filename(path) for path in input_files]
        if len(set(name.lower() for name in output_names)) != len(output_names):
            raise ValueError("Input filenames would produce duplicate output paths.")
        for input_path, output_name in zip(input_files, output_names):
            preprocess_file(
                input_path, output_dir / output_name,
                min_counts=args.min_counts,
                min_genes=args.min_genes,
                min_cells=args.min_cells,
                target_sum=args.target_sum,
            )
    except (OSError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    print(f"\nCompleted: {len(input_files)} file(s). Output directory: {output_dir}")


if __name__ == "__main__":
    main()
