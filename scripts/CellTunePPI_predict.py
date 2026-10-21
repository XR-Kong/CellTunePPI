import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from CellTunePPI_model import DEFAULT_CHECKPOINT, load_model_from_ckpt
from CellTunePPI_utils import (
    filter_valid_pairs,
    load_esm2_data,
    load_expression_data,
    load_pair_table,
    make_loader,
)



def load_block_re_module(module_path: Path):
    if not module_path.is_file():
        raise FileNotFoundError(f"Model script not found: {module_path}; set --model-script explicitly.")
    spec = importlib.util.spec_from_file_location("celltuneppi_model_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_scores(model, loader, pairs: pd.DataFrame, device: str) -> pd.DataFrame:
    """Return gene_i, gene_j, and the positive-class softmax score."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            logits = model(*(tensor.to(device) for tensor in batch))
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced NaN/Inf logits; check checkpoint and feature values.")
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    result = pairs[["gene_i", "gene_j"]].reset_index(drop=True).copy()
    result["score"] = np.concatenate(scores)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict PPI scores with a trained CellTunePPI checkpoint.")
    parser.add_argument("--ppi-tsv", required=True, help="TSV/CSV with gene_i and gene_j as its first two columns; optional header")
    parser.add_argument("--expr-file", required=True, help="Expression .h5ad (cells x genes), .pkl/.npz or headered table")
    parser.add_argument("--expr-layer", help="H5AD layer with model-ready expression; defaults to X; no normalization is applied")
    parser.add_argument("--esm2-file", "--esm2-pkl", dest="esm2_pkl", required=True, help="ESM2 gene-to-vector .pt, .pkl or .npz; legacy --esm2-pkl remains supported")
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CHECKPOINT), help="Rich-fusion checkpoint; defaults to the packaged HEK model in ../model")
    parser.add_argument("--output-tsv", default="ppi_scores_block_re.tsv", help="Prediction TSV with gene_i, gene_j, and score columns")
    parser.add_argument("--missing-report", help="Dropped-pair TSV; defaults to <output-stem>.dropped_pairs.tsv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-script", default=None, help="Optional compatible module providing load_model_from_ckpt; data loading always uses CellTunePPI_utils")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


def main():
    args = parse_args()
    model_loader = load_model_from_ckpt
    if args.model_script:
        model_loader = load_block_re_module(Path(args.model_script).resolve()).load_model_from_ckpt
    expr_dict = load_expression_data(args.expr_file, layer=args.expr_layer)
    esm2_dict = load_esm2_data(args.esm2_pkl)
    pairs = load_pair_table(args.ppi_tsv)
    output_path = Path(args.output_tsv)
    report_path = args.missing_report or str(output_path.with_name(output_path.stem + ".dropped_pairs.tsv"))
    if Path(report_path).resolve() == output_path.resolve():
        raise ValueError("Prediction output and missing-feature report must have different paths.")
    matched = filter_valid_pairs(pairs, expr_dict, esm2_dict, report_path=report_path)
    # Reuse the shared feature dataset and padding; no dummy labels.
    loader = make_loader(matched[["gene_i", "gene_j"]], expr_dict, esm2_dict, args.batch_size, shuffle=False)

    model, ckpt = model_loader(args.ckpt_path, args.device)
    esm_dim = int(next(iter(esm2_dict.values())).size)
    if esm_dim != int(ckpt["esm_dim"]):
        raise ValueError(f"ESM2 dim mismatch: ckpt expects {ckpt['esm_dim']}, got {esm_dim}")
    pred_df = predict_scores(model, loader, matched, args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, sep="\t", index=False)
    print(f"Input pairs: {len(pairs)} | scored: {len(pred_df)} | dropped: {len(pairs) - len(pred_df)}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
