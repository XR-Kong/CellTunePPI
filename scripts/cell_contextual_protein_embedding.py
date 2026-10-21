"""Extract expression-aware protein embeddings and their Euclidean distances.

The adjusted representation z is the trained encoder output after cross-attention
and final layer normalization. Distances are ||z_i - z_j||_2, not PPI scores.
Model and data helpers are shared through CellTunePPI_model and CellTunePPI_utils.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from CellTunePPI_model import DEFAULT_CHECKPOINT, load_model_from_ckpt
from CellTunePPI_utils import (
    _pad_1d_sequences,
    filter_valid_pairs,
    load_esm2_data,
    load_expression_data,
    load_pair_table,
)


@torch.no_grad()
def extract_z_for_protein(encoder, expr_vec, esm_vec, device: str) -> np.ndarray:
    expr = torch.from_numpy(np.asarray(expr_vec, dtype=np.float32).reshape(-1)).unsqueeze(0)
    lengths = torch.tensor([expr.shape[1]], dtype=torch.long)
    esm = torch.from_numpy(np.asarray(esm_vec, dtype=np.float32).reshape(-1)).unsqueeze(0)
    encoder.eval()
    z = encoder(expr.to(device), lengths.to(device), esm.to(device))
    if not torch.isfinite(z).all():
        raise ValueError("Encoder produced NaN/Inf embeddings.")
    return z.squeeze(0).cpu().numpy().copy()


@torch.no_grad()
def build_fusion_embeddings(
    encoder,
    expr_dict: Dict[str, np.ndarray],
    esm2_dict: Dict[str, np.ndarray],
    device: str,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    valid_proteins = sorted(set(expr_dict) & set(esm2_dict))
    if not valid_proteins:
        raise ValueError("No proteins occur in both expression and ESM2 data.")
    if batch_size < 1:
        raise ValueError("batch-size must be positive.")
    encoder.eval()
    embeddings = {}
    for start in range(0, len(valid_proteins), batch_size):
        proteins = valid_proteins[start:start + batch_size]
        expr, lengths = _pad_1d_sequences([torch.from_numpy(expr_dict[gene]) for gene in proteins])
        esm = torch.from_numpy(np.stack([esm2_dict[gene] for gene in proteins]))
        z = encoder(expr.to(device), lengths.to(device), esm.to(device))
        if not torch.isfinite(z).all():
            raise ValueError("Encoder produced NaN/Inf embeddings.")
        for gene, vector in zip(proteins, z.cpu().numpy()):
            embeddings[gene] = vector.copy()
        end = start + len(proteins)
        if start == 0 or end == len(valid_proteins) or start // 2048 != end // 2048:
            print(f"Extracted expression-aware embeddings: {end}/{len(valid_proteins)}", flush=True)
    return embeddings


def load_pair_file(path: str) -> pd.DataFrame:
    pairs = load_pair_table(path)
    # Keep the additional two-column headers accepted by the original tool.
    header = tuple(pairs.iloc[0][["gene_i", "gene_j"]].str.lower())
    if header in {("protein_i", "protein_j"), ("source", "target"), ("a", "b")}:
        pairs = pairs.iloc[1:].reset_index(drop=True)
    if pairs.empty:
        raise ValueError("Pair table has no data rows.")
    return pairs


def save_embeddings_tsv(embeddings: Dict[str, np.ndarray], out_path: Path) -> None:
    proteins = list(embeddings)
    matrix = np.stack(list(embeddings.values()))
    table = pd.DataFrame(matrix, columns=[f"z_{index}" for index in range(matrix.shape[1])])
    table.insert(0, "protein_id", proteins)
    table.to_csv(out_path, sep="\t", index=False)


def save_embeddings_pkl(embeddings: Dict[str, np.ndarray], out_path: Path) -> None:
    with out_path.open("wb") as handle:
        pickle.dump(embeddings, handle)


def compute_pair_distances(
    pairs: pd.DataFrame, embeddings: Dict[str, np.ndarray], strict_pairs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pairs.rename(columns={"gene_i": "protein_i", "gene_j": "protein_j"}).copy()
    has_i = pairs["protein_i"].isin(embeddings)
    has_j = pairs["protein_j"].isin(embeddings)
    missing = pairs.loc[~(has_i & has_j)].copy()
    missing["missing_protein_i"] = ~has_i.loc[missing.index]
    missing["missing_protein_j"] = ~has_j.loc[missing.index]
    if strict_pairs and not missing.empty:
        raise ValueError(f"{len(missing)} pairs contain proteins missing from embeddings.")
    result = pairs.loc[has_i & has_j].copy()
    result["d_z"] = [
        float(np.linalg.norm(embeddings[a] - embeddings[b], ord=2))
        for a, b in zip(result["protein_i"], result["protein_j"])
    ]
    return result.reset_index(drop=True), missing.reset_index(drop=True)


def run(args) -> None:
    outdir = Path(args.curve_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    expr_dict = load_expression_data(args.expr_file, layer=args.expr_layer)
    esm2_dict = load_esm2_data(args.esm2_pkl)
    model, ckpt = load_model_from_ckpt(args.ckpt_path, args.device)
    model.eval()
    esm_dim = int(next(iter(esm2_dict.values())).size)
    if esm_dim != int(ckpt["esm_dim"]):
        raise ValueError(f"ESM2 dim mismatch: ckpt expects {ckpt['esm_dim']}, got {esm_dim}")

    labels_path = args.test_labels_tsv or args.labels_tsv
    pairs_path = labels_path or args.pairs_tsv
    matched: Optional[pd.DataFrame] = None
    input_pair_count = None
    if pairs_path:
        pairs = load_pair_table(pairs_path, labelled=True) if labels_path else load_pair_file(pairs_path)
        input_pair_count = len(pairs)
        report_path = args.missing_report or str(outdir / "distance_dropped_pairs.tsv")
        matched = filter_valid_pairs(pairs, expr_dict, esm2_dict, report_path=report_path)
        if args.strict_pairs and len(matched) != len(pairs):
            raise ValueError(f"{len(pairs) - len(matched)} pairs lack features; see {report_path}.")

    embeddings = build_fusion_embeddings(
        model.encoder, expr_dict, esm2_dict, args.device, batch_size=args.batch_size,
    )
    embedding_tsv = outdir / "protein_fusion_embeddings.tsv"
    embedding_pkl = outdir / "protein_fusion_embeddings.pkl"
    save_embeddings_tsv(embeddings, embedding_tsv)
    save_embeddings_pkl(embeddings, embedding_pkl)
    print(f"Saved embeddings: {embedding_tsv} and {embedding_pkl}")
    if matched is not None:
        distances, _ = compute_pair_distances(matched, embeddings, strict_pairs=True)
        pair_path = outdir / "protein_pair_fusion_distances.tsv"
        distances.to_csv(pair_path, sep="\t", index=False)
        print(f"Saved {len(distances)} adjusted pair distances to: {pair_path}")
    metadata = {
        "expression_file": args.expr_file,
        "expression_layer": args.expr_layer,
        "esm2_file": args.esm2_pkl,
        "checkpoint": args.ckpt_path,
        "pair_file": pairs_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "fusion": "rich",
        "expression_gene_count": len(expr_dict),
        "esm2_gene_count": len(esm2_dict),
        "embedding_count": len(embeddings),
        "embedding_dimension": int(ckpt["d_model"]),
        "embedding_dtype": str(next(iter(embeddings.values())).dtype),
        "input_pair_count": input_pair_count,
        "retained_pair_count": len(matched) if matched is not None else None,
        "distance_definition": "Euclidean norm of z_i - z_j after cross-attention and final layer normalization",
    }
    (outdir / "distance_run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expr-file", required=True, help="Expression .h5ad (cells x genes), .pkl/.npz or headered table")
    parser.add_argument("--expr-layer", help="H5AD layer with model-ready expression; defaults to X; no normalization is applied")
    parser.add_argument("--esm2-file", "--esm2-pkl", dest="esm2_pkl", required=True, help="ESM2 gene-to-vector .pt, .pkl or .npz")
    parser.add_argument("--labels-tsv", help="Optional labelled TSV/CSV; optional header: gene_i gene_j label")
    parser.add_argument("--test-labels-tsv", default=None, help="Explicit test-label file; takes precedence over --labels-tsv")
    parser.add_argument("--pairs-tsv", help="Optional unlabelled pair TSV/CSV; its first two columns contain gene identifiers")
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CHECKPOINT), help="Rich-fusion checkpoint; defaults to the packaged HEK model in ../model")
    parser.add_argument("--curve-output-dir", "--out-dir", dest="curve_output_dir", default=".")
    parser.add_argument("--missing-report", help="Dropped-pair TSV; defaults to <curve-output-dir>/distance_dropped_pairs.tsv")
    parser.add_argument("--strict-pairs", action="store_true", help="Fail if any requested pair lacks expression or ESM2 features")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    if args.pairs_tsv and (args.labels_tsv or args.test_labels_tsv):
        parser.error("Use --pairs-tsv or labelled inputs, not both.")
    return args


if __name__ == "__main__":
    run(parse_args())
