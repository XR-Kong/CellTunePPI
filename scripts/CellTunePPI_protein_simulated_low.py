"""Measure PPI sensitivity to simulated-low expression using the shared test model.

The expression profile with the lowest mean across all input genes replaces
protein A, B, or both. The proteins' ESM2 embeddings remain unchanged.
The default checkpoint is the historical HEK rich-fusion model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from CellTunePPI_model import DEFAULT_CHECKPOINT, load_model_from_ckpt
from CellTunePPI_utils import (
    _pad_1d_sequences,
    binary_pr_auc,
    binary_roc_auc,
    f1_at_threshold,
    filter_valid_pairs,
    load_esm2_data,
    load_expression_data,
    load_pair_table,
    precision_at_k,
    recall_at_k,
)


def predict_scores_for_pairs(
    model: nn.Module,
    esm2_dict: Dict[str, np.ndarray],
    expr_source_a: Dict[str, np.ndarray],
    expr_source_b: Dict[str, np.ndarray],
    expr_keys_a: Sequence[str],
    expr_keys_b: Sequence[str],
    esm_keys_a: Sequence[str],
    esm_keys_b: Sequence[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    n = len(expr_keys_a)
    out = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            ea = [torch.from_numpy(np.asarray(expr_source_a[k], dtype=np.float32).reshape(-1)) for k in expr_keys_a[st:ed]]
            eb = [torch.from_numpy(np.asarray(expr_source_b[k], dtype=np.float32).reshape(-1)) for k in expr_keys_b[st:ed]]
            expr_a_pad, len_a = _pad_1d_sequences(ea)
            expr_b_pad, len_b = _pad_1d_sequences(eb)

            esm_a_np = [np.asarray(esm2_dict[k], dtype=np.float32).reshape(-1) for k in esm_keys_a[st:ed]]
            esm_b_np = [np.asarray(esm2_dict[k], dtype=np.float32).reshape(-1) for k in esm_keys_b[st:ed]]
            esm_a = torch.from_numpy(np.stack(esm_a_np, axis=0))
            esm_b = torch.from_numpy(np.stack(esm_b_np, axis=0))

            expr_a_pad = expr_a_pad.to(device)
            len_a = len_a.to(device)
            expr_b_pad = expr_b_pad.to(device)
            len_b = len_b.to(device)
            esm_a = esm_a.to(device)
            esm_b = esm_b.to(device)

            logits = model(expr_a_pad, len_a, expr_b_pad, len_b, esm_a, esm_b)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced NaN/Inf logits during simulated-low analysis.")
            prob_pos = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)
            out[st:ed] = prob_pos
            if st == 0 or ed == n or st // 8192 != ed // 8192:
                print(f"Scored pairs: {ed}/{n}", flush=True)
    return out


def compute_metrics_row(cell_line: str, condition: str, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    return {
        "cell_line": cell_line,
        "condition": condition,
        "AUROC": binary_roc_auc(y_true, y_score),
        "AUPRC": binary_pr_auc(y_true, y_score),
        "Precision_at_100": precision_at_k(y_true, y_score, 100),
        "Precision_at_500": precision_at_k(y_true, y_score, 500),
        "Precision_at_1000": precision_at_k(y_true, y_score, 1000),
        "Recall_at_500": recall_at_k(y_true, y_score, 500),
        "F1_threshold_0.5": f1_at_threshold(y_true, y_score, 0.5),
    }


def build_low_expression_vector_dict(expr_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], str, float]:
    """Map every gene to the expression profile of the lowest-mean-expression gene."""
    lowest_gene = ""
    lowest_vec: Optional[np.ndarray] = None
    lowest_mean = float("inf")

    for gene in sorted(expr_dict.keys()):
        arr = np.asarray(expr_dict[gene], dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        mean_val = float(np.nanmean(arr))
        if not np.isfinite(mean_val):
            continue
        if mean_val < lowest_mean:
            lowest_gene = str(gene)
            lowest_vec = arr.copy()
            lowest_mean = mean_val

    if lowest_vec is None:
        raise ValueError("No valid expression vector is available for simulated-low replacement.")

    out: Dict[str, np.ndarray] = {}
    for gene in expr_dict.keys():
        out[str(gene)] = lowest_vec.copy()
    return out, lowest_gene, lowest_mean


def save_dual(fig: plt.Figure, out_pdf: Path, out_png: Path, dpi: int = 300) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_workflow_figure_2(outdir: Path, cell_line: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.axis("off")
    steps = [
        "pair-specific\nexpression memory",
        "replace A/B/A+B with the\nlowest-expression gene memory",
        "re-predict PPI score",
        "compare score changes",
        "aggregate Delta AUROC\nand Delta AUPRC",
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (x, s) in enumerate(zip(xs, steps)):
        ax.text(x, 0.5, s, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="#f6f2fc"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 0.5), xytext=(x + 0.08, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title(f"Fig2A Protein-specific Simulated-low Workflow ({cell_line})", fontsize=12)
    save_dual(
        fig,
        outdir / "Fig2A_protein_specific_occlusion_workflow.pdf",
        outdir / "Fig2A_protein_specific_occlusion_workflow.png",
    )


def plot_metric_bar(
    metrics_df: pd.DataFrame,
    condition_order: List[str],
    metric_col: str,
    ylabel: str,
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    m = metrics_df.set_index("condition")
    y = [float(m.loc[c, metric_col]) for c in condition_order]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(np.arange(len(condition_order)), y, color="#4c78a8")
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(y):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_dual(fig, out_pdf, out_png)


def plot_delta_distribution(
    edge_df: pd.DataFrame,
    condition_to_delta_col: List[Tuple[str, str]],
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    base = np.arange(len(condition_to_delta_col)) * 3.0
    pos_color = "#2ca02c"
    neg_color = "#d62728"
    for i, (cond_name, delta_col) in enumerate(condition_to_delta_col):
        pos_vals = edge_df.loc[edge_df["label"] == 1, delta_col].to_numpy(dtype=np.float64)
        neg_vals = edge_df.loc[edge_df["label"] == 0, delta_col].to_numpy(dtype=np.float64)

        bp = ax.boxplot(
            [pos_vals, neg_vals],
            positions=[base[i] + 0.85, base[i] + 1.75],
            widths=0.7,
            patch_artist=True,
            showfliers=False,
        )
        bp["boxes"][0].set(facecolor=pos_color, alpha=0.45)
        bp["boxes"][1].set(facecolor=neg_color, alpha=0.45)

    ax.set_xticks(base + 1.3)
    ax.set_xticklabels([x[0] for x in condition_to_delta_col], rotation=18, ha="right")
    ax.set_ylabel("Delta score")
    ax.set_title(title)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=pos_color, alpha=0.45, label="positive pairs"),
            Patch(facecolor=neg_color, alpha=0.45, label="negative pairs"),
        ],
        loc="best",
    )
    fig.tight_layout()
    save_dual(fig, out_pdf, out_png)


def plot_single_row_heatmap(
    values: List[float],
    col_labels: List[str],
    value_label: str,
    title: str,
    out_pdf: Path,
    out_png: Path,
    cell_line: str = "sample",
) -> None:
    arr = np.array(values, dtype=np.float64).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlBu_r")
    ax.set_yticks([0])
    ax.set_yticklabels([cell_line])
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(value_label)
    for j, v in enumerate(values):
        ax.text(j, 0, f"{v:.3f}", ha="center", va="center", fontsize=9, color="black")
    fig.tight_layout()
    save_dual(fig, out_pdf, out_png)


def plot_edge_examples(edge_df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    pos_df = edge_df.loc[edge_df["label"] == 1].copy()
    if pos_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No positive pairs available for edge-level examples.", ha="center", va="center")
        save_dual(fig, out_pdf, out_png)
        return

    if len(pos_df) >= 5:
        selected = pos_df.sort_values(["delta_mask_A_plus_B", "score_original"], ascending=[False, False]).head(5)
    else:
        selected = pos_df.sort_values(["delta_mask_A_plus_B", "score_original"], ascending=[False, False])

    x_labels = ["original", "mask_A", "mask_B", "mask_A_plus_B"]
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for _, row in selected.iterrows():
        y = [
            float(row["score_original"]),
            float(row["score_mask_A"]),
            float(row["score_mask_B"]),
            float(row["score_mask_A_plus_B"]),
        ]
        tag = f"{row['protein_a']}-{row['protein_b']}"
        ax.plot(x, y, marker="o", linewidth=1.4, alpha=0.85, label=tag)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=18, ha="right")
    ax.set_ylabel("Predicted PPI score")
    ax.set_title("Fig2E Representative Positive Edges")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_dual(fig, out_pdf, out_png)


def run_protein_specific_experiment(
    model: nn.Module,
    filtered_df: pd.DataFrame,
    expr_dict: Dict[str, np.ndarray],
    esm2_dict: Dict[str, np.ndarray],
    args,
    outdir: Path,
    original_scores: np.ndarray,
) -> pd.DataFrame:
    genes_a = filtered_df["gene_a"].astype(str).tolist()
    genes_b = filtered_df["gene_b"].astype(str).tolist()
    esm_a = filtered_df["esm_key_a"].astype(str).tolist()
    esm_b = filtered_df["esm_key_b"].astype(str).tolist()
    y_true = filtered_df["label"].to_numpy(dtype=np.int64)

    low_expr_vec_dict, low_expr_gene, low_expr_mean = build_low_expression_vector_dict(expr_dict)
    print(
        "Using simulated-low replacement expression memory: "
        f"{low_expr_gene} (mean expression = {low_expr_mean:.6g})"
    )

    np.save(outdir / "01_simulated_low_reference.npy", low_expr_vec_dict[low_expr_gene])
    run_info = {
        "expression_file": args.expr_file,
        "expression_layer": args.expr_layer,
        "esm2_file": args.esm2_pkl,
        "labels_file": args.test_labels_tsv or args.labels_tsv,
        "checkpoint": args.ckpt_path,
        "cell_line": args.cell_line,
        "device": args.device,
        "batch_size": args.batch_size,
        "fusion": "rich",
        "reference_gene": low_expr_gene,
        "reference_mean_expression": low_expr_mean,
        "reference_vector_length": int(low_expr_vec_dict[low_expr_gene].size),
        "reference_selection": "Lowest mean expression across all loaded expression genes; alphabetical tie break",
        "perturbation": "Replace expression of A, B, or both; keep each protein's ESM2 vector",
        "retained_pairs": len(filtered_df),
        "positive_pairs": int((y_true == 1).sum()),
        "negative_pairs": int((y_true == 0).sum()),
        "auprc_implementation": "Shared v2 legacy per-row ranking implementation",
    }
    (outdir / "02_simulated_low_run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8",
    )

    print("Scoring condition: mask_A", flush=True)
    score_mask_a = predict_scores_for_pairs(
        model,
        esm2_dict,
        low_expr_vec_dict,
        expr_dict,
        genes_a,
        genes_b,
        esm_a,
        esm_b,
        args.batch_size,
        args.device,
    )
    print("Scoring condition: mask_B", flush=True)
    score_mask_b = predict_scores_for_pairs(
        model,
        esm2_dict,
        expr_dict,
        low_expr_vec_dict,
        genes_a,
        genes_b,
        esm_a,
        esm_b,
        args.batch_size,
        args.device,
    )
    print("Scoring condition: mask_A_plus_B", flush=True)
    score_mask_ab = predict_scores_for_pairs(
        model,
        esm2_dict,
        low_expr_vec_dict,
        low_expr_vec_dict,
        genes_a,
        genes_b,
        esm_a,
        esm_b,
        args.batch_size,
        args.device,
    )
    edge_df = filtered_df[["cell_line", "protein_a", "protein_b", "gene_a", "gene_b", "label"]].copy()
    edge_df["score_original"] = original_scores
    edge_df["score_mask_A"] = score_mask_a
    edge_df["score_mask_B"] = score_mask_b
    edge_df["score_mask_A_plus_B"] = score_mask_ab
    edge_df["delta_mask_A"] = edge_df["score_original"] - edge_df["score_mask_A"]
    edge_df["delta_mask_B"] = edge_df["score_original"] - edge_df["score_mask_B"]
    edge_df["delta_mask_A_plus_B"] = edge_df["score_original"] - edge_df["score_mask_A_plus_B"]
    edge_df.to_csv(outdir / "03_protein_specific_occlusion_edge_scores.csv", index=False)

    metric_rows = []
    cond_scores = {
        "original": original_scores,
        "mask_A": score_mask_a,
        "mask_B": score_mask_b,
        "mask_A_plus_B": score_mask_ab,
    }
    for cond, s in cond_scores.items():
        metric_rows.append(compute_metrics_row(args.cell_line, cond, y_true, s))
    mdf = pd.DataFrame(metric_rows)
    auroc0 = float(mdf.loc[mdf["condition"] == "original", "AUROC"].iloc[0])
    auprc0 = float(mdf.loc[mdf["condition"] == "original", "AUPRC"].iloc[0])
    mdf["Delta_AUROC"] = [0.0 if c == "original" else auroc0 - float(a) for c, a in zip(mdf["condition"], mdf["AUROC"])]
    mdf["Delta_AUPRC"] = [0.0 if c == "original" else auprc0 - float(a) for c, a in zip(mdf["condition"], mdf["AUPRC"])]
    mdf = mdf[
        [
            "cell_line",
            "condition",
            "AUROC",
            "AUPRC",
            "Delta_AUROC",
            "Delta_AUPRC",
            "Precision_at_100",
            "Precision_at_500",
            "Precision_at_1000",
            "Recall_at_500",
            "F1_threshold_0.5",
        ]
    ]
    mdf.to_csv(outdir / "04_protein_specific_occlusion_metrics.csv", index=False)

    plot_workflow_figure_2(outdir, args.cell_line)
    plot_delta_distribution(
        edge_df,
        [
            ("mask_A", "delta_mask_A"),
            ("mask_B", "delta_mask_B"),
            ("mask_A+B", "delta_mask_A_plus_B"),
        ],
        "Fig2B Delta-score Distribution by Protein-specific Simulated-low Occlusion",
        outdir / "Fig2B_protein_specific_delta_score_distribution.pdf",
        outdir / "Fig2B_protein_specific_delta_score_distribution.png",
    )
    plot_metric_bar(
        mdf,
        ["original", "mask_A", "mask_B", "mask_A_plus_B"],
        "AUPRC",
        "AUPRC",
        f"Fig2C Protein-specific Simulated-low Occlusion AUPRC ({args.cell_line})",
        outdir / "Fig2C_protein_specific_AUPRC.pdf",
        outdir / "Fig2C_protein_specific_AUPRC.png",
    )
    plot_metric_bar(
        mdf,
        ["original", "mask_A", "mask_B", "mask_A_plus_B"],
        "AUROC",
        "AUROC",
        f"Fig2C2 Protein-specific Simulated-low Occlusion AUROC ({args.cell_line})",
        outdir / "Fig2C2_protein_specific_AUROC.pdf",
        outdir / "Fig2C2_protein_specific_AUROC.png",
    )

    mindex = mdf.set_index("condition")
    vals = [
        float(mindex.loc["mask_A", "Delta_AUPRC"]),
        float(mindex.loc["mask_B", "Delta_AUPRC"]),
        float(mindex.loc["mask_A_plus_B", "Delta_AUPRC"]),
    ]
    plot_single_row_heatmap(
        vals,
        ["mask_A", "mask_B", "mask_A+B"],
        "Delta AUPRC",
        f"Fig2D Delta AUPRC Heatmap ({args.cell_line})",
        outdir / "Fig2D_protein_specific_delta_AUPRC_heatmap.pdf",
        outdir / "Fig2D_protein_specific_delta_AUPRC_heatmap.png",
        cell_line=args.cell_line,
    )
    vals_auroc = [
        float(mindex.loc["mask_A", "Delta_AUROC"]),
        float(mindex.loc["mask_B", "Delta_AUROC"]),
        float(mindex.loc["mask_A_plus_B", "Delta_AUROC"]),
    ]
    plot_single_row_heatmap(
        vals_auroc,
        ["mask_A", "mask_B", "mask_A+B"],
        "Delta AUROC",
        f"Fig2D2 Delta AUROC Heatmap ({args.cell_line})",
        outdir / "Fig2D2_protein_specific_delta_AUROC_heatmap.pdf",
        outdir / "Fig2D2_protein_specific_delta_AUROC_heatmap.png",
        cell_line=args.cell_line,
    )
    plot_edge_examples(
        edge_df,
        outdir / "Fig2E_edge_level_examples.pdf",
        outdir / "Fig2E_edge_level_examples.png",
    )
    return mdf


def print_ckpt_info(ckpt_path: str, device: str) -> None:
    _, ckpt = load_model_from_ckpt(ckpt_path, device)
    print("checkpoint keys:", sorted(ckpt))
    for key in ["esm_dim", "d_model", "n_heads", "n_cross_layers", "expr_token_hidden", "cls_hidden", "dropout", "fusion"]:
        print(f"{key}: {ckpt.get(key)}")
    print("model state_dict key count:", len(ckpt["model"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run_all", "inspect_ckpt"], default="run_all")
    parser.add_argument("--expr-file", "--expr-files", dest="expr_file", help="Expression .h5ad (cells x genes), .pkl/.npz or headered table")
    parser.add_argument("--expr-layer", help="H5AD layer with model-ready expression; defaults to X; no normalization is applied")
    parser.add_argument("--esm2-file", "--esm2-pkl", dest="esm2_pkl", help="ESM2 gene-to-vector .pt, .pkl or .npz")
    parser.add_argument("--labels-tsv", "--test-pairs", dest="labels_tsv", help="Labelled test TSV/CSV; optional header: gene_i gene_j label")
    parser.add_argument("--test-labels-tsv", default=None, help="Explicit test-label file; takes precedence over --labels-tsv")
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CHECKPOINT), help="Rich-fusion checkpoint; defaults to the packaged HEK model in ../model")
    parser.add_argument("--curve-output-dir", "--outdir", "--out-dir", dest="curve_output_dir", default=".")
    parser.add_argument("--missing-report", help="Dropped-pair TSV; defaults to <curve-output-dir>/simulated_low_dropped_pairs.tsv")
    parser.add_argument("--cell-line", help="Display name for tables and figures; defaults to the expression file stem")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    if args.mode == "run_all":
        if not args.expr_file or not args.esm2_pkl or not (args.test_labels_tsv or args.labels_tsv):
            parser.error("--expr-file, --esm2-file and --labels-tsv (or --test-labels-tsv) are required.")
        args.cell_line = args.cell_line or Path(args.expr_file).stem
    return args


def main():
    args = parse_args()
    if args.mode == "inspect_ckpt":
        print_ckpt_info(args.ckpt_path, args.device)
        return
    outdir = Path(args.curve_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    expr_dict = load_expression_data(args.expr_file, layer=args.expr_layer)
    esm2_dict = load_esm2_data(args.esm2_pkl)
    labels_path = args.test_labels_tsv or args.labels_tsv
    raw_pairs = load_pair_table(labels_path, labelled=True)
    report_path = args.missing_report or str(outdir / "simulated_low_dropped_pairs.tsv")
    pairs = filter_valid_pairs(raw_pairs, expr_dict, esm2_dict, report_path=report_path)
    model, ckpt = load_model_from_ckpt(args.ckpt_path, args.device)
    esm_dim = int(next(iter(esm2_dict.values())).size)
    if esm_dim != int(ckpt["esm_dim"]):
        raise ValueError(f"ESM2 dim mismatch: ckpt expects {ckpt['esm_dim']}, got {esm_dim}")
    filtered_df = pd.DataFrame({
        "cell_line": args.cell_line,
        "protein_a": pairs["gene_i"],
        "protein_b": pairs["gene_j"],
        "gene_a": pairs["gene_i"],
        "gene_b": pairs["gene_j"],
        "label": pairs["label"],
        "esm_key_a": pairs["gene_i"],
        "esm_key_b": pairs["gene_j"],
    })
    filtered_df[["cell_line", "protein_a", "protein_b", "gene_a", "gene_b", "label"]].to_csv(
        outdir / "00_filtered_test_pairs_used.csv", index=False,
    )
    print(f"Running original score prediction for {len(filtered_df)} pairs...", flush=True)
    scores = predict_scores_for_pairs(
        model, esm2_dict, expr_dict, expr_dict,
        filtered_df["gene_a"].tolist(), filtered_df["gene_b"].tolist(),
        filtered_df["esm_key_a"].tolist(), filtered_df["esm_key_b"].tolist(),
        args.batch_size, args.device,
    )
    metrics = run_protein_specific_experiment(
        model, filtered_df, expr_dict, esm2_dict, args, outdir, scores,
    )
    print("\nSimulated-low expression sensitivity metrics")
    print(metrics.to_string(index=False))
    print(f"Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
