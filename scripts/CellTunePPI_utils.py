"""Shared feature loading, pair datasets, splitting, metrics, and plotting.

This module has no dependency on the model implementation or command-line
entry points. Model-ready expression and ESM2 values are loaded as float32.
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _gene_name(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return ""
    return (value.decode("utf-8") if isinstance(value, bytes) else str(value)).strip()


def _read_table(path: str, header=None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    sep = "," if suffix == ".csv" else "\t" if suffix == ".tsv" else None
    return pd.read_csv(
        path, sep=sep, engine="python", header=header, dtype=str, keep_default_na=False,
    )


def _h5ad_expression_items(path: str, layer: Optional[str] = None):
    """Read cells x genes; expose each gene column as its expression vector."""
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("H5AD input requires anndata and scipy; install with: pip install anndata") from exc
    adata = ad.read_h5ad(path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{path}: expression AnnData must contain at least one cell and one gene.")
    if not adata.obs_names.is_unique:
        raise ValueError(f"{path}: duplicate cell identifiers in obs_names.")
    if not adata.var_names.is_unique:
        raise ValueError(f"{path}: duplicate gene identifiers in var_names.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{path}: expression layer {layer!r} not found; available layers: {list(adata.layers)}")
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError(f"{path}: selected expression matrix is empty; put model-ready values in X or specify --expr-layer.")
    if sparse.issparse(matrix):
        # Column-oriented slicing avoids densifying the whole sparse matrix.
        matrix = matrix.tocsc(copy=False)
        return ((gene, matrix[:, [index]].toarray().reshape(-1))
                for index, gene in enumerate(adata.var_names))
    return ((gene, matrix[:, index]) for index, gene in enumerate(adata.var_names))


def _load_features(path: str, kind: str, layer: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load gene-indexed float32 vectors without changing their values or order."""
    suffix = Path(path).suffix.lower()
    if layer is not None and not (kind == "expression" and suffix == ".h5ad"):
        raise ValueError("--expr-layer can only be used with an H5AD expression file.")
    if kind == "ESM2" and suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise TypeError(f"{path}: ESM2 PT must contain a gene-to-vector dict saved with torch.save().")
        if "mean_representations" in obj and isinstance(obj["mean_representations"], dict):
            raise ValueError(f"{path}: this is a single-sequence ESM extraction file. First collect the chosen layer's mean vectors into a gene-to-vector dict.")
        items = obj.items()
    elif kind == "expression" and suffix == ".h5ad":
        items = _h5ad_expression_items(path, layer=layer)
    elif suffix in {".pkl", ".pickle"}:
        # Backward compatibility: only load pickle files from trusted sources.
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a gene-to-vector dict in {path}, got {type(obj)}")
        items = obj.items()
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not {"genes", "values"}.issubset(archive.files):
                raise ValueError(f"{path}: NPZ must contain 'genes' and 'values' arrays.")
            genes, values = archive["genes"], archive["values"]
        if genes.ndim != 1 or genes.dtype.kind not in "US":
            raise ValueError(f"{path}: 'genes' must be a one-dimensional string array.")
        if values.ndim != 2 or len(genes) != len(values) or values.dtype.kind not in "biuf":
            raise ValueError(f"{path}: 'values' must be a numeric genes-by-features matrix.")
        items = zip(genes, values)
    elif kind == "expression" and suffix in {".csv", ".tsv", ".txt"}:
        table = _read_table(path, header=0)
        if table.shape[1] < 2:
            raise ValueError("Expression table must contain a gene column followed by values.")
        items = ((row[0], row[1:]) for row in table.itertuples(index=False, name=None))
    else:
        formats = ".h5ad, .pkl/.pickle, .npz or a headered .csv/.tsv/.txt table" if kind == "expression" else ".pt, .pkl/.pickle or .npz"
        raise ValueError(f"Unsupported {kind} format {suffix!r}; use {formats}.")

    out: Dict[str, np.ndarray] = {}
    seen, empty_count = set(), 0
    for key, value in items:
        gene = _gene_name(key)
        if not gene:
            raise ValueError(f"{path}: empty gene identifier in {kind} data.")
        if gene in seen:
            raise ValueError(f"{path}: duplicate {kind} gene identifier after trimming: {gene!r}")
        seen.add(gene)
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: non-numeric {kind} vector for gene {gene!r}") from exc
        if sum(size > 1 for size in arr.shape) > 1:
            raise ValueError(f"{path}: {kind} entry for {gene!r} must be a vector, got {arr.shape}.")
        arr = arr.reshape(-1)
        if arr.size == 0:
            empty_count += 1
            continue
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: NaN/Inf in {kind} vector for gene {gene!r}.")
        out[gene] = np.ascontiguousarray(arr)
    if empty_count:
        warnings.warn(f"{path}: ignored {empty_count} empty {kind} vectors; these genes count as missing.")
    if not out:
        raise ValueError(f"No valid {kind} entries loaded from {path}")
    if kind == "ESM2":
        dims = {v.size for v in out.values()}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent ESM2 dims detected: {sorted(dims)}")
    return out


def load_expression_data(expr_path: str, layer: Optional[str] = None) -> Dict[str, np.ndarray]:
    return _load_features(expr_path, "expression", layer=layer)


def load_esm2_data(esm2_pkl: str) -> Dict[str, np.ndarray]:
    return _load_features(esm2_pkl, "ESM2")


def load_pair_table(path: str, labelled: bool = False) -> pd.DataFrame:
    df = _read_table(path)
    expected = 3 if labelled else 2
    if df.shape[1] < expected or (labelled and df.shape[1] != 3):
        raise ValueError(f"{path}: expected {'exactly 3' if labelled else 'at least 2'} columns.")
    header = tuple(str(v).strip().lower() for v in df.iloc[0, :2])
    if header in {("gene_i", "gene_j"), ("gene_a", "gene_b"), ("protein_a", "protein_b")}:
        if labelled and str(df.iloc[0, 2]).strip().lower() != "label":
            raise ValueError(f"{path}: the third header must be 'label'.")
        df = df.iloc[1:].reset_index(drop=True)
    columns = ["gene_i", "gene_j", "label"] if labelled else ["gene_i", "gene_j"] + [f"extra_{i}" for i in range(df.shape[1] - 2)]
    df.columns = columns
    for column in ("gene_i", "gene_j"):
        df[column] = df[column].map(_gene_name)
    if df.empty:
        raise ValueError(f"{path}: pair table has no data rows.")
    if labelled:
        labels = pd.to_numeric(df["label"], errors="coerce")
        invalid = ~labels.isin([0, 1])
        if invalid.any():
            raise ValueError(f"{path}: labels must be 0 or 1; invalid data rows: {(np.flatnonzero(invalid)[:10] + 1).tolist()}")
        df["label"] = labels.astype(np.int64)
    return df


def load_labels(labels_tsv: str) -> pd.DataFrame:
    return load_pair_table(labels_tsv, labelled=True)


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for start, count in zip(first_idx, counts):
        if count > 1:
            end = start + count
            avg_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg_rank

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    # Preserve the legacy per-row ranking, including its tied-score ordering.
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def _curve_counts(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if not np.isfinite(y_score).all():
        raise ValueError("Predicted scores contain NaN/Inf.")
    order = np.argsort(-y_score, kind="stable")
    scores, labels = y_score[order], y_true[order]
    # A tied score defines one threshold; its internal row order is irrelevant.
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels == 1)[ends]
    fp = np.cumsum(labels == 0)[ends]
    return tp, fp


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tp, fp = _curve_counts(y_true, y_score)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return recall, precision


def save_curve_pdf(curves: List[Dict[str, object]], pdf_path: str, title: str):
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for item in curves:
        fpr, tpr = roc_curve_points(item["y_true"], item["y_score"])
        ax.plot(fpr, tpr, label=f"{item['label']} (AUC={item['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUROC")
    ax.legend(fontsize=8)

    ax = axes[1]
    for item in curves:
        recall, precision = pr_curve_points(item["y_true"], item["y_score"])
        ax.plot(recall, precision, label=f"{item['label']} (AUPRC={item['pr_auc']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AUPRC")
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 123):
    y = np.asarray(y).astype(np.int64)
    if n_splits < 2 or set(np.unique(y)) != {0, 1} or np.bincount(y).min() < n_splits:
        raise ValueError("Cross-validation requires both labels and at least n_splits samples per class.")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class_folds = {}
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        per_class_folds[int(cls)] = np.array_split(idx, n_splits)

    folds = []
    all_idx = np.arange(len(y))
    for fold_id in range(n_splits):
        val_parts = [per_class_folds[int(cls)][fold_id] for cls in classes]
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        tr_idx = all_idx[train_mask]
        folds.append((tr_idx, val_idx))
    return folds


def stratified_split_indices(y: np.ndarray, val_ratio: float = 0.1, seed: int = 123):
    y = np.asarray(y).astype(np.int64)
    if not 0 < val_ratio < 1:
        raise ValueError("full_val_ratio must be between 0 and 1; validation cannot reuse training data.")
    if set(np.unique(y)) != {0, 1} or np.bincount(y).min() < 2:
        raise ValueError("Internal validation requires at least two samples per class.")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    val_parts = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = min(len(idx) - 1, max(1, int(round(len(idx) * val_ratio))))
        val_parts.append(idx[:n_val])
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    tr_idx = np.arange(len(y))[mask]
    return tr_idx, val_idx


class MultiModalPairDataset(Dataset):
    """Pairs already checked by filter_valid_pairs; labels are optional at inference."""
    def __init__(self, df: pd.DataFrame, expr_dict: Dict[str, np.ndarray], esm2_dict: Dict[str, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.expr_dict = expr_dict
        self.esm2_dict = esm2_dict
        if len(self.df) == 0:
            raise ValueError("No valid samples left after expression/ESM2 matching.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gi, gj = str(row["gene_i"]), str(row["gene_j"])
        expr_a = torch.from_numpy(self.expr_dict[gi]).float()
        expr_b = torch.from_numpy(self.expr_dict[gj]).float()
        esm_a = torch.from_numpy(self.esm2_dict[gi]).float()
        esm_b = torch.from_numpy(self.esm2_dict[gj]).float()

        features = (expr_a, expr_b, esm_a, esm_b)
        if "label" in self.df.columns:
            return features + (torch.tensor(int(row["label"]), dtype=torch.long),)
        return features


def _pad_1d_sequences(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(s.numel()) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, s in enumerate(seqs):
        length = int(s.numel())
        padded[i, :length] = s
    return padded, lengths


def multimodal_collate_fn(batch):
    expr_a_list = [item[0] for item in batch]
    expr_b_list = [item[1] for item in batch]
    esm_a = torch.stack([item[2] for item in batch], dim=0)
    esm_b = torch.stack([item[3] for item in batch], dim=0)
    expr_a_pad, len_a = _pad_1d_sequences(expr_a_list)
    expr_b_pad, len_b = _pad_1d_sequences(expr_b_list)
    features = (expr_a_pad, len_a, expr_b_pad, len_b, esm_a, esm_b)
    if len(batch[0]) == 5:
        return features + (torch.stack([item[4] for item in batch], dim=0),)
    return features


def evaluate(
    model,
    loader,
    device,
    criterion: Optional[nn.Module] = None,
    return_scores: bool = False,
) -> Dict[str, object]:
    model.eval()
    ce = criterion if criterion is not None else nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    y_true_list, y_score_list = [], []

    with torch.no_grad():
        for expr_a_pad, len_a, expr_b_pad, len_b, esm_a, esm_b, y in loader:
            expr_a_pad = expr_a_pad.to(device)
            len_a = len_a.to(device)
            expr_b_pad = expr_b_pad.to(device)
            len_b = len_b.to(device)
            esm_a = esm_a.to(device)
            esm_b = esm_b.to(device)
            y = y.to(device)

            logits = model(expr_a_pad, len_a, expr_b_pad, len_b, esm_a, esm_b)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced NaN/Inf logits during evaluation.")
            loss = ce(logits, y)

            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += y.size(0)

            prob_pos = torch.softmax(logits, dim=-1)[:, 1]
            y_true_list.append(y.detach().cpu().numpy())
            y_score_list.append(prob_pos.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=np.int64)
    y_score = np.concatenate(y_score_list) if y_score_list else np.array([], dtype=np.float32)

    result = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "roc_auc": binary_roc_auc(y_true, y_score) if len(y_true) else float("nan"),
        "pr_auc": binary_pr_auc(y_true, y_score) if len(y_true) else float("nan"),
    }
    if return_scores:
        result["y_true"] = y_true
        result["y_score"] = y_score
    return result


def make_loader(
    df: pd.DataFrame,
    expr_dict: Dict[str, np.ndarray],
    esm2_dict: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = MultiModalPairDataset(df, expr_dict, esm2_dict)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
    )


def filter_valid_pairs(
    df: pd.DataFrame,
    expr_dict: Dict[str, np.ndarray],
    esm2_dict: Dict[str, np.ndarray],
    report_path: Optional[str] = None,
) -> pd.DataFrame:
    """Drop unscorable pairs, preserving order and recording each missing modality."""
    df = df.copy()
    missing = pd.DataFrame(index=df.index)
    for column in ("gene_i", "gene_j"):
        df[column] = df[column].map(_gene_name)
        missing[f"missing_expr_{column}"] = ~df[column].isin(expr_dict)
        missing[f"missing_esm2_{column}"] = ~df[column].isin(esm2_dict)
    keep = ~missing.any(axis=1)
    dropped = df.loc[~keep].copy()
    dropped.insert(0, "input_row", np.flatnonzero(~keep) + 1)
    for column in missing.columns:
        dropped[column] = missing.loc[~keep, column]
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        dropped.to_csv(report_path, sep="\t", index=False)
    missing_expr = missing.filter(like="missing_expr_").any(axis=1).sum()
    missing_esm = missing.filter(like="missing_esm2_").any(axis=1).sum()
    print(f"Feature matching | input={len(df)} retained={int(keep.sum())} dropped={len(dropped)} "
          f"missing_expression_pairs={missing_expr} missing_ESM2_pairs={missing_esm} (counts may overlap)")
    if report_path is not None:
        print(f"Missing-feature report: {report_path}")
    out = df.loc[keep].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No valid pairs remain: both genes must occur in both expression and ESM2 data. "
                         "Check gene identifiers and the missing-feature report; missing inputs are not zero-filled.")
    return out


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return float("nan")
    topk = min(k, y_true.size)
    order = np.argsort(-y_score)[:topk]
    return float((y_true[order] == 1).sum() / max(topk, 1))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    topk = min(k, y_true.size)
    order = np.argsort(-y_score)[:topk]
    tp = int((y_true[order] == 1).sum())
    return float(tp / n_pos)


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
