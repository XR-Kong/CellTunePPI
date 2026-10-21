"""Train, cross-validate, or evaluate CellTunePPI using shared public modules."""

import argparse
import copy
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Re-export shared names for callers using the previous training-module API.
from CellTunePPI_model import (
    CrossAttnBlock,
    GeneCrossAttentionEncoder,
    CrossAttnPPIPredictor,
    build_model_from_args,
    save_checkpoint,
    load_model_from_ckpt,
)
from CellTunePPI_utils import (
    _gene_name,
    _read_table,
    _h5ad_expression_items,
    _load_features,
    load_expression_data,
    load_esm2_data,
    load_pair_table,
    load_labels,
    binary_roc_auc,
    binary_pr_auc,
    _curve_counts,
    roc_curve_points,
    pr_curve_points,
    save_curve_pdf,
    stratified_kfold_indices,
    stratified_split_indices,
    MultiModalPairDataset,
    _pad_1d_sequences,
    multimodal_collate_fn,
    evaluate,
    make_loader,
    filter_valid_pairs,
)


def build_criterion(
    train_labels: np.ndarray,
    class_weight_mode: str,
    device: str,
) -> nn.Module:
    if class_weight_mode == "none":
        return nn.CrossEntropyLoss()

    y = np.asarray(train_labels).astype(np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


def run_one_epoch(model, loader, optimizer, criterion, device, args) -> Dict[str, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for expr_a_pad, len_a, expr_b_pad, len_b, esm_a, esm_b, y in loader:
        expr_a_pad = expr_a_pad.to(device)
        len_a = len_a.to(device)
        expr_b_pad = expr_b_pad.to(device)
        len_b = len_b.to(device)
        esm_a = esm_a.to(device)
        esm_b = esm_b.to(device)
        y = y.to(device)

        esm_a_in, esm_b_in = esm_a, esm_b
        if args.esm_drop_prob > 0:
            drop_mask = (torch.rand(esm_a.size(0), 1, device=device) < args.esm_drop_prob).float()
            esm_a_in = esm_a * (1.0 - drop_mask)
            esm_b_in = esm_b * (1.0 - drop_mask)

        logits = model(expr_a_pad, len_a, expr_b_pad, len_b, esm_a_in, esm_b_in)
        cls_loss = criterion(logits, y)
        loss = cls_loss

        if args.cf_weight > 0:
            perm = torch.randperm(expr_a_pad.size(0), device=device)
            logits_cf = model(
                expr_a_pad[perm],
                len_a[perm],
                expr_b_pad[perm],
                len_b[perm],
                esm_a_in,
                esm_b_in,
            )
            # Keep each loss branch's original graph for training reproducibility.
            prob_real = torch.softmax(logits, dim=-1)
            prob_cf = torch.softmax(logits_cf, dim=-1)
            idx = torch.arange(y.size(0), device=device)
            real_true_prob = prob_real[idx, y]
            cf_true_prob = prob_cf[idx, y]
            cf_margin_loss = torch.relu(args.cf_margin - (real_true_prob - cf_true_prob)).mean()
            loss = loss + args.cf_weight * cf_margin_loss

        if args.sim_weight > 0:
            # Independently shuffle complete expression vectors for the two proteins.
            perm_a = torch.randperm(expr_a_pad.size(0), device=device)
            perm_b = torch.randperm(expr_b_pad.size(0), device=device)
            expr_a_sim = expr_a_pad[perm_a]
            expr_b_sim = expr_b_pad[perm_b]
            len_a_sim = len_a[perm_a]
            len_b_sim = len_b[perm_b]

            logits_sim = model(expr_a_sim, len_a_sim, expr_b_sim, len_b_sim, esm_a_in, esm_b_in)
            prob_sim = torch.softmax(logits_sim, dim=-1)
            prob_uniform = torch.full_like(prob_sim, 0.5)
            sim_kl_to_uniform = torch.sum(
                prob_sim * (torch.log(prob_sim + 1e-8) - torch.log(prob_uniform + 1e-8)),
                dim=-1,
            ).mean()

            idx = torch.arange(y.size(0), device=device)
            prob_real = torch.softmax(logits, dim=-1)
            real_true_prob = prob_real[idx, y]
            sim_true_prob = prob_sim[idx, y]
            sim_margin_loss = torch.relu(args.sim_margin - (real_true_prob - sim_true_prob)).mean()
            loss = loss + args.sim_weight * (sim_kl_to_uniform + sim_margin_loss)

        if not torch.isfinite(loss):
            raise ValueError("Training loss became NaN/Inf; check inputs and optimization settings.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def train_with_early_stopping(model, train_loader, val_loader, train_labels, args) -> Dict[str, float]:
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = build_criterion(train_labels, args.class_weight_mode, args.device)

    best_score = -math.inf
    best_metrics = {}
    best_state = None
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        tr = run_one_epoch(model, train_loader, optimizer, criterion, args.device, args)
        scheduler.step()
        va = evaluate(model, val_loader, args.device, criterion=criterion)

        print(
            f"Epoch {ep:03d} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} "
            f"roc_auc {va['roc_auc']:.4f} pr_auc {va['pr_auc']:.4f}"
        )

        score = va["roc_auc"]
        if np.isnan(score):
            score = -math.inf

        if score > (best_score + args.early_stop_min_delta):
            best_score = score
            best_metrics = dict(va)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {ep:03d}: "
                    f"val_roc_auc not improved for {args.early_stop_patience} epochs."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


def load_matched_inputs(args, labels_path: str):
    expr_dict = load_expression_data(args.expr_file, layer=args.expr_layer)
    esm2_dict = load_esm2_data(args.esm2_pkl)
    report = args.missing_report or str(Path(args.curve_output_dir) / f"{args.mode}_dropped_pairs.tsv")
    df = filter_valid_pairs(load_labels(labels_path), expr_dict, esm2_dict, report_path=report)
    return expr_dict, esm2_dict, df


def run_cv(args):
    expr_dict, esm2_dict, df = load_matched_inputs(args, args.labels_tsv)

    y = df["label"].to_numpy().astype(np.int64)
    folds = stratified_kfold_indices(y=y, n_splits=args.n_splits, seed=args.seed)
    esm_dim = int(next(iter(esm2_dict.values())).shape[-1])

    fold_results: List[Dict[str, float]] = []
    curve_items: List[Dict[str, object]] = []
    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()

        tr_loader = make_loader(df_tr, expr_dict, esm2_dict, args.batch_size, shuffle=True)
        va_loader = make_loader(df_va, expr_dict, esm2_dict, args.batch_size, shuffle=False)

        print(f"\n===== Fold {fold_id}/{args.n_splits} | train={len(df_tr)} val={len(df_va)} =====")
        model = build_model_from_args(args, esm_dim).to(args.device)
        best = train_with_early_stopping(model, tr_loader, va_loader, df_tr["label"].to_numpy(), args)
        best_eval = evaluate(model, va_loader, args.device, return_scores=True)
        fold_results.append(best)
        curve_items.append(
            {
                "label": f"Fold {fold_id}",
                "roc_auc": float(best_eval["roc_auc"]),
                "pr_auc": float(best_eval["pr_auc"]),
                "y_true": best_eval["y_true"],
                "y_score": best_eval["y_score"],
            }
        )
        print(
            f"[Fold {fold_id}] best | acc {best.get('acc', float('nan')):.4f} | "
            f"roc_auc {best.get('roc_auc', float('nan')):.4f} | "
            f"pr_auc {best.get('pr_auc', float('nan')):.4f}"
        )

    acc_values = np.array([m.get("acc", float("nan")) for m in fold_results], dtype=float)
    roc_values = np.array([m.get("roc_auc", float("nan")) for m in fold_results], dtype=float)
    pr_values = np.array([m.get("pr_auc", float("nan")) for m in fold_results], dtype=float)

    print(f"\n===== Cross-Attn Block {args.n_splits}-Fold Summary =====")
    for i, m in enumerate(fold_results, start=1):
        print(
            f"Fold {i}: acc {m.get('acc', float('nan')):.4f}, "
            f"roc_auc {m.get('roc_auc', float('nan')):.4f}, "
            f"pr_auc {m.get('pr_auc', float('nan')):.4f}"
        )
    print(
        f"Mean+/-Std | acc {np.nanmean(acc_values):.4f}+/-{np.nanstd(acc_values):.4f} | "
        f"roc_auc {np.nanmean(roc_values):.4f}+/-{np.nanstd(roc_values):.4f} | "
        f"pr_auc {np.nanmean(pr_values):.4f}+/-{np.nanstd(pr_values):.4f}"
    )
    cv_pdf_path = str(Path(args.curve_output_dir) / args.cv_curve_pdf)
    save_curve_pdf(curve_items, cv_pdf_path, title=f"{args.n_splits}-Fold CV AUROC/AUPRC")
    print(f"Saved CV curves to: {cv_pdf_path}")


def run_train_full(args):
    expr_dict, esm2_dict, df = load_matched_inputs(args, args.labels_tsv)

    y = df["label"].to_numpy().astype(np.int64)
    tr_idx, va_idx = stratified_split_indices(y=y, val_ratio=args.full_val_ratio, seed=args.seed)
    df_tr = df.iloc[tr_idx].copy()
    df_va = df.iloc[va_idx].copy()

    tr_loader = make_loader(df_tr, expr_dict, esm2_dict, args.batch_size, shuffle=True)
    va_loader = make_loader(df_va, expr_dict, esm2_dict, args.batch_size, shuffle=False)

    esm_dim = int(next(iter(esm2_dict.values())).shape[-1])
    model = build_model_from_args(args, esm_dim).to(args.device)

    print(
        f"Train mode with internal early-stop split | train={len(df_tr)} val={len(df_va)} "
        f"(full_val_ratio={args.full_val_ratio})"
    )
    best = train_with_early_stopping(model, tr_loader, va_loader, df_tr["label"].to_numpy(), args)
    print(
        f"Best val metrics | acc {best.get('acc', float('nan')):.4f} | "
        f"roc_auc {best.get('roc_auc', float('nan')):.4f} | "
        f"pr_auc {best.get('pr_auc', float('nan')):.4f}"
    )

    save_checkpoint(args.ckpt_path, model, args, esm_dim)
    print(f"Saved checkpoint to: {args.ckpt_path}")


def run_test(args):
    test_labels = args.test_labels_tsv if args.test_labels_tsv else args.labels_tsv
    expr_dict, esm2_dict, df = load_matched_inputs(args, test_labels)

    model, ckpt = load_model_from_ckpt(args.ckpt_path, args.device)
    esm_dim = int(next(iter(esm2_dict.values())).shape[-1])
    if esm_dim != int(ckpt["esm_dim"]):
        raise ValueError(f"ESM2 dim mismatch: ckpt expects {ckpt['esm_dim']}, got {esm_dim}")

    test_loader = make_loader(df, expr_dict, esm2_dict, args.batch_size, shuffle=False)
    metrics = evaluate(model, test_loader, args.device, return_scores=True)
    print("Independent test metrics")
    print(
        f"loss {metrics['loss']:.4f} | acc {metrics['acc']:.4f} | "
        f"roc_auc {metrics['roc_auc']:.4f} | pr_auc {metrics['pr_auc']:.4f}"
    )
    test_pdf_path = str(Path(args.curve_output_dir) / args.test_curve_pdf)
    save_curve_pdf(
        [
            {
                "label": "Independent test",
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "y_true": metrics["y_true"],
                "y_score": metrics["y_score"],
            }
        ],
        test_pdf_path,
        title="Independent Test AUROC/AUPRC",
    )
    print(f"Saved test curves to: {test_pdf_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPI prediction with stacked cross-attention blocks and fixed rich pair fusion: ESM2 Query, expression Key/Value."
    )
    parser.add_argument("--mode", choices=["cv", "train_full", "test"], default="cv")

    parser.add_argument("--expr-file", required=True, help="Expression .h5ad (cells x genes), .pkl/.npz or headered table")
    parser.add_argument("--expr-layer", help="H5AD layer with model-ready expression; defaults to X; no normalization is applied")
    parser.add_argument("--esm2-file", "--esm2-pkl", dest="esm2_pkl", required=True, help="ESM2 gene-to-vector .pt, .pkl or .npz; legacy --esm2-pkl remains supported")
    parser.add_argument("--labels-tsv", help="Labelled TSV/CSV; optional header: gene_i gene_j label")
    parser.add_argument("--test-labels-tsv", default=None)
    parser.add_argument("--ckpt-path", default="ex_crossattn_block.pt")
    parser.add_argument("--curve-output-dir", default=".")
    parser.add_argument("--cv-curve-pdf", default="cv_auroc_auprc_block.pdf")
    parser.add_argument("--test-curve-pdf", default="test_auroc_auprc_block.pdf")
    parser.add_argument("--missing-report", help="Dropped-pair TSV; defaults to <curve-output-dir>/<mode>_dropped_pairs.tsv")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-cross-layers", type=int, default=2)
    parser.add_argument("--expr-token-hidden", type=int, default=32)
    parser.add_argument("--cls-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--esm-drop-prob", type=float, default=0.3)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.1)
    parser.add_argument("--sim-weight", type=float, default=0.7, help="Weight of the fixed independent expression-shuffle loss")
    parser.add_argument("--sim-margin", type=float, default=0.1)

    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--full-val-ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.labels_tsv is None and not (args.mode == "test" and args.test_labels_tsv):
        parser.error("--labels-tsv is required (or --test-labels-tsv in test mode).")
    if args.batch_size < 1 or args.epochs < 1 or args.early_stop_patience < 1:
        parser.error("batch-size, epochs, and early-stop-patience must be positive.")
    if args.n_heads < 1 or args.d_model < 1 or args.d_model % args.n_heads or args.cls_hidden < 2:
        parser.error("d-model must be positive and divisible by n-heads; cls-hidden must be >=2.")
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "cv":
        run_cv(args)
    elif args.mode == "train_full":
        run_train_full(args)
    else:
        run_test(args)
