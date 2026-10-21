"""Shared rich-fusion model definitions and checkpoint serialization.

Importing this module defines the model without starting training or inference.
Checkpoint tensor names and architecture metadata retain their original format.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "model" / "model_trained_by_HEK293.pt"


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=self.norm_attn(h),
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = h + self.drop_attn(attn_out)
        h = h + self.drop_ffn(self.ffn(self.norm_ffn(h)))
        return h


class GeneCrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_cross_layers: int = 2,
        expr_token_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.expr_token_proj = nn.Sequential(
            nn.Linear(1, expr_token_hidden),
            nn.ReLU(),
            nn.Linear(expr_token_hidden, d_model),
        )
        self.esm_proj = nn.Linear(esm_dim, d_model)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_cross_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, expr_pad: torch.Tensor, expr_len: torch.Tensor, esm_vec: torch.Tensor) -> torch.Tensor:
        expr_tokens = self.expr_token_proj(expr_pad.unsqueeze(-1))
        h = self.esm_proj(esm_vec).unsqueeze(1)

        bsz, seq_len, _ = expr_tokens.shape
        idx = torch.arange(seq_len, device=expr_tokens.device).unsqueeze(0).expand(bsz, seq_len)
        key_padding_mask = idx >= expr_len.unsqueeze(1)

        for block in self.blocks:
            h = block(h, expr_tokens, key_padding_mask)
        h = self.final_norm(h)
        return h.squeeze(1)


class CrossAttnPPIPredictor(nn.Module):
    """Predict PPIs with fixed rich pair features: [a, b, |a-b|, a*b]."""

    def __init__(
        self,
        esm_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_cross_layers: int = 2,
        expr_token_hidden: int = 32,
        cls_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = "rich"

        self.encoder = GeneCrossAttentionEncoder(
            esm_dim=esm_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            expr_token_hidden=expr_token_hidden,
            dropout=dropout,
        )

        in_dim = 4 * d_model

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, cls_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden // 2, 2),
        )

    def forward(
        self,
        expr_a_pad: torch.Tensor,
        len_a: torch.Tensor,
        expr_b_pad: torch.Tensor,
        len_b: torch.Tensor,
        esm_a: torch.Tensor,
        esm_b: torch.Tensor,
    ) -> torch.Tensor:
        z_a = self.encoder(expr_a_pad, len_a, esm_a)
        z_b = self.encoder(expr_b_pad, len_b, esm_b)

        feat = torch.cat([z_a, z_b, torch.abs(z_a - z_b), z_a * z_b], dim=-1)
        return self.classifier(feat)


def build_model_from_args(args, esm_dim: int) -> CrossAttnPPIPredictor:
    return CrossAttnPPIPredictor(
        esm_dim=esm_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_cross_layers=args.n_cross_layers,
        expr_token_hidden=args.expr_token_hidden,
        cls_hidden=args.cls_hidden,
        dropout=args.dropout,
    )


def save_checkpoint(path: str, model: nn.Module, args, esm_dim: int):
    ckpt = {
        "model": model.state_dict(),
        "esm_dim": esm_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_cross_layers": args.n_cross_layers,
        "expr_token_hidden": args.expr_token_hidden,
        "cls_hidden": args.cls_hidden,
        "dropout": args.dropout,
        "fusion": "rich",
        "training_args": dict(vars(args)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if ckpt.get("fusion", "rich") != "rich":
        raise ValueError(f"Only rich-fusion checkpoints are supported; got {ckpt['fusion']!r}.")
    model = CrossAttnPPIPredictor(
        esm_dim=int(ckpt["esm_dim"]),
        d_model=int(ckpt["d_model"]),
        n_heads=int(ckpt["n_heads"]),
        n_cross_layers=int(ckpt.get("n_cross_layers", 2)),
        expr_token_hidden=int(ckpt.get("expr_token_hidden", 32)),
        cls_hidden=int(ckpt["cls_hidden"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
