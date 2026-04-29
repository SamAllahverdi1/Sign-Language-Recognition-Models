import math

import torch
import torch.nn as nn

from config import TransformerASLConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerASL(nn.Module):
    def __init__(self, cfg: TransformerASLConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_keypoints, cfg.d_model)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.pos_encoder = PositionalEncoding(cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.RMSNorm(cfg.d_model)
        self.ctc_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        x = self.norm_out(x)
        logits = self.ctc_head(x)
        return logits.log_softmax(-1).transpose(0, 1)
