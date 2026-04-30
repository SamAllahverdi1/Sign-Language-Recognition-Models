import math

import torch
import torch.nn as nn

from config import TransformerASLConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        frame_positions = torch.arange(max_len).unsqueeze(1)
        even_dimension_steps = torch.arange(0, d_model, 2)

        frequency_scale = torch.exp(even_dimension_steps * (-math.log(10000.0) / d_model))

        position_table = torch.zeros(1, max_len, d_model)

        # weird formula
        position_table[0, :, 0::2] = torch.sin(frame_positions * frequency_scale)
        position_table[0, :, 1::2] = torch.cos(frame_positions * frequency_scale)

        self.register_buffer("position_table", position_table)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        frame_count = frame_features.size(1)
        position_slice = self.position_table[:, :frame_count, :]

        return frame_features + position_slice


class ASLTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.attention_norm = nn.RMSNorm(d_model)
        self.feedforward_norm = nn.RMSNorm(d_model)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(), # GELU Guassian Error not ReLU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, frame_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        
        attention_input = self.attention_norm(frame_features)

        attention_output, _ = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=padding_mask,
        )

        frame_features = frame_features + self.dropout(attention_output)

        feedforward_input = self.feedforward_norm(frame_features)
        feedforward_output = self.feedforward(feedforward_input)

        return frame_features + feedforward_output


class TransformerASL(nn.Module):
    def __init__(self, cfg: TransformerASLConfig):
        super().__init__()

        self.keypoint_projection = nn.Linear(cfg.n_keypoints, cfg.d_model)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.position_encoding = PositionalEncoding(cfg.d_model)

        self.transformer_layers = nn.ModuleList([
                ASLTransformerBlock(
                    cfg.d_model,
                    cfg.heads,
                    cfg.dim_feedforward,
                    cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ])

        self.output_norm = nn.RMSNorm(cfg.d_model)
        self.gloss_classifier = nn.Linear(cfg.d_model, cfg.vocab_size)

        # init values
        nn.init.xavier_uniform_(self.keypoint_projection.weight)
        nn.init.zeros_(self.keypoint_projection.bias)
        nn.init.xavier_uniform_(self.gloss_classifier.weight)
        nn.init.zeros_(self.gloss_classifier.bias)

    def forward(self, phoenix_keypoints: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        frame_features = self.keypoint_projection(phoenix_keypoints)
        frame_features = self.input_dropout(frame_features)
        frame_features = self.position_encoding(frame_features)

        for transformer_layer in self.transformer_layers:
            frame_features = transformer_layer(frame_features, padding_mask=src_key_padding_mask)

        frame_features = self.output_norm(frame_features)
        gloss_logits = self.gloss_classifier(frame_features)
        gloss_log_probs = gloss_logits.log_softmax(dim=-1)

        return gloss_log_probs.transpose(0, 1)