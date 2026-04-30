import torch
import torch.nn as nn

from config import LSTMASLConfig


class ConvBiLSTMCTC(nn.Module):

    def __init__(self, cfg: LSTMASLConfig):
        super().__init__()

        self.keypoint_conv = nn.Sequential(
            nn.Conv1d(cfg.n_keypoints, cfg.conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.conv_channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            nn.Conv1d(cfg.conv_channels, cfg.conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.conv_channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            nn.Conv1d(cfg.conv_channels, cfg.conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.conv_channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=cfg.conv_channels,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout,
        )

        self.output_norm = nn.LayerNorm(cfg.hidden_dim * 2)
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.token_classifier = nn.Linear(cfg.hidden_dim * 2, cfg.vocab_size)

    def forward(self, phoenix_keypoints: torch.Tensor) -> torch.Tensor:
        frame_features = phoenix_keypoints.transpose(1, 2)
        frame_features = self.keypoint_conv(frame_features)
        frame_features = frame_features.transpose(1, 2)

        frame_features, _ = self.bilstm(frame_features)

        frame_features = self.output_norm(frame_features)
        frame_features = self.output_dropout(frame_features)
        token_logits = self.token_classifier(frame_features)
        token_log_probs = token_logits.log_softmax(dim=-1)

        return token_log_probs.transpose(0, 1)