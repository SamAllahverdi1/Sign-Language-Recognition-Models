
#Phoenix HRNet layout: 
# body joints 0-16, dims 0-51
# feet joins 17-22, dims 51-68
# face joints 32-90, dims 69-272
# left hand joints 91-111, dims 273-335
# right hand joints 112-132, dims 336-298


import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from .config import MambaASLConfig


# dimension ranges for flat inputs
_LANDMARK_RANGES = {
    "body":        (0,    51),
    "feet":        (51,   69),
    "face":        (69,  273),
    "left_hand":   (273, 336),
    "right_hand":  (336, 399),
}


class LandmarkInputProjection(nn.Module):
    # Project landmark groups, softmax/normalize, recombine 
    # input (B,T,399)
    # output (B,T,d_model)
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projs = nn.ModuleDict({
            name: nn.Linear(end - start, d_model)
            for name, (start, end) in _LANDMARK_RANGES.items()
        })
        self.importance = nn.Parameter(torch.ones(len(_LANDMARK_RANGES)))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        for proj in self.projs.values():
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [
            self.projs[name](x[..., start:end])
            for name, (start, end) in _LANDMARK_RANGES.items()
        ]                            
        # 5 * (B, T, d_model)
        weights = F.softmax(self.importance, dim=0)
        # (5,)
        out = torch.stack(feats, dim=0)     
        # (5, B, T, d_model)
        out = (weights[:, None, None, None] * out).sum(0)
        # (B, T, d_model)
        return self.drop(self.norm(out))

    def landmark_weights(self) -> dict:
        w = F.softmax(self.importance.detach(), dim=0).tolist()
        return dict(zip(_LANDMARK_RANGES.keys(), w))


class MambaASLFast(nn.Module):
    def __init__(self, cfg: MambaASLConfig):
        super().__init__()
        self.cfg = cfg
        #input projection with landmarks 
        self.input_proj = LandmarkInputProjection(cfg.d_model, cfg.dropout)
        self.dropout    = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout(cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.ctc_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)
   
    def landmark_weights(self) -> dict:
        return self.input_proj.landmark_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model]
        x = self.input_proj(x)
        x = self.dropout(x)
        # add dropout
        for layer, norm, drop in zip(self.layers, self.norms, self.layer_dropouts):
            x = x + drop(layer(norm(x)))
        x = self.norm_out(x)
        # [B, T, vocab_size]
        logits = self.ctc_head(x)
        # [T, B, vocab_size]
        return logits.log_softmax(-1).transpose(0, 1)
