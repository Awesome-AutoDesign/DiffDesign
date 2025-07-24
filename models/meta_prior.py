import torch
import torch.nn as nn

class MetaPrior(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 4, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )
        self.res_weight = nn.Parameter(torch.ones(1))

    def forward(self, text_feat, img_feat, appearance_feat=None, spec_feat=None):
        feats = [text_feat, img_feat]
        if appearance_feat is not None:
            feats.append(appearance_feat)
        else:
            feats.append(torch.zeros_like(text_feat))
        if spec_feat is not None:
            feats.append(spec_feat)
        else:
            feats.append(torch.zeros_like(text_feat))
        feats = [self.norm(f) for f in feats]
        meta = torch.cat(feats, dim=-1)
        meta = self.fc(meta)
        # Residual trick
        return meta + self.res_weight * text_feat
