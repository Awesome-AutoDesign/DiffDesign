import torch
import torch.nn as nn
from .meta_prior import MetaPrior
from .clip_adapter import CLIPAdapter

class DiffDesign(nn.Module):
    def __init__(self, unet, tokenizer, meta_prior_dim=768, fusion_type='add', use_gating=True):
        super().__init__()
        self.unet = unet
        self.tokenizer = tokenizer
        self.meta_prior = MetaPrior(meta_prior_dim)
        self.fusion_type = fusion_type
        self.use_gating = use_gating
        self.clip_adapter = CLIPAdapter('openai/clip-vit-large-patch14')
        self.fusion_proj = nn.Linear(meta_prior_dim, unet.input_blocks[0][0].in_channels)
        self.gate = nn.Parameter(torch.ones(unet.input_blocks[0][0].in_channels))

    def encode_prompt(self, prompt):
        return self.clip_adapter.get_text_features(prompt)

    def encode_image(self, img):
        return self.clip_adapter.get_image_features(img)

    def fuse(self, x, meta_proj):
        if self.use_gating:
            meta_proj = meta_proj * self.gate.view(1, -1, 1, 1)
        if self.fusion_type == 'add':
            return x + meta_proj
        elif self.fusion_type == 'concat':
            return torch.cat([x, meta_proj], dim=1)
        elif self.fusion_type == 'mul':
            return x * (1 + torch.sigmoid(meta_proj))
        else:
            return x

    def forward(self, x, prompt, img, appearance_feat=None, spec_feat=None, mask=None):
        text_feat = self.encode_prompt(prompt)
        img_feat = self.encode_image(img)
        meta = self.meta_prior(text_feat, img_feat, appearance_feat, spec_feat)
        meta_proj = self.fusion_proj(meta).unsqueeze(-1).unsqueeze(-1)
        x = self.fuse(x, meta_proj)
        x = self.unet(x, mask=mask)
        return x

    def sample(self, prompt, img, steps=50, guidance_scale=7.5):
        with torch.no_grad():
            x = torch.randn(1, self.unet.input_blocks[0][0].in_channels, 256, 256).to(next(self.parameters()).device)
            for t in range(steps):
                x = self.forward(x, prompt, img)
                if (t+1) % 10 == 0:
                    print(f"Sampling step {t+1}/{steps}")
            return x

    def freeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = False
