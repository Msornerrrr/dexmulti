import torch
import torch.nn as nn
import torch.nn.functional as F

class BetterAttentionPointNetEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=128,
                 block_channel=(256, 256, 256, 512),
                 use_layernorm=False,
                 reduction='attention',
                 final_norm='none',
                 **kwargs):
        super().__init__()
        self.reduction = reduction
        self.out_channels = out_channels

        # shared MLP for per-point feature extraction
        layers = []
        last_c = in_channels
        for c in block_channel:
            layers.append(nn.Linear(last_c, c))
            if use_layernorm:
                layers.append(nn.LayerNorm(c))
            layers.append(nn.ReLU())
            last_c = c
        self.mlp = nn.Sequential(*layers)

        # reduction variants
        if reduction == 'learned':
            self.red_mlp = nn.Sequential(
                nn.Linear(block_channel[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif reduction == 'attention':
            self.key = nn.Parameter(torch.randn(block_channel[-1]))
            self.q_mlp = nn.Sequential(
                nn.Linear(block_channel[-1], 256),
                nn.ReLU(),
                nn.Linear(256, block_channel[-1])
            )

        # final projection to out_channels
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        else:
            self.final_projection = nn.Linear(block_channel[-1], out_channels)

    def forward(self, x, return_attn=False):
        """
        Args:
            x: [B, N, C] point cloud input
            return_attn: whether to return attention weights
        Returns:
            global_feat: [B, out_channels]
            (optionally) attn_weights: [B, N]
        """
        B, N, _ = x.shape
        feat = self.mlp(x)  # [B, N, F]

        if self.reduction == 'max':
            global_feat = feat.max(dim=1).values  # [B, F]
            attn_weights = None

        elif self.reduction == 'learned':
            logits = self.red_mlp(feat)  # [B, N, 1]
            weights = F.softmax(logits, dim=1)
            global_feat = torch.sum(feat * weights, dim=1)
            attn_weights = weights.squeeze(-1)

        elif self.reduction == 'attention':
            q = self.q_mlp(feat)  # [B, N, 256]
            logits = torch.einsum('bnd,d->bn', q, self.key) / (256 ** 0.5)
            weights = F.softmax(logits, dim=1)
            global_feat = torch.einsum('bn,bnf->bf', weights, feat)
            attn_weights = weights

        global_feat = self.final_projection(global_feat)

        if return_attn:
            return global_feat, attn_weights
        return global_feat
