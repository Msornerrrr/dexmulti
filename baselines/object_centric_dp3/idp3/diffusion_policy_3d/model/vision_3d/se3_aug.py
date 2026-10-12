import numpy as np
import torch

class SE3Augmentation:
    def __init__(self, cfg):
        self.rotation = cfg.rotation
        self.rotation_angle = torch.tensor(cfg.rotation_angle) * np.pi / 180.0
        self.translation = cfg.translation
        self.translation_scale = cfg.translation_scale
        self.jitter = cfg.jitter
        self.jitter_scale = cfg.jitter_scale

    def __call__(self, points):
        B, N, D = points.shape
        device = points.device
        xyz = points[..., :3]
        others = points[..., 3:] if D > 3 else None

        if self.rotation:
            rotation_angle = self.rotation_angle.to(device)
            angles = (torch.rand(B, 3, device=device) * 2 - 1) * rotation_angle
            cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
            sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])

            # Build batched rotation matrices [B, 3, 3]
            R = torch.stack([
                torch.stack([cy * cz, -cy * sz, sy], dim=1),
                torch.stack([sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], dim=1),
                torch.stack([-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy], dim=1),
            ], dim=1)

            xyz = torch.bmm(xyz, R.transpose(1, 2))

        if self.translation:
            translations = torch.empty((B, 1, 3), device=device).uniform_(
                -self.translation_scale, self.translation_scale
            )
            translations[..., 2] = 0
            xyz += translations

        if self.jitter:
            xyz += torch.randn((B, N, 3), device=device) * self.jitter_scale

        return torch.cat([xyz, others], dim=-1) if others is not None else xyz
