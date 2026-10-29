"""
Standalone Reusable PyTorch Module for Intelligent Medicine (Elsevier):
Uncertainty-Aware Cross-Modal Attention Fusion (CM-GAF) + Mondrian Split Conformal Triage
Authors:
  1. Dr. Juan Marcos Vilchez Canchari, PhD (Universidad César Vallejo, Lima, Peru; jvilchezca@ucv.edu.pe)
  2. Dra. Ruth Janett Palomino Rojas, PhD (Universidad Privada San Juan Bautista, Lima, Peru)
  3. Dra. Mirian Corina Cribillero Roca, PhD (Universidad Nacional del Callao, Callao, Peru)
  4. Dra. Luz Chavela De la Torre Guzmán, PhD (Universidad Nacional del Callao, Callao, Peru)
  5. Dra. Vilma María Arroyo Vigil, PhD (Universidad Nacional del Callao, Callao, Peru)
"""

import math
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import timm


class CrossModalGatedFusion(nn.Module):
    """Proposed CM-GAF module: Clinically guided FiLM modulation + Multi-Head Cross-Modal
    Attention + Adaptive Sigmoid Gating."""

    def __init__(self, vis_dim: int, tab_dim: int = 13, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, hidden_dim),
            nn.GELU()
        )
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.film_generator = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, v_feat: torch.Tensor, t_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.vis_proj(v_feat)
        t = self.tab_encoder(t_raw)
        gamma, beta = self.film_generator(t).chunk(2, dim=-1)
        v_mod = v * (1.0 + torch.tanh(gamma)) + beta

        q = t.unsqueeze(1)
        kv = torch.stack([v, v_mod], dim=1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        attn_vec = attn_out.squeeze(1)

        g = self.gate(torch.cat([v_mod, attn_vec], dim=-1))
        fused = self.norm_out(g * v_mod + (1.0 - g) * attn_vec)
        return fused, g.mean(dim=-1)


class SkinTriageNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny.fb_in22k",
        tab_dim: int = 13,
        mode: str = "cm_gaf",
        pretrained: bool = False,
        dropout: float = 0.25
    ):
        super().__init__()
        self.mode = mode
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        vis_dim = self.backbone.num_features

        if mode == "image_only":
            self.head = nn.Sequential(
                nn.LayerNorm(vis_dim),
                nn.Dropout(dropout),
                nn.Linear(vis_dim, 1)
            )
        elif mode == "late_concat":
            self.tab_mlp = nn.Sequential(
                nn.Linear(tab_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(vis_dim + 64),
                nn.Dropout(dropout),
                nn.Linear(vis_dim + 64, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )
        elif mode == "cm_gaf":
            self.fusion = CrossModalGatedFusion(
                vis_dim=vis_dim, tab_dim=tab_dim, hidden_dim=256, dropout=dropout
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        v_feat = self.backbone(image)
        if self.mode == "image_only":
            return self.head(v_feat).squeeze(-1)
        elif self.mode == "late_concat":
            t_feat = self.tab_mlp(tabular)
            return self.head(torch.cat([v_feat, t_feat], dim=-1)).squeeze(-1)
        else:
            fused, _ = self.fusion(v_feat, tabular)
            return self.head(fused).squeeze(-1)


def conformal_triage_decision(prob: float, tau_out: float = 0.1590, tau_in: float = 0.3984) -> str:
    """Three-zone clinical triage rule calibrated via Mondrian Split Conformal Prediction."""
    if prob < tau_out:
        return "Zone 1: Safe Rule-Out (Benign, FNR <= 5%)"
    elif prob < tau_in:
        return "Zone 2: Conformal Grey Zone (Defer to Specialist Dermoscopy)"
    else:
        return "Zone 3: Urgent Biopsy Rule-In (High Malignancy Risk)"

