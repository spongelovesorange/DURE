# /tokenlization_stage/models/vqvae.py

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# 從你的抽象基類導入
from .abstract_vq import AbstractVQ

# --- 複用你提供的 MLP 結構 ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1, out_channels=output_size, kernel_size=input_size,
                    bias=False, stride=input_size
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)


# --- 核心：標準 VQ 量化器 ---
class VectorQuantizer(nn.Module):
    """
    標準的 VQ-VAE 量化瓶頸層 (非殘差)。
    """
    def __init__(self, n_embed: int, embed_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.commitment_cost = commitment_cost

    def forward(self, z_e: torch.Tensor) -> tuple:
        # z_e: [B, D] (編碼器輸出)
        # 1. 計算輸入與碼本中所有向量的L2距離的平方
        distances = (
            torch.sum(z_e.pow(2), dim=1, keepdim=True)
            - 2 * torch.matmul(z_e, self.embedding.weight.t())
            + torch.sum(self.embedding.weight.pow(2), dim=1)
        )
        
        # 2. 找到最近的碼本向量索引
        code = torch.argmin(distances, dim=1)  # [B]

        # 3. 獲取量化後的向量
        z_q = self.embedding(code)  # [B, D]

        # 4. 計算損失
        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        latent_loss = loss_codebook + self.commitment_cost * loss_commit

        # 5. 使用直通估計器 (Straight-Through Estimator)
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, latent_loss, code


# --- VQVAE 主模型 ---
class VQVAE(AbstractVQ):
    def __init__(self, config: dict, input_size: int):
        # 遵循 AbstractVQ 規範
        super().__init__(config)
        
        model_params = config['vqvae']['model_params']
        train_params = config['vqvae']['training_params']

        # 讀取模型參數
        hidden_sizes = model_params['hidden_sizes']
        latent_size = model_params['latent_size']
        codebook_size = model_params['codebook_size']
        dropout = model_params['dropout']
        commitment_cost = model_params.get('commitment_cost', 0.25)

        # 讀取訓練參數
        self.latent_loss_weight = train_params.get('latent_loss_weight', 1.0)
        
        # 構建網路
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        rev_hidden_sizes = hidden_sizes.copy()
        rev_hidden_sizes.reverse()
        self.decoder = MLP(latent_size, rev_hidden_sizes, input_size, dropout=dropout)
        
        self.quantizer = VectorQuantizer(
            n_embed=codebook_size,
            embed_dim=latent_size,
            commitment_cost=commitment_cost
        )

    def forward(self, xs: torch.Tensor) -> tuple:
        """模型的前向傳播"""
        z_e = self.encoder(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decoder(z_q)
        return out, quant_loss, code

    @torch.no_grad()
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        """僅用於預測，獲取離散碼"""
        z_e = self.encoder(xs)
        _, _, code = self.quantizer(z_e)
        # VQVAE 的 code 是一維的 [B]，為了和你框架的 L-dim code [B, L] 兼容，增加一個維度
        return code.unsqueeze(1)

    def compute_loss(self, forward_outputs: tuple, xs: torch.Tensor) -> dict:
        """計算總損失"""
        out, quant_loss, _ = forward_outputs
        
        # 重構損失
        loss_recon = F.mse_loss(out, xs, reduction="mean")
        
        # 量化損失
        loss_latent = quant_loss
        
        # 加權求和
        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
        }