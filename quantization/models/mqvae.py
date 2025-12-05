# 文件路径: /quantization/models/mqvae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from typing import Dict, Any

# 假设 abstract_vq 在父目录或可導入
try:
    from .abstract_vq import AbstractVQ
except ImportError:
    # 简单的 Fallback
    class AbstractVQ(nn.Module):
        def __init__(self, config: dict): super().__init__(); self.config = config
        def forward(self, xs: torch.Tensor): raise NotImplementedError
        def get_codes(self, xs: torch.Tensor) -> torch.Tensor: raise NotImplementedError
        def compute_loss(self, *args, **kwargs) -> dict: raise NotImplementedError
        @property
        def is_iterative(self) -> bool: return True

# =================================================================
# 辅助模块 (从你的 RQVAE.py 中复制，确保文件独立)
# =================================================================

class MLP(nn.Module):
    """
    (从 RQVAE.py 复制)
    一个标准的多层感知机，用于 Encoder 和 Decoder。
    """
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
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)

class VQEmbedding(nn.Embedding):
    """
    (从 RQVAE.py 复制)
    VQ embedding module with ema update.
    """
    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        # +1 用于 [MASK] 标记 (我们将使用 padding_idx 作为 MASK)
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed # 真实 codebook 大小 (不含 MASK)

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone()) # 不更新 MASK token

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t() # 只与真实 code 比较
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq, inputs_flat, codebook_t, alpha=-2.0,
        )
        return distances.reshape(*inputs_shape[:-1], -1)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.n_embed, self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        n_vectors = vectors.shape[0]
        one_hot_idxs = vectors.new_zeros(n_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors))
        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed: vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            if dist.is_initialized(): dist.broadcast(_vectors_random, 0)
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (n * (self.cluster_size_ema + self.eps) / (n + self.n_embed * self.eps))
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        """
        与 find_nearest_embedding 不同, forward 负责量化和EMA更新。
        返回: 量化后的向量, 码本索引
        """
        embed_idxs = self.find_nearest_embedding(inputs) # [B, N]
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs) # [B, N, D]

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        return super().forward(idxs)

# =================================================================
# ================== 核心 MQ-VAE 模块 ==================
# =================================================================

class Masker(nn.Module):
    """
    MQ-VAE 论文中的 'Adaptive Mask Module' (Sec 3.2)。
    
    1. 使用评分网络为每个 patch/子向量 打分。
    2. 排序并选择 Top-N (重要的) features。
    3. 返回要量化的特征、原始特征以及它们的位置索引。
    """
    def __init__(self, num_patches, latent_dim, mask_ratio, hidden_dim_ratio=0.5):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        
        self.num_to_keep = int(num_patches * (1.0 - mask_ratio))
        
        # 论文: "lightweight scoring network fs... a two-layer MLP"
        scoring_hidden_dim = int(latent_dim * hidden_dim_ratio)
        self.scoring_network = nn.Sequential(
            nn.Linear(latent_dim, scoring_hidden_dim),
            nn.GELU(),
            nn.Linear(scoring_hidden_dim, 1)
        )

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Encoder 输出的特征序列, shape [B, L, D]
        
        Returns:
            dict: 包含 'sampled_features', 'sample_index' 等
        """
        B, L, D = z.shape
        assert L == self.num_patches, f"Masker 期望 L={self.num_patches} 但收到了 {L}"

        # 1. 评分
        scores = self.scoring_network(z).squeeze(-1) # [B, L]
        
        # 2. 排序和选择
        # Gumbel-Softmax/ST-estimator 在这里太复杂了，
        # 我们先用 'topk' 实现，并依赖论文中的 'score modulation' 来传递梯度
        
        # 排序
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True) # [B, L]
        
        # 根据分数重新排列特征
        z_sorted = torch.gather(z, dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, D)) # [B, L, D]
        
        # 3. 分割
        # 重要的 (Top-N)
        sampled_features = z_sorted[:, :self.num_to_keep, :] # [B, N, D]
        sample_index = sorted_indices[:, :self.num_to_keep]   # [B, N]
        
        # 冗余的 (M)
        remain_features = z_sorted[:, self.num_to_keep:, :]  # [B, M, D]
        remain_index = sorted_indices[:, self.num_to_keep:]    # [B, M]

        # 4. 论文中的梯度技巧 (Sec 3.2, Eq. 4)
        # "to enable the learning of fs, the predicted scores are further 
        # multiplied with the normalized region features"
        sampled_features_norm = F.layer_norm(sampled_features, [D])
        top_scores = sorted_scores[:, :self.num_to_keep].unsqueeze(-1)
        
        # (重要) 我们将这个 'scaled' 版本用于重构
        # 但将 'original' 版本用于量化 (计算最近邻)
        sampled_features_scaled = sampled_features_norm * top_scores
        
        return {
            "sampled_features_original": sampled_features,      # [B, N, D] -> 用于 VQ 查找
            "sampled_features_scaled": sampled_features_scaled, # [B, N, D] -> 用于 ST 梯度
            "sample_index": sample_index,                       # [B, N]
            "remain_features": remain_features,                 # [B, M, D]
            "remain_index": remain_index,                       # [B, M]
            "score_map": scores,                                # [B, L] (用于日志)
        }

class Demasker(nn.Module):
    """
    MQ-VAE 论文中的 'Adaptive De-mask Module' (Sec 3.2)。
    
    本质上是一个 Transformer Encoder，用于在填充了 [MASK] 标记的序列上
    运行，以恢复被掩码的特征。
    """
    def __init__(self, latent_dim, num_patches, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim = latent_dim
        
        # 可学习的绝对位置嵌入
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, latent_dim))

        # "direction-constrained self-attention" 在这里过于复杂,
        # 一个标准的 Transformer Encoder 已经足够学会恢复 MASK。
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=n_heads, 
            dim_feedforward=latent_dim * 4,
            dropout=dropout, 
            batch_first=True, 
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, filled_seq):
        """
        Args:
            filled_seq (torch.Tensor): [B, L, D] 
                已填充了 'sampled_quant_st' 和 'mask_token' 的序列。
        
        Returns:
            torch.Tensor: [B, L, D] 恢复后的完整序列
        """
        # 1. 添加位置嵌入
        x = filled_seq + self.pos_emb
        
        # 2. 通过 Transformer 恢复
        reconstructed_seq = self.transformer(x)
        return reconstructed_seq

# =================================================================
# ================== MQ-VAE 量化器主类 ==================
# =================================================================

class MQVAE(AbstractVQ):
    """
    Masked Quantization VAE (MQ-VAE)
    
    适配了你的 `quantization/trainer.py` 框架。
    - Encoder: MLP (D_in -> L*D_z)
    - Masker: 评分并选择 Top-N 个子向量
    - Quantizer: VQEmbedding (只量化 Top-N)
    - Demasker: Transformer (从 N 个恢复 L 个)
    - Decoder: MLP (L*D_z -> D_in)
    """
    def __init__(self, config: dict, input_size: int):
        super().__init__(config)
        
        # 1. 解析配置
        model_cfg = config.get('mqvae', {})
        model_params = model_cfg.get('model_params', {})
        train_params = model_cfg.get('training_params', {})

        # Encoder/Decoder 超参数
        hidden_sizes = model_params.get('hidden_sizes', [1024, 512])
        dropout = model_params.get('dropout', 0.1)
        
        # MQ-VAE 核心超参数
        self.latent_dim = model_params.get('latent_dim', 64)       # (D_z)
        self.num_patches = model_params.get('num_patches', 32)     # (L)
        self.mask_ratio = model_params.get('mask_ratio', 0.5)      # 掩码比例
        self.codebook_size = model_params.get('codebook_size', 1024) # (K)
        
        # Demasker 超参数
        demasker_layers = model_params.get('demasker_layers', 2)
        demasker_heads = model_params.get('demasker_heads', 4)

        # 损失权重
        self.latent_loss_weight = train_params.get('latent_loss_weight', 0.25)
        self.loss_type = train_params.get('loss_type', 'mse')

        # 2. 构建模块
        
        # Encoder (D_in -> L * D_z)
        encoder_latent_size = self.num_patches * self.latent_dim
        self.encoder = MLP(input_size, hidden_sizes, encoder_latent_size, dropout=dropout)
        
        # Masker (L, D_z)
        self.masker = Masker(self.num_patches, self.latent_dim, self.mask_ratio)
        
        # Quantizer (K, D_z)
        self.quantizer = VQEmbedding(
            n_embed=self.codebook_size,
            embed_dim=self.latent_dim,
            decay=train_params.get('vq_decay', 0.99),
        )
        # 码本大小 (不含 mask token)
        self.mask_code_idx = self.codebook_size # MASK token 的索引是 K

        # Demasker (L, D_z)
        self.demasker = Demasker(
            latent_dim=self.latent_dim,
            num_patches=self.num_patches,
            n_layers=demasker_layers,
            n_heads=demasker_heads,
            dropout=dropout
        )

        # Decoder (L * D_z -> D_in)
        rev_hidden_sizes = hidden_sizes.copy()
        rev_hidden_sizes.reverse()
        self.decoder = MLP(encoder_latent_size, rev_hidden_sizes, input_size, dropout=dropout)
        
        # [MASK] 标记 (使用 VQEmbedding 的 padding_idx 嵌入)
        # 我们不需要一个单独的 mask_token 参数

    @property
    def is_iterative(self) -> bool:
        return True # 这是一个需要迭代训练的模型

    def forward(self, xs: torch.Tensor):
        """
        完整的前向传播 (用于训练)。
        
        返回: (重构向量, 量化损失, (码, 索引))
        """
        B, D_in = xs.shape
        L, D_z = self.num_patches, self.latent_dim
        
        # 1. Encoder (D_in -> L*D_z -> [B, L, D_z])
        z_e_flat = self.encoder(xs)
        z_e = z_e_flat.view(B, L, D_z)
        
        # 2. Masker (获取重要的子向量)
        masker_output = self.masker(z_e)
        sampled_features_original = masker_output["sampled_features_original"] # [B, N, D_z]
        sampled_features_scaled = masker_output["sampled_features_scaled"]   # [B, N, D_z]
        sample_index = masker_output["sample_index"]                     # [B, N]
        
        # 3. Quantizer (只量化重要的)
        # sampled_quant: 量化后的向量, [B, N, D_z]
        # sampled_codes: 码本索引, [B, N]
        sampled_quant, sampled_codes = self.quantizer(sampled_features_original)
        
        # 4. 计算量化损失
        # 提交损失 (Commitment Loss)
        loss_commit = F.mse_loss(sampled_features_original.detach(), sampled_quant)
        
        # 直通梯度 (Straight-Through Estimator)
        # 我们使用论文中的 'scaled' 特征来传递梯度给评分网络
        sampled_quant_st = sampled_features_scaled + (sampled_quant - sampled_features_scaled).detach()
        
        # 5. Demasker (恢复全序列)
        
        # 获取 [MASK] 标记的嵌入 (即 VQEmbedding 的 padding_idx)
        mask_token_emb = self.quantizer.embed(
            torch.tensor([self.mask_code_idx], device=xs.device)
        ).expand(B, L, -1)
        
        # 将量化后的向量 [B, N, D_z] 散布回 [B, L, D_z] 的 MASK 序列中
        idx = sample_index.unsqueeze(-1).expand(-1, -1, D_z)
        filled_seq = mask_token_emb.scatter(dim=1, index=idx, src=sampled_quant_st)
        
        # 运行 Transformer 进行恢复
        z_q = self.demasker(filled_seq) # [B, L, D_z]
        
        # 6. Decoder
        z_q_flat = z_q.view(B, L * D_z)
        out = self.decoder(z_q_flat)
        
        # (返回码和索引，用于 get_codes)
        codes_tuple = (sampled_codes, sample_index)
        
        return out, loss_commit, codes_tuple

    @torch.no_grad()
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        """
        仅用于预测 (生成码本)。
        
        返回一个固定的 [B, L] 码本，未被选中的位置用 MASK 索引填充。
        这确保了 `trainer.py` 中的 `np.vstack` 可以工作。
        """
        B, D_in = xs.shape
        L, D_z = self.num_patches, self.latent_dim
        
        # 1. Encoder
        z_e = self.encoder(xs).view(B, L, D_z)
        
        # 2. Masker
        masker_output = self.masker(z_e)
        sampled_features_original = masker_output["sampled_features_original"] # [B, N, D_z]
        sample_index = masker_output["sample_index"]                     # [B, N]
        
        # 3. Quantizer (只获取索引)
        sampled_codes = self.quantizer.find_nearest_embedding(sampled_features_original) # [B, N]
        
        # 4. 创建固定的 [B, L] 码图
        # 用 MASK 索引填充
        full_codes = torch.full((B, L), self.mask_code_idx, dtype=torch.long, device=xs.device)
        
        # 5. 将真实码散布到对应位置
        full_codes.scatter_(dim=1, index=sample_index, src=sampled_codes)
        
        return full_codes # [B, L]

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """
        计算总损失。
        """
        out, loss_commit, _ = forward_outputs
        
        # 1. 重构损失
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, batch_data, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, batch_data, reduction="mean")
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
            
        # 2. 量化损失 (只有 commitment loss)
        loss_latent = loss_commit

        # 3. 总损失
        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
        }