# tokenlization_stage/models/rvq.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple

from .abstract_vq import AbstractVQ

# 复用你已有实现（若已在其它文件中，按需改成 from .xxx import MLP, VQEmbedding, RQBottleneck）
# 这里内联最小依赖，避免循环导入
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super().__init__()
        self.in_dropout = nn.Dropout(p=dropout)
        layers = []
        last = input_size
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.LayerNorm(hs), nn.ReLU(), nn.Dropout(dropout)]
            last = hs
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.proj = nn.Linear(last, latent_size) if last != latent_size else nn.Identity()

    def forward(self, x):
        x = self.in_dropout(x)
        x = self.mlp(x)
        x = self.proj(x)
        return x


class VQEmbedding(nn.Embedding):
    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()  # [D, K]
        inputs_flat = inputs.reshape(-1, codebook_t.shape[0])  # [N, D]
        inputs_norm_sq = inputs_flat.pow(2).sum(1, keepdim=True)  # [N, 1]
        cb_norm_sq = codebook_t.pow(2).sum(0, keepdim=True)       # [1, K]
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        distances = torch.addmm(inputs_norm_sq + cb_norm_sq, inputs_flat, codebook_t, alpha=-2.0)
        return distances.reshape(*inputs.shape[:-1], -1)  # [..., K]

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        return self.compute_distances(inputs).argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, D = x.shape
        reps = (target_n + B - 1) // B
        std = x.new_ones(D) * 0.01 / (D ** 0.5)
        x = x.repeat(reps, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        import torch.distributed as dist
        ema_decay = self.decay

        K = self.weight.shape[0] - 1
        D = self.weight.shape[1]
        v = vectors.reshape(-1, D)      # [N, D]
        i = idxs.reshape(-1)            # [N]

        one_hot = v.new_zeros(K, v.size(0))
        one_hot.scatter_(0, i.unsqueeze(0), 1.0)

        cluster_size = one_hot.sum(1)                # [K]
        embed_sum = one_hot @ v                      # [K, D]

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(ema_decay).add_(cluster_size, alpha=1 - ema_decay)
        self.embed_ema.mul_(ema_decay).add_(embed_sum, alpha=1 - ema_decay)

        # Re-init unused
        if self.restart_unused_codes:
            if v.size(0) < K:
                v = self._tile_with_noise(v, K)
            with torch.no_grad():
                usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()  # [K,1]
                randv = v[torch.randperm(v.size(0), device=v.device)[:K]]
                self.embed_ema.mul_(usage).add_(randv * (1 - usage))
                self.cluster_size_ema.mul_(usage.view(-1)).add_((1 - usage.view(-1)))

    @torch.no_grad()
    def _update_embedding(self):
        K = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        norm_cluster = n * (self.cluster_size_ema + self.eps) / (n + K * self.eps)
        self.weight[:-1, :] = self.embed_ema / norm_cluster.reshape(-1, 1)

    def forward(self, x):
        idxs = self.find_nearest_embedding(x)
        if self.training and self.ema:
            self._update_buffers(x, idxs)
        embeds = self.embed(idxs)
        if self.training and self.ema:
            self._update_embedding()
        return embeds, idxs

    def embed(self, idxs):
        return super().forward(idxs)


class RQBottleneck(nn.Module):
    """多级残差量化（Residual VQ）。"""
    def __init__(
        self,
        latent_dim: int,
        num_levels: int,
        codebook_size: int,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        shared_codebook: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.shared_codebook = shared_codebook

        if shared_codebook:
            cb = VQEmbedding(codebook_size, latent_dim, decay=decay, restart_unused_codes=restart_unused_codes)
            self.codebooks = nn.ModuleList([cb for _ in range(num_levels)])
        else:
            self.codebooks = nn.ModuleList([
                VQEmbedding(codebook_size, latent_dim, decay=decay, restart_unused_codes=restart_unused_codes)
                for _ in range(num_levels)
            ])

    def quantize(self, x):
        residual = x
        agg = torch.zeros_like(x)
        quants = []
        codes = []
        for l in range(self.num_levels):
            q, c = self.codebooks[l](residual)
            residual = residual - q
            agg = agg + q
            quants.append(agg.clone())
            codes.append(c.unsqueeze(-1))  # [..., 1]
        codes = torch.cat(codes, dim=-1)  # [..., L]
        return quants, codes

    def forward(self, x):
        quants, codes = self.quantize(x)
        # Straight-Through: 仅让梯度回传到输入
        z_q = quants[-1]
        z_q = x + (z_q - x).detach()
        # 承诺损失（逐级累积）
        commit = torch.mean(torch.stack([(x - q.detach()).pow(2).mean() for q in quants]))
        return z_q, commit, codes

    @torch.no_grad()
    def embed_code(self, codes):
        # codes: [..., L]
        embeds = []
        for l in range(self.num_levels):
            e = self.codebooks[l].embed(codes[..., l])
            embeds.append(e)
        return torch.stack(embeds, dim=-1).sum(dim=-1)  # [..., D]


class RVQ(AbstractVQ):
    """
    Residual Vector Quantizer
    - 输入: 连续向量 x ∈ R^{D}
    - 编码: 可选 MLP 将 D -> latent_size
    - 量化: 多级残差量化 (num_levels, codebook_size)
    - 解码: 可选 MLP 将 latent_size -> D（若 latent_size != D）
    - 输出: 重构 \hat{x}、承诺损失、codes（离散ID序列）
    """
    def __init__(self, config: Dict, input_size: int):
        super().__init__(config)
        mp = config["rvq"]["model_params"]
        tp = config["rvq"]["training_params"]

        self.loss_type = tp.get("loss_type", "mse")
        self.beta = tp.get("beta", 0.25)

        hidden_sizes = mp.get("hidden_sizes", [])
        latent_size = mp["latent_size"]
        num_levels = mp["num_levels"]
        codebook_size = mp["codebook_size"]
        dropout = mp.get("dropout", 0.0)
        shared = mp.get("shared_codebook", False)
        decay = mp.get("ema_decay", 0.99)
        restart = mp.get("restart_unused_codes", True)

        # 编码/解码
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout) if hidden_sizes else nn.Linear(input_size, latent_size)
        self.decoder = (
            MLP(latent_size, list(reversed(hidden_sizes)), input_size, dropout=dropout)
            if hidden_sizes else
            (nn.Linear(latent_size, input_size) if latent_size != input_size else nn.Identity())
        )

        # 残差量化瓶颈
        self.quantizer = RQBottleneck(
            latent_dim=latent_size,
            num_levels=num_levels,
            codebook_size=codebook_size,
            decay=decay,
            restart_unused_codes=restart,
            shared_codebook=shared,
        )

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(xs)                   # [B, D_lat]
        z_q, commit_loss, codes = self.quantizer(z_e)
        out = self.decoder(z_q)                  # [B, D_in]
        return out, commit_loss, codes

    @torch.no_grad()
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(xs)
        _, _, codes = self.quantizer(z_e)
        return codes  # [B, num_levels]

    def compute_loss(self, forward_outputs, batch_data=None) -> dict:
        """ 
        返回重構 Loss。
        【已修正】返回 detached 的 Tensor，切斷梯度流。
        """
        _, recon_loss, _ = forward_outputs # recon_loss 是一個 Tensor
        
        # ✅ 關鍵修正：返回 .detach() 後的 Tensor
        # 這會告訴 Trainer，這個 loss 不需要用於反向傳播更新參數
        # Trainer 仍然可以對其 .item()
        return { 
            "loss_total": recon_loss.detach(),
            "loss_recon": recon_loss.detach()
        }
