import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
import math

from .abstract_vq import AbstractVQ


class MLP(nn.Module):
    """Multi-layer perceptron with residual connections."""
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


class LETTERVQEmbedding(nn.Embedding):
    """
    LETTER VQ embedding module with semantic regularization and diversity loss.
    Based on the LETTER paper: Learnable Item Tokenization for Generative Recommendation.
    """

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
        semantic_regularization=True,
        diversity_weight=0.1,
        temperature=1.0,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.semantic_regularization = semantic_regularization
        self.diversity_weight = diversity_weight
        self.temperature = temperature

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            # padding index is not updated by EMA
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

        # For diversity loss computation
        self.register_buffer("code_usage_count", torch.zeros(n_embed))
        
        # Semantic regularization components
        if self.semantic_regularization:
            self.semantic_projection = nn.Linear(embed_dim, embed_dim)
            self.semantic_norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def compute_distances(self, inputs):
        """Compute distances between inputs and codebook entries."""
        codebook_t = self.weight[:-1, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        # Apply temperature scaling for better code assignment
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        
        # Apply temperature scaling
        distances = distances / self.temperature
        
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        """Find nearest embedding with improved assignment strategy."""
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    def compute_diversity_loss(self, embed_idxs):
        """
        Compute diversity loss to encourage balanced code usage.
        This addresses the code assignment bias mentioned in the LETTER paper.
        """
        batch_size = embed_idxs.numel()
        if batch_size == 0:
            return torch.tensor(0.0, device=embed_idxs.device)
        
        # Count code usage in current batch
        code_counts = torch.bincount(embed_idxs.flatten(), minlength=self.n_embed)
        code_probs = code_counts.float() / batch_size
        
        # Update global code usage statistics
        self.code_usage_count = self.code_usage_count * 0.99 + code_counts.float() * 0.01
        
        # Compute entropy-based diversity loss
        # Higher entropy means more balanced usage
        log_probs = torch.log(code_probs + self.eps)
        entropy = -(code_probs * log_probs).sum()
        max_entropy = math.log(self.n_embed)
        
        # Diversity loss encourages maximum entropy
        diversity_loss = (max_entropy - entropy) / max_entropy
        
        return diversity_loss

    def compute_semantic_regularization_loss(self, inputs, quantized):
        """
        Compute semantic regularization loss to maintain semantic consistency.
        """
        if not self.semantic_regularization:
            return torch.tensor(0.0, device=inputs.device)
        
        # Project inputs through semantic projection
        semantic_inputs = self.semantic_norm(self.semantic_projection(inputs))
        semantic_quantized = self.semantic_norm(self.semantic_projection(quantized))
        
        # Compute semantic consistency loss
        semantic_loss = F.mse_loss(semantic_inputs, semantic_quantized)
        
        return semantic_loss

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        """Tile tensor with noise for unused code restart."""
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        """Update EMA buffers for codebook learning."""
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors)
        )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][
                :n_embed
            ]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    @torch.no_grad()
    def _update_embedding(self):
        """Update embedding weights using EMA."""
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        """Forward pass with LETTER enhancements."""
        embed_idxs = self.find_nearest_embedding(inputs)
        
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        """Embed indices to vectors."""
        embeds = super().forward(idxs)
        return embeds

    def get_code_utilization(self):
        """Get code utilization statistics."""
        total_usage = self.code_usage_count.sum()
        if total_usage > 0:
            utilization = (self.code_usage_count > 0).float().mean()
        else:
            utilization = torch.tensor(0.0, device=self.code_usage_count.device)
        return utilization


class CollaborativeAlignmentModule(nn.Module):
    """
    Collaborative alignment module for incorporating collaborative signals.
    This implements the contrastive alignment loss from the LETTER paper.
    """
    
    def __init__(self, semantic_dim, cf_dim, temperature=0.1):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.cf_dim = cf_dim
        self.temperature = temperature
        
        # Projection layers for alignment
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, cf_dim),
            nn.LayerNorm(cf_dim),
            nn.ReLU(),
            nn.Linear(cf_dim, cf_dim)
        )
        
        self.cf_proj = nn.Sequential(
            nn.Linear(cf_dim, cf_dim),
            nn.LayerNorm(cf_dim),
            nn.ReLU(),
            nn.Linear(cf_dim, cf_dim)
        )
        
    def forward(self, semantic_features, cf_features):
        """
        Compute collaborative alignment loss.
        
        Args:
            semantic_features: Semantic features from quantized representations
            cf_features: Collaborative filtering features
            
        Returns:
            alignment_loss: Contrastive alignment loss
        """
        # Project features to common space
        semantic_proj = F.normalize(self.semantic_proj(semantic_features), dim=-1)
        cf_proj = F.normalize(self.cf_proj(cf_features), dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(semantic_proj, cf_proj.transpose(-2, -1)) / self.temperature
        
        # Create positive pairs (diagonal elements)
        batch_size = semantic_features.size(0)
        labels = torch.arange(batch_size, device=semantic_features.device)
        
        # Contrastive loss (InfoNCE)
        loss_semantic = F.cross_entropy(similarity, labels)
        loss_cf = F.cross_entropy(similarity.transpose(-2, -1), labels)
        
        alignment_loss = (loss_semantic + loss_cf) / 2
        
        return alignment_loss


class LETTERRQBottleneck(nn.Module):
    """
    LETTER-enhanced Residual Quantization bottleneck.
    Integrates semantic regularization, collaborative alignment, and diversity loss.
    """

    def __init__(
        self,
        latent_shape,
        code_shape,
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss="cumsum",
        semantic_regularization=True,
        diversity_weight=0.1,
        temperature=1.0,
        cf_embedding_dim=None,
    ):
        super().__init__()

        self.latent_shape = latent_shape
        self.code_shape = code_shape
        self.shared_codebook = shared_codebook
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = [n_embed for n_embed in code_shape]
        self.decay = [decay for _ in range(len(self.code_shape))]
        self.semantic_regularization = semantic_regularization
        self.diversity_weight = diversity_weight

        # Create LETTER-enhanced VQ embeddings
        if self.shared_codebook:
            codebook0 = LETTERVQEmbedding(
                self.n_embed[0],
                self.latent_shape,
                decay=self.decay[0],
                restart_unused_codes=restart_unused_codes,
                semantic_regularization=semantic_regularization,
                diversity_weight=diversity_weight,
                temperature=temperature,
            )
            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(len(self.code_shape))]
            )
        else:
            codebooks = [
                LETTERVQEmbedding(
                    self.n_embed[idx],
                    latent_shape,
                    decay=self.decay[idx],
                    restart_unused_codes=restart_unused_codes,
                    semantic_regularization=semantic_regularization,
                    diversity_weight=diversity_weight,
                    temperature=temperature,
                )
                for idx in range(len(self.code_shape))
            ]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss
        
        # Collaborative alignment module
        if cf_embedding_dim is not None:
            self.collaborative_alignment = CollaborativeAlignmentModule(
                semantic_dim=latent_shape,
                cf_dim=cf_embedding_dim,
                temperature=temperature
            )
        else:
            self.collaborative_alignment = None

    def quantize(self, x):
        """
        Quantize with LETTER enhancements.
        """
        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        
        for i in range(len(self.code_shape)):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def compute_diversity_loss(self):
        """Compute diversity loss across all codebooks."""
        diversity_losses = []
        for codebook in self.codebooks:
            if hasattr(codebook, 'code_usage_count'):
                total_usage = codebook.code_usage_count.sum()
                if total_usage > 0:
                    code_probs = codebook.code_usage_count / total_usage
                    log_probs = torch.log(code_probs + codebook.eps)
                    entropy = -(code_probs * log_probs).sum()
                    max_entropy = math.log(codebook.n_embed)
                    diversity_loss = (max_entropy - entropy) / max_entropy
                    diversity_losses.append(diversity_loss)
        
        if diversity_losses:
            return torch.stack(diversity_losses).mean()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x, cf_features=None):
        """Forward pass with LETTER enhancements."""
        quant_list, codes = self.quantize(x)

        # Compute standard commitment loss
        commitment_loss = self.compute_commitment_loss(x, quant_list)
        
        # Compute diversity loss
        diversity_loss = self.compute_diversity_loss()
        
        # Compute collaborative alignment loss if CF features provided
        alignment_loss = torch.tensor(0.0, device=x.device)
        if self.collaborative_alignment is not None and cf_features is not None:
            alignment_loss = self.collaborative_alignment(quant_list[-1], cf_features)
        
        # Compute semantic regularization loss
        semantic_loss = torch.tensor(0.0, device=x.device)
        if self.semantic_regularization:
            for i, codebook in enumerate(self.codebooks):
                if hasattr(codebook, 'compute_semantic_regularization_loss'):
                    semantic_loss += codebook.compute_semantic_regularization_loss(
                        x, quant_list[i]
                    )
            semantic_loss = semantic_loss / len(self.codebooks)

        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        """Compute commitment loss for residual quantization."""
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    def get_code_utilization(self):
        """Get average code utilization across all codebooks."""
        utilizations = []
        for codebook in self.codebooks:
            if hasattr(codebook, 'get_code_utilization'):
                utilizations.append(codebook.get_code_utilization())
        
        if utilizations:
            return torch.stack(utilizations).mean()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    @torch.no_grad()
    def embed_code(self, code):
        """Embed codes to vectors."""
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [
                self.codebooks[0].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]
        else:
            embeds = [
                self.codebooks[i].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        return embeds


class RQVAE_LETTER(nn.Module):
    def __init__(
        self,
        config: dict,
        input_size: int,
        cf_embedding_dim: int = None,
    ):
        super().__init__()
        
        # Parse model parameters from config
        model_params = config['rqvae_letter']['model_params']
        hidden_sizes = model_params['hidden_sizes']
        latent_size = model_params['latent_size']
        num_levels = model_params['num_levels']
        codebook_size = model_params['codebook_size']
        dropout = model_params['dropout']

        # Parse training parameters from config
        train_params = config['rqvae_letter']['training_params']
        self.latent_loss_weight = train_params.get('latent_loss_weight', 0.25)
        self.diversity_loss_weight = train_params.get('diversity_loss_weight', 0.1)
        self.alignment_loss_weight = train_params.get('alignment_loss_weight', 0.1)
        self.semantic_loss_weight = train_params.get('semantic_loss_weight', 0.1)
        self.ranking_loss_weight = train_params.get('ranking_loss_weight', 0.1)
        
        self.loss_type = train_params.get('loss_type', 'mse')
        self.temperature = train_params.get('temperature', 1.0)
        
        # Build encoder and decoder
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        rev_hidden_sizes = hidden_sizes.copy()
        rev_hidden_sizes.reverse()
        self.decoder = MLP(latent_size, rev_hidden_sizes, input_size, dropout=dropout)
        
        # Build LETTER-enhanced quantizer
        code_shape = [codebook_size] * num_levels
        self.quantizer = LETTERRQBottleneck(
            latent_shape=latent_size,
            code_shape=code_shape,
            semantic_regularization=True,
            diversity_weight=self.diversity_loss_weight,
            temperature=self.temperature,
            cf_embedding_dim=cf_embedding_dim,
        )
        
        # Ranking-guided generation components
        self.ranking_head = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.ReLU(),
            nn.Linear(latent_size // 2, 1)
        )

    def forward(self, xs, cf_features=None, target_rankings=None):
        """Forward pass with LETTER enhancements."""
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e, cf_features)
        out = self.decode(z_q)
        
        # Compute ranking scores if needed
        ranking_scores = None
        if target_rankings is not None:
            ranking_scores = self.ranking_head(z_q).squeeze(-1)
        
        return out, quant_loss, code, ranking_scores

    def encode(self, x):
        """Encode input to latent space."""
        z_e = self.encoder(x)
        return z_e

    def decode(self, z_q):
        """Decode quantized latent to output."""
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        """Get quantized codes for input."""
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    def compute_ranking_loss(self, ranking_scores, target_rankings):
        """
        Compute ranking-guided generation loss.
        This encourages the model to generate items with appropriate ranking scores.
        """
        if ranking_scores is None or target_rankings is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Use ranking loss (e.g., pairwise ranking loss or MSE)
        ranking_loss = F.mse_loss(ranking_scores, target_rankings)
        return ranking_loss

    def compute_loss(self, forward_outputs, xs=None, cf_features=None, target_rankings=None, valid=False):
        """
        Compute comprehensive LETTER loss including all components.
        """
        out, quant_loss, code, ranking_scores = forward_outputs
        
        # Reconstruction loss
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        # Quantization loss (commitment loss)
        loss_latent = quant_loss

        # Diversity loss
        loss_diversity = self.quantizer.compute_diversity_loss()

        # Collaborative alignment loss
        loss_alignment = torch.tensor(0.0, device=out.device)
        if self.quantizer.collaborative_alignment is not None and cf_features is not None:
            z_e = self.encode(xs)
            loss_alignment = self.quantizer.collaborative_alignment(z_e, cf_features)

        # Semantic regularization loss (already included in quantizer)
        loss_semantic = torch.tensor(0.0, device=out.device)

        # Ranking-guided generation loss
        loss_ranking = self.compute_ranking_loss(ranking_scores, target_rankings)

        # Total loss
        loss_total = (
            loss_recon +
            self.latent_loss_weight * loss_latent +
            self.diversity_loss_weight * loss_diversity +
            self.alignment_loss_weight * loss_alignment +
            self.semantic_loss_weight * loss_semantic +
            self.ranking_loss_weight * loss_ranking
        )

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
            "loss_diversity": loss_diversity,
            "loss_alignment": loss_alignment,
            "loss_semantic": loss_semantic,
            "loss_ranking": loss_ranking,
        }

    def get_code_utilization(self):
        """Get code utilization statistics."""
        return self.quantizer.get_code_utilization()

    def get_diversity_metrics(self):
        """Get diversity metrics for analysis."""
        return {
            "code_utilization": self.get_code_utilization(),
            "diversity_loss": self.quantizer.compute_diversity_loss(),
        }