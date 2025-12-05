# File Path: /tokenlization_stage/models/r_kmeans.py (Corrected Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import logging
import time # For Debugging

# Assuming abstract_vq is in the same directory or accessible
from .abstract_vq import AbstractVQ

# --- Distance Functions (with added detach for safety) ---
def _pairwise_euclidean2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Calculates squared Euclidean distance between two sets of vectors. """
    # Ensure calculations don't track gradients unnecessarily
    with torch.no_grad():
        a_norm = a.pow(2).sum(-1, keepdim=True)
        # Detach b to prevent potential gradient issues if b comes from parameters
        b_detached = b.detach()
        b_norm = b_detached.pow(2).sum(-1)
        # Calculate squared distance: a^2 - 2ab + b^2
        dist = a_norm - 2 * a @ b_detached.t() + b_norm
    # Clamp after calculation to ensure non-negativity due to floating point errors
    return dist.clamp_min_(0.0)

def _pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Calculates cosine distance (1 - cosine similarity) """
    # Ensure calculations don't track gradients unnecessarily
    with torch.no_grad():
        # Normalize vectors before dot product
        a_norm = F.normalize(a, dim=-1)
        # Detach b
        b_norm = F.normalize(b.detach(), dim=-1)
        # Calculate cosine similarity and convert to distance
        sim = (a_norm @ b_norm.t()).clamp(-1.0, 1.0) # Clamp for numerical stability
        dist = 1.0 - sim
    return dist

class RKMEANS(AbstractVQ):
    """
    Residual K-Means (Corrected Version - Inspired by Grid):
    - Residual normalization DISABLED by default.
    - Uses stable EMA updates for centroids.
    - Optimized empty cluster reseeding.
    - Layer-wise training.
    """
    def __init__(self, config: dict, input_size: Optional[int] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- Read Config (compatible with different naming) ---
        node = config.get("rkmeans", config.get("r_kmeans", {}))
        mcfg = node.get("model_params", config.get("model_params", {}))
        tcfg = node.get("training_params", config.get("training_params", {}))

        # --- Core Model Params ---
        # Infer code_dim from input_size if provided, otherwise use config
        self.code_dim: int = int(input_size if input_size is not None else mcfg.get("code_dim", 512))
        self.num_levels: int = int(mcfg.get("num_levels", 3))
        self.codebook_size: int = int(mcfg.get("codebook_size", 256))
        self.logger.info(f"RKMeans Params: Levels={self.num_levels}, K={self.codebook_size}, Dim={self.code_dim}")

        # --- Crucial Configs (Defaults based on Grid/Best Practices) ---
        self.normalize_residuals: bool = bool(mcfg.get("normalize_residuals", False)) # DEFAULT FALSE
        metric_str: str = mcfg.get("distance", "cosine") # DEFAULT COSINE
        self.metric: Literal["euclidean", "cosine"] = metric_str if metric_str in ("euclidean", "cosine") else "cosine"
        self._dist_fn = _pairwise_euclidean2 if self.metric == "euclidean" else _pairwise_cosine
        if not self.normalize_residuals: self.logger.info("Residual normalization DISABLED (Recommended).")
        else: self.logger.warning("Residual normalization ENABLED. This might hurt downstream performance.")
        self.logger.info(f"Using distance metric: {self.metric}")

        # --- Training Configs ---
        self.train_layer_wise: bool = bool(mcfg.get("train_layer_wise", True))
        self.init_buffer_size: int = int(mcfg.get("init_buffer_size", 4096))
        # Force EMA based on best practices
        self.update_mode: Literal["ema"] = "ema"
        self.ema_decay: float = float(tcfg.get("ema_decay", 0.99)) # Stable default
        self.level_train_steps: int = int(tcfg.get("level_train_steps", 1000)) # More steps per level
        self.reseed_empty: bool = bool(tcfg.get("reseed_empty", True))
        # Ensure reseed_threshold is float
        self.reseed_threshold: float = float(tcfg.get("reseed_threshold", 1e-5))
        self.logger.info(f"Update mode: EMA, Decay: {self.ema_decay}, Level steps: {self.level_train_steps}, Reseed empty: {self.reseed_empty} (Threshold: {self.reseed_threshold:.2e})")

        # --- Codebooks & EMA Statistics ---
        # Codebooks are Parameters for potential gradient flow in recon_loss
        self.register_buffer("codebooks", torch.empty(self.num_levels, self.codebook_size, self.code_dim)) # <-- 改成 buffer
        nn.init.xavier_uniform_(self.codebooks) # Use Xavier/Glorot initialization
        # EMA stats are buffers (non-trainable)
        self.register_buffer("cluster_sums", torch.zeros_like(self.codebooks.data))
        self.register_buffer("cluster_counts", torch.zeros(self.num_levels, self.codebook_size))

        # --- Initialization State ---
        self.register_buffer("initialized_levels", torch.zeros(self.num_levels, dtype=torch.bool))
        self._init_buffer = []; self._init_buffer_count = 0
        self.current_level: int = 0; self.level_steps: int = 0
        # Track device to ensure consistency
        self._current_device: Optional[torch.device] = None

    @property
    def is_iterative(self) -> bool:
         """ RKMeans requires iterative training. """
         return True

    # --- Utility Methods ---
    @torch.no_grad()
    def _kmeanspp_init(self, level: int, data: torch.Tensor):
        """ KMeans++ Initialization (Robust Version). Assumes data is on target device. """
        N, D = data.shape; K = self.codebook_size
        target_device = self.codebooks.device
        C = torch.empty(K, D, device=target_device, dtype=self.codebooks.dtype)

        # 1. Choose first center randomly
        idx0 = torch.randint(0, N, (1,), device=data.device); C[0] = data[idx0]
        min_d2_sum = torch.tensor(float('inf'), device=target_device) # Track sum of squares for stability

        # 2. Iteratively choose remaining centers
        for k in range(1, K):
            # Calculate distance from data points to current centers
            dist_to_C = self._dist_fn(data, C[:k]) # data & C[:k] are on target_device
            min_d2 = dist_to_C.min(dim=1).values # Min distance squared for each point
            current_sum = min_d2.sum()

            # Stability check: if distances are zero or not decreasing significantly, fall back to random
            if current_sum <= 1e-9 or not torch.isfinite(current_sum) or \
               (min_d2_sum.isfinite() and (min_d2_sum - current_sum) < 1e-6 * min_d2_sum):
                 self.logger.warning(f"KMeans++ stability issue at k={k} for level {level}. Random init for remaining.");
                 # Use randperm for unique random indices
                 rand_idx = torch.randperm(N, device=data.device)[:K-k]
                 C[k:] = data[rand_idx] # Assign remaining centers randomly
                 break
            min_d2_sum = current_sum # Update sum for next stability check

            # Sample next centroid using probabilities proportional to squared distance
            probs = (min_d2 / current_sum).clamp_min_(0)
            probs /= probs.sum() + 1e-12 # Ensure probabilities sum to 1

            try:
                 # Multinomial sampling - run on CPU for stability if needed
                 nxt = torch.multinomial(probs.cpu(), 1).to(data.device) # Sample index
                 C[k] = data[nxt] # Assign new center
            except RuntimeError as e: # Catch potential numerical errors in multinomial
                 self.logger.warning(f"Multinomial sampling failed at k={k} for level {level}: {e}. Using random index.")
                 rand_idx = torch.randint(0, N, (1,), device=data.device)
                 C[k] = data[rand_idx]

        # 3. Update codebook parameter and reset EMA stats
        self.codebooks.data[level].copy_(C)
        self.initialized_levels[level] = True
        self.cluster_sums[level].zero_()
        self.cluster_counts[level].zero_()
        self.logger.info(f"Level {level} initialized via KMeans++.")

    @torch.no_grad()
    def _maybe_collect_init(self, residual: torch.Tensor):
        """ Collect data for initialization buffer (store on CPU). """
        if self.initialized_levels[self.current_level]: return
        need = self.init_buffer_size - self._init_buffer_count;
        if need <= 0: return
        take = min(need, residual.size(0));
        # Store detached tensor on CPU to save GPU memory
        self._init_buffer.append(residual[:take].detach().clone().cpu());
        self._init_buffer_count += take
        # self.logger.debug(f"Collected init buffer: {self._init_buffer_count}/{self.init_buffer_size}") # Optional debug log

    @torch.no_grad()
    def _layer_init_if_needed(self, residual: torch.Tensor):
        """ Initialize the current layer using KMeans++ if buffer is full. """
        l = self.current_level
        if l >= self.num_levels:
            # self.logger.debug(f"_layer_init_if_needed called with l={l}, but num_levels={self.num_levels}. Returning.") # 可选的调试日志
            return
        # Skip if already initialized or training is finished
        if self.initialized_levels[l] or l >= self.num_levels: return
        # Collect data into buffer
        self._maybe_collect_init(residual)
        # If buffer is full, perform initialization
        if self._init_buffer_count >= self.init_buffer_size:
            # Concatenate CPU buffer and move to target device
            buf = torch.cat(self._init_buffer, dim=0)[: self.init_buffer_size].to(self.codebooks.device)
            self.logger.info(f"Initializing level {l} with KMeans++ (buffer size {buf.shape[0]})...")
            # Call KMeans++ init
            self._kmeanspp_init(l, buf)
            # Clear buffer after use
            self._init_buffer.clear(); self._init_buffer_count = 0

    @torch.no_grad()
    def _ema_update(self, level: int, batch: torch.Tensor, assign_idx: torch.Tensor):
        """ EMA update for centroids, inspired by grid """
        K = self.codebook_size
        target_device = self.codebooks.device # Use codebook's device

        # Ensure inputs are on the target device
        batch = batch.to(target_device)
        assign_idx = assign_idx.to(target_device)

        # --- Calculate Batch Statistics ---
        one_hot = F.one_hot(assign_idx, num_classes=K).to(batch.dtype) # (B, K)
        batch_counts = one_hot.sum(dim=0)                                    # (K,) Sum of assignments per cluster
        batch_sums = one_hot.t() @ batch                                     # (K, D) Sum of vectors per cluster

        # --- EMA Update Global Statistics (In-place) ---
        decay = self.ema_decay
        # Update sums: sums = decay * sums + (1 - decay) * batch_sums
        self.cluster_sums[level].mul_(decay).add_(batch_sums, alpha=1 - decay)
        # Update counts: counts = decay * counts + (1 - decay) * batch_counts
        self.cluster_counts[level].mul_(decay).add_(batch_counts, alpha=1 - decay)

        # --- Calculate New Centroids ---
        # Get EMA counts, add small epsilon for stability
        normalized_counts = self.cluster_counts[level].unsqueeze(1).clamp_min_(1e-10)
        # Calculate centroids: sums / counts
        new_centroids = self.cluster_sums[level] / normalized_counts

        # Update codebook parameter data using copy_
        self.codebooks[level].copy_(new_centroids)

        # --- Reseed Near-Empty Clusters ---
        # Reseed only after some initial steps to allow counts to stabilize
        if self.reseed_empty and self.level_steps > K * 2:
            # Identify clusters with EMA count below threshold
            near_empty_mask = self.cluster_counts[level] < self.reseed_threshold
            if near_empty_mask.any():
                num_empty = int(near_empty_mask.sum())
                self.logger.warning(f"Level {level}: Reseeding {num_empty} near-empty clusters (count < {self.reseed_threshold:.2e})...")

                # Reseed with random points from the current batch
                # Ensure random indices are valid and on the correct device
                ridx = torch.randint(0, batch.size(0), (num_empty,), device=batch.device)
                reseed_vectors = batch[ridx]

                # Update codebook centroids
                self.codebooks.data[level][near_empty_mask] = reseed_vectors
                # Reset EMA stats for reseeded clusters to a small initial value (like one sample)
                # alpha = 1 - decay helps initialize the EMA correctly
                self.cluster_sums.data[level][near_empty_mask] = reseed_vectors * (1 - decay)
                self.cluster_counts.data[level][near_empty_mask] = (1 - decay)

    # --- Forward Pass ---
    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs residual quantization. Updates centroids during training.
        Returns: (reconstruction, reconstruction_loss_mse, codes)
        """
        B, D = xs.shape
        if D != self.code_dim:
            raise ValueError(f"Input dimension {D} != model code_dim {self.code_dim}")

        # Ensure model parameters & buffers are on the same device as input
        current_device = xs.device
        if self._current_device is None or self._current_device != current_device:
             # This moves the nn.Parameter and registered buffers
             self.to(current_device)
             self._current_device = current_device
             self.logger.info(f"Moved RKMeans model to device: {current_device}")

        # --- Prepare Initial Residual (NO normalization by default) ---
        if self.normalize_residuals: # Should be False by default
            mean = xs.mean(dim=0, keepdim=True); std = xs.std(dim=0, keepdim=True).clamp_min_(1e-6)
            original_input_for_loss = (xs - mean) / std # Loss calculated on normalized input
        else:
            original_input_for_loss = xs # Loss calculated on original input

        current_residual = original_input_for_loss.clone() # Residual calculation starts here

        # Initialize outputs
        x_recon = torch.zeros_like(xs)
        codes = torch.zeros(B, self.num_levels, dtype=torch.long, device=current_device)

        # Determine active levels based on layer-wise training state
        active_levels_range = range(self.num_levels) if not self.train_layer_wise else range(self.current_level + 1)

        for l in active_levels_range:
            # --- Initialize layer if needed (during training, only for current level) ---
            if self.training and self.train_layer_wise and l == self.current_level:
                # Pass residual detached, init doesn't need grads
                self._layer_init_if_needed(current_residual.detach())
            
            # Skip quantization if layer is not initialized
            if l >= self.num_levels or not self.initialized_levels[l]:
                 # If training, still collect buffer for the layer to be initialized
                #  if self.training and self.train_layer_wise and l == self.current_level:
                #      self._maybe_collect_init(current_residual.detach())
                 continue # Move to next layer

            # --- Quantization Step ---
            # Get centroids detached for distance calculation (no gradient needed here)
            C = self.codebooks.data[l].detach()
            # Calculate distances using detached residual (no gradient needed here)
            dist = self._dist_fn(current_residual.detach(), C)
            idx = dist.argmin(dim=1) # Hard assignment indices

            # Get quantized vector using Parameter (connects gradient for recon_loss)
            q = F.embedding(idx, self.codebooks[l])

            # --- Update Reconstruction and Residual ---
            x_recon = x_recon + q # Accumulate reconstruction
            # Store residual *before* subtracting q, needed for EMA update
            residual_before_quant = current_residual.clone()
            # Update residual for *next* layer's input, detaching q
            current_residual = current_residual - q.detach()
            # Store codes
            codes[:, l] = idx

            # --- Update Centroids (EMA, only if training current layer) ---
            if self.training and self.train_layer_wise and l == self.current_level:
                # Update based on residual *before* quantization for level l
                self._ema_update(l, residual_before_quant, idx)
                self.level_steps += 1 # Increment steps for current level

                # Check if current level training is complete
                if self.level_steps >= self.level_train_steps and self.current_level < self.num_levels:
                    self.logger.info(f"Level {self.current_level} trained for {self.level_steps} steps. Moving to level {self.current_level + 1}.")
                    self.current_level += 1 # Move to next level
                    self.level_steps = 0    # Reset step counter
                    # If there are more levels, clear init buffer for the next level
                    if self.current_level < self.num_levels:
                         self._init_buffer.clear(); self._init_buffer_count = 0
                         # Try to collect for next layer immediately using current residual
                         # Use clone().detach() to avoid modifying current_residual inplace issues
                         self._maybe_collect_init(current_residual.clone().detach())
                    else:
                         self.logger.info("All levels trained.")
                         # Training finished for all layers

        # Calculate reconstruction loss based on the input used for the first residual
        recon_loss = F.mse_loss(x_recon, original_input_for_loss)

        # Return tuple: (reconstruction, loss, codes)
        return x_recon, recon_loss, codes

    # --- Inference Method ---
    @torch.no_grad()
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        """ Encodes input vectors into discrete codes using all trained levels. """
        self.eval() # Ensure model is in eval mode
        B, D = xs.shape; device = xs.device
        if D != self.code_dim: raise ValueError(f"Input dim {D} mismatch in get_codes")

        # Ensure model parameters & buffers are on the same device as input
        if self._current_device is None or self._current_device != device:
             self.to(device)
             self._current_device = device
             # self.logger.debug(f"Moved RKMeans model to device: {device} during get_codes")

        # --- Prepare Initial Residual (Consistent with forward, default NO normalization) ---
        if self.normalize_residuals:
             # Using batch stats during inference - potential mismatch with training if stats differ
             mean=xs.mean(dim=0, keepdim=True); std=xs.std(dim=0, keepdim=True).clamp_min_(1e-6)
             residual = (xs - mean) / std
        else:
             residual = xs.clone() # Use original input

        codes = torch.zeros(B, self.num_levels, dtype=torch.long, device=device)
        for l in range(self.num_levels):
            # Skip uninitialized levels, codes remain 0
            if not self.initialized_levels[l]: continue
            # Use codebook data directly for inference (no gradients needed)
            C = self.codebooks.data[l]
            # Calculate distance and find nearest centroid
            dist = self._dist_fn(residual, C)
            idx = dist.argmin(dim=1)
            codes[:, l] = idx
            # Update residual using codebook data (F.embedding works with .data)
            q = F.embedding(idx, C)
            residual = residual - q # Update residual for the next level
            
        return codes

    # --- Loss Computation for Trainer ---
    # --- Loss 計算 ---
    def compute_loss(self, forward_outputs, batch_data=None) -> dict:
        """ 
        返回重構 Loss。
        【已修正】返回 Tensor，而不是 float，以便 Trainer 進行反向傳播或日誌記錄。
        """
        _, recon_loss, _ = forward_outputs # recon_loss 是一個 Tensor
        
        # ✅ 關鍵修正：直接返回 Tensor
        return { 
            "loss_total": recon_loss, # 返回原始的 loss Tensor
            "loss_recon": recon_loss  # 返回原始的 loss Tensor
        }