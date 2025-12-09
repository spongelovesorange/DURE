import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging
from .decoder.GPT2 import GPT2
from .decoder.T5 import T5

logger = logging.getLogger(__name__)

class DUREAdapter(nn.Module):
    def __init__(self, input_dim, rank=16):
        super().__init__()
        # Initialize low-rank matrices A (down-projection) and B (up-projection)
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        self.activation = nn.ReLU()
        # Initialize B to zero to ensure no impact initially
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)

    def forward(self, x):
        return self.lora_B(self.activation(self.lora_A(x)))

class DURELayerWrapper(nn.Module):
    def __init__(self, frozen_layer, adapter):
        super().__init__()
        self.frozen_layer = frozen_layer # Original Transformer Layer (MLP)
        self.adapter = adapter           # Side Memory

    def forward(self, x, routing_mask=None):
        # 1. Main model path (frozen)
        with torch.no_grad():
            main_out = self.frozen_layer(x)
        
        # 2. Side memory path (trainable)
        side_out = self.adapter(x)
        
        # 3. Dynamic routing fusion
        # Check if routing_mask is passed or set as attribute
        mask = routing_mask if routing_mask is not None else getattr(self, 'routing_mask', None)
        
        # DEBUG PRINT
        if not hasattr(self, 'debug_printed'):
             print(f"DEBUG: DURELayerWrapper mask: {mask}")
             self.debug_printed = True
        
        if mask is not None:
            # routing_mask should be broadcastable to x
            # x shape: (batch, seq_len, hidden)
            # mask shape: (batch, 1, 1) or (batch, seq_len, 1)
            # Ensure mask is on the same device
            if isinstance(mask, torch.Tensor):
                mask = mask.to(x.device)
                # Expand dimensions if needed (e.g. from (B,) to (B, 1, 1))
                if mask.dim() == 1:
                    mask = mask.view(-1, 1, 1)
            return main_out + (side_out * mask)
        else:
            # Default behavior if no mask: assume 0 (Main Memory only) or 1?
            # DURE says: "Clean Queries directly penetrate main model".
            # So default should be 0 (Main only) if we are not sure.
            # But during training of Side Memory, we want it active.
            # Let's assume if no mask is set, we add it (lambda=1) or 0?
            # Safest is 0 if we want to preserve original behavior.
            # But if we are training, we set the mask.
            return main_out # Default to frozen model only

class SemanticAwareRouter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        # We might pool or take the last token representation
        # For simplicity, let's take the mean or last token
        # Here we assume last token represents the context
        context = hidden_states[:, -1, :] # (batch, hidden)
        score = self.sigmoid(self.classifier(context)) # (batch, 1)
        return score

class DUREGPT2(GPT2):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        
        self.dure_config = config.get('dure_params', {})
        self.rank = self.dure_config.get('rank', 16)
        self.router_threshold = self.dure_config.get('router_threshold', 0.5)
        
        # Freeze the main model
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        logger.info("Main GPT2 model frozen.")

        # Inject Adapters
        # GPT2 structure: self.gpt2.transformer.h is a ModuleList of GPT2Block
        # Each block has .mlp
        
        self.adapters = nn.ModuleList()
        
        # We only inject into the last few layers as suggested
        num_layers = len(self.gpt2.transformer.h)
        inject_layers = self.dure_config.get('inject_layers', [num_layers-1, num_layers-2, num_layers-3])
        
        self.inject_layer_indices = set(inject_layers)
        
        # We need to access the hidden size. 
        # GPT2Config has n_embd
        hidden_size = self.gpt2.config.n_embd
        
        for i, block in enumerate(self.gpt2.transformer.h):
            if i in self.inject_layer_indices:
                logger.info(f"Injecting DURE Adapter into layer {i}")
                adapter = DUREAdapter(hidden_size, self.rank)
                self.adapters.append(adapter) # Keep reference to register parameters
                
                # Wrap the MLP
                original_mlp = block.mlp
                block.mlp = DURELayerWrapper(original_mlp, adapter)
            else:
                self.adapters.append(None) # Placeholder

        # Router
        self.router = SemanticAwareRouter(hidden_size)
        
    def set_routing_mask(self, mask):
        """
        Set the routing mask for all injected adapters.
        mask: Tensor or None
        """
        for i, block in enumerate(self.gpt2.transformer.h):
            if i in self.inject_layer_indices and isinstance(block.mlp, DURELayerWrapper):
                block.mlp.routing_mask = mask

    def forward(self, batch: Dict, return_router_score=False) -> Dict:
        # We need to intercept the forward pass to handle routing mask
        # However, GPT2LMHeadModel doesn't easily allow passing extra args to internal blocks
        # without modifying the model class or using hooks.
        
        # A simpler approach for the Router in this architecture:
        # The Router determines if the Side Memory should be active for the *entire* sequence or batch.
        # But the Wrapper is deep inside.
        
        # We can use a global state or pass it via input_ids (hacky).
        # Or, we can compute the router score first, and then set a flag in the wrappers.
        
        # Let's compute router score using the embeddings or initial hidden states?
        # No, router needs context.
        
        # Ideally, we run the model. But we modified the blocks.
        # The blocks now expect 'routing_mask'.
        # But HF model forward won't pass 'routing_mask' to blocks.
        
        # Solution: We can use a thread-local context or a temporary attribute on the model
        # to store the routing mask before calling forward.
        
        input_ids = batch['input_ids']
        
        # 1. Compute Router Score
        # We need a representation. Let's use the embeddings of the input_ids.
        # This is an approximation. The paper says "Attention(H_user, Emb_dirty)".
        # If we are in training, we know if it's dirty or not (labels).
        # If inference, we need to compute it.
        
        # For now, let's assume we pass a 'routing_mask' in the batch if we know it (Training),
        # or we compute it.
        
        routing_mask = batch.get('routing_mask', None)
        
        if routing_mask is None:
             # Compute using Router (Implementation TBD properly, for now all 1s or 0s)
             # We need to run a partial forward to get context? 
             # Or just use the embedding of the last item?
             pass

        # Set routing mask on all wrappers
        self.set_routing_mask(routing_mask)

        # Call original forward
        # The Wrapper's forward needs to read self.routing_mask
        
        outputs = super().forward(batch)
        
        # Clean up
        self.set_routing_mask(None)
                 
        return outputs

class DURET5(T5):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        
        self.dure_config = config.get('dure_params', {})
        self.rank = self.dure_config.get('rank', 16)
        self.router_threshold = self.dure_config.get('router_threshold', 0.5)
        
        # Freeze Main Model
        for param in self.t5.parameters():
            param.requires_grad = False
            
        logger.info("Main T5 model frozen.")

        # Inject Adapters into Decoder
        # T5 structure: self.t5.decoder.block is ModuleList
        # Each block has .layer (ModuleList)
        # layer[2] is T5LayerFF, which has .DenseReluDense
        
        num_layers = len(self.t5.decoder.block)
        inject_layers = self.dure_config.get('inject_layers', [num_layers-1, num_layers-2, num_layers-3])
        
        self.inject_layer_indices = set(inject_layers)
        
        # T5 hidden size
        hidden_size = self.t5.config.d_model
        
        self.adapters = nn.ModuleList()
        
        for i, block in enumerate(self.t5.decoder.block):
            if i in self.inject_layer_indices:
                logger.info(f"Injecting DURE Adapter into T5 Decoder layer {i}")
                adapter = DUREAdapter(hidden_size, self.rank)
                self.adapters.append(adapter)
                
                # Wrap the MLP
                # Note: T5LayerFF structure might vary, but usually it has DenseReluDense
                if hasattr(block.layer[2], 'DenseReluDense'):
                    original_mlp = block.layer[2].DenseReluDense
                    block.layer[2].DenseReluDense = DURELayerWrapper(original_mlp, adapter)
                else:
                    logger.warning(f"Could not find DenseReluDense in T5 Decoder layer {i}")
            else:
                self.adapters.append(None)

        # Router
        self.router = SemanticAwareRouter(hidden_size)
        
    def set_routing_mask(self, mask):
        for i, block in enumerate(self.t5.decoder.block):
            if i in self.inject_layer_indices:
                if hasattr(block.layer[2], 'DenseReluDense') and isinstance(block.layer[2].DenseReluDense, DURELayerWrapper):
                    block.layer[2].DenseReluDense.routing_mask = mask

    def forward(self, batch: Dict, return_router_score=False) -> Dict:
        # Similar to DUREGPT2, we set the mask and call super().forward()
        
        routing_mask = batch.get('routing_mask', None)
        
        # If routing_mask is None, we might want to compute it using Router
        # For now, we rely on external injection or default
        
        self.set_routing_mask(routing_mask)
        
        outputs = super().forward(batch)
        
        self.set_routing_mask(None)
        
        return outputs
