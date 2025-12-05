# /tokenlization_stage/eval_metrics.py

import torch
import torch.nn.functional as F

def calculate_validation_metrics(model_outputs, original_data):
    """
    计算验证集上的指标。
    """
    reconstructed_data, latent_loss, _ = model_outputs
    
    # 1. 重构损失 (MSE)
    reconstruction_mse = F.mse_loss(reconstructed_data, original_data, reduction="mean")
    
    # 2. 完整的总损失
    loss_dict = model.compute_loss(model_outputs, original_data)
    total_loss = loss_dict['loss_total']
    
    return {
        'val_total_loss': total_loss.item(),
        'val_mse': reconstruction_mse.item(),
        'val_latent_loss': latent_loss.item()
    }