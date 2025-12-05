# metrics.py

import torch

def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    根据命中索引计算 Recall@k。
    
    Args:
        pos_index (torch.Tensor): 形状为 (B, maxk) 的布尔张量，True 表示命中。
        k (int): top-k 的 k 值。

    Returns:
        torch.Tensor: 形状为 (B,) 的每个样本的 Recall@k 值。
    """
    return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    根据命中索引计算 NDCG@k。
    假设每个样本只有一个正例。
    
    Args:
        pos_index (torch.Tensor): 形状为 (B, maxk) 的布尔张量，True 表示命中。
        k (int): top-k 的 k 值。

    Returns:
        torch.Tensor: 形状为 (B,) 的每个样本的 NDCG@k 值。
    """
    # 排名从 1 开始
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device, dtype=torch.float)
    
    # 计算 DCG，只有命中位置有值
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, device=dcg.device))
    
    # IDCG@k 在此场景下恒为 1.0（因为只有一个正例，且理想排名是1）
    # 所以 NDCG@k == DCG@k
    return dcg[:, :k].sum(dim=1).cpu().float()