# abstract_model.py (升级版)

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
import torch.nn as nn

class AbstractModel(nn.Module, ABC):
    """
    生成式推荐模型的抽象基类。

    它定义了一个所有模型都必须遵守的接口契约：
    1.  `__init__`: 接受一个配置字典。
    2.  `forward`: 执行前向传播，用于训练，必须返回包含 `loss` 的输出。
    3.  `evaluate_step`: 执行完整的单批次评估，返回一个指标字典。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def task_type(self) -> str:
        """返回模型的任務類型，例如 'generative' 或 'retrieval'。"""
        raise NotImplementedError

    @property
    def n_parameters(self) -> str:
        """计算模型的可训练参数数量。"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params:,}'

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        """
        【必须实现】执行前向传播以进行训练。
        
        Args:
            batch (Dict[str, torch.Tensor]): 包含 `input_ids`, `attention_mask`, `labels` 等的字典。

        Returns:
            Any: 通常是 Hugging Face 模型的输出对象，其中必须包含 `loss` 属性。
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        【必须实现】执行一个完整的评估步骤（生成 -> 后处理 -> 计算指标）。
        
        这个方法将模型专属的评估逻辑完全封装起来，供通用的 Trainer 调用。

        Args:
            batch (Dict[str, torch.Tensor]): 当前评估批次的数据。
            topk_list (List[int]): 需要计算指标的 top-k 列表 (e.g., [10, 20])。

        Returns:
            Dict[str, float]: 一个包含该批次所有指标平均值的字典, e.g.,
                              {'Recall@10': 0.5, 'NDCG@10': 0.4, ...}
        """
        raise NotImplementedError

    # generate 方法可以不是抽象的，因为 evaluate_step 内部会调用它。
    # 我们仍然可以保留它，以便在需要时单独调用。
    def generate(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError('The underlying model must have a generate method.')