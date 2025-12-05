# /tokenlization_stage/models/abstract_vq.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractVQ(nn.Module, ABC):
    """
    所有向量量化模型的抽象基类 (父类)。
    定义了所有子类模型都必须实现的通用接口。
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, xs: torch.Tensor):
        """
        模型的前向传播，必须返回重构的向量、量化损失和离散码。
        """
        pass

    @abstractmethod
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        """
        仅用于预测（生成码本），返回输入的离散码。
        """
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> dict:
        """
        计算损失函数，必须返回一个包含 'loss_total'键的字典。
        """
        pass

    @property
    def is_iterative(self) -> bool:
        """
        模型自我聲明其訓練範式。
        - True (預設): 需要迭代訓練 (e.g., VQ-VAE)。
        - False: 只需要一次性擬合 (e.g., OPQ, PQ)。
        """
        return True # 預設所有模型都需要迭代訓練