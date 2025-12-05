# 檔案路徑: tokenlization_stage/models/pq.py

import torch
from torch import nn
import numpy as np
import faiss
import os
import json
import logging
from sklearn.model_selection import train_test_split
import math
import time

from .abstract_vq import AbstractVQ

class PQ(AbstractVQ):
    """
    Product Quantization (PQ) 量化器 (已修正)。
    """
    def __init__(self, config: dict, input_size: int):
        super().__init__(config)
        self.config = config
        self.input_size = input_size
        self.logger = logging.getLogger(self.__class__.__name__) # 使用 logger

        # 从 config 的 'pq' 節點讀取參數
        node = config.get("pq", {}) # 使用 .get() 更安全
        model_params = node.get("model_params", {})
        train_params = node.get("training_params", {}) # 读取训练参数

        self.num_levels = model_params.get('num_levels')
        self.codebook_size = model_params.get('codebook_size')
        if self.num_levels is None or self.codebook_size is None:
             raise ValueError("PQ config 的 model_params 中必须包含 'num_levels' 和 'codebook_size'")
             
        # 添加 max_train_samples (与 OPQ 保持一致)
        self.max_train_samples = train_params.get("max_train_samples", 256 * 1000) 

        if 'faiss_omp_num_threads' in model_params:
            faiss.omp_set_num_threads(model_params['faiss_omp_num_threads'])
            self.logger.info(f"Faiss OMP 執行緒數設置為: {model_params['faiss_omp_num_threads']}")

        # 初始化內部狀態
        self.fitted = False
        self.pq: Optional[faiss.ProductQuantizer] = None # 类型提示
        self._embedding_buffer = []
        # _total_item_count 从 config 读取 (与 OPQ 一致)
        self._total_item_count = config.get('total_item_count', -1) 
        if self._total_item_count <= 0:
             raise ValueError("PQ 模型需要 'total_item_count' > 0 在 config 中被設置。")

        self.logger.info(f"PQ 初始化: Levels(M)={self.num_levels}, CodebookSize(K)={self.codebook_size}, InputDim={self.input_size}")
        self.logger.info("模型狀態: 未擬合 (等待數據)。")


    @property
    def is_iterative(self) -> bool:
        return False

    def _fit_faiss(self):
        """
        在數據攢齊後，執行一次性的 Faiss 擬合來訓練 ProductQuantizer 碼本。
        ✅ 使用 centroids_per_subquantizer 参数。
        """
        if self.fitted: return
        if not self._embedding_buffer:
             self.logger.error("拟合错误：Embedding 缓冲区为空。")
             return

        self.logger.info("PQ 数据已攒齐，开始一次性 Faiss 拟合...")
        start_time = time.time()
        
        embeddings_np = torch.cat(self._embedding_buffer, dim=0).cpu().numpy().astype('float32')
        self._embedding_buffer = [] 

        # 使用 max_train_samples 采样 (与 OPQ 一致)
        train_size = min(len(embeddings_np), self.max_train_samples) 
        if train_size < len(embeddings_np):
            train_indices = np.random.choice(len(embeddings_np), size=train_size, replace=False)
            train_vectors = embeddings_np[train_indices]
            self.logger.info(f"从 {len(embeddings_np)} 个向量中随机采样 {train_size} 个用于训练。")
        else:
            train_vectors = embeddings_np
            self.logger.info(f"使用全部 {len(embeddings_np)} 个向量进行训练。")
            
        if train_vectors.shape[0] == 0:
             self.logger.error("拟合错误：没有有效的训练向量。")
             return

        # ==================== 核心修改 ====================
        try:
            # 1. 计算每个子向量的维度 dsub
            d = self.input_size
            M = self.num_levels
            if d % M != 0:
                # Faiss PQ 要求输入维度必须能被 num_levels 整除
                # 如果不能整除，需要给出错误提示或进行处理 (如 PCA 降维)
                raise ValueError(f"输入维度 ({d}) 无法被 num_levels ({M}) 整除。Faiss PQ 要求 d % M == 0。")
            dsub = d // M

            # 2. 计算每个 codebook 需要多少 bit (nbits)
            # Faiss 要求每个 codebook 大小 <= 2^16 (65536)
            # 并且 nbits <= 16
            K = self.codebook_size
            if K > 65536:
                 raise ValueError(f"Codebook size ({K}) 不能超过 65536。")
                 
            # 找到最小的 nbits 使得 2^nbits >= K
            nbits = math.ceil(math.log2(K)) 
            if nbits > 16: # Faiss PQ 的限制
                 raise ValueError(f"计算出的 nbits ({nbits}) 超过 Faiss PQ 的最大限制 16。请减小 codebook_size。")
                 
            # 3. 使用正确的参数创建 ProductQuantizer
            # faiss.ProductQuantizer(d, M, nbits)
            # d: 总维度
            # M: 子空间数量 (num_levels)
            # nbits: 每个子 codebook 的比特数
            self.pq = faiss.ProductQuantizer(d, M, int(nbits)) 

            self.logger.info(f"创建 ProductQuantizer: d={d}, M={M}, nbits={int(nbits)} (对应 codebook_size={K})")
            
            # 4. 训练码本 (不变)
            self.logger.info(f"使用 {len(train_vectors)} 个向量训练 PQ 码本...")
            self.pq.train(train_vectors)
            
            self.fitted = True
            self.logger.info(f"PQ 碼本擬合完成，總耗時 {time.time() - start_time:.2f} 秒。")

        except Exception as e:
            self.logger.error(f"Faiss PQ 拟合过程中发生错误: {e}")
            self.fitted = False
            self.pq = None
            raise
        # ================= 修改结束 =================

    # --- forward, compute_loss, get_codes 保持不变 ---
    def forward(self, batch_data: torch.Tensor) -> tuple:
        """(保持不变)"""
        if self.fitted: return (None, None, None)
        self._embedding_buffer.append(batch_data.detach().cpu())
        if self._total_item_count is None: # 延迟读取 total_item_count
             self._total_item_count = self.config.get('total_item_count', -1)
             if self._total_item_count <= 0: raise ValueError("PQ total_item_count missing in config")
        current_count = sum(len(b) for b in self._embedding_buffer)
        if current_count >= self._total_item_count:
            try: self._fit_faiss()
            except Exception as e: self.logger.error(f"拟合在 forward 触发时失败: {e}")
        return (None, None, None)

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """(保持不变)"""
        # 返回 torch tensor 以适配 trainer
        return {'loss_total': torch.tensor(0.0, device=batch_data.device)} 

    @torch.no_grad()
    def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor:
        """(保持不变)"""
        if not self.fitted:
            self.logger.warning("PQ 在 get_codes 时仍未拟合，将执行紧急拟合。")
            try: self._fit_faiss()
            except Exception as e: raise RuntimeError(f"PQ 緊急擬合失敗: {e}")
            if not self.fitted: raise RuntimeError("PQ 緊急擬合後 fitted 仍为 False。")
            
        if self.pq is None: # 添加检查
             raise RuntimeError("PQ 对象为空，无法生成 codes。")
             
        batch_np = batch_data.cpu().numpy().astype('float32')
        try:
             codes_np = self.pq.compute_codes(batch_np)
             # 返回 int64 以匹配 trainer 中 build_dedup_layer 的期望
             return torch.from_numpy(codes_np.astype(np.int64)).long().to(batch_data.device) 
        except Exception as e:
             self.logger.error(f"PQ get_codes 失败: {e}")
             # 返回错误标记，例如全 -1
             error_codes = torch.full((batch_data.shape[0], self.num_levels), -1, 
                                      dtype=torch.long, device=batch_data.device)
             return error_codes