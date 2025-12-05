# 文件路徑: /quantization/models/opq.py (仅修改此文件版本)

import torch
from torch import nn
import numpy as np
import faiss
import os
import json
import logging
import math
import time
import pickle # 用於保存/加載 Faiss 對象狀態
from typing import Optional

# 假設 abstract_vq 在父目錄或可導入
try:
    from .abstract_vq import AbstractVQ
except ImportError:
    # Fallback if run directly or relative import fails
    class AbstractVQ(nn.Module):
        def __init__(self, config: dict): super().__init__(); self.config = config
        @property
        def is_iterative(self) -> bool: raise NotImplementedError
        def forward(self, batch_data: torch.Tensor): raise NotImplementedError
        def compute_loss(self, forward_outputs, batch_data) -> dict: raise NotImplementedError
        def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor: raise NotImplementedError

class OPQ(AbstractVQ):
    """
    Optimized Product Quantization (OPQ) 量化器，基於 Faiss 實現。
    採用「一次性擬合」策略，可保存/加載狀態以避免重複訓練。
    此版本仅修改 opq.py，通过读写独立的 .pkl 状态文件来解决 main.py 中 load_state_dict 无法恢复 Faiss 对象的问题。
    """
    def __init__(self, config: dict, input_size: int):
        super().__init__(config)
        self.config = config
        self.input_size = input_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- 讀取配置 ---
        node = config.get("opq", {})
        model_params = node.get("model_params", {})
        train_params = node.get("training_params", {})

        self.num_levels = model_params.get('num_levels', 8)
        self.codebook_size = model_params.get('codebook_size', 256)
        self.n_codebook_bits = int(math.log2(self.codebook_size))
        if 2**self.n_codebook_bits != self.codebook_size:
            raise ValueError(f"OPQ 的 codebook_size ({self.codebook_size}) 必須是 2 的冪。")

        if 'faiss_omp_num_threads' in model_params:
            faiss.omp_set_num_threads(model_params['faiss_omp_num_threads'])
            self.logger.info(f"Faiss OMP 執行緒數設置為: {model_params['faiss_omp_num_threads']}")
        self.max_train_samples = train_params.get("max_train_samples", 256 * 1000)

        # --- 內部狀態 ---
        self.fitted = False
        self.opq_matrix: Optional[faiss.OPQMatrix] = None
        self.pq: Optional[faiss.ProductQuantizer] = None
        self._embedding_buffer = []
        self._total_item_count = config.get('total_item_count', -1)
        if self._total_item_count <= 0:
             raise ValueError("OPQ 模型需要 'total_item_count' > 0 在 config 中被設置。")

        # === ✨ 推斷狀態文件路徑 ===
        # 假設 .pkl 文件與 .pth 文件在同一目錄，文件名類似
        # config['save_path'] 應該是 .pth 文件的完整路徑
        save_path_pth = config.get('save_path', None)
        if save_path_pth:
            self.state_path = save_path_pth.replace('_best.pth', '_fitted_state.pkl').replace('_fitted.pth', '_fitted_state.pkl')
            self.logger.info(f"推斷的 OPQ 狀態文件路徑: {self.state_path}")
        else:
            # 如果 config 沒有 save_path，則無法保存/加載狀態
            self.logger.warning("Config 中缺少 'save_path'，無法推斷狀態文件路徑，每次運行都將重新擬合。")
            self.state_path = None
        # === 推斷結束 ===


        self.logger.info(f"OPQ 初始化: Levels(M)={self.num_levels}, CodebookSize(K)={self.codebook_size}, Bits={self.n_codebook_bits}, InputDim={self.input_size}")
        # 嘗試在初始化時就加載狀態
        if self.state_path:
            self.load_fitted_state(self.state_path)
        if not self.fitted:
            self.logger.info("模型狀態: 未擬合 (等待數據)。")


    @property
    def is_iterative(self) -> bool: return False # 告訴 Trainer 這是一次性擬合

    # === 狀態保存與加載 (与之前版本一致) ===
    def save_fitted_state(self, path: str):
        """保存訓練好的 OPQ 狀態到 .pkl 文件。"""
        if not self.fitted or self.opq_matrix is None or self.pq is None:
            self.logger.error("OPQ 模型尚未成功擬合，無法保存狀態。")
            return
        state = { 'opq_matrix': self.opq_matrix, 'pq': self.pq, 'fitted': self.fitted }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'wb') as f: pickle.dump(state, f)
            self.logger.info(f"OPQ 狀態已保存到: {path}")
        except Exception as e: self.logger.error(f"保存 OPQ 狀態到 {path} 失敗: {e}")

    def load_fitted_state(self, path: str) -> bool:
        """從 .pkl 文件加載 OPQ 狀態。"""
        if not os.path.exists(path):
            # self.logger.info(f"未找到 OPQ 狀態文件: {path}。模型需要擬合。") # 减少日志干扰
            return False
        try:
            with open(path, 'rb') as f: state = pickle.load(f)
            if not isinstance(state.get('opq_matrix'), faiss.OPQMatrix): raise TypeError("opq_matrix type mismatch")
            if not isinstance(state.get('pq'), faiss.ProductQuantizer): raise TypeError("pq type mismatch")

            self.opq_matrix = state['opq_matrix']
            self.pq = state['pq']
            self.fitted = state.get('fitted', False)

            if self.fitted: self.logger.info(f"成功从 {path} 加载 OPQ 状态。")
            # (省略参数验证日志，保持简洁)
            return self.fitted
        except Exception as e:
            self.logger.error(f"加载 OPQ 状态失败 ({path}): {e}。模型需要重新拟合。")
            self.fitted = False; self.opq_matrix = None; self.pq = None
            return False

    # === Faiss 擬合邏輯 (只在需要時執行) ===
    def _fit_faiss(self):
        """在數據攢齊後，执行一次性的 Faiss 拟合，并保存状态（CPU 训练，避免 GpuIndexPQ 不存在的问题）。"""
        if self.fitted:
            return  # 已拟合/已加载，直接返回

        if not self._embedding_buffer:
            self.logger.error("拟合错误：Embedding 缓冲区为空。")
            return

        logging.info("OPQ 数据已攒齐，开始一次性 Faiss 拟合（CPU PQ 训练）...")
        total_start_time = time.time()

        # 收集并转换为 float32 numpy
        embeddings_np = torch.cat(self._embedding_buffer, dim=0).cpu().numpy().astype('float32')
        self._embedding_buffer = []  # 清空缓冲

        # 采样训练子集
        train_size = min(len(embeddings_np), self.max_train_samples)
        if train_size < len(embeddings_np):
            train_indices = np.random.choice(len(embeddings_np), size=train_size, replace=False)
            train_vectors = embeddings_np[train_indices]
            logging.info(f"从 {len(embeddings_np)} 个向量中随机采样 {train_size} 个用于训练。")
        else:
            train_vectors = embeddings_np
            logging.info(f"使用全部 {len(embeddings_np)} 个向量进行训练。")

        if train_vectors.shape[0] == 0:
            self.logger.error("拟合错误：没有有效的训练向量。")
            return

        try:
            # --- 步骤 1：训练 OPQ 旋转矩阵（CPU）---
            logging.info(f"步骤 1: 训练 OPQ 旋转矩阵 (M={self.num_levels})...")
            t0 = time.time()
            opq_matrix_obj = faiss.OPQMatrix(self.input_size, self.num_levels)
            opq_matrix_obj.train(train_vectors)
            self.opq_matrix = opq_matrix_obj
            logging.info(f"步骤 1 完成，耗时 {time.time() - t0:.2f} 秒。")

            # --- 步骤 2：应用旋转并训练 PQ 码本（CPU）---
            logging.info("步骤 2: 应用旋转矩阵并训练 PQ 码本（CPU）...")
            t0 = time.time()
            rotated_train_vectors = self.opq_matrix.apply_py(train_vectors)

            pq_cpu = faiss.ProductQuantizer(self.input_size, self.num_levels, self.n_codebook_bits)
            pq_cpu.train(rotated_train_vectors)   # 统一使用 CPU 训练，避免 GpuIndexPQ 不存在
            self.pq = pq_cpu

            logging.info(f"步骤 2 完成，耗时 {time.time() - t0:.2f} 秒。")

            self.fitted = True
            logging.info(f"OPQ 拟合完成，总耗时 {time.time() - total_start_time:.2f} 秒。")

            # 可选：保存 .pkl 状态（若配置了 state_path）
            if self.state_path:
                self.save_fitted_state(self.state_path)

        except Exception as e:
            self.logger.error(f"Faiss 拟合过程中发生错误: {e}")
            self.fitted = False
            self.opq_matrix = None
            self.pq = None
            raise


        except Exception as e:
            self.logger.error(f"Faiss 拟合过程中发生错误: {e}")
            self.fitted = False; self.opq_matrix = None; self.pq = None
            raise
        finally:
             if res is not None: del res

    # === 与 Trainer 交互 ===
    def forward(self, batch_data: torch.Tensor) -> tuple:
        """ forward 的职责是「攒数据」，直到数据量足够觸發擬合。"""
        if self.fitted: return (None, None, None) # 已拟合/加载，无需操作

        # 添加数据到缓冲区
        self._embedding_buffer.append(batch_data.detach().cpu())
        current_count = sum(len(b) for b in self._embedding_buffer)

        # 检查是否攒够数据
        if current_count >= self._total_item_count:
            try:
                self._fit_faiss() # 触发拟合
            except Exception as e:
                 # 如果拟合失败，仍然返回 None，让 Trainer 知道不用 backward
                 self.logger.error(f"拟合在 forward 触发时失败: {e}")
                 # 不需要清空 buffer，下次 forward 还会尝试
                 
        return (None, None, None) # Trainer 不会计算 loss

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """OPQ 没有迭代损失。"""
        return {'loss_total': torch.tensor(0.0, device=batch_data.device),
                'loss_recon': torch.tensor(0.0, device=batch_data.device)}

    # === 編碼方法 ===
    @torch.no_grad()
    def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor:
        """ 使用已拟合好的 OPQ/PQ 对输入 batch 进行量化。 """
        self.eval()

        # === ✨ 自救逻辑：如果未拟合，尝试从文件加载 ===
        if not self.fitted:
            self.logger.warning("OPQ 在 get_codes 时 fitted=False，尝试从状态文件加载...")
            if self.state_path and self.load_fitted_state(self.state_path):
                 self.logger.info("成功从状态文件恢复拟合状态。")
            else:
                 # 如果 state_path 不存在或加载失败，则抛出错误
                 self.logger.error("无法从状态文件恢复，OPQ 模型未拟合。")
                 raise RuntimeError("OPQ 模型在 get_codes 時未擬合，且无法从状态文件恢复。")
        # === 自救结束 ===

        # 检查 Faiss 对象是否存在 (以防万一加载失败但 fitted 意外为 True)
        if self.opq_matrix is None or self.pq is None:
             self.logger.error("OPQ Faiss 对象为空，即使 fitted=True。状态文件可能已损坏。")
             raise RuntimeError("OPQ Faiss 对象为空，无法执行 get_codes。")

        # 将输入转为 CPU numpy float32
        batch_np = batch_data.cpu().numpy().astype('float32')
        
        try:
            # 1. 应用 OPQ 旋转
            rotated_batch = self.opq_matrix.apply_py(batch_np)
            # 2. 计算 PQ 码 (uint8 numpy)
            codes_np_uint8 = self.pq.compute_codes(rotated_batch)
            # 3. 转为 int64 PyTorch Long Tensor 并移回原设备
            codes_tensor = torch.from_numpy(codes_np_uint8.astype(np.int64)).long().to(batch_data.device)
            return codes_tensor
        except Exception as e:
             self.logger.error(f"OPQ get_codes 失败: {e}")
             error_codes = torch.full((batch_data.shape[0], self.num_levels), -1, dtype=torch.long, device=batch_data.device)
             return error_codes

# --- (移除测试代码，保持脚本纯净) ---