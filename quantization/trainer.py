# /quantization/trainer.py (最终修正版 - 复用 TensorDataset)

import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
# ✅ 导入 TensorDataset
from torch.utils.data import DataLoader, Subset, TensorDataset 
from sklearn.model_selection import train_test_split
import utils # 假设 utils.py 在可导入路径
from collections import defaultdict
# 🚨 (移除) 不再需要 MultiModalDataset 或 EmbeddingDataset
# from dataset import EmbeddingDataset, MultiModalDataset 

class Trainer:
    """
    通用量化器 Trainer (复用 TensorDataset)
    """

    def __init__(self, config: dict, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.model_name = config.get("model_name", model.__class__.__name__)
        # ✅ 从正确的 config key 读取模型配置
        self.model_cfg = config.get(self.model_name.lower(), {}) 
        self.train_cfg = self.model_cfg.get("training_params", {})
        self.common_cfg = config.get("common", {}) # 获取 common 配置
        self.logger = logging.getLogger(f"Trainer[{self.model_name}]")

    # ====================================================
    # 🔹 1. 通用訓練邏輯 (智能調度入口)
    # ====================================================
    
    # (签名已修正：接收 embeddings_data)
    def fit(self, embeddings_data, ckpt_dir):
        """
        通用的 fit 方法，接收实际的 embedding 数据 (numpy array 或 tuple)。
        """
        # (调度逻辑不变)
        if self.model_name.lower() in ['opq']: # 添加其他 one-shot 模型
            return self._fit_one_shot(embeddings_data, ckpt_dir)
        else: # RQVAE, MM_RQVAE, VQVAE, RKMEANS 等
            return self._fit_iterative(embeddings_data, ckpt_dir)

    # ====================================================
    # 🔹 1a. 內部方法：迭代式訓練 (核心修改)
    # ====================================================
    
    # (签名已修正：接收 embeddings_data)
    def _fit_iterative(self, embeddings_data, ckpt_dir):
        """处理需要迭代训练的模型 (接收 numpy 数据)。"""
        self.logger.info(f"开始迭代式训练 ({self.model_name})...")

        # ✅ (核心修改) 根据 embeddings_data 创建 TensorDataset
        is_multimodal = isinstance(embeddings_data, tuple)
        dataset = None # 初始化
        try:
            if is_multimodal:
                embeddings_T, embeddings_I = embeddings_data
                tensor_T = torch.from_numpy(embeddings_T).float()
                tensor_I = torch.from_numpy(embeddings_I).float()
                dataset = TensorDataset(tensor_T, tensor_I) # 使用 TensorDataset
                self.logger.info("创建 TensorDataset (多模态 T+I)...")
            else:
                # 单模态，假设 embeddings_data 是 numpy array
                tensor_data = torch.from_numpy(embeddings_data).float()
                dataset = TensorDataset(tensor_data) # 只包含一个张量
                self.logger.info("创建 TensorDataset (单模态)...")
        except Exception as e:
            self.logger.error(f"创建 Dataset 失败: {e}", exc_info=True)
            raise # 重新抛出异常，因为无法继续

        # (划分训练/验证集)
        try:
            # 使用 common config 中的 test_size 或默认 0.05
            test_size = self.common_cfg.get('validation_split', 0.05)
            if test_size > 0:
                 train_idx, val_idx = train_test_split(list(range(len(dataset))), 
                                                       test_size=test_size, 
                                                       random_state=self.common_cfg.get('seed', 42))
                 self.logger.info(f"数据集已划分为 {1-test_size:.0%} 训练 / {test_size:.0%} 验证")
            else:
                 self.logger.info("未配置验证集划分 (validation_split <= 0)，将使用全部数据训练。")
                 train_idx = list(range(len(dataset)))
                 val_idx = []
                 
        except ValueError as e:
            self.logger.warning(f"无法划分验证集 ({e})，将使用全部数据训练。")
            train_idx = list(range(len(dataset)))
            val_idx = []

        # (创建 DataLoader)
        batch_size = self.train_cfg.get("batch_size", 1024) 
        # 从 common config 获取 num_workers
        num_workers = self.common_cfg.get('num_workers', 0) 
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, num_workers=num_workers, pin_memory=True) if val_idx else None
        self.logger.info(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
        
        # (优化器逻辑不变)
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer_name = self.train_cfg.get("optimizer", "AdamW")
        lr = float(self.train_cfg.get("lr", 1e-4))
        wd = float(self.train_cfg.get("weight_decay", 0.0))
        optimizer = None
        if params_to_optimize:
            try:
                optimizer_class = getattr(torch.optim, optimizer_name)
                optimizer = optimizer_class(params_to_optimize, lr=lr, weight_decay=wd)
                self.logger.info(f"优化器: {optimizer_name}, LR: {lr}, WeightDecay: {wd}")
            except AttributeError:
                 self.logger.error(f"无效的优化器名称: {optimizer_name}")
                 raise
        else:
            self.logger.info("模型没有可训练参数，不创建优化器。")

        # (训练循环逻辑...)
        best_loss, best_epoch = float("inf"), 0
        num_epochs = self.train_cfg.get("epochs", 100)
        best_path = os.path.join(ckpt_dir, f"{self.model_name}_best.pth")
        os.makedirs(os.path.dirname(best_path), exist_ok=True) 

        pbar = tqdm(range(num_epochs), desc=f"Training {self.model_name}", ncols=120)
        for epoch in pbar:
            self.model.train()
            epoch_loss_sum = defaultdict(float) # 使用 defaultdict 简化
            # --- 训练循环 ---
            for batch in train_loader:
                loss_dict = {} 
                # ✅ (核心修改) 正确解包 batch
                if is_multimodal:
                    batch_T, batch_I = batch # TensorDataset 会返回元组
                    batch_T, batch_I = batch_T.to(self.device), batch_I.to(self.device)
                    outputs = self.model(xs_T=batch_T, xs_I=batch_I)
                    loss_dict = self.model.compute_loss(outputs, xs_T=batch_T, xs_I=batch_I)
                else:
                    # 单模态 TensorDataset 返回只有一个元素的元组
                    batch_xs = batch[0].to(self.device) 
                    outputs = self.model(xs=batch_xs)
                    # 确保 compute_loss 接收正确的参数名 (xs)
                    loss_dict = self.model.compute_loss(outputs, xs=batch_xs) 
                
                loss_total = loss_dict.get("loss_total", torch.tensor(0.0, device=self.device))

                if optimizer and hasattr(loss_total, 'requires_grad') and loss_total.requires_grad:
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()
                
                for key, val in loss_dict.items():
                    item_val = val.item() if isinstance(val, torch.Tensor) else float(val) 
                    epoch_loss_sum[key] += item_val

            # 计算平均损失
            num_batches = len(train_loader)
            avg_losses = {k: v / num_batches for k, v in epoch_loss_sum.items()}

            # --- 验证循环 ---
            avg_val_loss = float('inf') 
            if val_loader: 
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss_dict = {} 
                        # ✅ (核心修改) 正确解包 batch
                        if is_multimodal:
                            batch_T, batch_I = batch
                            batch_T, batch_I = batch_T.to(self.device), batch_I.to(self.device)
                            outputs = self.model(xs_T=batch_T, xs_I=batch_I)
                            val_loss_dict = self.model.compute_loss(outputs, xs_T=batch_T, xs_I=batch_I)
                        else:
                            batch_xs = batch[0].to(self.device)
                            outputs = self.model(xs=batch_xs)
                            val_loss_dict = self.model.compute_loss(outputs, xs=batch_xs)
                            
                        loss_val = val_loss_dict.get('loss_total', torch.tensor(0.0, device=self.device))
                        val_loss_sum += loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
                        
                avg_val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else float('inf')
            
            # (更新进度条和日志)
            postfix_str = f"TrL={avg_losses.get('loss_total', 0):.4f}"
            if val_loader:
                 postfix_str += f"|VL={avg_val_loss:.4f}"
                 # 添加更多损失信息（如果存在）
                 if 'loss_recon' in avg_losses: postfix_str += f"|Rec={avg_losses['loss_recon']:.4f}"
                 if 'loss_latent' in avg_losses: postfix_str += f"|Lat={avg_losses['loss_latent']:.4f}"
                 if 'loss_recon_T' in avg_losses: postfix_str += f"|RecT={avg_losses['loss_recon_T']:.4f}"
                 if 'loss_recon_I' in avg_losses: postfix_str += f"|RecI={avg_losses['loss_recon_I']:.4f}"
            pbar.set_postfix_str(postfix_str)

            # (保存最佳模型 - 根据验证集或训练集)
            current_eval_loss = avg_val_loss if val_loader else avg_losses.get('loss_total', float('inf'))
            # 增加一点容忍度，防止因为浮点数精度问题不保存
            if current_eval_loss < best_loss - 1e-6: 
                best_loss = current_eval_loss
                best_epoch = epoch + 1
                if optimizer: # 只有可训练的模型才保存
                    try:
                        torch.save(self.model.state_dict(), best_path)
                    except Exception as e:
                         self.logger.error(f"保存模型失败: {e}", exc_info=True)


        pbar.close()
        self.logger.info("=" * 100)
        self.logger.info(f"🏁 迭代式训练完成 [{self.model_name}]")
        if val_loader:
            self.logger.info(f"📉 最佳验证集 Loss: {best_loss:.6f} (在 Epoch {best_epoch})")
        else:
             final_train_loss = avg_losses.get('loss_total', float('inf'))
             self.logger.info(f"📉 最终训练集 Loss: {final_train_loss:.6f}")
             
        if optimizer: self.logger.info(f"💾 最佳模型已保存至: {best_path}")
        self.logger.info("=" * 100)
        
        # 即使没有优化器（如RKMeans），也返回路径作为完成信号
        # 如果没有保存（例如 epochs=0 或从未改进），best_path 可能不存在，返回 None
        return best_path if os.path.exists(best_path) else None 


    # ====================================================
    # 🔹 1b. 內部方法：一次性擬合
    # ====================================================
    
    # (签名已修正：接收 embeddings_data)
    def _fit_one_shot(self, embeddings_data, ckpt_dir: str) -> str:
        """处理一次性拟合的模型 (接收 numpy 数据)。"""
        self.logger.info(f"开始 one-shot 拟合 ({self.model_name})...")
        self.model.train() 
        
        full_data_tensor = None
        if isinstance(embeddings_data, tuple):
             self.logger.warning("One-shot 模型接收到多模态输入，将只使用第一个模态 (文本)。")
             full_data_tensor = torch.from_numpy(embeddings_data[0]).float().to(self.device)
        else:
             full_data_tensor = torch.from_numpy(embeddings_data).float().to(self.device)
        
        # (调用模型进行拟合)
        try:
            if hasattr(self.model, 'fit') and callable(getattr(self.model, 'fit')):
                 self.logger.info("调用 model.fit()...")
                 self.model.fit(full_data_tensor) 
            else:
                 self.logger.info("调用 model forward()...")
                 self.model(full_data_tensor) 
        except Exception as e:
            self.logger.error(f"One-shot 拟合失败: {e}", exc_info=True)
            raise # 重新抛出，让 main.py 知道失败了
        
        # 使用空文件作为信号，因为模型状态可能在内部或不可保存
        fitted_signal_path = os.path.join(ckpt_dir, f"{self.model_name}_fitted.signal")
        os.makedirs(os.path.dirname(fitted_signal_path), exist_ok=True) 
        with open(fitted_signal_path, 'w') as f: f.write('fitted') # 写入内容表示完成

        self.logger.info("=" * 100)
        self.logger.info(f"🏁 One-shot 拟合完成 [{self.model_name}]")
        self.logger.info(f"💾 拟合完成信号已创建: {fitted_signal_path}")
        self.logger.info("=" * 100)
        # 注意：对于 one-shot，返回信号文件路径，而不是模型路径
        return fitted_signal_path 

    # ====================================================
    # 🔹 2. 通用碼本生成邏輯 (核心修改)
    # ====================================================
    
    # (签名已修正：接收 embeddings_data 和完整 output_path)
    @torch.no_grad()
    def predict(self, embeddings_data, output_path): 
        """生成码本 (接收 numpy 数据)"""
        self.logger.info(f"开始生成码本 ({self.model_name}) -> {output_path}")
        self.model.eval()
        
        # ✅ (核心修改) 根据 embeddings_data 创建 TensorDataset
        is_multimodal = isinstance(embeddings_data, tuple)
        dataset = None
        try:
            if is_multimodal:
                embeddings_T, embeddings_I = embeddings_data
                tensor_T = torch.from_numpy(embeddings_T).float()
                tensor_I = torch.from_numpy(embeddings_I).float()
                dataset = TensorDataset(tensor_T, tensor_I)
            else:
                tensor_data = torch.from_numpy(embeddings_data).float()
                dataset = TensorDataset(tensor_data)
        except Exception as e:
             self.logger.error(f"为 predict 创建 Dataset 失败: {e}", exc_info=True)
             return None # 返回 None 表示失败
            
        # (创建 DataLoader)
        # 使用 common config 中的 predict_batch_size 或默认 2048
        pred_batch_size = self.common_cfg.get('predict_batch_size', 2048) 
        num_workers = self.common_cfg.get('num_workers', 0)
        loader = DataLoader(dataset, batch_size=pred_batch_size, shuffle=False, num_workers=num_workers)
        
        all_codes = []

        try:
            for batch in tqdm(loader, desc="编码中"):
                codes = None 
                # ✅ (核心修改) 正确解包 batch 并调用模型
                if is_multimodal:
                    batch_T, batch_I = batch
                    batch_T, batch_I = batch_T.to(self.device), batch_I.to(self.device)
                    # 优先调用 get_codes
                    if hasattr(self.model, "get_codes"):
                        codes = self.model.get_codes(xs_T=batch_T, xs_I=batch_I)
                    # 否则尝试 encode + quantizer
                    elif hasattr(self.model, "encode") and hasattr(self.model, "quantizer"): 
                         z_e = self.model.encode(xs_T=batch_T, xs_I=batch_I)
                         _, _, codes = self.model.quantizer(z_e) 
                    else: raise ValueError(f"{self.model_name} 缺少 get_codes 或 encode+quantizer 方法")
                else: # 单模态
                    batch_xs = batch[0].to(self.device)
                    if hasattr(self.model, "get_codes"):
                        codes = self.model.get_codes(xs=batch_xs)
                    elif hasattr(self.model, "encode"):
                         output = self.model.encode(xs=batch_xs)
                         # 兼容不同模型的 encode 输出
                         if isinstance(output, torch.Tensor) and output.dtype in [torch.int, torch.long]:
                              codes = output # 假设 encode 直接返回 codes (e.g., OPQ)
                         elif hasattr(self.model, 'quantizer'): 
                              z_e = output
                              _, _, codes = self.model.quantizer(z_e)
                         else: # 尝试将输出直接作为 codes
                              codes = output 
                    else: raise ValueError(f"{self.model_name} 缺少 get_codes/encode 方法")
                
                if codes is not None and isinstance(codes, torch.Tensor):
                     all_codes.append(codes.detach().cpu().numpy().astype(np.int64))
                else:
                     self.logger.warning("模型未返回有效的 codes tensor，跳过此批次。")

        except Exception as e:
             self.logger.error(f"生成 codes 时出错: {e}", exc_info=True)
             return None # 返回 None 表示失败


        if not all_codes:
             self.logger.error("未能生成任何 codes。无法保存码本。")
             return None 

        base_codes = np.vstack(all_codes)
        self.logger.info(f"基础码本生成完毕，形状: {base_codes.shape}")

        # (添加去重层逻辑不变)
        final_codes = base_codes # 默认值
        try: # 将查找配置的操作放入 try-except
            # 使用 .get 避免 KeyError
            model_specific_cfg = self.config.get(self.model_name.lower(), {})
            model_params = model_specific_cfg.get("model_params", {})
            
            if model_params.get('has_dup_layer', True): # 默认 True
                self.logger.info("将构建去重层。")
                # 再次使用 .get 获取 codebook_size
                vocab_size = model_params.get("codebook_size") 
                if vocab_size is None or vocab_size <= 0: 
                    self.logger.error("无法获取有效的 'codebook_size'，无法构建去重层。")
                    # 选择是继续（无去重层）还是报错退出
                    # return None # 报错退出
                    self.logger.warning("将继续，但不添加去重层。")
                else:
                    dedup = utils.build_dedup_layer(base_codes, vocab_size)
                    final_codes = np.concatenate([base_codes, dedup], axis=1)
                    self.logger.info(f"添加去重层后维度: {final_codes.shape}")
            else:
                self.logger.info("配置中 'has_dup_layer' 设为 False，不构建去重层。")
                final_codes = base_codes
        except KeyError as e:
             self.logger.error(f"读取配置时出错 (KeyError: {e})，无法确定是否添加去重层。")
             # 选择继续还是报错
             self.logger.warning("将继续，但不添加去重层。")
             final_codes = base_codes
        except Exception as e:
             self.logger.error(f"处理去重层时发生未知错误: {e}", exc_info=True)
             self.logger.warning("将继续，但不添加去重层。")
             final_codes = base_codes


        # (保存逻辑不变，使用传入的 output_path)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True) 
            np.save(output_path, final_codes)
            
            # (可选 JSON 保存)
            json_path = output_path.replace(".npy", ".codebook.json") 
            json_dict = {str(i): " ".join([f"<L{l}_{v}>" for l, v in enumerate(row)]) 
                         for i, row in enumerate(final_codes)}
            with open(json_path, "w") as f: json.dump(json_dict, f, indent=2)

            self.logger.info(f"✅ 码本保存完成，最终形状: {final_codes.shape}，已保存至: {output_path} (及 .json)")
            return final_codes # 返回生成的码本
        except Exception as e:
             self.logger.error(f"保存码本失败: {e}", exc_info=True)
             return None # 返回 None 表示失败