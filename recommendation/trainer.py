# trainer.py

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import List, Dict
import logging

# trainer 不再需要关心 metrics 的具体实现
# from metrics import ... (这些引用可以删除了)

def train_one_epoch(model, train_loader, optimizer, device, scheduler=None):
    """执行一个训练周期 (此函数不变)"""
    model.train()
    total_loss = 0.0
    # 自动选择 AMP 精度：优先 bfloat16，其次 float16；CPU/无CUDA则禁用
    use_cuda = torch.cuda.is_available() and (str(device).startswith('cuda'))
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else None)
    # 使用新的 torch.amp API（兼容 PyTorch 2.x），指定設備類型 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    # 讀取梯度裁剪上限（如未配置，默認 1.0）
    try:
        max_grad_norm = float(getattr(model, 'config', {}).get('training_params', {}).get('grad_clip_norm', 1.0))
    except Exception:
        max_grad_norm = 1.0

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        if amp_dtype is not None:
            # 新 API：torch.amp.autocast('cuda', ...)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model.forward(batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            # NaN/Inf 保護：跳過本步
            if not torch.isfinite(loss):
                logging.warning("[Train] Non-finite loss detected (nan/inf). Skipping this step.")
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # 先反縮放，再裁剪
                scaler.unscale_(optimizer)
                try:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                except Exception:
                    pass
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
            else:
                loss.backward()
                try:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                except Exception:
                    pass
                optimizer.step()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
        else:
            outputs = model.forward(batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            if not torch.isfinite(loss):
                logging.warning("[Train] Non-finite loss detected (nan/inf). Skipping this step.")
                continue
            loss.backward()
            try:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            except Exception:
                pass
            optimizer.step()
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list: List[int], device) -> Dict[str, float]:
    """
    【已修正】在评估集上评估模型性能。
    - 此版本修复了“平均值的平均值”错误。
    - 它现在聚合真实的 总和(sum) 和 总计数(count) 来计算准确的平均指标。
    """
    model.eval()
    
    # ✅ 1. 初始化字典来收集 *总和* (不再是 list)
    total_metrics = {f'Recall@{k}': 0.0 for k in topk_list}
    total_metrics.update({f'NDCG@{k}': 0.0 for k in topk_list})
    total_count = 0.0 # ✅ 2. 初始化总计数
    
    with torch.no_grad():
        use_cuda = torch.cuda.is_available() and (str(device).startswith('cuda'))
        amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else None)
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ✅ 3. 调用模型自身的评估方法
            # 我们现在 *期望* batch_metrics 是一个包含 'count' 和 *指标总和* 的字典
            # e.g., {'count': 256, 'Recall@10': 15.0, 'NDCG@10': 7.5}
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    batch_metrics = model.evaluate_step(
                        batch=batch, 
                        topk_list=topk_list
                    )
            else:
                batch_metrics = model.evaluate_step(
                    batch=batch, 
                    topk_list=topk_list
                )
            
            # ✅ 4. 累加总和与计数
            # 从返回的字典中弹出 'count' 键
            current_batch_size = batch_metrics.pop('count', 0)
            
            if current_batch_size == 0:
                # 增加一个保护措施，以防模型忘记返回 'count'
                current_batch_size = batch.get('input_ids', torch.empty(0)).shape[0]
                if current_batch_size > 0:
                    # 使用 logging 模块打印警告
                    logging.warning("model.evaluate_step() did not return 'count'. Inferring from batch size.")

            total_count += current_batch_size
            
            # 累加指标的总和
            for metric, value in batch_metrics.items():
                if metric in total_metrics:
                    total_metrics[metric] += value # 不再是 .append()
    
    # ✅ 5. 计算所有批次的 *真实* 平均指标 (总和 / 总计数)
    avg_metrics = {k: v / total_count if total_count > 0 else 0.0 
                   for k, v in total_metrics.items()}
    
    # 返回一个包含所有平均指标的字典
    return avg_metrics