import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
import random

# 假设 load_json 是一个可以处理 JSON 文件的共享函数 
def load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

def verify_alignment(args):
    """主对齐检查函数"""
    print("=" * 60)
    print(f"🔹 启动对齐验证 ({args.dataset})")
    print("=" * 60)

    # --- 1. 文件路径设置 ---
    dataset_path = os.path.join(args.data_root, args.dataset)
    
    # 核心映射文件 (由 preprocess_data.py 生成)
    item2id_path = os.path.join(dataset_path, f"{args.dataset}.item2id")
    item_meta_path = os.path.join(dataset_path, f"{args.dataset}.item.json")
    
    # 图像信息文件 (由 download_images.py 生成)
    images_info_path = os.path.join(args.image_info_root, f"{args.dataset}_images_info.json")
    
    # ✅ 修正路径：图像文件夹应该在 args.image_root 下，以 args.dataset 命名
    # 例如：../datasets/amazon14/Images/Baby/
    image_dir = os.path.join(args.image_root, args.dataset) 

    print(f"检查 Item2ID 文件: {item2id_path}")
    print(f"检查 Item Meta 文件: {item_meta_path}")
    print(f"检查 Images Info 文件: {images_info_path}")
    print(f"检查 Image Directory: {image_dir}") # <-- 路径现在是正确的单层

    # 检查文件存在性
    if not os.path.exists(item2id_path):
        print(f"❌ 错误: 找不到 item2id 文件 ({item2id_path})。请先运行预处理。")
        return
    if not os.path.exists(item_meta_path):
        print(f"❌ 错误: 找不到 item.json 文件 ({item_meta_path})。请先运行预处理。")
        return
    if not os.path.exists(images_info_path):
        print(f"❌ 错误: 找不到 images_info 文件 ({images_info_path})。请先下载图片。")
        return

    # --- 2. 加载核心映射数据 ---
    newid_to_origid: Dict[str, str] = {}
    origid_to_newid: Dict[str, str] = {}
    try:
        with open(item2id_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    orig_id, new_id = parts
                    newid_to_origid[new_id] = orig_id
                    origid_to_newid[orig_id] = new_id
    except Exception as e:
        print(f"❌ 错误: 读取 item2id 文件失败: {e}")
        return
        
    item_meta = load_json(item_meta_path)
    if not isinstance(item_meta, dict):
        print(f"❌ 错误: 加载 item.json 失败或格式不正确。")
        return

    images_info = load_json(images_info_path)
    if not isinstance(images_info, dict):
        print(f"❌ 错误: 加载 images_info 失败或格式不正确。")
        return

    # --- 3. 跨文件一致性检查 ---
    total_items = len(newid_to_origid)
    print(f"\n[INFO] 总共找到 {total_items} 个物品需要检查。")
    
    meta_mismatch_count = 0
    img_coverage_count = 0
    img_file_missing_count = 0
    
    check_results: List[Tuple[str, str, str]] = [] # (new_id, title_status, image_status)
    
    for new_id in tqdm(newid_to_origid.keys(), desc="执行对齐检查"):
        orig_id = newid_to_origid[new_id]
        
        # 检查 Item Meta 对齐 (文本)
        meta_data = item_meta.get(new_id)
        if not meta_data:
            meta_status = "❌ META MISSING"
            meta_mismatch_count += 1
        else:
            meta_status = "✅ META OK"
        
        # 检查 Image Info 对齐 (图像)
        img_names = images_info.get(orig_id, [])
        img_info_status = "❌ IMG_INFO MISSING"
        img_file_status = "❌ FILE MISSING"
        
        if img_names and isinstance(img_names, list) and len(img_names) > 0:
            img_info_status = "✅ IMG_INFO OK"
            img_coverage_count += 1
            
            # 检查文件物理存在性 (只检查第一个文件)
            first_img_path = os.path.join(image_dir, img_names[0])
            if os.path.exists(first_img_path):
                 img_file_status = "✅ FILE OK"
            else:
                 img_file_status = "❌ FILE MISSING"
                 img_file_missing_count += 1
        
        check_results.append((new_id, orig_id, meta_data.get('title', 'N/A') if meta_data else 'N/A', img_info_status, img_file_status))

    # --- 4. 打印结果 ---
    print("\n" + "=" * 60)
    print("✅ 验证结果总结：")
    print("-" * 60)
    print(f"总物品数 (来自 {args.dataset}.item2id): {total_items}")
    print(f"1. 文本元数据 ({args.dataset}.item.json) 缺失数: {meta_mismatch_count}")
    print(f"2. 图片信息 ({os.path.basename(images_info_path)}) 覆盖数: {img_coverage_count}")
    print(f"3. 图片文件物理丢失数 (在 {image_dir} 中): {img_file_missing_count}")
    
    if meta_mismatch_count > 0:
         print(f"🚨 **严重警告:** 有 {meta_mismatch_count} 个新 ID 在 item.json 中找不到元数据。这可能意味着预处理有 BUG，或者 meta/rating 文件不一致。")
    if img_file_missing_count > 0:
         print(f"🚨 **警告:** 有 {img_file_missing_count} 个文件在 images_info 中有记录，但物理文件丢失。请重新运行下载脚本。")
         
    if total_items > 0:
         print(f"图片信息覆盖率: {img_coverage_count / total_items:.2%}")
    print("-" * 60)
    
    print("\n🚀 随机抽样检查 (5个 Items):")
    for new_id, orig_id, title, img_info_status, img_file_status in random.sample(check_results, min(5, total_items)):
        print(f"  新ID: {new_id} -> 原始ID: {orig_id}")
        print(f"    - 标题: {title[:50]}...")
        print(f"    - 图片信息状态: {img_info_status}")
        print(f"    - 图片文件状态: {img_file_status}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="多模态数据集对齐验证工具")
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., Baby)')
    parser.add_argument('--data_root', type=str, default='../datasets', help='预处理文件 (.item2id, .item.json) 所在的根目录')
    parser.add_argument('--image_info_root', type=str, default='../datasets/amazon14/Images', 
                        help='包含 _images_info.json 文件的目录。注意Amazon版本。')
    parser.add_argument('--image_root', type=str, default='../datasets/amazon14/Images',
                        help='包含实际图片文件的根目录 (例如 .../Images/)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 🚨 移除这一块导致路径重复的逻辑！
    # if 'Images' in args.image_root:
    #     args.image_root = os.path.join(args.image_root, args.dataset)
    
    verify_alignment(args)