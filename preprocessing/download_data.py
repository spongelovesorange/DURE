import os
import sys
import argparse
import requests
import gzip
import json
import csv
import zipfile
import re # 需要 re 模块来处理 MovieLens 标题中的年份
from tqdm import tqdm

# --- 通用輔助函數 (保持不变) ---
def download_file(url: str, filepath: str, description: str = "Downloading"):
    # ... (保持不变) ...
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"准备下载: {os.path.basename(filepath)}")
    print(f"从: {url}")
    if os.path.exists(filepath): print(f"文件已存在，跳过下载: {filepath}"); return True
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"  -> {description}")
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes: print("警告: 下载的文件大小与 Content-Length 不符。")
            print(f"  -> 下载完成，已保存到: {filepath}")
            return True
    except requests.exceptions.Timeout: print(f"下载超时: {url}"); return False
    except requests.exceptions.RequestException as e: print(f"下载失败: {e}"); return False

def extract_zip_file(zip_path: str, extract_to: str):
    # ... (保持不变) ...
    print(f"正在解压: {os.path.basename(zip_path)}")
    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  -> 解压完成，已保存到: {extract_to}")
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e: print(f"解压失败: {e}"); return False
    except Exception as e: print(f"解压时发生未知错误: {e}"); return False


def extract_ratings_from_amazon_reviews(reviews_gz_path: str, ratings_csv_path: str):
    # ... (保持不变) ...
    print(f"\n正在从 {os.path.basename(reviews_gz_path)} 提取 Amazon ratings 数据...")
    os.makedirs(os.path.dirname(ratings_csv_path), exist_ok=True)
    try:
        with gzip.open(reviews_gz_path, 'rt', encoding='utf-8') as f_in, \
             open(ratings_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            for line in tqdm(f_in, desc="  -> 提取中"):
                try:
                    review = json.loads(line); item_id = review.get('asin'); user_id = review.get('reviewerID'); rating = review.get('overall'); timestamp = review.get('unixReviewTime')
                    if all([item_id, user_id, rating is not None, timestamp is not None]): writer.writerow([item_id, user_id, rating, timestamp])
                except (json.JSONDecodeError, TypeError): pass
        print(f"  -> 提取完成，ratings 数据已保存到: {ratings_csv_path}")
        return True
    except FileNotFoundError: print(f"错误: 找不到 Review 文件: {reviews_gz_path}"); return False
    except Exception as e: print(f"提取失败: {e}"); return False

# --- 特定数据源处理逻辑 ---

def process_amazon(dataset_name: str, data_version: str, output_dir: str):
    # ... (保持不变) ...
    print("\n" + "="*15 + f" 处理 Amazon 数据集: {dataset_name} (v{data_version}) " + "="*15)
    category = dataset_name; base_output_dir = os.path.join(output_dir, f'amazon{data_version}'); base_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
    metadata_dir = os.path.join(base_output_dir, 'Metadata'); ratings_dir = os.path.join(base_output_dir, 'Ratings'); review_dir = os.path.join(base_output_dir, 'Review')
    meta_gz_filename = f'meta_{category}.json.gz'; review_gz_server_filename = f'reviews_{category}_5.json.gz' if data_version == '14' else f'{category}_5.json.gz'
    review_gz_local_filename = f'{category}_5.json.gz'; ratings_csv_filename = f'{category}.csv'
    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")
    meta_gz_path = os.path.join(metadata_dir, meta_gz_filename); meta_url = f'{base_url}{meta_gz_filename}'
    if not download_file(meta_url, meta_gz_path, description="下载元数据"): return False
    review_gz_temp_path = os.path.join(review_dir, review_gz_server_filename); review_gz_final_path = os.path.join(review_dir, review_gz_local_filename); review_url = f'{base_url}{review_gz_server_filename}'
    if not os.path.exists(review_gz_final_path):
        print("\n开始处理评论文件...")
        if not os.path.exists(review_gz_temp_path):
             if not download_file(review_url, review_gz_temp_path, description="下载评论"): return False
        else: print(f"临时文件已存在: {review_gz_temp_path}")
        try:
            os.rename(review_gz_temp_path, review_gz_final_path); print(f"  -> 文件已重命名为: {review_gz_local_filename}")
        except OSError as e: print(f"重命名失败: {e}"); return False
    else: print(f"\n评论文件已存在，跳过下载和重命名: {review_gz_final_path}")
    ratings_csv_path = os.path.join(ratings_dir, ratings_csv_filename)
    if not os.path.exists(ratings_csv_path):
        if not extract_ratings_from_amazon_reviews(review_gz_final_path, ratings_csv_path): return False
    else: print(f"\nRatings 文件已存在，跳过提取: {ratings_csv_path}")
    print("\nAmazon 数据下载和初步提取完成！"); print("生成的文件:"); print(f"  - {os.path.abspath(meta_gz_path)}"); print(f"  - {os.path.abspath(review_gz_final_path)}"); print(f"  - {os.path.abspath(ratings_csv_path)}")
    return True


def process_movielens(dataset_name: str, output_dir: str):
    """
    处理 MovieLens 数据集的下载、解压和格式转换 (修正版)。
    """
    print("\n" + "="*15 + f" 处理 MovieLens 数据集: {dataset_name} " + "="*15)

    dataset_urls = {
        'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }
    if dataset_name not in dataset_urls: print(f"错误: 不支持的 MovieLens 数据集 '{dataset_name}'"); return False

    url = dataset_urls[dataset_name]
    base_output_dir = os.path.join(output_dir, dataset_name)
    temp_dir = os.path.join(base_output_dir, 'raw')
    final_dir = os.path.join(base_output_dir, 'processed')
    zip_filename = f'{dataset_name}.zip'
    zip_path = os.path.join(temp_dir, zip_filename)
    
    # 修正：解压后的文件夹名可能不同
    extracted_folder_name = {
        'ml-1m': 'ml-1m',
        'ml-10m': 'ml-10M100K', # ml-10m 解压后文件夹名特殊
        'ml-20m': 'ml-20m'
    }.get(dataset_name, dataset_name) # 默认为数据集名

    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")

    # --- 1. 下载 ---
    if not download_file(url, zip_path, description=f"下载 {dataset_name}"): return False

    # --- 2. 解压 ---
    # 使用更准确的解压后文件夹名检查
    extracted_path = os.path.join(temp_dir, extracted_folder_name)
    if not os.path.exists(extracted_path): # 检查文件夹是否存在
        print(f"目标解压文件夹 {extracted_path} 不存在，开始解压...")
        if not extract_zip_file(zip_path, temp_dir): return False
    else:
        print(f"目标解压文件夹 {extracted_path} 已存在，跳过解压。")

    # --- 3. 处理数据 (格式转换) ---
    ratings_output_csv = os.path.join(final_dir, f'{dataset_name}.csv')
    # 修正：统一元数据文件名为 .item.json
    movies_output_json = os.path.join(final_dir, f'{dataset_name}.item.json') 

    if os.path.exists(ratings_output_csv) and os.path.exists(movies_output_json):
        print(f"\n处理后的文件已存在，跳过处理: {final_dir}")
        print("\nMovieLens 数据下载和初步处理完成！")
        print("生成的文件:"); print(f"  - {os.path.abspath(ratings_output_csv)}"); print(f"  - {os.path.abspath(movies_output_json)}")
        return True

    print("\n开始处理解压后的数据...")
    os.makedirs(final_dir, exist_ok=True)

    # 定位解压后的 ratings 和 movies 文件
    # 使用准确的解压路径
    ratings_file_path = os.path.join(extracted_path, 'ratings.dat' if dataset_name != 'ml-20m' else 'ratings.csv')
    movies_file_path = os.path.join(extracted_path, 'movies.dat' if dataset_name != 'ml-20m' else 'movies.csv')

    if not os.path.exists(ratings_file_path): print(f"错误: 找不到 ratings 文件: {ratings_file_path}"); return False
    if not os.path.exists(movies_file_path): print(f"错误: 找不到 movies 文件: {movies_file_path}"); return False

    print(f"  -> 找到 ratings 文件: {ratings_file_path}")
    print(f"  -> 找到 movies 文件: {movies_file_path}")

    # --- 处理 Ratings ---
    print("  -> 处理 Ratings...")
    is_csv = ratings_file_path.lower().endswith('.csv')
    skip_header = is_csv

    try:
        with open(ratings_file_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(ratings_output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)

            if is_csv: # CSV 使用 csv.reader
                reader = csv.reader(f_in)
                if skip_header:
                     try: next(reader)
                     except StopIteration: pass
                for row in tqdm(reader, desc="    Processing ratings CSV"):
                    if not row or len(row) < 4: continue
                    user_id, movie_id, rating, timestamp = row[0], row[1], row[2], row[3]
                    writer.writerow([movie_id, user_id, rating, timestamp])
            else: # DAT 使用手动 split
                for line in tqdm(f_in, desc="    Processing ratings DAT"):
                    line = line.strip()
                    if not line: continue
                    parts = line.split('::')
                    if len(parts) >= 4:
                        user_id, movie_id, rating, timestamp = parts[0], parts[1], parts[2], parts[3]
                        writer.writerow([movie_id, user_id, rating, timestamp])

    except Exception as e:
        print(f"处理 ratings 文件失败: {e}")
        if os.path.exists(ratings_output_csv): os.remove(ratings_output_csv)
        return False
    print(f"  -> Ratings CSV 已保存到: {ratings_output_csv}")

    # --- 处理 Movies ---
    print("  -> 处理 Movies...")
    is_csv = movies_file_path.lower().endswith('.csv')
    skip_header = is_csv
    movies_data = {}

    try:
        with open(movies_file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            if is_csv: # CSV 使用 csv.reader
                reader = csv.reader(f_in, quotechar='"')
                if skip_header:
                     try: next(reader)
                     except StopIteration: pass
                for row in tqdm(reader, desc="    Processing movies CSV"):
                    if not row or len(row) < 3: continue
                    try:
                         movie_id, title_raw, genres_raw = row[0], row[1], row[2]
                         # ... (后续处理逻辑与 DAT 相同) ...
                         title = title_raw.strip(); genres = genres_raw.strip().split('|') if genres_raw else []
                         year = None; title_clean = title; match = re.search(r'\((\d{4})\)$', title)
                         if match: year = int(match.group(1)); title_clean = title[:match.start()].strip()
                         description = f"{title_clean} ({year})" if year else title_clean
                         if genres: description += f". Genres: {', '.join(genres)}"
                         movies_data[movie_id] = {'title': title_clean, 'year': year, 'genres': genres, 'description': description}
                    except Exception as inner_e: print(f"Error processing movie line {row}: {inner_e}"); continue
            else: # DAT 使用手动 split
                 for line in tqdm(f_in, desc="    Processing movies DAT"):
                     line = line.strip()
                     if not line: continue
                     parts = line.split('::')
                     if len(parts) >= 3:
                         try:
                             movie_id, title_raw, genres_raw = parts[0], parts[1], parts[2]
                             # ... (后续处理逻辑与 CSV 相同) ...
                             title = title_raw.strip(); genres = genres_raw.strip().split('|') if genres_raw else []
                             year = None; title_clean = title; match = re.search(r'\((\d{4})\)$', title)
                             if match: year = int(match.group(1)); title_clean = title[:match.start()].strip()
                             description = f"{title_clean} ({year})" if year else title_clean
                             if genres: description += f". Genres: {', '.join(genres)}"
                             movies_data[movie_id] = {'title': title_clean, 'year': year, 'genres': genres, 'description': description}
                         except Exception as inner_e: print(f"Error processing movie line {parts}: {inner_e}"); continue

        with open(movies_output_json, 'w', encoding='utf-8') as f_out:
            json.dump(movies_data, f_out, indent=2, ensure_ascii=False)
        print(f"  -> Movies JSON 已保存到: {movies_output_json}")

    except Exception as e:
        print(f"处理 movies 文件失败: {e}")
        if os.path.exists(movies_output_json): os.remove(movies_output_json)
        return False

    print("\nMovieLens 数据下载和初步处理完成！")
    print("生成的文件:")
    print(f"  - {os.path.abspath(ratings_output_csv)}")
    print(f"  - {os.path.abspath(movies_output_json)}")
    return True


# --- 主程序入口 (保持不变) ---
def main():
    parser = argparse.ArgumentParser(description="下载并初步处理 Amazon 或 MovieLens 数据集。")
    parser.add_argument('--source', type=str, required=True, choices=['amazon', 'movielens'], help="数据源类型 ('amazon' 或 'movielens')")
    parser.add_argument('--dataset', type=str, required=True, help="具体的 数据集名称 (例如 'Musical_Instruments', 'Baby', 'ml-1m', 'ml-20m')")
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'], help="Amazon 数据集年份版本 (14 或 18)，仅当 source='amazon' 时需要。")
    parser.add_argument('--output_dir', type=str, default='../datasets', help="保存数据的根目录。")
    args = parser.parse_args()

    if args.source == 'amazon':
        process_amazon(args.dataset, args.data_version, args.output_dir)
    elif args.source == 'movielens':
        valid_ml_datasets = ['ml-1m', 'ml-10m', 'ml-20m']
        if args.dataset not in valid_ml_datasets: print(f"错误: 对于 source='movielens'，--dataset 必须是 {valid_ml_datasets} 中的一个。"); sys.exit(1)
        process_movielens(args.dataset, args.output_dir)
    else: print(f"错误: 不支持的数据源 '{args.source}'"); sys.exit(1)

if __name__ == '__main__':
    main()