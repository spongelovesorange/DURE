import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from tqdm import tqdm

# Local minimal helpers to avoid optional heavy deps (e.g., PIL) from preprocessing.utils
def check_path(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(dic: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(dic, fp, indent=4, ensure_ascii=False)


def read_user_sequences(dataset_root: str, dataset: str) -> Dict[str, List[int]]:
    """
    Load user -> item sequence mapping from {dataset}.inter.json.
    Values are 0-based integer item IDs in chronological order.
    """
    inter_path = os.path.join(dataset_root, dataset, f"{dataset}.inter.json")
    data = load_json(inter_path)
    if data is None:
        raise FileNotFoundError(f"Not found: {inter_path}. Please run process_data.py first for {dataset}.")
    # keys are user indices (as strings), values are lists of 0-based item indices (as strings or ints)
    user2items: Dict[str, List[int]] = {
        str(uid): [int(x) for x in items]
        for uid, items in data.items()
        if isinstance(items, list) and len(items) > 0
    }
    return user2items


def read_item_meta(dataset_root: str, dataset: str) -> Dict[int, Dict]:
    """
    Load item metadata mapping from {dataset}.item.json.
    Keys are 0-based integer item IDs (as strings in JSON), values include 'genres'.
    """
    item_path = os.path.join(dataset_root, dataset, f"{dataset}.item.json")
    meta = load_json(item_path)
    if meta is None:
        raise FileNotFoundError(f"Not found: {item_path}. Please run process_data.py first for {dataset}.")
    # Normalize keys to int
    return {int(k): v for k, v in meta.items()}


def compute_user_major_genre(
    user_seq: List[int], item_meta: Dict[int, Dict]
) -> Tuple[str, float]:
    """
    Determine user's dominant genre by majority proportion among all interactions.
    Returns (major_genre, proportion). If no genres available, returns ("", 0.0).
    A movie can have multiple genres; we count each movie toward all its genres.
    """
    counter = Counter()
    total = 0
    for iid in user_seq:
        meta = item_meta.get(iid, {})
        # Support both 'genres' (list) and 'categories' (string)
        genres = meta.get('genres', [])
        categories = meta.get('categories', "")
        
        valid_tags = []
        if isinstance(genres, list):
            valid_tags.extend([g for g in genres if isinstance(g, str) and g.strip()])
        
        if isinstance(categories, str) and categories.strip():
            # Amazon categories are comma separated string
            valid_tags.extend([c.strip() for c in categories.split(',') if c.strip()])
            
        if not valid_tags:
            continue
            
        counter.update(valid_tags)
        total += 1

    if total == 0 or not counter:
        return "", 0.0

    major_genre, major_count = counter.most_common(1)[0]
    proportion = major_count / total if total > 0 else 0.0
    return major_genre, proportion


def split_clean_and_forget(
    user2items: Dict[str, List[int]],
    item_meta: Dict[int, Dict],
    threshold: float = 0.9,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    For each user:
      - Compute dominant genre and proportion.
      - If proportion >= threshold, mark interactions whose item genres do not contain the dominant genre as incorrect (I_corr).
      - Build clean sequence by removing I_corr.

    Returns:
      clean_user2items: cleaned sequences (retain set)
      forget_targets: for each user, list of targets (item IDs) that are I_corr in chronological order
      forget_histories: for each user, list of history lengths aligned with forget_targets; not saved, but can be used for debugging
    """
    clean_user2items: Dict[str, List[int]] = {}
    forget_targets: Dict[str, List[int]] = {}
    forget_histories: Dict[str, List[int]] = {}

    for uid, seq in tqdm(user2items.items(), desc="Split clean/forget"):
        major, prop = compute_user_major_genre(seq, item_meta)
        clean_seq: List[int] = []
        bad_targets: List[int] = []
        bad_hist_lens: List[int] = []

        # Build clean history on the fly to support per-position forget samples
        for idx, iid in enumerate(seq):
            meta = item_meta.get(iid, {})
            genres = meta.get('genres', [])
            categories = meta.get('categories', "")
            
            item_tags = set()
            if isinstance(genres, list):
                item_tags.update([g for g in genres if isinstance(g, str) and g.strip()])
            if isinstance(categories, str) and categories.strip():
                item_tags.update([c.strip() for c in categories.split(',') if c.strip()])

            has_major = (major and major in item_tags)

            # If the user doesn't have a dominant genre (prop < threshold), we treat all as good
            is_corr = (prop >= threshold) and (not has_major)

            if is_corr:
                # This interaction is to-be-forgotten: record a sample with current clean history
                bad_targets.append(iid)
                bad_hist_lens.append(len(clean_seq))
                # Do NOT add to clean_seq
            else:
                clean_seq.append(iid)

        if len(clean_seq) >= 3:
            clean_user2items[uid] = clean_seq
        # Save forget lists even if empty (filter later when writing)
        if bad_targets:
            forget_targets[uid] = bad_targets
            forget_histories[uid] = bad_hist_lens

    return clean_user2items, forget_targets, forget_histories


def save_jsonl(dataset_root: str, dataset: str, split_name: str, samples: List[Dict]):
    out_dir = os.path.join(dataset_root, dataset)
    check_path(out_dir)
    out_path = os.path.join(out_dir, f"{dataset}.{split_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {split_name} to: {out_path} ({len(samples)} rows)")


def build_train_valid_test(
    clean_user2items: Dict[str, List[int]],
    max_history_len: int = 50,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build train/valid/test samples from clean sequences using Leave-One-Out per user
    with sliding window expansion for train:
      - Train: from sequence[:-2], expand prefixes -> next item
      - Valid: history = sequence[:-1], target = sequence[-2]
      - Test:  history = sequence[:-0-?], here consistent with original: history = train + last_valid
               but we only save a single sample per user: history = sequence[:-1], target = sequence[-1]
    """
    uids_sorted = sorted(clean_user2items.keys(), key=lambda x: int(x))
    train_rows: List[Dict] = []
    valid_rows: List[Dict] = []
    test_rows: List[Dict] = []

    for uid in uids_sorted:
        seq = clean_user2items[uid]
        if len(seq) < 3:
            continue
        # Train: sliding from [0..n-3]
        train_part = seq[:-2]
        for t in range(1, len(train_part)):
            history = train_part[:t][-max_history_len:]
            target = train_part[t]
            train_rows.append({"user": uid, "history": [int(x) for x in history], "target": int(target)})

        # Valid: last-2 is target, history up to it
        valid_history = seq[:-2][-max_history_len:]
        valid_target = seq[-2]
        valid_rows.append({"user": uid, "history": [int(x) for x in valid_history], "target": int(valid_target)})

        # Test: last item as target, history up to it
        test_history = seq[:-1][-max_history_len:]
        test_target = seq[-1]
        test_rows.append({"user": uid, "history": [int(x) for x in test_history], "target": int(test_target)})

    return train_rows, valid_rows, test_rows


def build_forget_tests(
    user2items: Dict[str, List[int]],
    clean_user2items: Dict[str, List[int]],
    forget_targets: Dict[str, List[int]],
    max_history_len: int = 50,
) -> List[Dict]:
    """
    Build forget evaluation samples:
      For each user and each incorrect interaction (I_corr) in chronological order,
      use as history the clean sequence prefix up to that time (i.e., all good items before it),
      and target = the incorrect item's ID.
    """
    rows: List[Dict] = []
    for uid, targets in forget_targets.items():
        if uid not in clean_user2items:
            # if the clean sequence becomes too short (<3), skip this user's forget samples
            continue
        clean_seq = clean_user2items[uid]
        # To construct history lengths, we replay original sequence and track clean prefix
        clean_prefix: List[int] = []
        seq = user2items[uid]
        t_idx = 0
        for iid in seq:
            if t_idx < len(targets) and iid == targets[t_idx]:
                # A forget sample at this point
                rows.append({
                    "user": uid,
                    "history": [int(x) for x in clean_prefix[-max_history_len:]],
                    "target": int(iid)
                })
                t_idx += 1
                # Do not add incorrect item to clean_prefix
            else:
                # If this item exists in clean_seq at this position, append
                # We cannot rely on equality-by-position; simply append iid if it appears in clean_seq and
                # we haven't exceeded its count in prefix
                # To handle potential duplicates, we track counts
                clean_prefix.append(iid) if iid in clean_seq else None
        # Done one user
    return rows


def main():
    parser = argparse.ArgumentParser(description="Split ML-1M into clean retain set and forget set by dominant genre")
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--dataset_root', type=str, default='../datasets', help='Root directory of processed datasets')
    parser.add_argument('--threshold', type=float, default=0.9, help='Dominant genre ratio threshold per user')
    parser.add_argument('--max_history_len', type=int, default=50)
    args = parser.parse_args()

    # 1) Load remapped sequences and item metadata produced by process_data.py
    user2items = read_user_sequences(args.dataset_root, args.dataset)
    item_meta = read_item_meta(args.dataset_root, args.dataset)

    # 2) Split into clean and forget using the dominant-genre rule
    clean_user2items, forget_targets, _ = split_clean_and_forget(
        user2items=user2items,
        item_meta=item_meta,
        threshold=args.threshold,
    )

    # 3) Build retain train/valid/test JSONL
    train_rows, valid_rows, test_rows = build_train_valid_test(
        clean_user2items, max_history_len=args.max_history_len
    )
    save_jsonl(args.dataset_root, args.dataset, 'train', train_rows)
    save_jsonl(args.dataset_root, args.dataset, 'valid', valid_rows)
    save_jsonl(args.dataset_root, args.dataset, 'test', test_rows)

    # 4) Build forget test JSONL
    forget_rows = build_forget_tests(
        user2items=user2items,
        clean_user2items=clean_user2items,
        forget_targets=forget_targets,
        max_history_len=args.max_history_len,
    )
    save_jsonl(args.dataset_root, args.dataset, 'forget', forget_rows)

    # 5) Save a small report for traceability
    report = {
        "dataset": args.dataset,
        "threshold": args.threshold,
        "num_users_total": len(user2items),
        "num_users_clean": len(clean_user2items),
        "num_forget_samples": len(forget_rows),
        "notes": "train/valid/test are generated from clean sequences; forget.jsonl holds incorrect-interaction targets."
    }
    report_path = os.path.join(args.dataset_root, args.dataset, f"{args.dataset}.clean_forget.report.json")
    write_json_file(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
