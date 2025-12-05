import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils import load_and_process_config, setup_logging
from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer
from trainer import evaluate
from utils import get_model_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a custom test JSONL (e.g., forget set)")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., TIGER, GPT2, RPG)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ml-1m)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq', 'vqvae', 'mm_rqvae'])
    parser.add_argument('--embedding_modality', type=str, default='text', choices=['text', 'image', 'fused', 'lfused'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to trained checkpoint (.pth). Defaults to config save_path')
    parser.add_argument('--test_json', type=str, default=None, help='Override test JSONL path. If not set, will use datasets/{dataset}/{dataset}.forget.jsonl')
    args = parser.parse_args()

    # 1) Load config
    config = load_and_process_config(
        args.model, args.dataset, args.quant_method, embedding_modality=args.embedding_modality
    )
    setup_logging(config['log_path'])
    logging.info("Loaded base configuration.")

    # 2) Resolve test_json path
    if args.test_json is None:
        # default to forget.jsonl under the dataset folder
        test_json = Path(config['test_json']).with_name(f"{args.dataset}.forget.jsonl")
    else:
        test_json = Path(args.test_json)
    if not test_json.is_file():
        raise FileNotFoundError(f"Test JSONL not found: {test_json}")

    # 3) Prepare token mappings
    item_to_code_map, _ = item2code(
        config['code_path'], config['vocab_sizes'], config['bases']
    )

    # 4) Build model and tokenizer
    ModelClass = get_model_class(args.model)
    model = ModelClass(config, prefix_trie=None)
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer_collate_fn = get_tokenizer(
        model_name=args.model, config=config, item_to_code_map=item_to_code_map
    )

    # 5) Load checkpoint
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else Path(config['save_path'])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded checkpoint from {ckpt_path}")

    # 6) Build test dataset (override the test_json for this run only)
    #    We reuse GenRecDataset by passing a shallow-copied config with test_json swapped
    config_for_eval = dict(config)
    config_for_eval['test_json'] = str(test_json)
    test_dataset = GenRecDataset(config=config_for_eval, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False,
        num_workers=config['training_params'].get('num_workers', 4),
        collate_fn=tokenizer_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # 7) Evaluate
    results = evaluate(
        model,
        test_loader,
        config['evaluation_params']['topk_list'],
        device
    )
    logging.info(f"Evaluation on {test_json.name}: {results}")
    print(results)


if __name__ == '__main__':
    main()
