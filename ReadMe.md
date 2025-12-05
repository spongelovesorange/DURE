## 📂 Project Structure

```text
.
├── asset/                  # Images and static assets
├── ckpt/                   # Model checkpoints
│   ├── pretrained/         # Pretrained models (e.g., t5-small)
│   ├── quantization/       # Quantization model checkpoints
│   └── recommendation/     # Recommendation model checkpoints
│       └── Digital_Music/
│           ├── GPT2_rqvae/ # Base GPT2 model
│           └── DURE/       # DURE adapter and frozen base model
├── datasets/               # Processed datasets
├── evaluation/             # Evaluation scripts
├── logs/                   # Training logs
├── preprocessing/          # Data preprocessing scripts
├── quantization/           # Quantization training code
└── recommendation/         # Recommendation training and evaluation code
```

---

## 🚀 Quick Start: DURE Evaluation

### 1. Evaluate DURE (Unlearning Model)
To evaluate the DURE (Dual-Process Unlearning) model on the Digital_Music dataset:

```bash
python recommendation/eval_dure.py \
    --dataset Digital_Music \
    --base_ckpt ckpt/recommendation/Digital_Music/DURE/base_frozen.pth \
    --adapter_ckpt ckpt/recommendation/Digital_Music/DURE/adapter.pth
```

### 2. Evaluate Base Model (Baseline)
To evaluate the Base Model (Baseline):

```bash
python recommendation/eval_base.py \
    --dataset Digital_Music \
    --base_ckpt ckpt/recommendation/Digital_Music/GPT2_rqvae/best_model.pth
```

---

训练 Base Model (GPT-2)
python recommendation/main.py --model GPT2 --dataset Digital_Music --quant_method rqvae

评估 Base Model
python recommendation/eval_base.py --dataset Digital_Music --base_ckpt ckpt/recommendation/Digital_Music/GPT2_rqvae/best_model.pth

训练 DURE Adapter
python recommendation/dure_main.py --dataset Digital_Music --data_root datasets

评估 DURE
python recommendation/eval_dure.py --dataset Digital_Music --base_ckpt ckpt/recommendation/Digital_Music/DURE/base_frozen.pth --adapter_ckpt ckpt/recommendation/Digital_Music/DURE/adapter.pth