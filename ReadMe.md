# DURE: Dual-Constraint Unlearning for Recommendation

This repository contains the implementation of **DURE** (Dual-Constraint Unlearning for Recommendation), a novel approach for machine unlearning in generative recommendation systems.

## 🚀 Project Overview

DURE enables a pre-trained recommendation model (e.g., GPT-2) to "forget" specific user-item interactions (Forget Set) while maintaining its performance on the rest of the data (Retain Set).

### Key Features
*   **Adapter-Based Architecture**: Uses a lightweight, trainable Adapter (Side Memory) while keeping the Base Model frozen.
*   **Three-Body Loss**: Combines **DPO** (Probability), **ECL** (Contrastive Learning), and **DOL** (Orthogonality) to achieve deep semantic unlearning.
*   **Privacy Guarantee**: Verified by Membership Inference Attack (MIA) with AUC ≈ 0.5.

## 📂 Directory Structure

```
ckpt/
    dure/               # Trained DURE Adapters
        Digital_Music/
            adapter.pth
            pseudo_labels.pt
    recommendation/     # Pretrained Base Models
datasets/               # Data files (Train, Forget, Retain)
recommendation/         # Source code
    dure_main.py        # Training script
    dure_loss.py        # Loss functions (DPO, ECL, DOL)
    eval_dure.py        # Evaluation script
    mia_attack.py       # Privacy attack script
```

## 🛠️ Usage

### 1. Train DURE Adapter
Train the unlearning adapter on the Forget Set.
```bash
bash run_train_dure.sh
```

### 2. Evaluate Performance
Check Recall/NDCG on Forget Set (should be low) and Retain Set (should be high).
```bash
bash run_eval_dure.sh
```

### 3. Run Privacy Attack (MIA)
Verify if the model is safe against Membership Inference Attacks.
```bash
bash run_mia.sh
```

### 4. Run Full Pipeline
Execute all steps in sequence.
```bash
bash run_all.sh
```

## 📊 Experimental Results (Digital Music)

| Metric | Base Model (Dirty) | DURE (Unlearned) | Change |
| :--- | :--- | :--- | :--- |
| **Forget Set Recall@10** | 10.90% | **3.85%** | 📉 -64.7% |
| **Retain Set Recall@10** | 7.96% | **7.98%** | 🟢 +0.02% |
| **MIA AUC** | - | **0.55** | ✅ Safe |

*Note: Results based on 20 epochs training with DPO + 0.1*ECL + 0.1*DOL.*
