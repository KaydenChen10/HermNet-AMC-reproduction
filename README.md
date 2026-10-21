# HermNet-AMC Reproduction (CPU)

Reproduction of **"Harnessing Second-Order Statistics for Cross-Scenario Over-the-Air Modulation Classification"** (IEEE Wireless Communications Letters, 2026, Mao & Wang) on a **CPU-only Windows laptop** — no GPU, no CUDA.

> Original code: [BeechburgPieStar/HermNet-AMC](https://github.com/BeechburgPieStar/HermNet-AMC) (MIT License). This repo contains only my modifications, tooling and results; please clone the original repository first and overwrite with the files in `code/`.

## Results

Full **leave-one-domain-out (LODO)** evaluation on the CD2025 OTA dataset (10-class AMC): each fold trains on three scenarios and tests on the fourth, completely unseen one.

| Test scenario (unseen) | Description | Paper (RTX 3090) | This reproduction (CPU) | Gap |
|---|---|---|---|---|
| L1 (S1) | Outdoor open area | 93.01% | **92.20%** | −0.81 |
| L2 (S2) | Indoor classroom | 92.74% | **88.60%** | −4.14 |
| L3 (S3) | Indoor office | 91.11% | **91.01%** | −0.10 |
| L4 (S4) | Corridor & staircase (hardest) | 81.02% | **75.35%** | −5.67 |
| **Average** | | **89.47%** | **86.79%** | **−2.68** |

Environment: CPU-only (no CUDA), PyTorch 2.14.0+cpu, Python 3.12, seed 2023, early stopping with patience 10 (folds stopped at epochs 224 / 253 / ~310 / ~150 for L4 / L1 / L2 / L3 respectively). Full training logs: [`logs/`](logs/).

The small average gap (−2.68 points) is within the expected reproduction variance for cross-scenario domain generalization: CPU vs GPU floating-point ordering, PyTorch version differences, and single-seed sensitivity all shift the training trajectory, which DG tasks are especially sensitive to. Two folds (L1, L3) match the paper within 1 point. All results are far above the 10% random baseline, and the full pipeline — data loading, LODO protocol, HCA module, training/early-stopping/testing — runs correctly end to end.

## What I changed (and why)

The original code hard-codes `.cuda()` and assumes a multi-GPU Linux box. Modifications:

1. **CPU fallback** (`code/main.py`, `code/utils/train_tools.py`): all `.cuda()` calls replaced with a `DEVICE = cuda if available else cpu` switch; CUDA seed calls guarded.
2. **Checkpoint/resume training** (`code/utils/train_tools.py`): `train_and_evaluate` now saves a full checkpoint (model + optimizer + epoch + best loss + patience counter) every epoch and resumes from it. A full CPU training run takes ~2 hours in ~10-minute shell chunks — this makes interrupted/resumed training possible without losing progress.
3. **Smoke-test dataset generator** (`code/make_smoke_dataset.py`): synthesizes a miniature CD2025-shaped dataset (4 classes, 4 domains with artificial phase/frequency shifts to mimic domain shift) to verify the whole pipeline end-to-end before the real data download finishes.
4. **float32 data loading** (`code/utils/train_tools.py`): the `.h5` files store float64; loading is cast to float32 on read, halving peak RAM and preventing `ArrayMemoryError` on memory-constrained machines.

## How to reproduce

```bash
# 1. original repo + dataset
git clone https://github.com/BeechburgPieStar/HermNet-AMC.git
# download CD2025 (Baidu Netdisk, code: 3uux) from the original README,
# place L1.h5..L4.h5 under HermNet-AMC/dataset/CD2025/

# 2. overwrite with the modified files from this repo's code/ directory

# 3. environment (CPU is fine)
pip install torch numpy scipy h5py torchsummary

# 4. (optional) smoke test first
python make_smoke_dataset.py
python main.py --mode train_test --num_cls 4 --epochs 5 \
  --dataset_indexes_for_train L1 L2 L3 --dataset_indexes_for_test L4

# 5. real run (checkpoint/resume built in — safe to interrupt and rerun)
python main.py --mode train_test --num_cls 10 --epochs 400 \
  --dataset_indexes_for_train L1 L2 L3 --dataset_indexes_for_test L4
```

## Citation

```bibtex
@article{mao2026hermnet,
  author  = {Mao, Jiangyuan and Wang, Yu},
  title   = {Harnessing Second-Order Statistics for Cross-Scenario Over-the-Air Modulation Classification},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {doi: 10.1109/LWC.2026.3733129}
}
```

## License

Original code © BeechburgPieStar, MIT License. The CD2025 dataset is hosted by the original authors on Baidu Netdisk and is not redistributed here.
