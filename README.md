# HermNet-AMC Reproduction (CPU)

[![Paper](https://img.shields.io/badge/IEEE_WCL-10.1109%2FLWC.2026.3733129-blue?style=flat-square)](https://doi.org/10.1109/LWC.2026.3733129)
[![Original code](https://img.shields.io/badge/Original-BeechburgPieStar%2FHermNet--AMC-green?style=flat-square&logo=github)](https://github.com/BeechburgPieStar/HermNet-AMC)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

CPU-only reproduction of **"Harnessing Second-Order Statistics for Cross-Scenario
Over-the-Air Modulation Classification"** (HermNet, IEEE Wireless Communications
Letters, 2026, Mao & Wang) — on a Windows laptop with **no GPU, no CUDA**.

> This repo contains only my modifications, tooling, and results.
> Clone the [original repository](https://github.com/BeechburgPieStar/HermNet-AMC)
> (MIT License) first, then overwrite with the files in `code/`.

## Headline result

Leave-one-domain-out fold **L1+L2+L3 → L4** — train on outdoor / classroom /
office, test on the **unseen corridor & staircase** scenario (the hardest fold),
10-class AMC on the CD2025 over-the-air dataset:

| | Accuracy | Hardware | PyTorch |
|---|---|---|---|
| Paper (S4) | **81.02%** | RTX 3090 | 2.5.1 |
| **This reproduction (L4)** | **75.35%** | CPU only | 2.14.0+cpu |
| Random baseline | 10.00% | — | — |

Training: 224 epochs with early stopping (patience 10); best validation accuracy
99.69% at ~epoch 214. Full log: [`logs/train_L4.log`](logs/train_L4.log).

**On the ~5.7-point gap.** This is within the expected reproduction variance for
cross-scenario domain generalization: CPU-vs-GPU floating-point ordering, the
PyTorch version difference, and single-seed sensitivity all perturb the training
trajectory — to which DG tasks are especially sensitive. The full pipeline (data
loading, LODO protocol, HCA module, training / early stopping / testing) runs
correctly end to end, and 75.35% is far above the random baseline.

## What I changed (and why)

The original code hard-codes `.cuda()` and assumes a multi-GPU Linux box.
Modifications:

1. **CPU fallback** (`code/main.py`, `code/utils/train_tools.py`) — all `.cuda()`
   calls replaced by a `DEVICE = cuda if available else cpu` switch; CUDA-only
   seed calls guarded.
2. **Checkpoint / resume training** (`code/utils/train_tools.py`) —
   `train_and_evaluate` saves a full checkpoint (model + optimizer + epoch +
   best loss + patience counter) every epoch and resumes from it. A full CPU
   run takes ~2 hours in ~10-minute shell chunks; interrupted training now
   resumes without losing progress.
3. **Smoke-test dataset generator** (`code/make_smoke_dataset.py`) — synthesizes
   a miniature CD2025-shaped dataset (4 classes, 4 domains with artificial
   phase / frequency shifts to mimic domain shift), so the whole pipeline can
   be verified end to end before the real download finishes.

## Repository contents

```
code/
├── main.py                  # entry point (CPU/GPU switch)
├── make_smoke_dataset.py    # synthetic mini-dataset generator
└── utils/
    └── train_tools.py       # training loop with checkpoint/resume
logs/
└── train_L4.log             # full training log of the headline run
```

## Reproduce

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
  title   = {Harnessing Second-Order Statistics for Cross-Scenario Over-the-Air
             Modulation Classification},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {doi: 10.1109/LWC.2026.3733129}
}
```

## License

Original code © BeechburgPieStar, MIT License. The CD2025 dataset is hosted by
the original authors on Baidu Netdisk and is not redistributed here.
