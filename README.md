# Modality Disentangled Learning for Incomplete Multimodal Emotion Recognition

<p align="center">
  <b>A Primitive Memory Distillation Perspective</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Incomplete%20Multimodal%20Emotion%20Recognition-2f6fed" alt="task">
  <img src="https://img.shields.io/badge/Method-PriMD-7c3aed" alt="method">
  <img src="https://img.shields.io/badge/Modules-SSSD%20%7C%20DPMC%20%7C%20DRAD-0f766e" alt="modules">
  <img src="https://img.shields.io/badge/Datasets-IEMOCAP%20%7C%20CMU--MOSI%20%7C%20CMU--MOSEI-f59e0b" alt="datasets">
  <img src="https://img.shields.io/badge/PyTorch-Ready-ee4c2c" alt="pytorch">
</p>

<p align="center">
  <a href="#abstract">Abstract</a> |
  <a href="#method-overview">Method Overview</a> |
  <a href="#installation--usage">Installation & Usage</a> |
  <a href="#experiments">Experiments</a> |
  <a href="#project-structure">Project Structure</a>
</p>

---

## 📌 Abstract

Multimodal Emotion Recognition (MER) often suffers from missing modalities in real-world scenarios. Existing methods usually generate, align, or distill missing modalities as a whole, overlooking the heterogeneity of information components within each modality. Such holistic treatment mixes inferable shared semantics with uncertain modality-specific details, causing unstable representations and weakening robustness. To address this issue, we propose the Primitive Memory Distillation (PriMD) framework. Unlike existing methods, PriMD takes an intra-modal perspective and focuses on the differences among information components within each modality. PriMD first disentangles cross-modal shared semantics from modality-specific representations, and then discretizes the latter into learnable semantic prototypes to construct modality-specific memory banks. When modalities are missing, the student uses the shared semantics of visible modalities as queries to dynamically retrieve prototypes. It compensates for missing modality-specific information within a constrained memory space and aligns with the teacher. Extensive experiments on IEMOCAP, CMU-MOSI, and CMU-MOSEI demonstrate that PriMD achieves state-of-the-art robustness across a wide range of missing-modality settings, while mitigating the instability caused by holistic feature inference.

---

## 🎇 Method Overview

<p align="center">
  <img width="900" alt="PriMD Architecture" src="./figures/PriMD.png">
</p>

PriMD is built around a two-stage learning process:

1. **Teacher stage.** A complete-modality teacher learns shared/specific
   representations and constructs primitive memory banks.
2. **Student stage.** An incomplete-modality student retrieves primitives from
   the frozen memory banks and learns from the complete-modality teacher.

| Component | Full Name | Function |
| --- | --- | --- |
| **SSSD** | Shared-Specific Semantic Decoupling | Separates cross-modal shared semantics from modality-specific information. |
| **DPMC** | Discrete Primitive Memory Construction | Quantizes modality-specific representations into learnable semantic prototypes. |
| **DRAD** | Dynamic Retrieval Augmented Distillation | Dynamically retrieves primitives to compensate missing modality-specific cues. |

---

## 💡 Key Features

- We propose PriMD, a framework that starts from the internal information structure of modalities and formulates incomplete MER as a unified process of shared semantic estimation and constrained compensation of modality-specific information.
- PriMD includes SSSD for separating shared and specific features, as well as DPMC and DRAD for discretizing the feature space and addressing missing-modality compensation.
- Extensive experiments show that PriMD mitigates the instability caused by holistic representation and outperforms existing methods under various modality-missing scenarios.

---

## 🚀 Installation & Usage

### 1. Environment

```bash
cd PriMD

# Optional: activate your own environment first.
# conda activate <your-env>

pip install -r requirements.txt
```

The workspace has been verified with:

| Item | Version / Setting |
| --- | --- |
| OS | Ubuntu 20.04 |
| Python | 3.8 |
| GPU | CUDA-capable NVIDIA GPU |
| Framework | PyTorch with CUDA |

### 2. Dataset Layout

PriMD uses pre-extracted utterance-level multimodal features:

```text
dataset/
|-- IEMOCAP/
|-- CMUMOSI/
`-- CMUMOSEI/
```

Default feature configuration:

| Modality | Feature Name |
| --- | --- |
| Acoustic | `wav2vec-large-c-UTT` |
| Textual | `deberta-large-4-UTT` |
| Visual | `manet_UTT` |

### 3. Quick Start

Run the IEMOCAP four-class experiment:

```bash
sh run_PriMD_iemocap4.sh
```

Run CMU-MOSI or CMU-MOSEI:

```bash
sh run_PriMD_cmumosi.sh
sh run_PriMD_cmumosei.sh
```

Run a single missing-modality condition manually:

```bash
python -u PriMD/train_PriMD.py \
  --dataset=IEMOCAPFour \
  --audio-feature=wav2vec-large-c-UTT \
  --text-feature=deberta-large-4-UTT \
  --video-feature=manet_UTT \
  --seed=66 \
  --batch-size=16 \
  --epoch=200 \
  --lr=0.0001 \
  --hidden=256 \
  --depth=4 \
  --num_heads=2 \
  --drop_rate=0.5 \
  --attn_drop_rate=0.0 \
  --test_condition=a \
  --stage_epoch=100 \
  --gpu=0
```

---

## 🧪 Experiments

### Missing-Modality Settings

`--test_condition` controls which modalities are observed in the student stage:

| Setting | Observed Modalities | Missing Modalities |
| --- | --- | --- |
| `a` | Acoustic | Textual, Visual |
| `t` | Textual | Acoustic, Visual |
| `v` | Visual | Acoustic, Textual |
| `at` | Acoustic, Textual | Visual |
| `av` | Acoustic, Visual | Textual |
| `tv` | Textual, Visual | Acoustic |
| `atv` | Acoustic, Textual, Visual | None |

### Training Outputs

Training logs and checkpoints are saved under:

```text
saved/
|-- log/main_result/
`-- model/main_result/
```

Running logs can also be monitored with:

```bash
tail -f run_logs/PriMD_IEMOCAPFour_a_*.log
```

For IEMOCAP, the final summary line has the following format:

```text
Folder avg: test_condition (...) --test_acc ... --test_ua ...
```

Here:

| Log Field | Metric |
| --- | --- |
| `test_acc` | Weighted Accuracy (WA) |
| `test_ua` | Unweighted Accuracy (UA) |

### Reference Comparison

For the IEMOCAP four-class `a` condition in the paper:

| Method | WA (%) | UA (%) |
| --- | ---: | ---: |
| MoMKE | 69.53 | 70.21 |
| PriMD | 73.82 | 74.46 |

---

## 📁 Project Structure

```text
PriMD/
|-- train_PriMD.py          # Training and evaluation entrypoint
|-- model_primd.py          # SSSD + DPMC + DRAD implementation
|-- utils.py                # Data loading, masking, and model builder
|-- loss.py                 # Masked CE / MSE losses
|-- dataloader_iemocap.py   # IEMOCAP dataloader
`-- dataloader_cmumosi.py   # CMU-MOSI / CMU-MOSEI dataloader
```

Top-level scripts:

```text
run_PriMD_iemocap4.sh
run_PriMD_cmumosi.sh
run_PriMD_cmumosei.sh
```

---

<p align="center">
  <b>PriMD disentangles shared semantics, retrieves primitive memories, and distills robust incomplete multimodal representations.</b>
</p>
