# KISS: Knowledge-aware Intent-guided Subgraph Sampling

> 🎉 **Accepted at [APWeb-WAIM 2026](https://conferences.sigappfr.org/apweb2026/about/)** — the 10th APWeb-WAIM Joint International Conference on Web and Big Data (Danang, Vietnam, Sept 7–9, 2026).

**KISS** is a knowledge graph–based recommendation model. Instead of reasoning over the full collaborative knowledge graph (CKG), it builds a compact subgraph for each user and keeps only the parts that matter to that user's intents:

- **Knowledge-aware** — expands a per-user subgraph over the CKG, linking interactions to item attributes and KG entities.
- **Intent-guided** — distills each user's behavior into a small set of latent intents via Slot Attention.
- **Efficient sampling** — uses those intents to drive Gumbel-TopK node selection, so reasoning stays personalized but cheap.

---

## Overview

| (2a) Subgraph Construction | (2b) Intent Modelling | (2c) Intent-guided Sampling |
|:---:|:---:|:---:|
| ![two-hop](assets/two-hop.svg) | ![intent](assets/intent_modelling.svg) | ![three-hop](assets/three-hop.svg) |

**Fig.** For a target user *u₁*, a user-centric computation graph is built via layer-wise CKG expansion **(2a)**. Item attribute tuples are encoded and aggregated into latent intents via Slot Attention **(2b)**. These intents guide adaptive node sampling, yielding a compact but informative subgraph **(2c)**.

---
## Requirements

- Python ≥ 3.10
- NVIDIA GPU with CUDA 12.x

---

## Clone

```bash
git clone https://github.com/tinta2510/GraphKISS.git
cd GraphKISS
```

---

## Installation

### Script (recommended)

```bash
bash scripts/install.sh        # CUDA 12.8 (default)
bash scripts/install.sh cu121  # CUDA 12.1
```

The script auto-installs `uv` if not present and falls back to plain `venv + pip` if that fails. A `.venv` environment is created in the project root.

### Manual (Python 3.12 venv)

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# PyTorch 2.8 + CUDA 12.8
pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# torch-scatter
pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Other dependencies
pip install numpy scipy tqdm pyyaml
```

> For CUDA 12.1 replace `cu128` → `cu121` and `torch==2.8.0` → `torch==2.3.1`.

---

## Data

Place each dataset under `data/<name>/`. The repo ships with four datasets:

```
data/
├── last-fm/
├── amazon-book/
├── new_last-fm/       # inductive split variant
└── new_amazon-book/   # inductive split variant
```

Each directory contains `train.txt` (or `train_1.txt` for `new_*`), the corresponding test file, `kg.txt`, and entity/relation lists.

---

## Training

```bash
python train.py --data_path data/last-fm/ --config configs/last-fm.yaml --gpu 0
```

Or use the helper script:

```bash
bash scripts/run_single.sh last-fm 0
```

Override individual config values without editing the YAML, and tag the run (the tag is appended to all output filenames):

```bash
python train.py --data_path data/last-fm/ --config configs/last-fm.yaml \
  --override K_intent=8 --override lr=5e-4 --tag myrun
```

Add `--deterministic` for a reproducible run.

Supported datasets and their config files:

| Dataset | Config |
|---|---|
| Last-FM | `configs/last-fm.yaml` |
| Amazon-Book | `configs/amazon-book.yaml` |
| Last-FM (inductive) | `configs/new_last-fm.yaml` |
| Amazon-Book (inductive) | `configs/new_amazon-book.yaml` |

---

## Config Reference

All hyperparameters live in `configs/<dataset>.yaml`. Key parameters:

| Parameter | Description |
|---|---|
| `K` | Max nodes kept per user subgraph per layer |
| `K_ppr` | PPR-based edge budget per (batch, head, rel) group |
| `K_intent` | Number of intent slots |
| `K_neg` | Negative samples per positive for BPR loss |
| `n_layer` | GNN hops |
| `hidden_dim` | Node/relation embedding dimension |
| `pred_rank` | Rank of the low-rank intent-item scorer |
| `lr` | Learning rate |
| `decay_rate` | Per-epoch learning-rate decay |
| `epochs` | Training epochs |
| `lamb` | L2 regularization weight |
| `lambda_intent_div` | Weight for intra-user intent diversity loss |
| `tau_min` / `tau_max` | Logsumexp temperature range; annealed from `tau_max` to `tau_min` over `tau_anneal_epochs` |
| `tau_g` | Gumbel-TopK sampling temperature |

---

## Results

Outputs are written automatically:

| Path | Contents |
|---|---|
| `results/<dataset>_perf.txt` | Config header + per-epoch Recall@20 and NDCG@20 |
| `results/<dataset>_epochs.csv` | Structured per-epoch log (metrics, timing, peak GPU, message counts) |
| `results/checkpoints/<dataset>_best.pt` | Best model checkpoint (saved whenever recall improves) |
| `results/ppr_cache/` | Precomputed PPR scores (reused across runs) |

When `--tag <name>` is set, `<dataset>` becomes `<dataset>_<name>` in all output filenames.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
