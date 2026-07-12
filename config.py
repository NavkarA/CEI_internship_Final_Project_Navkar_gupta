"""
Central configuration for Mini-GPT.

Everything here is deliberately small so the whole pipeline trains on a
CPU laptop in a few minutes, not hours. If you ever get access to a GPU,
bump these numbers up (commented "GPU" suggestions are included) and
you'd be most of the way to reproducing full GPT-2 (124M params).
"""

import torch

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_PATH = "data/input.txt"     # plain text file, one big corpus
TRAIN_SPLIT = 0.9                # 90% train / 10% val

# ---------------------------------------------------------------------------
# Model architecture (decoder-only Transformer, GPT-2 style)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128     # context length (max tokens attended to at once)   [GPU: 1024]
N_EMBD     = 192     # embedding dimension                              [GPU: 768]
N_HEAD     = 4        # number of attention heads                       [GPU: 12]
N_LAYER    = 6        # number of transformer blocks                    [GPU: 12]
DROPOUT    = 0.1

# This config lands around ~0.8M parameters (character-level vocab),
# vs. full GPT-2 small at 124M params. Same architecture, ~150x smaller.

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
MAX_ITERS      = 6000
EVAL_INTERVAL  = 250
EVAL_ITERS     = 100      # batches used to estimate loss
GRAD_CLIP      = 1.0

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337
CHECKPOINT_PATH = "checkpoints/mini_gpt.pt"
