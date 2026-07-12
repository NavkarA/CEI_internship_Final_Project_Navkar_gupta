# Mini-GPT: A Small GPT-2-Style Language Model From Scratch

A decoder-only Transformer language model, built from scratch in PyTorch,
inspired by Andrej Karpathy's "Let's build GPT" lecture and scaled down to
train comfortably on a CPU laptop. It learns to generate character-by-character
English text after training on a corpus of Shakespeare.

## What this project demonstrates

- **Tokenization** — a character-level tokenizer (each character is a token)
- **Self-attention** — causal (masked) multi-head self-attention, implemented
  from the matrix-multiply level up, not via a library black box
- **Transformer blocks** — pre-norm residual blocks (attention + MLP)
- **Autoregressive training** — next-character prediction with cross-entropy loss
- **Sampling** — temperature and top-k controlled text generation

## Architecture

This follows the same design as GPT-2, just much smaller:

```
input text
   │
   ▼
[token embedding] + [positional embedding]
   │
   ▼
┌─────────────────────────────┐
│  Transformer Block  x N     │
│  ┌────────────────────────┐ │
│  │ LayerNorm               │ │
│  │ Causal Multi-Head Attn  │ │
│  │ + residual connection   │ │
│  ├────────────────────────┤ │
│  │ LayerNorm               │ │
│  │ Feed-Forward MLP (GELU) │ │
│  │ + residual connection   │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
   │
   ▼
LayerNorm → Linear head → logits over vocabulary
```

| Setting | This project | Full GPT-2 (small) |
|---|---|---|
| Embedding dim | 128 | 768 |
| Attention heads | 4 | 12 |
| Layers | 4 | 12 |
| Context length | 128 | 1024 |
| Tokenizer | character-level (~65 tokens) | BPE (~50,000 tokens) |
| Parameters | ~0.8M | ~124M |

Same architecture end to end, just scaled down roughly 150x so it's
feasible to train on a laptop CPU in a few minutes instead of needing
a GPU cluster for days.

## Project structure

```
mini-gpt/
├── config.py       # all hyperparameters in one place
├── tokenizer.py     # character-level tokenizer
├── model.py         # the Transformer itself (attention, blocks, GPT model)
├── train.py         # training loop
├── generate.py       # generate text from a trained checkpoint
├── data/input.txt    # training corpus (tiny Shakespeare, ~1M characters)
└── checkpoints/       # trained model gets saved here
```

## How to run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train (a few minutes on a laptop CPU with default settings)
python train.py

# 3. generate text from the trained model
python generate.py --prompt "ROMEO:" --length 400
```

Training prints loss every `EVAL_INTERVAL` steps so you can watch it
converge, and automatically generates a sample at the end.

## Design choices worth explaining to evaluators

- **Character-level tokenizer instead of BPE**: real GPT-2 uses byte-pair
  encoding with ~50k tokens. A char-level tokenizer needs no external
  vocab file, keeps the embedding table small, and is very easy to reason
  about — a deliberate trade-off for a CPU-only, from-scratch build.
- **Small hyperparameters** (`config.py`): every dimension (embedding size,
  heads, layers, context length) is a straightforward scale-down of GPT-2's
  actual configuration, not a different architecture. Increasing the numbers
  in `config.py` alone moves this toward the real 124M GPT-2 config, given
  enough compute.
- **From-scratch attention**: `model.py`'s `CausalSelfAttention` computes
  Q/K/V, scaled dot-product scores, the causal mask, and softmax manually,
  rather than calling a pre-built attention module, so every step is visible.

## Ideas for extending this (bonus scope)

- Swap the character tokenizer for a subword/BPE tokenizer (e.g. via the
  `tiktoken` or `tokenizers` library) to shrink sequence lengths and let the
  model learn word-level structure.
- Train on a non-English or code-mixed dataset to test multilingual generation.
  If using multiple scripts (e.g. Devanagari + Latin), just point `DATA_PATH`
  at that text — the character-level tokenizer adapts automatically.
- Try a different attention variant (e.g. sliding-window attention, or
  relative positional encodings instead of learned absolute ones).
- Scale `N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE` up toward GPT-2's real
  124M configuration if you get access to a GPU.

## Notes on CPU feasibility

With the default config (~0.8M parameters, block size 128, batch size 32),
3000 training steps takes roughly 5-10 minutes on a typical laptop CPU
(varies by machine). If it's too slow, lower `MAX_ITERS`, `N_LAYER`, or
`BLOCK_SIZE` in `config.py` — the model will still learn something
recognizable, just less fluent.
