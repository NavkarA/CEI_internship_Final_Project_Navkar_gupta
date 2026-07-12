"""
Trains MiniGPT on data/input.txt (character-level language modelling).

Usage:
    python train.py

Everything is CPU-friendly by default (see config.py). Expect a few
minutes for MAX_ITERS=3000 with the default tiny model on a laptop CPU.
"""

import os
import time

import torch

import config
from model import MiniGPT
from tokenizer import CharTokenizer

torch.manual_seed(config.SEED)


def load_data():
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(config.TRAIN_SPLIT * len(data))
    train_data, val_data = data[:n], data[n:]
    return tokenizer, train_data, val_data


def get_batch(data, block_size, batch_size, device):
    # pick random starting points and slice out (input, target) chunks,
    # where target is input shifted by one character (next-token prediction)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.EVAL_ITERS)
        for i in range(config.EVAL_ITERS):
            x, y = get_batch(data, config.BLOCK_SIZE, config.BATCH_SIZE, config.DEVICE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Using device: {config.DEVICE}")

    tokenizer, train_data, val_data = load_data()
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    start_time = time.time()

    for it in range(config.MAX_ITERS):
        if it % config.EVAL_INTERVAL == 0 or it == config.MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start_time
            print(f"step {it:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed")

        xb, yb = get_batch(train_data, config.BLOCK_SIZE, config.BATCH_SIZE, config.DEVICE)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "block_size": config.BLOCK_SIZE,
            "n_embd": config.N_EMBD,
            "n_head": config.N_HEAD,
            "n_layer": config.N_LAYER,
            "dropout": config.DROPOUT,
        },
    }, config.CHECKPOINT_PATH)
    tokenizer.save(config.CHECKPOINT_PATH.replace(".pt", "_vocab.json"))

    print(f"\nDone. Checkpoint saved to {config.CHECKPOINT_PATH}")

    # quick sample so you immediately see what the trained model produces
    print("\n--- sample generation ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
    out_ids = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)[0].tolist()
    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
