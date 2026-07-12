"""
Generates text from a trained MiniGPT checkpoint.

Usage:
    python generate.py --prompt "ROMEO:" --length 300
    python generate.py --prompt "" --length 500 --temperature 1.0 --top_k 50
"""

import argparse

import torch

import config
from model import MiniGPT
from tokenizer import CharTokenizer


def load_model_and_tokenizer():
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    tokenizer = CharTokenizer.load(config.CHECKPOINT_PATH.replace(".pt", "_vocab.json"))

    model = MiniGPT(**checkpoint["config"]).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="Seed text to continue from")
    parser.add_argument("--length", type=int, default=300, help="Number of new characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Higher = more random")
    parser.add_argument("--top_k", type=int, default=40, help="Restrict sampling to top-k likely tokens")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if args.prompt:
        idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=config.DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)

    out_ids = model.generate(
        idx, max_new_tokens=args.length,
        temperature=args.temperature, top_k=args.top_k,
    )[0].tolist()

    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
