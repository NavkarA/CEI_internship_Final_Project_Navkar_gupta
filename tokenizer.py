"""
A minimal character-level tokenizer.

Why character-level instead of a BPE/subword tokenizer (like real GPT-2)?
  - It needs zero external vocabulary/merges files -> fully self-contained.
  - The vocabulary is tiny (~65-100 symbols for English text), which keeps
    the embedding table small -> trains fast on a CPU.
  - It's trivial to explain: "each character is one token."

This is a deliberate simplification vs. real GPT-2 (which uses byte-pair
encoding with a ~50k vocabulary). It's a great, honest thing to mention
in your project write-up as a design trade-off you made for a CPU-only
setup, and "swap in a BPE tokenizer" is a natural bonus extension.
"""

import json


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.stoi = data["stoi"]
        obj.itos = {int(v): k for k, v in obj.stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj
