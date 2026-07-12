"""
Mini-GPT: a small decoder-only Transformer, architecturally identical in
spirit to GPT-2 (token + positional embeddings -> stack of pre-norm
transformer blocks with causal self-attention -> final layernorm ->
linear head to vocabulary logits), just with far fewer parameters.

Reading order for understanding the file:
  1. CausalSelfAttention  - the core mechanism: each token looks at itself
                             and all *previous* tokens (never future ones).
  2. FeedForward          - a small per-token MLP applied after attention.
  3. Block                - one transformer layer = attention + MLP,
                             each wrapped in a residual connection and
                             pre-layernorm.
  4. MiniGPT              - embeddings + a stack of Blocks + output head.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask, so position i can only
    attend to positions <= i. This is what makes the model a *decoder*
    (autoregressive, left-to-right) rather than an encoder.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # one linear layer producing Q, K, V for all heads at once
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask, precomputed once and reused (not a learnable param)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embd)

        qkv = self.qkv_proj(x)                      # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)                # each (B, T, C)

        # reshape into (B, n_head, T, head_dim) so each head attends independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)   # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v                                 # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, apply non-linearity, project back down."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer decoder block: pre-norm attention + pre-norm MLP,
    each with a residual ("skip") connection around it."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.ffwd(self.ln2(x))   # residual around MLP
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # standard GPT-2 style init: small normal for weights, zero for biases
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than block_size"

        tok_emb = self.token_embedding(idx)                                   # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate `max_new_tokens` tokens after `idx`."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]         # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature        # last time step only

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        self.train()
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
