import micrograd.nn as nn
import numpy as np
from dataclasses import dataclass
from micrograd.engine import Tensor


@dataclass
class GPT2Config:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int = 1


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        batch_size, steps, channels = x.data.shape
        qkv = self.c_attn(x).reshape(batch_size, steps, 3, self.n_head, self.head_dim)
        q = qkv[:, :, 0].transpose((0, 2, 1, 3))
        k = qkv[:, :, 1].transpose((0, 2, 1, 3))
        v = qkv[:, :, 2].transpose((0, 2, 1, 3))

        scores = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        causal_mask = np.triu(np.ones((steps, steps), dtype=bool), k=1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], -1e9)
        probs = self.softmax(scores)
        context = probs @ v
        context = context.transpose((0, 2, 1, 3)).reshape(batch_size, steps, channels)
        return self.c_proj(context)


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        batch_size, steps = idx.shape
        positions = np.arange(steps, dtype=np.int64)[None, :]

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(positions)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
