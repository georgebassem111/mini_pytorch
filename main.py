import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import micrograd.nn as mnn
from micrograd.engine import Tensor
from micrograd.optim import Adam
from model import GPT2Config, SimpleGPT2


SEED = 37
MAX_STEPS = 500
BATCH_SIZE = 32
NUM_SAMPLES = 20
TEMPERATURE = 0.5


@dataclass
class DatasetBundle:
    docs: list[str]
    chars: list[str]
    bos_token: int
    vocab_size: int
    x_data: np.ndarray
    y_data: np.ndarray


def configure_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def load_docs():
    if not os.path.exists("input.txt"):
        import urllib.request

        names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
        urllib.request.urlretrieve(names_url, "input.txt")

    docs = [line.strip() for line in open("input.txt", encoding="utf-8").read().splitlines() if line.strip()]
    random.shuffle(docs)
    return docs


def build_dataset(block_size=16):
    docs = load_docs()
    chars = sorted(set("".join(docs)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    bos = len(chars)
    vocab_size = len(chars) + 1

    x_rows = []
    y_rows = []
    for doc in docs:
        tokens = [bos] + [stoi[ch] for ch in doc] + [bos]
        tokens = tokens[: block_size + 1]
        if len(tokens) < block_size + 1:
            tokens = tokens + [bos] * (block_size + 1 - len(tokens))
        x_rows.append(tokens[:-1])
        y_rows.append(tokens[1:])

    return DatasetBundle(
        docs=docs,
        chars=chars,
        bos_token=bos,
        vocab_size=vocab_size,
        x_data=np.asarray(x_rows, dtype=np.int64),
        y_data=np.asarray(y_rows, dtype=np.int64),
    )


def sample_batch(data, batch_size):
    idx = np.random.choice(len(data.x_data), size=batch_size, replace=False)
    return data.x_data[idx], data.y_data[idx]


def micro_ce_loss(model, x_batch, y_batch, vocab_size):
    logits = model(x_batch)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = Tensor(y_batch.reshape(-1), requires_grad=False)
    return mnn.CrossEntropyLoss()(flat_logits, flat_targets)


def sample_token_from_logits(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = np.asarray(logits, dtype=np.float64) / temperature

    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate(model, data, block_size, max_new_tokens=16, temperature=1.0):
    tokens = [data.bos_token]
    for _ in range(max_new_tokens):
        idx = np.asarray([tokens[-block_size:]], dtype=np.int64)
        logits = model(idx).data[0, -1]
        next_token = sample_token_from_logits(logits, temperature=temperature)
        if next_token == data.bos_token:
            break
        tokens.append(next_token)
    return "".join(data.chars[token] for token in tokens[1:] if token < len(data.chars))



def main():
    configure_seeds()
    # ensure_results_dir()

    data = build_dataset(block_size=16)
    config = GPT2Config(
        vocab_size=data.vocab_size,
        block_size=16,
        n_embd=16,
        n_head=4,
        n_layer=1,
    )

    model = SimpleGPT2(config).train()
    optimizer = Adam(model.parameters(), lr=0.01)
    losses = []

    print(f"dataset_size: {len(data.docs)}")
    print(f"vocab_size: {data.vocab_size}")
    
    print(f"num_params: {sum(param.data.size for param in model.parameters())}")

    for step in range(MAX_STEPS):
        x_batch, y_batch = sample_batch(data, BATCH_SIZE)
        optimizer.zero_grad()

        loss = micro_ce_loss(model, x_batch, y_batch, data.vocab_size)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.data))
        print(f"step {step + 1:3d}/{MAX_STEPS} | loss={losses[-1]:.4f}")

    model.eval()
    samples = [
        generate(
            model,
            data,
            config.block_size,
            temperature=TEMPERATURE,
        )
        for _ in range(NUM_SAMPLES)
    ]

    print("\nGenerated samples:")
    for i, sample in enumerate(samples, 1):
        print(f"{i:02d}: {sample}")


if __name__ == "__main__":
    main()
