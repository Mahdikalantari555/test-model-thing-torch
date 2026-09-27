import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import Model

# ponytail: minimal CoLA benchmark evaluation in PyTorch.

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def cola(filepath: str):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    data.append((parts[3].encode('utf-8'), int(parts[1])))
    except FileNotFoundError:
        pass
    return data


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100.0


def rollout(model: Model, b_s: bytes) -> torch.Tensor:
    model.reset()
    with torch.no_grad():
        for b in b_s:
            _, _ = model.step(torch.tensor(b, dtype=torch.long, device=model.device), frozen=True)
    return model.blocks[-1].states.clone()


def benchmark(model: Model, data: list, train: bool, head: Classification, optimizer: torch.optim.Optimizer):
    tp, tn, fp, fn = 0, 0, 0, 0
    device = next(head.parameters()).device

    for i, (b_s, label) in enumerate(data):
        if len(b_s) == 0:
            continue

        state = rollout(model, b_s)
        if state is None:
            continue

        state = state.to(device)

        if train:
            optimizer.zero_grad()
            choice = head(state)
            loss = F.cross_entropy(choice.unsqueeze(0), torch.tensor([label], device=device))
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                choice = head(state)

        predicted = int(choice.argmax().item())

        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1

        if i > 0 and i % 500 == 0:
            phase = 'train' if train else 'held'
            print(f'[{i} / {len(data) - 1}] {phase}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp, tn, fp, fn):.4f})')

    phase = 'train' if train else 'held'
    print(f'[{len(data) - 1} / {len(data) - 1}] {phase}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp, tn, fp, fn):.4f})')


def run(path: str, epochs: int, split: float, data: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model checkpoint not found at {path!r}.')

    model = Model(dim=512, layers=16, spread=32, temp=0.75, rate=5e-4, bound=(40000, 120000))
    model.load(path)
    model.eval()

    head = Classification(model.dim)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    rows = cola(data)
    if not rows or len(rows) < 2:
        raise FileNotFoundError('Invalid or missing CoLA dataset. Download it again from https://nyu-mll.github.io/CoLA/.')

    split_idx = int(len(rows) * (min(max(split, 0.0), 1.0)))
    train_data, held_data = rows[:split_idx], rows[split_idx:]

    print('Starting benchmark.')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}')
        benchmark(model, train_data, True, head, optimizer)
        benchmark(model, held_data, False, head, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoLA benchmark for test-model-thing')
    parser.add_argument('path')
    parser.add_argument('epochs', type=int)
    parser.add_argument('split', type=float)
    parser.add_argument('--data', default='CoLA/original/raw/in_domain_train.tsv')

    args = parser.parse_args()
    run(args.path, args.epochs, args.split, args.data)
