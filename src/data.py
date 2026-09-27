import glob
import itertools
import os
import random
from typing import Iterator
import torch
from torch.utils.data import Dataset, IterableDataset

# ponytail: minimal byte-pair datasets for streaming text. Upgrade to memory-mapped shards for multi-GB datasets.

class BytePairDataset(Dataset):
    def __init__(self, data: bytes | str):
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.data = data
        self.pairs = list(itertools.pairwise(self.data)) if len(self.data) >= 2 else []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, bool]:
        if idx < 0:
            idx += len(self.pairs)
        curr_b, next_b = self.pairs[idx]
        is_end = (idx == len(self.pairs) - 1)
        return curr_b, next_b, is_end

    @staticmethod
    def collate_fn(batch):
        currs = torch.tensor([item[0] for item in batch], dtype=torch.long)
        nexts = torch.tensor([item[1] for item in batch], dtype=torch.long)
        ends = torch.tensor([item[2] for item in batch], dtype=torch.bool)
        return currs, nexts, ends


class StreamingTextDataset(IterableDataset):
    def __init__(self, file_paths: list[str] | str, seed: int = 42, shuffle: bool = True):
        super().__init__()
        if isinstance(file_paths, str):
            self.file_paths = sorted(glob.glob(file_paths, recursive=True))
        else:
            self.file_paths = list(file_paths)
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[int, int, bool]]:
        rng = random.Random(self.seed)
        paths = list(self.file_paths)
        if self.shuffle:
            rng.shuffle(paths)

        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    data = line.encode('utf-8')
                    if len(data) < 2:
                        continue
                    pairs = list(itertools.pairwise(data))
                    for i, (c, n) in enumerate(pairs):
                        yield c, n, (i == len(pairs) - 1)


def load_corpus(name_or_path: str, seed: int = 42) -> BytePairDataset:
    if os.path.isfile(name_or_path):
        with open(name_or_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return BytePairDataset(content)

    # fallback synthetic sample for tests or known names
    if name_or_path.lower() in ("tinystories", "simplewiki", "sample"):
        sample_text = (
            "Once upon a time, there was a little girl named Lily. "
            "She loved to play in the green garden with her dog Spot. "
            "One day, Spot found a shiny red ball under the tall oak tree. "
            "Lily laughed and threw the ball, and Spot brought it right back."
        )
        return BytePairDataset(sample_text)

    return BytePairDataset(name_or_path)
