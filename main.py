import argparse
import glob
import itertools
import os
import random
import sys
import time
from datetime import datetime
import torch

from src.model.rtu import Model, Layer, Encoder, Decoder

# ponytail: minimal PyTorch runtime for test-model-thing matching MLX CLI.

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold
        self.step = 0

    def save(self):
        self.step += 1
        if self.step % 500 == 0:
            self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, save: bool, frozen: bool):
        outputs = self.model(c, n, end, frozen)
        if save:
            self.save()
        return outputs

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool):
        timestamp = None

        while True:
            text = input(f'\n[{self.now()} | {0 if timestamp is None else time.time() - timestamp:.4f}s]\nUser >> ')
            timestamp = time.time()

            data = (text + '\n').encode('utf-8')

            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, save, frozen)

            print(f'\n[{self.now()}]\nModel >> ', end='', flush=True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen)
                self.write(b)

                if stop > self.threshold:
                    print()
                    break

    def train(self, save: bool, frozen: bool, dataset: str):
        files = glob.glob(dataset, recursive=True)

        if not files:
            raise FileNotFoundError(
                f'Could not find training files with the following glob: {dataset!r}. Try downloading a dataset first.'
            )

        random.shuffle(files)

        while True:
            for file in files:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        if len(data) < 2:
                            continue

                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2, save, frozen)
                            self.write(b)

    def now(self):
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool):
        self.model.load(self.path)
        print(f'parameters: {self.model.count():,}\n')

        try:
            match mode:
                case 'train': self.train(save, frozen, dataset)
                case 'chat': self.chat(save, frozen)

        finally:
            if save:
                self.model.save(self.path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test-model-thing')
    parser.add_argument('path')
    parser.add_argument('mode', choices=['train', 'chat'])

    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--no-save', action='store_false')
    parser.add_argument('--dataset', default='wikipedia_clean/**/wiki_*')

    args = parser.parse_args()

    Runtime(
        path=args.path, threshold=0.35,
        dim=512, layers=16, spread=32, temp=0.75,
        rate=5e-4, bound=(40000, 120000)
    )(args.mode, args.dataset, args.no_save, args.frozen)
