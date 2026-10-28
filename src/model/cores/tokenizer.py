"""Tokenizer adapters for byte-level (V=256) and compact BPE (V=4096) evaluation.

Provides lossless UTF-8 byte-span tracking across tokenizers so held-out
Bits-Per-Byte (BPB) comparisons remain strictly invariant to vocabulary size.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.model.cores.contracts import ACTION_NAMES, EncodedSpan

SPECIAL_TOKEN_NAMES: Tuple[str, ...] = (
    "<PAD>",
    "<EOS>",
    "<OPT>",
    "<SEP>",
    "<ROUTE_GENERATE>",
    "<ROUTE_CLASSIFY>",
    "<ROUTE_RETRIEVE>",
    "<ROUTE_TOOL>",
    "<ROUTE_ABSTAIN>",
)

CONTROL_IDS: Dict[str, int] = {name: idx for idx, name in enumerate(SPECIAL_TOKEN_NAMES)}


class UTF8ByteTokenizer:
    """Raw UTF-8 byte tokenizer (`V=256`) with exact byte-span accounting.

    In `V=256` mode every UTF-8 byte maps directly to its integer value `0..255`.
    Control markers use reserved ASCII/C0 conventions (`PAD=0`, `EOS=3` ETX,
    `OPT=28` FS, `SEP=29` GS, and `ROUTE_*` mapped to `241..245` unassigned UTF-8
    leading bytes) so special control tokens occupy distinct, non-overlapping IDs
    within `0..255` without colliding with valid UTF-8 text bytes (`0xF5..0xF9`
    never appear in valid UTF-8!).
    """

    name = "utf8_bytes_256"

    def __init__(self) -> None:
        self.vocab_size = 256
        # Note: bytes 0xF5..0xFD (245..253) are strictly invalid in UTF-8 (RFC 3629),
        # so using 245..253 for the 9 control tokens guarantees ZERO collision with
        # any valid UTF-8 string while staying inside V=256!
        self.pad_id = 245
        self.eos_id = 246
        self.opt_id = 247
        self.sep_id = 248
        self.route_token_ids: Dict[str, int] = {
            "generate": 249,
            "classify": 250,
            "retrieve": 251,
            "tool": 252,
            "abstain": 253,
        }
        self._control_set = {
            self.pad_id,
            self.eos_id,
            self.opt_id,
            self.sep_id,
            *self.route_token_ids.values(),
        }

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        raw = list(text.encode("utf-8"))
        if add_eos:
            raw.append(self.eos_id)
        return raw

    def decode(self, token_ids: Sequence[int]) -> str:
        filtered = bytes(int(t) & 0xFF for t in token_ids if int(t) not in self._control_set)
        return filtered.decode("utf-8", errors="replace")

    def encode_span(self, text: str, add_eos: bool = False) -> EncodedSpan:
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        ctrl = 0
        if add_eos:
            ids.append(self.eos_id)
            ctrl = 1
        return EncodedSpan(
            token_ids=ids,
            utf8_byte_count=len(raw_bytes),
            control_token_count=ctrl,
            vocab_size=self.vocab_size,
        )


class CompactBPETokenizer:
    """Deterministic byte-level BPE tokenizer (`V=4096` default, offline-trainable).

    Token layout:
      * `0..8`: 9 reserved control tokens (`<PAD>`, `<EOS>`, `<OPT>`, `<SEP>`,
        `<ROUTE_GENERATE>`, `<ROUTE_CLASSIFY>`, `<ROUTE_RETRIEVE>`, `<ROUTE_TOOL>`,
        `<ROUTE_ABSTAIN>`).
      * `9..264`: all 256 single UTF-8 bytes (`byte + 9`), guaranteeing 100%
        lossless UTF-8 coverage with zero `<UNK>` tokens on unseen text.
      * `265..vocab_size - 1`: deterministic BPE merge tokens trained strictly on
        supplied training texts.
    """

    name = "compact_bpe"
    BYTE_OFFSET = len(SPECIAL_TOKEN_NAMES)  # 9
    BASE_VOCAB = BYTE_OFFSET + 256  # 265

    def __init__(
        self,
        vocab_size: int = 4096,
        merges: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        if vocab_size < self.BASE_VOCAB:
            raise ValueError(
                f"CompactBPETokenizer requires vocab_size >= {self.BASE_VOCAB}, got {vocab_size}"
            )
        self.vocab_size = int(vocab_size)
        self.pad_id = CONTROL_IDS["<PAD>"]
        self.eos_id = CONTROL_IDS["<EOS>"]
        self.opt_id = CONTROL_IDS["<OPT>"]
        self.sep_id = CONTROL_IDS["<SEP>"]
        self.route_token_ids: Dict[str, int] = {
            "generate": CONTROL_IDS["<ROUTE_GENERATE>"],
            "classify": CONTROL_IDS["<ROUTE_CLASSIFY>"],
            "retrieve": CONTROL_IDS["<ROUTE_RETRIEVE>"],
            "tool": CONTROL_IDS["<ROUTE_TOOL>"],
            "abstain": CONTROL_IDS["<ROUTE_ABSTAIN>"],
        }
        self._control_set = set(range(self.BYTE_OFFSET))

        self.merges: List[Tuple[int, int]] = []
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.token_bytes: Dict[int, bytes] = {
            self.BYTE_OFFSET + b: bytes([b]) for b in range(256)
        }
        if merges:
            self._install_merges(merges)

    def _install_merges(self, merges: Sequence[Tuple[int, int]]) -> None:
        max_merges = self.vocab_size - self.BASE_VOCAB
        self.merges = [(int(a), int(b)) for a, b in list(merges)[:max_merges]]
        self.merge_ranks.clear()
        for idx, pair in enumerate(self.merges):
            new_id = self.BASE_VOCAB + idx
            self.merge_ranks[pair] = new_id
            self.token_bytes[new_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]

    @classmethod
    def train_from_texts(
        cls,
        train_texts: Iterable[str],
        vocab_size: int = 4096,
        max_training_bytes: int = 250_000,
    ) -> "CompactBPETokenizer":
        """Deterministically learn BPE merges strictly from training split texts.

        Splits input on whitespace boundaries first (GPT-2 style pre-chunking) so
        merges do not bridge unrelated words, and breaks frequency ties by
        ascending `(id_a, id_b)` tuple for exact cross-platform reproducibility.
        """
        tok = cls(vocab_size=vocab_size)
        target_merges = vocab_size - cls.BASE_VOCAB
        if target_merges <= 0:
            return tok

        word_counts: Counter[Tuple[int, ...]] = Counter()
        consumed = 0
        for text in train_texts:
            if not text:
                continue
            raw = text.encode("utf-8")
            if consumed + len(raw) > max_training_bytes:
                raw = raw[: max(0, max_training_bytes - consumed)]
            consumed += len(raw)
            # Chunk into space-prefixed pieces for fast, clean BPE table training
            piece: List[int] = []
            for b in raw:
                if b in (32, 10, 9) and piece:
                    word_counts[tuple(piece)] += 1
                    piece = [cls.BYTE_OFFSET + b]
                else:
                    piece.append(cls.BYTE_OFFSET + b)
            if piece:
                word_counts[tuple(piece)] += 1
            if consumed >= max_training_bytes:
                break

        words: List[List[int]] = [list(w) for w in sorted(word_counts.keys())]
        freqs: List[int] = [word_counts[tuple(w)] for w in words]

        learned: List[Tuple[int, int]] = []
        for idx in range(target_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for word, freq in zip(words, freqs):
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if not pair_counts:
                break
            # Max count, deterministic tie-break by smallest token pair
            best_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p[0], p[1]))
            if pair_counts[best_pair] < 2:
                break
            new_id = cls.BASE_VOCAB + idx
            learned.append(best_pair)
            a, b = best_pair
            for w_idx, word in enumerate(words):
                if len(word) < 2:
                    continue
                merged: List[int] = []
                i = 0
                n = len(word)
                while i < n:
                    if i < n - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[w_idx] = merged

        tok._install_merges(learned)
        return tok

    def _bpe_piece(self, piece_ids: List[int]) -> List[int]:
        if len(piece_ids) < 2 or not self.merge_ranks:
            return piece_ids
        ids = list(piece_ids)
        while len(ids) >= 2:
            best_pair: Optional[Tuple[int, int]] = None
            best_new_id: Optional[int] = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                cand_id = self.merge_ranks.get(pair)
                if cand_id is not None and (best_new_id is None or cand_id < best_new_id):
                    best_new_id = cand_id
                    best_pair = pair
            if best_pair is None or best_new_id is None:
                break
            a, b = best_pair
            merged: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    merged.append(best_new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        raw = text.encode("utf-8")
        if not raw:
            return [self.eos_id] if add_eos else []
        out: List[int] = []
        piece: List[int] = []
        for b in raw:
            if b in (32, 10, 9) and piece:
                out.extend(self._bpe_piece(piece))
                piece = [self.BYTE_OFFSET + b]
            else:
                piece.append(self.BYTE_OFFSET + b)
        if piece:
            out.extend(self._bpe_piece(piece))
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, token_ids: Sequence[int]) -> str:
        buf = bytearray()
        for t in token_ids:
            tid = int(t)
            if tid in self._control_set:
                continue
            chunk = self.token_bytes.get(tid)
            if chunk is not None:
                buf.extend(chunk)
        return bytes(buf).decode("utf-8", errors="replace")

    def encode_span(self, text: str, add_eos: bool = False) -> EncodedSpan:
        raw_len = len(text.encode("utf-8"))
        ids = self.encode(text, add_eos=add_eos)
        return EncodedSpan(
            token_ids=ids,
            utf8_byte_count=raw_len,
            control_token_count=1 if add_eos else 0,
            vocab_size=self.vocab_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "compact_bpe",
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactBPETokenizer":
        merges = [(int(pair[0]), int(pair[1])) for pair in payload.get("merges", [])]
        return cls(vocab_size=int(payload["vocab_size"]), merges=merges)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompactBPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_tokenizer(
    kind: str = "bpe",
    vocab_size: int = 4096,
    train_texts: Optional[Iterable[str]] = None,
) -> UTF8ByteTokenizer | CompactBPETokenizer:
    """Factory returning either `UTF8ByteTokenizer` (`bytes`) or `CompactBPETokenizer` (`bpe`)."""
    norm = kind.strip().lower()
    if norm in ("bytes", "byte", "utf8", "utf8_bytes_256"):
        if vocab_size != 256:
            raise ValueError(f"Byte tokenizer requires vocab_size=256, got {vocab_size}")
        return UTF8ByteTokenizer()
    if norm in ("bpe", "compact_bpe"):
        if train_texts is not None:
            return CompactBPETokenizer.train_from_texts(train_texts, vocab_size=vocab_size)
        return CompactBPETokenizer(vocab_size=vocab_size)
    raise ValueError(f"Unknown tokenizer kind {kind!r}; expected 'bytes' or 'bpe'")
