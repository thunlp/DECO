"""Convert raw text into the indexed-dataset (.bin + .idx) format consumed by
``megatron.core.datasets.gpt_dataset.GPTDataset``.

The output is a single pair of files::

    <output-prefix>.bin   # raw uint16/int32 token stream (one document after another)
    <output-prefix>.idx   # offsets, sizes and document boundaries (Megatron format)

Plug the prefix (without the ``.bin``/``.idx`` suffix) into ``--data-path`` of
``pretrain_minicpm.py``::

    export DATA_PATH="1.0 /data/processed/fineweb_edu_text_document"

Two input modes are supported:

* ``--input``: a single ``.jsonl`` / ``.jsonl.gz`` file (one JSON object per line).
* ``--hf-dataset`` (``--hf-split``, ``--hf-streaming``): any HuggingFace dataset.

Example::

    python tools/data/preprocess_data.py \\
        --input /data/raw/fineweb_edu.jsonl \\
        --text-field text \\
        --tokenizer-type Llama2Tokenizer \\
        --tokenizer-model /models/tokenizer.model \\
        --output-prefix /data/processed/fineweb_edu \\
        --append-eod \\
        --workers 16

This produces ``/data/processed/fineweb_edu_text_document.{bin,idx}``. Use the
prefix ``/data/processed/fineweb_edu_text_document`` in your ``DATA_PATH``.

This script is a slimmed-down version of Megatron-LM's stock ``preprocess_data.py``
(removed: BPE/BERT tokenizers we don't use, multimodal blob field, SentencePiece
chat-template helpers). The on-disk format is byte-for-byte the standard
``IndexedDataset`` format.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy
import torch

# Add the project root to sys.path so that ``megatron.core.datasets`` resolves
# whether the script is launched from ``DECO/`` or from ``DECO/tools/data``.
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from megatron.core.datasets.indexed_dataset import (  # noqa: E402
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


# -----------------------------------------------------------------------------
# Input iterators
# -----------------------------------------------------------------------------

def iter_jsonl(path: str, text_field: str) -> Iterator[str]:
    """Yield strings from a (optionally gzipped) JSONL file."""
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field)
            if text:
                yield text


def iter_plaintext(path: str) -> Iterator[str]:
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def iter_hf_dataset(
    name: str, split: str, text_field: str, streaming: bool, config: Optional[str] = None
) -> Iterator[str]:
    from datasets import load_dataset  # optional dep

    ds = load_dataset(name, config, split=split, streaming=streaming)
    for row in ds:
        text = row.get(text_field)
        if text:
            yield text


# -----------------------------------------------------------------------------
# Tokeniser (SentencePiece) - matches `Llama2Tokenizer` used by pretrain_minicpm.py
# -----------------------------------------------------------------------------

class _SPMTokenizer:
    """Minimal wrapper around sentencepiece used by both ``Llama2Tokenizer`` and
    ``GPTSentencePieceTokenizer`` in Megatron. We replicate just the
    ``encode`` semantics here (the trainer does its own BOS handling)."""

    def __init__(self, model_path: str, prepend_bos: bool):
        import sentencepiece as spm  # delayed: not needed for HF tokenizer mode
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._prepend_bos = prepend_bos
        self.vocab_size = self._sp.vocab_size()
        self.bos_id = self._sp.bos_id()
        self.eos_id = self._sp.eos_id()

    def encode(self, text: str) -> List[int]:
        ids = self._sp.encode(text)
        if self._prepend_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        return ids


class _HFTokenizer:
    """Wrapper around transformers tokenizers (only used when --tokenizer-type=HuggingFaceTokenizer)."""

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer  # optional dep
        self._tok = AutoTokenizer.from_pretrained(name_or_path)
        self.vocab_size = self._tok.vocab_size
        self.eos_id = self._tok.eos_token_id or self._tok.pad_token_id

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=True)


_TOKENIZER = None
_APPEND_EOD = False


def _build_tokenizer(args: argparse.Namespace):
    if args.tokenizer_type in ("Llama2Tokenizer", "GPTSentencePieceTokenizer", "SentencePieceTokenizer"):
        # Llama2Tokenizer prepends BOS by default in Megatron; mirror that here.
        prepend_bos = args.tokenizer_type == "Llama2Tokenizer"
        return _SPMTokenizer(args.tokenizer_model, prepend_bos=prepend_bos)
    if args.tokenizer_type == "HuggingFaceTokenizer":
        return _HFTokenizer(args.tokenizer_model)
    raise SystemExit(f"Unsupported --tokenizer-type: {args.tokenizer_type}")


def _init_worker(args: argparse.Namespace) -> None:
    global _TOKENIZER, _APPEND_EOD
    _TOKENIZER = _build_tokenizer(args)
    _APPEND_EOD = args.append_eod


def _encode(text: str) -> Optional[List[int]]:
    assert _TOKENIZER is not None
    try:
        ids = _TOKENIZER.encode(text)
    except Exception:
        return None
    if not ids:
        return None
    if _APPEND_EOD:
        eod = getattr(_TOKENIZER, "eos_id", None)
        if eod is None or eod < 0:
            raise SystemExit("Tokenizer has no EOS id; cannot honour --append-eod.")
        ids = list(ids) + [int(eod)]
    return ids


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _build_input_iter(args: argparse.Namespace) -> Iterable[str]:
    if args.hf_dataset:
        return iter_hf_dataset(
            name=args.hf_dataset,
            split=args.hf_split,
            text_field=args.text_field,
            streaming=args.hf_streaming,
            config=args.hf_config,
        )
    if not args.input:
        raise SystemExit("must provide either --input or --hf-dataset")
    if args.input.endswith((".jsonl", ".jsonl.gz", ".jsonl.gzip")) or args.input_format == "jsonl":
        return iter_jsonl(args.input, args.text_field)
    return iter_plaintext(args.input)


def _select_dtype(vocab_size: int):
    # Mirror Megatron's preprocess_data.py rule: uint16 if it fits, else int32.
    # Return the numpy *class* (not numpy.dtype instance) - DType.code_from_dtype
    # uses ``cls[value.__name__]`` which only works on classes.
    if vocab_size < (1 << 16) - 1:
        return numpy.uint16
    return numpy.int32


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--input", type=str, help="JSONL[.gz] or plain-text file")
    src.add_argument("--input-format", type=str, choices=["jsonl", "text"], help="force input parser")
    src.add_argument("--hf-dataset", type=str, help="HuggingFace dataset id, e.g. HuggingFaceFW/fineweb-edu")
    src.add_argument("--hf-config", type=str, default=None, help="HF dataset config name")
    src.add_argument("--hf-split", type=str, default="train", help="HF split")
    src.add_argument("--hf-streaming", action="store_true", help="stream the HF dataset (no full download)")
    src.add_argument("--text-field", type=str, default="text", help="JSON / HF field that holds the document text")
    src.add_argument("--max-docs", type=int, default=0, help="stop after N documents (0 = no limit)")

    tok = p.add_argument_group("tokenisation")
    tok.add_argument("--tokenizer-type", type=str, default="Llama2Tokenizer",
                     choices=["Llama2Tokenizer", "GPTSentencePieceTokenizer", "SentencePieceTokenizer", "HuggingFaceTokenizer"],
                     help="must match the value passed to pretrain_minicpm.py")
    tok.add_argument("--tokenizer-model", type=str, required=True, help="path to the SentencePiece .model (or HF id)")
    tok.add_argument("--append-eod", action="store_true", help="append EOS/EOD after each document (recommended)")
    tok.add_argument("--workers", type=int, default=1, help="number of parallel tokenisation processes")

    out = p.add_argument_group("output")
    out.add_argument("--output-prefix", type=str, required=True,
                     help="output prefix; the script writes <prefix>_text_document.bin and .idx")
    out.add_argument("--log-every", type=int, default=10_000, help="log progress every N documents")
    args = p.parse_args()

    # Build a temporary tokenizer in the main process just to read vocab_size.
    tmp_tokenizer = _build_tokenizer(args)
    vocab_size = tmp_tokenizer.vocab_size
    dtype = _select_dtype(vocab_size)
    del tmp_tokenizer

    bin_path = get_bin_path(args.output_prefix + "_text_document")
    idx_path = get_idx_path(args.output_prefix + "_text_document")
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
    builder = IndexedDatasetBuilder(bin_path, dtype=dtype)

    iterator = _build_input_iter(args)
    if args.max_docs > 0:
        iterator = (x for i, x in enumerate(iterator) if i < args.max_docs)

    start = time.time()
    if args.workers <= 1:
        _init_worker(args)
        encoder = (_encode(x) for x in iterator)
    else:
        pool = mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(args,))
        encoder = pool.imap(_encode, iterator, chunksize=64)

    n_docs = 0
    n_tokens = 0
    try:
        for ids in encoder:
            if not ids:
                continue
            builder.add_item(torch.tensor(ids, dtype=torch.long))
            builder.end_document()
            n_docs += 1
            n_tokens += len(ids)
            if args.log_every and n_docs % args.log_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"[preprocess] {n_docs:>12d} docs | tokens={n_tokens:,} | {n_docs/elapsed:,.0f} docs/s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()
        builder.finalize(idx_path)

    elapsed = time.time() - start
    print(
        f"[preprocess] done. docs={n_docs:,} tokens={n_tokens:,} elapsed={elapsed:,.1f}s\n"
        f"             bin={bin_path}\n"
        f"             idx={idx_path}\n"
        f"             use prefix \"{args.output_prefix}_text_document\" in your DATA_PATH",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
