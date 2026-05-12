# DECO (BlockFFN-v2)

Source codes for pre-training DECO, introduced by the paper: *DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices*. DECO is an improved version of our previous architecture [BlockFFN](https://arxiv.org/pdf/2507.08771), with dense-comparable performance given the same budget of total parameters.

Links: [[Paper](https://arxiv.org/pdf/2605.10933)] [[Models](https://huggingface.co/SparseLLM)]

## Contents

* [Environment](#environment)
* [Data Preparation](#data-preparation)
* [Training](#training)
* [Citation](#citation)

## Environment

Our experiment is based on the framework of [Megatron-LM](https://github.com/nvidia/megatron-lm). All training sessions are conducted within a Docker container. You can quickly set up the environment with the helper script `init_docker.sh`.

Edit `init_docker.sh` to match your machine before running it (image registry, mount directory, CPU / memory / shared-memory limits, etc.).

## Data Preparation

The trainer consumes data in the **standard Megatron-LM indexed-dataset format**: each dataset is a pair of files,

```
<prefix>.bin     # raw token stream (uint16 / int32) of all documents
<prefix>.idx     # offsets, sizes and document boundaries (Megatron format)
```

`<prefix>` (without the `.bin` / `.idx` suffix) is what you pass to `--data-path`. The trainer's standard `GPTDataset` mmap-reads the `.bin` and the `BlendedMegatronDatasetBuilder` mixes multiple prefixes according to per-dataset weights, exactly as in upstream Megatron-LM.

### Step 1 — Get a SentencePiece tokenizer

You can obtain the tokenizer used in our original paper through our open-source [models](https://huggingface.co/SparseLLM). Each time you launch training, you should configure the tokenizer path through the environment variable `TOKENIZER_MODEL`.

### Step 2 — Convert raw text to indexed-dataset shards

[`tools/data/preprocess_data.py`](tools/data/preprocess_data.py) tokenises a corpus and writes the standard `IndexedDataset` `.bin` / `.idx` pair. It accepts either a local JSONL[`.gz`] file or any HuggingFace `datasets` source.

**From a local JSONL file** (one JSON object per line with a `text` field):

```bash
PYTHONPATH=. python tools/data/preprocess_data.py \
    --input /data/raw/fineweb_edu.jsonl \
    --text-field text \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --output-prefix /data/processed/fineweb_edu \
    --append-eod \
    --workers 16
```

This writes `/data/processed/fineweb_edu_text_document.{bin,idx}`. Use the prefix `/data/processed/fineweb_edu_text_document` in `DATA_PATH`.

**Streamed from the HuggingFace Hub** (no full download to disk):

```bash
PYTHONPATH=. python tools/data/preprocess_data.py \
    --hf-dataset HuggingFaceFW/fineweb-edu \
    --hf-split train \
    --hf-streaming \
    --text-field text \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --output-prefix /data/processed/fineweb_edu \
    --append-eod \
    --max-docs 8000000      # ~8B tokens at fineweb-edu's average length
```

Useful flags:

* `--append-eod`: append the tokenizer's EOS id after every document.
* `--workers N`: parallel tokenization processes.
* `--tokenizer-type`: must match the value used in the training script, selected from `{Llama2Tokenizer,GPTSentencePieceTokenizer,SentencePieceTokenizer,HuggingFaceTokenizer}`.
* `--max-docs N`: stop after N documents (useful with `--hf-streaming`).

### Step 3 — Wire the prefixes into a data conf

Copy `examples/data_conf/example_data.sh` to your own file and fill in `DATA_PATH` as a multi-line string of `"<weight> <prefix>"` entries:

```bash
export DATA_PATH="0.55 /data/processed/fineweb_edu_text_document
0.20 /data/processed/the_stack_v2_text_document
0.10 /data/processed/nemotron_cc_text_document
0.10 /data/processed/chinese_fineweb_edu_text_document
0.05 /data/processed/finemath_text_document"
```

Weights are normalised automatically; only their ratio matters. The same conf should be applied after the `source` command within each line of `run.sh`.

### Sanity check

Before launching multi-GPU training, verify your shards with a few lines of Python:

```python
from megatron.core.datasets.indexed_dataset import IndexedDataset
ds = IndexedDataset("/data/processed/fineweb_edu_text_document", mmap=True)
print("num documents:", len(ds))
print("doc 0 (first 16 tokens):", ds[0][:16].tolist())
print("doc 0 length:", len(ds[0]))
```

## Training

All launch commands for our main results, including DECO and the baselines (Dense, ReMoE, BlockFFN-v1, DeepSeek-V3, TopP), at scales of 0.1B / 0.2B / 0.5B / 1.2B, are provided in `run.sh`. Each command has the form:

```bash
export EXP_NAME=blockffn_01b_mul1002_withmean_d64_s128_lr1175e3_b64
export FOLDER=results/${EXP_NAME}
source examples/data_conf/example_data.sh && \
MASTER_ADDR=node1 WORLD_SIZE=1 RANK=0 \
bash examples/blockffn/scaling_blockffn_nomup/blockffn_01b.nvidia.sh \
    --tensorboard-dir ${FOLDER}/tensorboard/ \
    --save             ${FOLDER}/checkpoints/ \
    --wandb-project    scaling_blockffn2_mega \
    --wandb-exp-name   ${EXP_NAME} \
    --wandb-save-dir   ${FOLDER}/swanlab/ \
    --num-experts 42 --moe-ffn-hidden-size 64 --moe-shared-expert-intermediate-size 128 \
    --router-entropy-loss-coeff 1e-8 --router-entropy-loss-coeff-multiplier 1.002 \
    --expert-not-gated --expert-act-func norm_silu \
    --lr 1.1175e-3 --global-batch-size 64 --micro-batch-size 8 \
    --train-iters 15000 --lr-decay-iters 15000 --lr-wsd-decay-iters 1000
```

Before running, set:

* `MASTER_ADDR`, `WORLD_SIZE`, `RANK` for every rank in the cluster (the multi-node entries in `run.sh` show the `node1` / `node2` rank-0 / rank-1 pattern we used).
* `TOKENIZER_MODEL` if you use a tokenizer other than the default path inside the script.
* `examples/data_conf/example_data.sh` so that `DATA_PATH` points at your indexed-dataset prefixes.

Training logs and Wandb / SwanLab artefacts will be written under `results/${EXP_NAME}/`.

## Citation

If you find our work useful for your research, please kindly cite our paper as follows:

```
@article{song2026deco,
      title={{DECO}: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices}, 
      author={Chenyang Song, Weilin Zhao, Xu Han, Chaojun Xiao, Yingfa Chen, Zhiyuan Liu},
      journal={arXiv preprint arXiv:2605.10933},
      year={2026},
      url={https://arxiv.org/pdf/2605.10933}, 
}
```
