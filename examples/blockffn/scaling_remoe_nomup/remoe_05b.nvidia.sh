#!/bin/bash

# Runs DeepSeek-V3 model
set -ex

pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple swanlab

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=64321
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800

IS_CONVERSION=${IS_CONVERSION:-"0"}
GPUS_PER_NODE=${GPUS_PER_NODE:-"8"}
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"21345"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
MODEL_UNIQUE=${MODEL_UNIQUE:-"test_debug"}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/path/to/tokenizer"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 27
    --hidden-size 1280
    --ffn-hidden-size 3360
    --num-attention-heads 10
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 10
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --vocab-size 73448
    --make-vocab-size-divisible-by 1
    --transformer-impl transformer_engine
    --use-flash-attn
)

MOE_ARGS=(
    --router-type remoe

    --num-experts 79
    --moe-ffn-hidden-size 42
    --moe-shared-expert-intermediate-size 64
    --moe-layer-freq '([0]*1+[1]*26)'
    --moe-router-dtype fp32
    --moe-grouped-gemm

    --router-target-act-ratio 0.2
    --router-entropy-loss-coeff 0.0
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
    --data-cache-path results/${EXP_NAME}/data_cache
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 256

    --lr 1.04e-3
    --min-lr 0
    --train-iters 20000
    --lr-decay-iters 20000
    --lr-warmup-iters 100
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters 1000

    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

LOGGING_ARGS=(
    --log-interval 1 
    --save-interval 1000 \
    --eval-interval 1000000000 
    --eval-iters 100 
    # --save logs/$MODEL_UNIQUE/checkpoints 
    # --load $CHECKPOINT_PATH 
    # --tensorboard-dir "logs/$MODEL_UNIQUE/tensorboard" 
    # --no-load-optim 
    # --no-load-rng 
    --log-throughput 
)

mkdir -p results/${EXP_NAME}

if [[ $IS_CONVERSION == "0" ]]; then
    ENTRY_SCRIPT="pretrain_minicpm.py"
    LOG_PREFIX="train"
else
    ENTRY_SCRIPT="tools/checkpoint/dist_ckpt_to_hf_minicpm.py"
    LOG_PREFIX="convert"
fi

torchrun ${DISTRIBUTED_ARGS[@]} $ENTRY_SCRIPT \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    $@ > results/${EXP_NAME}/${LOG_PREFIX}_r${NODE_RANK}.log 2>&1
