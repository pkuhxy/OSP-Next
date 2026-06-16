echo "start process..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WANDB_MODE=${WANDB_MODE:-"offline"}
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "${WANDB_API_KEY}"
fi

export TOKENIZERS_PARALLELISM=false

export ASCEND_LAUNCH_BLOCKING=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600

# Processes per node.
NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS:-16}}

# Total node count.
NNODES=${NNODES:-${MA_NUM_HOSTS:-8}}

# Current node rank.
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}

# Master address: use the first VC_WORKER_HOSTS hostname when available.
if [ -z "${MASTER_ADDR:-}" ]; then
  if [ -n "${VC_WORKER_HOSTS:-}" ]; then
    MASTER_HOSTNAME=$(echo "${VC_WORKER_HOSTS}" | cut -d',' -f1)
    MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_HOSTNAME}'))")
  else
    MASTER_ADDR=127.0.0.1
  fi
fi

MASTER_PORT=${MASTER_PORT:-29503}
WORLD_SIZE=$((${NNODES} * ${NPROC_PER_NODE}))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "WORLD_SIZE=${WORLD_SIZE}"

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_osp_RL_luffy_stage3.py \
  --config configs/train/npu/osp_14b_RL_luffy_stage3.yaml
