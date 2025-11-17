#!/usr/bin/env bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=""
SHARED_STORAGE_PATH="/dev/shm/epd"
PID_FILE="${PID_FILE:-${CURRENT_DIR}/pid.txt}"
HOST_IP="127.0.0.1"

# Encoder default config
MAX_NUM_SEQS_ENCODER="${MAX_NUM_SEQS_ENCODER:-1}"
GPU_UTILIZATION_ENCODER=0.0
ENCODER_PORT_BASE="${ENCODER_PORT_BASE:-39000}"
ENCODER_DEVICE_ID_BASE=0
ENCODER_NUMBER=1

# PD default config
MAX_NUM_SEQS_PD="${MAX_NUM_SEQS_PD:-128}"
GPU_UTILIZATION_PD=0.95
PD_PORT_BASE="${PD_PORT_BASE:-40000}"
PD_DEVICE_ID_BASE=1
PD_NUMBER=1

# Proxy default config
PROXY_ADDR="${PROXY_ADDR:-127.0.0.1:38000}"           # number of prompts to send in benchmark
LOG_PATH="${CURRENT_DIR}/logs"
MOONCAKE_MASTER_LOG="$LOG_PATH/mooncake_master.log"
MOONCAKE_METADATA_LOG="$LOG_PATH/mooncake_metadata.log"

MOONCAKE_MASTER_PORT=50051
MOONCAKE_METADATA_PORT=8080
MOONCAKE_MASTER_IP="localhost"                      # producer
MOONCAKE_STORE_INSTANCE_IP="localhost"              # consumer
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((30 * 1073741824))   # 30 GB
MOONCAKE_LOCAL_BUFFER_SIZE=$((1 * 1073741824))      # 1 GB
MOONCAKE_REPLICA_NUM=1
MOONCAKE_FAST_TRANSFER=true
MOONCAKE_FAST_TRANSFER_BUFFER_SIZE=3                # 3 GB
mooncake_master \
  --rpc_port $MOONCAKE_MASTER_PORT \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=$MOONCAKE_METADATA_PORT \
  --rpc_thread_num 8 \
  --default_kv_lease_ttl 0 \
  --eviction_ratio 0.05 \
  --eviction_high_watermark_ratio 0.9 \
  >"$MOONCAKE_MASTER_LOG" 2>&1 &
PIDS+=($!)

export MC_MS_AUTO_DISC=0

IMAGE_FILE_PATH=""
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

function start_encoder() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m lm_service.entrypoints.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --transfer-protocol "tcp" \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_ENCODER \
        --max-num-seqs $MAX_NUM_SEQS_ENCODER \
        --enforce-eager \
        --no-enable-prefix-caching \
        --ec-transfer-config '{
            "ec_connector":"ECMooncakeStorageConnector",
            "ec_role":"ec_producer",
            "ec_connector_extra_config": {
                "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/producer.json",
                "ec_max_num_scheduled_tokens": "1000000000000000000"
            }
        }' \
        >"$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

function start_pd() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m lm_service.entrypoints.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --transfer-protocol "tcp" \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_PD \
        --max-num-seqs $MAX_NUM_SEQS_PD \
        --enforce-eager \
        --ec-transfer-config '{
            "ec_connector":"ECMooncakeStorageConnector",
            "ec_role":"ec_consumer",
            "ec_connector_extra_config": {
                "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/consumer.json"
            }
        }' \
        >"$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

function start_all() {
    mkdir -p "$LOG_PATH"
    if [ -f "$PID_FILE" ]; then
        rm "$PID_FILE"
    fi

    if [ ! -d "$SHARED_STORAGE_PATH" ]; then
        mkdir -p "$SHARED_STORAGE_PATH"
    fi

    echo "Starting encoder workers..."
    for ((i=0; i<ENCODER_NUMBER; i++)); do
        dev_id=$((ENCODER_DEVICE_ID_BASE + i))
        address="${HOST_IP}:$((ENCODER_PORT_BASE + i))"
        log_file="$LOG_PATH/encoder_$i.log"
        start_encoder $dev_id $address $PROXY_ADDR $log_file
        echo "  Encoder worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "Starting prefill/decode workers..."
    for ((i=0; i<PD_NUMBER; i++)); do
        dev_id=$((PD_DEVICE_ID_BASE + i))
        address="${HOST_IP}:$((PD_PORT_BASE + i))"
        log_file="$LOG_PATH/prefill_decode_$i.log"
        start_pd $dev_id $address $PROXY_ADDR $log_file
        echo "  Prefill/decode worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "All workers starting. PIDs are stored in $PID_FILE."
}

function stop_all() {
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" > /dev/null 2>&1; then
                echo "Stopping process $pid"
                kill "$pid"
                for i in {1..5}; do
                    sleep 1
                    if ! kill -0 "$pid" > /dev/null 2>&1; then
                        break
                    fi
                done
                if kill -0 "$pid" > /dev/null 2>&1; then
                    echo "Process $pid did not exit, killing with -9"
                    kill -9 "$pid"
                fi
            fi
        done < "$PID_FILE"
        rm "$PID_FILE"
    else
        echo "No PID file found. Are the workers running?"
    fi

}

function print_help() {
    echo "Usage: $0 [--model MODEL] [--shared-storage-path PATH]
              [--gpu-utilization-encoder FLOAT] [--gpu-utilization-pd FLOAT]
              [--encoder-device-id-base INT] [--encoder-number INT]
              [--pd-device-id-base INT] [--pd-number INT]
              [--image-file-path PATH] [--log-path PATH]
              [--stop] [--help]"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --shared-storage-path) SHARED_STORAGE_PATH="$2"; shift ;;
        --gpu-utilization-encoder) GPU_UTILIZATION_ENCODER="$2"; shift ;;
        --gpu-utilization-pd) GPU_UTILIZATION_PD="$2"; shift ;;
        --encoder-device-id-base) ENCODER_DEVICE_ID_BASE="$2"; shift ;;
        --encoder-number) ENCODER_NUMBER="$2"; shift ;;
        --pd-device-id-base) PD_DEVICE_ID_BASE="$2"; shift ;;
        --pd-number) PD_NUMBER="$2"; shift ;;
        --log-path) LOG_PATH="$2"; shift ;;
        --image-file-path) IMAGE_FILE_PATH="$2"; shift ;;
        --stop) stop_all; exit 0 ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ]; then
    echo "Error: --model is required."
    exit 1
fi

if [ -z "$IMAGE_FILE_PATH" ]; then
    echo "Error: --image-file-path is required."
    exit 1
fi

start_all

chat_with_image() {
    python $CURRENT_DIR/chat_with_image.py \
        --proxy-addr $PROXY_ADDR \
        --transfer-protocol "tcp" \
        --encode-addr-list $(for ((i=0; i<ENCODER_NUMBER; i++)); do echo -n "${HOST_IP}:$((ENCODER_PORT_BASE + i)) "; done) \
        --pd-addr-list $(for ((i=0; i<PD_NUMBER; i++)); do echo -n "${HOST_IP}:$((PD_PORT_BASE + i)) "; done) \
        --model-name $MODEL \
        --image-path $IMAGE_FILE_PATH
}

chat_with_image
