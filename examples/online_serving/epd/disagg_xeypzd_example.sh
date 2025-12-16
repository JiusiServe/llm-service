#!/usr/bin/env bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=""
SHARED_STORAGE_PATH="/dev/shm/epd"
PID_FILE="${PID_FILE:-${CURRENT_DIR}/pid.txt}"


# Encoder default config
MAX_NUM_SEQS_ENCODER="${MAX_NUM_SEQS_ENCODER:-1}"
GPU_UTILIZATION_ENCODER=0.0
ENCODER_ADDR_PREFIX="${ENCODER_ADDR_PREFIX:-/tmp/encoder}"
ENCODER_DEVICE_ID_BASE=0
ENCODER_NUMBER=1

# Prefill default config
MAX_NUM_SEQS_PREFILL="${MAX_NUM_SEQS_PREFILL:-128}"
GPU_UTILIZATION_PREFILL=0.95
PREFILL_ADDR_PREFIX="${PREFILL_ADDR_PREFIX:-/tmp/prefill}"
PREFILL_DEVICE_ID_BASE=1
PREFILL_NUMBER=1

# Decoder default config
MAX_NUM_SEQS_DECODER="${MAX_NUM_SEQS_DECODER:-128}"
GPU_UTILIZATION_D=0.95
DECODER_ADDR_PREFIX="${DECODER_ADDR_PREFIX:-/tmp/decoder}"
DECODER_DEVICE_ID_BASE=2
DECODER_NUMBER=1

# Proxy default config
PROXY_ADDR="${PROXY_ADDR:-/tmp/proxy}"

LOG_PATH="${CURRENT_DIR}/logs"
IMAGE_FILE_PATH=""

# Reduce memory fragmentation
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

function start_encoder() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m lm_service.entrypoints.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_ENCODER \
        --max-num-seqs $MAX_NUM_SEQS_ENCODER \
        --enforce-eager \
        --no-enable-prefix-caching \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
            }
        }' \
        >"$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

function start_prefill() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m lm_service.entrypoints.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_PREFILL \
        --max-num-seqs $MAX_NUM_SEQS_PREFILL \
        --enforce-eager \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
            }
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorStoreV1",
            "kv_role": "kv_producer",
            "mooncake_rpc_port": "50053",
            "kv_connector_extra_config": {
                "local_hostname": "localhost",
                "metadata_server": "http://localhost:8083/metadata",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "localhost:50053",
                "global_segment_size": 30000000000
            }
        }' \
        >"$log_file" 2>&1 &
    
    echo $! >> "$PID_FILE"
}

function start_decoder() {
    local dev_id=$1
    local address=$2
    local proxy_address=$3
    local log_file=$4

    VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=$dev_id python -m lm_service.entrypoints.worker \
        --proxy-addr $proxy_address \
        --worker-addr $address \
        --model $MODEL \
        --gpu-memory-utilization $GPU_UTILIZATION_D \
        --max-num-seqs $MAX_NUM_SEQS_DECODER \
        --enforce-eager \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorStoreV1",
            "kv_role": "kv_consumer",
            "mooncake_rpc_port": "50053",
            "kv_connector_extra_config": {
                "local_hostname": "localhost",
                "metadata_server": "http://localhost:8083/metadata",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "localhost:50053",
                "global_segment_size": 30000000000
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
        address="${ENCODER_ADDR_PREFIX}_$i"
        log_file="$LOG_PATH/encoder_$i.log"
        start_encoder $dev_id $address $PROXY_ADDR $log_file
        echo "  Encoder worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "Starting prefill workers..."
    for ((i=0; i<PREFILL_NUMBER; i++)); do
        dev_id=$((PREFILL_DEVICE_ID_BASE + i))
        address="${PREFILL_ADDR_PREFIX}_$i"
        log_file="$LOG_PATH/prefill_$i.log"
        start_prefill $dev_id $address $PROXY_ADDR $log_file
        echo "  Prefill worker $i starting on device $dev_id, address: $address, log: $log_file"
    done

    echo "Starting decode workers..."
    for ((i=0; i<DECODER_NUMBER; i++)); do
        dev_id=$((DECODER_DEVICE_ID_BASE + i))
        address="${DECODER_ADDR_PREFIX}_$i"
        log_file="$LOG_PATH/decoder_$i.log"
        start_decoder $dev_id $address $PROXY_ADDR $log_file
        echo "  Decode worker $i starting on device $dev_id, address: $address, log: $log_file"
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
              [--gpu-utilization-encoder FLOAT] [--encoder-device-id-base INT] [--encoder-number INT]
              [--gpu-utilization-prefill FLOAT] [--prefill-device-id-base INT] [--prefill-number INT]
              [--gpu-utilization-decode FLOAT] [--decode-device-id-base INT] [--decode-number INT]
              [--image-file-path PATH] [--log-path PATH]
              [--stop] [--help]"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --shared-storage-path) SHARED_STORAGE_PATH="$2"; shift ;;
        --gpu-utilization-encoder) GPU_UTILIZATION_ENCODER="$2"; shift ;;
        --encoder-device-id-base) ENCODER_DEVICE_ID_BASE="$2"; shift ;;
        --encoder-number) ENCODER_NUMBER="$2"; shift ;;
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
        --encode-addr-list $(for ((i=0; i<ENCODER_NUMBER; i++)); do echo -n "${ENCODER_ADDR_PREFIX}_$i "; done) \
        --p-addr-list $(for ((i=0; i<PREFILL_NUMBER; i++)); do echo -n "${PREFILL_ADDR_PREFIX}_$i "; done) \
        --d-addr-list $(for ((i=0; i<DECODER_NUMBER; i++)); do echo -n "${DECODER_ADDR_PREFIX}_$i "; done) \
        --model-name $MODEL \
        --image-path $IMAGE_FILE_PATH
}

chat_with_image
