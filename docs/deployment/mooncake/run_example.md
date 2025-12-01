# 快速使用指南

## 编译

Software

```
  * Python >= 3.9, < 3.12
  * CANN >= 8.3.rc1
  * PyTorch == 2.7.1, torch-npu == 2.7.1
  * vLLM：v0.11.0 branch
  * vLLM-Ascend：v0.11.0 branch
```  


Mooncake 目前使用 JiusiServe 仓 v6_support_v0.3.7.post2 分支，依赖的 yalantinglibs 组件，目前使用 main 分支。

完整编译脚本如下

```bash
#!/bin/bash

BUILD_ROOT="$(dirname "$(realpath "$0")")"
cd $BUILD_ROOT

# Check for root permissions
if [ $(id -u) -ne 0 ]; then
    echo "Require root permission, try sudo ./build.sh"
    exit 1
fi

# Install mooncake
MOONCAKE_PATH=${2:-$BUILD_ROOT/Mooncake}
git clone https://github.com/JiusiServe/Mooncake.git

cd ${MOONCAKE_PATH}
git reset --hard
git clean -f
git checkout v6_support_v0.3.7.post2

# Determine the package manager (Ubuntu vs CentOS)
if command -v apt-get >/dev/null 2>&1; then
    PACKAGE_MANAGER="apt-get"
    SYSTEM_PACKAGES="librdmacm-dev \
        libgflags-dev \
        libyaml-cpp-dev \
        libgtest-dev \
        libjsoncpp-dev \
        libunwind-dev \
        libnuma-dev \
        libboost-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libssl-dev \
        libgrpc-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libcurl4-openssl-dev \
        libhiredis-dev \
        patchelf"
elif command -v yum >/dev/null 2>&1; then
    PACKAGE_MANAGER="yum"
    SYSTEM_PACKAGES="rdma-core-devel \
        gflags-devel \
        yaml-cpp-devel \
        gtest-devel \
        jsoncpp-devel \
        libunwind-devel \
        numactl-devel \
        boost-devel \
        boost-system \
        boost-thread \
        openssl-devel \
        grpc-devel \
        protobuf-devel \
        protobuf-compiler \
        libcurl-devel \
        hiredis-devel \
        patchelf"
else
    echo "Neither apt-get nor yum found. Unsupported system."
    exit 1
fi

# Install system packages based on the package manager
echo "Installing system dependencies..."
if [ "$PACKAGE_MANAGER" == "apt-get" ]; then
    apt-get update
    apt-get install -y $SYSTEM_PACKAGES
elif [ "$PACKAGE_MANAGER" == "yum" ]; then
    yum install -y $SYSTEM_PACKAGES
fi

# Install yalantinglibs
echo "Installing yalantinglibs"

if [ ! -d "${MOONCAKE_PATH}/thirdparties" ]; then
    mkdir -p "${MOONCAKE_PATH}/thirdparties"
fi

cd "${MOONCAKE_PATH}/thirdparties"
if [ -d "yalantinglibs" ]; then
    echo -e "yalantinglibs directory already exists. Removing for fresh install..."
    rm -rf yalantinglibs
fi

git config --global http.sslVerify false
git clone https://github.com/JiusiServe/yalantinglibs.git

cd yalantinglibs
git checkout main
mkdir -p build
cd build

cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build . -j$(nproc)
cmake --install .

# Install glog
echo "Installing glog"
cd "${MOONCAKE_PATH}/thirdparties"
git clone https://github.com/google/glog.git

cd glog
git checkout v0.7.1

cmake -DWITH_GTEST=OFF -S . -B build -G "Unix Makefiles"
cmake --build build --target install -j$(nproc)

# Check if .gitmodules exists
if [ -f "${MOONCAKE_PATH}/.gitmodules" ]; then
    FIRST_SUBMODULE=$(grep "path" ${MOONCAKE_PATH}/.gitmodules | head -1 | awk '{print $3}')
    cd "${MOONCAKE_PATH}"

    if [ -d "${MOONCAKE_PATH}/${FIRST_SUBMODULE}/.git" ] || [ -f "${MOONCAKE_PATH}/${FIRST_SUBMODULE}/.git" ]; then
        echo -e "Git submodules already initialized. Skipping..."
    else
        git submodule update --init
    fi
else
    echo -e "No .gitmodules file found. Skipping..."
    exit 1
fi

# Build the project
if [ ! -d "${MOONCAKE_PATH}/build" ]; then
    mkdir -p "${MOONCAKE_PATH}/build"
fi
cd ${MOONCAKE_PATH}/build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
make -j"$((($(nproc) - 2)))"

make install
ldconfig

echo "Build and installation completed successfully."
```

## 启动Mooncake Master

```bash
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
mooncake_master
  --rpc_port 50051
  --enable_http_metadata_server=true
  --http_metadata_server_host=0.0.0.0
  --http_metadata_server_port=8081
  --metrics_port 9004
  --rpc_thread_num 8
  --eviction_ratio 0.05
  --eviction_high_watermark_ratio 0.9
  >"mooncake_master.log" 2>&1 &
```

各个参数的含义如下
- `local_hostname`表示本机的IP:Port或者可访问的域名（若不含端口则使用默认值）
- `enable_http_metadata_server`为true表示开启http元数据服务
- `http_metadata_server_host`指定HTTP元数据服务器绑定的主机地址。使用`"0.0.0.0"`可监听所有的可用网络接口，或指定特定IP
- `http_metadata_server_host`指定HTTP元数据服务器监听的TCP端口。该端口不可与其他服务冲突。
- `metrics_port`指定master通过HTTP暴露的Prometheus风格的监控指标。
- 根据可用的CPU核心数和工作负载，调整`rpc_thread_num`.
- 根据内存压力和对象更替情况调整`eviction_high_watermark_ratio`和`eviction_ratio`。

使用IPV6启动Mooncake Master

```bash
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
export MC_USE_IPV6=1
mooncake_master
  --rpc_port 50051
  --rpc_address=::
  --enable_http_metadata_server=true
  --http_metadata_server_host=::
  --http_metadata_server_port=8080
  --metrics_port 9003
  --rpc_thread_num 8
  --eviction_ratio 0.05
  --eviction_high_watermark_ratio 0.9
  >"mooncake_master.log" 2>&1 &
```

注意

- 如果需要指定网卡，避免使用自动探索网卡功能，可设置环境变量`export MC_MS_AUTO_DISC=0`。

- 建议通过设置环境变量`export MC_TCP_BIND_ADDRESS`指定RPC IP，避免自动连接到非预期IP。

- 可通过设置`--default_kv_lease_ttl"自定义租约超时时间，默认值为5000，单位为ms。

## 配置VLLM参数使用Mooncake Connector
### 使用EC Connector
### 1E1PD Encoder实例示例

- 使用ipv4时，启动模型时添加如下配置

$HOST_IP为本机IP， $MOONCAKE_MASTER_IP和$MOONCAKE_MASTER_PORT为mooncake master所在节点IP和port，
$MOONCAKE_METADATA_IP和$MOONCAKE_METADATA_PORT为元数据服务所在节点IP和port。

```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "transfer_timeout": 5,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

### 1E1PD PD实例示例

```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_consumer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 0,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "transfer_timeout": 5,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

- 使用ipv6时，注意添加中括号

```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer", # PD实例需要修改为ec_consumer
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://['$MOONCAKE_METADATA_IP']:'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "['$MOONCAKE_MASTER_IP']:'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "transfer_timeout": 5,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

### 使用EC connector和KV connector
### 1E1P1D Encoder实例

- 使用ipv4时，启动模型时添加如下配置
$HOST_IP为本机IP， $MOONCAKE_MASTER_IP和$MOONCAKE_MASTER_PORT为mooncake master所在节点IP和port，
$MOONCAKE_METADATA_IP和$MOONCAKE_METADATA_PORT为元数据服务所在节点IP和port。

- 根据网络延迟，可修改`transfer_timeout`mooncake传输超时等待时间，单位为s。

```bash
 --ec-transfer-config '{
     "ec_connector":"ECMooncakeStorageConnector",
     "ec_role":"ec_producer",
     "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 32212254720,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "transfer_timeout": 5,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1,
        "ec_max_num_scheduled_tokens": "1000000000000000000"
    }
 }'
```

### 1E1P1D P实例示例

```bash
 --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorStoreV1",
    "kv_role": "kv_producer",
    "mooncake_rpc_port":"0",
    "kv_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "global_segment_size": 30000000000
    }
 }'
 --ec-transfer-config '{
    "ec_connector":"ECMooncakeStorageConnector",
    "ec_role":"ec_consumer",
    "ec_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "global_segment_size": 0,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "replica_num": 1,
        "transfer_timeout": 5,
        "fast_transfer": true,
        "fast_transfer_buffer_size": 1
   }
 }'
```

### 1E1P1D D实例示例

```bash
 --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorStoreV1",
    "kv_role": "kv_consumer",
    "mooncake_rpc_port":"1",
    "kv_connector_extra_config": {
        "local_hostname": "'$HOST_IP'",
        "metadata_server": "http://'$MOONCAKE_METADATA_IP':'$MOONCAKE_METADATA_PORT'/metadata",
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "'$MOONCAKE_MASTER_IP':'$MOONCAKE_MASTER_PORT'",
        "global_segment_size": 30000000000
     }
 }'
```
