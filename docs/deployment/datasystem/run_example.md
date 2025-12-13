# openYuanrong datasystem 快速使用指南

## 概述
openYuanrong datasystem 是一个分布式缓存系统，利用计算集群的 HBM/DRAM/SSD 资源构建近计算多级缓存，提升模型训练及推理、大数据、微服务等场景数据访问性能。

## 环境要求
操作系统：openEuler 22.03 或更高版本
CANN：8.2.rc1 或更高版本
Python：3.9–3.11
etcd：3.5.12 或更高版本

## 部署 etcd
### 安装
1. 下载二进制包（参考 [etcd GitHub Releases](https://github.com/etcd-io/etcd/releases)）：
```bash
ETCD_VERSION="v3.5.12"
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz
```
2. 解压并安装：
```bash
tar -xvf etcd-${ETCD_VERSION}-linux-amd64.tar.gz
cd etcd-${ETCD_VERSION}-linux-amd64
sudo cp etcd etcdctl /usr/local/bin/
```
3. 验证安装：
```bash
etcd --version
etcdctl version
```
如果能输出版本号说明安装成功。

### 启动集群
> 提示：以下为最小化单节点部署示例。生产环境请参考 [官方集群部署文档](https://etcd.io/docs/current/op-guide/clustering/)。
1. 启动单节点 etcd 集群，并设置任意空闲端口（如 2379 和 2380 ）：
```bash
etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster etcd-single=http://0.0.0.0:2380 &
```
参数说明：
- `--name`：集群节点名称。
- `--data-dir`：数据存储目录。
- `--listen-client-urls`：客户端监听地址（0.0.0.0 允许任意 IP 访问）。
- `--advertise-client-urls`：对外暴露的客户端地址。
- `--listen-peer-urls`：集群节点间监听地址。
- `--initial-advertise-peer-urls`：对其他节点暴露的地址。
- `--initial-cluster`：初始节点列表，格式：节点名=节点peerURL。

2. 验证 etcd 是否正常运行：
```bash
etcdctl --endpoints "127.0.0.1:2379" put key "value"
etcdctl --endpoints "127.0.0.1:2379" get key
```
若能成功写入并读取，表示 etcd 已正确部署。

## 部署 openYuanrong datasystem

### 安装
#### pip 方式安装
使用 pip 安装预编译 wheel 包：
如果使用 Python 3.9，运行：
```bash
pip install https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/release/0.6.0/linux/aarch64/openyuanrong_datasystem-0.6.0-cp39-cp39-manylinux_2_34_aarch64.whl
```

如果使用 Python 3.10，运行：
```bash
pip install https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/release/0.6.0/linux/aarch64/openyuanrong_datasystem-0.6.0-cp310-cp310-manylinux_2_34_aarch64.whl
```

如果使用 Python 3.11，运行：
```bash
pip install https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/release/0.6.0/linux/aarch64/openyuanrong_datasystem-0.6.0-cp311-cp311-manylinux_2_34_aarch64.whl
```

#### 源码编译方式安装
使用源码编译方式安装 openYuanrong datasystem。Linux 环境请参考 [源码编译文档](https://gitee.com/openeuler/yuanrong-datasystem/blob/master/docs/source_zh_cn/installation/installation_linux.md)。

### 启动集群
安装完成后，即可通过随包自带的 `dscli` 命令行工具一键完成集群部署。
替换 `${ETCD_IP}` 为 etcd 所在节点的 IP， `${WORKER_IP_N}` 为所在节点 N 的 IP，在每个节点启动一个监听端口号为 31501 的服务端进程：
```bash
dscli start -w \
--worker_address "${WORKER_IP_N}:31501" \
--etcd_address "${ETCD_IP}:2379" \
```

### 停止集群
替换 `${WORKER_IP_N}` 为所在节点 N 的 IP。在每个节点执行以下命令：
```bash
dscli stop --worker_address "${WORKER_IP_N}:31501"
```

## 配置 VLLM 使用 Yuanrong Connector
Datasystem 支持通过 ECMooncakeStorageConnector（用于 EC 传输）和 YuanRongConnector（用于 KVC 传输）与 VLLM 对接。
先设置基础环境变量，替换 `${WORKER_IP}` 为所在节点的 IP，在每个 VLLM 节点执行以下命令：
```bash
export DS_WORKER_ADDR="${WORKER_IP}:31501"
export EC_STORE_TYPE="datasystem"
export USING_PREFIX_CONNECTOR=0
```

### 仅使用 EC Connector（1E1PD 架构）
Encoder 节点
```bash
vllm serve Qwen/Qwen3-8B \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeStorageConnector",
    "ec_role": "ec_producer"
  }'
```

Prefill-Decoder 节点
```bash
vllm serve Qwen/Qwen3-8B \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeStorageConnector",
    "ec_role": "ec_producer"
  }'
```

### 同时启用 EC Connector 与 KV Connector（1E1P1D 架构）
Encoder 节点
```bash
vllm serve Qwen/Qwen3-8B \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeStorageConnector",
    "ec_role": "ec_producer"
  }'
```

Prefill 节点
```bash
vllm serve Qwen/Qwen3-8B \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeStorageConnector",
    "ec_role": "ec_consumer"
  }' \
  --kv-transfer-config '{
    "kv_connector": "YuanRongConnector",
    "kv_role": "kv_producer"
  }'
```

Decoder 节点
```bash
vllm serve Qwen/Qwen3-8B \
  --kv-transfer-config '{
    "kv_connector": "YuanRongConnector",
    "kv_role": "kv_consumer"
  }'
```
