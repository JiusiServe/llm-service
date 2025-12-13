# API接口文档

本服务接口遵循 RESTful 设计，提供标准 HTTP 状态码返回。  
数据交互格式为 JSON（除 `/metrics` 及静态文档资源）。

---

## 目录

- [启动命令](#启动命令)
- [开放文档与元数据](#开放文档与元数据)
- [核心API](#核心api)
- [健康检查](#健康检查)
- [监控与其它接口](#监控与其它接口)
- [未实现接口](#未实现接口)

---

## 启动命令

多个proxy需要启动多个http server

### 非Redis启动命令

```bash

VLLM_USE_V1=1 python -m lm_service.entrypoints.openai.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --proxy-addr /tmp/proxy \
    --port 5580 \
    --host 0.0.0.0 \
    --encode-addr-list  /tmp/encoder_0 \
    --pd-addr-list /tmp/prefill_decode_0 \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/

```

### Redis启动命令

```bash

VLLM_USE_V1=1 python -m lm_service.entrypoints.openai.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --port 5580 \
    --host 0.0.0.0 \
    --metastore-client-config '{
        "metastore_client": "RedisMetastoreClient",
        "metastore_address": "redis://127.0.0.1:6379/0"
        }' \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/

```

---

## 开放文档与元数据

| 路径                  | 方法   | 功能                  |
| --------------------- | ------ | --------------------- |
| /openapi.json         | GET    | OpenAPI元数据（API描述信息） |
| /docs                 | GET    | Swagger UI可视化API文档 |
| /docs/oauth2-redirect | GET    | Swagger OAuth2重定向页面 |
| /redoc                | GET    | ReDoc可视化API文档    |

---

## 核心API

### 1. 聊天接口

- **路径**：`/v1/chat/completions`
- **方法**：POST
- **描述**：基于对话历史生成LLM回复
- **请求体举例**：

    ```bash

    curl -X POST  http://127.0.0.1:5580/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "/workspace/models/Qwen2.5-VL-7B-Instruct",
        "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///workspace/l00807937/EPD_Timecount_v0.11.0/image/work.jpg"}},
            {"type": "text", "text": "What is the text in the illustrate?"}
        ]}
        ]
        }'

    ```

- **响应体举例**：

    ```json

    {
      "id": "xxx",
      "object": "chat.completion",
      "created": 1234567890,
      "choices": [
        {
          "index": 0,
          "message": {
              "role": "assistant",
              "content": "你好！我是AI助手。"
           },
          "finish_reason": "stop"
        }
      ],
      "usage": { "prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22 }
    }

    ```

---

### 2. 传统补全接口

- **路径**：`/v1/completions`
- **方法**：POST
- **描述**：普通文本补全（非对话）。
- **请求体举例**：

    ```json

    {
      "model": "your-model-id",
      "prompt": "Once upon a time",
      "max_tokens": 16
    }

    ```

- **响应体举例**：

    ```json

    {
      "id": "xxx",
      "object": "text_completion",
      "created": 1234567890,
      "choices": [
        {
          "text": " there lived a wise king.",
          "index": 0,
          "finish_reason": "length"
        }
      ],
      "usage": { "prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10 }
    }

    ```

---

## 健康检查

- **路径**：`/check_health`
- **方法**：GET
- **描述**：服务健康状态自检。
- **成功响应**：

    ```json

    {"status": "ok"}

    ```

---

## 监控与其它接口

- **路径**：`/metrics`
- **方法**：GET
- **描述**：返回模型或服务实例各项运行指标。

    **响应示例：**

    ```text

    Server Type: PD_INSTANCE
      Address: ipc:///tmp/prefill_decode_0
        ec_role: PD_INSTANCE
        addr: ipc:///tmp/prefill_decode_0
        Avg e2e time requests: 4198.529 ms
        ...

    ```

---

## 未实现接口

- **路径**：`/abort`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

- **路径**：`/start_profile`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

- **路径**：`/stop_profile`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

---
