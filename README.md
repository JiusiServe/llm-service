# LM-Service

LM-Service is a high-performance proxy designed for disaggregated serving of LLMs and LMMs, such as encoder disaggregation, prefill disaggregation, and more. It provides an OpenAI-compatible API for easy out-of-the-box use and offers a Python SDK for integration into custom serving frameworks.

<img width="1204" height="1440" alt="image" src="docs/architecture/lm_service_arch.drawio.svg" />

## Key Components

- HTTP Server: provides OpenAI-compatible APIs along with APIs for health checks and monitoring.
- Python SDK: offers asynchronous Python API support, designed to integrate into the customer's serving framework.
- Disaggregation Inference API Core: delivers high-performance implementations for both HTTP and Python APIs.
  - Disagg Proxy: coordinates different types of Disagg Workers for disaggregated serving.
  - Load Balancer: distributes workloads across Disagg Workers of the same type, supporting cache-aware routing policies and extendable advanced routing policies.
  - Worker Monitor: tracks all Disagg Workers, collects metrics, performs health checks, and manages the registration and deregistration of new workers.
  - Text Tokenizer: converts text into tokens, potentially optimized with a Rust/C++ implementation for higher performance.
  - Multimodal Preprocessor: handles preprocessing of multimodal data such as images, videos, and audio, loading data into memory efficiently.
  - Tool Call: includes a tool parser for parsing model outputs in specific formats and a tool caller for native and MCP tools.
- RPC Framework: establishes connections and facilitates communication between the Disagg Proxy and Disagg Workers.
  - HTTP: supports OpenAI-compatible APIs and native vLLM/SGLang HTTP APIs.
  - ZeroMQ: a high-performance RPC library for sending and receiving structured messages defined by llm-service, offering lower costs than HTTP, especially for in-memory multimodal data.
- Disagg Worker: performs disaggregated serving tasks as an encoder, prefill, decoder, or generator in the future, powered by inference engines like vLLM or SGLang.
- Distributed Cache Pool: caches data across different types of worker, typically featuring a hierarchical memory pool for efficient data management.
