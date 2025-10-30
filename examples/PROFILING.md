# Profiling Example

This example demonstrates how to use the `start_profile` and `stop_profile` APIs to profile inference requests in the disaggregated vLLM service.

## Overview

The profiling functionality allows you to capture detailed performance traces during model inference. These traces can be used to:

- Identify performance bottlenecks
- Analyze GPU utilization
- Optimize inference performance
- Debug latency issues

## Prerequisites

1. Disaggregated workers must be running with appropriate configuration
2. Set the `VLLM_TORCH_PROFILER_DIR` environment variable to specify where profiling traces should be saved

## Usage

```bash
# Set the profiling output directory
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"

# Run the profiling example
python examples/profiling_example.py \
    --proxy-addr /tmp/proxy.sock \
    --encode-addr-list /tmp/encode1.sock /tmp/encode2.sock \
    --pd-addr-list /tmp/pd1.sock /tmp/pd2.sock \
    --model-name meta-llama/Meta-Llama-3-8B
```

## API Reference

### `Proxy.start_profile()`

Starts profiling on all PD (Prefill-Decode) workers.

```python
await proxy.start_profile()
```

### `Proxy.stop_profile()`

Stops profiling on all PD workers and writes trace files to disk.

```python
await proxy.stop_profile()
```

## Visualizing Traces

After profiling completes, trace files will be saved in the directory specified by `VLLM_TORCH_PROFILER_DIR`. You can visualize these traces using:

1. **Perfetto UI**: Upload trace files at https://ui.perfetto.dev/
2. **Chrome Tracing**: Open `chrome://tracing` in Chrome browser

## Additional Configuration

You can enable additional profiling details using environment variables:

- `VLLM_TORCH_PROFILER_RECORD_SHAPES`: Record tensor shapes
- `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY`: Profile memory usage
- `VLLM_TORCH_PROFILER_WITH_STACK`: Include Python stack traces
- `VLLM_TORCH_PROFILER_WITH_FLOPS`: Include FLOPS estimates

Example:

```bash
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
```

## Notes

- Profiling adds significant overhead and should only be used for debugging/optimization
- Profile only a small number of requests to keep trace file sizes manageable
- The `stop_profile()` operation may take several minutes for large models as it writes traces to disk
