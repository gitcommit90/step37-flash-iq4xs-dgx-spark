# Step-3.7-Flash IQ4_XS on NVIDIA DGX Spark (GB10)

Deployment package for running **[stepfun-ai/Step-3.7-Flash-GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF) (IQ4_XS, ~105 GB)** on a single **NVIDIA DGX Spark (GB10, 128 GB unified memory)** with llama.cpp.

This is a **deployment package, not new weights**. The GGUF is StepFun's official imatrix-calibrated IQ4_XS quant. This repo pins the runtime build, the exact measured serve command, and honest benchmark results.

## Model

- **Step-3.7-Flash**: 198B-parameter sparse MoE (~11B active/token), vision-language, 256K context, three reasoning levels (`low` / `medium` / `high`), agentic / tool-calling tuned. Apache-2.0.
- **Quant**: IQ4_XS, 3 shards, ~105 GB on disk.
- **Runtime**: llama.cpp built from StepFun's fork branch [`stepfun-ai/llama.cpp@step3.7`](https://github.com/stepfun-ai/llama.cpp/tree/step3.7) (commit `8f34864`). **Stock upstream llama.cpp is not the recommended runtime for this model** — use the vendor branch.

## Measured performance (GB10, this package)

Serve config: 64K ctx, `--parallel 10 --cont-batching`, `--flash-attn on`, KV cache `q4_0/q4_0`, all layers on GPU (`-ngl 99`). Prompt: technical explainer, `max_tokens=640`, `reasoning_effort=low`, streaming, warm.

| Concurrency | Success | Avg TTFT | Per-stream tok/s | Aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1  | 1/1  | 0.37 s | **27.1** | 27.1 |
| 4  | 4/4  | 0.54 s | 13.4 | 53.4 |
| 10 | 10/10 | 0.84 s | 7.1 | **67.6** |

- **Concurrent 10: achieved** (10/10 requests succeeded, no server crash).
- Single-stream decode measured by server timings (non-stream, thinking enabled): **30.2–30.8 tok/s** at 32K ctx.
- Reference: StepFun's own GB10 card bench for IQ4_XS reports ~24 tok/s TG at short ctx with much larger KV footprints; our 64K/q4_0-KV config measures faster.
- TTFT above is time to first **reasoning** delta (thinking is on by default). With default thinking, expect a few hundred reasoning tokens before `content` at `low` effort; more at `medium`/`high`.
- `bench_results.json` has the raw per-request data.

## Quickstart

### 1. Build the runtime (once)

```bash
git clone --depth 1 --branch step3.7 https://github.com/stepfun-ai/llama.cpp.git llama.cpp-step37
cd llama.cpp-step37
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build --config Release -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES=121` targets GB10 (SM121). Build takes ~5 min on the Spark.

### 2. Download the weights (~105 GB)

```bash
hf download stepfun-ai/Step-3.7-Flash-GGUF \
  --include "IQ4_XS/*" --local-dir ~/llm/step37-flash-iq4xs
```

### 3. Serve

```bash
./start.sh        # downloads nothing; assumes weights + build above
# or directly:
/home/tux/llama.cpp-step37/build/bin/llama-server \
  --model ~/llm/step37-flash-iq4xs/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8088 \
  --ctx-size 65536 --parallel 10 --cont-batching \
  --n-gpu-layers 99 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --metrics --no-webui
```

First load takes **~6–7 minutes** (105 GB read + fit). The API answers `503 "Loading model"` until ready — poll `/v1/models`, don't restart.

### 4. Chat

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Step-3.7-flash-IQ4_XS-00001-of-00003.gguf",
    "messages": [{"role":"user","content":"Write a Python fibonacci function."}],
    "max_tokens": 1200,
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }'
```

Notes:

- **Reasoning effort**: pass `chat_template_kwargs.reasoning_effort` = `low` | `medium` | `high`. Default thinking is ON; short `max_tokens` can be consumed entirely by `reasoning_content` with empty `content`. Give it budget (≥600 for low) or expect reasoning-only replies.
- **Vision**: download `mmproj-step3.7-flash-f16.gguf` from the same HF repo and add `--mmproj <path>` (not covered by this bench).
- **MTP draft heads** (`Step3.7-flash-mtp-*.gguf`) exist in the repo but were **not** used in this package.

## Files

| File | Role |
|---|---|
| `start.sh` / `stop.sh` | launch / stop the measured serve config |
| `bench_concurrent.py` | streaming concurrent benchmark (1/4/10) |
| `bench_results.json` | raw measured results from GB10 |
| `.gitignore` | keeps weights/caches/logs out of git |

## Hardware / environment

- NVIDIA DGX Spark (GB10, SM121), 128 GB unified memory, aarch64
- Ubuntu, CUDA toolkit at `/usr/local/cuda`, llama.cpp `step3.7` branch @ `8f34864`
- Host RAM during serve: ~110 GB used of 121 GB (weights + 64K KV q4_0 × 10 slots)

## License

Package scripts: MIT. Model weights: Apache-2.0 (StepFun). See the HF repo for calibration-data licenses.
