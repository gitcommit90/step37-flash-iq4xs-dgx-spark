# Step-3.7-Flash IQ4_XS on NVIDIA DGX Spark (GB10)

Deployment package for running **[stepfun-ai/Step-3.7-Flash-GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF) (IQ4_XS, ~105 GB)** on a single **NVIDIA DGX Spark (GB10, 128 GB unified memory)** with llama.cpp.

This is a **deployment package, not new weights**. The GGUF is StepFun's official imatrix-calibrated IQ4_XS quant. This repo pins the runtime build, the exact measured serve command, and honest benchmark results.

## Goal of this package

**Full 256K context window, as fast as possible on one GB10.**

Default serve is **single-slot, 262144 context**, not multi-slot short context. Concurrent multi-request is optional and trades per-stream context.

## Model

- **Step-3.7-Flash**: 198B-parameter sparse MoE (~11B active/token), vision-language, 256K context, three reasoning levels (`low` / `medium` / `high`), agentic / tool-calling tuned. Apache-2.0.
- **Quant**: IQ4_XS, 3 shards, ~105 GB on disk.
- **Runtime**: llama.cpp built from StepFun's fork branch [`stepfun-ai/llama.cpp@step3.7`](https://github.com/stepfun-ai/llama.cpp/tree/step3.7) (commit `8f34864`). **Stock upstream llama.cpp is not the recommended runtime for this model** — use the vendor branch.

## Measured performance (GB10, this package)

### A) Full 256K single-stream (`llama-batched-bench`)

Config: `-c 262272 -b 2048 -ub 1024 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -npl 1`, TG=128.

| PP | TG | N_KV | PP t/s | **TG t/s** | Peak GPU (process) |
|---:|---:|-----:|-------:|-----------:|-------------------:|
| 0 | 128 | 128 | — | **27.24** | ~100–105 GB during run |
| 2048 | 128 | 2176 | 664.7 | **26.42** | |
| 8192 | 128 | 8320 | 784.3 | **25.63** | |
| 16384 | 128 | 16512 | 762.7 | **24.35** | |
| 32768 | 128 | 32896 | 721.0 | **22.24** | |
| 65536 | 128 | 65664 | 653.2 | **18.79** | |
| 131072 | 128 | 131200 | 552.8 | **14.09** | |
| **262144** | 128 | **262272** | 422.6 | **9.92** | **~105.2 GB peak** |

Raw log: `batched-bench-256k-q4kv.log`.

**vs StepFun card (same quant / GB10, their published table):** short-ctx TG ~23.9 t/s, 256K TG ~8.6 t/s. This package is **~equal or slightly faster** at both ends with q4_0 KV.

### B) Live server (default package serve)

- Flags: `--ctx-size 262144 --parallel 1 --batch-size 2048 --ubatch-size 1024 --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99`
- Slot log: `n_slots = 1`, **`n_ctx = 262144`**
- Idle process GPU: **~104.4 GB** (`nvidia-smi`)
- Host after load: ~107 / 121 GiB used, ~14 GiB available
- Coherence sample (`reasoning_effort=low`): content `399` for 19×21; server decode **~28.2 t/s** at empty/short context

### C) Earlier multi-slot experiment (not the default)

`--ctx-size 65536 --parallel 10` → **only ~6.6k per stream**. Concurrent 10/10 worked (~27 t/s @1, ~67 aggregate @10) but **does not meet the 256K goal**. Kept only as historical `bench_results.json`.

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

### 2. Download the weights (~105 GB)

```bash
hf download stepfun-ai/Step-3.7-Flash-GGUF \
  --include "IQ4_XS/*" --local-dir ~/llm/step37-flash-iq4xs
```

### 3. Serve (256K single-stream — default)

```bash
./start.sh
# equivalent:
llama-server \
  --model ~/llm/step37-flash-iq4xs/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8088 \
  --ctx-size 262144 --parallel 1 \
  --batch-size 2048 --ubatch-size 1024 \
  --n-gpu-layers 99 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --metrics --no-webui
```

First load ~6–8 minutes. API answers `503 Loading model` until ready.

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

- **Reasoning effort**: `chat_template_kwargs.reasoning_effort` = `low` | `medium` | `high`. Default thinking is ON; budget tokens or expect long `reasoning_content`.
- **Speed vs filled context**: short ctx ~27–28 t/s decode; full 256K filled ~**10 t/s** TG (attention cost). That is the humanly-achievable envelope on one GB10 for this quant without MTP.
- **MTP draft heads** (`Step3.7-flash-mtp-*.gguf`) exist on HF but are **not** enabled in this package yet.
- **Vision**: optional `mmproj` not in default serve.

## Files

| File | Role |
|---|---|
| `start.sh` / `stop.sh` | 256K single-stream measured serve |
| `batched-bench-256k-q4kv.log` | full PP/TG table through 256K |
| `bench_concurrent.py` + `bench_results.json` | earlier multi-slot experiment |
| `.gitignore` | weights/caches/logs out of git |

## Hardware / environment

- NVIDIA DGX Spark (GB10, SM121), 128 GB unified memory, aarch64
- CUDA toolkit at `/usr/local/cuda`, llama.cpp `step3.7` @ `8f34864`
- Peak process GPU during 256K bench: **~105 GB**

## License

Package scripts: MIT. Model weights: Apache-2.0 (StepFun).
