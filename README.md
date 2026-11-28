# llama-notes

1. Install nvidia drivers, cuda-toolkit (https://developer.nvidia.com/cuda-downloads).

2. Compile llama.cpp (https://github.com/ggml-org/llama.cpp) with command:
```
cmake -B "~/llama.cpp/build" -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DGGML_LTO=ON -DGGML_CPU_KLEIDIAI=OFF -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_PDL=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120

-B - build directory
-G - generator
-DGGML_LTO - link-time optimization (supposedly, increases performance at runtime)
-DGGML_CPU_KLEIDIAI - optimization for ARM
-DGGML_CUDA - make use of CUDA
-DGGML_NATIVE - build for native GPU architecture 
-DGGML_CUDA_PDL - Programmatic Dependent Launch (PDL), optimization for modern NVidia cards
-DGGML_CUDA_GRAPHS - optimization for CUDA driver calls (chain of operations called as graphs)
-DGGML_CUDA_FA_ALL_QUANTS - flash attention built for all types of KV-cache quants (better to use DGGML_CUDA_FA_QUANTS with exact quants like q8_0)
-DCMAKE_BUILD_TYPE - release to exclude debugging stuff
-DCMAKE_CUDA_ARCHITECTURES - 120 means NVidia Blackwell
```

3. Build llama with command:
```
cmake --build build --config Release

--build - path to the built llama (from the previous step)
--config - Release to exclude debugging stuff.
```

4. Load the model with command (url in the output can be used for direct chat with the model) - optimized for 16GB VRAM:
Common sets of params for different tasks (model independent):
* problem solving: --temp 0.3 --min-p 0.05 --top-p 0.9 --top-k 40
* architecture: --temp 0.6 --min-p 0.1 --top-p 0.95
* implementation: --temp 0.1 --min-p 0.02 --top-p 1.0
* chat/writing: --temp 0.8 --min-p 0.05 --top-p 0.95
Parameters --presence-penalty 0.0 --repeat-penalty 1.0 --frequency-penalty 0.0 - default `turned off` (defense from cycling). Non default values slow models down up to 50%.
```
# Gemma 4 26B A4B. MOE.
## With expert offload, good quant - 50 t/s (40 for thinking mode)
./llama serve -m ~/llm_models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf --mmproj ~/llm_models/gemma-4-26B-A4B-it-mmproj-F16.gguf --ctx-size 98304 --n-gpu-layers 999 --n-cpu-moe 5 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size 512 --reasoning on --reasoning-budget 8192 --jinja --temp 0.3 --top-p 0.9 --top-k 40 --min-p 0.05 --presence-penalty 0.0 --repeat-penalty 1.0 --load-mode none --host 0.0.0.0
## Without expert offload, forced quant - 70 t/s (50 for thinking mode)
./llama serve -m ~/llm_models/gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf --mmproj ~/llm_models/gemma-4-26B-A4B-it-mmproj-F16.gguf --ctx-size 98304 --n-gpu-layers 999 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size 512 --reasoning on --jinja --temp 0.6 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --load-mode none --host 0.0.0.0

# Qwen 3.6 35B A3B. MOE
## With expert offload, good quant - 40 t/s (35 for thinking mode)
### Thinking mode
./llama serve -m ~/llm_models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --mmproj ~/llm_models/Qwen3.6-35B-A3B-mmproj-F16.gguf --no-mmproj-offload --ctx-size 98304 --n-gpu-layers 99 --n-cpu-moe 12 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --reasoning on --reasoning-budget 8192 --jinja --temp 0.3 --top-p 0.9 --top-k 40 --min-p 0.05 --presence-penalty 0.0 --repeat-penalty 1.0 --parallel 1 --load-mode none --host 0.0.0.0
### No thinking mode
./llama serve -m ~/llm_models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --mmproj ~/llm_models/Qwen3.6-35B-A3B-mmproj-F16.gguf --no-mmproj-offload --ctx-size 98304 --n-gpu-layers 99 --n-cpu-moe 12 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --reasoning off --reasoning-budget 0 --jinja --temp 0.3 --top-p 0.9 --top-k 40 --min-p 0.05 --presence-penalty 0.0 --repeat-penalty 1.0 --parallel 1 --load-mode none --host 0.0.0.0
## Without expert offload to CPU and without reasoning - 70 t/s (for fast tasks - no reasoning, forced quant)
./llama serve -m ~/llm_models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --mmproj ~/llm_models/Qwen3.6-35B-A3B-mmproj-F16.gguf --no-mmproj-offload --ctx-size 131072 --n-gpu-layers 99 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --reasoning off --reasoning-budget 0 --jinja --temp 0.6 --top-p 0.8 --top-k 21 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --parallel 1 --load-mode mmap --host 0.0.0.0

# Qwen 3.8 27B. Dense model.
## Quite slow (27B active params). Reasoning on (for complex tasks). Forced quant makes reasoning not relible (not recommended option).
./llama serve -m ~/llm_models/Qwen3.8-27B-UD-IQ3_S.gguf --mmproj ~/llm_models/Qwen3.8-27B-mmproj-F16.gguf --no-mmproj-offload --ctx-size 98304 --n-gpu-layers 999 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --jinja --temp 0.3 --top-p 0.95 --top-k 40 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --parallel 1 --host 0.0.0.0

serve - launch llama.cpp server with OpenAI API
-m - path to the model GGUF
--mmproj - path to image recognition file
--no-mmproj-offload - offload image recognition to CPU
--ctx-size - length of the context (max number of tokens session, size of KV-cache). Default for qwen 3.6 35B - 256K, which is too much for low-VRAM GPU (will occupy too much). Can be tuned by --cache-type-k --cache-type-v (q4_0 reduces KV-cache greatly, making model a little less accurate, but allowing more context). Exact number can be figured out by task needs+VRAM monitoring.
--n-gpu-layers - number of layers to go to GPU. Qwen 3.6 35B - 256 has 40 layers. But we use 99 to load all layers on GPU, because we offload layers with --n-cpu-moe. 
--n-cpu-moe - number of layers to go to CPU (for MOE - mixture of experts). Only slower layers go to CPU, attention and KV-cache will reside on GPU.
--flash-attn - optimized Attention mechanics (saves VRAM).
--cache-type-k, --cache-type-v - KV-cache key/value types. q8_0 - optimal quant, q4_0 max compressed.
--batch-size, --ubatch-size - logical/physical batch of tokens at a time to GPU. Bigger batch - faster prompt processing, but risks OOM with long context.
--reasoning - turning reasononing on/off. There are different recommended flags, following on and off.
    For off: --temp 0.7 --top-p 0.8
--reasoning-budget - number of tokens for reasoning (-1 - without restrictions)
--jinja - turn on model's tag logic (like <think>)
--temp - `temperature` (0.0 - most predictive, 1.1 - creative)
--min-p - min probability threshold of token (cuts off trash). For qwen recommended to zero --min-p for --top-p, --top-k
--top-p - takes only first tokens with max probability, which sum is top-p (0.8 - logical, 0.1-0.5 - strict (code generation), 1.0 - turned off, 0.95 - default)
--top-k - use first k probable tokens (1 - most predictive, 20 - focused, 40 - default, >40 - creative)
--presence-penalty - (negative - approves repetition, positive - disapproves, 0 - ignores). For qwen recommended --presence-penalty 1.5 --repeat-penalty 1.0
--repeat-penalty - stronger version of --presence-penalty (1.0 - turned off, 1.0-1.2 - penalty, <1.0 - approves repetition).
--parallel - parallel sequences. (1 - for chat sessions, 4-8 - for agents).
--load-mode - (mmap - fast load from disk (for local SSD), mlock - directly to RAM/VRAM, swap not allowed)
```

5. Using model with opencode agent.
- Install opencode-cli
- Add configuration file for the local model via llama.cpp to `~/.config/opencode/opencode.json` (like the one in the repo). For baseURL use the url, that `llama serve` command outputs, with `/v1`.
- Use command `opencode pair`. It gives url, which leads to WebUI of the opencode.








