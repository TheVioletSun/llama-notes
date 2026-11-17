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
4. Launch the model with command:
```
./llama <cli/serve> -m ~/Downloads/Qwen3.6-35B-A3B-MXFP4_MOE.gguf --ctx-size 131072 --n-gpu-layers 99 --n-cpu-moe 17 --flash-attn 1 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --reasoning off --reasoning-budget 0 --jinja --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0 --parallel 4 --load-mode none

cli - launch chat in the console
serve - launch llama.cpp server with OpenAI API
-m - path to the model GGUF
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
--load-mode - (none - load directly to RAM/VRAM, excluding swap)
```
