WIP:
Train vision-language models with reinforcement learning.
Using modal for compute, may make it platform-agnostic in the future

References:
- https://github.com/willccbb/verifiers

- https://huggingface.co/papers/2505.20793
  - LLaMA-Factory for SFT
  - EasyR1 for RLRF


Implementation of SVG RLRF paper (Im2SVG):

- [x] Implement mix of image similarity rewards
   - [ ] Small ViewBox Hack (use gt viewbox to render)
   - [ ] SVG Length Collapse: Reward weight scheduling
- [x] Run on modal
- [x] RL training setup for SVG fine-tuning (EasyR1)
  - [ ] C.3: Dynamic Max Length
- [x] Dataset
   - [x] Optimize SVGs, quantize
   - [x] Remove blank images
   - [x] Always rasterize to 512 smallest side, 1536 largest side
   - [ ] SVG-Stack-Hard Test Set
     - [x] filter out broken SVGs, white-background SVGs, and samples with low color entropy
     - [ ] only SVGs with at least 500 tokens
     - [ ] cluster the remaining samples using DINO image features and perform stratified sampling to select 600 examples
- [ ] Eval harness
  - [ ] Eval against openai, anthropic, google, vllm, etc.

Further ideas Im2SVG:
- [ ] Prompt model to start with the gt width/height/viewbox ie "generate an 100x100 SVG that …"
- [ ] Test out more featureful renderers
  - [ ] Skia?
  - [ ] headless chromium
- [ ] Test out data generation of easy examples instead of fine-tuning for bootstrapping
- [ ] Datasets 
  - [ ] https://huggingface.co/OmniSVG

## Usage:

### SFT
```sh
modal run --detach run_sft_modal.py
```

### RL fine-tuning
```sh
modal run --detach run_easyr1_modal.py::train_model_easyr1 --config svg
```

### Evaluation

Run evaluations against different models:

#### OpenAI Models
```sh
python svg_eval.py --client openai --model_name gpt-4o-mini --dataset simple-shapes -n 10
python svg_eval.py --client openai --model_name gpt-4o --dataset svg-stack -n 100 --temperature 0.1
python svg_eval.py --client openai-responses --model_name o1-mini --dataset simple-shapes -n 10
```

#### vLLM Models (Qwen2.5-VL)
```sh
# Run the vLLM server
MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct modal serve run_vllm_server_modal.py
```
The script will output the URL of the vLLM server.

```sh
# Run evaluation
python svg_eval.py --client vllm --vllm_endpoint https://prefix--prefix-rl-vllm-server-serve-dev.modal.run/v1/ --model_name "Qwen/Qwen2.5-VL-7B-Instruct" --num_eval_examples 100 --debug_dump --num_workers 16
```

#### Google Gemini
```sh
python svg_eval.py --client google --model_name gemini-2.0-flash --dataset simple-shapes -n 10
python svg_eval.py --client google --model_name gemini-2.5-flash --dataset svg-stack -n 100 --temperature 0.2
```

#### Anthropic Claude
```sh
python svg_eval.py --client anthropic --model_name claude-3-5-sonnet-20241022 --dataset simple-shapes -n 10
```

#### Debug with VS Code
Use the debug configurations in `.vscode/launch.json` for step-through debugging.

#### Key Parameters
- `--dataset`: `simple-shapes` (small) or `svg-stack` (large)
- `--temperature`: 0.1 (default) for consistent output, higher for creativity
- `--num_workers`: Parallel processing (default: 1)
- `--debug_dump`: Generate detailed HTML reports with image comparisons

