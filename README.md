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
     - [ ] filter out broken SVGs, white-background SVGs, and samples with low color entropy
     - [ ] only SVGs with at least 500 tokens
     - [ ] cluster the remaining samples using DINO image features and perform stratified sampling to select 600 examples

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

