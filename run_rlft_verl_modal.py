import modal
from modal import Image

app = modal.App("prefix-rl-verl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
torch_hub_cache_vol = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
dreamsim_cache_vol = modal.Volume.from_name("dreamsim-cache", create_if_missing=True)
data_parquet_vol = modal.Volume.from_name("svg-rlrf-parquet", create_if_missing=True)

# Slim image for dataset preparation only
dataset_prep_image = (
    Image.debian_slim()
    # Datasets 4.0.0 doesn't work, it saves a List[Image] instead of Sequence[Image] and we can't load it.
    # see https://github.com/huggingface/datasets/pull/7634
    .uv_pip_install("pillow", "datasets==3.6.0")
    .add_local_python_source("prefixrl")
)

# Use the official verl image for training
trainer_image = (
    Image.from_registry(
        "hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0"
    )
    .run_commands(
        # Reset to default Ubuntu mirrors
        "sed -i 's|https://mirrors.tuna.tsinghua.edu.cn|http://archive.ubuntu.com|g' /etc/apt/sources.list",
        "apt-get update",
        "apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 gdk-pixbuf2.0-0 libffi-dev libxml2 libpng-dev zlib1g",
    )
    .uv_pip_install("verl", "dreamsim", "cairosvg")
    .add_local_dir("env/svg", "/workspace/env/svg")
)

MINUTE = 60
HOUR = 60 * MINUTE


@app.function(
    image=trainer_image,
    gpu="H100:8",
    timeout=2 * HOUR,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/torch": torch_hub_cache_vol,
        "/workspace/data": data_parquet_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-darknoon"),
        modal.Secret.from_name("huggingface-write"),
    ],
)
def train_model_verl():
    import os
    import sys
    import subprocess

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "false"  # Disable to avoid multiprocessing issues
    )
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    os.environ["DREAMSIM_CACHE_DIR"] = "/root/.cache/torch"
    os.environ["HF_DATASETS_NUM_PROC"] = (
        "1"  # Reduce parallel workers for dataset processing
    )

    # Add the env/svg to Python path for reward function
    sys.path.append("/workspace/env/svg")

    # Run verl training with config file using Hydra syntax
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        "/workspace/env/svg",
        "--config-name",
        "config_svg_verl",
    ]

    result = subprocess.run(cmd, check=True)
    return result.returncode


@app.function(
    image=dataset_prep_image,
    timeout=30 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/workspace/data": data_parquet_vol,
    },
)
def prepare_svg_data(force: bool = False):
    from prefixrl.datasets import simple_shapes

    simple_shapes.prepare_for_verl(
        output_dir="/workspace/data/simple-shapes", force=force
    )


@app.local_entrypoint()
def main():
    # Force regeneration to pick up the new chat format
    prepare_svg_data.remote(force=True)
    train_model_verl.remote()
