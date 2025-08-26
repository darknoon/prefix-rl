import modal
from modal import Image

app = modal.App("prefix-rl-verl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
torch_hub_cache_vol = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Use the official verl image
trainer_image = (
    Image.from_registry(
        "hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 gdk-pixbuf2.0-0 libffi-dev libxml2 libpng-dev zlib1g",
    )
    # Install official verl and dependencies for svg rlrf
    .pip_install("verl", "dreamsim", "cairosvg")
    .add_local_dir("env/svg", "/root/svg_env")
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
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    # Add the svg_env to Python path for reward function
    sys.path.append("/root/svg_env")

    # Run verl training with config file using Hydra syntax
    cmd = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        "/root/svg_env",
        "--config-name",
        "config_svg_verl",
    ]

    result = subprocess.run(cmd, check=True)
    return result.returncode


@app.local_entrypoint()
def main():
    train_model_verl.remote()
