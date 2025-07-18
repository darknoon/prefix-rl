import modal
from modal import Image

app = modal.App("prefix-rl-sft")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sft_checkpoints_vol = modal.Volume.from_name(
    "prefix-rl-sft-checkpoints", create_if_missing=True
)
LLAMA_FACTORY_SHA = "8ffe7da"

# Create image with LLaMA-Factory
sft_image = (
    Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        # Core LLaMA-Factory dependencies (from official requirements.txt)
        "transformers>=4.45.0,<=4.52.4,!=4.46.*,!=4.47.*,!=4.48.0,!=4.52.0",
        "datasets>=2.16.0,<=3.6.0",
        "accelerate>=0.34.0,<=1.7.0",
        "peft>=0.14.0,<=0.15.2",
        "trl>=0.8.6,<=0.9.6",
        "tokenizers>=0.19.0,<=0.21.1",
        # Required for training
        "scipy",
        "einops",
        "sentencepiece",
        "tiktoken",
        "protobuf",
        "matplotlib>=3.7.0",
        "fire",
        "omegaconf",
        "packaging",
        "pyyaml",
        "numpy<2.0.0",
        "pydantic<=2.10.6",
        "pandas>=2.0.0",
        # Vision/multimodal support
        "av",
        "Pillow",
        # Optional but useful
        "wandb",
        "deepspeed",
    )
)

# Add LLaMA-Factory and config files
sft_image_with_files = (
    sft_image.run_commands(
        "git clone https://github.com/hiyouga/LLaMA-Factory.git /root/LLaMA-Factory",
        "cd /root/LLaMA-Factory && git checkout ${LLAMA_FACTORY_SHA}",
        # this is kinda dumb, don't think we really need to install as editable
        "cd /root/LLaMA-Factory && pip install -e '.[torch,metrics]'",
    )
    .workdir("/root/LLaMA-Factory")
    .add_local_dir("env/svg", "/root/LLaMA-Factory/env/svg")
)

MINUTE = 60
HOUR = 60 * MINUTE


# modal run --detach run_sft_modal.py
@app.function(
    image=sft_image_with_files,
    gpu="H100:4",
    timeout=8 * HOUR,  # SFT training should complete within 8 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/LLaMA-Factory/saves": sft_checkpoints_vol,
    },
    secrets=[
        # for logging to wandb and accessing HF models
        modal.Secret.from_name("wandb-darknoon"),
        modal.Secret.from_name("huggingface-write"),
    ],
)
def train_sft(config_path: str):
    import os
    import subprocess

    # Set environment variables
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "prefix-rl-sft"

    # first check that llamafactory-cli is installed correctly
    try:
        subprocess.run(
            ["llamafactory-cli", "version"], check=True, capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"llamafactory-cli is not installed correctly: {e}")
        raise

    print(f"Starting SFT training with config: {config_path}")

    cmd = ["llamafactory-cli", "train", config_path]
    print(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    print("SFT training completed successfully!")
    return result.returncode


@app.function(
    image=sft_image_with_files,
    timeout=30 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/LLaMA-Factory/saves": sft_checkpoints_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-write"),
    ],
)
def upload_model_to_hf(model_path: str, repo_name: str):
    """Upload the trained model to Hugging Face Hub"""
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from huggingface_hub import HfApi

    os.environ["PYTHONUNBUFFERED"] = "1"

    api = HfApi()

    print(f"Loading model from {model_path}")

    try:
        # Create repository if it doesn't exist (or do nothing if it exists)
        api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)

        # Load the model (this will only load the final model, not checkpoints)
        print("Loading model, tokenizer, and processor...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)

        print(f"Pushing model to {repo_name}")

        # Push to hub - this will automatically handle the model structure
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)

        print(f"Model successfully uploaded to {repo_name}")
    except Exception as e:
        print(f"Failed to upload model: {e}")
        raise


@app.local_entrypoint()
def main():
    """Main entrypoint to run SFT training"""
    print("Starting SFT training on Modal...")
    train_sft.remote("env/svg/config_svg_sft_3b.yaml")
    model_path = "/root/LLaMA-Factory/saves/qwen2_5vl-3b/full/sft"
    repo_name = "darknoon/svg-stack-filtered-sft-qwen2.5-vl-3b"
    upload_model_to_hf.remote(model_path, repo_name)
