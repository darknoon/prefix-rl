import modal
from modal import Image

app = modal.App("prefix-rl-sft")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sft_checkpoints_vol = modal.Volume.from_name(
    "prefix-rl-sft-checkpoints", create_if_missing=True
)

# Create image with LLaMA-Factory
sft_image = Image.from_registry(
    "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
).pip_install(
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

# Add LLaMA-Factory and config files
sft_image_with_files = (
    sft_image.run_commands(
        "git clone --branch 8ffe7da https://github.com/hiyouga/LLaMA-Factory.git /root/LLaMA-Factory",
    )
    .workdir("/root/LLaMA-Factory")
    .add_local_dir("env/svg", "/root/LLaMA-Factory/env/svg")
)

MINUTE = 60
HOUR = 60 * MINUTE


@app.function(
    image=sft_image_with_files,
    gpu="H100:8",
    timeout=4 * HOUR,  # SFT training should complete within 4 hours
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
def train_sft():
    import os
    import subprocess
    import sys

    # Set environment variables
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "prefix-rl-sft"

    # Change to LLaMA-Factory directory
    os.chdir("/root/LLaMA-Factory")

    # Install LLaMA-Factory in editable mode
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    print("Starting SFT training with config: env/svg/config_svg_sft_3b.yaml")

    # Run LLaMA-Factory training
    cmd = [sys.executable, "-m", "llamafactory.train", "env/svg/config_svg_sft_3b.yaml"]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("SFT training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise


@app.function(
    image=sft_image_with_files,
    gpu="H100:1",
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
    from huggingface_hub import HfApi

    os.environ["PYTHONUNBUFFERED"] = "1"

    api = HfApi()

    print(f"Uploading model from {model_path} to {repo_name}")

    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"Model successfully uploaded to {repo_name}")
    except Exception as e:
        print(f"Failed to upload model: {e}")
        raise


@app.function(
    image=sft_image_with_files,
    volumes={
        "/root/LLaMA-Factory/saves": sft_checkpoints_vol,
    },
)
def list_checkpoints():
    """List available checkpoints in the volume"""
    import os

    saves_dir = "/root/LLaMA-Factory/saves"
    if os.path.exists(saves_dir):
        print("Available checkpoints:")
        for root, dirs, files in os.walk(saves_dir):
            level = root.replace(saves_dir, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
    else:
        print("No checkpoints directory found")


@app.local_entrypoint()
def main():
    """Main entrypoint to run SFT training"""
    print("Starting SFT training on Modal...")

    # Run SFT training
    train_sft.remote()

    print("SFT training job submitted!")

    # Optionally upload to HF Hub after training
    # Uncomment the lines below if you want to auto-upload
    # model_path = "/root/LLaMA-Factory/saves/qwen2_5vl-3b/full/sft"
    # repo_name = "darknoon/qwen2.5-vl-3b-svg-sft"
    # upload_model_to_hf.remote(model_path, repo_name)


if __name__ == "__main__":
    main()
