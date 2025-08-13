import modal
from modal import Image

app = modal.App("prefix-rl-sft-trl")

# -----------------------------------------------------------------------------
# Shared volumes (Hugging Face cache & checkpoints) --------------------------------
# -----------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
trl_checkpoints_vol = modal.Volume.from_name(
    "prefix-rl-sft-trl-checkpoints", create_if_missing=True
)

# -----------------------------------------------------------------------------
# Docker image ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# We reuse the official PyTorch CUDA base image and install TRL + friends.
# Versions roughly match what we use elsewhere to remain compatible.
trl_image = (
    Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        # Core
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "tokenizers",
        # Vision / multimodal support
        "Pillow",
        # "timm",  # widely used by VLMs
        # Misc utilities
        "scipy",
        "einops",
        "sentencepiece",
        "tiktoken",
        "protobuf",
        "numpy",
        "pydantic",
        "pandas",
        # Optional monitoring
        "wandb",
        "deepspeed",
    )
    .workdir("/workspace")
    .add_local_file("sft_trl.py", "/workspace/sft_trl.py")
    .add_local_dir("env/svg", "/workspace/env/svg")
    .add_local_file("deepspeed_zero3.yaml", "/workspace/deepspeed_zero3.yaml")
)

# -----------------------------------------------------------------------------
# Constants ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
MINUTE = 60
HOUR = 60 * MINUTE


# -----------------------------------------------------------------------------
# Training function -------------------------------------------------------------
# -----------------------------------------------------------------------------
# modal run --detach run_sft_trl_modal.py
@app.function(
    image=trl_image,
    gpu="H100:4",
    timeout=8 * HOUR,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/workspace/checkpoints": trl_checkpoints_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-darknoon"),
        modal.Secret.from_name("huggingface-write"),
    ],
)
def train_sft_trl():
    """Fine-tune Qwen-2.5-VL on the given dataset using TRL (pure-Python).,
    
    eg
    accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
    """
    import os
    import subprocess

    # Environment tweaks to make logs stream nicely
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ.setdefault("WANDB_PROJECT", "prefix-rl-sft-trl")
    num_gpus = int(os.environ.get("NUM_GPUS", "4"))

    # Launch distributed training using accelerate
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "/workspace/deepspeed_zero3.yaml",
        "/workspace/sft_trl.py",
        "--dataset_name",
        "darknoon/svg-stack-filtered",
        "--model_name_or_path",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "--per_device_train_batch_size",
        "2",
        "--output_dir",
        "/workspace/checkpoints/qwen2_5vl-7b_sft_svg_filtered",
        "--bf16",
        "True",
        "--torch_dtype",
        "bfloat16",
        "--max_steps",
        "10000",
        "--learning_rate",
        "5e-6",
        "--save_steps",
        "1000",
        "--logging_steps",
        "10",
        "--gradient_checkpointing",
        "--max_length",
        "4096",
        # "32768", from the paper, should probably do this instead.
        "--trust_remote_code",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Training finished ✅")


# -----------------------------------------------------------------------------
# Helper to upload the final model to the Hub -----------------------------------
# -----------------------------------------------------------------------------
@app.function(
    image=trl_image,
    timeout=30 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/workspace/checkpoints": trl_checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-write")],
)
def upload_model_to_hf(model_path: str, repo_name: str):
    """Upload a checkpoint to the Hugging Face Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from huggingface_hub import HfApi
    import os

    os.environ["PYTHONUNBUFFERED"] = "1"

    api = HfApi()
    api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)

    print(f"Loading model from {model_path}…")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Pushing artifacts to {repo_name}…")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    processor.push_to_hub(repo_name)
    print("Upload complete ✅")


# -----------------------------------------------------------------------------
# Local entry-point -------------------------------------------------------------
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Run training + (optionally) upload the checkpoint when invoked locally."""
    MODEL_OUTPUT_PATH = "/workspace/checkpoints/qwen2_5vl-7b/full/sft"
    HF_REPO_NAME = "darknoon/svg-stack-filtered-sft-qwen2.5-vl-7b-trl"

    print("Launching distributed SFT job on Modal…")
    train_sft_trl.remote()

    # Uncomment if you want to immediately push after training; many users
    # prefer doing this manually to double-check the run.
    # upload_model_to_hf.remote(MODEL_OUTPUT_PATH, HF_REPO_NAME)
