import modal
from modal import Image

app = modal.App("prefix-rl-easyr1")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
torch_hub_cache_vol = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
dreamsim_cache_vol = modal.Volume.from_name("dreamsim-cache", create_if_missing=True)

trainer_image = (
    Image.from_registry(
        "hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 gdk-pixbuf2.0-0 libffi-dev libxml2 libpng-dev zlib1g",
    )
    # install for svg rlrf
    .pip_install("dreamsim", "cairosvg", "tqdm")
)

trainer_with_files = (
    trainer_image.run_commands(
        "git clone https://github.com/darknoon/EasyR1.git /root/EasyR1",
         # Pin to correct commit hash
        "cd /root/EasyR1 && git checkout cf78409" 
    )
    .workdir("/root/EasyR1")
    .add_local_dir("env/svg", "/root/EasyR1/env/svg")
)


MINUTE = 60
HOUR = 60 * MINUTE


default_args = {
    "config": "examples/config.yaml",
    "data": {
        "train_files": "hiyouga/geometry3k@train",
        "val_files": "hiyouga/geometry3k@test",
    },
    "worker": {"actor": {"model": {"model_path": "Qwen/Qwen2.5-VL-7B-Instruct"}}},
    "trainer": {
        "experiment_name": "qwen2_5_vl_7b_geo_grpo",
        "n_gpus_per_node": 8,
    },
}

args_svg_rlrf = {
    "config": "env/svg/config_svg_rlrf_7b.yaml",
    "trainer": {
        "experiment_name": "qwen2_5_vl_7b_svg_rlrf",
        "n_gpus_per_node": 8,
    },
}


@app.function(
    image=trainer_with_files,
    gpu="H100:8",
    timeout=1 * HOUR,  # will not complete but just test that it's working
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/torch": torch_hub_cache_vol,
    },
    secrets=[
        # for logging to wandb
        modal.Secret.from_name("wandb-darknoon"),
        modal.Secret.from_name("huggingface-write"),
    ],
)
def train_model_easyr1(*arglist):
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="default", choices=["default", "svg"]
    )
    cli_args = parser.parse_args(arglist)

    config = args_svg_rlrf if cli_args.config == "svg" else default_args

    print(f"Using config: {config}")

    os.environ["PYTHONUNBUFFERED"] = "1"

    from omegaconf import OmegaConf
    import ray
    from verl.trainer.main import Runner
    from verl.trainer.config import PPOConfig

    cli_args = OmegaConf.create(config)
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        print(f"Loaded config from {config_path}:")
        print(file_config)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                # For the DreamSim image comparator, used for the RL reward function svg_rlrf_reward.py
                "DREAMSIM_CACHE_DIR": "/root/.cache/torch",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


@app.function(
    image=trainer_with_files,
    cpu=2,  # CPU only, no GPU needed
    timeout=30 * MINUTE,  # Increased timeout for full benchmark
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/torch": torch_hub_cache_vol,
    },
)
def benchmark_reward():
    import sys
    import os
    import time
    import numpy as np
    from tqdm import tqdm

    sys.path.insert(0, "/root/EasyR1")
    os.makedirs("/root/.cache/torch", exist_ok=True)
    os.environ["DREAMSIM_CACHE_DIR"] = "/root/.cache/torch"

    from env.svg.svg_rlrf_reward import compute_rewards_dict, expected_response
    import torch

    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    def circle_svg(r: float) -> str:
        return f"<svg width='512' height='512'><circle cx='256' cy='256' r='{r:.0f}' stroke='black' stroke-width='3' fill='red' /></svg>"
    
    def create_batch(size: int, base_radius: int = 200) -> list:
        """Create a batch of test inputs with varying circle radii."""
        return [
            {
                "response": expected_response(
                    thinking=f"Drawing circle {i}", 
                    svg=circle_svg(base_radius + i * 5), 
                    add_line_breaks=True
                ),
                "ground_truth": circle_svg(base_radius)
            }
            for i in range(size)
        ]
    
    def benchmark_batch(batch_size: int, iterations: int = 10) -> dict:
        """Benchmark a specific batch size."""
        batch = create_batch(batch_size)
        
        # Warmup
        for _ in tqdm(range(2), desc=f"Warmup (batch={batch_size})", leave=False):
            compute_rewards_dict(batch)
        
        # Benchmark
        times = []
        for _ in tqdm(range(iterations), desc=f"Benchmark (batch={batch_size})", leave=False):
            start = time.perf_counter()
            rewards = compute_rewards_dict(batch)
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            "batch_size": batch_size,
            "total_mean": np.mean(times),
            "total_p50": np.percentile(times, 50),
            "per_item_mean": np.mean(times) / batch_size,
            "per_item_p50": np.percentile(times, 50) / batch_size,
            "num_rewards": len(rewards)
        }

    # Test batch size 32 only
    print("\n" + "="*50)
    print("BATCH PROCESSING PERFORMANCE (Batch Size: 32)")
    print("="*50)
    
    result = benchmark_batch(32, iterations=4)
    
    print(f"\nResults for Batch Size: {result['batch_size']}")
    print(f"  Total time:    {result['total_mean']:6.1f} ms (mean), {result['total_p50']:6.1f} ms (p50)")
    print(f"  Per item time: {result['per_item_mean']:6.1f} ms (mean), {result['per_item_p50']:6.1f} ms (p50)")
    print(f"  Throughput:    {1000 / result['per_item_mean']:6.1f} items/sec")
    
    return "Benchmark completed"


@app.local_entrypoint()
def main():
    train_model_easyr1.remote("--config", "svg")
