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
    .pip_install("dreamsim", "cairosvg")
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
    gpu="H100",
    timeout=10 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/torch": torch_hub_cache_vol,
    },
)
def benchmark_reward():
    import sys
    import os
    import time

    sys.path.insert(0, "/root/EasyR1")
    
    # Create cache directory if it doesn't exist
    os.makedirs("/root/.cache/torch", exist_ok=True)
    os.environ["DREAMSIM_CACHE_DIR"] = "/root/.cache/torch"

    from env.svg.svg_rlrf_reward import compute_rewards_dict, expected_response
    import torch
    import numpy as np

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def circle_svg(r: float) -> str:
        return f"<svg width='512' height='512'><circle cx='256' cy='256' r='{r:.0f}' stroke='black' stroke-width='3' fill='red' /></svg>"

    # Warmup
    print("\n=== Warmup ===")
    for i in range(3):
        response = expected_response(
            thinking="Test", svg=circle_svg(200), add_line_breaks=True
        )
        reward_input = {"response": response, "ground_truth": circle_svg(200)}
        start = time.perf_counter()
        rewards = compute_rewards_dict(reward_input)
        elapsed = time.perf_counter() - start
        print(f"Warmup {i + 1}: {elapsed * 1000:.1f} ms")

    # Benchmark
    print("\n=== Benchmark (100 iterations) ===")
    test_cases = [
        ("Identical", circle_svg(200), circle_svg(200)),
        ("Different", circle_svg(180), circle_svg(200)),
    ]

    for name, svg_pred, svg_gt in test_cases:
        print(f"\n{name}:")
        response = expected_response(
            thinking="Test", svg=svg_pred, add_line_breaks=True
        )
        reward_input = {"response": response, "ground_truth": svg_gt}

        times = []
        for i in range(100):
            start = time.perf_counter()
            rewards = compute_rewards_dict(reward_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        print(f"  Mean: {np.mean(times):.1f} ms")
        print(f"  P50: {np.percentile(times, 50):.1f} ms")
        print(f"  P95: {np.percentile(times, 95):.1f} ms")

    # Detailed profiling
    print("\n=== Profiling ===")
    from env.svg import svg_rlrf_reward
    import functools

    timing_results = {}

    def time_function(name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                if name not in timing_results:
                    timing_results[name] = []
                timing_results[name].append(elapsed * 1000)
                return result

            return wrapper

        return decorator

    # Patch functions
    svg_rlrf_reward.svg_env = time_function("svg_env")(svg_rlrf_reward.svg_env)
    svg_rlrf_reward.rasterize_svg = time_function("rasterize_svg")(
        svg_rlrf_reward.rasterize_svg
    )
    svg_rlrf_reward.canny = time_function("canny")(svg_rlrf_reward.canny)

    comparator = svg_rlrf_reward.get_image_comparator()
    comparator.compare_images = time_function("compare_images")(
        comparator.compare_images
    )

    # Time DreamSim calls
    original_model = comparator.dreamsim_model

    def timed_dreamsim(img1, img2):
        start = time.perf_counter()
        result = original_model(img1, img2)
        elapsed = time.perf_counter() - start
        timing_results.setdefault("dreamsim_forward", []).append(elapsed * 1000)
        return result

    comparator.dreamsim_model = timed_dreamsim

    response = expected_response(
        thinking="Profile", svg=circle_svg(200), add_line_breaks=True
    )
    reward_input = {"response": response, "ground_truth": circle_svg(200)}

    for i in range(10):
        timing_results.clear()
        rewards = compute_rewards_dict(reward_input)

    print("\nFunction timing (last run):")
    for func_name, times in timing_results.items():
        if times:
            print(f"  {func_name}: {times[-1]:.1f} ms")

    print(
        f"  TOTAL: {sum(times[-1] for times in timing_results.values() if times):.1f} ms"
    )
    return "Done"


@app.local_entrypoint()
def main():
    train_model_easyr1.remote("--config", "svg")
