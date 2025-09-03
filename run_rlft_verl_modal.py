import modal
from modal import Image

app = modal.App("prefix-rl-verl")

app_reward = modal.App("prefix-rl-verl-reward-service")

# Preprocessed data
data_parquet_vol = modal.Volume.from_name("svg-rlrf-parquet", create_if_missing=True)
# Results checkpoints
trl_checkpoints_vol = modal.Volume.from_name(
    "prefix-rl-rlft-verl-checkpoints", create_if_missing=True
)
# Intermediate results
generated_vol = modal.Volume.from_name("svg-rlrf-generated", create_if_missing=True)

# Shared caches
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Stuff specific to dreamsim
torch_hub_cache_vol = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
dreamsim_cache_vol = modal.Volume.from_name("dreamsim-cache", create_if_missing=True)
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
    .uv_pip_install("verl==0.5.0", "cairosvg")
    .add_local_dir("env/svg", "/workspace/env/svg")
)

MINUTE = 60
HOUR = 60 * MINUTE

# Lightweight image for reward computation service
reward_service_image = (
    Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 gdk-pixbuf2.0-0 libffi-dev libxml2 libpng-dev zlib1g",
    )
    .uv_pip_install(
        "torch", "torchvision", "dreamsim", "cairosvg", "pillow", "numpy", "wandb"
    )
    .add_local_dir("env/svg", "/workspace/env/svg")
)


@app_reward.cls(
    image=reward_service_image,
    gpu="A10G",  # Use A10G for cost-effective inference
    scaledown_window=60,  # Keep warm
    max_containers=2,
    volumes={
        "/root/.cache/torch": torch_hub_cache_vol,  # For DreamSim model cache
    },
)
class SVGRewardService:
    @modal.enter()
    def startup(self):
        """Initialize the reward service with DreamSim model loaded."""
        import sys
        import os

        sys.path.append("/workspace/env/svg")
        os.environ["DREAMSIM_CACHE_DIR"] = "/root/.cache/torch"

        from svg_rlrf_reward import get_image_comparator

        self.comparator = get_image_comparator()
        print(f"SVGRewardService initialized with device: {self.comparator.device}")

    @modal.method()
    def compute_rewards_batch(
        self,
        data_sources: list[str],
        solution_strs: list[str],
        ground_truths: list[str],
        extra_infos: list = None,
        **kwargs,
    ) -> list[dict[str, float]]:
        from svg_rlrf_reward import compute_rewards_dict

        return compute_rewards_dict(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            **kwargs,
        )


@app.function(
    image=trainer_image,
    gpu="H100:8",
    timeout=8 * HOUR,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/torch": torch_hub_cache_vol,
        "/workspace/data": data_parquet_vol,
        "/workspace/checkpoints": trl_checkpoints_vol,
        "/workspace/generated": generated_vol,
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
        "config_svg_rlrf_l2_7b_verl",
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


@app.function(
    image=reward_service_image,
    timeout=10 * MINUTE,
)
def test_reward_service():
    print("Everything I know about SVGRewardService:")

    # This service MUST BE DEPLOYED FIRST!
    SVGRewardService = modal.Cls.from_name(
        "prefix-rl-verl-reward-service", "SVGRewardService"
    )

    print("app.id", app.app_id)
    print("app.name", app.name)
    print("SVGRewardService.id", SVGRewardService().object_id)
    print("SVGRewardService.is_hydrated", SVGRewardService().is_hydrated)

    def circle_svg(r: float) -> str:
        return f"<svg width='512' height='512'><circle cx='256' cy='256' r='{r:.0f}' stroke='black' stroke-width='3' fill='red' /></svg>"

    def expected_response(thinking: str = "", svg: str = "") -> str:
        return f"<think>{thinking}</think><answer>{svg}</answer>"

    # Test data
    test_svg = circle_svg(200)
    test_responses = [
        expected_response("Circle 1", circle_svg(200)),  # Perfect match
        expected_response("Circle 2", circle_svg(180)),  # Slightly different
        expected_response("Circle 3", circle_svg(100)),  # Very different
    ]
    test_gts = [test_svg] * 3

    print("Testing remote reward service...")
    rewards = SVGRewardService().compute_rewards_batch.remote(
        data_sources=["test"] * 3,
        solution_strs=test_responses,
        ground_truths=test_gts,
        extra_infos=[None] * 3,
    )

    print("\nReward results:")
    for i, reward_dict in enumerate(rewards):
        print(
            f"  Response {i + 1}: overall={reward_dict['score']:.3f}, format={reward_dict['format']:.3f}, dreamsim={reward_dict['dreamsim']:.3f}"
        )

    return rewards


@app.local_entrypoint()
def main():
    prepare_svg_data.remote()
    train_model_verl.remote()


def pymain():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--interactive", action="store_true")
    args.add_argument("--detach", action="store_true")
    args = args.parse_args()

    app_reward.deploy()

    with app.run(interactive=args.interactive, detach=args.detach):
        prepare_svg_data.remote()
        train_model_verl.remote()
        test_reward_service.remote()

    app_reward.stop()


if __name__ == "__main__":
    pymain()
