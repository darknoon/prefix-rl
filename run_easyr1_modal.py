import modal
from modal import Image

app = modal.App("prefix-rl-easyr1")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
torch_hub_cache_vol = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

trainer_image = (
    Image.from_registry(
        "hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 gdk-pixbuf2.0-0 libffi-dev libxml2 libpng-dev zlib1g",
    )
    # install for svg rlrf
    .pip_install("dreamsim")
)

trainer_with_files = (
    trainer_image.run_commands(
        "git clone https://github.com/hiyouga/EasyR1.git /root/EasyR1"
    )
    .add_local_dir("env/svg", "/root/EasyR1/svg")
    .workdir("/root/EasyR1")
)


MINUTE = 60
HOUR = 60 * MINUTE


default_args = {
    "config": "examples/config.yaml",
    "data.train_files": "hiyouga/geometry3k@train",
    "data.val_files": "hiyouga/geometry3k@test",
    "worker.actor.model.model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "trainer.experiment_name": "qwen2_5_vl_7b_geo_grpo",
    "trainer.n_gpus_per_node": 8,
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
def train_model_easyr1(args=default_args):
    import os

    os.environ["PYTHONUNBUFFERED"] = "1"

    from omegaconf import OmegaConf
    import ray
    from verl.trainer.main import Runner
    from verl.trainer.config import PPOConfig

    cli_args = OmegaConf.from_dict(args)
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
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
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


@app.local_entrypoint()
def main():
    app.run(train_model_easyr1)
