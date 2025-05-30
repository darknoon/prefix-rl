import modal
from modal import Image

app = modal.App("prefix-rl-easyr1")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

trainer_image = Image.from_registry(
    "hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0"
)

trainer_with_files = trainer_image.run_commands(
    "git clone https://github.com/hiyouga/EasyR1.git /root/EasyR1"
).workdir("/root/EasyR1")


MINUTE = 60
HOUR = 60 * MINUTE


@app.function(
    image=trainer_with_files,
    gpu="H100:8",
    timeout=1 * HOUR,  # will not complete but just test that it's working
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        # for logging to wandb
        modal.Secret.from_name("wandb-darknoon"),
        modal.Secret.from_name("huggingface-write"),
    ],
)
def train_model_easyr1():
    import subprocess

    subprocess.run("bash examples/qwen2_5_vl_7b_geo3k_grpo.sh", shell=True)


@app.local_entrypoint()
def main():
    app.run(train_model_easyr1)
