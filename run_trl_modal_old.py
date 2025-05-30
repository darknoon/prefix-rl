import modal
from modal import Image
from typing import TypedDict
from dataclasses import dataclass

app = modal.App(name="prefix-rl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


base_image = Image.debian_slim(python_version="3.12")

browser_env_image = (
    base_image
    # https://modal.com/docs/examples/web-scraper#a-simple-web-scraper
    .run_commands(
        "apt-get update",
        "apt-get install -y software-properties-common",
        "apt-add-repository non-free",
        "apt-add-repository contrib",
        "pip install playwright==1.42.0",
        "playwright install-deps chromium",
        "playwright install chromium",
    ).add_local_file("svg_renderer.py", "/root/svg_renderer.py")
)

image_embedding_env = Image.from_registry(
    "pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime"
).pip_install(
    "transformers",
    "pillow",
    "numpy",
)

trl_trainer_image = Image.from_registry(
    "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
).pip_install(
    "trl[vllm]",
)


class ImageComparisonResult(TypedDict):
    l2: float
    clip_vit_b_32: float


def image_similarity_reward(image_bytes: bytes, reference_bytes: bytes) -> float:
    d = compare_images.remote(image_bytes, reference_bytes)
    l2 = d["l2"]
    l2_sim = 1.0 - l2
    sim = d["clip_vit_b_32"]
    return 0.5 * l2_sim + 0.5 * sim


@app.function(image=image_embedding_env, gpu="T4")
def compare_images(image_bytes: bytes, reference_bytes: bytes) -> ImageComparisonResult:
    """Compare two images using L2 distance and vision embedding similarity.

    Args:
        image_bytes: PNG image bytes of the first image
        reference_bytes: PNG image bytes of the reference image to compare against

    Returns:
        Dict containing l2_distance and embedding_similarity metrics
    """
    import torch
    import numpy as np
    from PIL import Image as PILImage
    import io
    import torch.nn.functional as F
    from transformers import AutoProcessor, AutoModel

    def process_image(image_bytes: bytes) -> PILImage:
        """Process image bytes into RGB PIL Image of specified size."""
        image = PILImage.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        return image

    @torch.no_grad()
    def get_image_embedding(image: PILImage, processor, model, size=(224, 224)):
        """Get normalized vision embedding for an image."""
        image = image.resize(size)
        inputs = processor(images=image, return_tensors="pt")
        embedding = model.get_image_features(**inputs)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    # Load and process images
    image = process_image(image_bytes)
    reference = process_image(reference_bytes)

    # Calculate L2 distance
    if image.size != reference.size:
        # resize image to reference size
        image = image.resize(reference.size)
        # warn
        print(
            f"Image size {image.size} does not match reference size {reference.size}, resizing image to reference size. Results may be inaccurate."
        )

    img_array = np.array(image)
    ref_array = np.array(reference)

    l2_distance = np.sqrt(np.sum((img_array - ref_array) ** 2)) / (img_array.size * 255)

    # Get vision embeddings using CLIP
    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", use_fast=True
    )
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

    # Get embeddings and calculate similarity
    emb_image = get_image_embedding(image, processor, model)
    emb_reference = get_image_embedding(reference, processor, model)
    embedding_similarity = F.cosine_similarity(emb_image, emb_reference).item()

    return {
        "l2": float(l2_distance),
        "clip_vit_b_32": float(embedding_similarity),
    }


@app.function(image=browser_env_image)
async def rasterize_svg(text: str, width: int = 512, height: int = 512) -> bytes:
    from svg_renderer_browser import rasterize_svg

    return await rasterize_svg(text, width, height)


@app.function(secrets=[modal.Secret.from_name("openai")])
async def rollout_step_openai(
    history: list[dict],
    model: str = "gpt-4o-mini",
    settings: dict = {},
):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(model=model, input=history, **settings)
    return response


HOUR = 60 * 60
MINUTE = 60
API_KEY = "sk-research-s4ruchosyo8b"

VLLM_PORT = 8000


@dataclass
class Hyperparameters:
    reward_siglip_weight: float = 0.5
    reward_l2_weight: float = 0.5
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # TODO: add a dataset for the training data
    dataset: str = "trl-lib/tldr"


params = Hyperparameters()


@app.function(image=base_image.pip_install("fastapi", "uvicorn"))
@modal.web_server(8000, startup_timeout=10 * MINUTE)
def my_file_server():
    with open("health_server.py", "w") as f:
        f.write("""
from fastapi import FastAPI
import uvicorn
from time import sleep
from contextlib import asynccontextmanager
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("health_server")
logger.info("Startup.")

wait_time = 2.4
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Lifespan waiting {wait_time} seconds")
    # Simulate slow startup like vllm_serve.py
    sleep(wait_time)
    logger.info("Lifespan ready")
    yield
    logger.info("Lifespan ending...")

app = FastAPI(lifespan=lifespan)

@app.get("/health/")
async def health():
    unique_id = str(uuid.uuid4())
    return {"status": f"waited {wait_time} seconds", "unique_id": unique_id}

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
""")

    import subprocess

    subprocess.Popen("python health_server.py", shell=True)


@app.function(
    max_containers=1,
    image=trl_trainer_image,
    gpu="A100:1",
    # gpu="A100:4",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    # WARN: this is actually not the real batch size, because we're batching in the trainer and sending one request at a time.
    max_inputs=100
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTE)
def vllm_server():
    import subprocess

    print(f"Starting vllm server for {params.model}")
    args = [
        "trl",
        "vllm-serve",
        "--model",
        params.model,
        "--log-level=info",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--max-model-len",
        "32768",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    print(f"$ {' '.join(args)}")
    subprocess.run(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# this will take ~8 days to train. Since we're going to get timed out, we should just run for an hour at a time then save/restore?
@app.function(image=trl_trainer_image, gpu="A100:8", timeout=24 * HOUR)
def train_model_trl(base_model: str = params.model):
    from datasets import load_dataset
    from trl import GRPOTrainer

    vllm_server_url = vllm_server.get_url()

    print(
        f"Hello from the main GPU server, connected to {vllm_server_url} for rollouts."
    )

    dataset = load_dataset(params.dataset, split="train")

    def rendered_code_rewards(completions, **kwargs):
        results = []
        for completion in completions:
            # todo: do we need to extract the svg from the completion?
            image_bytes = rasterize_svg.remote(completion, width=512, height=512)
            image_ref = open("image_ref.png", "rb").read()
            result = compare_images.remote(image_bytes, image_ref)
            print("Comparison results:", result)
            results.append(result["l2"])
        return results

    def text_rewards(completions, **kwargs):
        return [1.0 - len(c) / 8000 for c in completions]

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=[rendered_code_rewards, text_rewards],
        train_dataset=dataset,
        use_vllm=True,
        vllm_server_base_url=vllm_server_url,
    )
    trainer.train()


# todo: rollout_step_vllm

# @app.function(image=base_image)
# def collect_rollout_batch(index: int):
#     from datasets import load_dataset


@app.local_entrypoint()
def test_vllm_server(test_timeout=10 * MINUTE):
    import json
    import time
    import urllib

    vllm_server.spawn(params.model)

    print(f"Running health check for server at {vllm_server.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(
                vllm_server.get_web_url() + "/health"
            ) as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {vllm_server.get_web_url()}"

    print(f"Successful health check for server at {vllm_server.get_web_url()}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(
        f"Sending a sample message to {vllm_server.get_web_url()}", *messages, sep="\n"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"messages": messages, "model": params.model})
    req = urllib.request.Request(
        vllm_server.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))


@app.local_entrypoint()
def main():
    """
    run with:
    modal run run_modal.py
    """
    image_bytes = rasterize_svg.remote(
        "<svg width='512' height='512'><circle cx='256' cy='256' r='192' stroke='black' stroke-width='3' fill='red' /></svg>",
        width=512,
        height=512,
    )
    with open("image.png", "wb") as f:
        f.write(image_bytes)

    # Compare generated image with reference
    image_ref = open("image_ref.png", "rb").read()
    result = compare_images.remote(image_bytes, image_ref)
    print("Comparison results:", result)
