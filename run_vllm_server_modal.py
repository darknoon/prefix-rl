import json
import os
from typing import Any

import aiohttp
import modal

app = modal.App("prefix-rl-vllm-server")


# -----------------------------------------------------------------------------
# Model configuration -----------------------------------------------------------
# -----------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_REVISION = None


# -----------------------------------------------------------------------------
# Set up the container image ----------------------------------------------------
# -----------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    # .run_commands(
    #     # Install system dependencies for Qwen2.5-VL
    #     "apt-get update",
    #     "apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1",
    # )
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "qwen-vl-utils",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "1"})  # Use V1 engine for better performance
    .env({"MODEL_NAME": MODEL_NAME})
)


if not MODEL_NAME:
    raise ValueError("MODEL_NAME is not set")

# -----------------------------------------------------------------------------
# Shared volumes ----------------------------------------------------------------
# -----------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# -----------------------------------------------------------------------------
# Constants ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000
FAST_BOOT = True  # Set to False for better performance if you have multiple replicas


# -----------------------------------------------------------------------------
# vLLM Server function ---------------------------------------------------------
# -----------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)  # how many requests can one replica handle?
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Start a vLLM server for inference."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        os.environ["MODEL_NAME"],
        "--served-model-name",
        os.environ["MODEL_NAME"],
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Qwen2.5-VL specific settings
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        "image=5,video=5",
    ]

    # Add revision if specified
    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


# -----------------------------------------------------------------------------
# Testing functions -------------------------------------------------------------
# -----------------------------------------------------------------------------
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    """Test the vLLM server."""
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    if content is None:
        content = "Hello! How are you today?"

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, MODEL_NAME, messages)
        if twice:
            messages[0]["content"] = "You are a pirate."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, MODEL_NAME, messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    """Send a request to the vLLM server."""
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=1 * MINUTES
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()


# -----------------------------------------------------------------------------
# Local entry-point for custom configuration ------------------------------------
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(test=False):
    """Start the vLLM server with custom configuration."""

    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print("Server will be available at:", serve.get_web_url())

    if test:
        print("Running tests...")
        import asyncio

        asyncio.run(test())


if __name__ == "__main__":
    main()
