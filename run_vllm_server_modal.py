import modal

app = modal.App("prefix-rl-vllm-server")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "qwen-vl-utils",
        "openai",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "1"})  # Use V1 engine for better performance
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MINUTE = 60

# config (can't currently be passed in)
N_GPU = 1
FAST_BOOT = True  # Set to False for better performance if you have multiple replicas
VLLM_PORT = 8000

# account for at least this on cold start
VLLM_STARTUP_TIMEOUT = 4 * MINUTE
VLLM_REQUEST_TIMEOUT = 8 * MINUTE
VLLM_TOTAL_TIMEOUT = VLLM_STARTUP_TIMEOUT + VLLM_REQUEST_TIMEOUT
# misc


# -----------------------------------------------------------------------------
# vLLM Server class (parameterized) --------------------------------------------
# -----------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTE,
    timeout=VLLM_TOTAL_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
class VLLMServer:
    model_name: str = modal.parameter()
    model_revision: str = modal.parameter(default="")

    @modal.web_server(
        port=VLLM_PORT,
        startup_timeout=VLLM_STARTUP_TIMEOUT,
        requires_proxy_auth=True,
    )
    def serve(self):
        """Start a vLLM server for inference."""
        from subprocess import Popen

        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            self.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            # Qwen2.5-VL specific settings
            "--trust-remote-code",
            "--limit-mm-per-prompt",
            "image=5,video=5",
        ]

        if self.model_revision:
            cmd.extend(["--revision", self.model_revision])

        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        cmd_str = " ".join(cmd)
        print(f"Starting '{self.model_name}' vLLM server:\n{cmd_str}")
        Popen(cmd_str, shell=True)
        print("returned from serve(), vllm is starting in the background")


# vLLM Proxy endpoint. This isn't working yet :/

router_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi", "uvicorn", "httpx"
)

ALLOWED_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


@app.function(
    image=router_image,
    timeout=10 * MINUTE,
    secrets=[
        modal.Secret.from_name("vllm-proxy-allowed-keys"),
        modal.Secret.from_name("modal-proxy-key"),
    ],
)
@modal.concurrent(max_inputs=256)
@modal.asgi_app()
def vllm_proxy():
    """Proxy endpoint that routes requests to model-specific vLLM servers."""
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse
    import httpx
    import json
    import os

    router = FastAPI()
    # dummy model name to get the base URL :/
    base_url = VLLMServer(model_name="Qwen/Qwen2.5-VL-7B-Instruct").serve.get_web_url()
    print(f"Starting proxy server to base URL: {base_url}")

    # Resolve approved keys from env or fallback list
    env_keys = os.environ.get("VLLM_PROXY_API_KEYS", "")
    # allow connecting to the private vllm endpoint
    modal_key = os.environ["MODAL_PROXY_KEY"]
    modal_secret = os.environ["MODAL_PROXY_SECRET"]
    approved_keys = set(k.strip() for k in env_keys.split(",") if k.strip())
    print(f"vLLM proxy loaded {len(approved_keys)} approved API key(s)")

    def parse_body(body: bytes) -> dict:
        try:
            return json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    def require_api_key(request: Request):
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header.split(" ", 1)[1].strip()
        if token not in approved_keys:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Proxy chat completions to model-specific server, always streaming response."""
        require_api_key(request)
        raw_body = await request.body()
        body = parse_body(raw_body)

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name required")

        # Modal requires the class parameter for routing to the correct instance
        target_url = f"{base_url}/v1/chat/completions?model_name={model_name}"
        req_headers = {
            "content-type": "application/json",
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        }

        # Modal will throw a fit if any headers like `modal-function-call-id` are set, so filter them out
        def filter_headers(headers: dict[str, str]) -> dict[str, str]:
            return {k: v for k, v in headers.items() if not k.startswith("modal-")}

        client = httpx.AsyncClient(timeout=VLLM_TOTAL_TIMEOUT)
        try:
            req = client.build_request(
                "POST", target_url, content=raw_body, headers=req_headers
            )
            upstream = await client.send(req, stream=True)
            content_type = upstream.headers.get("content-type", "application/json")

            async def body_iter():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()

            # Only forward safe headers; Starlette will set transfer-encoding/content-length as needed
            return StreamingResponse(
                body_iter(),
                status_code=upstream.status_code,
                media_type=content_type,
                headers={"content-type": content_type},
            )
        except Exception:
            await client.aclose()
            raise

    @router.get("/v1/models")
    async def list_models(request: Request):
        require_api_key(request)
        """List available models with container status information."""
        import asyncio

        async def get_model_info(model_name):
            booted = False
            container_info = None
            error = None
            try:
                server = VLLMServer(model_name=model_name)
                # returns FunctionStats(backlog=0, num_total_runners=0)
                container_info = await server.serve.get_current_stats.aio()
                booted = container_info.num_total_runners > 0
            except Exception as e:
                booted = False
                error = str(e)
            return {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "vllm",
                "permission": [],
                "root": model_name,
                "parent": None,
                "modal_container_booted": booted,
                "modal_container_info": {
                    "num_total_runners": container_info.num_total_runners,
                    "backlog": container_info.backlog,
                    "_raw": str(container_info),
                }
                if container_info
                else None,
                "modal_error": error,
            }

        # Run all get_current_stats in parallel
        models = await asyncio.gather(
            *(get_model_info(model_name) for model_name in ALLOWED_MODELS)
        )

        return {"object": "list", "data": models}

    @router.get("/health")
    async def health():
        """Proxy health check."""
        return {"status": "healthy", "type": "proxy"}

    return router


# -----------------------------------------------------------------------------
# Testing functions -------------------------------------------------------------
# -----------------------------------------------------------------------------
async def _test_chat_completion(model_name: str, base_url: str, api_key: str = None):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        default_query={"model_name": model_name},
        api_key=api_key or "dummy-key",
    )

    messages = [
        {"role": "user", "content": "Hello"},
    ]

    print(f"Sending messages to {base_url}:", *messages, sep="\n\t")
    stream = await client.chat.completions.create(
        model=model_name, messages=messages, stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    print("Done")


async def _health_check(base_url: str, health_path: str = "/health"):
    from aiohttp import ClientSession

    async with ClientSession(base_url=base_url) as session:
        # health check will hang for a while if the server is starting up
        async with session.get(health_path, timeout=VLLM_REQUEST_TIMEOUT) as resp:
            assert resp.status == 200, f"Health check failed with status {resp.status}"


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Quick test of the vLLM server.
    For running the server just do `modal serve run_vllm_server_modal.py`
    """
    import asyncio
    import os

    use_proxy = True

    async def test_vllm_direct(model_name: str, base_url: str):
        await _health_check(base_url, health_path=f"/health?model_name={model_name}")
        await _test_chat_completion(model_name, base_url)

    async def test_vllm_proxy(model_name: str, proxy_url: str):
        await _health_check(proxy_url)
        await _test_chat_completion(
            model_name, proxy_url, api_key=os.environ["API_KEY"]
        )

    if use_proxy:
        url = vllm_proxy.get_web_url()
        asyncio.run(test_vllm_proxy(model_name, url))
    else:
        # the model_name= doesn't actually do much b/c it's shared for all parameterizations
        url = VLLMServer(model_name=model_name).serve.get_web_url()
        asyncio.run(test_vllm_direct(model_name, url))
