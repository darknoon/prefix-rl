import modal
from modal import Image

app = modal.App("prefix-rl")

browser_env_image = (
    Image.debian_slim(python_version="3.12")
    # https://modal.com/docs/examples/web-scraper#a-simple-web-scraper
    .run_commands(
        "apt-get update",
        "apt-get install -y software-properties-common",
        "apt-add-repository non-free",
        "apt-add-repository contrib",
        "pip install playwright==1.42.0",
        "playwright install-deps chromium",
        "playwright install chromium",
    )
    .add_local_file("svg_renderer.py", "/root/svg_renderer.py")
)


@app.function(image=browser_env_image)
async def env_svg(text: str, width: int = 512, height: int = 512) -> bytes:
    from svg_renderer import rasterize_svg

    image_bytes = await rasterize_svg(text, width, height)
    return image_bytes


@app.function(secrets=[modal.Secret.from_name("openai")])
async def rollout_step_openai(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    settings: dict = {},
):
    from openai import OpenAI

    client = OpenAI(api_key=modal.secrets.openai.api_key)
    response = client.chat.completions.create(
        model=model, messages=messages, **settings
    )
    return response.choices[0].message.content


# todo: rollout_step_vllm


@app.local_entrypoint()
def main():
    """
    run with:
    modal run run_modal.py
    """
    image_bytes = env_svg.remote(
        "<svg width='512' height='512'><circle cx='256' cy='256' r='192' stroke='black' stroke-width='3' fill='red' /></svg>",
        width=512,
        height=512,
    )
    with open("image.png", "wb") as f:
        f.write(image_bytes)
