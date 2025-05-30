from contextlib import asynccontextmanager
from playwright.async_api import async_playwright


@asynccontextmanager
async def browser(width=1200, height=800):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})
        try:
            yield page
        finally:
            await browser.close()


async def rasterize_svg(svg_content: str, width=1200, height=800) -> bytes:
    """Rasterize SVG content to PNG image bytes using browser rendering."""
    async with browser(width, height) as page:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; }}
                svg {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            {svg_content}
        </body>
        </html>
        """
        await page.set_content(html_content)
        image_bytes = await page.screenshot(type="png", scale="device", full_page=True)
        return image_bytes


if __name__ == "__main__":
    import asyncio

    svg_content = """
    <svg width='512' height='512'><circle cx='256' cy='256' r='192' stroke='black' stroke-width='3' fill='red' /></svg>
    """

    image_bytes = asyncio.run(rasterize_svg(svg_content, width=512, height=512))
    with open("image.png", "wb") as f:
        f.write(image_bytes)
