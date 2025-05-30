from cairosvg import svg2png


def rasterize_svg(svg_content: str, width=1200, height=800) -> bytes:
    """Rasterize SVG content to PNG image bytes using CairoSVG."""
    return svg2png(
        bytestring=svg_content.encode("utf-8"), output_width=width, output_height=height
    )


if __name__ == "__main__":
    svg_content = """
    <svg width='512' height='512'><circle cx='256' cy='256' r='192' stroke='black' stroke-width='3' fill='red' /></svg>
    """

    image_bytes = rasterize_svg(svg_content, width=512, height=512)
    with open("image_cairo.png", "wb") as f:
        f.write(image_bytes)
