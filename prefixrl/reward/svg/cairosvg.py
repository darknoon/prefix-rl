import sys
import os

import io
from PIL import Image as PILImage

# Set up DYLD_LIBRARY_PATH for macOS Cairo support (Homebrew installation)
if sys.platform == "darwin":  # macOS
    homebrew_lib = "/opt/homebrew/lib"
    if os.path.exists(homebrew_lib):
        current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
        if homebrew_lib not in current_path:
            os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

# Now we can import cairosvg
import cairosvg.parser
import cairosvg.surface


def get_svg_size(tree: cairosvg.parser.Tree) -> tuple[float, float]:
    width = tree.get("width")
    height = tree.get("height")

    # Handle percentage values - default to 512 for percentages
    if width and width.endswith("%"):
        width = "512"
    if height and height.endswith("%"):
        height = "512"

    if width is None or height is None:
        # Get viewBox if size not specified
        viewbox = tree.get("viewBox")
        if viewbox:
            # viewBox format is "min-x min-y width height"
            parts = viewbox.split()
            if len(parts) == 4:
                width = parts[2]
                height = parts[3]

    # Default to 512x512 if no size info found
    width = width or "512"
    height = height or "512"

    # Strip any remaining non-numeric characters (like 'px')
    width = "".join(c for c in str(width) if c.isdigit() or c == ".")
    height = "".join(c for c in str(height) if c.isdigit() or c == ".")

    # Final fallback if we still don't have valid numbers
    try:
        width = float(width) if width else 512.0
        height = float(height) if height else 512.0
    except ValueError:
        width = 512.0
        height = 512.0

    return width, height


def compute_svg_raster_scale(
    width: float, height: float, min_target: int = 512, max_target: int = 1536
) -> tuple[float, int, int]:
    """
    Compute the scale and output size for rasterizing an SVG such that:
    - The smallest side is at least min_target px
    - The largest side is at most max_target px
    - Aspect ratio is preserved
    Returns: (scale, output_width, output_height)
    """
    # Compute scale factors for both constraints
    scale_min = min_target / min(width, height)
    scale_max = max_target / max(width, height)

    # The scale must be at least scale_min, but not so large that the largest side exceeds max_target
    scale = max(scale_min, 1.0)
    if max(width, height) * scale > max_target:
        scale = scale_max

    output_width = int(round(width * scale))
    output_height = int(round(height * scale))
    return scale, output_width, output_height


def rasterize_svg(
    svg_content: str, min_target: int = 512, max_target: int = 1536
) -> tuple[PILImage.Image, cairosvg.parser.Tree, tuple[float, float]]:
    """Rasterize SVG content to PIL Image using CairoSVG.

    The output image will have its smallest side at least min_target,
    and its longest side at most max_target, preserving aspect ratio.

    Args:
        svg_content: SVG content as string
        min_target: Minimum size for smallest dimension (default: 512)
        max_target: Maximum size for largest dimension (default: 1536)

    Returns:
        tuple: (PIL_Image, Tree, (original_width, original_height))
    """
    image_bytes, tree, size = rasterize_svg_bytes(svg_content, min_target, max_target)
    return PILImage.open(io.BytesIO(image_bytes)), tree, size


def rasterize_svg_bytes(
    svg_content: str, min_target: int = 512, max_target: int = 1536
) -> tuple[bytes, cairosvg.parser.Tree, tuple[float, float]]:
    tree = cairosvg.parser.Tree(bytestring=svg_content.encode("utf-8"))
    # Get the intrinsic width and height from the SVG
    width, height = get_svg_size(tree)

    # Compute scale and output size using the factored-out function
    scale, output_width, output_height = compute_svg_raster_scale(
        width, height, min_target=min_target, max_target=max_target
    )

    output = io.BytesIO()
    dpi = 96
    parent_width = width
    parent_height = height
    background_color = "white"
    instance = cairosvg.surface.PNGSurface(
        tree,
        output,
        dpi,
        None,
        parent_width,
        parent_height,
        scale,
        output_width,
        output_height,
        background_color,
    )
    instance.finish()
    image_bytes = output.getvalue()
    return image_bytes, tree, (width, height)
