"""
Tool definitions for the agentic evaluation harness.
"""

from dataclasses import dataclass
from typing import Any
import json

from .virtual_fs import VirtualFS
from .metrics import MetricsComputer


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    data: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict:
        if self.error:
            return {"success": False, "error": self.error}
        return {"success": True, **self.data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# Tool definitions for Claude's tool_use format
TOOL_DEFINITIONS = [
    {
        "name": "write",
        "description": "Replace the entire content of result.svg with new SVG content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The SVG content to write to result.svg",
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_replace",
        "description": "Replace a string within result.svg. Useful for making small edits without rewriting the entire file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_string": {
                    "type": "string",
                    "description": "The text to find (must exist exactly once in the file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
            },
            "required": ["old_string", "new_string"],
        },
    },
    {
        "name": "rasterize_svg",
        "description": "Rasterize the current result.svg to a PNG image and compute distance metrics against the target image. Returns the generated image filename and metrics (l2_distance, dreamsim_distance). Lower values are better (0 = perfect match).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "zoom_image",
        "description": "Crop and zoom into a specific region of an image for detailed inspection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The image filename to zoom into",
                },
                "top": {
                    "type": "integer",
                    "description": "Top coordinate of the crop region",
                },
                "left": {
                    "type": "integer",
                    "description": "Left coordinate of the crop region",
                },
                "width": {
                    "type": "integer",
                    "description": "Width of the crop region",
                },
                "height": {
                    "type": "integer",
                    "description": "Height of the crop region",
                },
            },
            "required": ["file", "top", "left", "width", "height"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files currently available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read the content of a file. For text files (like .svg), returns the text content. For images, returns a reference to view the image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The filename to read",
                }
            },
            "required": ["file"],
        },
    },
]


class ToolExecutor:
    """Executes tools against a virtual filesystem."""

    def __init__(
        self,
        vfs: VirtualFS,
        target_filename: str = "target.png",
        svg_filename: str = "result.svg",
        raster_min_size: int = 512,
        raster_max_size: int = 1536,
    ):
        self.vfs = vfs
        self.target_filename = target_filename
        self.svg_filename = svg_filename
        self.raster_min_size = raster_min_size
        self.raster_max_size = raster_max_size
        self._metrics_computer: MetricsComputer | None = None
        self._rasterize_counter: int = 0

    @property
    def metrics_computer(self) -> MetricsComputer:
        """Lazy-load the metrics computer."""
        if self._metrics_computer is None:
            self._metrics_computer = MetricsComputer()
        return self._metrics_computer

    def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Execute a tool and return the result."""
        try:
            if tool_name == "write":
                return self._write(tool_input)
            elif tool_name == "search_replace":
                return self._search_replace(tool_input)
            elif tool_name == "rasterize_svg":
                return self._rasterize_svg(tool_input)
            elif tool_name == "zoom_image":
                return self._zoom_image(tool_input)
            elif tool_name == "list_files":
                return self._list_files(tool_input)
            elif tool_name == "read_file":
                return self._read_file(tool_input)
            else:
                return ToolResult(
                    success=False, data={}, error=f"Unknown tool: {tool_name}"
                )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    def _write(self, tool_input: dict) -> ToolResult:
        """Write content to result.svg."""
        content = tool_input.get("content", "")
        self.vfs.write(self.svg_filename, content)
        return ToolResult(
            success=True,
            data={
                "message": f"Successfully wrote {len(content)} bytes to {self.svg_filename}"
            },
        )

    def _search_replace(self, tool_input: dict) -> ToolResult:
        """Search and replace within result.svg."""
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")

        current_content = self.vfs.read_text(self.svg_filename)
        if current_content is None:
            return ToolResult(
                success=False,
                data={},
                error=f"{self.svg_filename} does not exist. Use 'write' first.",
            )

        count = current_content.count(old_string)
        if count == 0:
            return ToolResult(
                success=False,
                data={},
                error=f"String not found in {self.svg_filename}: {old_string[:100]}...",
            )

        new_content = current_content.replace(old_string, new_string)
        self.vfs.write(self.svg_filename, new_content)

        return ToolResult(
            success=True,
            data={
                "message": f"Replaced {count} occurrence(s)",
                "occurrences_replaced": count,
            },
        )

    def _rasterize_svg(self, tool_input: dict) -> ToolResult:
        """Rasterize the SVG and compute metrics."""
        from prefixrl.reward.svg.cairosvg import rasterize_svg

        svg_content = self.vfs.read_text(self.svg_filename)
        if svg_content is None:
            return ToolResult(
                success=False,
                data={},
                error=f"{self.svg_filename} does not exist. Use 'write' first.",
            )

        # Rasterize the SVG
        try:
            generated_image, _, _ = rasterize_svg(
                svg_content,
                min_target=self.raster_min_size,
                max_target=self.raster_max_size,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to rasterize SVG: {e}",
            )

        # Get target image
        target_image = self.vfs.read_image(self.target_filename)
        if target_image is None:
            return ToolResult(
                success=False,
                data={},
                error=f"Target image {self.target_filename} not found",
            )

        # Save the generated image to vfs with incrementing counter
        self._rasterize_counter += 1
        output_filename = f"rasterize_{self._rasterize_counter:03d}.png"
        self.vfs.write_image(output_filename, generated_image)

        # Compute metrics
        metrics = self.metrics_computer.compute(generated_image, target_image)

        return ToolResult(
            success=True,
            data={
                "image_file": output_filename,
                "metrics": metrics,
            },
        )

    def _zoom_image(self, tool_input: dict) -> ToolResult:
        """Zoom into a region of an image."""
        filename = tool_input.get("file", "")
        top = tool_input.get("top", 0)
        left = tool_input.get("left", 0)
        width = tool_input.get("width", 100)
        height = tool_input.get("height", 100)

        image = self.vfs.read_image(filename)
        if image is None:
            return ToolResult(
                success=False,
                data={},
                error=f"Image file not found: {filename}",
            )

        # Crop the image
        try:
            # PIL crop uses (left, top, right, bottom)
            cropped = image.crop((left, top, left + width, top + height))
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to crop image: {e}",
            )

        # Save the cropped image
        output_filename = self.vfs.next_image_filename(prefix="zoom_", suffix=".png")
        self.vfs.write_image(output_filename, cropped)

        return ToolResult(
            success=True,
            data={
                "image_file": output_filename,
                "original_size": {"width": image.width, "height": image.height},
                "crop_region": {
                    "top": top,
                    "left": left,
                    "width": width,
                    "height": height,
                },
            },
        )

    def _list_files(self, tool_input: dict) -> ToolResult:
        """List all files in the virtual filesystem."""
        files = self.vfs.list_files()
        return ToolResult(
            success=True,
            data={"files": files},
        )

    def _read_file(self, tool_input: dict) -> ToolResult:
        """Read a file from the virtual filesystem."""
        filename = tool_input.get("file", "")

        vf = self.vfs.read(filename)
        if vf is None:
            return ToolResult(
                success=False,
                data={},
                error=f"File not found: {filename}",
            )

        if vf.is_image:
            # For images, we return a marker that the evaluator will handle
            return ToolResult(
                success=True,
                data={
                    "type": "image",
                    "file": filename,
                    "message": f"Image file {filename} is available for viewing",
                },
            )
        else:
            # For text files, return the content
            return ToolResult(
                success=True,
                data={
                    "type": "text",
                    "file": filename,
                    "content": vf.as_text(),
                },
            )
