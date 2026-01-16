# Rasterize-Zoom Agentic Evaluation Harness

An agentic evaluation harness that allows models to iteratively refine their SVG output through multiple turns, using tools to inspect and modify their work.

## Overview

Unlike the single-shot evaluation in `svg_eval.py`, this harness gives the model multiple attempts to get the SVG right. The model can:

1. Write/edit SVG content
2. Rasterize and compare against the target image
3. Zoom in on specific regions to inspect details
4. Iterate until satisfied or turn limit reached

## Virtual Filesystem

To avoid sandboxing complexity, each model instance operates on a **virtual filesystem** - files are stored in a Python dict in memory. The model doesn't need to know this; it just uses the tools with filenames.

### Initial Files Available

| Filename | Description |
|----------|-------------|
| `00001_target.png` | The target image the model must recreate |
| `result.svg` | The output SVG file (starts empty) |

## Tools

### `write`

Replace the entire content of the SVG file.

**Parameters:**
- `content` (str): The SVG content to write

**Returns:** Confirmation message

**Example:**
```json
{
  "tool": "write",
  "content": "<svg width=\"100\" height=\"100\"><circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"red\"/></svg>"
}
```

---

### `search_replace`

Replace a string within the current SVG file. Useful for making small edits without rewriting everything.

**Parameters:**
- `old_string` (str): The text to find (must exist in the file)
- `new_string` (str): The replacement text

**Returns:** Confirmation message or error if old_string not found

**Example:**
```json
{
  "tool": "search_replace",
  "old_string": "fill=\"red\"",
  "new_string": "fill=\"blue\""
}
```

---

### `rasterize_svg`

Rasterize the current `result.svg` to a PNG image and compute distance metrics against the target.

**Parameters:** None

**Returns:**
- `image_file` (str): Filename of the generated raster image (e.g., `00002_attempt.png`)
- `metrics`:
  - `l2_distance` (float): Normalized L2 pixel distance (0 = identical, lower is better)
  - `dreamsim_distance` (float): Perceptual similarity using DreamSim (0 = identical, lower is better)

**Example Response:**
```json
{
  "image_file": "00002_attempt.png",
  "metrics": {
    "l2_distance": 0.0832,
    "dreamsim_distance": 0.2451
  }
}
```

---

### `zoom_image`

Crop and zoom into a specific region of an image for detailed inspection.

**Parameters:**
- `file` (str): The image filename to zoom into
- `bbox` (object): Bounding box with `top`, `left`, `width`, `height` (all integers)

**Returns:**
- `image_file` (str): Filename of the zoomed/cropped image

**Example:**
```json
{
  "tool": "zoom_image",
  "file": "00002_attempt.png",
  "bbox": {"top": 100, "left": 100, "width": 200, "height": 200}
}
```

---

### `list_files`

List all files currently available in the virtual filesystem.

**Parameters:** None

**Returns:** List of filenames

---

### `read_file`

Read the content of a file. For images, returns the image data (for multimodal models). For SVG/text files, returns the text content.

**Parameters:**
- `file` (str): The filename to read

**Returns:** File content (text or image data)

---

## Turn Limits

The model is limited to a configurable number of **turns**. A turn is defined as:

1. Model receives context (messages + tool results)
2. Model produces a response (possibly with tool calls)
3. Tool calls are executed and results added to context

**Default:** 10 turns

The evaluation ends when:
- The model indicates it's done (returns without tool calls)
- The turn limit is reached
- An unrecoverable error occurs

## Debug Output

For each evaluation instance, the harness writes detailed debug information:

### `debug/`

```
debug/
├── 00001/
│   ├── turn_01_response.md      # Model's response text
│   ├── turn_01_tool_calls.json  # Tool calls made
│   ├── turn_01_result.svg       # SVG state after this turn
│   ├── turn_01_raster.png       # Rasterized image (if rasterize was called)
│   ├── turn_01_metrics.json     # Metrics from this turn
│   ├── turn_02_response.md
│   ├── ...
│   ├── final_result.svg         # Final SVG output
│   ├── final_result.png         # Final rasterized image
│   └── summary.json             # Overall metrics and turn count
```

### `summary.json` Schema

```json
{
  "id": "00001",
  "target_image": "00001_target.png",
  "total_turns": 5,
  "final_metrics": {
    "l2_distance": 0.0234,
    "dreamsim_distance": 0.0891
  },
  "turn_history": [
    {
      "turn": 1,
      "tool_calls": ["write"],
      "metrics": {"l2_distance": 0.2341, "dreamsim_distance": 0.4521}
    },
    {
      "turn": 2,
      "tool_calls": ["rasterize_svg"],
      "metrics": {"l2_distance": 0.2341, "dreamsim_distance": 0.4521}
    },
    ...
  ],
  "status": "completed",  // or "turn_limit_reached", "error"
  "error": null
}
```

## Configuration

```python
@dataclass
class AgenticEvalConfig:
    max_turns: int = 10
    
    # Rasterization settings
    raster_min_size: int = 512
    raster_max_size: int = 1536
    
    # Whether to include canny edge detection in metrics
    include_canny_metrics: bool = False
    
    # Debug output
    debug_output: bool = True
    debug_dir: Path = Path("debug")
```

## Usage

### Programmatic

```python
from prefixrl.harness.rasterize_zoom import AgenticEvaluator, AgenticEvalConfig

config = AgenticEvalConfig(max_turns=10, debug_output=True)
evaluator = AgenticEvaluator(config)

# Run evaluation on a single example
result = await evaluator.evaluate(
    target_image=target_pil_image,
    model_client=my_client,  # Callable that handles tool-use conversation
    example_id="00001"
)

print(f"Final L2 distance: {result.final_metrics.l2_distance}")
print(f"Turns used: {result.total_turns}")
```

### CLI

```bash
# Run with Claude (requires ANTHROPIC_API_KEY)
uv run python -m prefixrl.harness.rasterize_zoom \
    --dataset simple-shapes \
    --model_name claude-sonnet-4-20250514 \
    --max_turns 10 \
    -n 10

# Output will be in: eval/darknoon_simple-shapes-svg_claude-sonnet-4-20250514_rasterize_zoom/
```

Available options:

```
--dataset {simple-shapes,svg-stack}  Dataset to evaluate on
-n, --num_examples N                 Number of examples (default: 10)
--model_name MODEL                   Claude model to use
--max_turns N                        Maximum turns per example (default: 10)
--output_dir DIR                     Custom output directory
--num_workers N                      Parallel workers (default: 1)
--temperature TEMP                   Generation temperature (default: 1.0)
--no_debug                           Disable debug output
```

## System Prompt Template

The model receives a system prompt explaining the task and available tools:

```
You are an AI assistant that recreates images as SVG files.

## Task
Recreate the target image as accurately as possible as an SVG.

## Available Files
- `00001_target.png`: The image you must recreate

## Available Tools
[Tool descriptions...]

## Strategy Tips
1. Start by analyzing the target image to understand its structure
2. Write an initial SVG attempt
3. Use `rasterize_svg` to see how your SVG compares to the target
4. Look at the metrics - lower is better (0 = perfect match)
5. Use `zoom_image` to inspect specific regions if needed
6. Iterate with `search_replace` for small fixes or `write` for major changes
7. Continue until you're satisfied or can't improve further

When you're done, respond without any tool calls.
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgenticEvaluator                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Virtual   │     │    Tool     │     │   Metrics   │       │
│  │ Filesystem  │◄────│  Executor   │────►│  Computer   │       │
│  │  (dict)     │     │             │     │             │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                   ▲                   │               │
│         │                   │                   │               │
│         ▼                   │                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │    Debug    │     │   Model     │     │   Image     │       │
│  │   Writer    │     │   Client    │◄────│  Comparator │       │
│  │             │     │ (external)  │     │             │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Extending

### Adding New Tools

1. Define the tool in `tools.py`:

```python
@tool_registry.register
class MyNewTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {...}
    
    def execute(self, vfs: VirtualFS, **kwargs) -> ToolResult:
        # Implementation
        pass
```

2. Add to the system prompt tool descriptions

### Custom Metrics

The `MetricsComputer` class can be extended to add custom image comparison metrics:

```python
class CustomMetrics(MetricsComputer):
    def compute(self, generated: Image, target: Image) -> dict:
        base_metrics = super().compute(generated, target)
        base_metrics["my_custom_metric"] = my_custom_function(generated, target)
        return base_metrics
```
