from pathlib import Path
from datasets import load_dataset
from .utils import export_parquet_if_changed


def prepare_for_verl(
    source_dataset: str = "darknoon/simple-shapes-svg",
    output_dir: str | Path = "./data/simple-shapes",
    train_split: str = "train[:-128]",
    # val split is the last 128 examples, so it doesn't take too long to evaluate
    val_split: str = "train[-128:]",
    num_proc: int = 16,
    force: bool = False,
) -> dict[str, str]:
    """Prepare simple shapes SVG dataset for verl training.

    - Loads darknoon/simple-shapes-svg dataset splits
    - Adds `data_source` field used by reward function routing
    - Injects `prompt` column with instructions for 512x512 SVGs

    Returns mapping of split name to parquet path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    print(f"Loading source dataset: {source_dataset} …")
    train_ds = load_dataset(source_dataset, split=train_split)
    val_ds = load_dataset(source_dataset, split=val_split)

    def process(example):
        ground_truth = example["svg"]
        image = example["image"]

        return {
            # Convert prompt string to messages
            "prompt": [
                {
                    "role": "user",
                    # ~125 tokens + image tokens
                    "content": """\
You first analyze the input image, think about how to convert it into SVG format, then generate SVG code that would render the image exactly as you see it. Think about the key shapes, paths, and visual elements that need to be represented and their x/y coordinates and any nesting or transformations necessary.

The svg should have width 512 and height 512.

Input image:
<image>

Formatting:
Your reasoning process MUST BE enclosed within <think></think> tags, immediately followed by <answer><svg ...></svg></answer> tags containing the final SVG code. No other text or backticks are permitted, or you will not receive any credit for your answer.
""",
                }
            ],
            # Verl wants images as a list
            "images": [image],
            # Add data source for reward routing
            "data_source": source_dataset,
            # Add ground truth (input to reward function)
            "reward_model": {"ground_truth": ground_truth},
        }

    # Remove any other columns
    cols = train_ds.column_names
    # Process and remove unwanted columns in one step
    train_ds = train_ds.map(process, remove_columns=cols, num_proc=num_proc)
    val_ds = val_ds.map(process, remove_columns=cols, num_proc=num_proc)

    export_parquet_if_changed(train_ds, train_path, force=force)
    export_parquet_if_changed(val_ds, val_path, force=force)

    return {"train": train_path, "val": val_path}
