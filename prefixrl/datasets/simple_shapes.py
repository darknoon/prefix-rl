from pathlib import Path
from datasets import load_dataset
from .utils import export_parquet_if_changed


def prepare_for_verl(
    source_dataset: str = "darknoon/simple-shapes-svg",
    output_dir: str | Path = "./data/simple-shapes",
    train_split: str = "train[:90%]",
    val_split: str = "train[90%:]",
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
                    "content": (
                        "Please recreate this image: <image> as an svg of width 512 and height 512."
                    ),
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
