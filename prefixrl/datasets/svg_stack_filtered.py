from pathlib import Path
from datasets import load_dataset
from .utils import export_parquet_if_changed


# Prepare for verl GRPO training (don't need completion in this case)
def prepare_for_verl(
    output_dir: str | Path = "./data/svg-rlrf",
    train_split: str = "train[:1000]",
    val_split: str = "test[:1000]",
    num_proc: int = 16,
    force: bool = False,
) -> dict[str, str]:
    """Prepare SVG Stack Filtered dataset for VERL training.

    - Loads darknoon/svg-stack-filtered dataset splits
    - Adds `data_source` field used by reward function routing
    - Injects `prompt` column with instructions using width/height from data
    - Writes train/val parquet with fingerprint-gated export

    Returns mapping of split name to parquet path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    source_dataset = "darknoon/svg-stack-filtered"

    print(f"Loading source dataset: {source_dataset} …")
    train_ds = load_dataset(source_dataset, split=train_split)
    val_ds = load_dataset(source_dataset, split=val_split)

    def process(example):
        # Add data source for reward routing
        example["data_source"] = source_dataset
        # Convert prompt string to messages
        prompt = example["prompt"]
        example["prompt"] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        # Don't need a completion
        return example

    keep_columns = ["prompt", "images", "data_source"]
    remove_columns = [col for col in train_ds.column_names if col not in keep_columns]

    # Process and remove unwanted columns in one step
    train_ds = train_ds.map(process, remove_columns=remove_columns, num_proc=num_proc)
    val_ds = val_ds.map(process, remove_columns=remove_columns, num_proc=num_proc)

    export_parquet_if_changed(train_ds, train_path, force=force)
    export_parquet_if_changed(val_ds, val_path, force=force)

    return {"train": train_path, "val": val_path}
