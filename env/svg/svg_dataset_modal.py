import modal
from typing import Literal
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    concatenate_datasets,
    Dataset,
)
import os

app = modal.App(name="svg-dataset")

base_image = (
    modal.Image.debian_slim()
    .apt_install("libcairo2")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("nodejs", "npm")
)
image = (
    base_image.add_local_file("env/svg/svg_dataset.py", "/root/svg_dataset.py")
    .add_local_file("env/svg/svg_rlrf_reward.py", "/root/svg_rlrf_reward.py")
    .add_local_file("env/svg/svg_dataset_modal.py", "/root/svg_dataset_modal.py")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
svg_dataset_vol = modal.Volume.from_name("svg-dataset-prep", create_if_missing=True)

volumes = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/svg-dataset-prep": svg_dataset_vol,
}

MINUTES = 60
HOURS = 60 * MINUTES


def get_split_query(split_name: str, test_split_ratio: float = 0.05):
    """
    Returns a [read instruction](https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/builder_classes#datasets.ReadInstruction) for the dataset split
    """
    pct = test_split_ratio * 100
    if split_name == "train":
        return f"train[{pct:0.0f}%:]"
    elif split_name == "test":
        return f"train[:{pct:0.0f}%]"
    elif split_name == "val":
        return "val"


# run this remotely so we can cache the dataset split on the modal volume before we spawn workers
@app.function(
    image=image,
    volumes=volumes,
    max_containers=1,
    timeout=4 * HOURS,
)
def build_dataset(
    input_dataset_name: str = "starvector/svg-stack",
    output_dataset_name: str = "darknoon/svg-stack-filtered",
    tokenizer_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    chunk_size: int = 10_000,
    test_split_ratio: float = 0.05,
):
    def run_split(
        input_dataset_name: str,
        split_name: str,
        tokenizer_name: str,
        chunk_size: int,
    ) -> str:
        output_path = f"/root/svg-dataset-prep/outputs/{split_name}/combined/"
        if os.path.exists(output_path):
            print(
                f"SPLIT ALREADY DONE: Output path {output_path} already exists, skipping!"
            )
            return output_path

        split_query = get_split_query(split_name, test_split_ratio)
        # using the context manager here will close file handles once done
        with load_dataset(input_dataset_name, split=split_query) as split:
            num_rows = len(split)
        print(f"Dataset {split_name}({split_query}) has {num_rows} rows")

        starts = range(0, num_rows, chunk_size)
        ends = list(range(chunk_size, num_rows, chunk_size)) + [num_rows]
        shard_paths = []
        failed_shards = []
        for shard_idx, shard_path in enumerate(
            process_dataset_chunk.map(
                starts,
                ends,
                kwargs={
                    "input_dataset_name": input_dataset_name,
                    "split_name": split_name,
                    "test_split_ratio": test_split_ratio,
                    "tokenizer_name": tokenizer_name,
                },
            )
        ):
            if shard_path is not None:
                print(f"SHARD processed and written to: {shard_path}")
                shard_paths.append(shard_path)
            else:
                print(
                    f"WARNING: Skipping empty shard {shard_idx} ie ({shard_idx * chunk_size} to {(shard_idx + 1) * chunk_size - 1} of {num_rows})"
                )
                failed_shards.append(shard_idx)

        if len(shard_paths) == 0:
            raise ValueError(f"All shards were empty for {split_name} split!")

        # To work around the bug in datasets where we can't call .reload() on the volume do this remotely
        concat_shards.remote(
            shard_paths=shard_paths,
            output_path=output_path,
        )
        return output_path

    print("Building train split")
    train_path = run_split(
        input_dataset_name=input_dataset_name,
        split_name="train",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    print("Building test split")
    test_path = run_split(
        input_dataset_name=input_dataset_name,
        split_name="test",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    print("Building val split")
    val_path = run_split(
        input_dataset_name=input_dataset_name,
        split_name="val",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    # Call this as a separate function to work around file handles not being closed in this container :/
    output_path = "/root/svg-dataset-prep/final/"
    merge_and_push_to_hub.remote(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        output_path=output_path,
        output_dataset_name=output_dataset_name,
    )


@app.function(
    image=image,
    volumes=volumes,
)
def concat_shards(
    shard_paths: list[str],
    output_path: str,
):
    shards = [load_from_disk(shard_path) for shard_path in shard_paths]
    dataset = concatenate_datasets(shards)
    dataset.save_to_disk(output_path)
    return output_path


def add_prompt(svg, width, height):
    def fmt(x: float, max_digits: int = 2) -> str:
        return f"{x:.{max_digits}f}".rstrip("0").rstrip(".")

    return {
        "prompt": f"Please recreate this image <image> as exactly as possible as an SVG of width {fmt(width)} and height {fmt(height)}.",
    }


def add_prompt_to_dataset(dataset: Dataset, num_proc: None):
    return dataset.map(
        add_prompt, input_columns=["Svg", "width", "height"], num_proc=num_proc
    )


# due to a mixup where I didn't add the prompts (but LLama-Factory really wants them unless you want to edit the code, I did this as a backfill)
# modal run env/svg/svg_dataset_modal.py::merge_and_push_to_hub --train-path /root/svg-dataset-prep/outputs/train/combined/ --test-path /root/svg-dataset-prep/outputs/test/combined/ --val-path /root/svg-dataset-prep/outputs/val/combined/ --output-path /root/svg-dataset-prep/final/ --output-dataset-name darknoon/svg-stack-filtered --force
@app.function(
    image=image,
    volumes=volumes,
    secrets=[modal.Secret.from_name("huggingface-write")],
    timeout=90 * MINUTES,
)
def merge_and_push_to_hub(
    train_path: str,
    test_path: str,
    val_path: str,
    output_path: str,
    output_dataset_name: str,
    force: bool = False,
):
    print(f"Loading train split from {train_path}")
    train = load_from_disk(train_path)
    print(f"Loading test split from {test_path}")
    test = load_from_disk(test_path)
    print(f"Loading val split from {val_path}")
    val = load_from_disk(val_path)
    print(f"train: {train.num_rows}, test: {test.num_rows}, val: {val.num_rows}")

    dataset = DatasetDict(
        train=train,
        test=test,
        val=val,
    )

    print("Formatting dataset into prompt/completion format…")
    dataset = dataset.map(
        add_prompt,
        input_columns=["Svg", "width", "height"],
        num_proc=os.cpu_count() or 4,
    )

    print(f"Saving final dataset to {output_path}")
    print(f"train: {train.num_rows}, test: {test.num_rows}, val: {val.num_rows}")
    if not os.path.exists(output_path) or force:
        dataset.save_to_disk(output_path)
    else:
        print(f"Already exists at {output_path}, skipping!")
    print(f"Uploading final dataset to HF as {output_dataset_name}")
    dataset.push_to_hub(output_dataset_name)


@app.function(
    cpu=8,
    image=image,
    volumes=volumes,
    max_containers=64,
    timeout=30 * MINUTES,
)
def process_dataset_chunk(
    range_start: int,
    range_end: int,
    split_name: Literal["train", "val"],
    input_dataset_name: str = None,
    tokenizer_name: str = None,
    test_split_ratio: float = None,
    skip_processing: bool = False,
):
    from svg_dataset import run_all_processing

    print(f"In process_dataset_chunk({range_start}-{range_end})")

    inputs_path = f"/root/svg-dataset-prep/inputs/{split_name}/{range_start:010d}-{range_end:010d}/"
    outputs_path = f"/root/svg-dataset-prep/outputs/{split_name}/{range_start:010d}-{range_end:010d}/"
    if not os.path.exists(inputs_path):
        shard = load_dataset(
            input_dataset_name,
            split=get_split_query(split_name, test_split_ratio),
        ).select(range(range_start, range_end))
        print("Writing inputs to volume")
        shard.save_to_disk(inputs_path)
    else:
        print(f"Inputs already written to {inputs_path}, loading")
        shard = load_from_disk(inputs_path)

    num_rows_before = shard.num_rows
    if num_rows_before != range_end - range_start:
        raise ValueError(
            f"Expected {range_end - range_start} rows, got {num_rows_before}"
        )

    if skip_processing:
        print("skipping actual processing, returning!")
        return inputs_path

    if os.path.exists(outputs_path):
        print(f"Outputs already written to {outputs_path}, skipping!")
        return outputs_path

    shard = run_all_processing(
        subset=shard,
        tokenizer_name=tokenizer_name,
        num_proc=8,
    )

    # Check if we have any data left after filtering
    if len(shard) == 0:
        print(
            f"WARNING: All examples were filtered out in chunk {range_start}-{range_end}"
        )
        return None  # Return None to indicate this chunk should be skipped

    shard.save_to_disk(outputs_path)

    print(
        f"ran {input_dataset_name} ({range_start}-{range_end}) {split_name} -> {shard.num_rows} rows (from {num_rows_before})"
    )
    print("Waiting for volume commit before returning")
    svg_dataset_vol.commit()
    return outputs_path


@app.function(
    image=image,
    volumes=volumes,
)
def cleanup_volume():
    print("Cleaning up volume")
    # svg_dataset_vol.remove_file("/inputs/", recursive=True)


@app.local_entrypoint()
def main():
    build_dataset.remote()
    # cleanup_volume.remote()
