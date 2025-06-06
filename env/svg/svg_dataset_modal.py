import modal
from typing import Literal
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
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


def split_train(dataset: DatasetDict, test_split_ratio: float):
    """
    Split [train] into [train] and [test], keeping [val] as is.
    """
    split = dataset["train"].train_test_split(test_size=test_split_ratio)
    dataset = dataset.copy()
    dataset["train"] = split["train"]
    dataset["test"] = split["test"]
    return dataset

    dataset = split_train(dataset, test_split_ratio)


def load_dataset_split(
    input_dataset_name: str, split_name: str, test_split_ratio: float
):
    print(f"Loading dataset {input_dataset_name}[{split_name}]")
    dataset = load_dataset(input_dataset_name)
    print("Splitting train into train and test")
    dataset = split_train(dataset, test_split_ratio)
    split = dataset[split_name]
    return split


# run this remotely so we can cache the dataset split on the modal volume before we spawn workers
@app.function(
    image=image,
    volumes=volumes,
    max_containers=1,
)
def build_dataset(
    input_dataset_name: str = "starvector/svg-stack",
    tokenizer_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    chunk_size: int = 10_000,
    test_split_ratio: float = 0.05,
):
    def run_split(
        input_dataset_name: str,
        split_name: str,
        tokenizer_name: str,
        chunk_size: int,
    ):
        split = load_dataset_split(input_dataset_name, split_name, test_split_ratio)
        num_rows = len(split)
        print(f"Dataset has {num_rows} rows")

        starts = range(0, num_rows, chunk_size)
        ends = list(range(chunk_size, num_rows, chunk_size)) + [num_rows]
        shard_paths = []
        for shard in process_dataset_chunk.map(
            starts,
            ends,
            kwargs={
                "input_dataset_name": input_dataset_name,
                "split_name": split_name,
                "test_split_ratio": test_split_ratio,
                "tokenizer_name": tokenizer_name,
            },
        ):
            print(f"SHARD processed and written to: {shard}")
            shard_paths.append(shard)

        print("Reloading svg-dataset-prep volume to get new chunks")
        svg_dataset_vol.reload()
        print(f"Loading {len(shard_paths)} shards from {shard_paths}")
        shards = [load_from_disk(shard_path) for shard_path in shard_paths]
        print(f"Concatenating {len(shards)} shards")
        split = concatenate_datasets(shards)
        print(f"Concatenated {len(split)} rows: {split}")
        output_path = f"/root/svg-dataset-prep/outputs/{split_name}/"
        split.save_to_disk(output_path)
        print(f"Saved completed {split_name} to {output_path}")
        return split

    print("Building train split")
    train = run_split(
        input_dataset_name=input_dataset_name,
        split_name="train",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    print("Building test split")
    test = run_split(
        input_dataset_name=input_dataset_name,
        split_name="test",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    print("Building val split")
    val = run_split(
        input_dataset_name=input_dataset_name,
        split_name="val",
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
    )
    output_path = "/root/svg-dataset-prep/final/"
    dataset_processed = DatasetDict(
        train=train,
        test=test,
        val=val,
    )
    print(f"Saving final dataset to {output_path}")
    print(f"train: {train.num_rows}, test: {test.num_rows}, val: {val.num_rows}")
    dataset_processed.save_to_disk(output_path)
    print("Done!")
    return dataset_processed


@app.function(
    cpu=8,
    image=image,
    volumes=volumes,
    max_containers=8,
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
        shard = load_dataset_split(
            input_dataset_name=input_dataset_name,
            split_name=split_name,
            test_split_ratio=test_split_ratio,
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
    shard.save_to_disk(outputs_path)

    print(
        f"ran {input_dataset_name} ({range_start}-{range_end}) {split_name} -> {shard.num_rows} rows (from {num_rows_before})"
    )

    return outputs_path


@app.function(
    image=image,
    volumes=volumes,
)
def cleanup_volume():
    print("Cleaning up volume")
    svg_dataset_vol.remove_file("/inputs/", recursive=True)


@app.local_entrypoint()
def main():
    build_dataset.remote()
    # cleanup_volume.remote()
