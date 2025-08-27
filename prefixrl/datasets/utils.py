"""Shared dataset utility functions for PrefixRL"""

import json
import pathlib
from datasets import Dataset, __version__ as datasets_version


def export_parquet_if_changed(
    dataset: Dataset, output_path: str, force: bool = False
) -> bool:
    """
    Writes an HF Dataset to parquet only if its fingerprint has changed.

    A small sidecar file `<name>.parquet.meta.json` stores the last fingerprint.

    Args:
        dataset: The HuggingFace Dataset to export
        output_path: Path where the parquet file should be written
        force: If True, always write the file regardless of fingerprint

    Returns:
        True if the file was written, False if it was skipped
    """
    path_obj = pathlib.Path(output_path)
    meta_path = path_obj.with_suffix(".parquet.meta.json")
    fingerprint = getattr(dataset, "_fingerprint", None)

    print(f"Writing {path_obj.name} …")

    # Check if we can skip writing
    if (
        not force
        and fingerprint is not None
        and path_obj.exists()
        and meta_path.exists()
    ):
        try:
            previous = json.loads(meta_path.read_text()).get("fingerprint")
            if previous == fingerprint:
                print(
                    f"  → Skipped {path_obj.name} (unchanged fingerprint: {fingerprint[:8]}...)"
                )
                return False
        except Exception:
            pass  # Fall through to writing if meta is corrupt

    # Write the parquet file
    dataset.to_parquet(str(path_obj))
    if fingerprint is not None:
        meta_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "datasets_version": datasets_version},
                indent=2,
            )
        )

    print(f"  ✓ Wrote {path_obj.name} ({len(dataset):,} examples)")
    return True
