from __future__ import annotations

import os
import subprocess
from pathlib import Path


def resolve_dreamsim_cache_dir(default: str = "./models") -> str:
    env_dir = os.environ.get("DREAMSIM_CACHE_DIR")
    if env_dir:
        return env_dir

    repo_root = _git_common_root()
    if repo_root is None:
        return default

    return str(repo_root / "models")


def _git_common_root() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None

    git_common_dir = result.stdout.strip()
    if not git_common_dir:
        return None

    common_dir = Path(git_common_dir).expanduser()
    if not common_dir.is_absolute():
        common_dir = (Path.cwd() / common_dir).resolve()

    return common_dir.parent
