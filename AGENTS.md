See README.md for more details.

Project structure:
- env/svg/ - SVG rendering task
- run_easyr1_modal.py - Launches the training loop on modal

Basic setup:

- We use uv to manage the dependencies.
- Since you don't have normal internet access, expect to not be able to install more dependencies. However you do have all of the dependencies in pyproject.toml installed already as part of setup.

Python Code:
- use `float | None` instead of `Optional[float]`

Before running anything you must activate the virtual environment with:
```sh
source .venv/bin/activate
```
