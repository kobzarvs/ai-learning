#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked --extra cpu
uv pip check
cargo fetch --locked --manifest-path examples/rust-burn/Cargo.toml
.venv/bin/python -c 'import sys, torch; print("Python", sys.version.split()[0], "· PyTorch", torch.__version__)'
rustc --version
printf '%s\n' 'AI Atlas готов. Команда для выбранного примера находится на сайте во вкладке Dev Container.'
