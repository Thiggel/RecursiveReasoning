#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="$WORK/.venv"

uv venv --python 3.12 "${UV_PROJECT_ENVIRONMENT}"
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
uv pip install -r "requirements.txt"
