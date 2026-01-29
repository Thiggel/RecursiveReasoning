#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="$WORK/.venv"

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

export WANDB_DIR="$WORK/recursive_models/wandb"
mkdir -p "$WANDB_DIR"

set -a
source .env
set +a

export http_proxy="http://proxy:80"
export https_proxy="http://proxy:80"
