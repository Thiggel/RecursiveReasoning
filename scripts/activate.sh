#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="$WORK/.venv"

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
