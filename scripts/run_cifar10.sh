#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/cifar10_dit.yaml"

CONFIG_PATH="${DEFAULT_CONFIG}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

python "${PROJECT_ROOT}/train_cifar10_dit.py" --config "${CONFIG_PATH}" "$@"
