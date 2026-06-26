#!/bin/bash
set -euo pipefail

# Optional PJLab-cluster helper.
#
# Source this file before running the demos only if you are using the same
# cluster-style cache/checkpoint layout. Otherwise, ignore this file and set
# PI0_PATH / PI05_PATH / MUJOCO_GL yourself.
#   source scripts/setup_cluster_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_ROOT="${USER_ROOT:-${HOME}}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${USER_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${XDG_CACHE_HOME}/libero/assets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${USER_ROOT}/dataset}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

export PI05_PATH="${PI05_PATH:-}"
export PI0_PATH="${PI0_PATH:-}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "LIBERO_ASSETS_PATH=${LIBERO_ASSETS_PATH}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PI05_PATH=${PI05_PATH}"
echo "PI0_PATH=${PI0_PATH}"
