#!/bin/bash
set -euo pipefail

# Source this file before running the demos on the PJLab cluster:
#   source scripts/setup_cluster_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/project/vla_interpretability_handoff}"
USER_ROOT="${USER_ROOT:-/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao}"

export XDG_CACHE_HOME="${USER_ROOT}/cache"
export HF_HOME="${USER_ROOT}/cache/huggingface"
export TORCH_HOME="${USER_ROOT}/cache/torch"
export LIBERO_ASSETS_PATH="${USER_ROOT}/cache/libero/assets"
export HF_LEROBOT_HOME="${USER_ROOT}/dataset"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

export PI05_PATH="${PI05_PATH:-${USER_ROOT}/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/dbf8a3f794a9c4297b44f40b752712f50073d945}"
export PI0_PATH="${PI0_PATH:-${USER_ROOT}/cache/huggingface/hub/models--lerobot--pi0_libero_finetuned/snapshots/45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f}"

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
