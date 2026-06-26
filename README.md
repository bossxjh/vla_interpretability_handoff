# VLA Interpretability Handoff

This repository is a compact handoff version of the VLA interpretability project. It keeps only three reproducible demos:

1. **PI0.5 layerwise linear probing** on LIBERO states.
2. **PI0 closed-loop rollout tracing** with rollout videos, full-token activations, rollout-time probes, and dashboard videos.
3. **PI0 activation ablation sweep** over 36 layers and 96 token bins.

Static S2D evaluation scripts, OpenVLA analysis, and exploratory notebooks/results were intentionally removed from this handoff repo.

## Download

Clone the handoff repository:

```bash
git clone https://github.com/bossxjh/vla_interpretability_handoff.git
cd vla_interpretability_handoff
```

All generated artifacts are ignored by git and should be written under `outputs/` or a large cluster storage path.

## Environment

There are two levels of environment setup:

- **Analysis-only environment**: enough for plotting, probe training from existing `.npz` activations, and CSV heatmaps.
- **Real PI0 / PI0.5 + LIBERO environment**: required for activation extraction, closed-loop rollouts, and ablation sweeps.

### Option A: Analysis-Only Environment

Use this if the activations or rollout CSVs have already been generated:

```bash
conda env create -f environment.yml
conda activate vla-interpretability
python -m pip install -r requirements.txt
```

Quick check:

```bash
python -m py_compile \
  scripts/03_train_layerwise_probe.py \
  scripts/05_plot_results.py \
  scripts/18_plot_pi0_ablation_heatmaps.py
```

### Option B: Real PI0 / PI0.5 + LIBERO Environment

Real model and simulator runs require Linux + GPU + MuJoCo/EGL. A generic setup is:

```bash
conda create -y -n vla-interpretability-real python=3.10
conda activate vla-interpretability-real
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "lerobot[pi,libero]@git+https://github.com/huggingface/lerobot.git"
```

On a headless cluster node, also install or provide EGL/OpenGL runtime libraries:

```bash
export MUJOCO_GL=egl
sudo apt update
sudo apt install -y libegl1 libopengl0 libgl1 mesa-utils
```

If `sudo` is unavailable inside the job container, use an image that already contains these libraries.

### Checkpoints and Optional Cache Settings

The core code only requires checkpoint paths and normal MuJoCo/LeRobot environment variables. Set these for your own machine:

```bash
export MUJOCO_GL=egl
export PI05_PATH=/path/to/pi05_libero_finetuned_snapshot_or_repo
export PI0_PATH=/path/to/pi0_libero_finetuned_snapshot_or_repo
```

Optional cache variables, useful on clusters but not required:

```bash
export HF_HOME=/path/to/huggingface_cache
export TORCH_HOME=/path/to/torch_cache
export LIBERO_ASSETS_PATH=/path/to/libero/assets
export HF_LEROBOT_HOME=/path/to/lerobot_dataset_cache
```

If the checkpoints are not present locally, download them first or point `PI05_PATH` / `PI0_PATH` to the local snapshot directories. Use `HF_HUB_OFFLINE=1` only if the checkpoints are already available locally.

For the original PJLab cluster layout used during development, `scripts/setup_cluster_env.sh` can be sourced after overriding `USER_ROOT`, `PI0_PATH`, and `PI05_PATH` as needed. If you are not on that cluster, you can ignore this helper.

### Environment Smoke Tests

Run these before launching long jobs:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy
from lerobot.envs.configs import LiberoEnv
print("lerobot pi/libero imports OK")
PY
```

Check LIBERO rendering:

```bash
python - <<'PY'
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("LIBERO_ASSETS_PATH =", os.environ.get("LIBERO_ASSETS_PATH"))
PY
```

## Demo 1: PI0.5 Layerwise Linear Probing

**Goal.** Test whether PI0.5 hidden states linearly encode task-relevant visuomotor variables. The main figure is a multi-curve R2 plot, `Layerwise linear decodability across probe targets`, covering:

- `offset`: target position minus gripper position.
- `target_position`: absolute target object position.
- `gripper_position`: absolute gripper/end-effector position.
- `action`: model single-step action output.
- `action_chunk`: flattened model action chunk target, when available.
- `gt_action` and `gt_action_chunk`: ground-truth action labels, when present in the state file.

**Experimental setting.** We sample LIBERO states from an early task phase, e.g. LIBERO-Spatial task 1. Each state stores RGB image, instruction, gripper position, target position, target offset, and optional ground-truth action labels. PI0.5 is run once per state. For every transformer layer, the sequence hidden state is mean-pooled over tokens into one layer-level representation. A ridge linear probe is then trained layer-by-layer.

**Reproduce.**

First collect or provide states:

```bash
python scripts/01_collect_states.py \
  --config configs/demo.yaml \
  --env libero_dataset \
  --task libero_spatial \
  --task-id 1 \
  --num-samples 500
```

Extract PI0.5 activations:

```bash
python scripts/02_extract_activations.py \
  --config configs/demo.yaml \
  --model pi05 \
  --pi05-path "$PI05_PATH"
```

Train all probes and plot the combined R2 figure:

```bash
python scripts/03_train_layerwise_probe.py --config configs/demo.yaml --target all
python scripts/05_plot_results.py --config configs/demo.yaml --target all
```

Expected outputs:

```text
outputs/activations/activations.npz
outputs/probes/layerwise_probe_results*.csv
outputs/probes/layerwise_probe_target_summary.csv
outputs/figures/layerwise_probe_targets_r2_comparison.png
```

For a smoke test:

```bash
python scripts/02_extract_activations.py --config configs/demo.yaml --model pi05 --pi05-path "$PI05_PATH" --max-samples 10
```

## Demo 2: PI0 Closed-Loop Rollout Tracing and Dynamic Dashboard

**Goal.** Observe PI0 during an actual LIBERO closed-loop rollout. The demo saves the robot execution video, policy outputs, robot/object metadata, and full-token activations for every captured replan/step. A second analysis stage trains rollout-time probes and renders a dashboard video that aligns:

- LIBERO execution image.
- Layer x token-bin activation heatmap.
- Layer x probe-target error heatmap.

**Experimental setting.** The default task is LIBERO-Spatial task 1:

```text
pick up the black bowl from table center and place it on the plate
```

By default, the policy replans every environment step and saves all 36 layers of full-token activations in float16. This is IO-heavy, so for long rollouts write to a large GPFS/TOS path.

**Collect rollouts.**

Choose an output directory first:

```bash
export ROLLOUT_DIR="$PWD/outputs/rollouts/pi0_libero_spatial_task1_full_tokens_30interval"
```

```bash
python scripts/09_collect_pi0_libero_rollouts.py \
  --config configs/demo.yaml \
  --pi0-path "$PI0_PATH" \
  --task libero_spatial \
  --task-id 1 \
  --instruction "pick up the black bowl from table center and place it on the plate" \
  --output-dir "$ROLLOUT_DIR" \
  --num-episodes 2 \
  --max-steps 250 \
  --replan-interval 30 \
  --save-video \
  --save-activations \
  --video-format mp4
```

Useful variants:

```bash
# Replan every step
--force-replan-every-step --replan-interval 1

# Less IO, capture less often
--no-save-activations
```

Expected rollout structure:

```text
rollout_dir/
  summary.json
  token_layout.json
  episode_000/
    steps.jsonl
    videos/*.mp4
    activations/step_*/layer_*.npz
```

**Analyze rollout-time probes and candidate dynamic circuit edges.**

```bash
python scripts/13_analyze_pi0_dynamic_circuit.py \
  --config configs/demo.yaml \
  --rollout-dir "$ROLLOUT_DIR" \
  --pooling mean \
  --targets pickup_offset place_offset action policy_pred_action progress
```

This writes a timestamped run under the corresponding `VLA-Probe-Analysis` directory.

**Render dashboard video.**

Set `ANALYSIS_DIR` to the timestamped directory printed by `scripts/13_analyze_pi0_dynamic_circuit.py`, for example `outputs/analysis/.../runs/20260626_120000_pool-mean_seed-42`.

```bash
export ANALYSIS_DIR=/path/to/pi0_dynamic_analysis_run

python scripts/14_render_pi0_dynamic_episode_video.py \
  --analysis-dir "$ANALYSIS_DIR" \
  --rollout-dir "$ROLLOUT_DIR" \
  --episode-index 0 \
  --targets pickup_offset place_offset action policy_pred_action progress \
  --format mp4 \
  --tmp-dir /tmp
```

Expected analysis outputs:

```text
pi0_dynamic_activation_manifest.csv
pi0_dynamic_activation_norms.csv
pi0_dynamic_samples.csv
pi0_dynamic_probe_summary.csv
pi0_dynamic_probe_*_sample_predictions.csv
pi0_dynamic_circuit_nodes.csv
pi0_dynamic_circuit_edges.csv
pi0_dynamic_episode_000_dashboard.mp4
```

## Demo 3: PI0 Full Activation Ablation Sweep

**Goal.** Measure which PI0 layer/token regions causally influence closed-loop policy behavior. Each ablation condition zeros one layer and one token bin during policy forward passes, then compares the resulting rollout against a baseline rollout.

**Experimental setting.**

- 36 layers.
- 96 token bins.
- `bin_stride = 4`, so each layer scans 24 bins.
- Total conditions: `36 x 24 = 864`.
- Default sharding: 8 shards, around 108 conditions per shard.
- Default sweep outputs only CSV files: no videos and no saved activations.

Primary metric for heatmaps:

```text
mean_policy_action_delta_l2
```

This is the average L2 difference between the ablated policy action and the baseline policy action at matched rollout steps. Larger values mean the ablated layer/bin has larger causal effect on action output.

Choose an output directory:

```bash
export ABLATION_DIR="$PWD/outputs/ablation/pi0_ablation_spatial_task1_full_sweep"
```

**Run one baseline.**

```bash
python scripts/16_sweep_pi0_activation_ablation.py \
  --config configs/demo.yaml \
  --pi0-path "$PI0_PATH" \
  --output-dir "$ABLATION_DIR" \
  --task libero_spatial \
  --task-id 1 \
  --instruction "pick up the black bowl from table center and place it on the plate" \
  --num-episodes 2 \
  --max-steps 250 \
  --layers all \
  --token-bins 96 \
  --bin-stride 4 \
  --baseline-only \
  --no-save-video \
  --no-save-activations
```

**Run a shard.**

```bash
python scripts/16_sweep_pi0_activation_ablation.py \
  --config configs/demo.yaml \
  --pi0-path "$PI0_PATH" \
  --output-dir "$ABLATION_DIR" \
  --baseline-dir "$ABLATION_DIR/baseline" \
  --skip-baseline \
  --task libero_spatial \
  --task-id 1 \
  --instruction "pick up the black bowl from table center and place it on the plate" \
  --num-episodes 2 \
  --max-steps 250 \
  --layers all \
  --token-bins 96 \
  --bin-stride 4 \
  --num-shards 8 \
  --shard-index 0 \
  --no-save-video \
  --no-save-activations
```

Repeat `--shard-index 0..7`.

**Submit via rjob.**

The helper scripts are:

```text
scripts/rjob_pi0_ablation_worker.sh
scripts/rjob_submit_pi0_ablation_sweep.sh
```

Example:

```bash
NUM_SHARDS=8 \
NUM_EPISODES=2 \
MAX_STEPS=250 \
BIN_STRIDE=4 \
PI0_PATH="$PI0_PATH" \
OUTPUT_DIR="$ABLATION_DIR" \
bash scripts/rjob_submit_pi0_ablation_sweep.sh
```

The rjob submitter is a PJLab-style template. On another cluster, either run `scripts/16_sweep_pi0_activation_ablation.py` directly or edit the rjob resource flags and mount settings in `scripts/rjob_submit_pi0_ablation_sweep.sh`.

If the cluster allows only four concurrent jobs, use:

```bash
NUM_SHARDS=4 BIN_STRIDE=4 bash scripts/rjob_submit_pi0_ablation_sweep.sh
```

**Merge and plot.**

```bash
python scripts/17_merge_pi0_ablation_shards.py \
  --input-dir "$ABLATION_DIR"

python scripts/18_plot_pi0_ablation_heatmaps.py \
  --input-dir "$ABLATION_DIR" \
  --metrics mean_policy_action_delta_l2 mean_gripper_position_delta_l2 success_gain \
  --annotate-top 20 \
  --top-k 50
```

For one unmerged shard:

```bash
python scripts/19_plot_single_pi0_ablation_shard.py \
  "$ABLATION_DIR/shard_00_of_08"
```

Expected outputs:

```text
ablation_sweep_results_merged.csv
figures/heatmap_mean_policy_action_delta_l2.png
figures/heatmap_mean_gripper_position_delta_l2.png
figures/heatmap_success_gain.png
figures/top50_by_mean_policy_action_delta_l2.csv
```

## Notes for the Next Developer

- Use GPFS for active sweeps. Copy final results to TOS/S3 after merge. Direct FUSE writes can be slow or flaky.
- MP4 rendering can fail on FUSE paths. Use `--tmp-dir /tmp` in dashboard rendering so encoding happens locally before copying.
- `success_gain` is noisy when only one or two episodes are used. Treat action-delta heatmaps as the main causal screening result unless the sweep has many episodes.
- Demo 1 uses token mean-pooling by default. Full-token probing is possible but memory-heavy.
- Demo 2 stores full-token activations as many small files to avoid giant NPZ files being killed by memory limits.
