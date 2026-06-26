from __future__ import annotations

import inspect
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .data import StateSample
from .utils import ensure_dir


class SyntheticToyEnv:
    """Draws a target red cube and a black gripper marker with randomized backgrounds."""

    def __init__(
        self,
        image_size: int = 224,
        instruction: str = "grasp the red cube",
        min_pixel_distance: int = 30,
        background_randomization: bool = True,
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.instruction = instruction
        self.min_pixel_distance = min_pixel_distance
        self.background_randomization = background_randomization
        self.rng = np.random.default_rng(seed)

    def collect(self, num_samples: int, image_dir: Path) -> list[StateSample]:
        ensure_dir(image_dir)
        samples: list[StateSample] = []
        for idx in tqdm(range(num_samples), desc="collect synthetic states"):
            sample = self._sample_one(idx, image_dir)
            samples.append(sample)
        return samples

    def _sample_one(self, sample_id: int, image_dir: Path) -> StateSample:
        margin = 28
        while True:
            target_px = self.rng.integers(margin, self.image_size - margin, size=2)
            gripper_px = self.rng.integers(margin, self.image_size - margin, size=2)
            if np.linalg.norm(target_px - gripper_px) >= self.min_pixel_distance:
                break

        target_xy = self._pixel_to_world(target_px)
        gripper_xy = self._pixel_to_world(gripper_px)
        z = 0.02
        target_position = [float(target_xy[0]), float(target_xy[1]), z]
        gripper_position = [float(gripper_xy[0]), float(gripper_xy[1]), z]
        target_offset = [
            float(target_position[0] - gripper_position[0]),
            float(target_position[1] - gripper_position[1]),
            0.0,
        ]

        background = self._background_color()
        img = Image.new("RGB", (self.image_size, self.image_size), tuple(background))
        draw = ImageDraw.Draw(img)
        self._draw_table_noise(draw)
        self._draw_target(draw, target_px)
        self._draw_gripper(draw, gripper_px)

        image_path = image_dir / f"sample_{sample_id:06d}.png"
        img.save(image_path)

        return StateSample(
            sample_id=sample_id,
            image_path=str(image_path),
            instruction=self.instruction,
            target_position=target_position,
            gripper_position=gripper_position,
            target_offset=target_offset,
            target_pixel=[int(target_px[0]), int(target_px[1])],
            gripper_pixel=[int(gripper_px[0]), int(gripper_px[1])],
            background_color=background,
        )

    def _pixel_to_world(self, pixel_xy: np.ndarray) -> np.ndarray:
        xy01 = pixel_xy.astype(np.float32) / float(self.image_size - 1)
        return (xy01 - 0.5) * 0.5

    def _background_color(self) -> list[int]:
        if not self.background_randomization:
            return [238, 238, 232]
        base = self.rng.integers(205, 246, size=3)
        return [int(v) for v in base]

    def _draw_table_noise(self, draw: ImageDraw.ImageDraw) -> None:
        if not self.background_randomization:
            return
        for _ in range(12):
            x0 = int(self.rng.integers(0, self.image_size))
            y0 = int(self.rng.integers(0, self.image_size))
            x1 = int(np.clip(x0 + self.rng.integers(-30, 31), 0, self.image_size))
            y1 = int(np.clip(y0 + self.rng.integers(-30, 31), 0, self.image_size))
            shade = int(self.rng.integers(190, 235))
            draw.line((x0, y0, x1, y1), fill=(shade, shade, shade), width=1)

    @staticmethod
    def _draw_target(draw: ImageDraw.ImageDraw, xy: np.ndarray) -> None:
        x, y = int(xy[0]), int(xy[1])
        r = 11
        draw.rectangle((x - r, y - r, x + r, y + r), fill=(218, 30, 30), outline=(130, 0, 0), width=2)

    @staticmethod
    def _draw_gripper(draw: ImageDraw.ImageDraw, xy: np.ndarray) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(20, 20, 20))
        draw.line((x - 15, y, x - 5, y), fill=(20, 20, 20), width=3)
        draw.line((x + 5, y, x + 15, y), fill=(20, 20, 20), width=3)


class LiberoDatasetEnv:
    """Samples diverse offline states from one LIBERO demonstration HDF5."""

    def __init__(
        self,
        hdf5_path: Path | None = None,
        dataset_dir: Path | None = None,
        image_size: int = 224,
        target_joint_name: str | None = None,
        target_role: str = "pickup",
        min_frame_fraction: float = 0.0,
        max_frame_fraction: float = 1.0,
        early_frame_indices: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.target_joint_name = target_joint_name
        if target_role not in ("pickup", "place"):
            raise ValueError(f"`target_role` must be `pickup` or `place`, got `{target_role}`.")
        self.target_role = target_role
        if not 0 <= min_frame_fraction < max_frame_fraction <= 1:
            raise ValueError(
                f"Expected `0 <= min_frame_fraction < max_frame_fraction <= 1`, "
                f"got min={min_frame_fraction}, max={max_frame_fraction}."
            )
        self.min_frame_fraction = min_frame_fraction
        self.max_frame_fraction = max_frame_fraction
        if early_frame_indices is not None and (not early_frame_indices or min(early_frame_indices) < 0):
            raise ValueError(f"`early_frame_indices` must contain non-negative frame indices, got {early_frame_indices}.")
        self.early_frame_indices = early_frame_indices
        self.rng = np.random.default_rng(seed)

    def collect(self, num_samples: int, image_dir: Path) -> list[StateSample]:
        hdf5_paths = self._hdf5_paths()
        ensure_dir(image_dir)
        with ExitStack() as stack:
            datasets = []
            candidates = []
            for dataset_index, hdf5_path in enumerate(hdf5_paths):
                h5 = stack.enter_context(h5py.File(hdf5_path, "r"))
                demos = h5["data"]
                instruction = self._instruction(demos)
                first_demo = demos[sorted(demos, key=_demo_sort_key)[0]]
                target_qpos_addr, target_joint_name = self._target_qpos(str(first_demo.attrs["model_file"]), instruction)
                datasets.append((hdf5_path, demos, instruction, target_qpos_addr, target_joint_name))
                candidates.extend(
                    (dataset_index, demo_id, frame_index)
                    for demo_id in sorted(demos, key=_demo_sort_key)
                    for frame_index in self._frame_indices(len(demos[demo_id]["states"]))
                )
            if not candidates:
                raise ValueError(f"No LIBERO states found in {[str(path) for path in hdf5_paths]}.")
            indices = self._balanced_candidate_indices(candidates, len(datasets), num_samples)
            samples = []
            for sample_id, index in enumerate(tqdm(indices, desc="sample LIBERO dataset states")):
                dataset_index, demo_id, frame_index = candidates[int(index)]
                hdf5_path, demos, instruction, target_qpos_addr, target_joint_name = datasets[dataset_index]
                samples.append(
                    self._sample_one(
                    sample_id=sample_id,
                    demo_id=demo_id,
                    frame_index=frame_index,
                    demos=demos,
                    instruction=instruction,
                    target_qpos_addr=target_qpos_addr,
                    target_joint_name=target_joint_name,
                    source_hdf5_path=hdf5_path,
                    image_dir=image_dir,
                )
                )
        return samples

    def _sample_one(
        self,
        sample_id: int,
        demo_id: str,
        frame_index: int,
        demos: h5py.Group,
        instruction: str,
        target_qpos_addr: int,
        target_joint_name: str,
        source_hdf5_path: Path,
        image_dir: Path,
    ) -> StateSample:
        demo = demos[demo_id]
        obs = demo["obs"]
        image = np.asarray(obs["agentview_rgb"][frame_index], dtype=np.uint8)
        wrist_image = np.asarray(obs["eye_in_hand_rgb"][frame_index], dtype=np.uint8)
        gripper_position = np.asarray(obs["ee_pos"][frame_index], dtype=np.float32)
        observation_state = np.concatenate(
            (
                np.asarray(obs["ee_states"][frame_index], dtype=np.float32),
                np.asarray(obs["gripper_states"][frame_index], dtype=np.float32),
            )
        )
        # RoboSuite flattens simulator state as [time, qpos, qvel, ...].
        state_qpos_addr = 1 + target_qpos_addr
        target_position = np.asarray(demo["states"][frame_index][state_qpos_addr : state_qpos_addr + 3], dtype=np.float32)
        target_offset = target_position - gripper_position
        actions = demo["actions"]
        gt_action = np.asarray(actions[frame_index], dtype=np.float32)
        gt_action_chunk = np.asarray(actions[frame_index:], dtype=np.float32)

        image_path = image_dir / f"sample_{sample_id:06d}.png"
        Image.fromarray(image).resize((self.image_size, self.image_size)).save(image_path)
        wrist_image_path = image_dir / f"sample_{sample_id:06d}_wrist.png"
        Image.fromarray(wrist_image).resize((self.image_size, self.image_size)).save(wrist_image_path)
        return StateSample(
            sample_id=sample_id,
            image_path=str(image_path),
            instruction=instruction,
            target_position=target_position.astype(float).tolist(),
            gripper_position=gripper_position.astype(float).tolist(),
            target_offset=target_offset.astype(float).tolist(),
            target_pixel=[-1, -1],
            gripper_pixel=[-1, -1],
            background_color=[-1, -1, -1],
            wrist_image_path=str(wrist_image_path),
            observation_state=observation_state.astype(float).tolist(),
            source_demo_id=demo_id,
            source_frame_index=frame_index,
            source_task_name=source_hdf5_path.name.removesuffix("_demo.hdf5"),
            source_hdf5_path=str(source_hdf5_path),
            target_role=self.target_role,
            target_joint_name=target_joint_name,
            gt_action=gt_action.astype(float).tolist(),
            gt_action_chunk=gt_action_chunk.astype(float).tolist(),
        )

    @staticmethod
    def _instruction(demos: h5py.Group) -> str:
        problem_info = json.loads(str(demos.attrs.get("problem_info", "{}")))
        return str(problem_info.get("language_instruction", "")).strip()

    def _target_qpos(self, model_xml: str, instruction: str) -> tuple[int, str]:
        joints = _xml_joint_qpos_addresses(model_xml)
        if self.target_joint_name is not None:
            if self.target_joint_name not in joints:
                raise ValueError(f"Target joint `{self.target_joint_name}` not found. Available joints: {sorted(joints)}")
            return joints[self.target_joint_name], self.target_joint_name
        candidates = []
        normalized_instruction = instruction.lower().replace("_", " ")
        for joint_name, qpos_addr in joints.items():
            object_name = _normalize_libero_object_name(joint_name)
            match = _instruction_object_match(normalized_instruction, object_name)
            if match is not None:
                candidates.append((match, qpos_addr, joint_name))
        if not candidates:
            raise ValueError(
                "Could not infer the target object free joint from the LIBERO instruction. "
                "Set `env.target_joint_name` or pass `--target-joint-name`. "
                f"Instruction: `{instruction}`. Available joints: {sorted(joints)}"
            )
        selected = min(candidates) if self.target_role == "pickup" else max(candidates)
        return selected[1], selected[2]

    def _frame_indices(self, trajectory_length: int) -> range | tuple[int, ...]:
        if self.early_frame_indices is not None:
            return range(0) if trajectory_length == 0 else tuple(index for index in self.early_frame_indices if index < trajectory_length)
        start = int(np.floor(trajectory_length * self.min_frame_fraction))
        stop = int(np.ceil(trajectory_length * self.max_frame_fraction))
        return range(start, max(start + 1, stop))

    def _hdf5_paths(self) -> list[Path]:
        if self.hdf5_path is not None and self.dataset_dir is not None:
            raise ValueError("Set only one of `hdf5_path` and `dataset_dir`.")
        if self.hdf5_path is not None:
            paths = [self.hdf5_path]
        elif self.dataset_dir is not None:
            paths = sorted(self.dataset_dir.glob("*_demo.hdf5"))
        else:
            raise ValueError("Set `hdf5_path` or `dataset_dir`.")
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"LIBERO dataset HDF5 not found: {missing[0]}")
        if not paths:
            raise FileNotFoundError(f"No `*_demo.hdf5` files found in {self.dataset_dir}.")
        return paths

    def _balanced_candidate_indices(
        self,
        candidates: list[tuple[int, str, int]],
        num_datasets: int,
        num_samples: int,
    ) -> np.ndarray:
        quotas = np.full(num_datasets, num_samples // num_datasets, dtype=int)
        quotas[self.rng.permutation(num_datasets)[: num_samples % num_datasets]] += 1
        selected = []
        for dataset_index, quota in enumerate(quotas):
            candidate_indices = [index for index, candidate in enumerate(candidates) if candidate[0] == dataset_index]
            selected.extend(self.rng.choice(candidate_indices, size=int(quota), replace=quota > len(candidate_indices)))
        self.rng.shuffle(selected)
        return np.asarray(selected, dtype=int)


class LiberoEnv:
    """Thin LeRobot/LIBERO collector.

    This is intentionally conservative: LIBERO state/object internals vary across
    wrappers, so target extraction is configurable via target_body_name and falls
    back to common MuJoCo body-name heuristics.
    """

    def __init__(
        self,
        image_size: int = 224,
        instruction: str = "grasp the red cube",
        task: str = "libero_object",
        task_id: int = 0,
        target_body_name: str | None = None,
        target_joint_name: str | None = None,
        target_site_name: str | None = None,
        camera_key: str = "observation.images.image",
        wrist_camera_key: str = "pixels.image2",
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.instruction = instruction
        self.task = task
        self.task_id = task_id
        self.target_body_name = target_body_name
        self.target_joint_name = target_joint_name
        self.target_site_name = target_site_name
        self.camera_key = camera_key
        self.wrist_camera_key = wrist_camera_key
        self.seed = seed
        self._env = self._make_env()

    def _make_env(self) -> Any:
        try:
            from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
            from lerobot.envs.factory import make_env
        except Exception as exc:
            raise RuntimeError(
                "LeRobot LIBERO is not installed. On a Linux machine, install it with "
                '`python -m pip install "lerobot[libero]@git+https://github.com/huggingface/lerobot.git"` '
                "and set `MUJOCO_GL=egl` for headless rendering. Otherwise use `--env synthetic`."
            ) from exc

        cfg_kwargs = {"task": self.task, "task_ids": [self.task_id], "max_parallel_tasks": 1}
        signature = inspect.signature(LeRobotLiberoConfig)
        cfg = LeRobotLiberoConfig(**{k: v for k, v in cfg_kwargs.items() if k in signature.parameters})
        env = make_env(cfg, n_envs=1)
        return self._unwrap_lerobot_env(env)

    def _unwrap_lerobot_env(self, env: Any) -> Any:
        """LeRobot LIBERO may return {suite: {task_id: env}}; select the configured task."""
        if isinstance(env, dict):
            cur = env
            while isinstance(cur, dict):
                if not cur:
                    raise RuntimeError("LeRobot make_env returned an empty environment dictionary.")
                key = self.task if self.task in cur else self.task_id if self.task_id in cur else next(iter(cur))
                cur = cur[key]
            env = cur
        return env

    def collect(self, num_samples: int, image_dir: Path) -> list[StateSample]:
        ensure_dir(image_dir)
        samples: list[StateSample] = []
        for idx in tqdm(range(num_samples), desc="collect LIBERO states"):
            obs, info = self._reset_one(idx)
            image = self._extract_image(obs)
            wrist_image = self._extract_wrist_image(obs)
            gripper_position = self._extract_gripper_position(obs)
            observation_state = self._extract_observation_state(obs)
            instruction = self._extract_instruction(info)
            target_position = self._extract_target_position(info, instruction)
            target_offset = target_position - gripper_position

            image_path = image_dir / f"sample_{idx:06d}.png"
            Image.fromarray(image).resize((self.image_size, self.image_size)).save(image_path)
            wrist_image_path = image_dir / f"sample_{idx:06d}_wrist.png"
            Image.fromarray(wrist_image).resize((self.image_size, self.image_size)).save(wrist_image_path)

            samples.append(
                StateSample(
                    sample_id=idx,
                    image_path=str(image_path),
                    instruction=instruction,
                    target_position=target_position.astype(float).tolist(),
                    gripper_position=gripper_position.astype(float).tolist(),
                    target_offset=target_offset.astype(float).tolist(),
                    target_pixel=[-1, -1],
                    gripper_pixel=[-1, -1],
                    background_color=[-1, -1, -1],
                    wrist_image_path=str(wrist_image_path),
                    observation_state=observation_state.astype(float).tolist(),
                )
            )
        return samples

    def _reset_one(self, idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            out = self._env.reset(seed=self.seed + idx)
        except TypeError:
            out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def _extract_instruction(self, info: dict[str, Any]) -> str:
        for key in ("task_description", "language_instruction", "instruction"):
            if key in info:
                value = _first_unbatched(info[key])
                return str(value)
        if hasattr(self._env, "call"):
            for attr in ("task_description", "task"):
                try:
                    value = self._env.call(attr)
                except (AttributeError, NotImplementedError):
                    continue
                if value:
                    return str(_first_unbatched(value))
        return self.instruction

    def _extract_image(self, obs: dict[str, Any]) -> np.ndarray:
        image = None
        candidate_keys = (
            self.camera_key,
            "observation.images.image",
            "pixels.image",
            "image",
            "agentview_image",
        )
        for key in candidate_keys:
            image = _get_nested(obs, key)
            if image is not None:
                break
        if image is None:
            available = ", ".join(_flatten_keys(obs))
            raise RuntimeError(
                f"Could not find a LIBERO camera image. Tried {candidate_keys}. "
                f"Available observation keys: {available}"
            )
        image_np = np.asarray(image)
        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.shape[0] in (1, 3) and image_np.ndim == 3:
            image_np = np.moveaxis(image_np, 0, -1)
        return image_np.astype(np.uint8)

    def _extract_wrist_image(self, obs: dict[str, Any]) -> np.ndarray:
        image = None
        candidate_keys = (
            self.wrist_camera_key,
            "observation.images.wrist_image",
            "observation.images.image2",
            "pixels.image2",
            "wrist_image",
            "robot0_eye_in_hand_image",
        )
        for key in candidate_keys:
            image = _get_nested(obs, key)
            if image is not None:
                break
        if image is None:
            available = ", ".join(_flatten_keys(obs))
            raise RuntimeError(
                f"Could not find a LIBERO wrist camera image. Tried {candidate_keys}. "
                f"Available observation keys: {available}"
            )
        image_np = np.asarray(image)
        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.shape[0] in (1, 3) and image_np.ndim == 3:
            image_np = np.moveaxis(image_np, 0, -1)
        return image_np.astype(np.uint8)

    @staticmethod
    def _extract_gripper_position(obs: dict[str, Any]) -> np.ndarray:
        state = None
        candidate_keys = (
            "observation.robot_state.eef.pos",
            "robot_state.eef.pos",
            "observation.state.eef_pos",
            "observation.state",
        )
        for key in candidate_keys:
            state = _get_nested(obs, key)
            if state is not None:
                break
        if state is None:
            available = ", ".join(_flatten_keys(obs))
            raise RuntimeError(
                f"Could not extract LIBERO EEF xyz. Tried {candidate_keys}. "
                f"Available observation keys: {available}"
            )
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim > 1:
            state_np = state_np[0]
        return state_np[:3]

    @staticmethod
    def _extract_observation_state(obs: dict[str, Any]) -> np.ndarray:
        eef_pos = _require_obs_array(obs, ("robot_state.eef.pos", "observation.robot_state.eef.pos"))
        eef_quat = _require_obs_array(obs, ("robot_state.eef.quat", "observation.robot_state.eef.quat"))
        gripper_qpos = _require_obs_array(
            obs, ("robot_state.gripper.qpos", "observation.robot_state.gripper.qpos")
        )
        return np.concatenate(
            (_first_vector(eef_pos), _quat2axisangle(_first_vector(eef_quat)), _first_vector(gripper_qpos))
        )

    def _extract_target_position(self, info: dict[str, Any], instruction: str) -> np.ndarray:
        for key in ("target_position", "object_position", "target_pos"):
            if key in info:
                return np.asarray(_first_unbatched(info[key]), dtype=np.float32)[:3]

        sim = self._find_attr(self._env, "sim")
        if sim is None:
            raise RuntimeError(
                "Could not find a MuJoCo `sim` object to extract target position. "
                "Set env.target_body_name in configs/demo.yaml or adapt LiberoEnv._extract_target_position."
            )
        if self.target_joint_name is not None:
            joint_qpos = np.asarray(sim.data.get_joint_qpos(self.target_joint_name), dtype=np.float32)
            return joint_qpos[:3]
        if self.target_site_name is not None:
            site_id = sim.model.site_name2id(self.target_site_name)
            return np.asarray(sim.data.site_xpos[site_id], dtype=np.float32)

        body_name = self.target_body_name or self._guess_target_body_name(sim, instruction)
        if body_name is None:
            names = self._mujoco_names(sim)
            raise RuntimeError(
                "Could not infer the LIBERO target object. Set one of `env.target_body_name`, "
                "`env.target_joint_name`, or `env.target_site_name` in configs/demo.yaml. "
                f"MuJoCo bodies: {names['bodies']}. Joints: {names['joints']}. Sites: {names['sites']}."
            )
        body_id = sim.model.body_name2id(body_name)
        return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)

    @staticmethod
    def _find_attr(obj: Any, name: str, depth: int = 0) -> Any:
        if depth > 5:
            return None
        if hasattr(obj, name):
            return getattr(obj, name)
        for child_name in ("env", "unwrapped", "_env", "venv", "envs"):
            if not hasattr(obj, child_name):
                continue
            child = getattr(obj, child_name)
            if isinstance(child, list) and child:
                child = child[0]
            found = LiberoEnv._find_attr(child, name, depth + 1)
            if found is not None:
                return found
        return None

    def _guess_target_body_name(self, sim: Any, instruction: str) -> str | None:
        names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        names = [name for name in names if name]
        if self.target_body_name in names:
            return self.target_body_name
        lowered_instruction = instruction.lower().replace("_", " ")
        candidates: list[tuple[int, str]] = []
        for name in names:
            object_name = _normalize_libero_object_name(name)
            if object_name and object_name in lowered_instruction:
                candidates.append((lowered_instruction.index(object_name), name))
        if candidates:
            return min(candidates)[1]
        return None

    @staticmethod
    def _mujoco_names(sim: Any) -> dict[str, list[str]]:
        model = sim.model
        return {
            "bodies": _model_names(model, "body", "nbody"),
            "joints": _model_names(model, "joint", "njnt"),
            "sites": _model_names(model, "site", "nsite"),
        }


def _get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in mapping:
        return mapping[dotted_key]
    cur: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_unbatched(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim > 1:
        return arr[0]
    if arr.ndim == 1 and arr.dtype == object and len(arr) == 1:
        return arr[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _flatten_keys(mapping: Any, prefix: str = "") -> list[str]:
    if not isinstance(mapping, dict):
        return [prefix] if prefix else []
    keys: list[str] = []
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)
        keys.extend(_flatten_keys(value, full_key))
    return keys


def _model_names(model: Any, kind: str, count_attr: str) -> list[str]:
    lookup = getattr(model, f"{kind}_id2name", None)
    count = int(getattr(model, count_attr, 0))
    if lookup is None:
        return []
    return [name for idx in range(count) if (name := lookup(idx))]


def _normalize_libero_object_name(name: str) -> str:
    normalized = name.lower().replace("_", " ")
    for suffix in (" main", " joint0", " default site", " contain region"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    parts = normalized.split()
    if parts and parts[-1].isdigit():
        parts.pop()
    return " ".join(parts)


def _xml_joint_qpos_addresses(model_xml: str) -> dict[str, int]:
    qpos_addr = 0
    addresses: dict[str, int] = {}
    widths = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}
    for joint in ET.fromstring(model_xml).iter("joint"):
        joint_name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type", "hinge")
        if joint_name is not None:
            addresses[joint_name] = qpos_addr
        qpos_addr += widths.get(joint_type, 1)
    return addresses


def _instruction_object_match(instruction: str, object_name: str) -> int | None:
    tokens = object_name.split()
    aliases = [object_name]
    aliases.extend(" ".join(tokens[start:]) for start in range(1, len(tokens) - 1))
    matches = [instruction.index(alias) for alias in aliases if alias and alias in instruction]
    return min(matches) if matches else None


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _require_obs_array(obs: dict[str, Any], candidate_keys: tuple[str, ...]) -> np.ndarray:
    for key in candidate_keys:
        value = _get_nested(obs, key)
        if value is not None:
            return np.asarray(value, dtype=np.float32)
    available = ", ".join(_flatten_keys(obs))
    raise RuntimeError(f"Could not find any of {candidate_keys}. Available observation keys: {available}")


def _first_vector(value: np.ndarray) -> np.ndarray:
    return value[0] if value.ndim > 1 else value


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32)
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    if quat[3] < 0:
        quat = -quat
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (xyz / sin_half * (2.0 * np.arctan2(sin_half, w))).astype(np.float32)
