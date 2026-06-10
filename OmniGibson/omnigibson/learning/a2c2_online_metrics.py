"""Online A2C2 rollout diagnostics for BEHAVIOR evaluation JSON files."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_DIM = 23
ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "base": (0, 1, 2),
    "torso": (3, 4, 5, 6),
    "left_arm": (7, 8, 9, 10, 11, 12, 13),
    "left_gripper": (14,),
    "right_arm": (15, 16, 17, 18, 19, 20, 21),
    "right_gripper": (22,),
}
EEF_FIELDS = {
    "left": {"pos": slice(186, 189), "quat": slice(189, 193), "gripper_qpos": slice(193, 195), "action": 14},
    "right": {"pos": slice(225, 228), "quat": slice(228, 232), "gripper_qpos": slice(232, 234), "action": 22},
}
GRIPPER_ACTION_OPEN_THRESHOLD = 0.0
GRIPPER_QPOS_OPEN_THRESHOLD = 0.025
GRIPPER_MATCH_WINDOW_STEPS = 5


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(value: Any, *, dim: int, name: str) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} must flatten to shape ({dim},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _optional_vector(value: Any, *, min_dim: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if arr.shape[0] < min_dim or not np.all(np.isfinite(arr)):
        return None
    return arr


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _quat_angle_delta(q0: np.ndarray, q1: np.ndarray) -> float | None:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 <= 0.0 or n1 <= 0.0:
        return None
    dot = abs(float(np.dot(q0 / n0, q1 / n1)))
    dot = min(max(dot, -1.0), 1.0)
    return float(2.0 * np.arccos(dot))


def _event_target(is_open: bool) -> str:
    return "open" if is_open else "close"


def _match_gripper_events(
    command_events: list[tuple[int, str]],
    realized_events: list[tuple[int, str]],
) -> dict[str, float | int | None]:
    used_realized: set[int] = set()
    delays: list[float] = []
    for command_step, target in command_events:
        match_idx = None
        for idx, (realized_step, realized_target) in enumerate(realized_events):
            if idx in used_realized or realized_target != target or realized_step < command_step:
                continue
            match_idx = idx
            delays.append(float(realized_step - command_step))
            break
        if match_idx is not None:
            used_realized.add(match_idx)

    matched = len(delays)
    within_window = sum(delay <= GRIPPER_MATCH_WINDOW_STEPS for delay in delays)
    return {
        "command_events": int(len(command_events)),
        "realized_events": int(len(realized_events)),
        "matched_events": int(matched),
        "unmatched_command_events": int(len(command_events) - matched),
        "mean_delay_steps": float(np.mean(delays)) if delays else None,
        "max_delay_steps": float(np.max(delays)) if delays else None,
        "accuracy_within_5_steps": float(within_window / matched) if matched else None,
    }


class A2C2OnlineMetricAccumulator:
    """Aggregates per-step A2C2 websocket diagnostics into compact rollout metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.clip_counts = np.zeros((ACTION_DIM,), dtype=np.int64)
        self.residual_l2: list[float] = []
        self.residual_linf: list[float] = []
        self.group_residual_abs: dict[str, list[float]] = {name: [] for name in ACTION_GROUPS}
        self.group_residual_l2: dict[str, list[float]] = {name: [] for name in ACTION_GROUPS}
        self.group_residual_rmse: dict[str, list[float]] = {name: [] for name in ACTION_GROUPS}
        self.action_velocity: list[float] = []
        self.action_acceleration: list[float] = []
        self.action_jerk: list[float] = []
        self.group_jerk: dict[str, list[float]] = {name: [] for name in ACTION_GROUPS}
        self.eef_position_delta: dict[str, list[float]] = {side: [] for side in EEF_FIELDS}
        self.eef_orientation_delta: dict[str, list[float]] = {side: [] for side in EEF_FIELDS}
        self.command_events: dict[str, list[tuple[int, str]]] = {side: [] for side in EEF_FIELDS}
        self.realized_events: dict[str, list[tuple[int, str]]] = {side: [] for side in EEF_FIELDS}
        self._prev_action: np.ndarray | None = None
        self._prev_velocity: np.ndarray | None = None
        self._prev_acceleration: np.ndarray | None = None
        self._prev_command_open: dict[str, bool | None] = {side: None for side in EEF_FIELDS}
        self._prev_realized_open: dict[str, bool | None] = {side: None for side in EEF_FIELDS}

    def step_callback(
        self,
        *,
        pre_obs: dict[str, Any],
        post_obs: dict[str, Any],
        action: Any,
        policy_info: dict[str, Any] | None,
    ) -> None:
        if not policy_info or "a2c2" not in policy_info:
            return

        info = policy_info["a2c2"]
        residual = _vector(info["residual_action"], dim=ACTION_DIM, name="a2c2.residual_action")
        corrected = _vector(action, dim=ACTION_DIM, name="action")
        payload_corrected = _vector(info["corrected_action"], dim=ACTION_DIM, name="a2c2.corrected_action")
        corrected = payload_corrected if np.allclose(corrected, payload_corrected, atol=1e-5) else corrected
        clip_mask = _to_numpy(info.get("clip_mask", np.zeros((ACTION_DIM,), dtype=np.bool_))).astype(np.bool_).reshape(-1)
        if clip_mask.shape[0] != ACTION_DIM:
            raise ValueError(f"a2c2.clip_mask must have shape ({ACTION_DIM},), got {clip_mask.shape}.")

        self.clip_counts += clip_mask.astype(np.int64)
        self.residual_l2.append(float(np.linalg.norm(residual)))
        self.residual_linf.append(float(np.max(np.abs(residual))))
        for name, indices in ACTION_GROUPS.items():
            group = residual[list(indices)]
            self.group_residual_abs[name].append(float(np.mean(np.abs(group))))
            self.group_residual_l2[name].append(float(np.linalg.norm(group)))
            self.group_residual_rmse[name].append(float(np.sqrt(np.mean(np.square(group)))))

        self._update_action_smoothness(corrected)
        self._update_gripper_timing(corrected, post_obs)
        self._update_eef_proxy(pre_obs, post_obs)
        self.steps += 1

    def _update_action_smoothness(self, action: np.ndarray) -> None:
        if self._prev_action is None:
            self._prev_action = action.copy()
            return

        velocity = action - self._prev_action
        self.action_velocity.append(float(np.linalg.norm(velocity)))
        if self._prev_velocity is not None:
            acceleration = velocity - self._prev_velocity
            self.action_acceleration.append(float(np.linalg.norm(acceleration)))
            if self._prev_acceleration is not None:
                jerk = acceleration - self._prev_acceleration
                self.action_jerk.append(float(np.linalg.norm(jerk)))
                for name, indices in ACTION_GROUPS.items():
                    self.group_jerk[name].append(float(np.linalg.norm(jerk[list(indices)])))
            self._prev_acceleration = acceleration
        self._prev_velocity = velocity
        self._prev_action = action.copy()

    def _update_gripper_timing(self, action: np.ndarray, post_obs: dict[str, Any]) -> None:
        proprio = _optional_vector(post_obs.get("robot_r1::proprio"), min_dim=236)
        for side, fields in EEF_FIELDS.items():
            command_open = bool(action[fields["action"]] > GRIPPER_ACTION_OPEN_THRESHOLD)
            prev_command = self._prev_command_open[side]
            if prev_command is not None and command_open != prev_command:
                self.command_events[side].append((self.steps, _event_target(command_open)))
            self._prev_command_open[side] = command_open

            if proprio is None:
                continue
            qpos = proprio[fields["gripper_qpos"]]
            realized_open = bool(float(np.mean(qpos)) > GRIPPER_QPOS_OPEN_THRESHOLD)
            prev_realized = self._prev_realized_open[side]
            if prev_realized is not None and realized_open != prev_realized:
                self.realized_events[side].append((self.steps, _event_target(realized_open)))
            self._prev_realized_open[side] = realized_open

    def _update_eef_proxy(self, pre_obs: dict[str, Any], post_obs: dict[str, Any]) -> None:
        pre = _optional_vector(pre_obs.get("robot_r1::proprio"), min_dim=232)
        post = _optional_vector(post_obs.get("robot_r1::proprio"), min_dim=232)
        if pre is None or post is None:
            return
        for side, fields in EEF_FIELDS.items():
            self.eef_position_delta[side].append(float(np.linalg.norm(post[fields["pos"]] - pre[fields["pos"]])))
            angle = _quat_angle_delta(pre[fields["quat"]], post[fields["quat"]])
            if angle is not None:
                self.eef_orientation_delta[side].append(angle)

    def gather_results(
        self,
        *,
        success: bool,
        q_score_final: float | None,
        steps: int,
        n_trials: int,
        n_success_trials: int,
    ) -> dict[str, Any]:
        if self.steps <= 0:
            return {}

        success_rate = float(n_success_trials / n_trials) if n_trials else None
        return {
            "a2c2_online": {
                "rollout": {
                    "success": bool(success),
                    "rollout_success_rate": success_rate,
                    "success_rate_for_json": success_rate,
                    "q_score_final": float(q_score_final) if q_score_final is not None else None,
                    "steps": int(steps),
                    "collected_policy_steps": int(self.steps),
                },
                "actions": self._action_results(),
                "gripper_timing": {
                    side: _match_gripper_events(self.command_events[side], self.realized_events[side])
                    for side in EEF_FIELDS
                },
                "end_effector_proxy": {
                    side: {
                        "position_delta": _summary(self.eef_position_delta[side]),
                        "orientation_delta_rad": _summary(self.eef_orientation_delta[side]),
                    }
                    for side in EEF_FIELDS
                },
                "smoothness": self._smoothness_results(),
            }
        }

    def _action_results(self) -> dict[str, Any]:
        denom = max(self.steps * ACTION_DIM, 1)
        return {
            "out_of_range_action_ratio": float(self.clip_counts.sum() / denom),
            "out_of_range_by_group": {
                name: float(self.clip_counts[list(indices)].sum() / max(self.steps * len(indices), 1))
                for name, indices in ACTION_GROUPS.items()
            },
            "clip_count_by_dim": [int(value) for value in self.clip_counts.tolist()],
            "residual_l2": _summary(self.residual_l2),
            "residual_linf": _summary(self.residual_linf),
            "per_group_residual_error": {
                name: {
                    "l1_mean": _summary(self.group_residual_abs[name])["mean"],
                    "l2_mean": _summary(self.group_residual_l2[name])["mean"],
                    "rmse_mean": _summary(self.group_residual_rmse[name])["mean"],
                }
                for name in ACTION_GROUPS
            },
        }

    def _smoothness_results(self) -> dict[str, Any]:
        return {
            "action_velocity_norm": _summary(self.action_velocity),
            "action_acceleration_norm": _summary(self.action_acceleration),
            "action_jerk_norm": _summary(self.action_jerk),
            "per_group_jerk": {name: _summary(values) for name, values in self.group_jerk.items()},
        }
