from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Humanoid Robot Simulation Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------

LIVE_FPS = 8.0
DEFAULT_LIVE_CHUNK_S = 0.75

PREVIEW_DURATION_S = 2.0
PREVIEW_FPS = 10
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180

USE_LIVE_FRAGMENT = False
if hasattr(st, "fragment"):
    try:
        USE_LIVE_FRAGMENT = "run_every" in inspect.signature(st.fragment).parameters
    except Exception:
        USE_LIVE_FRAGMENT = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    seed: int = 7
    dt: float = 0.002
    render_fps: int = 30
    duration_s: float = 14.0
    width: int = 640
    height: int = 360

    kp: float = 180.0
    kd: float = 22.0
    torque_limit: float = 90.0
    friction: float = 0.05
    delay_ms: float = 4.0

    contact_stiffness: float = 150000.0
    contact_damping: float = 2200.0
    margin: float = 0.0005
    mass_total: float = 35.0

    gyro_std: float = 0.012
    accel_std: float = 0.25
    encoder_std: float = 0.0005
    ft_std: float = 1.5
    camera_noise: float = 0.015

    push_time: float = 7.0
    push_duration: float = 0.10
    push_impulse: float = 18.0
    scripted_start: float = 10.0
    drop_height: float = 0.04

    root_kp: float = 180.0
    root_kd: float = 28.0

    max_penetration_m: float = 0.005
    energy_drift_max_pct: float = 1.0
    rtf_min: float = 1.0
    determinism_tol: float = 1e-9


# ---------------------------------------------------------------------------
# Lightweight humanoid model
# ---------------------------------------------------------------------------


DOF_NAMES: Tuple[str, ...] = (
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "l_shoulder_pitch",
    "l_elbow_pitch",
    "r_shoulder_pitch",
    "r_elbow_pitch",
    "torso_yaw",
    "head_pitch",
)
N_DOF = len(DOF_NAMES)

JOINT_LOWER = np.array(
    [
        -0.60, 0.00, -0.50,
        -0.60, 0.00, -0.50,
        -1.20, -1.60,
        -1.20, -1.60,
        -0.40, -0.30,
    ],
    dtype=float,
)

JOINT_UPPER = np.array(
    [
         1.00, 1.50,  0.50,
         1.00, 1.50,  0.50,
         1.20,  0.00,
         1.20,  0.00,
         0.40,  0.30,
    ],
    dtype=float,
)

JOINT_INERTIA = np.array(
    [
        0.25, 0.18, 0.08,
        0.25, 0.18, 0.08,
        0.08, 0.04,
        0.08, 0.04,
        0.50, 0.05,
    ],
    dtype=float,
)


@dataclass
class Node:
    name: str
    parent: Optional[int]
    offset: Tuple[float, float, float]
    axis: Optional[str]
    dof: Optional[int]


NODES: List[Node] = [
    Node("pelvis", None, (0.0, 0.0, 0.0), None, None),
    Node("torso", 0, (0.0, 0.0, 0.12), "z", 10),
    Node("neck", 1, (0.0, 0.0, 0.20), None, None),
    Node("head", 2, (0.0, 0.0, 0.12), "x", 11),

    Node("l_shoulder", 1, (0.0, 0.18, 0.16), "y", 6),
    Node("l_elbow", 4, (0.0, 0.0, -0.24), "y", 7),
    Node("l_hand", 5, (0.0, 0.0, -0.22), None, None),

    Node("r_shoulder", 1, (0.0, -0.18, 0.16), "y", 8),
    Node("r_elbow", 7, (0.0, 0.0, -0.24), "y", 9),
    Node("r_hand", 8, (0.0, 0.0, -0.22), None, None),

    Node("l_hip", 0, (0.0, 0.09, -0.06), "y", 0),
    Node("l_knee", 10, (0.0, 0.0, -0.38), "y", 1),
    Node("l_ankle", 11, (0.0, 0.0, -0.38), "y", 2),
    Node("l_foot", 12, (0.09, 0.0, -0.04), None, None),

    Node("r_hip", 0, (0.0, -0.09, -0.06), "y", 3),
    Node("r_knee", 14, (0.0, 0.0, -0.38), "y", 4),
    Node("r_ankle", 15, (0.0, 0.0, -0.38), "y", 5),
    Node("r_foot", 16, (0.09, 0.0, -0.04), None, None),
]

LINKS: List[Tuple[int, int]] = [
    (node.parent, i) for i, node in enumerate(NODES) if node.parent is not None
]

FOOT_IDS: Tuple[int, int] = (13, 17)
FOOT_NAMES: Tuple[str, str] = ("left_foot", "right_foot")

SEGMENT_MASSES = np.array(
    [
        8.0, 7.0, 1.0, 2.0,
        1.0, 0.8, 0.5,
        1.0, 0.8, 0.5,
        1.5, 1.2, 0.8, 0.7,
        1.5, 1.2, 0.8, 0.7,
    ],
    dtype=float,
)

HOME_PELVIS_Z = 0.86


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _axis_rotation(axis: Optional[str], angle: float) -> np.ndarray:
    if axis == "x":
        return _rot_x(angle)
    if axis == "y":
        return _rot_y(angle)
    if axis == "z":
        return _rot_z(angle)
    return np.eye(3, dtype=float)


def euler_to_rot(rpy: np.ndarray) -> np.ndarray:
    return _rot_z(float(rpy[2])) @ _rot_y(float(rpy[1])) @ _rot_x(float(rpy[0]))


def forward_kinematics(
    q: np.ndarray,
    root_pos: np.ndarray,
    rpy: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    root_rot = euler_to_rot(rpy)
    transforms: List[Tuple[np.ndarray, np.ndarray]] = []
    positions = np.zeros((len(NODES), 3), dtype=float)
    rotations: List[np.ndarray] = []

    for i, node in enumerate(NODES):
        if node.parent is None:
            rot = root_rot
            pos = np.asarray(root_pos, dtype=float)
        else:
            parent_rot, parent_pos = transforms[node.parent]
            offset = np.asarray(node.offset, dtype=float)
            pos = parent_pos + parent_rot @ offset

            if node.axis is not None and node.dof is not None:
                rot = parent_rot @ _axis_rotation(node.axis, float(q[node.dof]))
            else:
                rot = parent_rot

        transforms.append((rot, pos))
        positions[i] = pos
        rotations.append(rot)

    return positions, rotations


def project_points(
    pts: np.ndarray,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=float)
    forward = cam_target - cam_pos
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = forward / forward_norm

    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up_world)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right = right / right_norm

    up = np.cross(right, forward)

    rel = pts - cam_pos
    x = rel @ right
    y = rel @ up
    z = rel @ forward

    safe_z = np.where(z > 1e-4, z, 1e-4)
    focal = 0.95 * float(height)

    u = 0.5 * width + focal * x / safe_z
    v = 0.5 * height - focal * y / safe_z
    valid = z > 0.02

    return np.column_stack((u, v)), valid, z


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_humanoid(
    positions: np.ndarray,
    t: float,
    cfg: SimConfig,
    camera: str,
    rng: np.random.Generator,
    metric: Dict[str, Any],
    label: str,
) -> np.ndarray:
    width, height = int(cfg.width), int(cfg.height)
    img = np.full((height, width, 3), (28, 28, 34), dtype=np.uint8)

    presets = {
        "front": (np.array([2.45, 0.0, 1.05]), np.array([0.0, 0.0, 0.74])),
        "side": (np.array([0.0, 2.45, 1.05]), np.array([0.0, 0.0, 0.74])),
        "iso": (np.array([1.85, 1.85, 1.38]), np.array([0.0, 0.0, 0.72])),
    }
    cam_pos, cam_target = presets.get(camera, presets["iso"])

    grid_step = 0.5 if width < 420 else 0.25
    grid_values = np.arange(-1.5, 1.51, grid_step)

    for gv in grid_values:
        pts_x = np.array([[gv, -1.5, 0.0], [gv, 1.5, 0.0]], dtype=float)
        uv, valid, _ = project_points(pts_x, cam_pos, cam_target, width, height)
        if valid.all():
            cv2.line(
                img,
                tuple(uv[0].astype(int)),
                tuple(uv[1].astype(int)),
                (65, 65, 78),
                1,
                cv2.LINE_AA,
            )

        pts_y = np.array([[-1.5, gv, 0.0], [1.5, gv, 0.0]], dtype=float)
        uv, valid, _ = project_points(pts_y, cam_pos, cam_target, width, height)
        if valid.all():
            cv2.line(
                img,
                tuple(uv[0].astype(int)),
                tuple(uv[1].astype(int)),
                (65, 65, 78),
                1,
                cv2.LINE_AA,
            )

    for fid in FOOT_IDS:
        shadow = positions[fid].copy()
        shadow[2] = 0.002
        uv, valid, _ = project_points(shadow[None, :], cam_pos, cam_target, width, height)
        if valid[0]:
            cv2.circle(
                img,
                tuple(uv[0].astype(int)),
                max(4, int(width * 0.016)),
                (18, 18, 18),
                -1,
                cv2.LINE_AA,
            )

    projected, valid, depths = project_points(positions, cam_pos, cam_target, width, height)

    for a, b in LINKS:
        if valid[a] and valid[b]:
            depth = 0.5 * (float(depths[a]) + float(depths[b]))
            thickness = int(np.clip(width * 0.018 - depth * 1.0, 2, 10))
            cv2.line(
                img,
                tuple(projected[a].astype(int)),
                tuple(projected[b].astype(int)),
                (90, 200, 255),
                thickness,
                cv2.LINE_AA,
            )

    for i in range(len(positions)):
        if valid[i]:
            depth = float(depths[i])
            base_radius = int(np.clip(width * 0.023 - depth * 0.9, 3, 12))
            if i in FOOT_IDS:
                radius = base_radius + 2
                color = (0, 255, 160)
            else:
                radius = base_radius
                color = (240, 240, 240)

            cv2.circle(
                img,
                tuple(projected[i].astype(int)),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )

    top_h = max(24, int(height * 0.11))
    cv2.rectangle(img, (0, 0), (width, top_h), (0, 0, 0), -1)

    font_scale = max(0.30, float(width) / 1100.0)
    thickness = max(1, int(round(font_scale * 2.0)))

    if width < 420:
        top_text = f"t={t:4.2f}s {label}"
    else:
        top_text = f"t={t:5.2f}s  {label}"

    cv2.putText(
        img,
        top_text,
        (8, max(16, int(top_h * 0.68))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )

    rtf = float(metric.get("rtf", 0.0))
    pen_mm = float(metric.get("penetration", 0.0)) * 1000.0
    energy = float(metric.get("energy", 0.0))
    contacts = int(metric.get("contact_count", 0))

    if width < 420:
        bottom = f"RTF {rtf:.1f} pen {pen_mm:.1f}mm E {energy:.0f}J C{contacts}"
    else:
        bottom = f"RTF {rtf:4.2f} | pen {pen_mm:5.2f} mm | E {energy:6.1f} J | C {contacts}"

    cv2.putText(
        img,
        bottom,
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.26, float(width) / 1500.0),
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    exposure = 1.0 + 0.035 * math.sin(t * 2.2)
    img_float = img.astype(np.float32) * exposure

    if cfg.camera_noise > 0.0:
        img_float += rng.normal(0.0, cfg.camera_noise * 255.0, size=img_float.shape)

    img = np.clip(img_float, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class HumanoidSim:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.q = np.zeros(N_DOF, dtype=float)
        self.qdot = np.zeros(N_DOF, dtype=float)

        self.masses = SEGMENT_MASSES * (cfg.mass_total / float(SEGMENT_MASSES.sum()))

        self.root_pos = np.array([0.0, 0.0, HOME_PELVIS_Z + cfg.drop_height], dtype=float)
        self.root_vel = np.zeros(3, dtype=float)
        self.rpy = np.zeros(3, dtype=float)
        self.angvel = np.zeros(3, dtype=float)

        delay_steps = max(1, int(round(cfg.delay_ms * 1e-3 / cfg.dt)))
        self.delay_buffer: deque[np.ndarray] = deque(
            [self.q.copy() for _ in range(delay_steps)],
            maxlen=delay_steps,
        )

        self.positions, self.rotations = forward_kinematics(self.q, self.root_pos, self.rpy)
        self.prev_positions = self.positions.copy()

        self.last_contact_forces: Dict[str, np.ndarray] = {
            name: np.zeros(3, dtype=float) for name in FOOT_NAMES
        }
        self.last_acc = np.zeros(3, dtype=float)
        self.last_sensors: Dict[str, float] = {}
        self.last_image: Optional[np.ndarray] = None

        self.last_metric: Dict[str, Any] = {
            "time": 0.0,
            "penetration": 0.0,
            "energy": float(np.sum(self.masses * 9.81 * self.positions[:, 2])),
            "contact_count": 0,
            "tau_norm": 0.0,
            "tracking_error": 0.0,
            "root_x": float(self.root_pos[0]),
            "root_y": float(self.root_pos[1]),
            "root_z": float(self.root_pos[2]),
            "root_vel_x": 0.0,
            "root_vel_norm": 0.0,
            "rtf": 0.0,
        }

        self.history: List[Dict[str, Any]] = []
        self.sensor_history: List[Dict[str, Any]] = []

        self.t = 0.0
        self.frame = 0
        self.sim_active = 0.0
        self.wall_active = 0.0
        self.run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{cfg.seed}"

    def sync_config(self) -> None:
        expected = max(1, int(round(self.cfg.delay_ms * 1e-3 / self.cfg.dt)))
        if self.delay_buffer.maxlen != expected:
            self.delay_buffer = deque(
                [self.q.copy() for _ in range(expected)],
                maxlen=expected,
            )

    def state_label(self) -> str:
        if self.t < self.cfg.push_time - 0.2:
            return "DROP & SETTLE"
        if self.t < self.cfg.push_time + self.cfg.push_duration:
            return "EXTERNAL PUSH"
        if self.t < self.cfg.scripted_start:
            return "PD STAND"
        return "SCRIPTED MOTION"

    def _targets(self) -> np.ndarray:
        target = np.zeros(N_DOF, dtype=float)
        t = self.t

        if t >= self.cfg.scripted_start:
            phase = t - self.cfg.scripted_start
            two_pi = 2.0 * math.pi

            target[6] = 0.55 * math.sin(two_pi * 0.45 * phase)
            target[7] = -0.65 * (0.5 + 0.5 * math.sin(two_pi * 0.45 * phase + 0.4))

            target[8] = -0.35 * math.sin(two_pi * 0.45 * phase)
            target[9] = -0.55 * (0.5 + 0.5 * math.cos(two_pi * 0.45 * phase))

            leg_phase = max(0.0, math.sin(two_pi * 0.25 * phase))
            target[1] = 0.25 * leg_phase
            target[2] = -0.08 * leg_phase

            target[3] = -0.08 * leg_phase
            target[4] = 0.08 * leg_phase
            target[5] = 0.04 * leg_phase

            target[10] = 0.06 * math.sin(two_pi * 0.20 * phase)
            target[11] = 0.04 * math.sin(two_pi * 0.35 * phase)

        return np.clip(target, JOINT_LOWER, JOINT_UPPER)

    def _recover_if_nan(self) -> None:
        finite = (
            np.all(np.isfinite(self.q))
            and np.all(np.isfinite(self.qdot))
            and np.all(np.isfinite(self.root_pos))
            and np.all(np.isfinite(self.root_vel))
            and np.all(np.isfinite(self.rpy))
            and np.all(np.isfinite(self.angvel))
        )
        if finite:
            return

        self.q = np.clip(self.q, JOINT_LOWER, JOINT_UPPER)
        self.q[~np.isfinite(self.q)] = 0.0
        self.qdot[~np.isfinite(self.qdot)] = 0.0

        self.root_pos[~np.isfinite(self.root_pos)] = 0.0
        self.root_pos[2] = HOME_PELVIS_Z + self.cfg.drop_height

        self.root_vel[~np.isfinite(self.root_vel)] = 0.0
        self.rpy[~np.isfinite(self.rpy)] = 0.0
        self.angvel[~np.isfinite(self.angvel)] = 0.0

        self.positions, self.rotations = forward_kinematics(self.q, self.root_pos, self.rpy)
        self.prev_positions = self.positions.copy()

        kin = (
            0.5 * self.cfg.mass_total * float(np.dot(self.root_vel, self.root_vel))
            + 0.5 * float(np.sum(JOINT_INERTIA * self.qdot**2))
        )
        pot = float(np.sum(self.masses * 9.81 * self.positions[:, 2]))

        self.last_metric = {
            "time": float(self.t),
            "penetration": 0.0,
            "energy": float(kin + pot),
            "contact_count": 0,
            "tau_norm": 0.0,
            "tracking_error": 0.0,
            "root_x": float(self.root_pos[0]),
            "root_y": float(self.root_pos[1]),
            "root_z": float(self.root_pos[2]),
            "root_vel_x": float(self.root_vel[0]),
            "root_vel_norm": float(np.linalg.norm(self.root_vel)),
            "rtf": float(self.last_metric.get("rtf", 0.0)),
        }

    def _physics_step(self) -> Dict[str, Any]:
        cfg = self.cfg
        dt = cfg.dt

        target = self._targets()
        delayed = self.delay_buffer[0].copy()
        self.delay_buffer.append(target)

        tau = JOINT_INERTIA * (cfg.kp * (delayed - self.q) - cfg.kd * self.qdot)
        tau -= cfg.friction * np.tanh(self.qdot / 0.05)
        tau = np.clip(tau, -cfg.torque_limit, cfg.torque_limit)

        qddot = tau / JOINT_INERTIA
        self.qdot += qddot * dt
        self.q += self.qdot * dt

        self.q = np.clip(self.q, JOINT_LOWER, JOINT_UPPER)
        at_lower = self.q <= JOINT_LOWER + 1e-9
        at_upper = self.q >= JOINT_UPPER - 1e-9
        self.qdot[at_lower & (self.qdot < 0.0)] = 0.0
        self.qdot[at_upper & (self.qdot > 0.0)] = 0.0

        positions, rotations = forward_kinematics(self.q, self.root_pos, self.rpy)
        foot_indices = list(FOOT_IDS)
        foot_vel = (positions[foot_indices] - self.prev_positions[foot_indices]) / dt

        total_f = np.array([0.0, 0.0, -cfg.mass_total * 9.81], dtype=float)
        penetration = 0.0
        contact_count = 0
        contact_forces: Dict[str, np.ndarray] = {}

        for idx, (fid, name) in enumerate(zip(FOOT_IDS, FOOT_NAMES)):
            p = positions[fid]
            v = foot_vel[idx]
            pen = max(0.0, -float(p[2]))

            if pen > 0.0 or float(p[2]) < cfg.margin:
                fz = cfg.contact_stiffness * pen
                if float(v[2]) < 0.0:
                    fz += cfg.contact_damping * (-float(v[2]))
                fz = float(np.clip(fz, 0.0, 12000.0))

                mu = 0.90
                fx = -mu * fz * math.tanh(float(v[0]) / 0.08) if abs(float(v[0])) > 1e-6 else 0.0
                fy = -mu * fz * math.tanh(float(v[1]) / 0.08) if abs(float(v[1])) > 1e-6 else 0.0

                ft_norm = math.hypot(fx, fy)
                max_ft = mu * fz
                if ft_norm > max_ft and ft_norm > 1e-9:
                    scale = max_ft / ft_norm
                    fx *= scale
                    fy *= scale

                f = np.array([fx, fy, fz], dtype=float)
                total_f += f
                contact_forces[name] = f
                penetration = max(penetration, pen)
                contact_count += 1
            else:
                contact_forces[name] = np.zeros(3, dtype=float)

        push_dur = max(1e-3, float(cfg.push_duration))
        if cfg.push_time <= self.t < cfg.push_time + push_dur:
            total_f[0] += cfg.push_impulse / push_dur

        acc = total_f / cfg.mass_total
        acc -= 0.20 * self.root_vel

        self.last_acc = acc.copy()
        self.root_vel += acc * dt
        self.root_pos += self.root_vel * dt

        ang_acc = cfg.root_kp * (-self.rpy) - cfg.root_kd * self.angvel
        self.angvel += ang_acc * dt
        self.rpy += self.angvel * dt
        self.rpy = np.clip(self.rpy, [-0.35, -0.35, -0.8], [0.35, 0.35, 0.8])

        kin = (
            0.5 * cfg.mass_total * float(np.dot(self.root_vel, self.root_vel))
            + 0.5 * float(np.sum(JOINT_INERTIA * self.qdot**2))
        )
        pot = float(np.sum(self.masses * 9.81 * positions[:, 2]))
        energy = kin + pot

        metric = {
            "penetration": float(penetration),
            "energy": float(energy),
            "contact_count": int(contact_count),
            "tau_norm": float(np.linalg.norm(tau)),
            "tracking_error": float(np.linalg.norm(delayed - self.q)),
            "root_x": float(self.root_pos[0]),
            "root_y": float(self.root_pos[1]),
            "root_z": float(self.root_pos[2]),
            "root_vel_x": float(self.root_vel[0]),
            "root_vel_norm": float(np.linalg.norm(self.root_vel)),
        }

        self.last_metric = metric
        self.last_contact_forces = contact_forces
        self.positions = positions
        self.rotations = rotations
        self.prev_positions = positions.copy()
        self.t += dt

        return metric

    def _sample_sensors(self) -> Dict[str, float]:
        cfg = self.cfg
        r_root = self.rotations[0]

        specific_world = self.last_acc + np.array([0.0, 0.0, 9.81], dtype=float)
        accel_body = r_root.T @ specific_world
        gyro_body = r_root.T @ self.angvel

        out: Dict[str, float] = {}

        out["imu_gx"] = float(gyro_body[0] + self.rng.normal(0.0, cfg.gyro_std))
        out["imu_gy"] = float(gyro_body[1] + self.rng.normal(0.0, cfg.gyro_std))
        out["imu_gz"] = float(gyro_body[2] + self.rng.normal(0.0, cfg.gyro_std))

        out["imu_ax"] = float(accel_body[0] + self.rng.normal(0.0, cfg.accel_std))
        out["imu_ay"] = float(accel_body[1] + self.rng.normal(0.0, cfg.accel_std))
        out["imu_az"] = float(accel_body[2] + self.rng.normal(0.0, cfg.accel_std))

        for name, value in zip(DOF_NAMES, self.q):
            out[f"encoder_{name}"] = float(value + self.rng.normal(0.0, cfg.encoder_std))

        for name in FOOT_NAMES:
            f = self.last_contact_forces.get(name, np.zeros(3, dtype=float))
            f_noisy = f + self.rng.normal(0.0, cfg.ft_std, size=3)
            out[f"{name}_fx"] = float(f_noisy[0])
            out[f"{name}_fy"] = float(f_noisy[1])
            out[f"{name}_fz"] = float(f_noisy[2])

        out["gt_root_z"] = float(self.root_pos[2])
        out["gt_contact_count"] = float(
            sum(1 for f in self.last_contact_forces.values() if float(np.linalg.norm(f)) > 1e-6)
        )

        return out

    def _render(self, camera: str) -> np.ndarray:
        return render_humanoid(
            self.positions,
            self.t,
            self.cfg,
            camera,
            self.rng,
            self.last_metric,
            self.state_label(),
        )

    def step_render_frame(
        self,
        speed: float = 1.0,
        camera: str = "iso",
        target_fps: int = 30,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
        self.sync_config()
        fps = max(1, int(target_fps))

        if speed <= 0.0:
            image = self._render(camera)
            self.last_image = image
            sensors = self._sample_sensors()
            self.last_sensors = sensors
            metric = dict(self.last_metric)
            metric["time"] = float(self.t)
            metric["rtf"] = float(self.sim_active / max(self.wall_active, 1e-9))
            return image, metric, sensors

        frame_sim_time = float(speed) / float(fps)
        steps = max(1, int(math.ceil(frame_sim_time / self.cfg.dt)))

        wall0 = time.perf_counter()
        for _ in range(steps):
            self._physics_step()
        wall1 = time.perf_counter()

        self.wall_active += wall1 - wall0
        self.sim_active += steps * self.cfg.dt

        self._recover_if_nan()

        metric = dict(self.last_metric)
        metric["time"] = float(self.t)
        metric["rtf"] = float(self.sim_active / max(self.wall_active, 1e-9))

        sensors = self._sample_sensors()
        self.last_sensors = sensors

        image = self._render(camera)
        self.last_image = image

        self.history.append(metric)
        self.sensor_history.append({"time": float(self.t), **sensors})

        if len(self.history) > 2400:
            self.history = self.history[-1800:]
        if len(self.sensor_history) > 2400:
            self.sensor_history = self.sensor_history[-1800:]

        self.frame += 1
        return image, metric, sensors


# ---------------------------------------------------------------------------
# Analysis, determinism, and video generation
# ---------------------------------------------------------------------------


def compute_summary(
    cfg: SimConfig,
    hist: pd.DataFrame,
    sens: pd.DataFrame,
    sim: HumanoidSim,
    display_path: str,
    download_path: str,
    codec: str,
    total_frames: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "run_id": sim.run_id,
        "display_path": display_path,
        "download_path": download_path,
        "codec": codec,
        "frames": total_frames,
        "duration_s": cfg.duration_s,
        "rtf_measured": float(sim.sim_active / max(sim.wall_active, 1e-9)),
        "max_penetration_mm": float("nan"),
        "energy_drift_pct_settled_stand": float("nan"),
        "standing_root_xy_std_m": float("nan"),
        "mean_contact_count": float("nan"),
        "final_tracking_error_rad": float("nan"),
        "scripted_mean_tracking_error_rad": float("nan"),
        "push_peak_root_vel_m_s": float("nan"),
        "push_recovery_time_s": float("nan"),
        "imu_gyro_noise_std_meas_rad_s": float("nan"),
        "imu_accel_noise_std_meas_m_s2": float("nan"),
        "encoder_noise_std_meas_rad": float("nan"),
        "configured_gyro_std": cfg.gyro_std,
        "configured_accel_std": cfg.accel_std,
        "configured_encoder_std": cfg.encoder_std,
        "configured_ft_std": cfg.ft_std,
        "configured_camera_noise": cfg.camera_noise,
    }

    if hist.empty:
        return summary

    summary["max_penetration_mm"] = float(hist["penetration"].max() * 1000.0)
    summary["mean_contact_count"] = float(hist["contact_count"].mean())
    summary["final_tracking_error_rad"] = float(hist["tracking_error"].iloc[-1])

    stand_end = min(float(cfg.push_time), float(cfg.scripted_start))
    settled_start = max(0.0, stand_end - 1.0)
    stand = hist[(hist["time"] >= settled_start) & (hist["time"] <= stand_end)]

    if len(stand) < 2:
        stand = hist[hist["time"] <= stand_end]

    if len(stand) > 1:
        e0 = float(stand["energy"].iloc[0])
        e1 = float(stand["energy"].iloc[-1])
        summary["energy_drift_pct_settled_stand"] = float(
            100.0 * (e1 - e0) / max(abs(e0), 1e-9)
        )

        if "root_x" in stand and "root_y" in stand:
            xy_std = float(
                np.sqrt(
                    np.var(stand["root_x"].to_numpy(dtype=float))
                    + np.var(stand["root_y"].to_numpy(dtype=float))
                )
            )
            summary["standing_root_xy_std_m"] = xy_std

    script = hist[hist["time"] >= cfg.scripted_start]
    if not script.empty:
        summary["scripted_mean_tracking_error_rad"] = float(script["tracking_error"].mean())

    push = hist[(hist["time"] >= cfg.push_time) & (hist["time"] <= cfg.push_time + 2.0)]
    if not push.empty:
        summary["push_peak_root_vel_m_s"] = float(push["root_vel_norm"].max())

    after = hist[hist["time"] > cfg.push_time + cfg.push_duration]
    recovered = after[after["root_vel_norm"] < 0.08]
    if not recovered.empty:
        summary["push_recovery_time_s"] = float(
            recovered["time"].iloc[0] - (cfg.push_time + cfg.push_duration)
        )

    if not sens.empty:
        stand_sens = sens[(sens["time"] >= settled_start) & (sens["time"] <= stand_end)]
        if len(stand_sens) < 10:
            stand_sens = sens[sens["time"] <= stand_end]

        if len(stand_sens) > 10:
            gyro_cols = [c for c in ("imu_gx", "imu_gy", "imu_gz") if c in stand_sens]
            accel_cols = [c for c in ("imu_ax", "imu_ay", "imu_az") if c in stand_sens]
            enc_cols = [f"encoder_{n}" for n in DOF_NAMES if f"encoder_{n}" in stand_sens]

            if gyro_cols:
                summary["imu_gyro_noise_std_meas_rad_s"] = float(
                    np.mean([stand_sens[c].std() for c in gyro_cols])
                )
            if accel_cols:
                summary["imu_accel_noise_std_meas_m_s2"] = float(
                    np.mean([stand_sens[c].std() for c in accel_cols])
                )
            if enc_cols:
                summary["encoder_noise_std_meas_rad"] = float(
                    np.mean([stand_sens[c].std() for c in enc_cols])
                )

    return summary


def _final_state_vector(sim: HumanoidSim) -> np.ndarray:
    mech = np.concatenate(
        [
            sim.q.astype(float),
            sim.qdot.astype(float),
            sim.root_pos.astype(float),
            sim.root_vel.astype(float),
            sim.rpy.astype(float),
            sim.angvel.astype(float),
        ]
    )

    sensor_keys = sorted(sim.last_sensors.keys())
    if sensor_keys:
        sensors = np.array([sim.last_sensors[k] for k in sensor_keys], dtype=float)
    else:
        sensors = np.zeros(0, dtype=float)

    if sim.last_image is not None:
        image_scalar = np.array([float(np.mean(sim.last_image))], dtype=float)
    else:
        image_scalar = np.zeros(1, dtype=float)

    return np.concatenate([mech, sensors, image_scalar])


def _run_short(config_json: str, seed_override: int, duration: float, fps: int) -> np.ndarray:
    config_dict = json.loads(config_json)
    config_dict["seed"] = int(seed_override)
    config_dict["duration_s"] = float(duration)

    cfg = SimConfig(**config_dict)
    sim = HumanoidSim(cfg)
    frames = max(1, int(float(duration) * float(max(1, fps))))

    for _ in range(frames):
        sim.step_render_frame(speed=1.0, camera="iso", target_fps=max(1, fps))

    return _final_state_vector(sim)


@st.cache_data(show_spinner=False)
def run_determinism_check(config_json: str) -> Dict[str, Any]:
    config_dict = json.loads(config_json)
    cfg = SimConfig(**config_dict)

    seed = int(cfg.seed)
    tol = float(cfg.determinism_tol)
    duration = 0.4
    fps = 8

    a = _run_short(config_json, seed, duration, fps)
    b = _run_short(config_json, seed, duration, fps)
    c = _run_short(config_json, seed + 1, duration, fps)

    max_same = float(np.max(np.abs(a - b))) if a.size else 0.0
    max_diff = float(np.max(np.abs(a - c))) if a.size else 0.0

    noise_present = any(
        float(x) > 0.0
        for x in (
            cfg.gyro_std,
            cfg.accel_std,
            cfg.encoder_std,
            cfg.ft_std,
            cfg.camera_noise,
        )
    )

    return {
        "determinism_same_seed_pass": bool(max_same <= tol),
        "determinism_different_seed_pass": bool(max_diff > tol or not noise_present),
        "determinism_max_abs_same": max_same,
        "determinism_max_abs_different_seed": max_diff,
        "determinism_noise_present": bool(noise_present),
    }


@st.cache_data(show_spinner="Generating humanoid demo video...")
def generate_demo_video(config_json: str) -> Tuple[str, str, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = SimConfig(**json.loads(config_json))
    sim = HumanoidSim(cfg)

    tmp_dir = tempfile.mkdtemp(prefix="humanoid_demo_")
    mp4_path = os.path.join(tmp_dir, "humanoid_demo.mp4")
    gif_path = os.path.join(tmp_dir, "humanoid_demo.gif")

    fps = max(1, int(cfg.render_fps))
    writer = None
    writer_ok = False

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            mp4_path,
            fourcc,
            fps,
            (int(cfg.width), int(cfg.height)),
        )
        writer_ok = bool(writer is not None and writer.isOpened())
        if not writer_ok:
            writer = None
    except Exception:
        writer = None
        writer_ok = False

    total_frames = max(1, int(round(float(cfg.duration_s) * float(fps))))
    gif_stride = max(1, math.ceil(total_frames / 90))
    gif_frames: List[Image.Image] = []

    for i in range(total_frames):
        frame, _, _ = sim.step_render_frame(speed=1.0, camera="iso", target_fps=fps)

        if writer_ok and writer is not None:
            try:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception:
                writer_ok = False
                writer = None

        if i % gif_stride == 0 and len(gif_frames) < 90:
            gif_frames.append(Image.fromarray(frame))

    if writer is not None and writer_ok:
        try:
            writer.release()
        except Exception:
            pass

    gif_saved = False
    try:
        if gif_frames:
            effective_fps = max(1.0, float(fps) / float(gif_stride))
            duration_ms = max(20, int(1000.0 / effective_fps))
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0,
            )
            gif_saved = os.path.exists(gif_path) and os.path.getsize(gif_path) > 0
    except Exception:
        gif_saved = False

    mp4_ok = bool(writer_ok and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0)

    if gif_saved:
        display_path = gif_path
    elif mp4_ok:
        display_path = mp4_path
    else:
        display_path = ""

    if mp4_ok:
        download_path = mp4_path
    elif gif_saved:
        download_path = gif_path
    else:
        download_path = ""

    if mp4_ok and gif_saved:
        codec = "MP4 + GIF"
    elif mp4_ok:
        codec = "MP4"
    elif gif_saved:
        codec = "GIF"
    else:
        codec = "none"

    hist = pd.DataFrame(sim.history)
    sens = pd.DataFrame(sim.sensor_history)
    summary = compute_summary(cfg, hist, sens, sim, display_path, download_path, codec, total_frames)
    summary.update(run_determinism_check(config_json))

    summary["penetration_pass"] = bool(
        np.isfinite(summary["max_penetration_mm"])
        and float(summary["max_penetration_mm"]) / 1000.0 <= float(cfg.max_penetration_m)
    )
    summary["energy_drift_pass"] = bool(
        np.isfinite(summary["energy_drift_pct_settled_stand"])
        and abs(float(summary["energy_drift_pct_settled_stand"])) <= float(cfg.energy_drift_max_pct)
    )
    summary["rtf_pass"] = bool(
        np.isfinite(summary["rtf_measured"])
        and float(summary["rtf_measured"]) >= float(cfg.rtf_min)
    )
    summary["determinism_pass"] = bool(
        summary.get("determinism_same_seed_pass", False)
        and summary.get("determinism_different_seed_pass", False)
    )

    return display_path, download_path, hist, sens, summary


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------


def make_error_result(msg: str) -> Tuple[str, str, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    return "", "", pd.DataFrame(), pd.DataFrame(), {"error": msg}


def reset_simulation() -> None:
    st.session_state.sim = HumanoidSim(st.session_state.cfg)
    st.session_state.live_running = True
    st.session_state.live_error = None
    st.session_state.frame_count = 0

    frame, metric, sensors = st.session_state.sim.step_render_frame(
        speed=0.0,
        camera=str(st.session_state.camera),
        target_fps=int(round(LIVE_FPS)),
    )

    st.session_state.last_frame = frame
    st.session_state.last_metric = metric
    st.session_state.last_sensors = sensors
    st.session_state.last_seed = int(st.session_state.cfg.seed)
    st.session_state.last_drop = float(st.session_state.cfg.drop_height)
    st.session_state.last_camera = str(st.session_state.camera)


def advance_live_once() -> None:
    try:
        frame, metric, sensors = st.session_state.sim.step_render_frame(
            speed=float(st.session_state.speed),
            camera=str(st.session_state.camera),
            target_fps=int(round(LIVE_FPS)),
        )
        st.session_state.last_frame = frame
        st.session_state.last_metric = metric
        st.session_state.last_sensors = sensors
        st.session_state.last_camera = str(st.session_state.camera)
        st.session_state.frame_count = int(st.session_state.get("frame_count", 0)) + 1
        st.session_state.live_error = None
    except Exception as exc:
        st.session_state.live_error = f"{type(exc).__name__}: {exc}"
        st.session_state.live_running = False


def ensure_static_render() -> None:
    try:
        if (
            st.session_state.get("last_frame") is None
            or st.session_state.get("last_camera") != st.session_state.camera
        ):
            frame, metric, sensors = st.session_state.sim.step_render_frame(
                speed=0.0,
                camera=str(st.session_state.camera),
                target_fps=int(round(LIVE_FPS)),
            )
            st.session_state.last_frame = frame
            st.session_state.last_metric = metric
            st.session_state.last_sensors = sensors
            st.session_state.last_camera = str(st.session_state.camera)
    except Exception as exc:
        st.session_state.live_error = f"{type(exc).__name__}: {exc}"


def live_should_run() -> bool:
    return bool(
        st.session_state.live_running
        and float(st.session_state.speed) > 0.0
        and not st.session_state.live_error
    )


def recent_series(key: str, n: int = 90) -> List[float]:
    hist = st.session_state.sim.history[-n:]
    return [float(item.get(key, 0.0)) for item in hist]


def show_metric(
    slot: Any,
    label: str,
    value: str,
    chart_values: Optional[List[float]] = None,
) -> None:
    try:
        if chart_values:
            slot.metric(label=label, value=value, chart_values=chart_values)
        else:
            slot.metric(label=label, value=value)
    except Exception:
        slot.metric(label=label, value=value)


def show_generated_video(
    result: Tuple[str, str, pd.DataFrame, pd.DataFrame, Dict[str, Any]],
    key_prefix: str,
) -> None:
    display_path, download_path, _, _, summary = result

    if display_path and os.path.exists(display_path):
        if display_path.lower().endswith(".gif"):
            try:
                st.image(display_path, use_container_width=True)
            except Exception:
                with open(display_path, "rb") as fh:
                    st.image(fh.read(), use_container_width=True)
        else:
            try:
                st.video(display_path, autoplay=True, muted=True, loop=True)
            except TypeError:
                st.video(display_path)
            except Exception as exc:
                st.warning(f"Video display failed: {exc}")
    else:
        st.warning("No playable animated file was generated. The live simulation below still runs.")

    if download_path and os.path.exists(download_path):
        suffix = os.path.splitext(download_path)[1].lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".gif": "image/gif",
            ".avi": "video/x-msvideo",
        }
        mime = mime_map.get(suffix, "application/octet-stream")

        with open(download_path, "rb") as fh:
            video_bytes = fh.read()

        file_name = os.path.basename(download_path)
        st.download_button(
            "Download generated demo file",
            data=video_bytes,
            file_name=file_name,
            mime=mime,
            key=f"{key_prefix}-download",
        )

    if summary:
        st.dataframe(pd.DataFrame([summary]).T, use_container_width=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


if "cfg" not in st.session_state:
    st.session_state.cfg = SimConfig()

if "camera" not in st.session_state:
    st.session_state.camera = "iso"

if "speed" not in st.session_state:
    st.session_state.speed = 1.0

if "chunk_s" not in st.session_state:
    st.session_state.chunk_s = DEFAULT_LIVE_CHUNK_S

if "live_running" not in st.session_state:
    st.session_state.live_running = True

if "live_error" not in st.session_state:
    st.session_state.live_error = None

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "last_metric" not in st.session_state:
    st.session_state.last_metric = {}

if "last_sensors" not in st.session_state:
    st.session_state.last_sensors = {}

if "last_seed" not in st.session_state:
    st.session_state.last_seed = int(st.session_state.cfg.seed)

if "last_drop" not in st.session_state:
    st.session_state.last_drop = float(st.session_state.cfg.drop_height)

if "last_camera" not in st.session_state:
    st.session_state.last_camera = str(st.session_state.camera)

if "preview_result" not in st.session_state:
    st.session_state.preview_result = None

if "full_result" not in st.session_state:
    st.session_state.full_result = None

if "sim" not in st.session_state:
    reset_simulation()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


cfg = st.session_state.cfg

st.sidebar.title("🎛️ Simulation controls")
st.sidebar.caption(
    "One-page humanoid demo. Live simulation starts automatically. "
    "No tabs, no API key, no external asset download."
)

cfg.seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=int(cfg.seed),
    step=1,
    key="seed_widget",
)

cfg.drop_height = st.sidebar.slider(
    "Drop height (m)",
    min_value=0.0,
    max_value=0.12,
    value=float(cfg.drop_height),
    step=0.005,
    key="drop_widget",
)

st.sidebar.subheader("Actuator / PD")

cfg.kp = st.sidebar.slider(
    "Joint PD kp",
    min_value=20.0,
    max_value=500.0,
    value=float(cfg.kp),
    step=5.0,
    key="kp_widget",
)

cfg.kd = st.sidebar.slider(
    "Joint PD kd",
    min_value=0.5,
    max_value=60.0,
    value=float(cfg.kd),
    step=0.5,
    key="kd_widget",
)

cfg.torque_limit = st.sidebar.slider(
    "Torque limit (Nm)",
    min_value=10.0,
    max_value=180.0,
    value=float(cfg.torque_limit),
    step=5.0,
    key="torque_widget",
)

cfg.delay_ms = st.sidebar.slider(
    "Actuator delay (ms)",
    min_value=0.0,
    max_value=30.0,
    value=float(cfg.delay_ms),
    step=1.0,
    key="delay_widget",
)

st.sidebar.subheader("Sensors")

cfg.gyro_std = st.sidebar.slider(
    "IMU gyro noise std (rad/s)",
    min_value=0.0,
    max_value=0.08,
    value=float(cfg.gyro_std),
    step=0.002,
    key="gyro_widget",
)

cfg.accel_std = st.sidebar.slider(
    "IMU accel noise std (m/s²)",
    min_value=0.0,
    max_value=1.0,
    value=float(cfg.accel_std),
    step=0.02,
    key="accel_widget",
)

cfg.encoder_std = st.sidebar.slider(
    "Encoder noise std (rad)",
    min_value=0.0,
    max_value=0.005,
    value=float(cfg.encoder_std),
    step=0.0001,
    format="%.4f",
    key="encoder_widget",
)

cfg.camera_noise = st.sidebar.slider(
    "Camera noise",
    min_value=0.0,
    max_value=0.08,
    value=float(cfg.camera_noise),
    step=0.005,
    key="camera_noise_widget",
)

st.sidebar.subheader("Scenario")

cfg.push_time = st.sidebar.slider(
    "Push time (s)",
    min_value=1.0,
    max_value=15.0,
    value=float(cfg.push_time),
    step=0.5,
    key="push_time_widget",
)

cfg.push_impulse = st.sidebar.slider(
    "Push impulse (Ns)",
    min_value=0.0,
    max_value=80.0,
    value=float(cfg.push_impulse),
    step=1.0,
    key="push_impulse_widget",
)

cfg.scripted_start = st.sidebar.slider(
    "Scripted motion start (s)",
    min_value=2.0,
    max_value=20.0,
    value=float(cfg.scripted_start),
    step=0.5,
    key="script_widget",
)

cfg.duration_s = st.sidebar.slider(
    "Full video duration (s)",
    min_value=4.0,
    max_value=30.0,
    value=float(cfg.duration_s),
    step=1.0,
    key="duration_widget",
)

st.session_state.camera = st.sidebar.selectbox(
    "Camera",
    options=["iso", "front", "side"],
    index=["iso", "front", "side"].index(st.session_state.camera),
    key="camera_widget",
)

st.session_state.speed = st.sidebar.slider(
    "Live speed",
    min_value=0.0,
    max_value=3.0,
    value=float(st.session_state.speed),
    step=0.25,
    key="speed_widget",
)

st.session_state.chunk_s = st.sidebar.slider(
    "Live refresh chunk (s)",
    min_value=0.3,
    max_value=2.0,
    value=float(st.session_state.chunk_s),
    step=0.05,
    key="chunk_widget",
    help="Lower means more responsive but more page refreshes.",
)

use_fragment = st.sidebar.checkbox(
    "Use smoother fragment mode",
    value=False,
    disabled=not USE_LIVE_FRAGMENT,
    key="use_fragment_widget",
    help="Only use this if your Streamlit version supports st.fragment(run_every=...).",
)

engine_mode = "fragment" if USE_LIVE_FRAGMENT and use_fragment else "loop"

btn1, btn2 = st.sidebar.columns(2)

if btn1.button("⏸️ Pause", use_container_width=True):
    st.session_state.live_running = False

if btn2.button("▶️ Resume", use_container_width=True):
    st.session_state.live_running = True
    st.session_state.live_error = None

if st.sidebar.button("🔄 Reset simulation", use_container_width=True):
    reset_simulation()

st.sidebar.info(
    "Default mode is the compatible full-page live loop. "
    "If the robot does not move, keep 'Use smoother fragment mode' unchecked."
)


# ---------------------------------------------------------------------------
# Apply config changes
# ---------------------------------------------------------------------------


if (
    int(cfg.seed) != int(st.session_state.last_seed)
    or abs(float(cfg.drop_height) - float(st.session_state.last_drop)) > 1e-9
):
    reset_simulation()

if not live_should_run():
    ensure_static_render()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


st.title("🤖 Humanoid Robot Physics Simulation — Clear One-Page Demo")
st.caption(
    "Live simulation, generated animated demo, sensor telemetry, and validation checks are all visible on one page. "
    f"Animation engine: {engine_mode}."
)


# ---------------------------------------------------------------------------
# Live simulation placeholders
# ---------------------------------------------------------------------------


st.subheader("1. Live humanoid simulation")

live_image_slot = st.empty()
metric_cols = st.columns(5)
metric_slots = [col.empty() for col in metric_cols]
status_slot = st.empty()
error_slot = st.empty()


def draw_live() -> None:
    frame = st.session_state.last_frame
    metric = st.session_state.last_metric
    sensors = st.session_state.last_sensors

    if frame is None:
        frame = np.zeros(
            (int(st.session_state.cfg.height), int(st.session_state.cfg.width), 3),
            dtype=np.uint8,
        )

    live_image_slot.image(frame, channels="RGB", use_container_width=True)

    show_metric(
        metric_slots[0],
        "Sim time",
        f"{float(metric.get('time', 0.0)):.2f} s",
        recent_series("time"),
    )
    show_metric(
        metric_slots[1],
        "RTF",
        f"{float(metric.get('rtf', 0.0)):.2f}x",
        recent_series("rtf"),
    )
    show_metric(
        metric_slots[2],
        "Penetration",
        f"{float(metric.get('penetration', 0.0)) * 1000.0:.2f} mm",
        [v * 1000.0 for v in recent_series("penetration")],
    )
    show_metric(
        metric_slots[3],
        "Energy",
        f"{float(metric.get('energy', 0.0)):.1f} J",
        recent_series("energy"),
    )
    show_metric(
        metric_slots[4],
        "Contacts",
        f"{int(metric.get('contact_count', 0))}",
        recent_series("contact_count"),
    )

    status = "running" if live_should_run() else "paused"
    status_slot.caption(
        f"Status: {status} | engine: {engine_mode} | camera: {st.session_state.camera} | "
        f"speed: {float(st.session_state.speed):.2f}x | "
        f"seed: {st.session_state.cfg.seed} | "
        f"frames: {int(st.session_state.frame_count)} | "
        f"run_id: {st.session_state.sim.run_id}"
    )

    if st.session_state.live_error:
        error_slot.error(f"Live simulation error: {st.session_state.live_error}")
    else:
        error_slot.empty()


draw_live()


# ---------------------------------------------------------------------------
# Optional fragment mode
# ---------------------------------------------------------------------------


if engine_mode == "fragment":
    @st.fragment(run_every=1.0 / max(1.0, LIVE_FPS))
    def live_fragment() -> None:
        try:
            if live_should_run():
                advance_live_once()
            else:
                ensure_static_render()
            draw_live()
        except Exception as exc:
            st.session_state.live_error = f"{type(exc).__name__}: {exc}"
            st.session_state.live_running = False
            draw_live()

    live_fragment()


# ---------------------------------------------------------------------------
# Generated demo file
# ---------------------------------------------------------------------------


st.divider()
st.subheader("2. Generated animated demo file")

preview_cfg = SimConfig(
    seed=7,
    duration_s=PREVIEW_DURATION_S,
    render_fps=PREVIEW_FPS,
    width=PREVIEW_WIDTH,
    height=PREVIEW_HEIGHT,
    push_time=1.2,
    scripted_start=1.6,
    drop_height=0.04,
)
preview_json = json.dumps(asdict(preview_cfg), sort_keys=True)

if st.session_state.preview_result is None:
    try:
        with st.spinner("Generating short animated demo file..."):
            st.session_state.preview_result = generate_demo_video(preview_json)
    except Exception as exc:
        st.session_state.preview_result = make_error_result(f"{type(exc).__name__}: {exc}")

show_generated_video(st.session_state.preview_result, "preview")

st.markdown("#### Generate a longer video with current sidebar settings")

if st.button("Generate full demo video", use_container_width=True):
    full_values = asdict(st.session_state.cfg)
    full_values.update(
        {
            "duration_s": min(12.0, max(6.0, float(st.session_state.cfg.duration_s))),
            "render_fps": 12,
            "width": 384,
            "height": 216,
        }
    )
    full_json = json.dumps(full_values, sort_keys=True)

    try:
        with st.spinner("Rendering full demo video..."):
            st.session_state.full_result = generate_demo_video(full_json)
    except Exception as exc:
        st.session_state.full_result = make_error_result(f"{type(exc).__name__}: {exc}")

if st.session_state.full_result is not None:
    st.markdown("#### Full generated video")
    show_generated_video(st.session_state.full_result, "full")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


st.divider()
st.subheader("3. Validation checks")

active_result = st.session_state.full_result or st.session_state.preview_result
active_summary = active_result[4] if active_result else {}

if active_summary and "error" not in active_summary:
    pen = float(active_summary.get("max_penetration_mm", float("nan")))
    drift = float(active_summary.get("energy_drift_pct_settled_stand", float("nan")))
    rtf = float(active_summary.get("rtf_measured", float("nan")))
    det_same = float(active_summary.get("determinism_max_abs_same", float("nan")))
    det_diff = float(active_summary.get("determinism_max_abs_different_seed", float("nan")))

    check_rows = [
        {
            "Check": "Max penetration",
            "Measured": f"{pen:.3f} mm",
            "Threshold": f"{cfg.max_penetration_m * 1000.0:.1f} mm",
            "Result": "PASS" if active_summary.get("penetration_pass", False) else "FAIL",
        },
        {
            "Check": "Settled-standing energy drift",
            "Measured": f"{drift:.3f} %",
            "Threshold": f"±{cfg.energy_drift_max_pct:.1f} %",
            "Result": "PASS" if active_summary.get("energy_drift_pass", False) else "FAIL",
        },
        {
            "Check": "Physics real-time factor",
            "Measured": f"{rtf:.2f}x",
            "Threshold": f">={cfg.rtf_min:.1f}x",
            "Result": "PASS" if active_summary.get("rtf_pass", False) else "FAIL",
        },
        {
            "Check": "Determinism same seed",
            "Measured": f"{det_same:.3e}",
            "Threshold": f"<={cfg.determinism_tol:.1e}",
            "Result": "PASS" if active_summary.get("determinism_same_seed_pass", False) else "FAIL",
        },
        {
            "Check": "Different seed changes noisy outputs",
            "Measured": f"{det_diff:.3e}",
            "Threshold": f">{cfg.determinism_tol:.1e} when noise enabled",
            "Result": "PASS" if active_summary.get("determinism_different_seed_pass", False) else "FAIL",
        },
    ]

    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)
else:
    st.info("No validation summary available yet.")


# ---------------------------------------------------------------------------
# Live telemetry
# ---------------------------------------------------------------------------


st.divider()
st.subheader("4. Live sensor telemetry")

sensors = st.session_state.last_sensors

if sensors:
    ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)

    with ic1:
        st.metric("Gyro X", f"{float(sensors.get('imu_gx', 0.0)):.4f}")
    with ic2:
        st.metric("Gyro Y", f"{float(sensors.get('imu_gy', 0.0)):.4f}")
    with ic3:
        st.metric("Gyro Z", f"{float(sensors.get('imu_gz', 0.0)):.4f}")
    with ic4:
        st.metric("Accel X", f"{float(sensors.get('imu_ax', 0.0)):.3f}")
    with ic5:
        st.metric("Accel Y", f"{float(sensors.get('imu_ay', 0.0)):.3f}")
    with ic6:
        st.metric("Accel Z", f"{float(sensors.get('imu_az', 0.0)):.3f}")

    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        st.metric("Left foot Fz", f"{float(sensors.get('left_foot_fz', 0.0)):.2f} N")
    with fc2:
        st.metric("Right foot Fz", f"{float(sensors.get('right_foot_fz', 0.0)):.2f} N")
    with fc3:
        st.metric("Root Z", f"{float(sensors.get('gt_root_z', 0.0)):.3f} m")
    with fc4:
        st.metric("Contact count", f"{int(float(sensors.get('gt_contact_count', 0.0)))}")

    sensor_tail = pd.DataFrame(st.session_state.sim.sensor_history).tail(12)
    if not sensor_tail.empty:
        st.dataframe(sensor_tail, use_container_width=True)
    else:
        st.info("No sensor history yet.")
else:
    st.info("No live sensor data yet.")


# ---------------------------------------------------------------------------
# Generated demo analytics
# ---------------------------------------------------------------------------


st.divider()
st.subheader("5. Generated demo analytics")

if active_result:
    _, _, active_hist, active_sens, _ = active_result
else:
    active_hist = pd.DataFrame()
    active_sens = pd.DataFrame()

if not active_hist.empty:
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("###### Energy")
        st.line_chart(active_hist[["time", "energy"]].set_index("time"))

    with ch2:
        st.markdown("###### Penetration")
        st.line_chart(active_hist[["time", "penetration"]].set_index("time"))

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("###### Root height")
        st.line_chart(active_hist[["time", "root_z"]].set_index("time"))

    with ch4:
        st.markdown("###### Tracking error")
        st.line_chart(active_hist[["time", "tracking_error"]].set_index("time"))
else:
    st.info("No generated analytics available yet.")

if not active_sens.empty:
    sh1, sh2 = st.columns(2)

    with sh1:
        st.markdown("###### IMU gyro")
        st.line_chart(active_sens[["time", "imu_gx", "imu_gy", "imu_gz"]].set_index("time"))

    with sh2:
        st.markdown("###### Foot vertical force")
        fz_cols = [c for c in active_sens.columns if c.endswith("_fz")]
        if fz_cols:
            st.line_chart(active_sens[["time"] + fz_cols].set_index("time"))
        else:
            st.info("No foot force columns available.")


# ---------------------------------------------------------------------------
# Download live logs
# ---------------------------------------------------------------------------


st.divider()
st.subheader("6. Download live logs")

live_hist = pd.DataFrame(st.session_state.sim.history)
live_sens = pd.DataFrame(st.session_state.sim.sensor_history)

d1, d2 = st.columns(2)

with d1:
    if not live_hist.empty:
        st.download_button(
            "Download live metrics CSV",
            data=live_hist.to_csv(index=False).encode("utf-8"),
            file_name="live_metrics.csv",
            mime="text/csv",
            key="download-live-metrics",
            use_container_width=True,
        )
    else:
        st.info("No live metrics yet.")

with d2:
    if not live_sens.empty:
        st.download_button(
            "Download live sensors CSV",
            data=live_sens.to_csv(index=False).encode("utf-8"),
            file_name="live_sensors.csv",
            mime="text/csv",
            key="download-live-sensors",
            use_container_width=True,
        )
    else:
        st.info("No live sensor data yet.")


# ---------------------------------------------------------------------------
# Notes and limitations
# ---------------------------------------------------------------------------


st.divider()
st.subheader("7. What this demo is and is not")

st.markdown(
    """
    **What it shows**

    - A humanoid drops, settles, stands with PD control, resists a push, and plays a scripted motion.
    - Simulated IMU, encoder, foot force/torque, and camera noise are visible.
    - An animated demo file is generated automatically.
    - Determinism and basic validation checks are computed from scripted runs.

    **Limitations**

    - This is a lightweight built-in simulation surrogate for fast visualization.
    - It is not a certified MuJoCo, Isaac Sim, PyBullet, or ROS 2 hardware validator.
    - Contacts are simplified penalty contacts.
    - Results are useful for demo, debugging, and control visualization, not real-robot safety certification.
    """
)


# ---------------------------------------------------------------------------
# Forced full-page live loop fallback
# ---------------------------------------------------------------------------


if engine_mode == "loop" and live_should_run():
    interval = 1.0 / max(1.0, LIVE_FPS)
    end_time = time.time() + float(st.session_state.chunk_s)

    try:
        while time.time() < end_time:
            loop_start = time.time()

            advance_live_once()

            if st.session_state.live_error:
                break

            draw_live()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    except Exception as exc:
        st.session_state.live_error = f"{type(exc).__name__}: {exc}"
        st.session_state.live_running = False

    st.rerun()