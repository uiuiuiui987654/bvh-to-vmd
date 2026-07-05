import argparse
import math
import os
import struct
from dataclasses import dataclass

import numpy as np

from render_mmd_pose import apply_pose, load_pmx_bones


BONE_NAME_SIZE = 15
IDENTITY_QUAT = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
FINGER_BONES = (
    "左親指０",
    "左親指１",
    "左親指２",
    "左人指１",
    "左人指２",
    "左人指３",
    "左中指１",
    "左中指２",
    "左中指３",
    "左薬指１",
    "左薬指２",
    "左薬指３",
    "左小指１",
    "左小指２",
    "左小指３",
    "右親指０",
    "右親指１",
    "右親指２",
    "右人指１",
    "右人指２",
    "右人指３",
    "右中指１",
    "右中指２",
    "右中指３",
    "右薬指１",
    "右薬指２",
    "右薬指３",
    "右小指１",
    "右小指２",
    "右小指３",
)
GROUND_FIT_BONES = (
    "センター",
    "グルーブ",
    "腰",
    "下半身",
    "上半身",
    "上半身2",
    "首",
    "頭",
    "左肩",
    "左腕",
    "左ひじ",
    "左手首",
    "右肩",
    "右腕",
    "右ひじ",
    "右手首",
    "左足",
    "左ひざ",
    "左足首",
    "左つま先",
    "右足",
    "右ひざ",
    "右足首",
    "右つま先",
)


@dataclass
class BvhJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]
    channel_start: int = 0


def fixed_sjis(text, size):
    raw = text.encode("shift_jis", errors="ignore")
    if len(raw) <= size:
        return raw + b"\x00" * (size - len(raw))
    out = bytearray()
    for char in text:
        chunk = char.encode("shift_jis", errors="ignore")
        if len(out) + len(chunk) > size:
            break
        out.extend(chunk)
    return bytes(out) + b"\x00" * (size - len(out))


def default_interpolation():
    return bytes([20, 20, 107, 107] * 4 + [0] * 48)


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def rot_x(deg):
    r = math.radians(float(deg))
    c, s = math.cos(r), math.sin(r)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)


def rot_y(deg):
    r = math.radians(float(deg))
    c, s = math.cos(r), math.sin(r)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)


def rot_z(deg):
    r = math.radians(float(deg))
    c, s = math.cos(r), math.sin(r)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def axis_rot(axis, deg):
    if axis == "X":
        return rot_x(deg)
    if axis == "Y":
        return rot_y(deg)
    if axis == "Z":
        return rot_z(deg)
    raise ValueError(axis)


def mat_to_quat(mat):
    m = np.asarray(mat, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-8)) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-8)) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-8)) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array((x, y, z, w), dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return IDENTITY_QUAT.copy()
    q /= n
    if q[3] < 0.0:
        q = -q
    return q


def quat_slerp_identity(quat, strength):
    strength = max(0.0, min(1.0, float(strength)))
    quat = np.asarray(quat, dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-8:
        return IDENTITY_QUAT.copy()
    quat = quat / n
    if quat[3] < 0.0:
        quat = -quat
    dot = max(-1.0, min(1.0, float(quat[3])))
    if dot > 0.9995:
        out = IDENTITY_QUAT + strength * (quat - IDENTITY_QUAT)
        out /= max(float(np.linalg.norm(out)), 1e-8)
        return out
    theta = math.acos(dot)
    s0 = math.sin((1.0 - strength) * theta) / math.sin(theta)
    s1 = math.sin(strength * theta) / math.sin(theta)
    out = s0 * IDENTITY_QUAT + s1 * quat
    out /= max(float(np.linalg.norm(out)), 1e-8)
    return out


def quat_clamp_angle(quat, max_degrees):
    max_degrees = float(max_degrees)
    if max_degrees <= 0.0:
        return IDENTITY_QUAT.copy()
    quat = np.asarray(quat, dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-8:
        return IDENTITY_QUAT.copy()
    quat = quat / n
    if quat[3] < 0.0:
        quat = -quat
    angle = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, float(quat[3])))))
    if angle <= max_degrees:
        return quat
    return quat_slerp_identity(quat, max_degrees / max(angle, 1e-8))


def rotation_between(a, b):
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return np.eye(3)
    v = np.cross(a, b)
    c = max(-1.0, min(1.0, float(np.dot(a, b))))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, np.array((1.0, 0.0, 0.0), dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(a, np.array((0.0, 1.0, 0.0), dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        x, y, z = axis
        return np.array(
            (
                (2 * x * x - 1, 2 * x * y, 2 * x * z),
                (2 * x * y, 2 * y * y - 1, 2 * y * z),
                (2 * x * z, 2 * y * z, 2 * z * z - 1),
            ),
            dtype=np.float64,
        )
    vx = np.array(((0.0, -v[2], v[1]), (v[2], 0.0, -v[0]), (-v[1], v[0], 0.0)), dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def parse_bvh(path):
    lines = [line.strip() for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    joints = []
    stack = []
    current = None
    channel_cursor = 0
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        head = parts[0]
        if head in {"ROOT", "JOINT"}:
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(parts[1], parent, np.zeros(3), []))
            current = len(joints) - 1
        elif head == "End":
            depth = 0
            i += 1
            while i < len(lines):
                if "{" in lines[i]:
                    depth += 1
                if "}" in lines[i]:
                    depth -= 1
                    if depth <= 0:
                        break
                i += 1
        elif head == "{":
            if current is not None:
                stack.append(current)
                current = None
        elif head == "}":
            if stack:
                stack.pop()
        elif head == "OFFSET":
            joints[stack[-1]].offset = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
        elif head == "CHANNELS":
            count = int(parts[1])
            channels = parts[2 : 2 + count]
            joints[stack[-1]].channels = channels
            joints[stack[-1]].channel_start = channel_cursor
            channel_cursor += count
        elif head == "MOTION":
            break
        i += 1

    while i < len(lines) and not lines[i].startswith("Frames:"):
        i += 1
    frame_count = int(lines[i].split()[1])
    i += 1
    frame_time = float(lines[i].split()[-1])
    i += 1
    data = []
    for line in lines[i : i + frame_count]:
        if line:
            data.append([float(x) for x in line.split()])
    return joints, np.asarray(data, dtype=np.float64), frame_time


def bvh_global_positions(joints, motion):
    t_count = motion.shape[0]
    j_count = len(joints)
    positions = np.zeros((t_count, j_count, 3), dtype=np.float64)
    rotations = np.zeros((t_count, j_count, 3, 3), dtype=np.float64)
    for t in range(t_count):
        for i, joint in enumerate(joints):
            values = motion[t, joint.channel_start : joint.channel_start + len(joint.channels)]
            local_pos = np.zeros(3, dtype=np.float64)
            local_rot = np.eye(3, dtype=np.float64)
            for channel, value in zip(joint.channels, values):
                if channel.endswith("position"):
                    local_pos["XYZ".index(channel[0])] = value
                elif channel.endswith("rotation"):
                    local_rot = local_rot @ axis_rot(channel[0], value)
            if joint.parent < 0:
                rotations[t, i] = local_rot
                positions[t, i] = joint.offset + local_pos
            else:
                p = joint.parent
                rotations[t, i] = rotations[t, p] @ local_rot
                positions[t, i] = positions[t, p] + rotations[t, p] @ (joint.offset + local_pos)
    return positions


def bvh_global_rotations(joints, motion):
    t_count = motion.shape[0]
    j_count = len(joints)
    rotations = np.zeros((t_count, j_count, 3, 3), dtype=np.float64)
    for t in range(t_count):
        for i, joint in enumerate(joints):
            values = motion[t, joint.channel_start : joint.channel_start + len(joint.channels)]
            local_rot = np.eye(3, dtype=np.float64)
            for channel, value in zip(joint.channels, values):
                if channel.endswith("rotation"):
                    local_rot = local_rot @ axis_rot(channel[0], value)
            if joint.parent < 0:
                rotations[t, i] = local_rot
            else:
                rotations[t, i] = rotations[t, joint.parent] @ local_rot
    return rotations


def bvh_local_rotations(joints, motion):
    t_count = motion.shape[0]
    j_count = len(joints)
    rotations = np.zeros((t_count, j_count, 3, 3), dtype=np.float64)
    for t in range(t_count):
        for i, joint in enumerate(joints):
            values = motion[t, joint.channel_start : joint.channel_start + len(joint.channels)]
            local_rot = np.eye(3, dtype=np.float64)
            for channel, value in zip(joint.channels, values):
                if channel.endswith("rotation"):
                    local_rot = local_rot @ axis_rot(channel[0], value)
            rotations[t, i] = local_rot
    return rotations


def to_mmd_space(points):
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., 2] *= -1.0
    return out


def estimate_bvh_height(global_positions):
    first = np.asarray(global_positions[0], dtype=np.float64)
    return float(np.max(first[:, 1]) - np.min(first[:, 1]))


def estimate_pmx_height(pmx_bones):
    positions = np.array([bone["pos"] for bone in pmx_bones], dtype=np.float64)
    return float(np.max(positions[:, 1]) - np.min(positions[:, 1]))


def resolve_position_scale(value, source_global_positions, pmx_bones):
    text = str(value).strip().lower()
    if text not in {"auto", "0", "0.0"}:
        scale = float(text)
        if scale <= 0.0:
            raise ValueError("--position-scale must be positive, 0, or auto")
        return scale, "manual"

    source_height = estimate_bvh_height(source_global_positions)
    target_height = estimate_pmx_height(pmx_bones)
    if source_height <= 1e-6 or target_height <= 1e-6:
        return 0.0025, "fallback"
    return target_height / source_height, "auto"


def rotation_to_mmd_space(rot):
    mirror = np.diag((1.0, 1.0, -1.0))
    return mirror @ rot @ mirror


def bone_frame(name, frame, pos=(0.0, 0.0, 0.0), quat=IDENTITY_QUAT):
    return b"".join(
        (
            fixed_sjis(name, BONE_NAME_SIZE),
            struct.pack("<I", int(frame)),
            struct.pack("<3f", float(pos[0]), float(pos[1]), float(pos[2])),
            struct.pack("<4f", float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            default_interpolation(),
        )
    )


def model_display_frame(frame, ik_names, enabled=False):
    out = bytearray()
    out += struct.pack("<I", int(frame))
    out += struct.pack("<B", 1)
    out += struct.pack("<I", len(ik_names))
    for name in ik_names:
        out += fixed_sjis(name, 20)
        out += struct.pack("<B", 1 if enabled else 0)
    return bytes(out)


def write_vmd(path, chunks, frame_numbers, foot_ik_display=False):
    out = bytearray()
    out += fixed_sjis("Vocaloid Motion Data 0002", 30)
    out += fixed_sjis("KimodoSolver", 20)
    out += struct.pack("<I", len(chunks))
    out += b"".join(chunks)
    out += struct.pack("<IIII", 0, 0, 0, 0)
    displays = [model_display_frame(frame, ("左足ＩＫ", "右足ＩＫ"), enabled=foot_ik_display) for frame in frame_numbers]
    out += struct.pack("<I", len(displays))
    out += b"".join(displays)
    open(path, "wb").write(out)


def stabilize_foot_targets(targets_by_name, frame_numbers, ground_threshold=0.65, min_segment_len=3):
    stabilized = {}
    for name, frame_to_target in targets_by_name.items():
        targets = np.array([frame_to_target[frame] for frame in frame_numbers], dtype=np.float64)
        if len(targets) == 0:
            stabilized[name] = {}
            continue
        ground_y = float(np.percentile(targets[:, 1], 8))
        contact = targets[:, 1] <= ground_y + ground_threshold
        out = targets.copy()
        i = 0
        while i < len(targets):
            if not contact[i]:
                i += 1
                continue
            j = i + 1
            while j < len(targets) and contact[j]:
                j += 1
            if j - i >= min_segment_len:
                anchor = targets[i:j].mean(axis=0)
                anchor[1] = ground_y
                for k in range(i, j):
                    out[k, 0] = anchor[0]
                    out[k, 1] = anchor[1]
                    out[k, 2] = anchor[2]
            i = j

        # Keep flight frames slightly above ground and smooth hard jumps at contact borders.
        out[:, 1] = np.maximum(out[:, 1], ground_y)
        for _ in range(2):
            smoothed = out.copy()
            smoothed[1:-1] = out[:-2] * 0.25 + out[1:-1] * 0.5 + out[2:] * 0.25
            smoothed[contact] = out[contact]
            out = smoothed
        stabilized[name] = {frame: out[idx] for idx, frame in enumerate(frame_numbers)}
    return stabilized


def source_foot_target(
    src,
    src_name_to_idx,
    pmx_bones,
    name_to_idx,
    center,
    side,
):
    src_leg = f"{side}Leg"
    src_foot = f"{side}Foot"
    mmd_leg = "左足" if side == "Left" else "右足"
    mmd_knee = "左ひざ" if side == "Left" else "右ひざ"
    mmd_ankle = "左足首" if side == "Left" else "右足首"
    if src_leg not in src_name_to_idx or src_foot not in src_name_to_idx:
        return None
    if any(name not in name_to_idx for name in (mmd_leg, mmd_knee, mmd_ankle)):
        return None
    source_vec = src[src_name_to_idx[src_foot]] - src[src_name_to_idx[src_leg]]
    source_len = float(np.linalg.norm(source_vec))
    if source_len < 1e-8:
        return None
    pmx_len = float(
        np.linalg.norm(pmx_bones[name_to_idx[mmd_knee]]["pos"] - pmx_bones[name_to_idx[mmd_leg]]["pos"])
        + np.linalg.norm(pmx_bones[name_to_idx[mmd_ankle]]["pos"] - pmx_bones[name_to_idx[mmd_knee]]["pos"])
    )
    target = pmx_bones[name_to_idx[mmd_leg]]["pos"] + source_vec * (pmx_len / source_len)
    target = target + center
    return target


def source_body_frame(src, src_name_to_idx):
    needed = ("Hips", "Chest", "LeftShoulder", "RightShoulder", "LeftLeg", "RightLeg")
    if any(name not in src_name_to_idx for name in needed):
        return None
    hips = src[src_name_to_idx["Hips"]]
    chest = src[src_name_to_idx["Chest"]]
    left_shoulder = src[src_name_to_idx["LeftShoulder"]]
    right_shoulder = src[src_name_to_idx["RightShoulder"]]
    left_leg = src[src_name_to_idx["LeftLeg"]]
    right_leg = src[src_name_to_idx["RightLeg"]]
    up = normalize(chest - hips)
    shoulder_side = normalize(right_shoulder - left_shoulder)
    hip_side = normalize(right_leg - left_leg)
    side = None
    if shoulder_side is not None and hip_side is not None:
        side = normalize(shoulder_side * 0.45 + hip_side * 0.55)
    elif shoulder_side is not None:
        side = shoulder_side
    elif hip_side is not None:
        side = hip_side
    if up is None or side is None:
        return None
    side = side - up * float(np.dot(side, up))
    side = normalize(side)
    if side is None:
        return None
    forward = normalize(np.cross(side, up))
    if forward is None:
        return None
    side = normalize(np.cross(up, forward))
    if side is None:
        return None
    return np.stack((side, up, forward), axis=1)


def infer_body_rotations(src_positions, src_name_to_idx, frame_numbers, mode):
    rotations = {frame: np.eye(3, dtype=np.float64) for frame in frame_numbers}
    if mode == "none":
        return rotations, False
    frames = []
    for frame in frame_numbers:
        frame_mat = source_body_frame(src_positions[frame], src_name_to_idx)
        frames.append(frame_mat)
    valid = [mat for mat in frames if mat is not None]
    if not valid:
        return rotations, False
    up_y = np.array([mat[1, 1] for mat in valid], dtype=np.float64)
    should_apply = mode == "infer" or float(up_y.min()) < -0.35
    if not should_apply:
        return rotations, False
    base = frames[0] if frames[0] is not None else valid[0]
    base_inv = base.T
    for frame, mat in zip(frame_numbers, frames):
        if mat is not None:
            rotations[frame] = mat @ base_inv
    return rotations, True


def body_up_vector(src, src_name_to_idx):
    if "Hips" not in src_name_to_idx or "Chest" not in src_name_to_idx:
        return None
    return normalize(src[src_name_to_idx["Chest"]] - src[src_name_to_idx["Hips"]])


def infer_body_up_rotations(src_positions, src_name_to_idx, frame_numbers, mode):
    rotations = {frame: np.eye(3, dtype=np.float64) for frame in frame_numbers}
    if mode == "none":
        return rotations, False
    base_up = body_up_vector(src_positions[frame_numbers[0]], src_name_to_idx)
    if base_up is None:
        return rotations, False
    ups = []
    for frame in frame_numbers:
        ups.append(body_up_vector(src_positions[frame], src_name_to_idx))
    valid = [up for up in ups if up is not None]
    if not valid:
        return rotations, False
    up_y = np.array([up[1] for up in valid], dtype=np.float64)
    should_apply = mode == "infer" or float(up_y.min()) < -0.35
    if not should_apply:
        return rotations, False
    for frame, up in zip(frame_numbers, ups):
        if up is not None:
            rotations[frame] = rotation_between(base_up, up)
    return rotations, True


def infer_hips_rotations(src_rotations, src_name_to_idx, frame_numbers, mode):
    rotations = {frame: np.eye(3, dtype=np.float64) for frame in frame_numbers}
    if mode == "none" or "Hips" not in src_name_to_idx:
        return rotations, False
    hips_idx = src_name_to_idx["Hips"]
    base = rotation_to_mmd_space(src_rotations[frame_numbers[0], hips_idx])
    base_inv = base.T
    rels = []
    for frame in frame_numbers:
        rel = rotation_to_mmd_space(src_rotations[frame, hips_idx]) @ base_inv
        rels.append(rel)
    up_y = np.array([rel[1, 1] for rel in rels], dtype=np.float64)
    should_apply = mode == "infer" or float(up_y.min()) < -0.35
    if not should_apply:
        return rotations, False
    for frame, rel in zip(frame_numbers, rels):
        rotations[frame] = rel
    return rotations, True


def apply_inverse_body_rotation(src, src_name_to_idx, body_rotation):
    if "Hips" not in src_name_to_idx:
        return src
    origin = src[src_name_to_idx["Hips"]]
    local = (body_rotation.T @ (src - origin).T).T + origin
    return local


def transform_body_rotation(rot, mode):
    if mode == "normal":
        return rot
    if mode == "inverse":
        return rot.T
    if mode == "mirror-x":
        mirror = np.diag((-1.0, 1.0, 1.0))
        return mirror @ rot @ mirror
    if mode == "mirror-y":
        mirror = np.diag((1.0, -1.0, 1.0))
        return mirror @ rot @ mirror
    if mode == "mirror-z":
        mirror = np.diag((1.0, 1.0, -1.0))
        return mirror @ rot @ mirror
    raise ValueError(mode)


def solve_frame_from_local_rotations(
    src_local_rotations,
    frame,
    src_name_to_idx,
    strengths,
    include_feet=True,
):
    source_for_mmd = {
        "下半身": "Hips",
        "上半身": "Spine1",
        "上半身2": "Chest",
        "首": "Neck2",
        "右肩": "RightShoulder",
        "右腕": "RightArm",
        "右ひじ": "RightForeArm",
        "右手首": "RightHand",
        "左肩": "LeftShoulder",
        "左腕": "LeftArm",
        "左ひじ": "LeftForeArm",
        "左手首": "LeftHand",
        "右足": "RightLeg",
        "右ひざ": "RightShin",
        "右足首": "RightFoot",
        "左足": "LeftLeg",
        "左ひざ": "LeftShin",
        "左足首": "LeftFoot",
    }
    solved = {}
    for mmd_name, src_name in source_for_mmd.items():
        if not include_feet and mmd_name in {"左足首", "右足首"}:
            continue
        if src_name not in src_name_to_idx:
            continue
        idx = src_name_to_idx[src_name]
        base = rotation_to_mmd_space(src_local_rotations[0, idx])
        cur = rotation_to_mmd_space(src_local_rotations[frame, idx])
        rel = cur @ base.T
        quat = mat_to_quat(rel)
        strength = strengths.get(mmd_name, 1.0)
        if strength < 1.0:
            quat = quat_slerp_identity(quat, strength)
        solved[mmd_name] = quat
    return solved


def solve_frame_from_global_rotations(
    pmx_bones,
    name_to_idx,
    src_global_rotations,
    frame,
    src_name_to_idx,
    strengths,
    center_rotation=None,
    include_feet=False,
):
    source_for_mmd = {
        "下半身": "Hips",
        "上半身": "Spine1",
        "上半身2": "Chest",
        "首": "Neck2",
        "右肩": "RightShoulder",
        "右腕": "RightArm",
        "右ひじ": "RightForeArm",
        "右手首": "RightHand",
        "左肩": "LeftShoulder",
        "左腕": "LeftArm",
        "左ひじ": "LeftForeArm",
        "左手首": "LeftHand",
        "右足": "RightLeg",
        "右ひざ": "RightShin",
        "右足首": "RightFoot",
        "左足": "LeftLeg",
        "左ひざ": "LeftShin",
        "左足首": "LeftFoot",
    }
    order = [name for name, _pair in build_mappings() if name in source_for_mmd]
    local = {}
    global_rots = [np.eye(3, dtype=np.float64) for _ in pmx_bones]
    center_inv = np.eye(3, dtype=np.float64) if center_rotation is None else center_rotation.T

    def refresh_globals():
        for i, bone in enumerate(pmx_bones):
            parent = bone["parent"]
            lr = local.get(bone["name"], np.eye(3, dtype=np.float64))
            if parent < 0:
                global_rots[i] = lr
            else:
                global_rots[i] = global_rots[parent] @ lr

    refresh_globals()
    for mmd_name in order:
        if not include_feet and mmd_name in {"左足首", "右足首"}:
            continue
        src_name = source_for_mmd[mmd_name]
        if src_name not in src_name_to_idx or mmd_name not in name_to_idx:
            continue
        src_idx = src_name_to_idx[src_name]
        base = rotation_to_mmd_space(src_global_rotations[0, src_idx])
        cur = rotation_to_mmd_space(src_global_rotations[frame, src_idx])
        wanted_global = center_inv @ (cur @ base.T)
        idx = name_to_idx[mmd_name]
        parent = pmx_bones[idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        q = mat_to_quat(parent_global.T @ wanted_global)
        strength = strengths.get(mmd_name, 1.0)
        if strength < 1.0:
            q = quat_slerp_identity(q, strength)
        local[mmd_name] = quat_to_mat(q)
        refresh_globals()
    return {name: mat_to_quat(rot) for name, rot in local.items()}


def make_frame_from_primary_lateral(primary, lateral):
    x = normalize(primary)
    if x is None:
        return None
    lateral = np.asarray(lateral, dtype=np.float64)
    lateral = lateral - x * float(np.dot(lateral, x))
    y = normalize(lateral)
    if y is None:
        return None
    z = np.cross(x, y)
    z = normalize(z)
    if z is None:
        return None
    y = np.cross(z, x)
    y = normalize(y)
    if y is None:
        return None
    return np.stack((x, y, z), axis=1)


def hand_frame_from_lookup(get_pos, wrist, index, middle, pinky):
    try:
        wrist_pos = get_pos(wrist)
        index_pos = get_pos(index)
        middle_pos = get_pos(middle)
        pinky_pos = get_pos(pinky)
    except KeyError:
        return None
    primary = middle_pos - wrist_pos
    lateral = index_pos - pinky_pos
    return make_frame_from_primary_lateral(primary, lateral)


def solve_frame(
    pmx_bones,
    name_to_idx,
    src,
    src_name_to_idx,
    mappings,
    strengths,
    hand_outward=0.0,
    hand_forward=0.0,
    hand_down=0.0,
    knee_hinge=False,
    knee_hinge_sign=1.0,
    leg_solver_mode="ccd",
    leg_max_angle=0.0,
):
    local = {}
    global_rots = [np.eye(3, dtype=np.float64) for _ in pmx_bones]
    global_pos = [bone["pos"].copy() for bone in pmx_bones]

    def refresh_globals():
        for i, bone in enumerate(pmx_bones):
            parent = bone["parent"]
            lr = local.get(bone["name"], np.eye(3, dtype=np.float64))
            if parent < 0:
                global_rots[i] = lr
                global_pos[i] = bone["pos"].copy()
            else:
                rest_offset = bone["pos"] - pmx_bones[parent]["pos"]
                global_rots[i] = global_rots[parent] @ lr
                global_pos[i] = global_pos[parent] + global_rots[parent] @ rest_offset

    def set_global_rotation(bone_name, wanted_global):
        idx = name_to_idx[bone_name]
        parent = pmx_bones[idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        local[bone_name] = parent_global.T @ wanted_global
        refresh_globals()

    def apply_direction(name, src_a, src_b):
        if name not in name_to_idx or src_a not in src_name_to_idx or src_b not in src_name_to_idx:
            return
        child_name = child_for_bone(name)
        if child_name not in name_to_idx:
            return
        idx = name_to_idx[name]
        rest_dir = pmx_bones[name_to_idx[child_name]]["pos"] - pmx_bones[idx]["pos"]
        desired_dir = src[src_name_to_idx[src_b]] - src[src_name_to_idx[src_a]]
        target_global = rotation_between(rest_dir, desired_dir)
        q = mat_to_quat((np.eye(3) if pmx_bones[idx]["parent"] < 0 else global_rots[pmx_bones[idx]["parent"]]).T @ target_global)
        strength = strengths.get(name, 1.0)
        if strength < 1.0:
            q = quat_slerp_identity(q, strength)
            target_global = (np.eye(3) if pmx_bones[idx]["parent"] < 0 else global_rots[pmx_bones[idx]["parent"]]) @ quat_to_mat(q)
        set_global_rotation(name, target_global)

    def ccd_limb(upper_name, mid_name, end_name, src_upper, src_mid, src_end, iterations=8, target_offset=None):
        required = (upper_name, mid_name, end_name)
        if any(name not in name_to_idx for name in required):
            return
        if any(name not in src_name_to_idx for name in (src_upper, src_mid, src_end)):
            return

        upper_idx = name_to_idx[upper_name]
        mid_idx = name_to_idx[mid_name]
        end_idx = name_to_idx[end_name]
        target_len = (
            np.linalg.norm(pmx_bones[mid_idx]["pos"] - pmx_bones[upper_idx]["pos"])
            + np.linalg.norm(pmx_bones[end_idx]["pos"] - pmx_bones[mid_idx]["pos"])
        )
        source_len = (
            np.linalg.norm(src[src_name_to_idx[src_mid]] - src[src_name_to_idx[src_upper]])
            + np.linalg.norm(src[src_name_to_idx[src_end]] - src[src_name_to_idx[src_mid]])
        )
        desired = src[src_name_to_idx[src_end]] - src[src_name_to_idx[src_upper]]
        desired_dir = normalize(desired)
        if desired_dir is None or source_len < 1e-8:
            return
        desired_dist = min(float(np.linalg.norm(desired)) * target_len / source_len, target_len * 0.98)
        desired_dist = max(desired_dist, target_len * 0.08)
        target = global_pos[upper_idx] + desired_dir * desired_dist
        if target_offset is not None:
            target = target + np.asarray(target_offset, dtype=np.float64)

        for _ in range(iterations):
            for bone_name in (mid_name, upper_name):
                b_idx = name_to_idx[bone_name]
                joint = global_pos[b_idx]
                cur_vec = global_pos[end_idx] - joint
                target_vec = target - joint
                if np.linalg.norm(cur_vec) < 1e-8 or np.linalg.norm(target_vec) < 1e-8:
                    continue
                delta = rotation_between(cur_vec, target_vec)
                set_global_rotation(bone_name, delta @ global_rots[b_idx])

    def weak_end_direction(end_name, child_name, src_a, src_b, strength):
        if strength <= 0.0:
            return
        if end_name not in name_to_idx or child_name not in name_to_idx:
            return
        if src_a not in src_name_to_idx or src_b not in src_name_to_idx:
            return
        idx = name_to_idx[end_name]
        rest_dir = pmx_bones[name_to_idx[child_name]]["pos"] - pmx_bones[idx]["pos"]
        desired_dir = src[src_name_to_idx[src_b]] - src[src_name_to_idx[src_a]]
        target_global = rotation_between(rest_dir, desired_dir)
        parent = pmx_bones[idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        q = mat_to_quat(parent_global.T @ target_global)
        q = quat_slerp_identity(q, strength)
        local[end_name] = quat_to_mat(q)
        refresh_globals()

    def apply_hand_basis(wrist_name, prefix_jp, prefix_src, strength):
        if strength <= 0.0 or wrist_name not in name_to_idx:
            return False
        jp_names = (
            wrist_name,
            f"{prefix_jp}人指１",
            f"{prefix_jp}中指１",
            f"{prefix_jp}小指１",
        )
        src_names = (
            f"{prefix_src}Hand",
            f"{prefix_src}HandIndex1",
            f"{prefix_src}HandMiddle1",
            f"{prefix_src}HandPinky1",
        )
        if any(name not in name_to_idx for name in jp_names):
            return False
        if any(name not in src_name_to_idx for name in src_names):
            return False
        rest_frame = hand_frame_from_lookup(
            lambda name: pmx_bones[name_to_idx[name]]["pos"],
            *jp_names,
        )
        source_frame = hand_frame_from_lookup(
            lambda name: src[src_name_to_idx[name]],
            *src_names,
        )
        if rest_frame is None or source_frame is None:
            return False
        target_global = source_frame @ rest_frame.T
        idx = name_to_idx[wrist_name]
        parent = pmx_bones[idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        q = mat_to_quat(parent_global.T @ target_global)
        q = quat_slerp_identity(q, strength)
        local[wrist_name] = quat_to_mat(q)
        refresh_globals()
        return True

    refresh_globals()
    for mmd_name, src_pair in mappings:
        if mmd_name in {"左手首", "右手首", "左足首", "右足首"}:
            continue
        apply_direction(mmd_name, src_pair[0], src_pair[1])

    ccd_limb(
        "左腕",
        "左ひじ",
        "左手首",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        target_offset=(hand_outward, -hand_down, hand_forward),
    )
    ccd_limb(
        "右腕",
        "右ひじ",
        "右手首",
        "RightArm",
        "RightForeArm",
        "RightHand",
        target_offset=(-hand_outward, -hand_down, hand_forward),
    )
    if leg_solver_mode == "ccd":
        ccd_limb("左足", "左ひざ", "左足首", "LeftLeg", "LeftShin", "LeftFoot")
        ccd_limb("右足", "右ひざ", "右足首", "RightLeg", "RightShin", "RightFoot")
    elif leg_solver_mode == "frozen":
        for leg_name in ("左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"):
            if leg_name in local:
                local[leg_name] = np.eye(3, dtype=np.float64)
        refresh_globals()

    if knee_hinge:
        for knee_name in ("左ひざ", "右ひざ"):
            if knee_name in local:
                mat = local[knee_name]
                angle = math.degrees(math.atan2(float(mat[2, 1]), float(mat[1, 1])))
                angle = max(-160.0, min(160.0, angle * float(knee_hinge_sign)))
                local[knee_name] = rot_x(angle)
                refresh_globals()

    if not apply_hand_basis("左手首", "左", "Left", strengths.get("左手首", 0.0)):
        weak_end_direction("左手首", "左中指１", "LeftHand", "LeftHandMiddle1", strengths.get("左手首", 0.0))
    if not apply_hand_basis("右手首", "右", "Right", strengths.get("右手首", 0.0)):
        weak_end_direction("右手首", "右中指１", "RightHand", "RightHandMiddle1", strengths.get("右手首", 0.0))

    for foot_name in ("左足首", "右足首"):
        if foot_name in name_to_idx:
            local[foot_name] = np.eye(3, dtype=np.float64)
            refresh_globals()

    solved = {name: mat_to_quat(rot) for name, rot in local.items()}
    if leg_max_angle > 0.0:
        for name in ("左足", "左ひざ", "右足", "右ひざ"):
            if name in solved:
                solved[name] = quat_clamp_angle(solved[name], leg_max_angle)
    return solved


def quat_to_mat(quat):
    x, y, z, w = quat
    n = math.sqrt(float(x * x + y * y + z * z + w * w))
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        (
            (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w),
            (2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w),
            (2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y),
        ),
        dtype=np.float64,
    )


def child_for_bone(name):
    return {
        "下半身": "上半身",
        "上半身": "上半身2",
        "上半身2": "首",
        "首": "頭",
        "右肩": "右腕",
        "右腕": "右ひじ",
        "右ひじ": "右手首",
        "左肩": "左腕",
        "左腕": "左ひじ",
        "左ひじ": "左手首",
        "右足": "右ひざ",
        "右ひざ": "右足首",
        "右足首": "右つま先",
        "左足": "左ひざ",
        "左ひざ": "左足首",
        "左足首": "左つま先",
        "右手首": "右中指１",
        "左手首": "左中指１",
    }.get(name)


def build_mappings():
    return [
        ("下半身", ("Hips", "Spine1")),
        ("上半身", ("Spine1", "Chest")),
        ("上半身2", ("Chest", "Neck2")),
        ("首", ("Neck2", "Head")),
        ("右肩", ("RightShoulder", "RightArm")),
        ("右腕", ("RightArm", "RightForeArm")),
        ("右ひじ", ("RightForeArm", "RightHand")),
        ("右手首", ("RightHand", "RightHandMiddle1")),
        ("左肩", ("LeftShoulder", "LeftArm")),
        ("左腕", ("LeftArm", "LeftForeArm")),
        ("左ひじ", ("LeftForeArm", "LeftHand")),
        ("左手首", ("LeftHand", "LeftHandMiddle1")),
        ("右足", ("RightLeg", "RightShin")),
        ("右ひざ", ("RightShin", "RightFoot")),
        ("右足首", ("RightFoot", "RightToeBase")),
        ("左足", ("LeftLeg", "LeftShin")),
        ("左ひざ", ("LeftShin", "LeftFoot")),
        ("左足首", ("LeftFoot", "LeftToeBase")),
    ]


def main():
    parser = argparse.ArgumentParser(description="Convert Kimodo BVH motion to an MMD PMX-friendly VMD.")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--pmx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--position-scale",
        default="auto",
        help="Center/root translation scale. Use auto or 0 to match BVH height to PMX height; old fixed value was 0.0025.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--arm-strength", type=float, default=1.0)
    parser.add_argument("--wrist-strength", type=float, default=0.45)
    parser.add_argument("--hand-outward", type=float, default=0.9)
    parser.add_argument("--hand-forward", type=float, default=-0.25)
    parser.add_argument("--hand-down", type=float, default=2.0)
    parser.add_argument("--torso-strength", type=float, default=0.7)
    parser.add_argument("--flip-torso-strength", type=float, default=0.25)
    parser.add_argument("--leg-strength", type=float, default=1.0)
    parser.add_argument("--flip-leg-strength", type=float, default=0.65)
    parser.add_argument("--foot-strength", type=float, default=0.35)
    parser.add_argument(
        "--motion-fidelity",
        choices=("preserve", "stable"),
        default="preserve",
        help="preserve keeps BVH motion strength during flips; stable uses reduced flip strengths to avoid twisting.",
    )
    parser.add_argument("--leg-solver-mode", choices=("ccd", "direction", "frozen"), default="ccd")
    parser.add_argument("--leg-max-angle", type=float, default=0.0)
    parser.add_argument("--knee-hinge", choices=("off", "flip", "always"), default="flip")
    parser.add_argument("--knee-hinge-sign", type=float, default=1.0)
    parser.add_argument("--pose-solver-mode", choices=("position", "local-rot", "global-rot"), default="position")
    parser.add_argument("--local-rot-feet", choices=("omit", "include"), default="omit")
    parser.add_argument("--foot-ik-mode", choices=("auto", "source", "fk", "none"), default="auto")
    parser.add_argument(
        "--foot-rotation-mode",
        choices=("none", "follow-body"),
        default="none",
        help="Rotate foot IK with inferred body rotation during flips so feet do not stay fixed in world space.",
    )
    parser.add_argument("--no-foot-ik-tracks", action="store_true")
    parser.add_argument("--foot-lock", action="store_true")
    parser.add_argument("--foot-ground-threshold", type=float, default=0.65)
    parser.add_argument("--ground-fit-mode", choices=("none", "first", "range", "all"), default="none")
    parser.add_argument("--ground-fit-start", type=int, default=0)
    parser.add_argument("--ground-fit-end", type=int, default=80)
    parser.add_argument("--ground-fit-strength", type=float, default=1.0)
    parser.add_argument("--ground-fit-target", type=float, default=0.0)
    parser.add_argument("--enable-foot-ik", action="store_true")
    parser.add_argument("--keep-foot-ik", action="store_true")
    parser.add_argument(
        "--body-rotation-mode",
        choices=("none", "auto", "infer"),
        default="auto",
        help="Infer center rotation from body orientation for flips; auto keeps walking motions unchanged.",
    )
    parser.add_argument(
        "--body-frame-mode",
        choices=("full", "up", "hips"),
        default="full",
        help="full uses hips/chest/shoulders; up follows torso up; hips uses BVH Hips global rotation.",
    )
    parser.add_argument(
        "--body-rotation-transform",
        choices=("normal", "inverse", "mirror-x", "mirror-y", "mirror-z"),
        default="normal",
    )
    parser.add_argument(
        "--body-local-compensation",
        choices=("on", "off"),
        default="on",
        help="When center rotation is inferred, solve limbs in center-local space. Turn off to debug side flips.",
    )
    parser.add_argument(
        "--finger-mode",
        choices=("omit", "neutral"),
        default="omit",
        help="Kimodo does not provide reliable finger animation; omit by default or write neutral finger keys.",
    )
    args = parser.parse_args()

    joints, motion, frame_time = parse_bvh(args.bvh)
    src_names = [joint.name for joint in joints]
    src_name_to_idx = {name: i for i, name in enumerate(src_names)}
    source_global_positions = bvh_global_positions(joints, motion)
    src_positions = to_mmd_space(source_global_positions)
    src_rotations = bvh_global_rotations(joints, motion)
    src_local_rotations = bvh_local_rotations(joints, motion)
    pmx_bones = load_pmx_bones(args.pmx)
    name_to_idx = {bone["name"]: i for i, bone in enumerate(pmx_bones)}
    position_scale, position_scale_mode = resolve_position_scale(args.position_scale, source_global_positions, pmx_bones)

    strengths = {}
    for name, _pair in build_mappings():
        if name in {"下半身", "上半身", "上半身2", "首"}:
            strengths[name] = args.torso_strength
        elif name in {"左手首", "右手首"}:
            strengths[name] = args.wrist_strength
        elif name in {"左足首", "右足首"}:
            strengths[name] = args.foot_strength
        elif name.endswith("足") or "ひざ" in name or "足首" in name:
            strengths[name] = args.leg_strength
        else:
            strengths[name] = args.arm_strength

    hips_idx = src_name_to_idx.get("Hips")
    if hips_idx is None:
        raise ValueError("BVH is missing Hips")
    center0 = src_positions[0, hips_idx].copy()
    frame_numbers = list(range(0, src_positions.shape[0], max(1, args.stride)))
    if args.body_frame_mode == "hips":
        body_rotations, body_rotation_enabled = infer_hips_rotations(
            src_rotations,
            src_name_to_idx,
            frame_numbers,
            args.body_rotation_mode,
        )
    elif args.body_frame_mode == "up":
        body_rotations, body_rotation_enabled = infer_body_up_rotations(
            src_positions,
            src_name_to_idx,
            frame_numbers,
            args.body_rotation_mode,
        )
    else:
        body_rotations, body_rotation_enabled = infer_body_rotations(
            src_positions,
            src_name_to_idx,
            frame_numbers,
            args.body_rotation_mode,
        )
    if body_rotation_enabled and args.body_rotation_transform != "normal":
        body_rotations = {
            frame: transform_body_rotation(rot, args.body_rotation_transform)
            for frame, rot in body_rotations.items()
        }
    effective_foot_ik_mode = args.foot_ik_mode
    if effective_foot_ik_mode == "auto":
        effective_foot_ik_mode = "none" if body_rotation_enabled else "source"
    if body_rotation_enabled and args.motion_fidelity == "stable":
        for name, _pair in build_mappings():
            if name in {"下半身", "上半身", "上半身2", "首"}:
                strengths[name] = args.flip_torso_strength
            elif name.endswith("足") or "ひざ" in name or "足首" in name:
                strengths[name] = args.flip_leg_strength
    chunks = []
    foot_targets = {"左足ＩＫ": {}, "右足ＩＫ": {}}
    foot_rotations = {"左足ＩＫ": {}, "右足ＩＫ": {}}
    hand_outward = args.hand_outward if args.motion_fidelity == "stable" else 0.0
    hand_forward = args.hand_forward if args.motion_fidelity == "stable" else 0.0
    hand_down = args.hand_down if args.motion_fidelity == "stable" else 0.0
    for frame in frame_numbers:
        center = (src_positions[frame, hips_idx] - center0) * position_scale
        body_rotation = body_rotations.get(frame, np.eye(3, dtype=np.float64))
        center_quat = mat_to_quat(body_rotation)
        use_local_compensation = body_rotation_enabled and args.body_local_compensation == "on"
        solve_src = apply_inverse_body_rotation(src_positions[frame], src_name_to_idx, body_rotation) if use_local_compensation else src_positions[frame]
        if args.pose_solver_mode == "local-rot":
            solved = solve_frame_from_local_rotations(
                src_local_rotations,
                frame,
                src_name_to_idx,
                strengths,
                include_feet=args.local_rot_feet == "include",
            )
        elif args.pose_solver_mode == "global-rot":
            solved = solve_frame_from_global_rotations(
                pmx_bones,
                name_to_idx,
                src_rotations,
                frame,
                src_name_to_idx,
                strengths,
                center_rotation=body_rotation if body_rotation_enabled else None,
                include_feet=args.local_rot_feet == "include",
            )
        else:
            solved = solve_frame(
                pmx_bones,
                name_to_idx,
                solve_src,
                src_name_to_idx,
                build_mappings(),
                strengths,
                hand_outward=hand_outward,
                hand_forward=hand_forward,
                hand_down=hand_down,
                knee_hinge=args.knee_hinge == "always" or (args.knee_hinge == "flip" and body_rotation_enabled),
                knee_hinge_sign=args.knee_hinge_sign,
                leg_solver_mode=args.leg_solver_mode,
                leg_max_angle=args.leg_max_angle,
            )
        for wrist_name, twist_name in (("左手首", "左手捩"), ("右手首", "右手捩")):
            if wrist_name in solved and twist_name in name_to_idx:
                half = quat_slerp_identity(solved[wrist_name], 0.5)
                solved[twist_name] = half
                solved[wrist_name] = half

        frame_pose = {
            "センター": (center, center_quat),
            "グルーブ": (np.zeros(3), IDENTITY_QUAT.copy()),
            "腰": (np.zeros(3), IDENTITY_QUAT.copy()),
        }
        for name, quat in solved.items():
            frame_pose[name] = (np.zeros(3), quat)

        should_ground_fit = args.ground_fit_mode == "all" or (
            args.ground_fit_mode in {"first", "range"} and args.ground_fit_start <= frame <= args.ground_fit_end
        )
        if should_ground_fit:
            posed_for_ground = apply_pose(pmx_bones, frame_pose)
            ground_indices = [name_to_idx[name] for name in GROUND_FIT_BONES if name in name_to_idx]
            min_y = float(np.min(posed_for_ground[ground_indices, 1])) if ground_indices else float(np.min(posed_for_ground[:, 1]))
            if min_y > args.ground_fit_target:
                center = center.copy()
                center[1] -= (min_y - args.ground_fit_target) * max(0.0, min(1.0, args.ground_fit_strength))
                frame_pose["センター"] = (center, center_quat)

        chunks.append(bone_frame("センター", frame, center, center_quat))
        chunks.append(bone_frame("グルーブ", frame))
        chunks.append(bone_frame("腰", frame))
        for name in sorted(solved):
            chunks.append(bone_frame(name, frame, quat=solved[name]))
        if not args.no_foot_ik_tracks and effective_foot_ik_mode != "none":
            posed = apply_pose(pmx_bones, frame_pose)
            for side, ankle_name, ik_name in (
                ("Left", "左足首", "左足ＩＫ"),
                ("Right", "右足首", "右足ＩＫ"),
            ):
                if ankle_name in name_to_idx and ik_name in name_to_idx:
                    if effective_foot_ik_mode == "source":
                        target = source_foot_target(
                            src_positions[frame],
                            src_name_to_idx,
                            pmx_bones,
                            name_to_idx,
                            center,
                            side,
                        )
                        if target is None:
                            target = posed[name_to_idx[ankle_name]]
                    else:
                        target = posed[name_to_idx[ankle_name]]
                    foot_targets[ik_name][frame] = target.copy()
                    if args.foot_rotation_mode == "follow-body" and body_rotation_enabled:
                        foot_rotations[ik_name][frame] = center_quat.copy()
                    else:
                        foot_rotations[ik_name][frame] = IDENTITY_QUAT.copy()
        neutral_bones = ("頭",)
        if args.finger_mode == "neutral":
            neutral_bones = neutral_bones + FINGER_BONES
        for name in neutral_bones:
            if name in name_to_idx:
                chunks.append(bone_frame(name, frame))

    if not args.no_foot_ik_tracks and effective_foot_ik_mode != "none":
        write_targets = foot_targets
        if args.foot_lock:
            write_targets = stabilize_foot_targets(
                foot_targets,
                frame_numbers,
                ground_threshold=args.foot_ground_threshold,
            )
        for frame in frame_numbers:
            for ik_name in ("左足ＩＫ", "右足ＩＫ"):
                if frame in write_targets.get(ik_name, {}) and ik_name in name_to_idx:
                    rest = pmx_bones[name_to_idx[ik_name]]["pos"]
                    quat = foot_rotations.get(ik_name, {}).get(frame, IDENTITY_QUAT.copy())
                    chunks.append(bone_frame(ik_name, frame, pos=write_targets[ik_name][frame] - rest, quat=quat))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_vmd(args.out, chunks, frame_numbers, foot_ik_display=args.enable_foot_ik or args.keep_foot_ik)
    print(f"Wrote {args.out}")
    print(f"frames={len(frame_numbers)} source_fps={1.0 / frame_time:.3f} bones={len(chunks)}")
    print(f"position_scale={position_scale:.8f} mode={position_scale_mode}")
    print(f"body_rotation={'on' if body_rotation_enabled else 'off'} mode={args.body_rotation_mode}")
    print(f"body_frame_mode={args.body_frame_mode}")
    print(f"body_rotation_transform={args.body_rotation_transform}")
    print(f"body_local_compensation={args.body_local_compensation}")
    print(f"motion_fidelity={args.motion_fidelity}")
    print(f"effective_hand_offset=({hand_outward:.3f},{hand_forward:.3f},{hand_down:.3f})")
    print(f"foot_ik_mode={effective_foot_ik_mode} requested={args.foot_ik_mode}")
    print(f"foot_rotation_mode={args.foot_rotation_mode}")
    print(f"knee_hinge={args.knee_hinge} sign={args.knee_hinge_sign}")
    print(f"leg_solver_mode={args.leg_solver_mode}")
    print(f"leg_max_angle={args.leg_max_angle}")
    print(f"pose_solver_mode={args.pose_solver_mode}")
    print(f"local_rot_feet={args.local_rot_feet}")


if __name__ == "__main__":
    main()
