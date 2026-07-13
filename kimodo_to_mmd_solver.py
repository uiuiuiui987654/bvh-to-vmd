import argparse
import json
import math
import os
import struct
from dataclasses import dataclass

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependency: numpy. Install it with: python -m pip install -r requirements.txt") from exc

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


def build_finger_local_rotation_map():
    mappings = []
    for jp, src in (("左", "Left"), ("右", "Right")):
        mappings.extend(
            (
                (f"{jp}親指０", f"{src}HandThumb1"),
                (f"{jp}親指１", f"{src}HandThumb2"),
                (f"{jp}親指２", f"{src}HandThumb3"),
            )
        )
        for jp_finger, src_finger in (
            ("人指", "Index"),
            ("中指", "Middle"),
            ("薬指", "Ring"),
            ("小指", "Pinky"),
        ):
            mappings.extend(
                (
                    (f"{jp}{jp_finger}１", f"{src}Hand{src_finger}1"),
                    (f"{jp}{jp_finger}２", f"{src}Hand{src_finger}2"),
                    (f"{jp}{jp_finger}３", f"{src}Hand{src_finger}3"),
                )
            )
    return mappings

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


def rotation_around_axis(axis, deg):
    axis = normalize(axis)
    if axis is None:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis
    r = math.radians(float(deg))
    c, s = math.cos(r), math.sin(r)
    t = 1.0 - c
    return np.array(
        (
            (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
        ),
        dtype=np.float64,
    )


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


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    out = np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dtype=np.float64,
    )
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
            joints.append(BvhJoint(" ".join(parts[1:]), parent, np.zeros(3), []))
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


def add_source_name_aliases(name_to_idx):
    aliases = {
        "Hips": ("Bip001 Pelvis", "mixamorig:Hips", "mixamorig1:Hips"),
        "HipsRotation": ("Bip001 Spine",),
        "Spine1": ("Bip001 Spine1", "Bip001 Spine", "mixamorig:Spine", "mixamorig1:Spine"),
        "Chest": ("Bip001 Spine2", "Bip001 Spine1", "mixamorig:Spine2", "mixamorig1:Spine2", "mixamorig:Spine1"),
        "Neck2": ("Bip001 Neck", "mixamorig:Neck", "mixamorig1:Neck"),
        "Head": ("Bip001 Head", "mixamorig:Head", "mixamorig1:Head"),
        "RightShoulder": ("Bip001 R Clavicle", "mixamorig:RightShoulder", "mixamorig1:RightShoulder"),
        "RightArm": ("Bip001 R UpperArm", "mixamorig:RightArm", "mixamorig1:RightArm"),
        "RightForeArm": ("Bip001 R Forearm", "mixamorig:RightForeArm", "mixamorig1:RightForeArm"),
        "RightHand": ("Bip001 R Hand", "mixamorig:RightHand", "mixamorig1:RightHand"),
        "LeftShoulder": ("Bip001 L Clavicle", "mixamorig:LeftShoulder", "mixamorig1:LeftShoulder"),
        "LeftArm": ("Bip001 L UpperArm", "mixamorig:LeftArm", "mixamorig1:LeftArm"),
        "LeftForeArm": ("Bip001 L Forearm", "mixamorig:LeftForeArm", "mixamorig1:LeftForeArm"),
        "LeftHand": ("Bip001 L Hand", "mixamorig:LeftHand", "mixamorig1:LeftHand"),
        "RightLeg": ("Bip001 R Thigh", "mixamorig:RightUpLeg", "mixamorig1:RightUpLeg"),
        "RightShin": ("Bip001 R Calf", "mixamorig:RightLeg", "mixamorig1:RightLeg"),
        "RightFoot": ("Bip001 R Foot", "mixamorig:RightFoot", "mixamorig1:RightFoot"),
        "RightToeBase": ("Bip001 R Toe0", "mixamorig:RightToeBase", "mixamorig1:RightToeBase"),
        "LeftLeg": ("Bip001 L Thigh", "mixamorig:LeftUpLeg", "mixamorig1:LeftUpLeg"),
        "LeftShin": ("Bip001 L Calf", "mixamorig:LeftLeg", "mixamorig1:LeftLeg"),
        "LeftFoot": ("Bip001 L Foot", "mixamorig:LeftFoot", "mixamorig1:LeftFoot"),
        "LeftToeBase": ("Bip001 L Toe0", "mixamorig:LeftToeBase", "mixamorig1:LeftToeBase"),
        "RightHandThumb1": ("Bip001 R Finger0",),
        "RightHandThumb2": ("Bip001 R Finger01",),
        "RightHandThumb3": ("Bip001 R Finger02",),
        "RightHandIndex1": ("Bip001 R Finger1",),
        "RightHandIndex2": ("Bip001 R Finger11",),
        "RightHandIndex3": ("Bip001 R Finger12",),
        "RightHandMiddle1": ("Bip001 R Finger2",),
        "RightHandMiddle2": ("Bip001 R Finger21",),
        "RightHandMiddle3": ("Bip001 R Finger22",),
        "RightHandRing1": ("Bip001 R Finger3",),
        "RightHandRing2": ("Bip001 R Finger31",),
        "RightHandRing3": ("Bip001 R Finger32",),
        "RightHandPinky1": ("Bip001 R Finger4",),
        "RightHandPinky2": ("Bip001 R Finger41",),
        "RightHandPinky3": ("Bip001 R Finger42",),
        "LeftHandThumb1": ("Bip001 L Finger0",),
        "LeftHandThumb2": ("Bip001 L Finger01",),
        "LeftHandThumb3": ("Bip001 L Finger02",),
        "LeftHandIndex1": ("Bip001 L Finger1",),
        "LeftHandIndex2": ("Bip001 L Finger11",),
        "LeftHandIndex3": ("Bip001 L Finger12",),
        "LeftHandMiddle1": ("Bip001 L Finger2",),
        "LeftHandMiddle2": ("Bip001 L Finger21",),
        "LeftHandMiddle3": ("Bip001 L Finger22",),
        "LeftHandRing1": ("Bip001 L Finger3",),
        "LeftHandRing2": ("Bip001 L Finger31",),
        "LeftHandRing3": ("Bip001 L Finger32",),
        "LeftHandPinky1": ("Bip001 L Finger4",),
        "LeftHandPinky2": ("Bip001 L Finger41",),
        "LeftHandPinky3": ("Bip001 L Finger42",),
    }
    for target, candidates in aliases.items():
        if target in name_to_idx:
            continue
        for candidate in candidates:
            if candidate in name_to_idx:
                name_to_idx[target] = name_to_idx[candidate]
                break


def bvh_global_positions(joints, motion, root_position_indices=None):
    root_position_indices = set(root_position_indices or ())
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
                    if joint.parent < 0 or i in root_position_indices:
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


def bvh_rest_positions(joints):
    """Reconstruct the BVH bind pose from hierarchy offsets only."""
    positions = np.zeros((len(joints), 3), dtype=np.float64)
    for index, joint in enumerate(joints):
        if joint.parent < 0:
            positions[index] = joint.offset
        else:
            positions[index] = positions[joint.parent] + joint.offset
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


def source_uses_joint_positions(joints, hips_idx):
    for i, joint in enumerate(joints):
        if i == hips_idx or joint.parent < 0:
            continue
        if any(channel.endswith("position") for channel in joint.channels):
            return True
    return False


def detect_source_up(global_positions, src_name_to_idx):
    hips_idx = src_name_to_idx.get("Hips")
    head_idx = src_name_to_idx.get("Head")
    if hips_idx is None or head_idx is None:
        return "y"
    vector = global_positions[0, head_idx] - global_positions[0, hips_idx]
    return "z" if abs(float(vector[2])) > abs(float(vector[1])) * 1.4 else "y"


def resolve_center_index(src_positions, src_name_to_idx, mode):
    if mode != "auto":
        explicit = {
            "hips": "Hips",
            "spine": "Spine1",
            "chest": "Chest",
            "head": "Head",
        }[mode]
        return src_name_to_idx.get(explicit, src_name_to_idx.get("Hips"))
    hips_idx = src_name_to_idx.get("Hips")
    chest_idx = src_name_to_idx.get("Chest")
    spine_idx = src_name_to_idx.get("Spine1")
    if hips_idx is None:
        return chest_idx if chest_idx is not None else spine_idx
    hips_range = float(np.linalg.norm(np.ptp(src_positions[:, hips_idx, [0, 2]], axis=0)))
    candidates = []
    for name, idx in (("Chest", chest_idx), ("Spine1", spine_idx)):
        if idx is not None:
            horizontal = float(np.linalg.norm(np.ptp(src_positions[:, idx, [0, 2]], axis=0)))
            candidates.append((horizontal, idx, name))
    if hips_range < 1e-4 and candidates:
        candidates.sort(reverse=True)
        if candidates[0][0] > 0.5:
            return candidates[0][1]
    return hips_idx


def load_object_motion(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    if not frames:
        return None
    max_frame = max(int(frame["frame"]) for frame in frames)
    loc = np.zeros((max_frame + 1, 3), dtype=np.float64)
    rot = np.zeros((max_frame + 1, 3, 3), dtype=np.float64)
    scale = np.ones((max_frame + 1, 3), dtype=np.float64)
    for frame in frames:
        idx = int(frame["frame"])
        loc[idx] = np.asarray(frame["location"], dtype=np.float64)
        rot[idx] = np.asarray(frame["rotation"], dtype=np.float64)
        scale[idx] = np.asarray(frame.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    return {"location": loc, "rotation": rot, "scale": scale}


def object_motion_at(object_motion, frame, source_up, origin_mode="delta"):
    if object_motion is None:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    loc = object_motion["location"]
    rot = object_motion["rotation"]
    scale = object_motion["scale"]
    idx = min(int(frame), loc.shape[0] - 1)
    base_scale = np.maximum(np.abs(scale[0]), 1e-8)
    loc_delta = loc[idx] - loc[0]
    if origin_mode in {"horizontal", "absolute"}:
        loc_delta[:2] = loc[idx, :2]
    if origin_mode == "absolute":
        loc_delta[2] = loc[idx, 2]
    loc_delta = loc_delta / base_scale
    # Object motion is sampled from Blender's world matrix, not from the BVH
    # coordinate system. mmd_tools maps MMD (X, Y, Z) to Blender (X, Z, Y).
    loc_mmd = blender_world_position_to_mmd(loc_delta.reshape(1, 3))[0]
    rot_delta = rot[idx] @ rot[0].T
    rot_mmd = rotation_to_mmd_space(rot_delta, source_up)
    return loc_mmd, rot_mmd


def blender_world_position_to_mmd(points):
    points = np.asarray(points, dtype=np.float64)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def to_mmd_space(points, source_up="y"):
    points = np.asarray(points, dtype=np.float64)
    if source_up == "z":
        out = np.empty_like(points)
        out[..., 0] = points[..., 1]
        out[..., 1] = points[..., 2]
        out[..., 2] = -points[..., 0]
        return out
    out = points.copy()
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


def source_to_mmd_basis(source_up):
    if source_up == "z":
        return np.array(
            (
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
    return np.diag((1.0, 1.0, -1.0))


def rotation_to_mmd_space(rot, source_up="y"):
    basis = source_to_mmd_basis(source_up)
    return basis @ rot @ basis.T


def yaw_only_rotation(rot):
    forward = rot @ np.array((0.0, 0.0, 1.0), dtype=np.float64)
    horizontal = np.array((forward[0], forward[2]), dtype=np.float64)
    length = float(np.linalg.norm(horizontal))
    if length < 1e-8:
        return np.eye(3, dtype=np.float64)
    yaw = math.atan2(horizontal[0], horizontal[1])
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        (
            (c, 0.0, s),
            (0.0, 1.0, 0.0),
            (-s, 0.0, c),
        ),
        dtype=np.float64,
    )


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


def vmd_position_space(pos, root_rotation, root_rotation_bone):
    if root_rotation_bone == "parent":
        return root_rotation.T @ pos
    return pos.copy()


def world_position_space(pos, root_rotation, root_rotation_bone):
    if root_rotation_bone == "parent":
        return root_rotation @ pos
    return pos.copy()


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
    foot_ik_names = ("左足ＩＫ", "右足ＩＫ", "左つま先ＩＫ", "右つま先ＩＫ")
    displays = [
        model_display_frame(frame, foot_ik_names, enabled=foot_ik_display)
        for frame in frame_numbers
    ]
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
    base = frames[0] if frames[0] is not None else valid[0]
    yaw_values = []
    for mat in valid:
        rel = mat @ base.T
        yaw_values.append(math.degrees(math.atan2(float(rel[0, 2]), float(rel[2, 2]))))
    yaw_range = float(max(yaw_values) - min(yaw_values)) if yaw_values else 0.0
    should_apply = mode == "infer" or float(up_y.min()) < -0.35 or yaw_range > 12.0
    if not should_apply:
        return rotations, False
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


def infer_hips_rotations(src_rotations, src_name_to_idx, frame_numbers, mode, source_up="y"):
    rotations = {frame: np.eye(3, dtype=np.float64) for frame in frame_numbers}
    if mode == "none" or "Hips" not in src_name_to_idx:
        return rotations, False
    hips_idx = src_name_to_idx["Hips"]
    base = rotation_to_mmd_space(src_rotations[frame_numbers[0], hips_idx], source_up)
    base_inv = base.T
    rels = []
    for frame in frame_numbers:
        rel = rotation_to_mmd_space(src_rotations[frame, hips_idx], source_up) @ base_inv
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


def retarget_rotation_source(mmd_name, src_name_to_idx, default_name):
    if mmd_name == "下半身" and "HipsRotation" in src_name_to_idx:
        return "HipsRotation"
    return default_name


def solve_frame_from_local_rotations(
    src_local_rotations,
    frame,
    src_name_to_idx,
    strengths,
    include_feet=True,
    source_up="y",
):
    source_for_mmd = {
        "下半身": "Hips",
        "上半身": "Spine1",
        "上半身2": "Chest",
        "首": "Neck2",
        "頭": "Head",
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
        base = rotation_to_mmd_space(src_local_rotations[0, idx], source_up)
        cur = rotation_to_mmd_space(src_local_rotations[frame, idx], source_up)
        rel = cur @ base.T
        quat = mat_to_quat(rel)
        strength = strengths.get(mmd_name, 1.0)
        if strength < 1.0:
            quat = quat_slerp_identity(quat, strength)
        solved[mmd_name] = quat
    return solved


def solve_fingers_from_local_rotations(
    src_local_rotations,
    frame,
    src_name_to_idx,
    strength=1.0,
    max_angle=45.0,
    source_up="y",
):
    solved = {}
    for mmd_name, src_name in build_finger_local_rotation_map():
        if src_name not in src_name_to_idx:
            continue
        src_name = retarget_rotation_source(mmd_name, src_name_to_idx, src_name)
        idx = src_name_to_idx[src_name]
        base = rotation_to_mmd_space(src_local_rotations[0, idx], source_up)
        cur = rotation_to_mmd_space(src_local_rotations[frame, idx], source_up)
        rel = cur @ base.T
        quat = mat_to_quat(rel)
        if strength < 1.0:
            quat = quat_slerp_identity(quat, strength)
        if max_angle > 0.0:
            quat = quat_clamp_angle(quat, max_angle)
        solved[mmd_name] = quat
    return solved


def solve_thumbs_from_positions(pmx_bones, name_to_idx, src, src_name_to_idx, base_solved):
    local = {
        name: quat_to_mat(quat)
        for name, quat in base_solved.items()
        if name in name_to_idx
    }
    thumb_names = tuple(
        f"{side}親指{number}"
        for side in ("左", "右")
        for number in "０１２"
    )
    for name in thumb_names:
        if name in name_to_idx:
            local[name] = np.eye(3, dtype=np.float64)

    global_rots = [np.eye(3, dtype=np.float64) for _ in pmx_bones]

    def refresh_globals():
        for index, bone in enumerate(pmx_bones):
            parent = bone["parent"]
            parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
            global_rots[index] = parent_global @ local.get(bone["name"], np.eye(3, dtype=np.float64))

    def target_rest_frame(side):
        names = (f"{side}手首", f"{side}人指１", f"{side}中指１", f"{side}小指１")
        return hand_frame_from_lookup(lambda name: pmx_bones[name_to_idx[name]]["pos"], *names)

    refresh_globals()
    for side, source_side in (("左", "Left"), ("右", "Right")):
        wrist_name = f"{side}手首"
        source_hand_names = (
            f"{source_side}Hand",
            f"{source_side}HandIndex1",
            f"{source_side}HandMiddle1",
            f"{source_side}HandPinky1",
        )
        required_target = (wrist_name, f"{side}親指０", f"{side}親指１", f"{side}親指２")
        required_source = source_hand_names + (
            f"{source_side}HandThumb1",
            f"{source_side}HandThumb2",
            f"{source_side}HandThumb3",
        )
        if any(name not in name_to_idx for name in required_target):
            continue
        if any(name not in src_name_to_idx for name in required_source):
            continue
        rest_frame = target_rest_frame(side)
        source_frame = hand_frame_from_lookup(
            lambda name: src[src_name_to_idx[name]],
            *source_hand_names,
        )
        if rest_frame is None or source_frame is None:
            continue
        wrist_global = global_rots[name_to_idx[wrist_name]]
        direction_map = (wrist_global @ rest_frame) @ source_frame.T
        chain = (
            (f"{side}親指０", f"{side}親指１", f"{source_side}HandThumb1", f"{source_side}HandThumb2"),
            (f"{side}親指１", f"{side}親指２", f"{source_side}HandThumb2", f"{source_side}HandThumb3"),
        )
        for bone_name, child_name, source_a, source_b in chain:
            bone_index = name_to_idx[bone_name]
            child_index = name_to_idx[child_name]
            rest_direction = pmx_bones[child_index]["pos"] - pmx_bones[bone_index]["pos"]
            desired_direction = direction_map @ (
                src[src_name_to_idx[source_b]] - src[src_name_to_idx[source_a]]
            )
            current_direction = global_rots[bone_index] @ rest_direction
            wanted_global = rotation_between(current_direction, desired_direction) @ global_rots[bone_index]
            parent = pmx_bones[bone_index]["parent"]
            parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
            local[bone_name] = parent_global.T @ wanted_global
            refresh_globals()

    return {
        name: mat_to_quat(local[name])
        for name in thumb_names
        if name in local
    }


def solve_frame_from_global_rotations(
    pmx_bones,
    name_to_idx,
    src_global_rotations,
    frame,
    src_name_to_idx,
    strengths,
    center_rotation=None,
    include_feet=False,
    source_up="y",
):
    source_for_mmd = {
        "下半身": "Hips",
        "上半身": "Spine1",
        "上半身2": "Chest",
        "首": "Neck2",
        "頭": "Head",
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
        src_name = retarget_rotation_source(mmd_name, src_name_to_idx, source_for_mmd[mmd_name])
        if src_name not in src_name_to_idx or mmd_name not in name_to_idx:
            continue
        src_idx = src_name_to_idx[src_name]
        base = rotation_to_mmd_space(src_global_rotations[0, src_idx], source_up)
        cur = rotation_to_mmd_space(src_global_rotations[frame, src_idx], source_up)
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


def solve_legs_from_positions(
    pmx_bones,
    name_to_idx,
    src,
    src_name_to_idx,
    leg_solver_mode="ccd",
    leg_max_angle=0.0,
    base_solved=None,
    pole_state=None,
    ankle_rotation_mode="neutral",
    ankle_max_angle=35.0,
    source_base=None,
):
    leg_bones = {"左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"}
    base_leg_local = {
        name: quat_to_mat(quat)
        for name, quat in (base_solved or {}).items()
        if name in leg_bones
    }
    local = {
        name: quat_to_mat(quat)
        for name, quat in (base_solved or {}).items()
        if name in name_to_idx and name not in leg_bones
    }
    pole_state = pole_state if pole_state is not None else {}
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
        idx = name_to_idx[name]
        if child_name not in name_to_idx:
            return
        rest_dir = pmx_bones[name_to_idx[child_name]]["pos"] - pmx_bones[idx]["pos"]
        desired_dir = src[src_name_to_idx[src_b]] - src[src_name_to_idx[src_a]]
        set_global_rotation(name, rotation_between(rest_dir, desired_dir))

    def ccd_limb(upper_name, mid_name, end_name, src_upper, src_mid, src_end, iterations=6):
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

    def solve_pole_leg(side, upper_name, mid_name, end_name, src_upper, src_mid, src_end):
        required = (upper_name, mid_name, end_name)
        if any(name not in name_to_idx for name in required):
            return
        if any(name not in src_name_to_idx for name in (src_upper, src_mid, src_end)):
            return

        upper_idx = name_to_idx[upper_name]
        mid_idx = name_to_idx[mid_name]
        end_idx = name_to_idx[end_name]
        rest_upper = pmx_bones[mid_idx]["pos"] - pmx_bones[upper_idx]["pos"]
        rest_lower = pmx_bones[end_idx]["pos"] - pmx_bones[mid_idx]["pos"]
        upper_len = float(np.linalg.norm(rest_upper))
        lower_len = float(np.linalg.norm(rest_lower))
        if upper_len < 1e-8 or lower_len < 1e-8:
            return

        src_upper_pos = src[src_name_to_idx[src_upper]]
        src_mid_pos = src[src_name_to_idx[src_mid]]
        src_end_pos = src[src_name_to_idx[src_end]]
        src_upper_vec = src_mid_pos - src_upper_pos
        src_lower_vec = src_end_pos - src_mid_pos
        src_upper_dir = normalize(src_upper_vec)
        src_lower_dir = normalize(src_lower_vec)
        chain_dir = normalize(src_end_pos - src_upper_pos)
        if src_upper_dir is None or src_lower_dir is None or chain_dir is None:
            return

        # Keep the source knee angle while adapting both segment lengths to the
        # PMX. A tiny bend prevents a straight leg from losing its pole plane.
        bend_cos = max(
            -0.9994,
            min(math.cos(math.radians(1.5)), float(np.dot(src_upper_dir, src_lower_dir))),
        )
        target_dist = math.sqrt(
            max(
                upper_len * upper_len
                + lower_len * lower_len
                + 2.0 * upper_len * lower_len * bend_cos,
                1e-8,
            )
        )
        target_dist = min(target_dist, (upper_len + lower_len) * 0.9999)
        target_dist = max(target_dist, abs(upper_len - lower_len) + 1e-5)

        raw_pole = src_mid_pos - (
            src_upper_pos + chain_dir * float(np.dot(src_mid_pos - src_upper_pos, chain_dir))
        )
        source_total_len = float(np.linalg.norm(src_upper_vec) + np.linalg.norm(src_lower_vec))
        source_pole = (
            normalize(raw_pole)
            if float(np.linalg.norm(raw_pole)) > source_total_len * 0.01
            else None
        )

        parent = pmx_bones[upper_idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        rest_chain_dir = normalize(rest_upper + rest_lower)
        rest_hinge = normalize(np.cross(rest_upper, rest_lower))
        rest_pole = None
        if rest_chain_dir is not None:
            rest_pole = rest_upper - rest_chain_dir * float(np.dot(rest_upper, rest_chain_dir))
            rest_pole = normalize(parent_global @ rest_pole)

        expected_hinge = None
        if rest_hinge is not None:
            expected_upper = parent_global @ base_leg_local.get(upper_name, np.eye(3, dtype=np.float64))
            expected_hinge = normalize(expected_upper @ rest_hinge)
        expected_pole = (
            normalize(np.cross(chain_dir, expected_hinge))
            if expected_hinge is not None
            else None
        )

        previous_pole = pole_state.get(side)
        if previous_pole is not None:
            previous_pole = previous_pole - chain_dir * float(np.dot(previous_pole, chain_dir))
            previous_pole = normalize(previous_pole)
        # The source knee position becomes ambiguous whenever the leg crosses a
        # straight pose. The source thigh rotation still carries a stable roll
        # axis, so it is the authoritative pole direction for MMD's hinge knee.
        pole = expected_pole
        if pole is None:
            pole = source_pole
        if pole is None:
            pole = previous_pole if previous_pole is not None else rest_pole
        if pole is None:
            fallback = np.array((0.0, 0.0, -1.0), dtype=np.float64)
            pole = fallback - chain_dir * float(np.dot(fallback, chain_dir))
            pole = normalize(pole)
        if pole is None:
            return
        pole_state[side] = pole.copy()

        along = (
            upper_len * upper_len - lower_len * lower_len + target_dist * target_dist
        ) / (2.0 * target_dist)
        height = math.sqrt(max(upper_len * upper_len - along * along, 0.0))
        target_upper_pos = global_pos[upper_idx].copy()
        target_end_pos = target_upper_pos + chain_dir * target_dist
        target_mid_pos = target_upper_pos + chain_dir * along + pole * height
        target_upper_dir = normalize(target_mid_pos - target_upper_pos)
        target_lower_dir = normalize(target_end_pos - target_mid_pos)
        if target_upper_dir is None or target_lower_dir is None:
            return

        target_hinge = normalize(np.cross(target_upper_dir, target_lower_dir))
        rest_frame = (
            make_frame_from_primary_lateral(rest_upper, rest_hinge)
            if rest_hinge is not None
            else None
        )
        target_frame = (
            make_frame_from_primary_lateral(target_upper_dir, target_hinge)
            if target_hinge is not None
            else None
        )
        if rest_frame is not None and target_frame is not None:
            upper_global = target_frame @ rest_frame.T
        else:
            upper_global = rotation_between(rest_upper, target_upper_dir)
        set_global_rotation(upper_name, upper_global)

        current_lower_dir = normalize(global_rots[upper_idx] @ rest_lower)
        if current_lower_dir is None:
            return
        knee_delta = rotation_between(current_lower_dir, target_lower_dir)
        set_global_rotation(mid_name, knee_delta @ global_rots[upper_idx])

    def solve_ankle_pitch(
        upper_name,
        mid_name,
        end_name,
        toe_name,
        src_end,
        src_toe,
    ):
        required = (upper_name, mid_name, end_name, toe_name)
        if any(name not in name_to_idx for name in required):
            return
        if src_end not in src_name_to_idx or src_toe not in src_name_to_idx:
            return

        upper_idx = name_to_idx[upper_name]
        mid_idx = name_to_idx[mid_name]
        end_idx = name_to_idx[end_name]
        toe_idx = name_to_idx[toe_name]
        rest_upper = pmx_bones[mid_idx]["pos"] - pmx_bones[upper_idx]["pos"]
        rest_lower = pmx_bones[end_idx]["pos"] - pmx_bones[mid_idx]["pos"]
        rest_foot = pmx_bones[toe_idx]["pos"] - pmx_bones[end_idx]["pos"]
        hinge_local = normalize(np.cross(rest_upper, rest_lower))
        if hinge_local is None:
            hinge_local = np.array((-1.0, 0.0, 0.0), dtype=np.float64)

        parent_global = global_rots[mid_idx]
        axis_global = normalize(parent_global @ hinge_local)
        current_dir = normalize(parent_global @ rest_foot)
        desired_dir = normalize(src[src_name_to_idx[src_toe]] - src[src_name_to_idx[src_end]])
        if axis_global is None or current_dir is None or desired_dir is None:
            return
        current_plane = current_dir - axis_global * float(np.dot(current_dir, axis_global))
        desired_plane = desired_dir - axis_global * float(np.dot(desired_dir, axis_global))
        current_plane = normalize(current_plane)
        desired_plane = normalize(desired_plane)
        if current_plane is None or desired_plane is None:
            return

        angle = math.degrees(
            math.atan2(
                float(np.dot(axis_global, np.cross(current_plane, desired_plane))),
                max(-1.0, min(1.0, float(np.dot(current_plane, desired_plane)))),
            )
        )
        angle = max(-float(ankle_max_angle), min(float(ankle_max_angle), angle))
        local[end_name] = rotation_around_axis(hinge_local, angle)
        refresh_globals()

    def solve_ankle_relative_flex(side, mid_name, end_name, toe_name, src_mid, src_end, src_toe):
        required = (mid_name, end_name, toe_name, "左足", "右足")
        if any(name not in name_to_idx for name in required):
            return
        if source_base is None or any(
            name not in src_name_to_idx
            for name in (src_mid, src_end, src_toe, "LeftLeg", "RightLeg")
        ):
            return

        def joint_angle(points, mid, end, toe):
            down = normalize(points[src_name_to_idx[end]] - points[src_name_to_idx[mid]])
            forward = normalize(points[src_name_to_idx[toe]] - points[src_name_to_idx[end]])
            if down is None or forward is None:
                return None
            return math.acos(max(-1.0, min(1.0, float(np.dot(down, forward)))))

        source_base_angle = joint_angle(source_base, src_mid, src_end, src_toe)
        source_angle = joint_angle(src, src_mid, src_end, src_toe)
        if source_base_angle is None or source_angle is None:
            return

        # VMD ankle-axis calibration on the actual PMX deformation chain
        # shows that foot pitch is the standard 足首 X channel.  Joint
        # positions do not contain enough information to recover twist around
        # the ankle-to-toe axis; trying to synthesize that missing roll from
        # the pelvis line is what made 足首D point toward the toe.  Preserve the
        # reliable flex component and let the leg/root chains carry heading.
        flex_degrees = math.degrees(source_angle - source_base_angle)
        flex_degrees = max(
            -float(ankle_max_angle), min(float(ankle_max_angle), flex_degrees)
        )
        local[end_name] = rotation_around_axis(
            np.array((1.0, 0.0, 0.0), dtype=np.float64), flex_degrees
        )
        refresh_globals()

    def solve_ankle_direction(end_name, toe_name, src_end, src_toe):
        required = (end_name, toe_name)
        if any(name not in name_to_idx for name in required):
            return
        if src_end not in src_name_to_idx or src_toe not in src_name_to_idx:
            return

        end_idx = name_to_idx[end_name]
        toe_idx = name_to_idx[toe_name]
        parent = pmx_bones[end_idx]["parent"]
        parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[parent]
        rest_foot = pmx_bones[toe_idx]["pos"] - pmx_bones[end_idx]["pos"]
        current_dir = normalize(parent_global @ rest_foot)
        desired_dir = normalize(src[src_name_to_idx[src_toe]] - src[src_name_to_idx[src_end]])
        if current_dir is None or desired_dir is None:
            return

        delta_global = rotation_between(current_dir, desired_dir)
        local_rotation = parent_global.T @ delta_global @ parent_global
        quat = mat_to_quat(local_rotation)
        if ankle_max_angle > 0.0:
            quat = quat_clamp_angle(quat, ankle_max_angle)
        local[end_name] = quat_to_mat(quat)
        refresh_globals()

    def solve_ankle_frame(mid_name, end_name, toe_name, src_mid, src_end, src_toe):
        required = (mid_name, end_name, toe_name)
        if any(name not in name_to_idx for name in required):
            return
        if any(name not in src_name_to_idx for name in (src_mid, src_end, src_toe)):
            return

        mid_idx = name_to_idx[mid_name]
        end_idx = name_to_idx[end_name]
        toe_idx = name_to_idx[toe_name]
        rest_shin = pmx_bones[end_idx]["pos"] - pmx_bones[mid_idx]["pos"]
        rest_forward = pmx_bones[toe_idx]["pos"] - pmx_bones[end_idx]["pos"]
        rest_lateral = np.cross(rest_shin, rest_forward)
        rest_frame = make_frame_from_primary_lateral(rest_forward, rest_lateral)

        source_shin = src[src_name_to_idx[src_end]] - src[src_name_to_idx[src_mid]]
        source_forward = src[src_name_to_idx[src_toe]] - src[src_name_to_idx[src_end]]
        source_lateral = np.cross(source_shin, source_forward)
        source_frame = make_frame_from_primary_lateral(source_forward, source_lateral)
        if rest_frame is None or source_frame is None:
            solve_ankle_direction(end_name, toe_name, src_end, src_toe)
            return

        parent_global = global_rots[mid_idx]
        wanted_global = source_frame @ rest_frame.T
        local_rotation = parent_global.T @ wanted_global
        quat = mat_to_quat(local_rotation)
        if ankle_max_angle > 0.0:
            quat = quat_clamp_angle(quat, ankle_max_angle)
        local[end_name] = quat_to_mat(quat)
        refresh_globals()

    refresh_globals()
    if leg_solver_mode == "pole":
        solve_pole_leg("Left", "左足", "左ひざ", "左足首", "LeftLeg", "LeftShin", "LeftFoot")
        solve_pole_leg("Right", "右足", "右ひざ", "右足首", "RightLeg", "RightShin", "RightFoot")
    else:
        for name, src_pair in (
            ("左足", ("LeftLeg", "LeftShin")),
            ("左ひざ", ("LeftShin", "LeftFoot")),
            ("右足", ("RightLeg", "RightShin")),
            ("右ひざ", ("RightShin", "RightFoot")),
        ):
            apply_direction(name, src_pair[0], src_pair[1])

    if leg_solver_mode == "ccd":
        ccd_limb("左足", "左ひざ", "左足首", "LeftLeg", "LeftShin", "LeftFoot")
        ccd_limb("右足", "右ひざ", "右足首", "RightLeg", "RightShin", "RightFoot")
    elif leg_solver_mode == "frozen":
        for leg_name in ("左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"):
            local[leg_name] = np.eye(3, dtype=np.float64)
        refresh_globals()

    if ankle_rotation_mode != "neutral" and leg_solver_mode != "frozen":
        if ankle_rotation_mode == "relative-flex":
            solve_ankle_relative_flex(
                "Left", "左ひざ", "左足首", "左つま先", "LeftShin", "LeftFoot", "LeftToeBase"
            )
            solve_ankle_relative_flex(
                "Right", "右ひざ", "右足首", "右つま先", "RightShin", "RightFoot", "RightToeBase"
            )
        elif ankle_rotation_mode == "pitch":
            solve_ankle_pitch(
                "左足", "左ひざ", "左足首", "左つま先", "LeftFoot", "LeftToeBase"
            )
            solve_ankle_pitch(
                "右足", "右ひざ", "右足首", "右つま先", "RightFoot", "RightToeBase"
            )
        elif ankle_rotation_mode == "direction":
            solve_ankle_direction("左足首", "左つま先", "LeftFoot", "LeftToeBase")
            solve_ankle_direction("右足首", "右つま先", "RightFoot", "RightToeBase")
        else:
            solve_ankle_frame(
                "左ひざ", "左足首", "左つま先", "LeftShin", "LeftFoot", "LeftToeBase"
            )
            solve_ankle_frame(
                "右ひざ", "右足首", "右つま先", "RightShin", "RightFoot", "RightToeBase"
            )
    else:
        for foot_name in ("左足首", "右足首"):
            if foot_name in name_to_idx:
                local[foot_name] = np.eye(3, dtype=np.float64)
                refresh_globals()

    solved = {name: mat_to_quat(local[name]) for name in leg_bones if name in local}
    if leg_max_angle > 0.0:
        for name in ("左足", "左ひざ", "右足", "右ひざ"):
            if name in solved:
                solved[name] = quat_clamp_angle(solved[name], leg_max_angle)
    return solved


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
        idx = name_to_idx[name]
        if child_name in name_to_idx:
            rest_dir = pmx_bones[name_to_idx[child_name]]["pos"] - pmx_bones[idx]["pos"]
        elif name == "頭" and pmx_bones[idx]["parent"] >= 0:
            rest_dir = pmx_bones[idx]["pos"] - pmx_bones[pmx_bones[idx]["parent"]]["pos"]
        else:
            return
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
        ("頭", ("Head", "HeadEnd")),
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
    parser.add_argument("--center-source", choices=("auto", "hips", "spine", "chest", "head"), default="auto")
    parser.add_argument("--source-up", choices=("auto", "y", "z"), default="auto")
    parser.add_argument("--object-motion-json", default="")
    parser.add_argument("--object-motion-strength", type=float, default=1.0)
    parser.add_argument(
        "--object-origin-mode",
        choices=("delta", "horizontal", "absolute"),
        default="delta",
        help="Discard the FBX object origin, preserve its horizontal placement, or preserve its complete world placement.",
    )
    parser.add_argument("--object-rotation-strength", type=float, default=1.0)
    parser.add_argument(
        "--object-root-rotation",
        choices=("auto", "object", "full", "off"),
        default="auto",
        help="When object motion is present, choose what drives the VMD root turn. auto/object uses the FBX object transform; full also includes inferred body tilt.",
    )
    parser.add_argument(
        "--object-root-axis",
        choices=("yaw", "full"),
        default="yaw",
        help="Use only horizontal yaw for FBX object root rotation, or preserve the full object rotation.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--arm-strength", type=float, default=1.0)
    parser.add_argument("--wrist-strength", type=float, default=0.45)
    parser.add_argument("--hand-outward", type=float, default=0.9)
    parser.add_argument("--hand-forward", type=float, default=-0.25)
    parser.add_argument("--hand-down", type=float, default=2.0)
    parser.add_argument("--torso-strength", type=float, default=0.7)
    parser.add_argument("--neck-strength", type=float, default=0.45)
    parser.add_argument("--head-strength", type=float, default=0.35)
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
    parser.add_argument("--leg-solver-mode", choices=("ccd", "direction", "pole", "frozen"), default="ccd")
    parser.add_argument("--leg-max-angle", type=float, default=0.0)
    parser.add_argument(
        "--ankle-rotation-mode",
        choices=("neutral", "relative-flex", "pitch", "direction", "frame"),
        default="neutral",
        help="Keep the ankle neutral, use hinge pitch, align only the toe direction, or align the complete foot frame.",
    )
    parser.add_argument("--ankle-max-angle", type=float, default=35.0)
    parser.add_argument("--knee-hinge", choices=("off", "flip", "always"), default="flip")
    parser.add_argument("--knee-hinge-sign", type=float, default=1.0)
    parser.add_argument(
        "--pose-solver-mode",
        choices=("position", "local-rot", "global-rot", "hybrid-leg-position"),
        default="position",
    )
    parser.add_argument("--local-rot-feet", choices=("omit", "include"), default="omit")
    parser.add_argument("--foot-ik-mode", choices=("auto", "source", "fk", "none"), default="auto")
    parser.add_argument(
        "--foot-rotation-mode",
        choices=("none", "follow-body"),
        default="follow-body",
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
        "--rotation-compensation",
        choices=("auto", "on", "off"),
        default="auto",
        help="Subtract whole-body root rotation from global-rot/local solving. FBX may need off to bake waist turns into bones.",
    )
    parser.add_argument(
        "--root-rotation-bone",
        choices=("center", "parent", "lower", "none"),
        default="center",
        help="Where to write inferred whole-body turning: center, parent/all-parent, lower body, or none.",
    )
    parser.add_argument(
        "--finger-mode",
        choices=("omit", "neutral", "source"),
        default="omit",
        help="omit by default, write neutral finger keys, or convert source BVH finger directions.",
    )
    parser.add_argument("--finger-strength", type=float, default=0.75)
    parser.add_argument("--finger-max-angle", type=float, default=45.0)
    parser.add_argument("--thumb-mode", choices=("generic", "direction", "neutral"), default="generic")
    args = parser.parse_args()

    joints, motion, frame_time = parse_bvh(args.bvh)
    src_names = [joint.name for joint in joints]
    src_name_to_idx = {name: i for i, name in enumerate(src_names)}
    add_source_name_aliases(src_name_to_idx)
    hips_idx = src_name_to_idx.get("Hips")
    root_position_indices = {hips_idx} if hips_idx is not None and source_uses_joint_positions(joints, hips_idx) else set()
    source_global_positions = bvh_global_positions(joints, motion, root_position_indices=root_position_indices)
    source_up = detect_source_up(source_global_positions, src_name_to_idx) if args.source_up == "auto" else args.source_up
    src_positions = to_mmd_space(source_global_positions, source_up)
    source_rest_positions = to_mmd_space(bvh_rest_positions(joints), source_up)
    src_rotations = bvh_global_rotations(joints, motion)
    src_local_rotations = bvh_local_rotations(joints, motion)
    object_motion = load_object_motion(args.object_motion_json)
    pmx_bones = load_pmx_bones(args.pmx)
    name_to_idx = {bone["name"]: i for i, bone in enumerate(pmx_bones)}
    position_scale, position_scale_mode = resolve_position_scale(args.position_scale, src_positions, pmx_bones)

    strengths = {}
    for name, _pair in build_mappings():
        if name == "首":
            strengths[name] = args.neck_strength
        elif name == "頭":
            strengths[name] = args.head_strength
        elif name in {"下半身", "上半身", "上半身2"}:
            strengths[name] = args.torso_strength
        elif name in {"左手首", "右手首"}:
            strengths[name] = args.wrist_strength
        elif name in {"左足首", "右足首"}:
            strengths[name] = args.foot_strength
        elif name.endswith("足") or "ひざ" in name or "足首" in name:
            strengths[name] = args.leg_strength
        else:
            strengths[name] = args.arm_strength

    if hips_idx is None:
        raise ValueError("BVH is missing Hips")
    center_idx = resolve_center_index(src_positions, src_name_to_idx, args.center_source)
    if center_idx is None:
        center_idx = hips_idx
    center_source_name = src_names[center_idx]
    center0 = src_positions[0, center_idx].copy()
    frame_numbers = list(range(0, src_positions.shape[0], max(1, args.stride)))
    if args.body_frame_mode == "hips":
        body_rotations, body_rotation_enabled = infer_hips_rotations(
            src_rotations,
            src_name_to_idx,
            frame_numbers,
            args.body_rotation_mode,
            source_up=source_up,
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
    use_rotation_compensation = body_rotation_enabled and (
        args.rotation_compensation == "on"
        or (args.rotation_compensation == "auto" and args.root_rotation_bone in {"center", "parent"})
    )
    effective_foot_ik_mode = args.foot_ik_mode
    if effective_foot_ik_mode == "auto":
        effective_foot_ik_mode = "source"
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
    leg_pole_state = {}
    for frame in frame_numbers:
        center_world = (src_positions[frame, center_idx] - center0) * position_scale
        object_center, object_rotation = object_motion_at(
            object_motion,
            frame,
            source_up,
            origin_mode=args.object_origin_mode,
        )
        center_world = center_world + object_center * position_scale * args.object_motion_strength
        if args.object_rotation_strength < 1.0:
            object_rotation_quat = quat_slerp_identity(mat_to_quat(object_rotation), args.object_rotation_strength)
            object_rotation = quat_to_mat(object_rotation_quat)
        object_root_rotation = yaw_only_rotation(object_rotation) if args.object_root_axis == "yaw" else object_rotation
        body_rotation = object_rotation @ body_rotations.get(frame, np.eye(3, dtype=np.float64))
        if object_motion is not None and args.object_root_rotation in {"auto", "object"}:
            root_rotation = object_root_rotation
        elif object_motion is not None and args.object_root_rotation == "off":
            root_rotation = np.eye(3, dtype=np.float64)
        else:
            root_rotation = body_rotation
        root_quat = mat_to_quat(root_rotation)
        center_quat = root_quat if args.root_rotation_bone == "center" else IDENTITY_QUAT.copy()
        center_vmd = vmd_position_space(center_world, root_rotation, args.root_rotation_bone)
        use_local_compensation = body_rotation_enabled and args.body_local_compensation == "on"
        solve_src = apply_inverse_body_rotation(src_positions[frame], src_name_to_idx, body_rotation) if use_local_compensation else src_positions[frame]
        if args.pose_solver_mode == "local-rot":
            solved = solve_frame_from_local_rotations(
                src_local_rotations,
                frame,
                src_name_to_idx,
                strengths,
                include_feet=args.local_rot_feet == "include",
                source_up=source_up,
            )
        elif args.pose_solver_mode == "global-rot":
            solved = solve_frame_from_global_rotations(
                pmx_bones,
                name_to_idx,
                src_rotations,
                frame,
                src_name_to_idx,
                strengths,
                center_rotation=body_rotation if use_rotation_compensation else None,
                include_feet=args.local_rot_feet == "include",
                source_up=source_up,
            )
        elif args.pose_solver_mode == "hybrid-leg-position":
            solved = solve_frame_from_global_rotations(
                pmx_bones,
                name_to_idx,
                src_rotations,
                frame,
                src_name_to_idx,
                strengths,
                center_rotation=body_rotation if use_rotation_compensation else None,
                include_feet=False,
                source_up=source_up,
            )
            solved.update(
                solve_legs_from_positions(
                    pmx_bones,
                    name_to_idx,
                    solve_src,
                    src_name_to_idx,
                    leg_solver_mode=args.leg_solver_mode,
                    leg_max_angle=args.leg_max_angle,
                    base_solved=solved,
                    pole_state=leg_pole_state,
                    ankle_rotation_mode=args.ankle_rotation_mode,
                    ankle_max_angle=args.ankle_max_angle,
                    source_base=source_rest_positions,
                )
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
        if args.finger_mode == "source":
            finger_solved = solve_fingers_from_local_rotations(
                src_local_rotations,
                frame,
                src_name_to_idx,
                strength=args.finger_strength,
                max_angle=args.finger_max_angle,
                source_up=source_up,
            )
            for name in FINGER_BONES:
                if name in finger_solved and name in name_to_idx:
                    solved[name] = finger_solved[name]
            if args.thumb_mode == "direction":
                solved.update(
                    solve_thumbs_from_positions(
                        pmx_bones,
                        name_to_idx,
                        solve_src,
                        src_name_to_idx,
                        solved,
                    )
                )
            elif args.thumb_mode == "neutral":
                for side in ("左", "右"):
                    for number in "０１２":
                        solved[f"{side}親指{number}"] = IDENTITY_QUAT.copy()
        for wrist_name, twist_name in (("左手首", "左手捩"), ("右手首", "右手捩")):
            if wrist_name in solved and twist_name in name_to_idx:
                half = quat_slerp_identity(solved[wrist_name], 0.5)
                solved[twist_name] = half
                solved[wrist_name] = half

        frame_pose = {
            "センター": (center_vmd, center_quat),
            "グルーブ": (np.zeros(3), IDENTITY_QUAT.copy()),
            "腰": (np.zeros(3), IDENTITY_QUAT.copy()),
        }
        if args.root_rotation_bone == "parent" and "全ての親" in name_to_idx:
            frame_pose["全ての親"] = (np.zeros(3), root_quat)
        elif args.root_rotation_bone == "lower" and "下半身" in name_to_idx:
            solved["下半身"] = quat_mul(root_quat, solved.get("下半身", IDENTITY_QUAT.copy()))
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
                center_vmd = center_vmd.copy()
                center_vmd[1] -= (min_y - args.ground_fit_target) * max(0.0, min(1.0, args.ground_fit_strength))
                frame_pose["センター"] = (center_vmd, center_quat)
        center_world_for_targets = world_position_space(center_vmd, root_rotation, args.root_rotation_bone)

        if args.root_rotation_bone == "parent" and "全ての親" in name_to_idx:
            chunks.append(bone_frame("全ての親", frame, quat=root_quat))
        chunks.append(bone_frame("センター", frame, center_vmd, center_quat))
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
                            center_world_for_targets,
                            side,
                        )
                        if target is None:
                            target = posed[name_to_idx[ankle_name]]
                    else:
                        target = posed[name_to_idx[ankle_name]]
                    target = vmd_position_space(target, root_rotation, args.root_rotation_bone)
                    foot_targets[ik_name][frame] = target.copy()
                    if args.foot_rotation_mode == "follow-body" and body_rotation_enabled:
                        foot_rotations[ik_name][frame] = root_quat.copy()
                    else:
                        foot_rotations[ik_name][frame] = IDENTITY_QUAT.copy()
        neutral_bones = ()
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
    print(f"source_up={source_up} requested={args.source_up}")
    print(f"root_position_indices={','.join(src_names[i] for i in sorted(root_position_indices)) if root_position_indices else 'none'}")
    print(f"center_source={center_source_name} requested={args.center_source}")
    print(f"position_scale={position_scale:.8f} mode={position_scale_mode}")
    print(f"body_rotation={'on' if body_rotation_enabled else 'off'} mode={args.body_rotation_mode}")
    print(f"body_frame_mode={args.body_frame_mode}")
    print(f"body_rotation_transform={args.body_rotation_transform}")
    print(f"body_local_compensation={args.body_local_compensation}")
    print(f"motion_fidelity={args.motion_fidelity}")
    print(f"neck_strength={strengths.get('首', 0.0):.3f}")
    print(f"head_strength={strengths.get('頭', 0.0):.3f}")
    print(f"effective_hand_offset=({hand_outward:.3f},{hand_forward:.3f},{hand_down:.3f})")
    print(f"foot_ik_mode={effective_foot_ik_mode} requested={args.foot_ik_mode}")
    print(f"foot_rotation_mode={args.foot_rotation_mode}")
    print(f"knee_hinge={args.knee_hinge} sign={args.knee_hinge_sign}")
    print(f"leg_solver_mode={args.leg_solver_mode}")
    print(f"leg_max_angle={args.leg_max_angle}")
    print(f"ankle_rotation_mode={args.ankle_rotation_mode} max={args.ankle_max_angle}")
    print(f"pose_solver_mode={args.pose_solver_mode}")
    print(f"local_rot_feet={args.local_rot_feet}")


if __name__ == "__main__":
    main()
