import math
import os
import struct
import sys
from collections import defaultdict

import numpy as np


TRACKED_BONES = (
    "センター",
    "下半身",
    "上半身",
    "上半身2",
    "首",
    "左腕",
    "左ひじ",
    "左手首",
    "右腕",
    "右ひじ",
    "右手首",
    "左足",
    "左ひざ",
    "左足首",
    "右足",
    "右ひざ",
    "右足首",
    "左足ＩＫ",
    "右足ＩＫ",
)


def read_vmd_name(raw):
    return raw.split(b"\x00", 1)[0].decode("shift_jis", errors="replace")


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return q / n


def quat_angle_delta(a, b):
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = abs(float(np.dot(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def parse_vmd_bones(path):
    by_bone = defaultdict(dict)
    with open(path, "rb") as f:
        header = read_vmd_name(f.read(30))
        model = read_vmd_name(f.read(20))
        if not header.startswith("Vocaloid Motion Data"):
            raise ValueError(f"not a VMD file: {path}")
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            name = read_vmd_name(f.read(15))
            frame = struct.unpack("<I", f.read(4))[0]
            pos = np.array(struct.unpack("<3f", f.read(12)), dtype=np.float64)
            quat = np.array(struct.unpack("<4f", f.read(16)), dtype=np.float64)
            by_bone[name][frame] = (pos, quat_normalize(quat))
            f.seek(64, os.SEEK_CUR)
    return model, by_bone


def segments(mask):
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_vmd_motion_energy.py file.vmd [file2.vmd ...]")
    for path in sys.argv[1:]:
        model, by_bone = parse_vmd_bones(path)
        frames = sorted({frame for frames_by_bone in by_bone.values() for frame in frames_by_bone})
        if len(frames) < 2:
            print(f"{path}: not enough frames")
            continue

        tracked = [name for name in TRACKED_BONES if name in by_bone]
        per_bone_energy = {name: [] for name in tracked}
        total_energy = []
        center_move = []
        for a, b in zip(frames, frames[1:]):
            e = 0.0
            contributors = 0
            for name in tracked:
                if a not in by_bone[name] or b not in by_bone[name]:
                    continue
                apos, aquat = by_bone[name][a]
                bpos, bquat = by_bone[name][b]
                pos_delta = float(np.linalg.norm(bpos - apos))
                rot_delta = quat_angle_delta(aquat, bquat)
                bone_e = pos_delta + rot_delta * 0.05
                per_bone_energy[name].append(bone_e)
                e += bone_e
                contributors += 1
                if name == "センター":
                    center_move.append(pos_delta)
            total_energy.append(e / max(1, contributors))

        energy = np.array(total_energy, dtype=np.float64)
        print(f"VMD: {path}")
        print(f"model={model} frames={frames[0]}-{frames[-1]} count={len(frames)} tracked={len(tracked)}")
        print(f"energy min/median/max={energy.min():.4f}/{np.median(energy):.4f}/{energy.max():.4f}")
        if center_move:
            c = np.array(center_move, dtype=np.float64)
            print(f"centerMove min/median/max={c.min():.4f}/{np.median(c):.4f}/{c.max():.4f}")

        low_threshold = max(0.02, float(np.percentile(energy, 15)))
        high_threshold = float(np.percentile(energy, 85))
        print(f"low_threshold={low_threshold:.4f} high_threshold={high_threshold:.4f}")
        print("low_motion_segments")
        for start, end in segments(energy <= low_threshold):
            if end - start >= 5:
                print(f"{frames[start]}-{frames[end]} ({end - start} steps)")
        print("high_motion_segments")
        for start, end in segments(energy >= high_threshold):
            if end - start >= 3:
                print(f"{frames[start]}-{frames[end]} ({end - start} steps)")

        print("top_bone_energy median/max")
        scored = []
        for name, values in per_bone_energy.items():
            if values:
                arr = np.array(values, dtype=np.float64)
                scored.append((float(np.median(arr)), float(np.max(arr)), name))
        for median, max_value, name in sorted(scored, reverse=True)[:12]:
            print(f"{name}: {median:.4f}/{max_value:.4f}")
        print()


if __name__ == "__main__":
    main()
