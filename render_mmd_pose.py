import argparse
import math
import os
import struct

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependency: numpy. Install it with: python -m pip install -r requirements.txt") from exc


def read_text(f, encoding):
    n = struct.unpack("<i", f.read(4))[0]
    return f.read(n).decode("utf-16-le" if encoding == 0 else "utf-8", errors="replace")


def read_index(f, size):
    if size == 1:
        return struct.unpack("<b", f.read(1))[0]
    if size == 2:
        return struct.unpack("<h", f.read(2))[0]
    return struct.unpack("<i", f.read(4))[0]


def skip_index(f, size):
    f.seek(size, os.SEEK_CUR)


def load_pmx_bones(path):
    with open(path, "rb") as f:
        if f.read(4) != b"PMX ":
            raise ValueError("not PMX")
        f.seek(4, os.SEEK_CUR)
        header_size = f.read(1)[0]
        header = f.read(header_size)
        encoding = header[0]
        add_uv = header[1]
        vertex_index_size, texture_index_size, bone_index_size = header[2], header[3], header[5]
        for _ in range(4):
            read_text(f, encoding)
        vertex_count = struct.unpack("<i", f.read(4))[0]
        for _ in range(vertex_count):
            f.seek(12 + 12 + 8 + 4 * add_uv * 4, os.SEEK_CUR)
            weight_type = f.read(1)[0]
            if weight_type == 0:
                skip_index(f, bone_index_size)
            elif weight_type == 1:
                skip_index(f, bone_index_size)
                skip_index(f, bone_index_size)
                f.seek(4, os.SEEK_CUR)
            elif weight_type == 2:
                for _ in range(4):
                    skip_index(f, bone_index_size)
                f.seek(16, os.SEEK_CUR)
            elif weight_type == 3:
                skip_index(f, bone_index_size)
                skip_index(f, bone_index_size)
                f.seek(40, os.SEEK_CUR)
            elif weight_type == 4:
                for _ in range(4):
                    skip_index(f, bone_index_size)
                f.seek(64, os.SEEK_CUR)
            f.seek(4, os.SEEK_CUR)
        face_count = struct.unpack("<i", f.read(4))[0]
        f.seek(face_count * vertex_index_size, os.SEEK_CUR)
        texture_count = struct.unpack("<i", f.read(4))[0]
        for _ in range(texture_count):
            read_text(f, encoding)
        material_count = struct.unpack("<i", f.read(4))[0]
        for _ in range(material_count):
            read_text(f, encoding)
            read_text(f, encoding)
            f.seek(65, os.SEEK_CUR)
            skip_index(f, texture_index_size)
            skip_index(f, texture_index_size)
            f.seek(1, os.SEEK_CUR)
            toon_flag = f.read(1)[0]
            if toon_flag == 0:
                skip_index(f, texture_index_size)
            else:
                f.seek(1, os.SEEK_CUR)
            read_text(f, encoding)
            f.seek(4, os.SEEK_CUR)
        bone_count = struct.unpack("<i", f.read(4))[0]
        bones = []
        for _ in range(bone_count):
            name = read_text(f, encoding)
            read_text(f, encoding)
            pos = np.array(struct.unpack("<3f", f.read(12)), dtype=np.float64)
            parent = read_index(f, bone_index_size)
            f.seek(4, os.SEEK_CUR)
            flags = struct.unpack("<H", f.read(2))[0]
            if flags & 0x0001:
                skip_index(f, bone_index_size)
            else:
                f.seek(12, os.SEEK_CUR)
            if flags & 0x0100 or flags & 0x0200:
                skip_index(f, bone_index_size)
                f.seek(4, os.SEEK_CUR)
            if flags & 0x0400:
                f.seek(12, os.SEEK_CUR)
            if flags & 0x0800:
                f.seek(24, os.SEEK_CUR)
            if flags & 0x2000:
                f.seek(4, os.SEEK_CUR)
            if flags & 0x0020:
                skip_index(f, bone_index_size)
                f.seek(8, os.SEEK_CUR)
                link_count = struct.unpack("<i", f.read(4))[0]
                for _ in range(link_count):
                    skip_index(f, bone_index_size)
                    has_limit = f.read(1)[0]
                    if has_limit:
                        f.seek(24, os.SEEK_CUR)
            bones.append({"name": name, "parent": parent, "pos": pos})
    return bones


def read_vmd_name(raw):
    return raw.split(b"\x00", 1)[0].decode("shift_jis", errors="replace")


def quat_to_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_vmd_frame(path, frame_no):
    result = {}
    with open(path, "rb") as f:
        f.seek(50)
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            name = read_vmd_name(f.read(15))
            frame = struct.unpack("<I", f.read(4))[0]
            pos = np.array(struct.unpack("<3f", f.read(12)), dtype=np.float64)
            quat = np.array(struct.unpack("<4f", f.read(16)), dtype=np.float64)
            f.seek(64, os.SEEK_CUR)
            if frame == frame_no:
                result[name] = (pos, quat)
    return result


def apply_pose(bones, frame):
    name_to_idx = {b["name"]: i for i, b in enumerate(bones)}
    local_rots = [np.eye(3) for _ in bones]
    local_pos = [np.zeros(3) for _ in bones]
    for name, (pos, quat) in frame.items():
        if name in name_to_idx:
            i = name_to_idx[name]
            local_rots[i] = quat_to_mat(quat)
            local_pos[i] = pos

    global_rots = [np.eye(3) for _ in bones]
    global_pos = [b["pos"].copy() for b in bones]
    for i, bone in enumerate(bones):
        parent = bone["parent"]
        if parent < 0:
            global_rots[i] = local_rots[i]
            global_pos[i] = bone["pos"] + local_pos[i]
        else:
            rest_offset = bone["pos"] - bones[parent]["pos"]
            global_rots[i] = global_rots[parent] @ local_rots[i]
            global_pos[i] = global_pos[parent] + global_rots[parent] @ rest_offset + local_pos[i]
    return np.array(global_pos)


def draw_projection(bones, pts, out_path, axes=(0, 1)):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Missing optional dependency: Pillow. Install it with: python -m pip install pillow") from exc

    key_names = {
        "センター", "グルーブ", "腰", "下半身", "上半身", "上半身2", "首", "頭",
        "左肩", "左腕", "左ひじ", "左手首", "右肩", "右腕", "右ひじ", "右手首",
        "左足", "左ひざ", "左足首", "左つま先", "右足", "右ひざ", "右足首", "右つま先",
    }
    key = [i for i, b in enumerate(bones) if b["name"] in key_names]
    xy = pts[key][:, list(axes)]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    scale = min(700 / max(mx[0] - mn[0], 1e-6), 700 / max(mx[1] - mn[1], 1e-6))
    img = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(img)

    def project(p):
        q = (p[list(axes)] - mn) * scale + 90
        return int(q[0]), int(810 - q[1])

    for i in key:
        p = bones[i]["parent"]
        if p >= 0 and p in key:
            draw.line([project(pts[p]), project(pts[i])], fill=(30, 80, 180), width=4)
    for i in key:
        x, y = project(pts[i])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(200, 40, 40))
        draw.text((x + 7, y - 7), bones[i]["name"], fill=(0, 0, 0))
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmx", required=True)
    parser.add_argument("--vmd", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bones = load_pmx_bones(args.pmx)
    frame = load_vmd_frame(args.vmd, args.frame)
    pts = apply_pose(bones, frame)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.vmd))[0]
    draw_projection(bones, pts, os.path.join(args.out, f"{stem}_{args.frame:04d}_front.png"), axes=(0, 1))
    draw_projection(bones, pts, os.path.join(args.out, f"{stem}_{args.frame:04d}_side.png"), axes=(2, 1))


if __name__ == "__main__":
    main()
