import argparse
import json
import os
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Import FBX in Blender and export its armature animation as BVH.")
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--motion-json", default="")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def find_main_armature():
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("FBX import did not create an armature.")
    armatures.sort(key=lambda obj: len(obj.data.bones), reverse=True)
    return armatures[0]


def action_frame_range(armature):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        for obj in bpy.context.scene.objects:
            if obj.animation_data and obj.animation_data.action:
                action = obj.animation_data.action
                break
    if action is None:
        return bpy.context.scene.frame_start, bpy.context.scene.frame_end
    start, end = action.frame_range
    return int(start), int(end)


def select_armature_hierarchy(armature):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    for obj in bpy.context.scene.objects:
        if obj.parent == armature:
            obj.select_set(True)


def export_bvh(path, start, end):
    kwargs = {
        "filepath": path,
        "frame_start": start,
        "frame_end": end,
        "root_transform_only": False,
    }
    try:
        bpy.ops.export_anim.bvh(**kwargs)
    except TypeError:
        kwargs.pop("root_transform_only", None)
        bpy.ops.export_anim.bvh(**kwargs)


def write_object_motion(path, armature, start, end):
    if not path:
        return
    frames = []
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        mat = armature.matrix_world.copy()
        rot = mat.to_3x3()
        frames.append(
            {
                "frame": frame - start,
                "location": [float(v) for v in mat.translation],
                "rotation": [[float(rot[row][col]) for col in range(3)] for row in range(3)],
                "scale": [float(v) for v in armature.scale],
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"start": start, "end": end, "frames": frames}, f)
    print(f"Exported object motion: {path}")


def main():
    args = parse_args()
    fbx_path = os.path.abspath(args.fbx)
    out_path = os.path.abspath(args.out)
    if not os.path.isfile(fbx_path):
        raise FileNotFoundError(fbx_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    armature = find_main_armature()
    start, end = action_frame_range(armature)
    if args.start is not None:
        start = args.start
    if args.end is not None:
        end = args.end
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    select_armature_hierarchy(armature)
    write_object_motion(args.motion_json, armature, start, end)
    export_bvh(out_path, start, end)
    print(f"Exported BVH: {out_path}")
    print(f"armature={armature.name} bones={len(armature.data.bones)} frames={start}-{end}")


if __name__ == "__main__":
    main()
