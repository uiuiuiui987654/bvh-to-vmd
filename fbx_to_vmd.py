import argparse
import os
import subprocess
import sys
import tempfile


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_BRIDGE = os.path.join(APP_DIR, "fbx_to_bvh_blender.py")
SOLVER = os.path.join(APP_DIR, "kimodo_to_mmd_solver.py")
DEFAULT_BLENDER_PATHS = (
    r"E:\SteamLibrary\steamapps\common\Blender\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender\blender.exe",
)


def find_blender(explicit_path):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(DEFAULT_BLENDER_PATHS)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return "blender"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FBX animation to VMD by exporting FBX to BVH with Blender first."
    )
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--pmx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blender", default="")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--fbx-start", type=int, default=None)
    parser.add_argument("--fbx-end", type=int, default=None)
    args, solver_args = parser.parse_known_args()
    return args, solver_args


def run_checked(cmd, label):
    print(f"[{label}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=APP_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout or "")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def has_arg(args, option):
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def add_default_arg(args, option, value):
    if not has_arg(args, option):
        args.extend([option, str(value)])


def apply_fbx_solver_defaults(solver_args):
    out = list(solver_args)
    defaults = (
        ("--position-scale", "auto"),
        ("--center-source", "hips"),
        ("--source-up", "auto"),
        ("--motion-fidelity", "preserve"),
        ("--foot-ik-mode", "none"),
        ("--foot-rotation-mode", "none"),
        ("--wrist-strength", "0.45"),
        ("--finger-mode", "source"),
        ("--finger-strength", "1.0"),
        ("--finger-max-angle", "90"),
        ("--thumb-mode", "direction"),
        ("--body-rotation-mode", "auto"),
        ("--body-frame-mode", "full"),
        ("--body-rotation-transform", "normal"),
        ("--body-local-compensation", "off"),
        ("--root-rotation-bone", "parent"),
        ("--rotation-compensation", "off"),
        ("--object-root-rotation", "object"),
        ("--object-root-axis", "yaw"),
        ("--object-origin-mode", "horizontal"),
        ("--ground-fit-mode", "all"),
        ("--ground-fit-strength", "1.0"),
        ("--knee-hinge", "off"),
        ("--leg-solver-mode", "pole"),
        ("--ankle-rotation-mode", "relative-flex"),
        ("--ankle-max-angle", "35"),
        ("--pose-solver-mode", "hybrid-leg-position"),
        ("--local-rot-feet", "omit"),
    )
    for option, value in defaults:
        add_default_arg(out, option, value)
    return out


def main():
    args, solver_args = parse_args()
    args.fbx = os.path.abspath(args.fbx)
    args.pmx = os.path.abspath(args.pmx)
    args.out = os.path.abspath(args.out)
    if not os.path.isfile(args.fbx):
        raise FileNotFoundError(args.fbx)
    if not os.path.isfile(args.pmx):
        raise FileNotFoundError(args.pmx)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    blender = find_blender(args.blender)
    temp_dir = tempfile.mkdtemp(prefix="fbx_to_vmd_")
    bvh_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(args.fbx))[0] + ".bvh")
    motion_json_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(args.fbx))[0] + "_object_motion.json")

    blender_cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        BLENDER_BRIDGE,
        "--",
        "--fbx",
        args.fbx,
        "--out",
        bvh_path,
        "--motion-json",
        motion_json_path,
    ]
    if args.fbx_start is not None:
        blender_cmd.extend(["--start", str(args.fbx_start)])
    if args.fbx_end is not None:
        blender_cmd.extend(["--end", str(args.fbx_end)])
    run_checked(blender_cmd, "fbx-to-bvh")

    solver_cmd = [
        sys.executable,
        SOLVER,
        "--bvh",
        bvh_path,
        "--pmx",
        args.pmx,
        "--out",
        args.out,
        "--object-motion-json",
        motion_json_path,
    ]
    solver_cmd.extend(apply_fbx_solver_defaults(solver_args))
    run_checked(solver_cmd, "bvh-to-vmd")

    if args.keep_temp:
        print(f"Temporary BVH kept: {bvh_path}")
        print(f"Temporary object motion kept: {motion_json_path}")


if __name__ == "__main__":
    main()
