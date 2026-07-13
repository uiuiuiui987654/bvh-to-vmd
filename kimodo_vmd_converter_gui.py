import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(APP_DIR, "kimodo_to_mmd_solver.py")
FBX_TO_VMD = os.path.join(APP_DIR, "fbx_to_vmd.py")


DEFAULT_BVH = ""
DEFAULT_PMX = ""
DEFAULT_OUT = ""


class ConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kimodo VMD 转换器")
        self.geometry("760x520")
        self.minsize(680, 460)

        self.messages = queue.Queue()
        self.worker = None

        self.bvh_var = tk.StringVar(value=DEFAULT_BVH)
        self.pmx_var = tk.StringVar(value=DEFAULT_PMX)
        self.out_var = tk.StringVar(value=DEFAULT_OUT)
        self.finger_var = tk.StringVar(value="omit")
        self.body_rotation_var = tk.StringVar(value="auto")
        self.body_frame_var = tk.StringVar(value="full")
        self.body_transform_var = tk.StringVar(value="normal")
        self.motion_fidelity_var = tk.StringVar(value="preserve")
        self.foot_ik_mode_var = tk.StringVar(value="auto")
        self.foot_rotation_var = tk.StringVar(value="follow-body")
        self.knee_hinge_var = tk.StringVar(value="flip")
        self.leg_solver_var = tk.StringVar(value="ccd")
        self.leg_max_angle_var = tk.DoubleVar(value=0.0)
        self.pose_solver_var = tk.StringVar(value="position")
        self.local_rot_feet_var = tk.StringVar(value="omit")
        self.ground_fit_var = tk.StringVar(value="none")
        self.ground_fit_start_var = tk.IntVar(value=0)
        self.ground_fit_end_var = tk.IntVar(value=80)
        self.ground_fit_strength_var = tk.DoubleVar(value=1.0)
        self.foot_ik_display_var = tk.BooleanVar(value=False)
        self.position_scale_var = tk.StringVar(value="auto")

        self.hand_outward_var = tk.DoubleVar(value=0.9)
        self.hand_forward_var = tk.DoubleVar(value=-0.25)
        self.hand_down_var = tk.DoubleVar(value=2.0)
        self.wrist_strength_var = tk.DoubleVar(value=0.45)

        self._build_ui()
        self.after(100, self._drain_messages)

    def _build_ui(self):
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(9, weight=1)

        self._file_row(root, 0, "输入 BVH/FBX", self.bvh_var, self._pick_bvh)
        self._file_row(root, 1, "目标 PMX", self.pmx_var, self._pick_pmx)
        self._file_row(root, 2, "输出 VMD", self.out_var, self._pick_out)

        options = ttk.LabelFrame(root, text="稳定参数")
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        for col in range(8):
            options.columnconfigure(col, weight=1)

        self._spin(options, 0, "手部外移", self.hand_outward_var, 0.0, 2.5, 0.05)
        self._spin(options, 2, "手部前后", self.hand_forward_var, -2.0, 2.0, 0.05)
        self._spin(options, 4, "手腕下压", self.hand_down_var, 0.0, 4.0, 0.05)
        self._spin(options, 6, "手腕强度", self.wrist_strength_var, 0.0, 1.0, 0.05)
        ttk.Label(options, text="位移比例").grid(row=1, column=0, sticky="e", padx=(8, 4), pady=(0, 8))
        ttk.Entry(options, textvariable=self.position_scale_var, width=12).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(0, 8))

        finger = ttk.Frame(root)
        finger.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(finger, text="手指轨道").pack(side="left")
        ttk.Radiobutton(finger, text="不写入（推荐）", variable=self.finger_var, value="omit").pack(side="left", padx=(12, 4))
        ttk.Radiobutton(finger, text="静态默认手型", variable=self.finger_var, value="neutral").pack(side="left", padx=4)
        ttk.Label(finger, text="身体翻转").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.body_rotation_var,
            values=("auto", "none", "infer"),
            width=8,
            state="readonly",
        ).pack(side="left")
        ttk.Combobox(
            finger,
            textvariable=self.body_frame_var,
            values=("full", "up", "hips"),
            width=7,
            state="readonly",
        ).pack(side="left", padx=(4, 0))
        ttk.Combobox(
            finger,
            textvariable=self.body_transform_var,
            values=("normal", "inverse", "mirror-x", "mirror-y", "mirror-z"),
            width=9,
            state="readonly",
        ).pack(side="left", padx=(4, 0))
        ttk.Label(finger, text="动作").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.motion_fidelity_var,
            values=("preserve", "stable"),
            width=9,
            state="readonly",
        ).pack(side="left")
        ttk.Label(finger, text="脚 IK").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.foot_ik_mode_var,
            values=("auto", "source", "fk", "none"),
            width=8,
            state="readonly",
        ).pack(side="left")
        ttk.Label(finger, text="脚旋转").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.foot_rotation_var,
            values=("none", "follow-body"),
            width=11,
            state="readonly",
        ).pack(side="left")
        ttk.Label(finger, text="膝盖").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.knee_hinge_var,
            values=("flip", "off", "always"),
            width=8,
            state="readonly",
        ).pack(side="left")
        ttk.Label(finger, text="腿").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            finger,
            textvariable=self.leg_solver_var,
            values=("ccd", "direction", "frozen"),
            width=9,
            state="readonly",
        ).pack(side="left")
        ttk.Checkbutton(finger, text="强制打开足 IK 显示", variable=self.foot_ik_display_var).pack(side="right")

        solver = ttk.Frame(root)
        solver.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(solver, text="姿态解算").pack(side="left")
        ttk.Combobox(
            solver,
            textvariable=self.pose_solver_var,
            values=("position", "global-rot", "local-rot"),
            width=10,
            state="readonly",
        ).pack(side="left", padx=(12, 4))
        ttk.Label(solver, text="局部旋转脚腕").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            solver,
            textvariable=self.local_rot_feet_var,
            values=("omit", "include"),
            width=8,
            state="readonly",
        ).pack(side="left")
        ttk.Label(solver, text="腿最大角").pack(side="left", padx=(18, 4))
        ttk.Spinbox(solver, textvariable=self.leg_max_angle_var, from_=0, to=180, increment=5, width=8).pack(side="left")

        ground = ttk.Frame(root)
        ground.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(ground, text="贴地").pack(side="left")
        ttk.Combobox(
            ground,
            textvariable=self.ground_fit_var,
            values=("none", "first", "all"),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(12, 4))
        ttk.Label(ground, text="开始").pack(side="left", padx=(12, 4))
        ttk.Spinbox(ground, textvariable=self.ground_fit_start_var, from_=0, to=9999, increment=1, width=7).pack(side="left")
        ttk.Label(ground, text="结束").pack(side="left", padx=(12, 4))
        ttk.Spinbox(ground, textvariable=self.ground_fit_end_var, from_=0, to=9999, increment=1, width=7).pack(side="left")
        ttk.Label(ground, text="强度").pack(side="left", padx=(12, 4))
        ttk.Spinbox(ground, textvariable=self.ground_fit_strength_var, from_=0, to=1, increment=0.05, width=7).pack(side="left")

        buttons = ttk.Frame(root)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        self.convert_btn = ttk.Button(buttons, text="开始转换", command=self._start_convert)
        self.convert_btn.pack(side="left")
        ttk.Button(buttons, text="恢复稳定参数", command=self._reset_stable).pack(side="left", padx=8)
        ttk.Button(buttons, text="打开输出文件夹", command=self._open_output_folder).pack(side="right")

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.grid(row=8, column=0, columnspan=3, sticky="ew")

        self.log = tk.Text(root, height=12, wrap="word")
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.log.configure(state="disabled")

    def _file_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="浏览", command=command).grid(row=row, column=2, sticky="ew", pady=4)

    def _spin(self, parent, col, label, var, from_, to, step):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, columnspan=2, sticky="ew", padx=6, pady=8)
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Spinbox(frame, textvariable=var, from_=from_, to=to, increment=step, width=8).pack(fill="x")

    def _pick_bvh(self):
        path = filedialog.askopenfilename(
            title="选择 BVH 或 FBX",
            filetypes=[("BVH/FBX", "*.bvh *.fbx"), ("BVH", "*.bvh"), ("FBX", "*.fbx"), ("所有文件", "*.*")],
        )
        if path:
            self.bvh_var.set(path)
            base = os.path.splitext(path)[0]
            suffix = "_fbx_fixed.vmd" if path.lower().endswith(".fbx") else "_kimodo_fixed.vmd"
            self.out_var.set(base + suffix)

    def _pick_pmx(self):
        path = filedialog.askopenfilename(title="选择 MMD PMX", filetypes=[("PMX", "*.pmx"), ("所有文件", "*.*")])
        if path:
            self.pmx_var.set(path)

    def _pick_out(self):
        path = filedialog.asksaveasfilename(title="输出 VMD", defaultextension=".vmd", filetypes=[("VMD", "*.vmd"), ("所有文件", "*.*")])
        if path:
            self.out_var.set(path)

    def _reset_stable(self):
        self.hand_outward_var.set(0.9)
        self.hand_forward_var.set(-0.25)
        self.hand_down_var.set(2.0)
        self.wrist_strength_var.set(0.45)
        self.finger_var.set("omit")
        self.body_rotation_var.set("auto")
        self.body_frame_var.set("full")
        self.body_transform_var.set("normal")
        self.motion_fidelity_var.set("preserve")
        self.foot_ik_mode_var.set("auto")
        self.foot_rotation_var.set("follow-body")
        self.knee_hinge_var.set("flip")
        self.leg_solver_var.set("ccd")
        self.leg_max_angle_var.set(0.0)
        self.pose_solver_var.set("position")
        self.local_rot_feet_var.set("omit")
        self.ground_fit_var.set("none")
        self.position_scale_var.set("auto")
        self.ground_fit_start_var.set(0)
        self.ground_fit_end_var.set(80)
        self.ground_fit_strength_var.set(1.0)
        self.foot_ik_display_var.set(False)

    def _start_convert(self):
        if self.worker and self.worker.is_alive():
            return

        bvh = self.bvh_var.get().strip()
        pmx = self.pmx_var.get().strip()
        out = self.out_var.get().strip()
        if not os.path.isfile(bvh):
            messagebox.showerror("文件不存在", "找不到 BVH/FBX 文件")
            return
        if not os.path.isfile(pmx):
            messagebox.showerror("文件不存在", "找不到 PMX 文件")
            return
        if not out:
            messagebox.showerror("路径错误", "请设置输出 VMD 路径")
            return

        is_fbx = bvh.lower().endswith(".fbx")
        if is_fbx:
            args = [
                sys.executable,
                FBX_TO_VMD,
                "--fbx",
                bvh,
                "--pmx",
                pmx,
                "--out",
                out,
            ]
        else:
            ground_fit_mode = self.ground_fit_var.get()
            args = [
                sys.executable,
                SOLVER,
                "--bvh",
                bvh,
                "--pmx",
                pmx,
                "--out",
                out,
                "--position-scale",
                self.position_scale_var.get().strip() or "auto",
                "--foot-ik-mode",
                self.foot_ik_mode_var.get(),
                "--foot-rotation-mode",
                self.foot_rotation_var.get(),
                "--wrist-strength",
                str(self.wrist_strength_var.get()),
                "--hand-outward",
                str(self.hand_outward_var.get()),
                "--hand-forward",
                str(self.hand_forward_var.get()),
                "--hand-down",
                str(self.hand_down_var.get()),
                "--finger-mode",
                self.finger_var.get(),
                "--body-rotation-mode",
                self.body_rotation_var.get(),
                "--body-frame-mode",
                self.body_frame_var.get(),
                "--body-rotation-transform",
                self.body_transform_var.get(),
                "--motion-fidelity",
                self.motion_fidelity_var.get(),
                "--knee-hinge",
                self.knee_hinge_var.get(),
                "--leg-solver-mode",
                self.leg_solver_var.get(),
                "--leg-max-angle",
                str(self.leg_max_angle_var.get()),
                "--pose-solver-mode",
                self.pose_solver_var.get(),
                "--local-rot-feet",
                self.local_rot_feet_var.get(),
                "--ground-fit-mode",
                ground_fit_mode,
                "--ground-fit-start",
                str(self.ground_fit_start_var.get()),
                "--ground-fit-end",
                str(self.ground_fit_end_var.get()),
                "--ground-fit-strength",
                str(self.ground_fit_strength_var.get()),
            ]
        if self.foot_ik_display_var.get():
            args.append("--enable-foot-ik")

        self._clear_log()
        self._log("开始转换...\n")
        self.convert_btn.configure(state="disabled")
        self.progress.start(10)
        self.worker = threading.Thread(target=self._run_solver, args=(args,), daemon=True)
        self.worker.start()

    def _run_solver(self, args):
        try:
            proc = subprocess.run(
                args,
                cwd=APP_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.messages.put(("log", proc.stdout or ""))
            if proc.returncode == 0:
                self.messages.put(("done", "转换完成"))
            else:
                self.messages.put(("error", "转换失败"))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _drain_messages(self):
        try:
            while True:
                kind, text = self.messages.get_nowait()
                if kind == "log":
                    self._log(text)
                elif kind == "done":
                    self.progress.stop()
                    self.convert_btn.configure(state="normal")
                    self._log("\n完成。\n")
                    messagebox.showinfo("完成", text)
                elif kind == "error":
                    self.progress.stop()
                    self.convert_btn.configure(state="normal")
                    self._log("\n" + text + "\n")
                    messagebox.showerror("错误", text)
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_output_folder(self):
        out = self.out_var.get().strip()
        folder = os.path.dirname(out) if out else ""
        if os.path.isdir(folder):
            os.startfile(folder)
        else:
            messagebox.showerror("路径错误", "输出文件夹不存在")


if __name__ == "__main__":
    ConverterApp().mainloop()
