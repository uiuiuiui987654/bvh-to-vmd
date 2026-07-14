# Kimodo VMD Converter

把 Kimodo 输出的 BVH 转成 MMD 可用的 VMD。

当前默认参数以“尽量按 BVH 翻译”为主：

- 位移比例：`auto`，按 BVH 骨架身高和 PMX 身高自动换算中心位移
- 动作模式：`preserve`，保留 BVH 翻转时的躯干和腿部动作强度
- 脚步：`auto`，默认保留 source foot IK；空翻时建议搭配 `follow-body` 让脚跟随身体旋转
- 中心：保留位移，空翻时可自动写入 `センター` 旋转
- 手部：`preserve` 模式不加人工偏移；`stable` 模式才使用外移/下压修正
- 手腕强度：`0.45`
- 手指：Kimodo BVH 没有可靠手指参数，默认不写入；FBX 会读取并转换 30 根手指骨
- 空翻：自动检测身体翻转，并把翻转写入 `センター` 旋转

## 图形界面

第一次使用前安装依赖：

```bat
python -m pip install -r requirements.txt
```

双击：

```bat
KimodoVMD转换器.cmd
```

选择：

1. Kimodo 输出的 `.bvh`
2. 目标 MMD `.pmx`
3. 输出 `.vmd`

然后点击“开始转换”。

身体、腿部与脚部解算会自动使用已验证的稳定配置，无需手动调整。

## 命令行

`convert_cli.cmd` 需要依次传入 BVH、PMX 和输出 VMD 路径：

```bat
convert_cli.cmd "input.bvh" "model.pmx" "output.vmd"
```

`convert_cli.cmd` 默认使用和 GUI 一致的保真参数：

- `--position-scale auto`
- `--motion-fidelity preserve`
- `--foot-ik-mode auto`
- `--foot-rotation-mode follow-body`
- `--body-rotation-mode auto`
- `--knee-hinge flip`
- `--pose-solver-mode position`

## FBX 转 VMD

FBX 转换需要 Blender。程序会先由 Blender 导出临时 BVH 和物体变换，再转换为 VMD：

```bat
convert_fbx_cli.cmd "input.fbx" "model.pmx" "output.vmd"
```

FBX 默认配置会：

- 保留 FBX 的移动和整体转向
- 按 Blender 世界坐标映射 VMD 中心位移，并保留首帧水平原点
- 使用 FK 腿部和相对脚腕弯曲，关闭足 IK 与足尖 IK，避免脚掌被 IK 拉扭
- 转换左右手共 30 根手指骨
- 单独按掌面和拇指链方向解算拇指，避免套用普通手指骨轴

可用 `--object-origin-mode delta` 忽略 FBX 首帧放置位置，或用 `horizontal` 保留水平位置。`absolute` 还会保留垂直位置，通常不建议与贴地修正同时使用。

## 可选手指模式

直接转换 Kimodo BVH 时默认不写手指轨道，因为 Kimodo 没有可靠的手指参数。FBX 转换默认使用：

```bat
--finger-mode source --finger-strength 1.0 --finger-max-angle 90 --thumb-mode direction
```

如果需要强制写入静态默认手型，可以在命令行改用：

```bat
--finger-mode neutral
```

## 打包 exe

如果当前 Python 环境安装了 PyInstaller，运行：

```bat
build_exe.cmd
```

生成的 exe 会在 `dist` 文件夹中。
