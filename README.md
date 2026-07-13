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

“身体翻转”选项：

- `auto`：推荐。走路不写中心旋转，空翻自动写中心旋转
- `none`：完全关闭中心翻转修正
- `infer`：强制从身体姿态推断中心旋转

身体参考选项：

- `full`：使用胯、胸、肩、腿推断完整身体方向
- `up`：只跟随胯到胸的上下方向，更适合侧空翻排查整体扭曲
- `hips`：直接使用 BVH 的 `Hips` 全局旋转，推荐给空翻/侧翻排查

身体方向选项：

- `normal`：默认方向
- `inverse`：反向翻转
- `mirror-x` / `mirror-y` / `mirror-z`：按指定轴镜像中心翻转，用于排查侧翻方向错误

“动作”选项：

- `preserve`：推荐。尽量按照 BVH 动作翻译，不额外削弱翻转动作，也不默认移动手部目标
- `stable`：稳定修正版。会降低翻转时的躯干/腿部强度，并启用手部外移/下压修正，适合模型明显扭曲时排查

“脚 IK”选项：

- `auto`：推荐。保留 Kimodo/BVH 的 source foot IK 轨迹
- `source`：强制写入 Kimodo 脚部轨迹
- `fk`：从当前 FK 姿态写脚 IK
- `none`：不写脚 IK 轨道

“脚旋转”选项：

- `none`：脚 IK 只写位置，不写旋转
- `follow-body`：推荐。脚 IK 跟随身体/中心翻转旋转，适合空翻时保持脚掌相对身体方向

“膝盖”选项：

- `flip`：推荐。只在空翻/身体翻转时把膝盖限制为单轴铰链，减少脚和小腿扭曲
- `off`：关闭膝盖限制
- `always`：所有动作都启用膝盖限制

“腿”选项：

- `ccd`：默认，按目标脚位置求解腿部
- `direction`：只按腿骨方向解算，不用 CCD 拉脚
- `frozen`：冻结腿部，用来判断脚扭曲是否来自腿部解算

“腿最大角”选项：

- `0`：不限制
- `90` / `120`：限制大腿和膝盖最大旋转角度，用于减少空翻时脚部极限扭曲

“姿态解算”选项：

- `position`：默认。根据关节位置重建 MMD 骨骼，适合走路和一般动作
- `global-rot`：实验项。用 BVH 全局旋转按 PMX 层级重新计算局部旋转，可用于空翻排查
- `local-rot`：实验项。当前 Kimodo BVH 与 MMD rest 轴不一致，容易出现 A pose 和扭曲，暂不推荐

“局部旋转脚腕”选项：

- `omit`：不写脚腕轨道，减少脚掌扭曲
- `include`：写入 BVH 脚腕局部旋转，用于对比脚掌是否更自然

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
