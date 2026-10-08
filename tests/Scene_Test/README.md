# 地形生成器：用法与参数速查

本目录的 `terrain_generator.py` 是 Newton 1.0.0 的独立地形生成源码。本指南按**本目录实际代码**编写。它生成顶点和三角面；资产导出由 `creator1.py` 中的 Trimesh 完成。

快速跳转：

- [生成与运行](#先生成一个资产)
- [坐标与摆放](#坐标为什么看起来竖着)
- [两个公开入口](#两个公开入口)
- [七种地形参数速查](#全部内置地形参数速查)
- [常用示例](#常用完整示例)
- [导入 OBJ](#导入-obj拼接成场景资产)
- [SDF 烘焙与重建](#sdf-烘焙与重建)
- [输出格式](#输出格式速查)
- [全部辅助函数](#内部函数完整索引)
- [常见问题](#常见问题速查)

## 先生成一个资产

在仓库根目录运行：

```powershell
conda activate mimickit-newton
python tests/Scene_Test/creator1.py --terrain wave
```

当前输出为 **`tests/Scene_Test/Assets/procedural/<类型名>.obj`**，例如 `wave.obj`。同一类型重复运行会覆盖原文件，不会添加编号。输出路径相对于脚本定位，因此从其他工作目录运行也会写入同一个分类目录。

当前脚本生成一块 6×4 米的波浪地形，幅度为 0.75 米，每个方向有 30 个采样点；导出默认采用 **Y-up**。脚本不启动查看器。

### CLI 速查

```powershell
# 选择一种地形
python tests/Scene_Test/creator1.py --terrain pyramid_stairs

# 显式调用七种独立生成函数，各自保存为一个 OBJ
python tests/Scene_Test/creator1.py --terrain all

# 使用 Z-up，并固定随机种子
python tests/Scene_Test/creator1.py --terrain random_grid --up-axis Z --seed 42

# 从二维 .npy 数组生成有底面和侧壁的地形
python tests/Scene_Test/creator1.py --terrain heightfield --heightfield height.npy

# 查看全部 CLI 参数
python tests/Scene_Test/creator1.py --help
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--terrain` | `wave` | 七种内置类型之一，或 `all` 分别导出全部七种 |
| `--up-axis` | `Y` | `Y` 或 `Z`，也接受小写 |
| `--seed` | 不固定 | 控制 `random_grid` 的随机布局 |
| `--heightfield` | 不读取文件 | 用于 `heightfield` 或 `all`；文件路径相对于运行目录 |
| `--ground-z` | 最低地表高度减 0.3 米 | 用于 `heightfield` 或 `all`；底面必须低于最低地表 |

`heightfield` 未传数组时，脚本会生成一张 41×31 的浮点高度图，包含一处隆起和一处凹坑，四周平滑衰减到零高度。底面位于最低地表下方 0.3 米，并连接侧壁。这是 CLI 的示例预设；生成器本身不传高度图时仍返回无厚度平面。波浪保留原脚本参数，其余五种类型的参数值与生成器默认值一致。

当前 `generate_heightfield()` 保留 `grid_size=(6, 6)`：同一张高度图重复排列 36 块，每块 6×4 米，整体为 36×24 米。只想看一处隆起和凹坑时，将该函数中的 `grid_size` 改为 `(1, 1)`。

默认高度图的可调参数：

| 参数 | 当前值 | 观察效果 |
|---|---|---|
| `resolution_x`、`resolution_y` | `41`、`31` | 每轴采样点数；越大曲面越细 |
| `hill_height` | `1.0` 米 | 隆起强度；经过边缘衰减后实际峰值稍低 |
| `pit_depth` | `0.6` 米 | 凹坑强度；越大凹陷越深 |
| `hill_radius`、`pit_radius` | `0.18`、`0.16` | 相对于块尺寸的宽度比例 |
| `hill_center` | `(0.35, 0.45)` | 隆起中心在块长宽中的相对位置 |
| `pit_center` | `(0.70, 0.60)` | 凹坑中心在块长宽中的相对位置 |

这些参数放在 `generate_heightfield()` 内，用来构造示例数组；它们不是原始 Newton API 的参数。提供 `--heightfield` 文件时使用文件中的高度，不使用上述示例。

### 每种地形的代码在哪里？

`creator1.py` 为每种地形提供独立函数，各自直接写出 `grid_size`、`block_size`、类型名和对应参数。每个参数旁均有注释，解释控制的形状、方向、单位或取值含义。修改某种地形时，直接修改该函数中的数值即可。

| 类型 | 生成函数 | 参数位置 |
|---|---|---|
| `flat` | `generate_flat()` | 高度 `height` |
| `pyramid_stairs` | `generate_pyramid_stairs()` | 台阶宽度、高度增量、平台宽度 |
| `random_grid` | `generate_random_grid(seed=None)` | 格子宽度、高度范围、随机种子 |
| `wave` | `generate_wave()` | 波浪幅度、频率、采样点数 |
| `box` | `generate_box()` | 平台高度、宽度 |
| `gap` | `generate_gap()` | 间隔宽度、平台宽度 |
| `heightfield` | `generate_heightfield(heightfield_path=None, ground_z=None)` | 高度数组、中心位置、底面高度 |

选择器 `select_generator(类型名)` 返回对应函数，单种地形生成函数返回 `(vertices, indices)`；`export_mesh()` 负责向上轴转换和保存。`all` 对应 `generate_all()`，内部显式调用七种函数，不使用生成循环，分别导出并返回七个文件路径。各类型参数仍在各自的生成函数中修改。

也可以从本目录的其他 Python 脚本直接调用：

```python
from creator1 import generate_wave, select_generator, export_mesh

# 直接使用指定生成函数
vertices, indices = generate_wave()
output_path = export_mesh("wave", vertices, indices, up_axis="Y")

# 或通过选择器获取函数；仅选择时不会生成资产
generator = select_generator("pyramid_stairs")
vertices, indices = generator()
output_path = export_mesh("pyramid_stairs", vertices, indices, up_axis="Y")
```

## 目录、依赖与处理流程

```text
Scene_Test/
├── terrain_generator.py   # 生成器源码
├── creator1.py            # CLI、坐标转换、资产导出
├── scene_creator.py       # 独立的阶梯 + OBJ 放置与布尔并集示例
├── repair_bunny.py        # 单独补 bunny 底部开口并另存副本
├── sdf_baker.py           # 读取布尔 mesh、烘焙 SDF、重建和导出切片
├── README.md              # 本指南
└── Assets/
    ├── origin/           # bunny、奶牛等导入模型，以及 bunny 修复副本
    ├── procedural/       # wave.obj 等纯程序化地形
    ├── output/           # 拼接/布尔场景、SDF、重建 OBJ、切片 PNG
    └── .warp_cache/      # Warp 编译缓存
```

| 部分 | 依赖 | 作用 |
|---|---|---|
| `terrain_generator.py` | NumPy；标准库 `typing` | 生成几何、排列地形块、拼接索引 |
| `creator1.py` | NumPy、Trimesh；标准库 `pathlib`、`argparse` | CLI、生成 mesh、旋转、写文件 |
| `scene_creator.py` / `repair_bunny.py` | NumPy、Trimesh、manifold3d | 实体布尔 / bunny 局部补洞 |
| `sdf_baker.py` | NumPy、Trimesh、Warp、Matplotlib；CUDA | SDF 烘焙、保存、重建和切片 |

独立使用生成器时不需要导入 Newton，也不需要 Warp、CUDA 或物理求解器。源码文件内已包含所有辅助函数；若复制或分发该文件，请保留原有 Apache-2.0 许可声明。

生成流程：**参数 → 顶点和三角面 → Trimesh → 可选坐标转换 → 文件**。

## 坐标：为什么看起来竖着？

生成器始终使用 **Z-up**：X、Y 是地面方向，Z 是高度。OBJ 文件本身不统一规定向上轴，也不统一规定长度单位；本指南按米使用生成参数。

如果查看器采用 Y-up，原始 Z-up 地面可能被显示为竖直面。`creator1.py` 默认在导出前绕 X 轴旋转 **−90°**：

```text
原始 Z-up： (x, y, z)
导出 Y-up： (x, z, -y)
```

| 查看或使用方式 | 设置 |
|---|---|
| 查看器采用 Y-up | `--up-axis Y`，当前默认 |
| 查看器采用 Z-up，或给 Newton/Warp 使用 | `--up-axis Z` |
| 查看器能在导入时转换轴 | 选择一种转换方式，避免脚本和查看器重复旋转 |

查看器名称尚未指定，所以默认 Y-up 是针对“竖着”的现象采用的假设。若旋转后仍不符合预期，先确认查看器的向上轴。

Warp 本身不强制向上轴；网格、采样点和后续场景采用同一个坐标约定即可。这里建议 Z-up，是为了与此生成器和当前 Newton 场景保持一致。

### 地形放在哪里？

`create_mesh_terrain()` 的常规块范围从 `(0, 0)` 开始，沿正方向排列：

```text
X ∈ [0, 列数 × 每块 X 长度]
Y ∈ [0, 行数 × 每块 Y 长度]
```

例如 `grid_size=(2, 3)`、`block_size=(6, 4)`，整体覆盖 X 方向 18 米、Y 方向 8 米。行沿 Y，列沿 X。

若要把整体水平中心放到原点，**在轴转换前**执行：

```python
rows, cols = 2, 3
size_x, size_y = 6.0, 4.0
mesh.apply_translation([-cols * size_x / 2, -rows * size_y / 2, 0])
```

自定义函数或指定 heightfield 中心时，可以改变局部范围；上述覆盖范围针对默认块布局。

## 两个公开入口

### 入口一：`create_mesh_terrain()`

适合生成一种地形，或把多种地形按网格排列。

```python
vertices, indices = create_mesh_terrain(
    grid_size=(4, 4),
    block_size=(5.0, 5.0),
    terrain_types=None,
    terrain_params=None,
    seed=None,
)
```

| 参数 | 默认值 | 用法 |
|---|---|---|
| `grid_size` | `(4, 4)` | `(行数, 列数)`；使用正整数 |
| `block_size` | `(5.0, 5.0)` | 每块的 `(X 长度, Y 长度)`，米；应为正数 |
| `terrain_types` | `None` | 类型字符串、类型列表，或者自定义函数 |
| `terrain_params` | `None` | 按类型提供参数，例如 `{"wave": {"wave_amplitude": 0.1}}` |
| `seed` | `None` | 给内置 `random_grid` 生成可复现的随机种子 |

`terrain_types` 的行为：

| 写法 | 行为 |
|---|---|
| `"wave"` | 每块都是波浪 |
| `["flat", "wave", "box"]` | 按行遍历，依次循环这三个类型；不是随机抽取 |
| `None` | 依次循环 `flat`、`pyramid_stairs`、`random_grid`、`wave`、`box`、`gap` |
| 自定义函数 | 每块调用这个函数，传入 `block_size` 和对应参数 |

不同块只做位置偏移和网格拼接，**不会自动平滑接缝、焊接顶点或执行布尔并集**。此文件也不提供自动难度课程、比例随机采样或训练时重建地形逻辑。

### 入口二：`create_mesh_heightfield()`

适合把自定义二维高度数组转为有底面和侧壁的地形实体。

```python
vertices, indices = create_mesh_heightfield(
    heightfield,
    extent_x,
    extent_y,
    center_x=0.0,
    center_y=0.0,
    ground_z=0.0,
)
```

| 参数 | 默认值 | 用法 |
|---|---|---|
| `heightfield` | 必填 | 二维 NumPy 数组；建议 `float32`；两个方向都至少有 2 个点 |
| `extent_x` | 必填 | X 方向总长度，米，必须大于 0 |
| `extent_y` | 必填 | Y 方向总长度，米，必须大于 0 |
| `center_x` | `0.0` | 地形中心 X 坐标，米 |
| `center_y` | `0.0` | 地形中心 Y 坐标，米 |
| `ground_z` | `0.0` | 底面高度，米；应严格低于所有地表高度 |

**数组第一维对应 X，第二维对应 Y**。`heightfield[i, j]` 是对应 `(x_i, y_j)` 的 Z 高度。支持非正方形数组。

默认覆盖范围为 `[-extent_x/2, extent_x/2] × [-extent_y/2, extent_y/2]`，与入口一的默认起点不同。

源码会检查二维数组、最小尺寸和正的物理长度，但不会替你检查底面是否低于地表。使用浮点数组；当前实现用 `np.full_like(heightfield, ground_z)` 创建底面，整数数组会截断非整数底面高度。

### 返回值怎么处理？

两个入口都返回：

| 返回值 | 形状与类型 | 意义 |
|---|---|---|
| `vertices` | `(N, 3)`，`float32` | 每个顶点的 X、Y、Z |
| `indices` | `(3T,)`，`int32` | 展平的三角形顶点索引 |

导出时将索引还原为每行一个三角形：

```python
mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=indices.reshape(-1, 3),
    process=False,
)
```

## 全部内置地形参数速查

以下参数放在 `terrain_params[类型名称]` 中。`size` 由主入口的 `block_size` 自动传入，不要再在该字典里传 `size`。

| 类型名称 | 适合生成 | 可调参数 |
|---|---|---|
| `flat` | 无厚度的平整地面 | `height` |
| `pyramid_stairs` | 向中心升高的台阶 | `step_width`、`step_height`、`platform_width` |
| `random_grid` | 随机高度格子 | `grid_width`、`grid_height_range`、`platform_width`（未使用）、`seed` |
| `wave` | 无厚度的连续波浪 | `wave_amplitude`、`wave_frequency`、`resolution` |
| `box` | 地面加中心凸起平台 | `box_height`、`platform_width` |
| `gap` | 中心平台四周留空 | `gap_width`、`platform_width` |
| `heightfield` | 自定义高度图；提供数组时带底面和侧壁 | `heightfield`、`center_x`、`center_y`、`ground_z` |

### `flat`：平面

| 参数 | 默认值 | 作用 |
|---|---|---|
| `height` | `0.0` | 平面 Z 高度，米 |

```python
terrain_types="flat"
terrain_params={"flat": {"height": 0.0}}
```

输出只有 4 个顶点、2 个三角形；没有厚度、侧壁和底面。

### `pyramid_stairs`：向中心升高的台阶

| 参数 | 默认值 | 作用 |
|---|---|---|
| `step_width` | `0.5` | 每圈台阶的水平宽度，米 |
| `step_height` | `0.1` | 台阶高度增量，米 |
| `platform_width` | `1.0` | 中心正方形平台边长，米 |

```python
terrain_types="pyramid_stairs"
terrain_params={
    "pyramid_stairs": {
        "step_width": 0.4,
        "step_height": 0.1,
        "platform_width": 1.5,
    }
}
```

建议宽度和高度为正，平台小于块的短边。台阶圈数由两个方向能容纳的圈数的较小值决定。当前实现将中心平台顶面设为 `(num_steps + 2) * step_height`，不要假定最后一次上平台一定只升高一个台阶。

它由多个箱体网格拼接构成，没有做实体并集；不同箱体可能有共享或内部表面。

### `random_grid`：随机高度方块

| 参数 | 默认值 | 作用 |
|---|---|---|
| `grid_width` | `0.5` | 每个小格子的边长，米 |
| `grid_height_range` | `(-0.15, 0.15)` | 顶面相对 Z=0 的随机高度范围，米 |
| `platform_width` | `None` | 当前未使用，设置它不会创建平坦出生区 |
| `seed` | `None` | 本类型的固定随机种子，可覆盖主入口种子 |

```python
terrain_types="random_grid"
terrain_params={
    "random_grid": {
        "grid_width": 0.5,
        "grid_height_range": (-0.1, 0.2),
    }
}
# 在 create_mesh_terrain() 中另设 seed=42
```

格子数为 `int(size_x / grid_width)`、`int(size_y / grid_width)`；不能整除时，尾部余量不会补齐。每个方向至少应容纳一个格子，高度范围应有序且高于箱体底面 Z=−1。

主入口设置 `seed` 后，每块随机地形通常得到不同、但可复现的子种子；如果在类型参数中设置固定 `seed`，每块会使用同一个种子，得到重复的高度布局。自定义函数的随机性不会自动由主入口管理。

**当前实现的几何限制：**源码只移动顶面的 4 个顶点，侧面使用的重复顶点没有同步移动；高度变化会让顶面与侧面出现接缝。不要直接把它当作封闭 SDF 输入；需要单独处理几何，或改用封闭高度图。

### `wave`：正弦波表面

| 参数 | 默认值 | 作用 |
|---|---|---|
| `wave_amplitude` | `0.3` | 波浪幅度，米 |
| `wave_frequency` | `2.0` | 每个方向跨越多少个半周期；2 表示一个完整周期 |
| `resolution` | `50` | 每个方向采样点数，整数且至少为 2 |

```python
terrain_types="wave"
terrain_params={
    "wave": {
        "wave_amplitude": 0.1,
        "wave_frequency": 2.0,
        "resolution": 60,
    }
}
```

高度公式为：

```text
z = A × sin(f × π × x / size_x) × sin(f × π × y / size_y)
```

频率取整数时，各块边界高度接近零；任意非整数频率不保证所有边界为零。连续拼接仍不保证边界坡度相同。

顶点数为 `resolution²`，三角形数为 `2 × (resolution−1)²`。提高分辨率会让几何更细，也会增加资产大小。

输出仅有地表；需要封闭波浪实体时，使用后面的高度图示例。

### `box`：中央凸起平台

| 参数 | 默认值 | 作用 |
|---|---|---|
| `box_height` | `0.5` | 平台顶面相对地面的高度，米 |
| `platform_width` | `1.5` | 中心正方形平台边长，米 |

```python
terrain_types="box"
terrain_params={"box": {"box_height": 0.4, "platform_width": 1.5}}
```

地面箱体覆盖 Z∈[−1, 0]，平台箱体覆盖 Z∈[−1, box_height]。建议平台高度为正，宽度小于块的短边。这是两个重叠箱体的网格拼接，不是布尔并集。

### `gap`：围绕中心平台的沟壑

| 参数 | 默认值 | 作用 |
|---|---|---|
| `gap_width` | `0.8` | 中心平台与外围地面之间的间隔，米 |
| `platform_width` | `1.2` | 中心正方形平台边长，米 |

```python
terrain_types="gap"
terrain_params={"gap": {"gap_width": 0.6, "platform_width": 1.2}}
```

这是中心平台加外围四条地面，中间留空；不是沿一个方向贯穿地形的直线沟。箱体底面为 Z=−1、顶面为 Z=0，没有沟底。

建议满足：`platform_width + 2 * gap_width < min(block_size)`，保证外围地面有正宽度。

### `heightfield`：自定义高度图块

| 参数 | 默认值 | 作用 |
|---|---|---|
| `heightfield` | `None` | 二维浮点 NumPy 数组；未提供时退化为无厚度平面 |
| `center_x` | `None` | 默认取块 X 长度的一半，使块从 X=0 开始 |
| `center_y` | `None` | 默认取块 Y 长度的一半，使块从 Y=0 开始 |
| `ground_z` | `0.0` | 底面高度，米；应低于最低地表 |

```python
terrain_types="heightfield"
terrain_params={
    "heightfield": {
        "heightfield": height.astype(np.float32),
        "ground_z": float(height.min()) - 0.3,
    }
}
```

高度图水平尺寸由 `block_size` 决定。当前字典会被所有同类型块复用，因此各块使用同一张高度图；每块需要不同高度图时，可自行循环生成，或用自定义函数。

## 常用完整示例

### 单块地形，保存到 Assets/procedural

```python
from pathlib import Path
import numpy as np
import trimesh
from terrain_generator import create_mesh_terrain

vertices, indices = create_mesh_terrain(
    grid_size=(1, 1),
    block_size=(6.0, 4.0),
    terrain_types="wave",
    terrain_params={"wave": {"wave_amplitude": 0.1, "resolution": 60}},
)
mesh = trimesh.Trimesh(vertices, indices.reshape(-1, 3), process=False)

# 需要 Y-up 时旋转；需要 Z-up 时省略这行。
mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
output = Path(__file__).resolve().parent / "Assets" / "procedural" / "terrain.obj"
output.parent.mkdir(parents=True, exist_ok=True)
mesh.export(output)
```

### 混合多块地形

```python
vertices, indices = create_mesh_terrain(
    grid_size=(2, 3),
    block_size=(6.0, 4.0),
    terrain_types=["flat", "wave", "box"],
    terrain_params={
        "flat": {"height": 0.0},
        "wave": {"wave_amplitude": 0.1, "resolution": 40},
        "box": {"box_height": 0.3, "platform_width": 1.5},
    },
)
```

排列为两行，每行依次为 flat、wave、box。`seed` 不会改变类型顺序。

### 封闭的起伏地形，适合后续距离场流程

```python
import numpy as np
from terrain_generator import create_mesh_heightfield

x = np.linspace(-3, 3, 61)
y = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
height = (0.1 * np.sin(X) * np.cos(Y)).astype(np.float32)

vertices, indices = create_mesh_heightfield(
    heightfield=height,
    extent_x=6.0,
    extent_y=4.0,
    ground_z=float(height.min()) - 0.3,
)
```

这块地形默认以原点为水平中心。将 `height` 替换为全零浮点数组可生成有厚度的平整地板。封闭地形仍需在实际资产上检查法线、退化面和与 OBJ 的穿插关系。

### 自定义生成函数

函数接收 `(size_x, size_y)`，返回 `(vertices, flattened_indices)`。尽量使用局部 `[0, size_x] × [0, size_y]` 坐标，使主入口能按块排列。

```python
def custom_surface(size, amplitude=0.1, resolution=32):
    x = np.linspace(0, size[0], resolution)
    y = np.linspace(0, size[1], resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    height = (amplitude * np.sin(X) * np.cos(Y)).astype(np.float32)
    return create_mesh_heightfield(
        height, size[0], size[1],
        center_x=size[0] / 2,
        center_y=size[1] / 2,
        ground_z=float(height.min()) - 0.3,
    )

vertices, indices = create_mesh_terrain(
    grid_size=(1, 2),
    block_size=(6.0, 4.0),
    terrain_types=custom_surface,
    terrain_params={custom_surface: {"amplitude": 0.2}},
)
```

当前实现按 `terrain_name` 查找参数，因此自定义函数应作为字典键，而不是其名称字符串。自定义函数需要自行管理随机种子。

## 导入 OBJ，生成布尔场景资产

### 直接运行独立场景脚本

`scene_creator.py` 直接使用 `terrain_generator.py`，不依赖 `creator1.py`。它生成一块 6×4 米的上升阶梯，将 OBJ 放到中心平台，使用 Trimesh + Manifold 做实体布尔并集。`Assets/output/stairs_scene.obj` 是同位置的拼接对照，`Assets/output/stairs_scene_bool.obj` 是成功完成布尔后的结果；同名输出覆盖。

当前 `mimickit-newton` 环境已验证 Trimesh 5.1.1 + manifold3d 3.5.4 可执行布尔并集。其他环境需要 `python -m pip install trimesh manifold3d numpy`。脚本先检查后端，不可用时立即停止。

**bunny 修复副本：** 原始 `Assets/origin/bunny.obj` 底部附近有 4 个开口，共 42 条边界边。独立运行 `repair_bunny.py` 会添加 34 个补洞三角形，另存为 `Assets/origin/bunny_repaired.obj`，保留原始文件。补面采用开口轮廓三角化，保留原有三维顶点和表面；副本已通过封闭性和 Manifold 输入检查。场景脚本不自动补洞或重建；原始 bunny 即使将洞埋入平台仍不能作 Manifold 输入，`check_volume=False` 也不能绕过这一要求。

当前默认模型为本目录 `Assets/origin/spot_triangulated_good.obj` 奶牛，在仓库根目录运行：

```powershell
python tests/Scene_Test/scene_creator.py

# 单独修复 bunny，并用修复副本测试场景。
python tests/Scene_Test/repair_bunny.py
python tests/Scene_Test/scene_creator.py --obj tests/Scene_Test/Assets/origin/bunny_repaired.obj --scale 5 --obj-up-axis Y

# 或指定自己的 OBJ 路径；相对路径以运行目录为基准。
python tests/Scene_Test/scene_creator.py --obj "D:/Models/chair.obj" --scale 1 --obj-up-axis Z

# 假设模型单位为厘米、向上轴为 Y，并在地面上旋转 90 度。
python tests/Scene_Test/scene_creator.py --obj "D:/Models/chair.obj" --scale 0.01 --obj-up-axis Y --yaw 90

# 调整自动放置时的下沉深度，单位为米；0 表示与顶面齐平。
python tests/Scene_Test/scene_creator.py --sink 0.02

# 手动指定 OBJ 底部中心的位置；此处采用 Z-up，不再叠加 --sink。
python tests/Scene_Test/scene_creator.py --position 1.0 2.0 0.1
```

不提供 `--position` 时，OBJ 的包围盒底部中心放在中央平台，并默认下沉 0.02 米，约为 `(3, 2, 0.38)` 米。平台顶面约为 0.4 米，大小为 1×1 米。奶牛四只脚均与平台相交；实际求交得到四块正体积区域，整体并集是一个封闭连通网格。可用 `--sink` 控制嵌入深度；显式 `--position` 是最终底部中心坐标，不再叠加下沉。输出默认 Y-up，可以用 `--up-axis Z` 保留 Z-up。放置坐标始终按 Z-up 填写，与输出向上轴无关。

当前奶牛原始向上轴是 Y，原始 X/Y/Z 尺寸约 `0.943×1.690×1.718`。脚本默认 `--obj-up-axis Y --scale 0.5`，转成 Z-up 后，宽、深、高约为 `0.472×0.859×0.845` 米，完整落在中央平台范围内。脚本顶部的 `DEFAULT_OBJ_SCALE`、`DEFAULT_OBJ_UP_AXIS` 和 `DEFAULT_OBJ_SINK` 可以调整默认大小、朝向和嵌入深度。换用修复后的 bunny 时应显式设置 `--scale 5 --obj-up-axis Y`；OBJ 文件本身不记录统一的单位和向上轴。

脚本按步骤拆成函数：

| 函数 | 作用 |
|---|---|
| `generate_stairs()` | 独立生成阶梯；在这里修改阶梯尺寸、台阶参数 |
| `load_and_place_obj()` | 导入 OBJ，统一向上轴，缩放、旋转，再对齐底部中心 |
| `combine_meshes()` | 追加顶点和三角面，并调整 OBJ 面的顶点编号 |
| `prepare_solid()` | 仅焊接同位置的重复顶点，不补洞或修复几何 |
| `union_stair_boxes()` | 按当前生成器每箱体 12 个连续三角形恢复独立箱体，焊接后做地形并集 |
| `boolean_union_meshes()` | 检查实体封闭性，再调用 Manifold 做并集并验证输出 |
| `simplify_mesh_manifold()` | 对最终布尔实体副本做小容差简化；使用 Mesh64，不平滑或补洞 |
| `report_simplification()` | 比较三角形数、最小角、瘦长面、封闭性、连通性和体积 |
| `validate_output_paths()` | 导出前拒绝输入和不同输出指向同一文件 |
| `export_scene_mesh()` | 转换整个场景的向上轴并导出 |
| `check_boolean_backend()` | 检查当前 Python 环境中的 Manifold 后端 |
| `create_scene()` | 保存摆放对照和原布尔结果，可选简化分支另存 OBJ；返回原布尔路径 |

数组操作旁附有中文注释：顶点数组每行是 `[x, y, z]`；面数组每行是三个顶点编号。拼接对照只追加数组，保留内部面；布尔输出则裁剪相交部分，删除并集内部面。布尔需要输入为有效封闭实体，不保留完整材质与纹理，也不会自动填补实际存在的物体间隙。

### 可选 Manifold 小容差简化

默认关闭。启用后先保存原布尔结果，再对其副本调用 `Manifold.simplify()`，**不会用简化结果替换原布尔文件**。三个默认输出为：

| 结果 | 默认文件 |
|---|---|
| 原始拼接对照 | `Assets/output/stairs_scene.obj` |
| 原布尔并集 | `Assets/output/stairs_scene_bool.obj` |
| 可选简化副本 | `Assets/output/stairs_scene_bool_simplified.obj` |

```powershell
# 开启：默认表面容差为 0.001 米（1 毫米）。
python tests/Scene_Test/scene_creator.py --simplify

# 使用更小的 0.0001 米（0.1 毫米）容差。
python tests/Scene_Test/scene_creator.py --simplify --simplify-tolerance 0.0001

# 显式关闭；未提供 --simplify 时也默认关闭。
python tests/Scene_Test/scene_creator.py --no-simplify

# 自定义原输出，简化文件自动派生为 Assets/output/custom_bool_simplified.obj。
python tests/Scene_Test/scene_creator.py --simplify --output tests/Scene_Test/Assets/output/custom_bool.obj

# 或显式指定另一份独立简化 OBJ。
python tests/Scene_Test/scene_creator.py --simplify --simplified-output tests/Scene_Test/Assets/output/custom_simplified.obj

# 按需使用简化副本烘焙；SDF 默认输入仍是原布尔结果。
python tests/Scene_Test/sdf_baker.py --input tests/Scene_Test/Assets/output/stairs_scene_bool_simplified.obj
```

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--simplify` / `--no-simplify` | 关闭 | 控制独立简化分支；关闭时不调用简化、不更新简化文件 |
| `--simplify-tolerance` | `0.001` 米 | 允许的表面容差，必须是有限正值；仅开启时使用，不是目标面数 |
| `--simplified-output` | 原布尔输出文件名追加 `_simplified` | 简化副本路径；只能与开启分支一起使用 |

输入和所有输出在写文件前检查互不覆盖，包括相对路径别名、符号链接和硬链接。各输出可以覆盖自己的上次结果，但不能指向输入或另一种输出。未开启分支却指定 `--simplified-output` 会报错；开启时容差小于 Manifold 实体实际最小容差也会明确报错，不会自动放大容差。

简化使用 `Mesh64 → Manifold.simplify() → to_mesh64()`，避免额外的 float32 转换损失，原 mesh 不被修改。它允许表面在容差内变化，不保证得到均匀三角形或消除全部瘦长面，也不做全局平滑、补洞或重网格。终端同时报告两份实体的三角形数、最小内角、小于 1° 的面数、封闭性、连通分量和体积变化；体积变化不能代替距离误差。

简化失败时原布尔输出仍保留，并明确报错。已有简化文件不会被主动删除，因此失败后同名文件可能是上次的结果。原件和简化副本采用相同的输出向上轴。

回归验证：`python -m unittest discover -s tests/Scene_Test -p test_scene_creator.py`。

### 手动拼接片段

以下片段接在已经创建 `mesh` 的代码后，且在整个场景转 Y-up 之前运行。示例假设 OBJ 也采用 Z-up，尺寸单位已统一为米。

```python
asset_dir = Path(__file__).resolve().parent / "Assets"
obj = trimesh.load(asset_dir / "origin" / "input.obj", force="mesh")
obj.apply_scale(1.0)
obj.apply_translation([2.0, 1.0, 0.5])  # 按实际原点和尺寸调整
scene_mesh = trimesh.util.concatenate([mesh, obj])

# 场景几何统一坐标后，再对全部几何做一次向上轴转换。
scene_mesh.apply_transform(
    trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
)
(asset_dir / "output").mkdir(parents=True, exist_ok=True)
scene_mesh.export(asset_dir / "output" / "scene.obj")
```

这一步将多个几何拼为一个网格资产，不会裁剪到长方体范围、自动贴地、修补孔洞或处理相交实体的布尔并集。若要保留复杂多材质和场景层级，应另用 Trimesh Scene 等场景表示。

## SDF 烘焙与重建

`sdf_baker.py` 独立读取已有布尔 mesh，不依赖场景创建脚本，也不启动 viewer。使用当前 `mimickit-newton` 环境的 Warp、Trimesh、NumPy、Matplotlib；需要 CUDA。

```powershell
conda activate mimickit-newton
python tests/Scene_Test/sdf_baker.py

# 自选封闭资产、采样精度和水平切片高度；单位均为米。
python tests/Scene_Test/sdf_baker.py --input tests/Scene_Test/Assets/output/stairs_scene_bool.obj --voxel-size 0.01 --slice-height 0.41
```

| 参数 | 默认值 | 控制内容 |
|---|---|---|
| `--input` | `Assets/output/stairs_scene_bool.obj` | 已经生成的封闭布尔 mesh，保持原文件不变 |
| `--voxel-size` | `0.01` | 距离采样点间距；越小越细，三维网格内存约按其倒数的三次方增加 |
| `--padding` | `0.05` | 包围盒每侧外扩，至少为采样间距的两倍，避免重建表面被边界裁开 |
| `--output-dir` | 脚本所在目录的 `Assets/output` | 输出目录，文件名由输入文件名派生，同名覆盖 |
| `--device` | `cuda:0` | CUDA 设备；不可用时停止，不自动改为 CPU |
| `--up-axis` | `Y` | 输入坐标的向上轴；只用于元数据和切片，不旋转模型 |
| `--slice-height` | `0.41` | 水平切片沿向上轴的高度；当前平台约高 0.4 米 |

默认在 `Assets/output` 保存四份结果：

| 文件 | 如何使用 |
|---|---|
| `stairs_scene_bool_sdf.npz` | 完整 float32 距离数组和坐标元数据；方便 NumPy 读取与分析 |
| `stairs_scene_bool_sdf.nvdb` | NanoVDB SDF，可重新加载为 `wp.Volume`；查看器需支持 NanoVDB，不能假定普通 OBJ 查看器能打开 |
| `stairs_scene_bool_sdf_reconstructed.obj` | 从同一份 SDF 的零等值面提取的重建 mesh，使用现有 OBJ 查看器打开 |
| `stairs_scene_bool_sdf_slices.png` | 三方向切片：红色为内部负距离，蓝色为外部正距离，黑线为零等值线 |

### 读取保存的 SDF

```python
import numpy as np
import warp as wp
from pathlib import Path

asset_root = Path("tests/Scene_Test/Assets")  # 示例从仓库根目录运行。
assets = asset_root / "output"
with np.load(assets / "stairs_scene_bool_sdf.npz", allow_pickle=False) as data:
    values = data["values"]       # [X编号, Y编号, Z编号]，不会因 up-axis 换序。
    origin = data["origin"]       # 数组 [0,0,0] 的世界位置。
    spacing = float(data["voxel_size"])
    # values[i,j,k] 对应 origin + spacing * [i,j,k]；单位为米，内部为负。

wp.config.kernel_cache_dir = str(asset_root.resolve() / ".warp_cache")
wp.init()
with open(assets / "stairs_scene_bool_sdf.nvdb", "rb") as file:
    volume = wp.Volume.load_from_nvdb(file, device="cuda:0")
# Warp kernel 内：先 volume_world_to_index，再 volume_sample_f(..., wp.Volume.LINEAR)。
```

NPZ 还记录 `shape`、`up_axis`、`axis_order="XYZ"`、`unit="m"`、`sign_convention="negative_inside"`、来源路径和格式版本。NanoVDB 会将分配区域补齐到 8 的倍数；**仅 NPZ 定义的采样域具有完整距离意义**，域外和补齐部分使用正背景值，不是该处真实距离。在线性插值时也应保持在有效采样域内。

### 精度和检查

脚本按最近表面距离与绕数烘焙，不做全局平滑、距离偏移或补洞；使用 Warp Marching Cubes 重建，只焊接重复点、移除零面积面。SDF 保存后会检查 NPZ 全数组一致性，以及 NanoVDB 网格点和域内插值的坐标一致性。

默认场景的网格为约 `611×139×411`，单个 SDF 数组约 133 MiB。Marching Cubes 中间数组比 SDF 本身大很多，脚本会报告保守显存估计；不足时停止并提示增大采样间距，不会偷偷降低精度。

终端报告重建的封闭性、朝向、连通分量、体积及双向各 2 万点的表面误差。误差按表面积采样，平均值可能被大片平面主导；采样最大值也不是严格的 Hausdorff 上界，不能作为脚部精度保证。体素重建可能改变锐角和细节，也不保证消除所有瘦长三角形。

中央竖切片未必穿过奶牛的四只脚；切片上身体与平台分离不表示整个三维模型悬空。水平切片默认位于平台上方，用于观察四只脚的内部区域。图片颜色只显示 ±5 个采样间距，保存的距离数值未截断。

输入、参数或显存检查失败时停止；如果重建阶段失败，已经保存的 SDF 和切片保留，已导出的异常 OBJ 用于诊断，并明确报错。

## 输出格式速查

| 格式 | 用途与注意事项 |
|---|---|
| `.obj` | 第一版推荐；广泛兼容；材质可能额外产生 `.mtl` 和纹理文件 |
| `.ply` | 几何和顶点颜色；是否包含颜色取决于 mesh 是否设置颜色 |
| `.stl` | 三角形几何；通常不包含材质和颜色 |
| `.glb` | 单文件场景/网格，可包含材质和纹理 |
| `.gltf` | 可包含材质和纹理，可能产生多个关联文件 |
| `.off` | 简单几何交换格式 |

使用 `mesh.export(output_path)` 按扩展名导出。生成器只提供几何，不会凭空生成材质、纹理或 UV。

## 内部函数完整索引

以下函数以 `_` 开头，属于实现辅助函数；优先使用两个公开入口。直接调用时必须自己传 `size`、做位置偏移和导出。

| 函数 | 参数与默认值 | 作用 |
|---|---|---|
| `_create_box` | `size` 三维全尺寸；`position=None` | 构造箱体，默认以原点为中心；返回面索引为 `(12, 3)` |
| `_flat_terrain` | `size`；`height=0.0` | 平面 |
| `_pyramid_stairs_terrain` | `size`；`step_width=0.5`；`step_height=0.1`；`platform_width=1.0` | 台阶 |
| `_random_grid_terrain` | `size`；`grid_width=0.5`；`grid_height_range=(-0.15, 0.15)`；`platform_width=None`；`seed=None` | 随机格子 |
| `_wave_terrain` | `size`；`wave_amplitude=0.3`；`wave_frequency=2.0`；`resolution=50` | 波浪表面 |
| `_box_terrain` | `size`；`box_height=0.5`；`platform_width=1.5` | 中央平台 |
| `_gap_terrain` | `size`；`gap_width=0.8`；`platform_width=1.2` | 环绕沟壑 |
| `_heightfield_terrain` | `size`；`heightfield=None`；`center_x=None`；`center_y=None`；`ground_z=0.0` | 主入口使用的高度图适配器 |
| `_combine_meshes` | `meshes`，元素为 `(vertices, faces)` | 拼接顶点并修正面索引偏移，返回展平索引 |
| `_to_newton_mesh` | `vertices, indices` | 仅将数组转换为 `float32` / `int32`，不构造 Newton Mesh，也不自动展平索引 |

箱体按面复制顶点以保持锐边，因此未经焊接时，拓扑检查可能认为边不共享；视觉上闭合不等于拓扑上已闭合。

## 常见问题速查

| 现象 | 检查和处理 |
|---|---|
| 地形竖着 | 检查向上轴；Y-up 查看器用导出前绕 X 轴 −90° |
| 资产找不到 | 导入模型位于 `Assets/origin`，纯地形输出到 `Assets/procedural`，场景/SDF 输出到 `Assets/output`；终端打印绝对路径 |
| 高度、尺寸不对 | OBJ 不约定单位；统一米制，并区分高度幅度与总体宽度 |
| 未识别的地形类型 | 名称必须是七种内置字符串之一，或传入自定义函数 |
| 提示不认识某个参数 | 只在对应地形字典中传该类型支持的参数 |
| 索引形状不对 | 使用 `indices.reshape(-1, 3)` 构造 Trimesh |
| 多块接缝不平 | 生成器仅拼接；调整边界高度或自行做过渡 |
| 随机格子每块重复 | 检查是否在 `random_grid` 参数中固定了同一个 `seed` |
| 高度图平板退化 | `ground_z` 要低于最低高度；不要让顶面与底面重合 |
| 波浪有表面但没有体积 | 内置 wave 是表面网格；换用封闭高度图入口 |
| 合并 OBJ 后不能直接作统一 SDF | 检查封闭性、法线、重叠和内部表面；拼接不等于实体并集 |

后续给 Newton/Warp 使用场景时，保持一致的坐标约定；若资产已经转成 Y-up，应整体转回 Z-up，或在导入时明确处理。
