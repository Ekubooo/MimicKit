# 地形生成器：用法与参数速查

本目录的 `terrain_generator.py` 是 Newton 1.0.0 的独立地形生成源码。本指南按**本目录实际代码**编写。它生成顶点和三角面；资产导出由 `creator1.py` 中的 Trimesh 完成。

快速跳转：

- [生成与运行](#先生成一个资产)
- [坐标与摆放](#坐标为什么看起来竖着)
- [两个公开入口](#两个公开入口)
- [七种地形参数速查](#全部内置地形参数速查)
- [常用示例](#常用完整示例)
- [导入 OBJ](#导入-obj拼接成场景资产)
- [输出格式](#输出格式速查)
- [全部辅助函数](#内部函数完整索引)
- [常见问题](#常见问题速查)

## 先生成一个资产

在仓库根目录运行：

```powershell
conda activate mimickit-newton
python tests/Scene_Test/creator1.py --terrain wave
```

当前输出为 **`tests/Scene_Test/Assets/<类型名>.obj`**，例如 `wave.obj`。同一类型重复运行会覆盖原文件，不会添加编号。输出路径相对于脚本定位，因此从其他工作目录运行也会写入同一个 Assets 目录。

当前脚本生成一块 6×4 米的波浪地形，幅度为 0.75 米，每个方向有 30 个采样点；导出默认采用 **Y-up**。脚本不启动查看器。

### CLI 速查

```powershell
# 选择一种地形
python tests/Scene_Test/creator1.py --terrain pyramid_stairs

# 一次生成全部七种类型，各自保存为一个 OBJ
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
| `--terrain` | `wave` | 七种内置类型之一；`all` 导出全部类型 |
| `--up-axis` | `Y` | `Y` 或 `Z`，也接受小写 |
| `--seed` | 不固定 | 控制 `random_grid` 的随机布局 |
| `--heightfield` | 不读取文件 | 仅用于 `heightfield` 或 `all`；文件路径相对于运行目录 |
| `--ground-z` | 最低地表高度减 0.3 米 | 仅用于 `heightfield` 或 `all`；底面必须低于最低地表 |

`heightfield` 未传数组时，脚本会提供一张 30×30 的全零浮点高度图，生成 0.3 米厚的平整地板。这是 CLI 的预设；生成器本身不传高度图时仍返回无厚度平面。除波浪保留原脚本参数外，其余类型采用生成器默认参数。修改 `BLOCK_SIZE` 可调整所有类型的尺寸，修改 `terrain_params` 可设置下面列出的类型参数。

## 目录、依赖与处理流程

```text
Scene_Test/
├── terrain_generator.py   # 生成器源码
├── creator1.py            # CLI、坐标转换、资产导出
├── README.md              # 本指南
└── Assets/
    ├── wave.obj           # 波浪资产
    └── <类型名>.obj       # 其他类型的资产
```

| 部分 | 依赖 | 作用 |
|---|---|---|
| `terrain_generator.py` | NumPy；标准库 `typing` | 生成几何、排列地形块、拼接索引 |
| `creator1.py` | NumPy、Trimesh；标准库 `pathlib`、`argparse` | CLI、生成 mesh、旋转、写文件 |

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

### 单块地形，保存到 Assets

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
output = Path(__file__).resolve().parent / "Assets" / "terrain.obj"
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

## 导入 OBJ，拼接成场景资产

以下片段接在已经创建 `mesh` 的代码后，且在整个场景转 Y-up 之前运行。示例假设 OBJ 也采用 Z-up，尺寸单位已统一为米。

```python
asset_dir = Path(__file__).resolve().parent / "Assets"
obj = trimesh.load(asset_dir / "input.obj", force="mesh")
obj.apply_scale(1.0)
obj.apply_translation([2.0, 1.0, 0.5])  # 按实际原点和尺寸调整
scene_mesh = trimesh.util.concatenate([mesh, obj])

# 场景几何统一坐标后，再对全部几何做一次向上轴转换。
scene_mesh.apply_transform(
    trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
)
scene_mesh.export(asset_dir / "scene.obj")
```

这一步将多个几何拼为一个网格资产，不会裁剪到长方体范围、自动贴地、修补孔洞或处理相交实体的布尔并集。若要保留复杂多材质和场景层级，应另用 Trimesh Scene 等场景表示。

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
| 资产找不到 | 当前输出固定在本目录 `Assets`，终端会打印绝对路径 |
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
