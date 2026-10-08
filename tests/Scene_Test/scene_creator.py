"""生成阶梯、放置一个 OBJ，并导出场景网格；不依赖 creator1.py。"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from terrain_generator import create_mesh_terrain


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
DEFAULT_OBJ_PATH = ASSET_DIR / "bunny.obj"  # 将自己的 OBJ 放在这里，或用 --obj 指定路径。
DEFAULT_OUTPUT_PATH = ASSET_DIR / "stairs_scene.obj"  # 重复运行覆盖同名文件。
DEFAULT_OBJ_SCALE = 5.0  # 当前 bunny 原始尺寸约 0.15 米；放大 5 倍后约高 0.77 米。
DEFAULT_OBJ_UP_AXIS = "Y"  # 当前 bunny 是 Y-up；OBJ 不提供统一 up 信息，不同模型需手动指定。


def generate_stairs():
    """生成一块 Z-up 阶梯；在本函数中修改阶梯尺寸和参数。"""
    vertices, indices = create_mesh_terrain(
        grid_size=(1, 1),                   # (行数, 列数)，这里生成一块。
        block_size=(6.0, 4.0),              # X 长度和 Y 长度，米。
        terrain_types="pyramid_stairs",     # 四周低、中心高的同心阶梯。
        terrain_params={
            "pyramid_stairs": {
                "step_width": 0.25,         # 每圈台阶的水平宽度，米。
                "step_height": 0.05,        # 每圈高度增量，米；当前源码应使用正值。
                "platform_width": 1.0,      # 中心正方形平台的边长，米。
            }
        },
    )

    # vertices 是 (N, 3) 数组：N 个顶点，每一行是一个 [x, y, z]。
    # indices 是一维数组，如 [0, 1, 2, 2, 3, 0]：每三个数字表示一个三角形。
    # reshape(-1, 3) 将它变成 [[0, 1, 2], [2, 3, 0]]；-1 表示自动计算行数。
    faces = indices.reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_and_place_obj(obj_path, terrain, scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                       obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None):
    """先统一 OBJ 坐标，再把它的底部中心移到指定位置。

    scale：尺寸倍率，当前 bunny 默认 5；例如厘米模型转米使用 0.01。
    yaw_deg：绕竖直 Z 轴旋转的角度，单位为度。
    obj_up_axis：原始 OBJ 的向上轴，Y 或 Z；当前 bunny 默认 Y。
    position：目标底部中心 [x, y, z]，单位为米，采用 Z-up；None 放在阶梯顶端中心。
    """
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"找不到 OBJ：{obj_path}；默认读取 Assets/{DEFAULT_OBJ_PATH.name}，或使用 --obj 指定文件。")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError("请提供 .obj 文件")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须是大于零的有限数值")
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw_deg 必须是有限数值")
    obj_up_axis = obj_up_axis.upper()
    if obj_up_axis not in ("Y", "Z"):
        raise ValueError("obj_up_axis 必须是 Y 或 Z")

    obj = trimesh.load(obj_path, force="mesh", process=False)
    if len(obj.vertices) == 0 or len(obj.faces) == 0 or not np.isfinite(obj.vertices).all():
        raise ValueError("OBJ 必须包含有效的顶点和三角面")

    # 整个放置过程统一使用 Z-up：X、Y 在地面上，Z 表示高度。
    # 按指定轴转换：如果原始 OBJ 为 Y-up，绕 X 轴 +90° 转为 Z-up。
    if obj_up_axis == "Y":
        obj.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    obj.apply_scale(scale)
    obj.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])
    )

    # bounds 是 (2, 3) 数组：第一行 [最小X, 最小Y, 最小Z]，第二行是各轴最大值。
    # 这些数已经包含刚才的旋转和缩放效果。
    x_min, y_min, z_min = obj.bounds[0]
    x_max, y_max, _ = obj.bounds[1]  # 下划线表示这里不需要最大 Z。

    # 用包围盒定义“底部中心”：X、Y 取中点，Z 取最低点。
    # 模型原点不一定在脚底，因此直接移动模型原点可能造成悬空或埋入地面。
    bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])

    if position is None:
        # 当前阶梯的最高位置就是中央平台。
        # 默认底部中心为 [3, 2, 约0.4]；根据实际生成的网格计算，避免手写高度。
        ground_min_x, ground_min_y, _ = terrain.bounds[0]
        ground_max_x, ground_max_y, platform_z = terrain.bounds[1]
        target = np.array([
            (ground_min_x + ground_max_x) / 2,
            (ground_min_y + ground_max_y) / 2,
            platform_z,
        ])
    else:
        # 例如 [1.0, 2.0, 0.1]；自选位置时，Z 要按该处台阶高度填写。
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("position 必须是三个有限数值 [x, y, z]")

    # 数组相减逐项进行：[目标X-当前X, 目标Y-当前Y, 目标Z-当前Z]。
    # 例如底部中心为 [0, 0, -0.5]，目标 [3, 2, 0.4]，移动量就是 [3, 2, 0.9]。
    translation = target - bottom_center
    obj.apply_translation(translation)  # 给物体的每一个顶点加上同一个移动量。
    return obj


def combine_meshes(terrain, obj):
    """手动拼接两套顶点和三角面，便于理解数组和索引的关系。"""
    # vstack 沿竖直方向追加数组的行，不改变任何顶点的坐标。
    # 地形有 N 个顶点、OBJ 有 M 个顶点，合并后就是 (N+M, 3)。
    vertices = np.vstack([terrain.vertices, obj.vertices])

    # 两个原始模型都从顶点编号 0 开始。
    # 追加 OBJ 后，它的顶点在合并数组中从编号 N 开始，所以面索引也要整体加 N。
    # 例如地形有 100 个顶点，OBJ 的面 [0, 1, 2] 就改成 [100, 101, 102]。
    vertex_offset = len(terrain.vertices)
    obj_faces = obj.faces + vertex_offset

    # 地形面索引保持原样，再追加已经调整编号的 OBJ 面索引。
    faces = np.vstack([terrain.faces, obj_faces])
    # 这里拼接几何，不执行实体布尔并集；各部分的内部面仍然存在。
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def create_scene(obj_path=DEFAULT_OBJ_PATH, output_path=DEFAULT_OUTPUT_PATH,
                 scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                 obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None, up_axis="Y"):
    """生成、放置、拼接、导出；返回输出路径。所有位置参数均按 Z-up 填写。"""
    up_axis = up_axis.upper()
    if up_axis not in ("Y", "Z"):
        raise ValueError("up_axis 必须是 Y 或 Z")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".obj":
        raise ValueError("输出路径必须以 .obj 结尾")
    if Path(obj_path).resolve() == output_path.resolve():
        raise ValueError("输入 OBJ 和输出场景不能是同一个文件")

    terrain = generate_stairs()
    obj = load_and_place_obj(
        obj_path=obj_path, terrain=terrain, scale=scale,
        yaw_deg=yaw_deg, obj_up_axis=obj_up_axis, position=position,
    )
    scene = combine_meshes(terrain, obj)

    # 放置完成后才转换整个场景的向上轴，确保地形和物体始终一起旋转。
    if up_axis == "Y":
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ_PATH, help=f"输入 OBJ；默认 Assets/{DEFAULT_OBJ_PATH.name}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 OBJ；默认 Assets/stairs_scene.obj")
    parser.add_argument("--scale", type=float, default=DEFAULT_OBJ_SCALE, help="模型尺寸倍率；当前 bunny 默认 5，厘米转米用 0.01")
    parser.add_argument("--yaw", type=float, default=0.0, help="模型绕 Z 轴旋转的角度，单位：度")
    parser.add_argument("--obj-up-axis", type=str.upper, choices=("Y", "Z"), default=DEFAULT_OBJ_UP_AXIS, help="输入 OBJ 的向上轴；当前 bunny 默认 Y")
    parser.add_argument("--up-axis", type=str.upper, choices=("Y", "Z"), default="Y", help="输出场景的向上轴")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="OBJ 底部中心的位置，Z-up、米；默认中央平台顶面")
    args = parser.parse_args()
    try:
        output = create_scene(
            obj_path=args.obj,
            output_path=args.output,
            scale=args.scale,
            yaw_deg=args.yaw,
            obj_up_axis=args.obj_up_axis,
            position=args.position,
            up_axis=args.up_axis,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"已导出：{output.resolve()}（{args.up_axis}-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
