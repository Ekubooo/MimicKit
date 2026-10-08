"""通过 CLI 生成地形，保存到 Assets/<地形类型>.obj，同名文件覆盖。"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from terrain_generator import create_mesh_terrain


TERRAIN_TYPES = (
    "flat", "pyramid_stairs", "random_grid", "wave", "box", "gap", "heightfield",
)
ASSET_DIR = Path(__file__).resolve().parent / "Assets"
BLOCK_SIZE = (6.0, 4.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain", choices=(*TERRAIN_TYPES, "all"), default="wave",
        help="地形类型，默认 wave；all 一次导出全部七种类型",
    )
    parser.add_argument(
        "--up-axis", type=str.upper, choices=("Y", "Z"), default="Y",
        help="导出向上轴，默认 Y；与 Newton 场景一致时使用 Z",
    )
    parser.add_argument("--seed", type=int, help="random_grid 的随机种子")
    parser.add_argument(
        "--heightfield", type=Path, metavar="FILE.npy",
        help="heightfield 使用的二维高度数组；省略时生成有厚度的平整地板",
    )
    parser.add_argument(
        "--ground-z", type=float,
        help="heightfield 在轴转换前的底面高度；默认比最低地表低 0.3 米",
    )
    args = parser.parse_args()
    selected = TERRAIN_TYPES if args.terrain == "all" else (args.terrain,)
    if "heightfield" not in selected and (
        args.heightfield is not None or args.ground_z is not None
    ):
        parser.error("--heightfield 和 --ground-z 仅用于 --terrain heightfield 或 all")

    # 保留原脚本的波浪参数，其余类型采用生成器默认参数。
    terrain_params = {
        "wave": {"wave_amplitude": 0.75, "wave_frequency": 2.0, "resolution": 30}
    }
    if "heightfield" in selected:
        try:
            height = (
                np.asarray(np.load(args.heightfield, allow_pickle=False), dtype=np.float32)
                if args.heightfield is not None else np.zeros((30, 30), dtype=np.float32)
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"无法读取高度图：{error}")
        if height.ndim != 2 or min(height.shape) < 2 or not np.isfinite(height).all():
            parser.error("高度图必须是两个方向至少各有 2 个点的二维有限数值数组")
        ground_z = float(height.min()) - 0.3 if args.ground_z is None else args.ground_z
        if not np.isfinite(ground_z) or ground_z >= float(height.min()):
            parser.error("--ground-z 必须是有限数值，且严格低于最低地表高度")
        terrain_params["heightfield"] = {"heightfield": height, "ground_z": ground_z}

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for terrain_type in selected:
        vertices, indices = create_mesh_terrain(
            grid_size=(1, 1),
            block_size=BLOCK_SIZE,
            terrain_types=terrain_type,
            terrain_params=terrain_params,
            seed=args.seed,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=indices.reshape(-1, 3), process=False,
        )
        if args.up_axis == "Y":
            # (x, y, z) -> (x, z, -y)，保持右手坐标系。
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            )
        output_path = ASSET_DIR / f"{terrain_type}.obj"
        mesh.export(output_path)
        print(f"已导出：{output_path}（{args.up_axis}-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
