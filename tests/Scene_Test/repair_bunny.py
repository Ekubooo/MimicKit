"""单独修复 bunny 底部的四个开口，另存副本；场景脚本不会自动调用本文件。"""

from pathlib import Path

import manifold3d
import numpy as np
import trimesh


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
ORIGIN_DIR = ASSET_DIR / "origin"  # 导入模型及已整理在此处的修复副本。
INPUT_PATH = ORIGIN_DIR / "bunny.obj"
OUTPUT_PATH = ORIGIN_DIR / "bunny_repaired.obj"


def find_boundary_loops(mesh):
    """返回开口的顶点编号闭环，方向与原表面的边相反，方便正确补面。"""
    # 一条边只被一个三角形使用，说明它位于开口；正常实体边应被两个面共享。
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    if np.any(counts > 2):
        raise ValueError("输入存在非流形边，不能只通过补底面修复")
    boundary = mesh.edges[counts[mesh.edges_unique_inverse] == 1]
    # 原面使用 a→b，新补面使用 b→a，保证共享边两侧的面方向一致。
    next_vertex = {int(b): int(a) for a, b in boundary}
    if (len(next_vertex) != len(boundary)
            or len(set(next_vertex.values())) != len(boundary)
            or set(next_vertex) != set(next_vertex.values())):
        raise ValueError("开口边界有分叉，无法组织成独立闭环")

    remaining = set(next_vertex)
    loops = []
    while remaining:
        start = min(remaining)
        current = start
        loop = []
        visited = set()
        while current not in visited:
            if current not in remaining:
                raise ValueError("开口边界不是独立闭环")
            loop.append(current)
            visited.add(current)
            current = next_vertex[current]
        if current != start or len(loop) < 3:
            raise ValueError("开口边界不能形成有效多边形")
        remaining.difference_update(loop)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def triangulate_cap(mesh, loop):
    """用拟合平面决定补面连接方式，保留所有原有三维顶点坐标。"""
    points = mesh.vertices[loop]
    centered = points - points.mean(axis=0)
    # 前两个主方向形成局部二维坐标轴，便于对开口轮廓做多边形三角化。
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1] <= 1e-12:
        raise ValueError("开口边界退化，无法三角化")
    polygon = centered @ basis[:2].T
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                  - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if area < 0:
        polygon[:, 1] *= -1  # Manifold 要求二维外轮廓为逆时针。
    local_faces = np.asarray(manifold3d.triangulate([polygon]), dtype=np.int64)
    if local_faces.shape != (len(loop) - 2, 3):
        raise ValueError("开口轮廓没有得到完整三角化")
    # 三角化只返回轮廓内部编号；用 loop 映射回原 mesh 顶点编号。
    return loop[local_faces]


def repair_bunny(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """仅给开口补面，保存到另一个 OBJ，不改变原文件或模型尺寸、朝向。"""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("修复结果必须保存为另一个文件")
    mesh = trimesh.load(input_path, force="mesh", process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    loops = find_boundary_loops(mesh)
    caps = [triangulate_cap(mesh, loop) for loop in loops]
    if caps:
        mesh.faces = np.vstack([mesh.faces, *caps])  # 追加新三角形，保留原来的表面。
    if not mesh.is_volume:
        raise ValueError("补面后仍不是有效封闭实体，停止导出")

    # 也验证 Manifold 能接受这个副本，避免只通过 Trimesh 的拓扑检查。
    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    ))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f"Manifold 拒绝修复结果：{solid.status()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(f"修复 {len(loops)} 个开口，增加 {sum(len(cap) for cap in caps)} 个三角形")
    print(f"另存副本：{output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    repair_bunny()
