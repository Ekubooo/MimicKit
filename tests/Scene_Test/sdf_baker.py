"""读取封闭布尔 mesh，使用 Warp 烘焙 SDF、保存并重建 OBJ；不启动 viewer。"""

import argparse
import gc
from pathlib import Path

import numpy as np
import trimesh
import warp as wp


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
OUTPUT_DIR = ASSET_DIR / "output"  # 已合成的场景、SDF、重建 mesh 和切片图。
DEFAULT_INPUT = OUTPUT_DIR / "stairs_scene_bool.obj"
DEFAULT_VOXEL_SIZE = 0.01  # 相邻距离采样点间距，米；越小越细，三维网格内存约按其倒数的三次方增加。
DEFAULT_PADDING = 0.05  # 输入包围盒每侧外扩距离，米；让零等值面远离采样边界。
DEFAULT_SLICE_HEIGHT = 0.41  # 水平切片高度，米；当前平台约 0.4 米，查看其上方的奶牛脚部。


@wp.kernel(enable_backward=False)
def sample_grid_kernel(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    spacing: float,
    max_distance: float,
    field: wp.array3d(dtype=wp.float32),
    failures: wp.array(dtype=wp.int32),
):
    # 每个线程负责一个网格点。数组始终按 [X编号, Y编号, Z编号] 存储，不按向上轴换序。
    i, j, k = wp.tid()
    # origin 是编号 (0,0,0) 的采样位置，不额外加半个体素，保存和重建也采用同一约定。
    point = origin + spacing * wp.vec3(float(i), float(j), float(k))
    # accuracy=2.0 是绕数近似精度参数，threshold=0.5 是内部判定阈值，和 Newton 的烘焙一致。
    query = wp.mesh_query_point_sign_winding_number(mesh_id, point, max_distance, 2.0, 0.5)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        field[i, j, k] = query.sign * wp.length(point - closest)
    else:
        field[i, j, k] = max_distance
        wp.atomic_add(failures, 0, 1)  # 查询失败不能悄悄当作正确的外部距离。


@wp.kernel(enable_backward=False)
def sample_points_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_distance: float,
    distances: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh_id, points[i], max_distance, 2.0, 0.5)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[i] = query.sign * wp.length(points[i] - closest)
    else:
        distances[i] = wp.nan


@wp.kernel(enable_backward=False)
def sample_volume_kernel(
    volume_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    mode: int,
    distances: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # Volume 采样接收索引坐标，必须先把世界坐标转换回来，避免导出后整体错位。
    index = wp.volume_world_to_index(volume_id, points[i])
    distances[i] = wp.volume_sample_f(volume_id, index, mode)


def initialize_device(device_name="cuda:0"):
    """设置可写缓存；NanoVDB 创建需要 CUDA，不自动退回 CPU。"""
    wp.config.kernel_cache_dir = str(ASSET_DIR / ".warp_cache")
    wp.init()
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise ValueError("此脚本需要 CUDA 设备来创建 wp.Volume；请使用 --device cuda:0")
    return device


def load_mesh(input_path):
    """在内存副本中焊接同位置顶点，仅用于拓扑检查，不补洞或改变表面。"""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入 mesh：{input_path}")
    mesh = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("输入必须包含三角形 mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("输入顶点包含 NaN 或无穷值")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise ValueError("输入必须是封闭、朝向一致且体积为正的实体；本脚本不会修补输入")
    if np.any(mesh.area_faces <= 0):
        raise ValueError("输入包含零面积三角形，请先检查原 mesh")
    return mesh


def make_grid(mesh, voxel_size=DEFAULT_VOXEL_SIZE, padding=DEFAULT_PADDING):
    """返回首个采样点和 XYZ 点数；点数比对应的网格单元数多 1。"""
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel-size 必须是大于零的有限数值，单位为米")
    if not np.isfinite(padding) or padding < 2 * voxel_size:
        raise ValueError("padding 必须是有限数值，且至少为 voxel-size 的两倍")
    origin = mesh.bounds[0] - padding
    upper = mesh.bounds[1] + padding
    # ceil 向上取整，确保末端采样点也覆盖外扩后的包围盒。
    shape = np.ceil((upper - origin) / voxel_size).astype(np.int64) + 1
    if np.any(shape < 2) or np.any(shape > np.iinfo(np.int32).max):
        raise ValueError("网格尺寸超出可用范围")
    return origin, tuple(int(n) for n in shape)


def check_memory(shape, device):
    """报告稠密场大小，并保守估计 Marching Cubes 等中间数组的峰值。"""
    count = int(np.prod(shape, dtype=object))
    scalar_bytes = count * 4  # float32 每个采样值 4 字节。
    peak_estimate = count * 64  # MC 的边编号、计数、前缀和等远大于单个 SDF 数组。
    print(f"网格 XYZ={shape}，SDF 数组={scalar_bytes / 1024**2:.1f} MiB，"
          f"显存峰值粗估={peak_estimate / 1024**3:.2f} GiB", flush=True)
    if count * 3 > np.iinfo(np.int32).max:
        raise ValueError("网格过大，超过当前 Warp Marching Cubes 的 int32 边编号范围")
    if peak_estimate > 0.8 * device.free_memory:
        raise ValueError("估计显存不足，请增大 --voxel-size；本脚本不自动降低精度")


def make_warp_mesh(mesh, device):
    """Warp 使用 float32 顶点和展平的 int32 面索引；启用绕数所需的额外数据。"""
    return wp.Mesh(
        points=wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(np.asarray(mesh.faces, dtype=np.int32).ravel(), dtype=wp.int32, device=device),
        support_winding_number=True,
    )


def bake_sdf(mesh, origin, shape, voxel_size, device):
    """以输入表面为目标烘焙完整距离场，内部负、外部正，不做平滑或偏移。"""
    warp_mesh = make_warp_mesh(mesh, device)
    field = wp.empty(shape=shape, dtype=wp.float32, device=device)
    failures = wp.zeros(1, dtype=wp.int32, device=device)
    # 整个采样包围盒的对角线足以覆盖任意采样点到 mesh 的最近距离。
    max_distance = float(np.linalg.norm((np.asarray(shape) - 1) * voxel_size) + voxel_size)
    wp.launch(sample_grid_kernel, dim=shape,
              inputs=[warp_mesh.id, wp.vec3(*origin), voxel_size, max_distance],
              outputs=[field, failures], device=device)
    if int(failures.numpy()[0]):
        raise ValueError("部分网格点未找到最近表面，停止保存错误的 SDF")
    values = field.numpy()
    if not np.isfinite(values).all() or not (values.min() < 0 < values.max()):
        raise ValueError("SDF 必须有限且包含内外两种符号；请检查输入或提高采样精度")
    # 六个包围盒边界都应在实体外面，否则重建表面可能被裁开。
    boundaries = (values[0], values[-1], values[:, 0], values[:, -1], values[:, :, 0], values[:, :, -1])
    if any(np.any(side <= 0) for side in boundaries):
        raise ValueError("采样边界存在非正距离，请检查输入或增大 padding")
    return field, values


def save_sdf(values, origin, voxel_size, up_axis, input_path, npz_path, nvdb_path, device):
    """保存同一份数据为 NPZ 和 NanoVDB；先保存 NPZ，后续失败仍保留烘焙结果。"""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # values[i,j,k] 对应 origin + voxel_size * [i,j,k]；负值为实体内部。
    np.savez_compressed(
        npz_path, values=values, origin=np.asarray(origin, dtype=np.float64),
        voxel_size=np.float64(voxel_size), shape=np.asarray(values.shape, dtype=np.int64),
        up_axis=np.asarray(up_axis), axis_order=np.asarray("XYZ"), unit=np.asarray("m"),
        sign_convention=np.asarray("negative_inside"), source=np.asarray(str(Path(input_path).resolve())),
        format_version=np.int32(1),
    )
    print(f"SDF NPZ：{npz_path.resolve()}", flush=True)
    # 未分配区域使用正背景值；它仅表示域外，不是该处真实距离，NPZ 中记录的网格才是有效域。
    background = float(np.linalg.norm((np.asarray(values.shape) - 1) * voxel_size) + voxel_size)
    volume = wp.Volume.load_from_numpy(values, min_world=tuple(origin), voxel_size=voxel_size,
                                       bg_value=background, device=device)
    volume.save_to_nvdb(str(nvdb_path), codec="zip")
    print(f"SDF NanoVDB：{Path(nvdb_path).resolve()}", flush=True)
    return volume


def load_sdf(npz_path):
    """读取本脚本的 NPZ，返回数组、首点、间距和向上轴；无需 allow_pickle。"""
    with np.load(npz_path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1 or str(data["axis_order"]) != "XYZ":
            raise ValueError("不支持的 SDF NPZ 格式")
        return data["values"], data["origin"], float(data["voxel_size"]), str(data["up_axis"])


def sample_volume(volume, points, device, mode=wp.Volume.LINEAR):
    """在世界坐标位置采样保存/加载后的 wp.Volume，供坐标和数值核验。"""
    points_wp = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    result = wp.empty(len(points), dtype=wp.float32, device=device)
    wp.launch(sample_volume_kernel, dim=len(points), inputs=[volume.id, points_wp, mode], outputs=[result], device=device)
    return result.numpy()


def verify_saved_sdf(values, origin, voxel_size, up_axis, npz_path, nvdb_path, device):
    """核验 NPZ 全数组，并抽查 NanoVDB 首末点及随机网格点，检测换轴/偏移错误。"""
    loaded, loaded_origin, loaded_spacing, loaded_axis = load_sdf(npz_path)
    if not (np.array_equal(loaded, values) and np.array_equal(loaded_origin, origin)
            and loaded_spacing == voxel_size and loaded_axis == up_axis):
        raise ValueError("NPZ 保存/加载后的数组或坐标不一致")
    del loaded
    shape = np.asarray(values.shape)
    indices = np.vstack([np.zeros(3, dtype=int), shape - 1,
                         np.random.default_rng(0).integers(0, shape, size=(128, 3))])
    points = origin + indices * voxel_size
    # Warp 1.12 接收二进制文件对象/字节缓冲区，不能直接传路径字符串。
    with open(nvdb_path, "rb") as file:
        volume = wp.Volume.load_from_nvdb(file, device=device)
    # 这里核验恰好在网格点上的已保存值，用最近点采样。
    # 最末点的 float32 坐标舍入可能令线性插值少量混入域外背景，不能当作序列化误差。
    actual = sample_volume(volume, points, device, mode=wp.Volume.CLOSEST)
    expected = values[indices[:, 0], indices[:, 1], indices[:, 2]]
    error = float(np.max(np.abs(actual - expected)))
    tolerance = max(1e-5, voxel_size * 1e-3)  # 世界坐标与变换使用 float32，允许其舍入误差。
    if not np.isfinite(actual).all() or error > tolerance:
        raise ValueError(f"NanoVDB 坐标/数值核验失败，最大误差 {error:.6g} m")
    # 再查域内非整数位置的三线性插值，确保小于半格的坐标偏移也不会被最近点核验漏掉。
    indices = np.random.default_rng(1).integers(0, shape - 1, size=(128, 3))
    fraction = np.array([0.23, 0.47, 0.71])
    points = origin + (indices + fraction) * voxel_size
    expected_linear = np.zeros(len(indices), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz])
                weight = np.prod(np.where(offset, fraction, 1 - fraction))
                corners = indices + offset
                expected_linear += weight * values[corners[:, 0], corners[:, 1], corners[:, 2]]
    interpolated = sample_volume(volume, points, device)
    linear_error = float(np.max(np.abs(interpolated - expected_linear)))
    if not np.isfinite(interpolated).all() or linear_error > tolerance:
        raise ValueError(f"NanoVDB 域内插值核验失败，最大误差 {linear_error:.6g} m")
    print(f"NPZ 全数组一致；NanoVDB 网格点误差={error:.3g} m，插值误差={linear_error:.3g} m", flush=True)


def reconstruct_mesh(field, origin, voxel_size, output_path):
    """提取零等值面；仅焊接重复点及移除零面积面，不平滑、不补洞。"""
    # 上界是最后一个采样点的位置，不能用 shape * spacing，否则会把 mesh 拉大一个体素。
    upper = origin + (np.asarray(field.shape) - 1) * voxel_size
    vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
        field, threshold=0.0, domain_bounds_lower_corner=tuple(origin), domain_bounds_upper_corner=tuple(upper),
    )
    if len(vertices) == 0 or len(indices) == 0:
        raise ValueError("没有提取到零等值面；已保存的 SDF 保留")
    mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=indices.numpy().reshape(-1, 3), process=False)
    # 零值恰好落在网格点上时，多条网格边可生成同位置顶点，需要使其真正共享拓扑。
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.area_faces > 0)
    mesh.remove_unreferenced_vertices()
    mesh.export(output_path)
    print(f"重建 OBJ：{Path(output_path).resolve()}", flush=True)
    return mesh


def save_slices(values, origin, voxel_size, up_axis, slice_height, output_path):
    """两张中央竖切片加一张指定高度的水平切片；图片不旋转实际资产。"""
    import matplotlib
    matplotlib.use("Agg")  # 只写图片，不弹窗口。
    import matplotlib.pyplot as plt

    height_axis = 1 if up_axis == "Y" else 2
    horizontal_axes = [axis for axis in range(3) if axis != height_axis]
    shape = np.asarray(values.shape)
    height_index = int(np.rint((slice_height - origin[height_axis]) / voxel_size))
    if not 0 <= height_index < shape[height_axis]:
        raise ValueError("slice-height 不在 SDF 的采样高度范围内")
    cuts = [(horizontal_axes[0], int((shape[horizontal_axes[0]] - 1) // 2)),
            (horizontal_axes[1], int((shape[horizontal_axes[1]] - 1) // 2)),
            (height_axis, height_index)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    try:
        limit = 5 * voxel_size  # 仅截断显示颜色，不截断保存的距离值。
        names = "XYZ"
        for ax, (fixed_axis, index) in zip(axes, cuts):
            remaining = [axis for axis in range(3) if axis != fixed_axis]
            # 将高度放到图的纵轴；水平图则使用两个水平方向。
            x_axis = next((a for a in remaining if a != height_axis), remaining[0])
            y_axis = next(a for a in remaining if a != x_axis)
            cut = np.take(values, index, axis=fixed_axis)
            image = cut.T if remaining == [x_axis, y_axis] else cut
            x = origin[x_axis] + np.arange(shape[x_axis]) * voxel_size
            y = origin[y_axis] + np.arange(shape[y_axis]) * voxel_size
            # imshow 的 extent 表示像素边缘；扩半格使像素中心与采样点一致。
            extent = (x[0] - voxel_size / 2, x[-1] + voxel_size / 2,
                      y[0] - voxel_size / 2, y[-1] + voxel_size / 2)
            artist = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu", vmin=-limit, vmax=limit,
                               interpolation="nearest", aspect="equal")
            if image.min() < 0 < image.max():
                ax.contour(x, y, image, levels=[0], colors="black", linewidths=0.6)
            actual_height = origin[fixed_axis] + index * voxel_size
            ax.set(title=f"{names[fixed_axis]} = {actual_height:.4f} m",
                   xlabel=f"{names[x_axis]} (m)", ylabel=f"{names[y_axis]} (m)")
        fig.colorbar(artist, ax=axes, label="Signed distance (m); negative = inside")
        fig.suptitle(f"SDF slices ({up_axis}-up, voxel={voxel_size:g} m); black = zero contour")
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    print(f"SDF 切片：{Path(output_path).resolve()}", flush=True)


def distances_to_mesh(mesh, points, device):
    """点到 mesh 的有符号距离，用于误差统计和解析箱体验证。"""
    warp_mesh = make_warp_mesh(mesh, device)
    points_wp = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    result = wp.empty(len(points), dtype=wp.float32, device=device)
    all_points = np.vstack([mesh.bounds, np.asarray(points)])
    max_distance = float(np.linalg.norm(np.ptp(all_points, axis=0)) + 1)
    wp.launch(sample_points_kernel, dim=len(points), inputs=[warp_mesh.id, points_wp, max_distance],
              outputs=[result], device=device)
    distances = result.numpy()
    if not np.isfinite(distances).all():
        raise ValueError("误差统计的距离查询失败")
    return distances


def report_reconstruction(source, reconstructed, device, count=20000):
    """双向面积加权表面采样；结果是采样误差，不是严格的 Hausdorff 上界。"""
    valid = reconstructed.is_volume
    print(f"重建检查：封闭={reconstructed.is_watertight}，朝向一致={reconstructed.is_winding_consistent}，"
          f"连通分量={len(reconstructed.split(only_watertight=False))}，"
          f"三角形={len(reconstructed.faces)}，体积={reconstructed.volume:.6f} m^3", flush=True)
    print(f"原体积={source.volume:.6f} m^3，体积变化={(reconstructed.volume / source.volume - 1) * 100:.3f}%")
    for label, sampled_mesh, target_mesh in (("原 mesh → 重建", source, reconstructed),
                                             ("重建 → 原 mesh", reconstructed, source)):
        points, _ = trimesh.sample.sample_surface(sampled_mesh, count, seed=0)
        errors = np.abs(distances_to_mesh(target_mesh, points, device))
        print(f"{label}：{count} 点，平均={errors.mean():.6f} m，P95={np.percentile(errors, 95):.6f} m，"
              f"采样最大值={errors.max():.6f} m", flush=True)
    return valid


def create_sdf_assets(input_path=DEFAULT_INPUT, voxel_size=DEFAULT_VOXEL_SIZE, padding=DEFAULT_PADDING,
                      output_dir=OUTPUT_DIR, device_name="cuda:0", up_axis="Y",
                      slice_height=DEFAULT_SLICE_HEIGHT):
    """完整独立流程；所有输出由输入文件名派生，同名输出覆盖，输入不覆盖。"""
    up_axis = up_axis.upper()
    if up_axis not in ("Y", "Z") or not np.isfinite(slice_height):
        raise ValueError("up-axis 必须为 Y/Z，slice-height 必须是有限数值")
    mesh = load_mesh(input_path)
    origin, shape = make_grid(mesh, voxel_size, padding)
    height_axis = 1 if up_axis == "Y" else 2
    upper = origin + (np.asarray(shape) - 1) * voxel_size
    if not origin[height_axis] <= slice_height <= upper[height_axis]:
        raise ValueError("slice-height 不在外扩后的采样高度范围内，请显式指定")
    device = initialize_device(device_name)
    check_memory(shape, device)
    output_dir = Path(output_dir)
    stem = Path(input_path).stem
    outputs = {
        "npz": output_dir / f"{stem}_sdf.npz",
        "nvdb": output_dir / f"{stem}_sdf.nvdb",
        "mesh": output_dir / f"{stem}_sdf_reconstructed.obj",
        "slices": output_dir / f"{stem}_sdf_slices.png",
    }
    if Path(input_path).resolve() in [p.resolve() for p in outputs.values()]:
        raise ValueError("输出不能覆盖输入文件")
    print(f"输入：{Path(input_path).resolve()}（按 {up_axis}-up 和米解释，不转换坐标）", flush=True)
    print("开始烘焙完整 SDF…", flush=True)
    field, values = bake_sdf(mesh, origin, shape, voxel_size, device)
    print(f"距离范围：[{values.min():.6f}, {values.max():.6f}] m", flush=True)
    volume = save_sdf(values, origin, voxel_size, up_axis, input_path, outputs["npz"], outputs["nvdb"], device)
    del volume
    gc.collect()
    verify_saved_sdf(values, origin, voxel_size, up_axis, outputs["npz"], outputs["nvdb"], device)
    save_slices(values, origin, voxel_size, up_axis, slice_height, outputs["slices"])
    del values
    gc.collect()
    print("开始从 SDF=0 重建 mesh…", flush=True)
    reconstructed = reconstruct_mesh(field, origin, voxel_size, outputs["mesh"])
    del field
    gc.collect()
    if not report_reconstruction(mesh, reconstructed, device):
        raise ValueError("重建未通过封闭实体检查；SDF、切片和诊断用 OBJ 均保留，不自动补洞")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="已有封闭布尔 mesh；默认 Assets/output/stairs_scene_bool.obj")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, help="采样间距，米；默认 0.01")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING, help="包围盒每侧外扩，米；默认 0.05，至少两层体素")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="四种输出的目录；默认 Assets/output，同名覆盖")
    parser.add_argument("--device", default="cuda:0", help="Warp CUDA 设备；默认 cuda:0，不退回 CPU")
    parser.add_argument("--up-axis", type=str.upper, choices=("Y", "Z"), default="Y", help="输入向上轴，仅用于元数据和切片；不旋转 mesh")
    parser.add_argument("--slice-height", type=float, default=DEFAULT_SLICE_HEIGHT, help="水平切片高度，米；默认 0.41，位于当前平台上方")
    args = parser.parse_args()
    try:
        create_sdf_assets(args.input, args.voxel_size, args.padding, args.output_dir,
                          args.device, args.up_axis, args.slice_height)
    except (OSError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        parser.exit(1, f"SDF 烘焙/重建停止：{error}\n")
    print("SDF、重建 mesh 和距离切片已保存。")


if __name__ == "__main__":
    main()
