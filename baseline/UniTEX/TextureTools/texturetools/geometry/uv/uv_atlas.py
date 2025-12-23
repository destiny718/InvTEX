import os
from typing import Tuple, Union
import numpy as np
import trimesh
import open3d as o3d
import xatlas
import torch

from ...mesh.trimesh_utils import align_to_bbox


def _trimesh_to_o3d_triangle_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int64)),
    )


def _load_mesh_as_o3d_triangle_mesh(path: str) -> o3d.geometry.TriangleMesh:
    ext = os.path.splitext(path)[1].lower()

    # Open3D 读取 glb/gltf 在一些文件上会得到空 mesh 或触发后续 UVAtlas 崩溃，
    # 这里优先用 trimesh 读取并合并 scene 中的几何体。
    if ext in {'.glb', '.gltf'}:
        loaded = trimesh.load(path, force='scene', process=False)
        if isinstance(loaded, trimesh.Scene):
            # NOTE: 必须 bake scene.graph 的节点变换，否则会丢失 glTF 的节点 transform，
            # 造成坐标系/朝向不一致（常见表现：看起来“方向反了/镜像了”）
            geometry = loaded.dump()
            geometries = [
                g for g in geometry
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0 and len(g.vertices) > 0
            ]
            if len(geometries) == 0:
                raise ValueError(f'No valid triangle mesh found in {path}.')
            mesh_tm = trimesh.util.concatenate(geometries)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh_tm = loaded
        else:
            raise ValueError(f'Unsupported mesh type loaded from {path}: {type(loaded)}')
        return _trimesh_to_o3d_triangle_mesh(mesh_tm)

    mesh_o3d = o3d.io.read_triangle_mesh(path, enable_post_processing=False)
    if len(mesh_o3d.vertices) == 0 or len(mesh_o3d.triangles) == 0:
        # 对少数格式做一次 trimesh 回退
        try:
            mesh_tm = trimesh.load(path, force='mesh', process=False)
            if isinstance(mesh_tm, trimesh.Trimesh) and len(mesh_tm.faces) > 0 and len(mesh_tm.vertices) > 0:
                return _trimesh_to_o3d_triangle_mesh(mesh_tm)
        except Exception:
            pass
    return mesh_o3d


def _unwrap_uv_with_xatlas(mesh: o3d.t.geometry.TriangleMesh, resolution: int = 2048, padding: int = 4) -> np.ndarray:
    atlas = xatlas.Atlas()
    v = mesh.vertex.positions.numpy()
    f = mesh.triangle.indices.numpy()
    atlas.add_mesh(v, f)

    chart_config = {
        'max_chart_area': 0.0,
        'max_boundary_length': 0.0,
        'normal_deviation_weight': 2.0,
        'roundness_weight': 0.01,
        'straightness_weight': 6.0,
        'normal_seam_weight': 4.0,
        'texture_seam_weight': 0.5,
        'max_cost': 16.0,
        'max_iterations': 1,
        'use_input_mesh_uvs': False,
        'fix_winding': False,
    }
    pack_config = {
        'max_chart_size': 0,
        'padding': int(padding),
        'texels_per_unit': 0.0,
        'resolution': int(resolution),
        'bilinear': True,
        'blockAlign': False,
        'bruteForce': False,
        'create_image': False,
        'rotate_charts_to_axis': True,
        'rotate_charts': True,
    }

    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    for k, v in chart_config.items():
        setattr(chart_options, k, v)
    for k, v in pack_config.items():
        setattr(pack_options, k, v)

    atlas.generate(chart_options, pack_options, verbose=False)
    _, indices, uvs = atlas.get_mesh(0)
    indices = np.asarray(indices, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    # Open3D legacy 的 triangle_uvs 是 per-corner: [F*3, 2]
    triangle_uvs = uvs[indices.reshape(-1)]
    return triangle_uvs


def preprocess_blank_mesh_trimesh(
    mesh:Union[trimesh.Trimesh, str, o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh], 
    min_faces=20_000, 
    max_faces=200_000, 
    scale=1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Preprocess Function for UV Geometry Image Baking

    vertices, faces: [V, 3], [F, 3]
    vertices_2d, faces_2d: [V_2d, 3], [F_2d, 3]
    '''
    # load open3d mesh
    if isinstance(mesh, trimesh.Trimesh):
        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices), 
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
    elif isinstance(mesh, str):
        mesh_o3d = o3d.io.read_triangle_mesh(mesh, enable_post_processing=False)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh_o3d = mesh
    elif isinstance(mesh, o3d.t.geometry.TriangleMesh):
        mesh_o3d = mesh.to_legacy()
    else:
        raise NotImplementedError(f'type of mesh {mesh} is not supported')

    # preprocess mesh
    device_o3d = o3d.core.Device('CPU:0')
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    aaa = bbox.min_bound
    bbb = bbox.max_bound
    sss = (bbb - aaa).max() / (2.0 * scale)
    transform = np.eye(4, dtype=np.float64)
    transform[[0, 1, 2], [0, 1, 2]] = 1.0 / sss
    transform[:3, 3] = - (aaa + bbb) / (2.0 * sss)
    mesh_o3d = mesh_o3d.transform(transform)
    mesh_o3d = mesh_o3d.remove_non_manifold_edges()
    mesh_o3d = mesh_o3d.remove_degenerate_triangles()
    mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
    F = len(mesh_o3d.triangles)
    if F > max_faces:
        # NOTE: o3d.geometry.TriangleMesh is slower than o3d.t.geometry.TriangleMesh
        # mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=max_faces, maximum_error=np.inf, boundary_weight=1.0)
        mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d, device=device_o3d)
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_reduction=1.0 - max_faces / F, preserve_volume=True)
        mesh_o3d = mesh_o3d.to_legacy()
        mesh_o3d = mesh_o3d.remove_non_manifold_edges()
        mesh_o3d = mesh_o3d.remove_degenerate_triangles()
        mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
    elif F < min_faces:
        mesh_o3d = mesh_o3d.subdivide_loop(number_of_iterations=2)
        mesh_o3d = mesh_o3d.remove_non_manifold_edges()
        mesh_o3d = mesh_o3d.remove_degenerate_triangles()
        mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d = mesh_o3d.merge_close_vertices(eps=1e-8)

    # unwrap uv
    mesh_2d_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=3, filter_scope=o3d.geometry.FilterScope.Vertex)
    mesh_2d_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_2d_o3d, device=device_o3d)
    try:
        mesh_2d_o3d.compute_uvatlas(size=2048, gutter=4.0, max_stretch=0.1667, parallel_partitions=4, nthreads=0)
        faces_uv = torch.as_tensor(mesh_2d_o3d.triangle.texture_uvs.numpy(), dtype=torch.float32, device='cuda')
        vertices_2d, faces_2d = torch.unique(faces_uv.reshape(-1, 2), dim=0, sorted=False, return_inverse=True, return_counts=False)
        vertices_2d = vertices_2d * 2.0 - 1.0
        faces_2d = faces_2d.reshape(-1, 3)
    except RuntimeError:  # UVAtlasPartition: Non-manifold mesh
        try:
            unwrap_uv_atlas_success = False
            atlas = xatlas.Atlas()
            atlas.add_mesh(mesh_2d_o3d.vertex.positions.numpy(), mesh_2d_o3d.triangle.indices.numpy())
            chart_config = {
                'max_chart_area': 0.0,
                'max_boundary_length': 0.0,
                'normal_deviation_weight': 2.0,
                'roundness_weight': 0.01,
                'straightness_weight': 6.0,
                'normal_seam_weight': 4.0,
                'texture_seam_weight': 0.5,
                'max_cost': 16.0,  # NOTE: avoid small charts
                'max_iterations': 1,
                'use_input_mesh_uvs': False,
                'fix_winding': False,
            }
            pack_config = {
                'max_chart_size': 0,
                'padding': 4,  # NOTE: avoid adjoint
                'texels_per_unit': 0.0,
                'resolution': 2048,
                'bilinear': True,
                'blockAlign': False,
                'bruteForce': False,
                'create_image': False,
                'rotate_charts_to_axis': True,
                'rotate_charts': True,
            }
            # NOTE: incompatible constructor arguments
            chart_options = xatlas.ChartOptions()
            pack_options = xatlas.PackOptions()
            for k, v in chart_config.items():
                setattr(chart_options, k, v)
            for k, v in pack_config.items():
                setattr(pack_options, k, v)
            atlas.generate(chart_options, pack_options, verbose=False)
            _, faces_2d, vertices_2d = atlas.get_mesh(0)
            vertices_2d = torch.as_tensor(vertices_2d, dtype=torch.float32, device='cuda')
            faces_2d = torch.as_tensor(faces_2d, dtype=torch.int64, device='cuda')
            vertices_2d = vertices_2d * 2.0 - 1.0
        except:
            vertices_2d, faces_2d = None, None
    except:
        vertices_2d, faces_2d = None, None

    # move to cuda
    vertices = torch.as_tensor(np.asarray(mesh_o3d.vertices, dtype=np.float32), dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(np.asarray(mesh_o3d.triangles, dtype=np.int64), dtype=torch.int64, device='cuda')
    return vertices, faces, vertices_2d, faces_2d


def preprocess_blank_mesh_o3d(mesh:o3d.geometry.TriangleMesh, min_faces=20_000, max_faces=200_000, scale=1.0) -> o3d.geometry.TriangleMesh:
    '''
    Preprocess Function for RGB/PBR Texture Generation

    scale: scale of bounding box
    '''
    ## rescale mesh
    mesh_o3d = mesh
    device_o3d = o3d.core.Device('CPU:0')
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    aaa = bbox.min_bound
    bbb = bbox.max_bound
    sss = (bbb - aaa).max() / (2.0 * scale)
    transform = np.eye(4, dtype=np.float64)
    transform[[0, 1, 2], [0, 1, 2]] = 1.0 / sss
    transform[:3, 3] = - (aaa + bbb) / (2.0 * sss)
    mesh_o3d = mesh_o3d.transform(transform)
    # get uv unwarpping
    if np.asarray(mesh_o3d.triangle_uvs).shape[0] == 0:
        mesh_o3d = mesh_o3d.remove_non_manifold_edges()
        mesh_o3d = mesh_o3d.remove_degenerate_triangles()
        mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
        F = len(mesh_o3d.triangles)
        if F == 0 or len(mesh_o3d.vertices) == 0:
            raise ValueError(
                'Mesh has no valid triangles/vertices after cleanup; cannot unwrap UVs. '
                'If you are using .glb/.gltf, try exporting to a triangulated .obj/.ply, '
                'or ensure the file contains at least one visible mesh primitive.'
            )
        if F > max_faces:
            # NOTE: o3d.geometry.TriangleMesh is slower than o3d.t.geometry.TriangleMesh
            # mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=max_faces, maximum_error=np.inf, boundary_weight=1.0)
            mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d, device=device_o3d)
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_reduction=1.0 - max_faces / F, preserve_volume=True)
            mesh_o3d = mesh_o3d.to_legacy()
            mesh_o3d = mesh_o3d.remove_non_manifold_edges()
            mesh_o3d = mesh_o3d.remove_degenerate_triangles()
            mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
        elif F < min_faces:
            mesh_o3d = mesh_o3d.subdivide_loop(number_of_iterations=2)
            mesh_o3d = mesh_o3d.remove_non_manifold_edges()
            mesh_o3d = mesh_o3d.remove_degenerate_triangles()
            mesh_o3d = mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d = mesh_o3d.merge_close_vertices(eps=1e-8)
        mesh_2d_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=3, filter_scope=o3d.geometry.FilterScope.Vertex)
        mesh_2d_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_2d_o3d, device=device_o3d)

        # Open3D 的 compute_uvatlas 在部分网格/版本上会：
        # - 抛 IndexError: _Map_base::at
        # - 或者直接卡死且 Ctrl+C 无法中断（C++ 多线程/死锁）
        # 因此默认使用 xatlas（更稳定），仅当显式开启环境变量时才尝试 Open3D 路径。
        use_open3d_uvatlas = os.environ.get('UNITEX_USE_OPEN3D_UVATLAS', '0').strip() == '1'
        if use_open3d_uvatlas:
            try:
                # 强制单线程+单分区，最大化避免卡死/不可中断问题
                mesh_2d_o3d.compute_uvatlas(
                    size=2048,
                    gutter=4.0,
                    max_stretch=0.1667,
                    parallel_partitions=1,
                    nthreads=1,
                )
                mesh_2d_legacy = mesh_2d_o3d.to_legacy()
                mesh_o3d.triangle_uvs = mesh_2d_legacy.triangle_uvs
                return mesh_o3d
            except (RuntimeError, IndexError, Exception):
                pass

        triangle_uvs = _unwrap_uv_with_xatlas(mesh_2d_o3d, resolution=2048, padding=4)
        mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh_o3d


def preprocess_blank_mesh(input_obj_path, output_obj_path, min_faces=20_000, max_faces=200_000, scale=1.0):
    mesh_o3d = _load_mesh_as_o3d_triangle_mesh(input_obj_path)
    mesh_o3d = preprocess_blank_mesh_o3d(
        mesh=mesh_o3d,
        min_faces=min_faces,
        max_faces=max_faces,
        scale=scale,
    )
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    o3d.io.write_triangle_mesh(
        output_obj_path,
        mesh_o3d,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=True,
    )


def preprocess_textured_mesh(input_obj_path, output_obj_path, scale=1.0):
    mesh:trimesh.Scene = trimesh.load(input_obj_path, force='scene', process=False)
    mesh = align_to_bbox(mesh, scale=scale)
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    mesh.export(output_obj_path)


