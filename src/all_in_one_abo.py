from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import math
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
import os, json
import argparse
import csv
import subprocess
from datetime import datetime
import os
os.environ["EGL_PLATFORM"] = "surfaceless"   # 让 Mesa 走无窗口 EGL
import open3d as o3d


fieldnames = [
    "file_identifier",
    "caption",
    "aesthetic_score"
]

def compute_extrinsics(azimuth, elevation, distance):
    """
    Compute the world-to-camera extrinsics matrix given azimuth (phi), elevation (theta), and distance.
    :param azimuth: Azimuth angle in degrees (ϕ, measured counterclockwise from +Z axis).
    :param elevation: Elevation angle in degrees (θ, measured from the XY plane).
    :param distance: Distance from the origin.
    :return: 4x4 torch.Tensor representing the extrinsics matrix (world-to-camera transform).
    """
    # Convert degrees to radians
    phi = torch.deg2rad(torch.tensor(azimuth))  # Azimuth
    theta = torch.deg2rad(torch.tensor(elevation))  # Elevation
    
    # Compute camera position in world space
    C = torch.tensor([
        distance * torch.sin(phi) * torch.cos(theta),
        distance * torch.cos(phi) * torch.cos(theta), 
        distance * torch.sin(theta),   
    ])
    
    # Forward vector (camera looks at the origin, so it's -C normalized)
    z_c = -C / torch.linalg.norm(C)
    
    # Right vector (cross product of z_c with world up [0,1,0])
    world_up = torch.tensor([0.0, 0.0, -1.0])
    x_c = torch.cross(world_up, z_c)
    if torch.linalg.norm(x_c) < 1e-6:  # Handle singularity at poles
        x_c = torch.tensor([1.0, 0.0, 0.0])
    x_c = x_c / torch.linalg.norm(x_c)
    
    # Up vector
    y_c = torch.cross(z_c, x_c)
    
    # Rotation matrix (camera-to-world)
    R = torch.stack([x_c, y_c, z_c], dim=1)
    
    # Construct the world-to-camera extrinsics matrix
    extrinsics = torch.eye(4)
    extrinsics[:3, :3] = R.T  # Transpose of rotation matrix
    extrinsics[:3, 3] = -R.T @ C  # Transform camera position
    
    return extrinsics


def blender_to_gl(R_blender, t_blender, K_blender):
    # Coordinate system transformation matrix
    # This accounts for Blender's camera looking down -Z
    R_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    # Convert rotation matrix
    R_gaussian = R_convert @ R_blender
    # Convert translation vector
    t_gaussian = R_convert @ t_blender
    # Intrinsics typically remain unchanged in this conversion
    K_gaussian = K_blender.copy()
    
    return R_gaussian, t_gaussian, K_gaussian

def get_intrinsic_extrinsic(transform_matrix, fov_degree):
    c2w = transform_matrix
    fov_angle = math.radians(fov_degree)
    width, height = 1, 1
    focal_length_pixels = height / (2 * math.tan(fov_angle / 2))
    cx = width / 2
    cy = height / 2
    intrinsic = np.array([
        [focal_length_pixels, 0, cx],
        [0, focal_length_pixels, cy],
        [0, 0, 1]
    ])
    world_to_camera = np.linalg.inv(c2w)
    extrinsic = world_to_camera
    R_gaussian, t_gaussian, K_gaussian = blender_to_gl(extrinsic[:3, :3], extrinsic[:3, 3:], intrinsic)
    extrinsic = np.eye(4, 4)
    extrinsic[:3, :3] = R_gaussian
    extrinsic[:3, 3:] = t_gaussian
    intrinsic = K_gaussian

    return extrinsic, intrinsic


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def normalize_mesh_verts(vertices):
    """
    Normalize vertices to fit inside [-0.5, 0.5]^3, preserving aspect ratio.
    
    Args:
        vertices (np.ndarray): shape (N, 3)
    Returns:
        vertices_normalized (np.ndarray): shape (N, 3)
        scale (float): the applied uniform scale factor
        center (np.ndarray): the original center (3,)
    """
    # Compute bounding box
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    
    # Compute center
    center = (vmin + vmax) / 2.0
    
    # Shift vertices to center at origin
    vertices_centered = vertices - center
    
    # Compute the maximum range among x, y, z
    max_range = (vmax - vmin).max()
    
    # Compute scale to fit in [-0.5, 0.5]
    scale = 1.0 / max_range
    
    # Apply scaling
    vertices_normalized = vertices_centered * scale
    
    return vertices_normalized, scale, center

transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
def coords_trimesh_to_blender(coords: np.ndarray):
    return (transform_matrix @ coords.T).T



def load_mesh_with_uvs(
    filepath, 
    coord_system: Literal["blender", "trimesh"] = "blender"
):
    """
    Load a mesh using trimesh and return vertices, faces, UVs, and normals.

    Args:
        filepath (str): Path to the mesh file.

    Returns:
        vertices (np.ndarray): (N, 3) array of normalized vertex positions.
        faces (np.ndarray): (M, 3) array of triangle indices.
        uvs (np.ndarray or None): (N, 2) array of UV coordinates or None.
        texture_image (PIL.Image or None): Texture image or None.
        normals (np.ndarray): (N, 3) array of vertex normals.
    """
    try:
        mesh = trimesh.load(filepath, process=False, force='mesh')
    except Exception as e:
        return None, None, None, None, None, None, None

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a single Trimesh object. It might be a scene.")

    vertices = mesh.vertices
    faces = mesh.faces

    # trimesh 会自动计算顶点法线
    normals = mesh.vertex_normals

    # if coord_system == "blender":
    vertices = coords_trimesh_to_blender(vertices)
    

    # 同样对法线应用旋转变换
    normals = coords_trimesh_to_blender(normals)
    # 确保法线在旋转后仍然是单位向量
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    vertices_norm, scale, center_bl = normalize_mesh_verts(vertices) # 注意：法线不需要归一化（平移和缩放）

    # UVs
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
    else:
        uvs = None

    material = mesh.visual.material
    print(type(material), material.__dict__)

    # Texture
    texture_image = getattr(material, 'baseColorTexture', None)
        
    return vertices_norm, faces, uvs, texture_image, normals, scale, center_bl


class ColorRenderer:
    """Trainable color renderer for sparse grid"""
    def __init__(
        self, 
        rendering_options={}, 
        device='cuda'
    ):
        super().__init__()
        self.rendering_options = rendering_options
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device
        self.resolution = self.rendering_options["resolution"]

    def render(self, **kwargs):
        return self(**kwargs)


    def __call__(
        self, 
        extrinsics: torch.Tensor, 
        intrinsics: torch.Tensor, 
        mesh_v: torch.Tensor, 
        mesh_f: torch.Tensor,
        mesh_uv: Optional[torch.Tensor] = None,
        texture: Optional[torch.Tensor] = None,
        mesh_n: Optional[torch.Tensor] = None,
    ):
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        with torch.no_grad():
            perspective = intrinsics_to_projection(intrinsics, near, far)
            
            RT = extrinsics.unsqueeze(0)
            full_proj = (perspective @ extrinsics).unsqueeze(0)
            
            vertices = mesh_v.unsqueeze(0)
            uvs = mesh_uv.unsqueeze(0)

            if mesh_n is not None:
                normals = mesh_n.unsqueeze(0)
            # --------------------------------------------------

            with torch.autocast("cuda", dtype=torch.float32):
                vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
                vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
                vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
                faces_int = mesh_f.int()

            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, faces_int, (resolution[0] * ssaa, resolution[1] * ssaa))
            
            valid_mask = rast[..., 3] > 0
            # valid_mask = valid_mask.squeeze()

            depth_image = dr.interpolate(
                vertices_camera[..., 2:3].contiguous(), rast, faces_int
            )[0]

            if mesh_n is not None:
                normal_image = dr.interpolate(normals.contiguous(), rast, faces_int)[0]
                # 确保插值后的法线仍然是单位向量
                normal_image = F.normalize(normal_image, p=2, dim=-1)
                # normal_image[~valid_mask] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            else:
                # 如果没有提供法线，返回一个空图像
                print("No normals provided, returning zero normal image.")
                normal_image = torch.zeros((resolution[0] * ssaa, resolution[1] * ssaa, 3), device=self.device)

            coords_image = dr.interpolate(vertices.contiguous(), rast, faces_int)[0]
            camera_pos_image = dr.interpolate(vertices_camera[..., :3].contiguous(), rast, faces_int)[0]
            uv_image = dr.interpolate(uvs.contiguous(), rast, faces_int)[0]
            uv_image = torch.cat([uv_image[..., 0:1],1 - uv_image[..., 1:]], dim=-1)
            texture_image = dr.texture(texture[None, ...], uv_image)

            return coords_image, depth_image, normal_image, camera_pos_image, texture_image
        


def ffmpeg_silent_with_log(
    feature_name,
    output_path,
    log_path="ffmpeg_log.txt",
):
    """
    静默执行 FFmpeg，将 PNG 序列生成无损视频，并记录到日志。

    返回:
        True  -> 执行成功
        False -> 执行失败（已记录日志）
    """

    # 确保输出路径存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if feature_name == "rgb":
        input_pattern = os.path.join(os.path.dirname(output_path), "rgb", "rgb_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
            "-movflags", "+faststart",
        ]
    elif feature_name == "rgb_pbr":
        input_pattern = os.path.join(os.path.dirname(output_path), "rgb_pbr", "rgb_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-crf", str(18),
            "-preset", "veryslow",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
    elif feature_name == "depth":
        input_pattern = os.path.join(os.path.dirname(output_path), "depth", "depth_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-vf", "format=gray",
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
        ]
    elif feature_name == "normal":
        input_pattern = os.path.join(os.path.dirname(output_path), "normal", "normal_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
            "-movflags", "+faststart",
        ]
    elif feature_name == "mask":
        input_pattern = os.path.join(os.path.dirname(output_path), "mask", "mask_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-vf", "format=gray",
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
        ]
    elif feature_name == "ccm":
        input_pattern = os.path.join(os.path.dirname(output_path), "ccm", "ccm_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
            "-movflags", "+faststart",
        ]
    elif feature_name == "position":
        input_pattern = os.path.join(os.path.dirname(output_path), "position", "position_%04d.png")

        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-framerate", str(16),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-crf", str(0),
            "-preset", "veryslow",
            "-pix_fmt", "yuv444p",
            "-movflags", "+faststart",
        ]
    
    cmd.append(output_path)

    # 执行并静默
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        _write_log(log_path, cmd, "SUCCESS")

    except subprocess.CalledProcessError as e:
        _write_log(log_path, cmd, f"ERROR: Return code {e.returncode}")

    except Exception as e:
        _write_log(log_path, cmd, f"EXCEPTION: {repr(e)}")


def _write_log(log_path, cmd, status):
    """
    写日志文件，包含时间戳、命令、执行状态。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"[{timestamp}] {status}\n")
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata file.")
    parser.add_argument("--base_glb_dir", type=str, required=True, help="Base directory for the file identifiers.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for rendered images and metadata.")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames to render.")
    parser.add_argument("--fov", type=float, default=39.6, help="Field of view in degrees.")
    parser.add_argument("--resolution_x", type=int, default=1024, help="Rendering resolution x-axis.")
    parser.add_argument("--resolution_y", type=int, default=1024, help="Rendering resolution y-axis.")
    parser.add_argument("--render_rgb", type=bool, default=False, help="Whether to render RGB images.")
    parser.add_argument("--render_depth", type=bool, default=False, help="Whether to render depth images.")
    parser.add_argument("--render_normal", type=bool, default=False, help="Whether to render normal images.")
    parser.add_argument("--render_position", type=bool, default=False, help="Whether to render position images.")
    parser.add_argument("--render_ccm", type=bool, default=False, help="Whether to render camera coordinate maps.")
    parser.add_argument("--render_mask", type=bool, default=False, help="Whether to render masks.")
    args = parser.parse_args()

    # ===== 可调参数 =====
    FOV_DEG = args.fov         # 固定内参（由FOV计算一次，后续全程复用）
    FRAMES  = args.frames           # 帧数
    OUT_DIR = args.out_dir    # 输出目录
    BASE_GLB_DIR = args.base_glb_dir

    # os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)
    # os.makedirs(os.path.join(OUT_DIR, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(OUT_DIR, "normal"), exist_ok=True)
    # os.makedirs(os.path.join(OUT_DIR, "position"), exist_ok=True)
    # os.makedirs(os.path.join(OUT_DIR, "mask"), exist_ok=True)
    # os.makedirs(os.path.join(OUT_DIR, "ccm"), exist_ok=True)

    # ===== 读取 metadata =====
    all_data = []

    meta_path = args.metadata_path
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_identifier = row.get("file_identifier", "").strip()
            captions = row.get("captions", "")
            if len(captions) == 0:
                # 跳过没有 caption 的样本
                continue

            captions = json.loads(captions)
            caption = captions[0] if isinstance(captions, list) else captions

            full_glb_mesh_path = os.path.join(BASE_GLB_DIR, file_identifier)
            data_dict = {}
            data_dict["file_identifier"] = file_identifier
            data_dict["caption"] = caption
            data_dict["full_glb_mesh_path"] = full_glb_mesh_path
            data_dict["aesthetic_score"] = row.get("aesthetic_score", "0.0")
            all_data.append(data_dict)

    # ===== 首帧：Blender 风格 c2w（你给定的外参）=====
    transform_mtx = [
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0, -1.0, -3.0],
        [0.0, 1.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  1.0]
    ]
    c2w_bl0 = np.array(transform_mtx, dtype=np.float32)

    # ===== 固定相机内参：仅用首帧计算一次并全程复用 =====
    extr0, intr0 = get_intrinsic_extrinsic(c2w_bl0, FOV_DEG)  # intr0 固定
    device = "cuda"
    intr0_t = torch.from_numpy(intr0).float().to(device)

    

    # ===== 在 Blender 坐标系下构造 z = x 平面上的圆周路径 =====
    # 平面 z=x 的法向量：n_bl = (1, 0, -1)  （满足 n·v=0 等价于 z-x=0）
    n_bl = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    n_bl /= (np.linalg.norm(n_bl) + 1e-9)

    def proj_to_plane(v, n):
        """将向量 v 投影到法向 n 的平面上（Blender坐标系）"""
        return v - (v @ n) * n

    # 初始相机位置（Blender世界坐标）
    C_bl0 = c2w_bl0[:3, 3]
    # 投影到 z=x 平面（如果本来就在平面上，此步不改变）
    C_bl0_proj = proj_to_plane(C_bl0, n_bl)
    r = np.linalg.norm(C_bl0_proj)
    if r < 1e-8:
        raise ValueError("初始相机位置在 z=x 平面投影长度为0，无法构造圆周路径。")

    # 平面内的两条正交单位基：u1 指向初始投影方向；u2 = 归一化(n × u1)
    u1 = C_bl0_proj / r
    u2 = np.cross(n_bl, u1)
    u2 /= (np.linalg.norm(u2) + 1e-9)

    world_up_bl = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def build_c2w_lookat_blender(C_bl: np.ndarray) -> np.ndarray:
        """
        由 Blender 世界中的相机位置 C_bl 构造 Blender 风格 c2w（相机始终看向原点）：
        约定：相机局部 -Z 朝前，Y 向上，X 向右；c2w 的列为 [X, Y, Z] 轴在世界坐标下的方向。
        """
        forward = -C_bl
        forward /= (np.linalg.norm(forward) + 1e-9)
        Z = -forward
        X = np.cross(world_up_bl, Z); xnorm = np.linalg.norm(X)
        if xnorm < 1e-8:
            # 退化：当前前向与 world_up 共线时换一个上方向
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            X = np.cross(alt_up, Z); xnorm = np.linalg.norm(X)
        X /= (xnorm + 1e-9)
        Y = np.cross(Z, X)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([X, Y, Z], axis=1)
        c2w[:3, 3]  = C_bl
        return c2w

    # ===== 预先计算所有 Blender 风格外参（首帧自然等于 transform_mtx）=====
    c2w_list = []
    for i in range(FRAMES):
        theta = 2.0 * math.pi * (i / FRAMES)  # 均匀采样
        C_bl = r * (math.cos(theta) * u1 + math.sin(theta) * u2)  # z=x 平面上的圆周
        c2w_bl = build_c2w_lookat_blender(C_bl)
        c2w_list.append(c2w_bl)

    render_options = {
        "resolution": [args.resolution_x, args.resolution_y],
        "near": 0.01,
        "far": 1000,
        "ssaa": 1,
    }

    # ===== 构造 open3d 渲染器 =====
    o3d_renderer = o3d.visualization.rendering.OffscreenRenderer(args.resolution_x, args.resolution_y)
    o3d_scene = o3d_renderer.scene

    o3d_scene.set_background([0.0, 0.0, 0.0, 1.0])
    o3d_scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        np.array([0.577, -0.577, -0.577], dtype=np.float32)
    )

    existing_ids = []

    out_csv = os.path.join(OUT_DIR, "rendered_metadata_1.csv")

    with open(out_csv, "r", encoding="utf-8") as f:
        existing_ids = [row["file_identifier"] for row in csv.DictReader(f)]

    for data in tqdm(all_data, total=len(all_data), desc="Rendering meshes"):
        file_identifier = data["file_identifier"]
        file_identifier = file_identifier[:-4] if file_identifier.endswith(".glb") else file_identifier

        if file_identifier in existing_ids:
            print(f"Skipping {file_identifier} as it is already rendered.")
            continue
        
        file_output_path = os.path.join(OUT_DIR, file_identifier)
        os.makedirs(file_output_path, exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "mask"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "normal"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "position"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "ccm"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "rgb"), exist_ok=True)
        

        # ===== 载入网格（含UV和法线）=====
        mesh_v, mesh_f, mesh_uv, texture, mesh_n, norm_scale, norm_center_bl = load_mesh_with_uvs(data["full_glb_mesh_path"])

        if texture is None:
            continue

        mesh_v = torch.from_numpy(mesh_v).float().to(device)
        mesh_f = torch.from_numpy(mesh_f).int().to(device)
        mesh_uv = torch.from_numpy(mesh_uv).float().to(device)
        texture = torch.from_numpy(np.array(texture)).float().to(device) / 255.0
        mesh_n = torch.from_numpy(mesh_n).float().to(device)

        renderer = ColorRenderer(rendering_options=render_options, device=device)

        # ===== 逐帧渲染（每帧用 Blender c2w → get_intrinsic_extrinsic → GL 外参）=====
        frames_meta = []
        for i, c2w_bl in enumerate(c2w_list):
            extr_i, _ = get_intrinsic_extrinsic(c2w_bl, FOV_DEG)
            extr_i_t = torch.from_numpy(extr_i).float().to(device)

            # 渲染：返回 position(=世界坐标可视化)、rgb、depth、normal、CCM、rgb
            pos_img, depth_img, normal_img, camera_pos_image, texture_image = renderer(
                extr_i_t, intr0_t, mesh_v, mesh_f, mesh_uv, texture, mesh_n=mesh_n
            )

            # ---- 保存结果 ----

            # depth：对有效像素线性归一化后存8bit
            depth_np = depth_img[0].detach().cpu().numpy()
            valid = depth_np > 0
            if valid.any():
                d_min, d_max = depth_np[valid].min(), depth_np[valid].max()
                # depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-9)   # 近黑远白
                depth_norm = (d_max - depth_np) / (d_max - d_min + 1e-9)     # 近白远黑
                depth_norm[~valid] = 0.0
                mask_norm = depth_norm.copy()
                mask_norm[valid] = 1.0
            else:
                depth_norm = np.zeros_like(depth_np)

            valid_3 = np.repeat(valid, 3, axis=-1)

            if args.render_depth:
                depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
                Image.fromarray(depth_vis).save(os.path.join(file_output_path, "depth", f"depth_{i:04d}.png"))

            if args.render_mask:
                mask_vis = (mask_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
                Image.fromarray(mask_vis).save(os.path.join(file_output_path, "mask", f"mask_{i:04d}.png"))

            if args.render_position:
                # position [-0.5,0.5] → [0,255]
                pos_vis = ((pos_img[0].detach().cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                pos_vis[~valid_3] = 0
                Image.fromarray(pos_vis).save(os.path.join(file_output_path, "position", f"position_{i:04d}.png"))

            # CCM
            if args.render_ccm:
                camera_pos_np = camera_pos_image[0].detach().cpu().numpy()
                ccm_vis = ((camera_pos_np + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                ccm_vis[~valid_3] = 0
                Image.fromarray(ccm_vis).save(os.path.join(file_output_path, "ccm", f"ccm_{i:04d}.png"))

            if args.render_normal:
                # normal [-1,1] → [0,255]
                normal_np = normal_img[0].detach().cpu().numpy()
                normal_vis = ((normal_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                normal_vis[~valid_3] = 0
                Image.fromarray(normal_vis).save(os.path.join(file_output_path, "normal", f"normal_{i:04d}.png"))

            # rgb [0,1] → [0,255]
            if args.render_rgb:
                rgb_vis = (texture_image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                if rgb_vis.shape[2] == 3:
                    rgb_vis[~valid_3] = 0
                else:
                    # RGBA
                    rgba_vis = rgb_vis.copy()
                    rgb_vis = rgba_vis[..., :3]
                    alpha_channel = rgba_vis[..., 3] / 255.0
                    rgb_vis[..., 0] = rgb_vis[..., 0] * alpha_channel
                    rgb_vis[..., 1] = rgb_vis[..., 1] * alpha_channel
                    rgb_vis[..., 2] = rgb_vis[..., 2] * alpha_channel
                    rgb_vis[~valid_3] = 0

                Image.fromarray(rgb_vis).save(os.path.join(file_output_path, "rgb", f"rgb_{i:04d}.png"))

            # 累积 Blender 风格外参
            frames_meta.append({
                "frame_id": i,
                "transform_matrix": c2w_bl.tolist()
            })

            # ===== 使用 Open3D 渲染器生成缩略图 =====
            o3d_model = o3d.io.read_triangle_model(
                data["full_glb_mesh_path"]
            )
            o3d_scene.add_model("model", o3d_model)

            # ---------- 让 Open3D 的几何和 nvdiffrast 的几何一致 ----------
            # nvdiffrast 顶点： v_nd = norm_scale * ( R_tb @ v_tm - norm_center_bl )
            # 其中 R_tb = transform_matrix
            R_tb = transform_matrix.astype(np.float64)           # 3x3
            center_bl = norm_center_bl.astype(np.float64)        # (3,)
            s = float(norm_scale)

            T_model = np.eye(4, dtype=np.float64)
            T_model[:3, :3] = s * R_tb                           # 旋转 + 缩放
            T_model[:3, 3]  = -s * center_bl                     # 平移到原点

            # 应用到 Open3D 的模型几何上
            o3d_scene.set_geometry_transform("model", T_model)   # 像素级对齐的关键

            os.makedirs(os.path.join(file_output_path, "rgb_pbr"), exist_ok=True)
            # -------- Open3D PBR RGB ----------
            c2w_np = c2w_bl.astype(np.float32)
            eye    = c2w_np[:3, 3]
            up     = c2w_np[:3, 1]  # 相机局部 +Y 轴
            center = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 物体已被归一化到原点

            o3d_renderer.setup_camera(
                FOV_DEG,
                center,
                eye,
                up
            )
            img_o3d = o3d_renderer.render_to_image()
            o3d.io.write_image(
                os.path.join(file_output_path, "rgb_pbr", f"rgb_{i:04d}.png"),
                img_o3d
            )

        ffmpeg_silent_with_log(
            feature_name="rgb_pbr",
            output_path=os.path.join(file_output_path, "rgb_pbr.mp4"),
            log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
        )

        if args.render_rgb:
            ffmpeg_silent_with_log(
                feature_name="rgb",
                output_path=os.path.join(file_output_path, "rgb.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )
        if args.render_depth:
            ffmpeg_silent_with_log(
                feature_name="depth",
                output_path=os.path.join(file_output_path, "depth.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )
        if args.render_normal:
            ffmpeg_silent_with_log(
                feature_name="normal",
                output_path=os.path.join(file_output_path, "normal.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )
        if args.render_position:
            ffmpeg_silent_with_log(
                feature_name="position",
                output_path=os.path.join(file_output_path, "position.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )
        if args.render_ccm:
            ffmpeg_silent_with_log(
                feature_name="ccm",
                output_path=os.path.join(file_output_path, "ccm.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )
        if args.render_mask:
            ffmpeg_silent_with_log(
                feature_name="mask",
                output_path=os.path.join(file_output_path, "mask.mp4"),
                log_path=os.path.join(OUT_DIR, "ffmpeg_log.txt"),
            )

        single_out = {
            "file_identifier": file_identifier,
            "caption": data["caption"],
            "aesthetic_score": data["aesthetic_score"],
        }
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(single_out)


    # ===== 一次性写出 JSON（Blender 风格外参 + FOV）=====
    meta = {
        "fov_degrees": FOV_DEG,
        "resolution": [args.resolution_x, args.resolution_y],
        "frames": frames_meta
    }
    with open(os.path.join(OUT_DIR, "cameras_blender.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
