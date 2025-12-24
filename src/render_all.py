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



RESOLUTION = 1024


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
    mesh = trimesh.load(filepath, process=False, force='mesh')

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

    if texture_image is not None:
        print("Texture size:", texture_image.size)
    else:
        print("No texture found in material.")
        
    return vertices_norm, faces, uvs, texture_image, normals, scale, center_bl


class ColorRenderer:
    """Trainable color renderer for sparse grid"""
    def __init__(
        self, 
        rendering_options={}, 
        device='cuda'
    ):
        super().__init__()
        self.rendering_options = {
            "resolution": RESOLUTION,
            "near": 0.01,
            "far": 1000,
            "ssaa": 1,
        }
        
        self.rendering_options.update(rendering_options)
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
                self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa))
            
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
                normal_image = torch.zeros((resolution * ssaa, resolution * ssaa, 3), device=self.device)

            coords_image = dr.interpolate(vertices.contiguous(), rast, faces_int)[0]
            uv_image = dr.interpolate(uvs.contiguous(), rast, faces_int)[0]
            uv_image = torch.cat([uv_image[..., 0:1],1 - uv_image[..., 1:]], dim=-1)
            texture_image = dr.texture(texture[None, ...], uv_image)

            return coords_image, texture_image, depth_image, normal_image


if __name__ == "__main__":
    import os, json
    mesh_path = "1b28eef9b0d0e7783d0017b1b14c99e3afaa8ee986b45e4fdced506c0b4465d9.glb"

    # ===== 可调参数 =====
    FOV_DEG = 39.6         # 固定内参（由FOV计算一次，后续全程复用）
    FRAMES  = 80           # 帧数
    OUT_DIR = "outputs_1"    # 输出目录
    os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "depth"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "normal"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "position"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "mask"), exist_ok=True)

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

    # ===== 载入网格（含UV和法线）=====
    mesh_v, mesh_f, mesh_uv, texture, mesh_n, norm_scale, norm_center_bl = load_mesh_with_uvs(mesh_path)
    if texture is not None:
        texture.save(os.path.join(OUT_DIR, "texture.png"))

    mesh_v = torch.from_numpy(mesh_v).float().to(device)
    mesh_f = torch.from_numpy(mesh_f).int().to(device)
    mesh_uv = torch.from_numpy(mesh_uv).float().to(device)
    texture = torch.from_numpy(np.array(texture)).float().to(device) / 255.0
    mesh_n = torch.from_numpy(mesh_n).float().to(device)

    # ===== 构造渲染器 =====
    renderer = ColorRenderer(device=device)

    print("Loaded mesh with {} vertices, {} faces.".format(
        mesh_v.shape[0], mesh_f.shape[0]
    ))

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

    # ===== 逐帧渲染（每帧用 Blender c2w → get_intrinsic_extrinsic → GL 外参）=====
    frames_meta = []
    for i, c2w_bl in enumerate(tqdm(c2w_list, total=len(c2w_list), desc="Rendering")):
        extr_i, _ = get_intrinsic_extrinsic(c2w_bl, FOV_DEG)
        extr_i_t = torch.from_numpy(extr_i).float().to(device)

        # 渲染：返回 position(=世界坐标可视化)、rgb、depth、normal
        pos_img, rgb_img, depth_img, normal_img = renderer(
            extr_i_t, intr0_t, mesh_v, mesh_f, mesh_uv, texture, mesh_n=mesh_n
        )

        # ---- 保存四类结果 ----

        # depth：对有效像素线性归一化后存8bit
        depth_np = depth_img[0].detach().cpu().numpy()
        valid = depth_np > 0
        if valid.any():
            d_min, d_max = depth_np[valid].min(), depth_np[valid].max()
            depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-9)
            depth_norm[~valid] = 0.0
            mask_norm = depth_norm.copy()
            mask_norm[valid] = 1.0
        else:
            depth_norm = np.zeros_like(depth_np)

        valid = np.repeat(valid, 3, axis=-1)

        mask_vis = (mask_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
        Image.fromarray(mask_vis).save(os.path.join(OUT_DIR, "mask", f"mask_{i:04d}.png"))
        
        depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
        Image.fromarray(depth_vis).save(os.path.join(OUT_DIR, "depth", f"depth_{i:04d}.png"))

        # position [-0.5,0.5] → [0,255]
        pos_vis = ((pos_img[0].detach().cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        pos_vis[~valid] = 0
        Image.fromarray(pos_vis).save(os.path.join(OUT_DIR, "position", f"position_{i:04d}.png"))

        # rgb [0,1] → [0,255]
        rgb_vis = (rgb_img[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if rgb_vis.shape[2] == 4:
            rgba_vis = rgb_vis.copy()
            rgb_vis = rgba_vis[..., :3]
            alpha_channel = rgba_vis[..., 3] / 255.0
            rgb_vis[..., 0] = rgb_vis[..., 0] * alpha_channel
            rgb_vis[..., 1] = rgb_vis[..., 1] * alpha_channel
            rgb_vis[..., 2] = rgb_vis[..., 2] * alpha_channel
            rgb_vis[~valid] = 0
        else:
            rgb_vis[~valid] = 0
        Image.fromarray(rgb_vis).save(os.path.join(OUT_DIR, "rgb", f"rgb_{i:04d}.png"))

        # normal [-1,1] → [0,255]
        normal_np = normal_img[0].detach().cpu().numpy()
        normal_vis = ((normal_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        normal_vis[~valid] = 0
        Image.fromarray(normal_vis).save(os.path.join(OUT_DIR, "normal", f"normal_{i:04d}.png"))

        # 累积 Blender 风格外参
        frames_meta.append({
            "frame_id": i,
            "transform_matrix": c2w_bl.tolist()
        })

    # ===== 一次性写出 JSON（Blender 风格外参 + FOV）=====
    meta = {
        "fov_degrees": FOV_DEG,
        "resolution": [1024, 1024],
        "frames": frames_meta
    }
    with open(os.path.join(OUT_DIR, "cameras_blender.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
