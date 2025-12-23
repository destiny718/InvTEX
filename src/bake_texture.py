import os, json
from typing import *
import numpy as np
import torch
import utils3d
import nvdiffrast.torch as dr
from tqdm import tqdm
import imageio.v2 as iio
from PIL import Image
import math
import cv2


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-vertex normals (world space).

    vertices: (V, 3) float32 (CUDA)
    faces:    (F, 3) int32   (CUDA)
    returns:  (V, 3) float32 (CUDA)
    """
    # Face vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Unnormalized face normals
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)

    # Accumulate to vertices
    vn = torch.zeros_like(vertices)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)

    vn = torch.nn.functional.normalize(vn, dim=-1, eps=eps)
    return vn


@torch.no_grad()
def rasterize_pos_nrm(
    drctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vnormals: torch.Tensor,
    view: torch.Tensor,
    projection: torch.Tensor,
    H: int,
    W: int,
):
    """Rasterize world position + world normal per pixel using nvdiffrast.

    Returns:
        posw_px: (1,H,W,3)
        nrmw_px: (1,H,W,3)
        rast_m:  (1,H,W) bool
    """
    # Clip-space positions
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    v_h = torch.cat([vertices, ones], dim=1)  # (V,4)
    v_clip = (v_h @ view.t()) @ projection.t()  # (V,4)
    v_clip = v_clip.contiguous()

    rast, rast_db = dr.rasterize(drctx, v_clip[None], faces.contiguous(), resolution=(H, W))
    rast = rast.contiguous()
    if rast_db is not None:
        rast_db = rast_db.contiguous()
    rast_m = rast[..., 3] > 0

    pos_attr = vertices[None].contiguous()   # (1,V,3)
    nrm_attr = vnormals[None].contiguous()  # (1,V,3)

    posw_px, _ = dr.interpolate(pos_attr, rast, faces.contiguous(), rast_db=rast_db)
    nrmw_px, _ = dr.interpolate(nrm_attr, rast, faces.contiguous(), rast_db=rast_db)

    posw_px = posw_px.contiguous()
    nrmw_px = torch.nn.functional.normalize(nrmw_px, dim=-1, eps=1e-8).contiguous()

    return posw_px, nrmw_px, rast_m


@torch.no_grad()
def compute_w_ang(
    posw_px: torch.Tensor,
    nrmw_px: torch.Tensor,
    view: torch.Tensor,
    rast_m: torch.Tensor,
    cos_min: float = 0.05,
    power: float = 1.0,
    w_min: float = 0.05,
):
    """View-angle weight: suppress grazing-angle observations.

    w_ang = clamp(|n·v|, cos_min, 1)^power, then clamp to w_min.
    """
    cam2world = torch.linalg.inv(view)
    cam_pos = cam2world[:3, 3].view(1, 1, 1, 3)
    vdir = cam_pos - posw_px
    vdir = torch.nn.functional.normalize(vdir, dim=-1, eps=1e-8)
    cos = (nrmw_px * vdir).sum(dim=-1).abs()  # (1,H,W)
    cos = torch.clamp(cos, min=cos_min, max=1.0)
    w = torch.pow(cos, power)
    w = torch.clamp(w, min=w_min)
    w = w * rast_m.float()
    return w


def compute_w_edge(mask_bool: np.ndarray, edge_width: int = 16, w_min: float = 0.05) -> np.ndarray:
    """Edge weight via distance transform (0 near boundary, 1 inside).

    mask_bool: (H,W) foreground bool.
    returns:   (H,W) float32 in [w_min, 1] inside foreground, 0 outside.
    """
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(np.bool_)
    m = (mask_bool.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    w = np.clip(dist / max(1, int(edge_width)), 0.0, 1.0).astype(np.float32)
    # Clamp inside-foreground to avoid all-zero weights
    w = np.where(mask_bool, np.maximum(w, w_min), 0.0).astype(np.float32)
    return w


def weighted_charbonnier_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted Charbonnier loss.

    X, Y: (N, 3) or (N, C)
    W:    (N,) non-negative weights
    """
    diff = X - Y
    err = torch.sqrt(diff * diff + eps).mean(dim=-1)  # (N,)
    w = torch.clamp(W, min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (w * err).sum() / denom


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
def coords_trimesh_to_blender(coords: np.ndarray):
    return (transform_matrix @ coords.T).T


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
def get_intrinsic_extrinsic(world_to_camera, fov_angle_radians):
    width, height = 1.0, 1.0

    fx = width / 2 / math.tan(fov_angle_radians[0] / 2)
    fy = height / 2 / math.tan(fov_angle_radians[1] / 2)
    cx = width / 2.0
    cy = height / 2.0
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
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


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    fov: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    lambda_tv: float = 1e-2,
    verbose: bool = True,
    # multiscale tuning
    min_steps_per_level: int = 300,
    max_steps_per_level: int = 2500,
    stability_window: int = 50,
    stable_patience: int = 20,
    stable_rel_tol: float = 1e-2,
    # visualization
    save_view_weights: bool = True,
    view_weights_dir: Optional[str] = None,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        texture_size (int): Target final texture size; multiscale starts at 256 and upscales to this.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """
    H, W = observations[0].shape[0], observations[0].shape[1]

    vertices = torch.tensor(vertices).cuda().float()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]


    views: List[torch.Tensor] = []
    projections: List[torch.Tensor] = []

    fov_radians = math.radians(float(fov))
    for extr in extrinsics:
        extr_, intr_ = get_intrinsic_extrinsic(extr, (fov_radians, fov_radians))
        views.append(torch.tensor(extr_, dtype=torch.float32, device='cuda'))
        intr_t = torch.tensor(intr_, dtype=torch.float32, device='cuda')
        projections.append(intrinsics_to_projection(intr_t, near, far).to(dtype=torch.float32))

    rastctx = utils3d.torch.RastContext(backend='cuda')

    _uv = []
    _uv_dr = []
    for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
        with torch.no_grad():
            rast = utils3d.torch.rasterize_triangle_faces(
                rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
            )
            _uv.append(rast['uv'].detach())
            _uv_dr.append(rast['uv_dr'].detach())

    # --- Precompute per-view confidence weights C_vis = W_ang * W_edge ---
    #   W_ang: penalize grazing angles using world normal + view direction
    #   W_edge: down-weight silhouette boundary via distance-to-edge inside the mask
    # Notes:
    #   - We only store C_vis (H,W) per view to keep memory manageable.
    #   - This does NOT change the existing optimization loop structure; it only reweights per-pixel loss.
    with torch.no_grad():
        vnormals = compute_vertex_normals(vertices, faces)
        drctx_cuda = dr.RasterizeCudaContext()

        cvis_list: List[torch.Tensor] = []
        edge_width_px = 16
        edge_wmin = 0.05
        ang_cos_min = 0.05
        ang_pow = 1.0

        # Where to dump per-view weight visualizations.
        # Default: next to other debug renders.
        if view_weights_dir is None:
            view_weights_dir = os.path.join('/home/cjh/invtex', 'tmp_steps', 'weights')
        os.makedirs(view_weights_dir, exist_ok=True)

        for view_idx, (m, view, projection) in enumerate(
            tqdm(zip(masks, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): C_vis')
        ):
            # Rasterize position/normal once per view to estimate view-angle confidence.
            posw_px, nrmw_px, rast_mask = rasterize_pos_nrm(
                drctx_cuda, vertices, faces, vnormals, view, projection, H, W
            )
            # View-angle weight in [0,1] (1,H,W)
            w_ang = compute_w_ang(posw_px, nrmw_px, view, rast_mask, cos_min=ang_cos_min, power=ang_pow, w_min=ang_cos_min)[0]

            # Edge weight in [0,1] (H,W) computed from the provided object mask
            w_edge_np = compute_w_edge(m.detach().cpu().numpy(), edge_width=edge_width_px, w_min=edge_wmin)
            w_edge = torch.from_numpy(w_edge_np).to(device='cuda', dtype=torch.float32)

            c_vis = (w_edge * w_ang).clamp_(0.0, 1.0).to(dtype=torch.float16)
            cvis_list.append(c_vis.detach())

            # --- Save per-view weight visualizations ---
            if save_view_weights:
                try:
                    w_ang_u8 = (w_ang.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
                    w_edge_u8 = (w_edge.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
                    c_vis_u8 = (c_vis.to(torch.float32).clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()

                    # Ensure 2D arrays
                    if w_ang_u8.ndim == 3:
                        w_ang_u8 = w_ang_u8.squeeze(0)
                    if w_edge_u8.ndim == 3:
                        w_edge_u8 = w_edge_u8.squeeze(0)
                    if c_vis_u8.ndim == 3:
                        c_vis_u8 = c_vis_u8.squeeze(0)

                    # Grayscale
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_w_ang.png"), w_ang_u8)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_w_edge.png"), w_edge_u8)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_c_vis.png"), c_vis_u8)

                    # Colormap (use TURBO for better contrast)
                    w_ang_cm = cv2.applyColorMap(w_ang_u8, cv2.COLORMAP_TURBO)
                    w_edge_cm = cv2.applyColorMap(w_edge_u8, cv2.COLORMAP_TURBO)
                    c_vis_cm = cv2.applyColorMap(c_vis_u8, cv2.COLORMAP_TURBO)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_w_ang_cm.png"), w_ang_cm)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_w_edge_cm.png"), w_edge_cm)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_c_vis_cm.png"), c_vis_cm)

                    # Optional: save the binary mask used for edge distance
                    mask_u8 = (m.detach().cpu().numpy().astype(np.uint8) * 255)
                    cv2.imwrite(os.path.join(view_weights_dir, f"view_{view_idx:04d}_mask.png"), mask_u8)
                except Exception as e:
                    print(f"[warn] 保存 view 权重可视化失败 (view={view_idx}): {e}")

    _cvis = cvis_list

    final_texture_size = int(texture_size)

    def _build_texture_schedule(target: int) -> List[int]:
        # Start at 256 and grow by ×2 each level; ensure the final target is included.
        schedule = [256]
        while schedule[-1] < target:
            next_size = schedule[-1] * 2
            if next_size > target:
                break
            schedule.append(next_size)
        if schedule[-1] != target:
            schedule.append(target)
        return schedule

    texture_sizes = _build_texture_schedule(final_texture_size)

    def _valid_max_mip_level(size: int) -> int:
        # nvdiffrast mip stack requires even extents for each downsample step.
        # We allow as many mip levels as repeated /2 keeps integer and even before next downsample.
        levels = 0
        s = int(size)
        while s > 1 and (s % 2 == 0):
            levels += 1
            s //= 2
        return max(0, levels)

    def _make_texture_param(size: int, init_from: Optional[torch.Tensor] = None) -> torch.nn.Parameter:
        if init_from is None:
            t = torch.nn.Parameter(torch.full((1, size, size, 3), 0.5, device='cuda'))
        else:
            # init_from: [1, H, W, 3]
            t_chw = init_from.permute(0, 3, 1, 2)
            t_up = torch.nn.functional.interpolate(t_chw, size=(size, size), mode='bilinear', align_corners=False)
            t = t_up.permute(0, 2, 3, 1).contiguous()
        return torch.nn.Parameter(t)

    base_dir = '/home/cjh/invtex'
    ren_dir = os.path.join(base_dir, 'tmp_steps')
    os.makedirs(ren_dir, exist_ok=True)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    def tv_loss(texture):
        return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])

    loss_history: List[float] = []  # 记录全程 loss
    charbonnier_loss = L1_Charbonnier_loss()
    global_step = 0
    texture: Optional[torch.nn.Parameter] = None

    for level_idx, level_size in enumerate(texture_sizes):
        if texture is None:
            texture = _make_texture_param(level_size)
        else:
            with torch.no_grad():
                texture = _make_texture_param(level_size, init_from=texture.detach())

        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-3)

        desc = f"Texture baking (opt): optimizing @{level_size}"
        recent_losses: List[float] = []
        stable_counter = 0
        max_mip_level = _valid_max_mip_level(level_size)
        rel_change = 0.0

        with tqdm(total=max_steps_per_level, disable=not verbose, desc=desc) as pbar:
            for step in range(max_steps_per_level):
                optimizer.zero_grad()
                batch_size = min(4, len(views))
                if batch_size <= 0:
                    raise ValueError("views 为空，无法优化")
                selected_idxes = np.random.choice(len(views), size=batch_size, replace=False)

                loss = 0.0
                last_selected = int(selected_idxes[-1])
                charbonnier_loss_val = 0.0
                tv_loss_val = 0.0
                for selected in selected_idxes:
                    uv, uv_dr, observation, mask = _uv[int(selected)], _uv_dr[int(selected)], observations[int(selected)], masks[int(selected)]
                    render = dr.texture(texture, uv, uv_dr, max_mip_level=max_mip_level)[0]
                    # View-dependent confidence weight (C_vis = W_ang * W_edge).
                    # We keep the original masking logic and only reweight the per-pixel loss.
                    conf = _cvis[int(selected)].to(dtype=torch.float32)
                    w = conf[mask].reshape(-1)
                    if w.numel() == 0 or float(w.sum().item()) < 1e-8:
                        charbonnier_loss_val = charbonnier_loss(render[mask], observation[mask])
                    else:
                        charbonnier_loss_val = weighted_charbonnier_loss(
                            render[mask], observation[mask], w, eps=float(getattr(charbonnier_loss, 'eps', 1e-6))
                        )
                    loss = loss + charbonnier_loss_val
                    
                tv_loss_val = tv_loss(texture) * lambda_tv
                loss = loss + tv_loss_val 
                loss = loss / batch_size

                    

                loss.backward()
                optimizer.step()

                # 轻量 cosine anneal（每个 level 内）
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, max_steps_per_level, 1e-2, 1e-5)

                loss_val = float(loss.item())
                loss_history.append(loss_val)
                recent_losses.append(loss_val)
                if len(recent_losses) > stability_window:
                    recent_losses.pop(0)

                if step % 100 == 0:
                    with torch.no_grad():
                        ren_img = (render.clamp(0, 1) * 255).byte().detach().cpu().numpy()
                        iio.imwrite(
                            os.path.join(ren_dir, f"lvl{level_idx}_{level_size}_g{global_step:06d}_s{step:05d}_view_{last_selected:04d}.png"),
                            ren_img,
                        )

                # 稳定判据：在达到 min_steps 后，窗口均值相对变化很小则计数；计数达到 patience 则进入下一 level
                if step + 1 >= min_steps_per_level and len(recent_losses) == stability_window:
                    prev = np.mean(recent_losses[: stability_window // 2])
                    curr = np.mean(recent_losses[stability_window // 2 :])
                    denom = max(abs(prev), 1e-8)
                    rel_change = (prev - curr) / denom
                    if rel_change < stable_rel_tol:
                        stable_counter += 1
                    else:
                        stable_counter = 0
                    if stable_counter >= stable_patience:
                        if verbose:
                            print(f"[info] level {level_size} 稳定 (rel_change={rel_change:.2e})，进入下一分辨率")
                        pbar.set_postfix({'loss': loss_val, 'chan_loss': charbonnier_loss_val.item(), 'tv_loss': tv_loss_val.item(), 'stable': stable_counter, 'rel': f"{rel_change:.2e}"})
                        pbar.update()
                        break

                pbar.set_postfix({'loss': loss_val, 'chan_loss': charbonnier_loss_val.item(), 'tv_loss': tv_loss_val.item(), 'stable': stable_counter, 'rel': f"{rel_change:.2e}"})
                pbar.update()
                global_step += 1

    # 保存 loss 曲线（整个训练过程）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(loss_history)), loss_history, color='tab:blue', linewidth=1)
        plt.xlabel('Step')
        plt.ylabel('Loss (L1 + TV)')
        plt.title('Optimization Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "loss.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[warn] 绘制 loss 曲线失败: {e}")
        
    if texture is None:
        raise RuntimeError("texture 未初始化")
    texture = np.clip(texture[0].detach().flip(0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask = 1 - utils3d.torch.rasterize_triangle_faces(
        rastctx, (uvs * 2 - 1)[None], faces, final_texture_size, final_texture_size
    )['mask'][0].detach().cpu().numpy().astype(np.uint8)
    texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    return texture


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


def _load_glb_vertices_faces_uvs(glb_path: str):
    import trimesh
    import numpy as np

    scene = trimesh.load(glb_path, force='scene', skip_materials=False, process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)  # ★ 应用所有节点变换并合并
    else:
        mesh = scene


    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    vertices = coords_trimesh_to_blender(vertices)

    vertices, _, _ = normalize_mesh_verts(vertices)

    # vertices = np.asarray(vertices, dtype=np.float32)

    # 读取 UV；若无则用 xatlas 生成
    # if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
    #     uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    #     # 顶点数与 uv 数不一致时，回退重新参数化
    #     if uvs.shape[0] != vertices.shape[0]:
    #         print(f"警告: GLB 顶点数 {vertices.shape[0]} 与 UV 数 {uvs.shape[0]} 不匹配，重新参数化")
    #         vertices, faces, uvs = parametrize_mesh(vertices, faces)
    # else:
    #     vertices, faces, uvs = parametrize_mesh(vertices, faces)

    uvs = np.asarray(mesh.visual.uv, dtype=np.float32)

    return vertices, faces, uvs

def _read_blender_camera_poses(json_path: str):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    # intrinsics_K: 3x3
    # K = np.array(meta["intrinsics_K"], dtype=np.float32)
    fov = meta["fov_degrees"]
    width, height = meta["resolution"]  # [W, H]
    frames = meta["frames"]

    # 优先使用 world_to_camera_4x4，如果不存在则由 camera_matrix_world_4x4 或者 transform_matrix 求逆
    extrinsics = []
    for fr in frames:
        if "world_to_camera_4x4" in fr:
            E = np.array(fr["world_to_camera_4x4"], dtype=np.float32)
        elif "transform_matrix" in fr:
            Twc = np.array(fr["transform_matrix"], dtype=np.float32)
            E = np.linalg.inv(Twc).astype(np.float32)
        else:
            Twc = np.array(fr["camera_matrix_world_4x4"], dtype=np.float32)
            E = np.linalg.inv(Twc).astype(np.float32)
        extrinsics.append(E)

    # 每帧使用同一 K
    frame_ids = [fr["frame_id"] for fr in frames]
    return (width, height), extrinsics, fov, frame_ids

def _load_images(images_dir: str, frame_ids: list[int], image_pattern: str, expected_size: tuple[int,int]):
    """
    image_pattern 示例:
      - '{frame:04d}.png'  -> 0001.png, 0002.png, ...
      - 'frame_{frame:04d}.png' -> frame_0001.png, ...
    expected_size: (W, H)
    """
    from PIL import Image
    import numpy as np

    W, H = expected_size
    observations = []
    for fid in frame_ids:
        filename = image_pattern.format(frame=fid)
        path = os.path.join(images_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到图像: {path}")
        img = Image.open(path)

        if img.size != (W, H):
            raise ValueError(f"图像 {path} 分辨率 {img.size} 与期望 {expected_size} 不符")

        else:
            rgb = np.array(img.convert("RGB"))
            obs = rgb

        observations.append(obs)

    return observations


def _load_masks(images_dir: str, frame_ids: list[int], image_pattern: str, expected_size: tuple[int,int]):
    """
    仅加载遮罩
    image_pattern 示例:
      - '{frame:04d}.png'  -> 0001.png, 0002.png, ...
      - 'frame_{frame:04d}.png' -> frame_0001.png, ...
    expected_size: (W, H)
    """
    from PIL import Image
    import numpy as np

    W, H = expected_size
    masks = []
    for fid in frame_ids:
        filename = image_pattern.format(frame=fid)
        path = os.path.join(images_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到图像: {path}")
        img = Image.open(path)

        if img.size != (W, H):
            raise ValueError(f"图像 {path} 分辨率 {img.size} 与期望 {expected_size} 不符")

        else:
            rgb = np.array(img.convert("RGB"))
            m = np.any(rgb > 5.0, axis=-1)

        masks.append(m)
    return masks




def bake_texture_from_blender_glb(
    glb_path: str,
    poses_json_path: str,
    images_dir: str,
    images_dir_mask: str,
    image_pattern: str = "{frame:04d}.png",
    mask_pattern: str = "mask_{frame:04d}.png",
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    lambda_tv: float = 1e-2,
    verbose: bool = True,
):
    """
    使用 Blender 导出的 camera_poses.json 与对应帧图像，为 glb 网格烘焙纹理。
    约定 images_dir 下图像按 image_pattern 命名，并与 frames[].frame 对齐。

    返回:
      texture: np.ndarray[H_tex, W_tex, 3], uint8
      (vertices, faces, uvs): 供需要时进一步保存 glb 使用
    """
    # 1) 读取网格
    vertices, faces, uvs = _load_glb_vertices_faces_uvs(glb_path)

    # 2) 读取相机姿态与内参（Blender: world_to_camera 已给出，即 w2c）
    (W, H), extrinsics, fov, frame_ids = _read_blender_camera_poses(poses_json_path)

    # 3) 读取观测图像与遮罩（与帧号对齐）
    observations = _load_images(images_dir, frame_ids, image_pattern, (W, H))
    masks = _load_masks(images_dir_mask, frame_ids, mask_pattern, (W, H))

    # 4) 调用烘焙
    texture = bake_texture(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        observations=observations,
        masks=masks,
        extrinsics=extrinsics,
        fov=fov,
        texture_size=texture_size,
        near=near,
        far=far,
        lambda_tv=lambda_tv,
        verbose=verbose,
    )
    return texture, (vertices, faces, uvs)



if __name__ == "__main__":
    # tex, (V, F, UV) = bake_texture_from_blender_glb(
    #     glb_path="/home/cjh/invtex/tiger_warrior.glb",
    #     poses_json_path="/home/cjh/TRELLIS/camera_poses.json",
    #     images_dir="/home/cjh/invtex/frames",
    #     images_dir_mask="/home/cjh/invtex/mask_frames",
    #     image_pattern="frame_{frame:04d}.png",
    #     texture_size=2048,
    #     lambda_tv=1e-2,
    #     # near=2.0, far=6.0,
    #     verbose=True,
    # )

    tex, (V, F, UV) = bake_texture_from_blender_glb(
        glb_path="/home/cjh/invtex/nezuko.glb",
        poses_json_path="/home/cjh/invtex/outputs_nezuko/cameras_blender.json",
        images_dir="/home/cjh/invtex/outputs_nezuko/result",
        images_dir_mask="/home/cjh/invtex/outputs_nezuko/mask",
        image_pattern="frame_{frame:04d}.png",
        mask_pattern="mask_{frame:04d}.png",
        texture_size=4096,
        lambda_tv=4,
        near=0.01, far=1000.0,
        verbose=True,
    )


# tex 为 np.uint8 的 HxWx3，可保存
import imageio.v2 as iio
iio.imwrite("baked_texture_nezuko_new.png", tex)
