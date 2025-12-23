#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end pipeline that downloads Sketchfab models from a CSV manifest
and then renders per-model assets using the existing nvdiffrast + Open3D setup.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime
from typing import Literal, Optional, List, Set
from urllib.parse import urlparse

import numpy as np
import os
os.environ["EGL_PLATFORM"] = "surfaceless"   # 让 Mesa 走无窗口 EGL
import open3d as o3d
import requests
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

import nvdiffrast.torch as dr


LOG_FILE = "pipeline_render_sketchfab.log"
API_ROOT = "https://api.sketchfab.com/v3"

fieldnames = [
    "file_identifier",
    "caption",
    "aesthetic_score",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def get_api_token() -> str:
    # token = os.getenv("SKETCHFAB_API_TOKEN")
    # if token:
    #     return token
    return "214f90959eae4743a947ee8e28877139"


def extract_uid_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None
        return parts[-1]
    except Exception:
        return None


def request_download_info(uid: str, token: str) -> Optional[dict]:
    url = f"{API_ROOT}/models/{uid}/download"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        log(f"[WARN] uid={uid} download info failed: HTTP {resp.status_code}")
        return None
    try:
        return resp.json()
    except Exception as exc:
        log(f"[WARN] uid={uid} decode JSON failed: {exc}")
        return None


def choose_download_entry(info: dict) -> Optional[tuple[str, str, str]]:
    if "glb" in info and isinstance(info["glb"], dict) and info["glb"].get("url"):
        return ("glb", info["glb"]["url"], ".glb")
    return None


def download_file(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
        log(
            f"[OK] download complete: {out_path} ({downloaded/1024/1024:.2f} MB, server {total/1024/1024:.2f} MB)"
        )


def captions_non_empty(captions: str) -> bool:
    if captions is None:
        return False
    s = captions.strip()
    if not s:
        return False
    if s == "[]":
        return False
    return True


def parse_caption_field(raw: str) -> str:
    if not captions_non_empty(raw):
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
    except json.JSONDecodeError:
        pass
    return raw.strip()


def prepare_records(csv_path: str, download_dir: str) -> List[dict]:
    records: List[dict] = []
    seen_paths: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            captions_raw = row.get("captions", "")
            if not captions_non_empty(captions_raw):
                continue
            url = row.get("file_identifier", "").strip()
            if not url:
                log(f"[WARN] row {idx}: missing file_identifier")
                continue
            uid = extract_uid_from_url(url)
            if not uid:
                log(f"[WARN] row {idx}: cannot extract uid from {url}")
                continue
            mesh_name = f"{uid}.glb"
            mesh_path = os.path.join(download_dir, mesh_name)
            if mesh_path in seen_paths:
                continue
            record = {
                "sha256": row.get("sha256", ""),
                "url": url,
                "caption": parse_caption_field(captions_raw),
                "aesthetic_score": row.get("aesthetic_score", "0.0"),
                "uid": uid,
                "mesh_path": mesh_path,
                "output_id": os.path.splitext(mesh_name)[0],
            }
            records.append(record)
            seen_paths.add(mesh_path)
    log(f"[INFO] parsed {len(records)} render jobs from {csv_path}")
    return records


def ensure_model_downloaded(record: dict, token: str) -> bool:
    mesh_path = record["mesh_path"]
    if os.path.exists(mesh_path):
        return True
    uid = record["uid"]
    log(f"[INFO] uid={uid} requesting download info...")
    info = request_download_info(uid, token)
    if not info:
        return False
    entry = choose_download_entry(info)
    if not entry:
        log(f"[WARN] uid={uid} has no downloadable glb entry")
        return False
    fmt, dl_url, ext = entry
    out_path = os.path.splitext(mesh_path)[0] + ext
    log(f"[INFO] uid={uid} downloading format={fmt} -> {out_path}")
    try:
        download_file(dl_url, out_path)
        return True
    except Exception as exc:
        log(f"[ERROR] uid={uid} download failed: {exc}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


def blender_to_gl(R_blender, t_blender, K_blender):
    R_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    R_gaussian = R_convert @ R_blender
    t_gaussian = R_convert @ t_blender
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
        [0, 0, 1],
    ])
    world_to_camera = np.linalg.inv(c2w)
    extrinsic = world_to_camera
    R_gaussian, t_gaussian, K_gaussian = blender_to_gl(
        extrinsic[:3, :3], extrinsic[:3, 3:], intrinsic
    )
    extrinsic = np.eye(4, 4)
    extrinsic[:3, :3] = R_gaussian
    extrinsic[:3, 3:] = t_gaussian
    intrinsic = K_gaussian
    return extrinsic, intrinsic


def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def normalize_mesh_verts(vertices):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    vertices_centered = vertices - center
    max_range = (vmax - vmin).max()
    scale = 1.0 / max_range if max_range != 0 else 1.0
    vertices_normalized = vertices_centered * scale
    return vertices_normalized, scale, center


transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def coords_trimesh_to_blender(coords: np.ndarray):
    return (transform_matrix @ coords.T).T


def load_mesh_with_uvs(
    filepath,
    coord_system: Literal["blender", "trimesh"] = "blender",
):
    try:
        mesh = trimesh.load(filepath, process=False, force="mesh")
    except Exception as exc:
        log(f"[ERROR] failed to load mesh {filepath}: {exc}")
        return None, None, None, None, None, None, None

    if not isinstance(mesh, trimesh.Trimesh):
        log(f"[WARN] mesh {filepath} is not a single Trimesh object")
        return None, None, None, None, None, None, None

    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals

    if coord_system == "blender":
        vertices = coords_trimesh_to_blender(vertices)
        normals = coords_trimesh_to_blender(normals)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    vertices_norm, scale, center_bl = normalize_mesh_verts(vertices)

    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
    else:
        uvs = None

    material = mesh.visual.material
    texture_image = getattr(material, "baseColorTexture", None)

    return vertices_norm, faces, uvs, texture_image, normals, scale, center_bl


class ColorRenderer:
    def __init__(self, rendering_options: dict, device: str = "cuda"):
        super().__init__()
        self.rendering_options = rendering_options
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

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

            with torch.autocast("cuda", dtype=torch.float32):
                vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
                vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
                vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
                faces_int = mesh_f.int()

            rast, _ = dr.rasterize(
                self.glctx,
                vertices_clip,
                faces_int,
                (resolution[0] * ssaa, resolution[1] * ssaa),
            )

            depth_image = dr.interpolate(
                vertices_camera[..., 2:3].contiguous(), rast, faces_int
            )[0]

            if mesh_n is not None:
                normal_image = dr.interpolate(normals.contiguous(), rast, faces_int)[0]
                normal_image = F.normalize(normal_image, p=2, dim=-1)
            else:
                normal_image = torch.zeros(
                    (resolution[0] * ssaa, resolution[1] * ssaa, 3), device=self.device
                )

            coords_image = dr.interpolate(vertices.contiguous(), rast, faces_int)[0]
            camera_pos_image = dr.interpolate(vertices_camera[..., :3].contiguous(), rast, faces_int)[0]
            uv_image = dr.interpolate(uvs.contiguous(), rast, faces_int)[0]
            uv_image = torch.cat([uv_image[..., 0:1], 1 - uv_image[..., 1:]], dim=-1)
            texture_image = dr.texture(texture[None, ...], uv_image)

            return coords_image, depth_image, normal_image, camera_pos_image, texture_image


def ffmpeg_silent_with_log(feature_name, output_path, log_path="ffmpeg_log.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if feature_name == "rgb":
        input_pattern = os.path.join(os.path.dirname(output_path), "rgb", "rgb_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
        ]
    elif feature_name == "rgb_pbr":
        input_pattern = os.path.join(os.path.dirname(output_path), "rgb_pbr", "rgb_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    elif feature_name == "depth":
        input_pattern = os.path.join(os.path.dirname(output_path), "depth", "depth_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-vf",
            "format=gray",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
        ]
    elif feature_name == "normal":
        input_pattern = os.path.join(os.path.dirname(output_path), "normal", "normal_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
        ]
    elif feature_name == "mask":
        input_pattern = os.path.join(os.path.dirname(output_path), "mask", "mask_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-vf",
            "format=gray",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
        ]
    elif feature_name == "ccm":
        input_pattern = os.path.join(os.path.dirname(output_path), "ccm", "ccm_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
        ]
    elif feature_name == "position":
        input_pattern = os.path.join(os.path.dirname(output_path), "position", "position_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "16",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
        ]
    else:
        return

    cmd.append(output_path)

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        _write_log(log_path, cmd, "SUCCESS")
    except subprocess.CalledProcessError as exc:
        _write_log(log_path, cmd, f"ERROR: Return code {exc.returncode}")
    except Exception as exc:
        _write_log(log_path, cmd, f"EXCEPTION: {repr(exc)}")


def _write_log(log_path, cmd, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"[{timestamp}] {status}\n")
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")


def build_camera_path(frames: int, fov_deg: float):
    transform_mtx = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -3.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    c2w_bl0 = np.array(transform_mtx, dtype=np.float32)
    extr0, intr0 = get_intrinsic_extrinsic(c2w_bl0, fov_deg)

    n_bl = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    n_bl /= (np.linalg.norm(n_bl) + 1e-9)

    def proj_to_plane(v, n):
        return v - (v @ n) * n

    C_bl0 = c2w_bl0[:3, 3]
    C_bl0_proj = proj_to_plane(C_bl0, n_bl)
    r = np.linalg.norm(C_bl0_proj)
    if r < 1e-8:
        raise ValueError("camera origin projection length is zero, cannot build path")

    u1 = C_bl0_proj / r
    u2 = np.cross(n_bl, u1)
    u2 /= (np.linalg.norm(u2) + 1e-9)

    world_up_bl = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def build_c2w(C_bl: np.ndarray) -> np.ndarray:
        forward = -C_bl
        forward /= (np.linalg.norm(forward) + 1e-9)
        Z = -forward
        X = np.cross(world_up_bl, Z)
        xnorm = np.linalg.norm(X)
        if xnorm < 1e-8:
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            X = np.cross(alt_up, Z)
            xnorm = np.linalg.norm(X)
        X /= (xnorm + 1e-9)
        Y = np.cross(Z, X)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([X, Y, Z], axis=1)
        c2w[:3, 3] = C_bl
        return c2w

    extr_list = []
    frames_meta = []
    for i in range(frames):
        theta = 2.0 * math.pi * (i / frames)
        C_bl = r * (math.cos(theta) * u1 + math.sin(theta) * u2)
        c2w_bl = build_c2w(C_bl)
        extr_i, _ = get_intrinsic_extrinsic(c2w_bl, fov_deg)
        extr_list.append(extr_i)
        frames_meta.append({"frame_id": i, "transform_matrix": c2w_bl.tolist()})

    return extr_list, intr0, frames_meta


def render_models(records: List[dict], args, token: str) -> None:
    if not records:
        log("[INFO] no records to render")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    extr_np_list, intr_np, frames_meta = build_camera_path(args.frames, args.fov)
    intr0_t = torch.from_numpy(intr_np).float().to(args.device)
    extr_tensors = [torch.from_numpy(extr).float().to(args.device) for extr in extr_np_list]

    render_options = {
        "resolution": [args.resolution_x, args.resolution_y],
        "near": 0.01,
        "far": 1000,
        "ssaa": 1,
    }

    renderer = ColorRenderer(rendering_options=render_options, device=args.device)
    o3d_renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.resolution_x, args.resolution_y
    )
    o3d_scene = o3d_renderer.scene
    o3d_scene.set_background([0.0, 0.0, 0.0, 1.0])
    o3d_scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        np.array([0.577, -0.577, -0.577], dtype=np.float32),
    )

    out_csv = args.out_csv_path
    existing_ids: Set[str] = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8") as f:
            existing_ids = {row["file_identifier"] for row in csv.DictReader(f)}

    ffmpeg_log_path = os.path.join(args.out_dir, "ffmpeg_log.txt")

    for record in tqdm(records, desc="Rendering meshes"):
        file_identifier = record["output_id"]
        mesh_path = record["mesh_path"]
        if file_identifier in existing_ids:
            log(f"[INFO] skip existing render: {file_identifier}")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            continue

        already_present = os.path.exists(mesh_path)
        if not ensure_model_downloaded(record, token):
            log(f"[WARN] unable to download mesh for uid={record['uid']}")
            continue
        if args.sleep > 0 and not already_present:
            time.sleep(args.sleep)

        if not os.path.exists(mesh_path):
            log(f"[WARN] missing mesh after download: {mesh_path}")
            continue

        file_output_path = os.path.join(args.out_dir, file_identifier)
        os.makedirs(file_output_path, exist_ok=True)
        for feature in ["depth", "mask", "normal", "position", "ccm", "rgb", "rgb_pbr"]:
            os.makedirs(os.path.join(file_output_path, feature), exist_ok=True)

        mesh_v, mesh_f, mesh_uv, texture, mesh_n, norm_scale, norm_center_bl = load_mesh_with_uvs(
            mesh_path
        )
        if mesh_v is None or mesh_uv is None or texture is None or mesh_n is None:
            log(f"[WARN] incomplete mesh data for {mesh_path}, skipping")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            continue

        mesh_v_t = torch.from_numpy(mesh_v).float().to(args.device)
        mesh_f_t = torch.from_numpy(mesh_f).int().to(args.device)
        mesh_uv_t = torch.from_numpy(mesh_uv).float().to(args.device)
        texture_np = np.asarray(texture)
        texture_t = torch.from_numpy(texture_np).float().to(args.device) / 255.0
        mesh_n_t = torch.from_numpy(mesh_n).float().to(args.device)

        o3d_model_loaded = False
        try:
            if hasattr(o3d_scene, "clear_geometry"):
                o3d_scene.clear_geometry()
            else:
                for name in list(o3d_scene.get_geometry_names()):
                    o3d_scene.remove_geometry(name)
            o3d_model = o3d.io.read_triangle_model(mesh_path)
            o3d_scene.add_model("model", o3d_model)
            for mat in o3d_model.materials:
                # 关闭 normal map
                if hasattr(mat, "normal_img"):
                    mat.normal_img = None

                # 关闭透明
                if hasattr(mat, "has_alpha"):
                    mat.has_alpha = False

                # base_color 把 alpha 拉满（防止材质本身带透明）
                if hasattr(mat, "base_color"):
                    # 有的版本是类似 np.ndarray / Eigen 向量，先转成 list 再改
                    bc = list(mat.base_color)
                    if len(bc) == 4:
                        bc[3] = 1.0
                    mat.base_color = bc

                # **关键：关掉 PBR 透射 / 玻璃效果**
                if hasattr(mat, "transmission"):
                    mat.transmission = 0.0   # 不再透过去
                if hasattr(mat, "thickness"):
                    mat.thickness = 0.0      # 没有体积感
                if hasattr(mat, "absorption_distance"):
                    mat.absorption_distance = 1e9  # 形同无吸收
                if hasattr(mat, "absorption_color"):
                    mat.absorption_color = [1.0, 1.0, 1.0]

                # 强制用不透明 PBR shader（避免带透明/SSR 的变体）
                if hasattr(mat, "shader"):
                    mat.shader = "defaultLit"
            R_tb = transform_matrix.astype(np.float64)
            center_bl = norm_center_bl.astype(np.float64)
            s = float(norm_scale)
            T_model = np.eye(4, dtype=np.float64)
            T_model[:3, :3] = s * R_tb
            T_model[:3, 3] = -s * center_bl
            o3d_scene.set_geometry_transform("model", T_model)
            o3d_model_loaded = True
        except Exception as exc:
            log(f"[WARN] Open3D load failed for {mesh_path}: {exc}")

        frames_rendered = 0
        for i, extr_t in enumerate(extr_tensors):
            pos_img, depth_img, normal_img, camera_pos_image, texture_image = renderer(
                extr_t, intr0_t, mesh_v_t, mesh_f_t, mesh_uv_t, texture_t, mesh_n=mesh_n_t
            )

            depth_np = depth_img[0].detach().cpu().numpy()
            valid = depth_np > 0
            if valid.any():
                d_min, d_max = depth_np[valid].min(), depth_np[valid].max()
                depth_norm = (d_max - depth_np) / (d_max - d_min + 1e-9)
                depth_norm[~valid] = 0.0
                mask_norm = depth_norm.copy()
                mask_norm[valid] = 1.0
            else:
                depth_norm = np.zeros_like(depth_np)
                mask_norm = np.zeros_like(depth_np)
            valid_3 = np.repeat(valid, 3, axis=-1)

            if args.render_depth:
                depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
                Image.fromarray(depth_vis).save(
                    os.path.join(file_output_path, "depth", f"depth_{i:04d}.png")
                )

            if args.render_mask:
                mask_vis = (mask_norm * 255.0).clip(0, 255).astype(np.uint8).squeeze()
                Image.fromarray(mask_vis).save(
                    os.path.join(file_output_path, "mask", f"mask_{i:04d}.png")
                )

            if args.render_position:
                pos_vis = ((pos_img[0].detach().cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype(
                    np.uint8
                )
                pos_vis[~valid_3] = 0
                Image.fromarray(pos_vis).save(
                    os.path.join(file_output_path, "position", f"position_{i:04d}.png")
                )

            if args.render_ccm:
                camera_pos_np = camera_pos_image[0].detach().cpu().numpy()
                ccm_vis = ((camera_pos_np + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                ccm_vis[~valid_3] = 0
                Image.fromarray(ccm_vis).save(
                    os.path.join(file_output_path, "ccm", f"ccm_{i:04d}.png")
                )

            if args.render_normal:
                normal_np = normal_img[0].detach().cpu().numpy()
                normal_vis = ((normal_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                normal_vis[~valid_3] = 0
                Image.fromarray(normal_vis).save(
                    os.path.join(file_output_path, "normal", f"normal_{i:04d}.png")
                )

            if args.render_rgb:
                rgb_vis = (texture_image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(
                    np.uint8
                )
                if rgb_vis.shape[2] == 3:
                    rgb_vis[~valid_3] = 0
                else:
                    rgba_vis = rgb_vis.copy()
                    rgb_vis = rgba_vis[..., :3]
                    alpha_channel = rgba_vis[..., 3] / 255.0
                    rgb_vis[..., 0] = rgb_vis[..., 0] * alpha_channel
                    rgb_vis[..., 1] = rgb_vis[..., 1] * alpha_channel
                    rgb_vis[..., 2] = rgb_vis[..., 2] * alpha_channel
                    rgb_vis[~valid_3] = 0
                Image.fromarray(rgb_vis).save(
                    os.path.join(file_output_path, "rgb", f"rgb_{i:04d}.png")
                )

            if o3d_model_loaded:
                c2w_np = frames_meta[i]["transform_matrix"]
                c2w_np = np.array(c2w_np, dtype=np.float32)
                eye = c2w_np[:3, 3]
                up = c2w_np[:3, 1]
                center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                o3d_renderer.setup_camera(args.fov, center, eye, up)
                img_o3d = o3d_renderer.render_to_image()
                o3d.io.write_image(
                    os.path.join(file_output_path, "rgb_pbr", f"rgb_{i:04d}.png"),
                    img_o3d,
                )

            frames_rendered += 1

        if o3d_model_loaded:
            try:
                o3d_scene.remove_geometry("model")
            except Exception as exc:
                log(f"[WARN] remove_geometry failed for {file_identifier}: {exc}")

        if frames_rendered == 0:
            log(f"[WARN] no frames rendered for {file_identifier}")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            continue

        if args.render_rgb:
            ffmpeg_silent_with_log("rgb", os.path.join(file_output_path, "rgb.mp4"), ffmpeg_log_path)
            ffmpeg_silent_with_log(
                "rgb_pbr", os.path.join(file_output_path, "rgb_pbr.mp4"), ffmpeg_log_path
            )
        if args.render_depth:
            ffmpeg_silent_with_log("depth", os.path.join(file_output_path, "depth.mp4"), ffmpeg_log_path)
        if args.render_normal:
            ffmpeg_silent_with_log(
                "normal", os.path.join(file_output_path, "normal.mp4"), ffmpeg_log_path
            )
        if args.render_position:
            ffmpeg_silent_with_log(
                "position", os.path.join(file_output_path, "position.mp4"), ffmpeg_log_path
            )
        if args.render_ccm:
            ffmpeg_silent_with_log("ccm", os.path.join(file_output_path, "ccm.mp4"), ffmpeg_log_path)
        if args.render_mask:
            ffmpeg_silent_with_log("mask", os.path.join(file_output_path, "mask.mp4"), ffmpeg_log_path)

        single_out = {
            "file_identifier": file_identifier,
            "caption": record["caption"],
            "aesthetic_score": record["aesthetic_score"],
        }
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(single_out)
        existing_ids.add(file_identifier)
        log(f"[OK] rendered {file_identifier}")

        if os.path.exists(mesh_path):
            try:
                os.remove(mesh_path)
            except OSError as exc:
                log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")

    camera_json = {
        "fov_degrees": args.fov,
        "resolution": [args.resolution_x, args.resolution_y],
        "frames": frames_meta,
    }
    with open(os.path.join(args.out_dir, "cameras_blender.json"), "w", encoding="utf-8") as f:
        json.dump(camera_json, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Download + render pipeline")
    parser.add_argument("--csv", required=True, help="Path to ObjaverseXL_sketchfab.csv")
    parser.add_argument("--download-dir", required=True, help="Directory to store downloaded GLBs")
    parser.add_argument("--out-dir", required=True, help="Output directory for renders")
    parser.add_argument("--out-csv-path", default="objaversexl_sketchfab/rendered_metadata.csv", help="Output CSV path")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between download requests")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames to render")
    parser.add_argument("--fov", type=float, default=39.6, help="Camera field of view (degrees)")
    parser.add_argument("--resolution_x", type=int, default=1024, help="Render resolution width")
    parser.add_argument("--resolution_y", type=int, default=1024, help="Render resolution height")
    parser.add_argument("--render_rgb", type=bool, default=True, help="Render RGB views")
    parser.add_argument("--render_depth", type=bool, default=True, help="Render depth maps")
    parser.add_argument("--render_normal", type=bool, default=True, help="Render normals")
    parser.add_argument("--render_position", type=bool, default=True, help="Render positions")
    parser.add_argument("--render_ccm", type=bool, default=True, help="Render camera coordinate maps")
    parser.add_argument("--render_mask", type=bool, default=True, help="Render masks")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use (e.g. cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    records = prepare_records(args.csv, args.download_dir)
    if not records:
        log("[INFO] no records after parsing CSV; exiting")
        return

    token = get_api_token()
    render_models(records, args, token)


if __name__ == "__main__":
    main()
