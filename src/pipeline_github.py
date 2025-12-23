#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end pipeline that downloads GitHub-hosted models from a CSV manifest
and then renders per-model assets using the existing nvdiffrast + Open3D setup.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import shutil
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from types import SimpleNamespace
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
import psutil


LOG_FILE = "pipeline_render_github.log"

METADATA_FIELDNAMES = [
    "file_identifier",
    "caption",
    "aesthetic_score",
]

FAILURE_FIELDNAMES = [
    "file_identifier",
    "caption",
    "aesthetic_score",
    "reason",
]

MAX_MESH_BYTES = 25 * 1024 * 1024  # 25 MB hard limit for downloaded meshes


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def log_main_memory(note: str = "") -> None:
    proc = psutil.Process(os.getpid())
    mem_gb = proc.memory_info().rss / (1024 ** 3)
    note_suffix = f" ({note})" if note else ""
    log(f"[DEBUG] main pid={proc.pid} rss={mem_gb:.2f}GB{note_suffix}")


def extract_uid_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None
        last = parts[-1]
        if "." in last:
            last = os.path.splitext(last)[0]
        return last
    except Exception:
        return None


def to_github_raw_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception as exc:
        log(f"[WARN] failed to parse URL {url}: {exc}")
        return None

    netloc = parsed.netloc.lower()
    path_parts = [p for p in parsed.path.split("/") if p]
    cleaned = parsed._replace(query="", fragment="", params="").geturl()

    if "raw.githubusercontent.com" in netloc:
        return cleaned

    if "github.com" in netloc:
        if len(path_parts) >= 5 and path_parts[2] in {"blob", "raw"}:
            owner, repo, _, branch = path_parts[:4]
            file_path = "/".join(path_parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        return cleaned if path_parts else None

    if parsed.scheme in {"http", "https"}:
        return cleaned

    log(f"[WARN] unsupported URL scheme for {url}")
    return None


def derive_mesh_name(url: str, uid: Optional[str]) -> str:
    basename = ""
    try:
        parsed = urlparse(url)
        basename = os.path.basename(parsed.path)
    except Exception:
        basename = ""

    if basename:
        if not basename.lower().endswith(".glb"):
            basename = f"{basename}.glb"
        return basename

    if uid:
        return f"{uid}.glb"

    timestamp = int(time.time() * 1000)
    return f"model_{timestamp}.glb"


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
            uid = extract_uid_from_url(url)
            mesh_name = derive_mesh_name(url, uid)
            mesh_path = os.path.join(download_dir, mesh_name)
            if mesh_path in seen_paths:
                continue
            sha256 = row.get("sha256", "").strip()
            record = {
                "sha256": sha256,
                "url": url,
                "caption": parse_caption_field(captions_raw),
                "aesthetic_score": row.get("aesthetic_score", "0.0"),
                "uid": uid,
                "mesh_path": mesh_path,
                "output_id": sha256,
            }
            records.append(record)
            seen_paths.add(mesh_path)
    log(f"[INFO] parsed {len(records)} render jobs from {csv_path}")
    return records


def ensure_model_downloaded(record: dict) -> bool:
    mesh_path = record["mesh_path"]
    if os.path.exists(mesh_path):
        return True

    source_url = record["url"]
    raw_url = to_github_raw_url(source_url)
    if not raw_url:
        log(f"[WARN] uid={record['uid']} has unsupported URL: {source_url}")
        return False

    log(f"[INFO] uid={record['uid']} downloading from GitHub -> {mesh_path}")
    try:
        download_file(raw_url, mesh_path)
        return True
    except Exception as exc:
        log(f"[ERROR] uid={record['uid']} download failed: {exc}")
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
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
    scale = 1.0 / max_range if max_range != 0 else np.inf
    vertices_normalized = vertices_centered * scale
    return vertices_normalized, scale, center


transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def coords_trimesh_to_blender(coords: np.ndarray):
    return (transform_matrix @ coords.T).T


def _clear_o3d_scene(o3d_scene) -> None:
    try:
        if hasattr(o3d_scene, "clear_geometry"):
            o3d_scene.clear_geometry()
            return
    except Exception:
        pass

    try:
        names = list(o3d_scene.get_geometry_names())
    except Exception:
        names = []

    for name in names:
        try:
            o3d_scene.remove_geometry(name)
        except Exception:
            pass


def _aabb_is_valid(bounds: np.ndarray) -> bool:
    bounds = np.asarray(bounds)
    if bounds.shape != (2, 3):
        return False
    if not np.isfinite(bounds).all():
        return False
    extent = bounds[1] - bounds[0]
    # Filament 的 empty AABB 通常来自“无有效几何/NaN/Inf”，而不是单轴厚度为 0。
    if (extent < 0).any():
        return False
    return np.max(extent) > 0


def _validate_trimesh_geometry(geom: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        vertices = np.asarray(geom.vertices)
        faces = np.asarray(geom.faces)
    except Exception:
        return False, "mesh_geometry_array_error"

    if vertices.size == 0 or vertices.shape[0] == 0:
        return False, "mesh_no_vertices"
    if faces.size == 0 or faces.shape[0] == 0:
        return False, "mesh_no_faces"
    if not np.isfinite(vertices).all():
        return False, "mesh_vertices_nonfinite"

    try:
        bounds = np.asarray(geom.bounds)
    except Exception:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        bounds = np.stack([vmin, vmax], axis=0)

    if not _aabb_is_valid(bounds):
        return False, "mesh_invalid_aabb"

    return True, ""


def validate_mesh_file_for_render(mesh_path: str) -> tuple[bool, str]:
    """在进入 Open3D/Filament 之前做一次离线预检，避免 empty AABB 触发崩溃。"""
    try:
        obj = trimesh.load(mesh_path, process=False)
    except Exception as exc:
        return False, f"mesh_trimesh_load_failed:{type(exc).__name__}"

    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            return False, "mesh_scene_empty"
        for name, geom in obj.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            ok, reason = _validate_trimesh_geometry(geom)
            if not ok:
                return False, f"mesh_invalid_submesh:{name}:{reason}"
        return True, ""

    if isinstance(obj, trimesh.Trimesh):
        return _validate_trimesh_geometry(obj)

    return False, f"mesh_unsupported_type:{type(obj).__name__}"


def _iter_o3d_triangle_model_meshes(o3d_model):
    meshes = getattr(o3d_model, "meshes", None)
    if meshes is None:
        return
    for idx, item in enumerate(meshes):
        mesh = getattr(item, "mesh", None)
        if mesh is None:
            mesh = item
        yield idx, mesh


def validate_o3d_triangle_model_for_filament(o3d_model) -> tuple[bool, str]:
    """检查 Open3D TriangleModel 的子网格是否会触发 Filament empty AABB。"""
    bad_reasons: list[str] = []
    for idx, mesh in _iter_o3d_triangle_model_meshes(o3d_model):
        try:
            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.triangles)
        except Exception as exc:
            bad_reasons.append(f"submesh[{idx}]:mesh_array_error:{type(exc).__name__}")
            continue

        if v.size == 0 or v.shape[0] == 0:
            bad_reasons.append(f"submesh[{idx}]:empty_vertices")
            continue
        if f.size == 0 or f.shape[0] == 0:
            bad_reasons.append(f"submesh[{idx}]:empty_triangles")
            continue
        if not np.isfinite(v).all():
            bad_reasons.append(f"submesh[{idx}]:nonfinite_vertices")
            continue

        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        bounds = np.stack([vmin, vmax], axis=0)
        if not _aabb_is_valid(bounds):
            bad_reasons.append(f"submesh[{idx}]:invalid_aabb")
            continue

    if bad_reasons:
        # 合并成一条 reason，便于写入 failed csv
        joined = ";".join(bad_reasons[:8])
        if len(bad_reasons) > 8:
            joined += f";...(+{len(bad_reasons)-8})"
        return False, f"mesh_o3d_invalid:{joined}"

    return True, ""


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

    if vertices is None or len(vertices) == 0:
        log(f"[WARN] mesh {filepath} has no vertices")
        return None, None, None, None, None, None, None
    if faces is None or len(faces) == 0:
        log(f"[WARN] mesh {filepath} has no faces")
        return None, None, None, None, None, None, None
    if not np.isfinite(vertices).all():
        log(f"[WARN] mesh {filepath} has non-finite vertices")
        return None, None, None, None, None, None, None

    if coord_system == "blender":
        vertices = coords_trimesh_to_blender(vertices)
        normals = coords_trimesh_to_blender(normals)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    vertices_norm, scale, center_bl = normalize_mesh_verts(vertices)

    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
    else:
        uvs = None

    if uvs is not None:
        try:
            if len(uvs) != len(vertices):
                log(f"[WARN] mesh {filepath} uv count mismatch (uv={len(uvs)} v={len(vertices)})")
                uvs = None
        except Exception:
            uvs = None

    try:
        material = mesh.visual.material
        texture_image = getattr(material, "baseColorTexture", None)
    except Exception as exc:
        log(f"[WARN] failed to get material or texture for mesh {filepath}: {exc}")
        texture_image = None

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


_WORKER_CONTEXT = {}


def _failure_result(record: dict, reason: str = "") -> dict:
    return {
        "status": "fail",
        "file_identifier": record.get("output_id") or record.get("sha256") or "",
        "caption": record.get("caption", ""),
        "aesthetic_score": record.get("aesthetic_score", ""),
        "reason": reason,
    }


def _cleanup_failure_outputs(file_output_path: Optional[str], mesh_path: Optional[str]) -> None:
    # Remove generated outputs and the downloaded mesh when a case fails
    try:
        if file_output_path and os.path.isdir(file_output_path):
            shutil.rmtree(file_output_path, ignore_errors=True)
    except Exception:
        pass

    try:
        if mesh_path and os.path.exists(mesh_path):
            os.remove(mesh_path)
    except Exception:
        pass


def _init_worker(config: dict) -> None:
    global _WORKER_CONTEXT
    args = SimpleNamespace(**config["args_dict"])
    intr0_t = torch.from_numpy(config["intr_np"]).float().to(args.device)
    extr_tensors = [torch.from_numpy(extr).float().to(args.device) for extr in config["extr_np_list"]]
    renderer = ColorRenderer(rendering_options=config["render_options"], device=args.device)
    o3d_renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.resolution_x, args.resolution_y
    )
    o3d_scene = o3d_renderer.scene
    o3d_scene.set_background([0.0, 0.0, 0.0, 1.0])
    o3d_scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        np.array([0.577, -0.577, -0.577], dtype=np.float32),
    )

    _WORKER_CONTEXT = {
        "args": args,
        "intr0_t": intr0_t,
        "extr_tensors": extr_tensors,
        "frames_meta": config["frames_meta"],
        "renderer": renderer,
        "o3d_renderer": o3d_renderer,
        "o3d_scene": o3d_scene,
        "ffmpeg_log_path": config["ffmpeg_log_path"],
    }


def _process_record(record: dict) -> Optional[dict]:
    ctx = _WORKER_CONTEXT
    args: SimpleNamespace = ctx["args"]
    renderer: ColorRenderer = ctx["renderer"]
    o3d_scene = ctx["o3d_scene"]
    o3d_renderer = ctx["o3d_renderer"]
    intr0_t = ctx["intr0_t"]
    extr_tensors = ctx["extr_tensors"]
    frames_meta = ctx["frames_meta"]
    ffmpeg_log_path = ctx["ffmpeg_log_path"]

    file_identifier = record["output_id"]
    mesh_path = record["mesh_path"]
    file_output_path = os.path.join(args.out_dir, file_identifier)
    
    # test memory usage
    proc = psutil.Process(os.getpid())

    try:
        # 关键：每个 record 开始先强制清场，避免上一个 record 残留导致重叠渲染
        _clear_o3d_scene(o3d_scene)

        already_present = os.path.exists(mesh_path)
        if not ensure_model_downloaded(record):
            log(f"[WARN] unable to download mesh for uid={record['uid']}")
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "download_failed")
        if args.sleep > 0 and not already_present:
            time.sleep(args.sleep)

        if not os.path.exists(mesh_path):
            log(f"[WARN] missing mesh after download: {mesh_path}")
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "mesh_missing_after_download")

        # Reject overly large meshes to avoid OOM
        mesh_size = os.path.getsize(mesh_path)
        if mesh_size > MAX_MESH_BYTES:
            log(
                f"[WARN] mesh too large ({mesh_size/1024/1024:.2f} MB) for {file_identifier}, skipping"
            )
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "model_too_large")

        ok, reason = validate_mesh_file_for_render(mesh_path)
        if not ok:
            log(f"[WARN] invalid mesh geometry for {mesh_path}: {reason}")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, reason)

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
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "mesh_incomplete_data")

        if np.isinf(norm_scale) or np.isnan(norm_scale) or norm_scale <= 0:
            log(f"[WARN] invalid normalization scale for {mesh_path}, skipping")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "mesh_invalid_normalization")

        # 关键：用 Open3D 路径预检 TriangleModel 的子网格，捕获 trimesh 拼接看不出的坏 primitive。
        # 发现会触发 Filament empty AABB 的子网格则直接 fail，避免崩溃 & 避免场景残留。
        try:
            o3d_model_pre = o3d.io.read_triangle_model(mesh_path)
            ok_o3d, reason_o3d = validate_o3d_triangle_model_for_filament(o3d_model_pre)
            if not ok_o3d:
                log(f"[WARN] Open3D TriangleModel preflight failed for {mesh_path}: {reason_o3d}")
                _cleanup_failure_outputs(file_output_path, mesh_path)
                return _failure_result(record, reason_o3d)
        except Exception as exc:
            log(f"[WARN] Open3D TriangleModel preflight error for {mesh_path}: {exc}")
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, f"mesh_o3d_preflight_error:{type(exc).__name__}")

        mesh_v_t = torch.from_numpy(mesh_v).float().to(args.device)
        mesh_f_t = torch.from_numpy(mesh_f).int().to(args.device)
        mesh_uv_t = torch.from_numpy(mesh_uv).float().to(args.device)
        texture_np = np.asarray(texture)
        texture_t = torch.from_numpy(texture_np).float().to(args.device) / 255.0
        mesh_n_t = torch.from_numpy(mesh_n).float().to(args.device)

        o3d_model_loaded = False
        try:
            _clear_o3d_scene(o3d_scene)
            # 复用预检时加载的 model，避免重复解析
            o3d_model = o3d_model_pre
            o3d_scene.add_model("model", o3d_model)
            o3d_model_loaded = True
            for mat in o3d_model.materials:
                if hasattr(mat, "normal_img"):
                    mat.normal_img = None
                if hasattr(mat, "has_alpha"):
                    mat.has_alpha = False
                if hasattr(mat, "base_color"):
                    bc = list(mat.base_color)
                    if len(bc) == 4:
                        bc[3] = 1.0
                    mat.base_color = bc
                if hasattr(mat, "transmission"):
                    mat.transmission = 0.0
                if hasattr(mat, "thickness"):
                    mat.thickness = 0.0
                if hasattr(mat, "absorption_distance"):
                    mat.absorption_distance = 1e9
                if hasattr(mat, "absorption_color"):
                    mat.absorption_color = [1.0, 1.0, 1.0]
                if hasattr(mat, "shader"):
                    mat.shader = "defaultLit"
            R_tb = transform_matrix.astype(np.float64)
            center_bl = norm_center_bl.astype(np.float64)
            s = float(norm_scale)
            T_model = np.eye(4, dtype=np.float64)
            T_model[:3, :3] = s * R_tb
            T_model[:3, 3] = -s * center_bl
            o3d_scene.set_geometry_transform("model", T_model)
        except Exception as exc:
            log(f"[WARN] Open3D load failed for {mesh_path}: {exc}")
            # 即使中间步骤失败，也确保不残留几何体
            _clear_o3d_scene(o3d_scene)
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, f"mesh_o3d_add_model_failed:{type(exc).__name__}")

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
                log(
                    f"[WARN] no valid depth for {file_identifier} at frame {i}, "
                    "treat as render failure"
                )
                if o3d_model_loaded:
                    try:
                        o3d_scene.remove_geometry("model")
                    except Exception as exc:
                        log(f"[WARN] remove_geometry failed for {file_identifier}: {exc}")
                _cleanup_failure_outputs(file_output_path, mesh_path)
                return _failure_result(record, f"no_valid_depth_frame_{i}")
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
            _cleanup_failure_outputs(file_output_path, mesh_path)
            return _failure_result(record, "no_frames_rendered")

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

        if os.path.exists(mesh_path):
            try:
                os.remove(mesh_path)
            except OSError as exc:
                log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")

        log(f"[OK] rendered {file_identifier}")

        # test 每处理完一个 model 打印当前进程 RSS
        mem_gb = proc.memory_info().rss / (1024 ** 3)
        log(f"[DEBUG] pid={proc.pid} rss={mem_gb:.2f}GB after {file_identifier}")

        return {
            "status": "success",
            "file_identifier": file_identifier,
            "caption": record["caption"],
            "aesthetic_score": record["aesthetic_score"],
        }

    except Exception as exc:
        log(
            f"[ERROR] worker failure for {record.get('output_id', 'unknown')}: {exc}\n{traceback.format_exc()}"
        )
        if os.path.exists(mesh_path):
            try:
                os.remove(mesh_path)
            except OSError:
                pass
        _cleanup_failure_outputs(file_output_path, mesh_path)
        return _failure_result(record, f"exception:{exc}")

    finally:
        # 无论成功/失败/中途 return，都强制清场，避免同进程后续模型叠加渲染
        try:
            _clear_o3d_scene(o3d_scene)
        except Exception:
            pass


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
    except subprocess.CalledProcessError as exc:
        # Suppress ffmpeg logging; only fail silently on non-zero return codes
        pass
    except Exception as exc:
        # Suppress ffmpeg logging for unexpected exceptions as well
        pass

    # After video is generated (or attempted), remove the source image folder
    frame_dir = os.path.dirname(input_pattern)
    try:
        if os.path.isdir(frame_dir):
            for root, dirs, files in os.walk(frame_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            os.rmdir(frame_dir)
    except OSError:
        pass


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


def render_models(records: List[dict], args) -> None:
    if not records:
        log("[INFO] no records to render")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    extr_np_list, intr_np, frames_meta = build_camera_path(args.frames, args.fov)
    render_options = {
        "resolution": [args.resolution_x, args.resolution_y],
        "near": 0.01,
        "far": 1000,
        "ssaa": 1,
    }

    out_csv = args.out_csv_path
    failed_csv = args.failed_csv_path

    existing_ids: Set[str] = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8") as f:
            existing_ids = {row.get("file_identifier", "") for row in csv.DictReader(f)}

    failed_ids: Set[str] = set()
    if os.path.exists(failed_csv):
        with open(failed_csv, "r", encoding="utf-8") as f:
            failed_ids = {row.get("file_identifier", "") for row in csv.DictReader(f)}

    ffmpeg_log_path = os.path.join(args.out_dir, "ffmpeg_log.txt")
    records_to_process: List[dict] = []
    for record in records:
        file_identifier = record["output_id"]
        mesh_path = record["mesh_path"]
        if file_identifier in existing_ids or file_identifier in failed_ids:
            log(f"[INFO] skip existing render: {file_identifier}")
            if os.path.exists(mesh_path):
                try:
                    os.remove(mesh_path)
                except OSError as exc:
                    log(f"[WARN] failed to delete mesh {mesh_path}: {exc}")
            continue
        records_to_process.append(record)

    if records_to_process:
        for csv_path in [out_csv, failed_csv]:
            dir_path = os.path.dirname(csv_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        worker_config = {
            "args_dict": vars(args).copy(),
            "extr_np_list": extr_np_list,
            "intr_np": intr_np,
            "frames_meta": frames_meta,
            "render_options": render_options,
            "ffmpeg_log_path": ffmpeg_log_path,
        }
        worker_count = max(1, min(args.workers, len(records_to_process)))
        ctx = get_context("spawn")
        
        log_main_memory("before_pool")
        
        with ctx.Pool(
            processes=worker_count, initializer=_init_worker, initargs=(worker_config,), maxtasksperchild=100
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_record, records_to_process),
                total=len(records_to_process),
                desc="Rendering meshes",
            ):
                if not result:
                    continue

                status = result.get("status")
                file_id = result.get("file_identifier", "")

                if status == "success":
                    if file_id in existing_ids:
                        continue
                    with open(out_csv, "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow({
                            "file_identifier": result.get("file_identifier", ""),
                            "caption": result.get("caption", ""),
                            "aesthetic_score": result.get("aesthetic_score", ""),
                        })
                    existing_ids.add(file_id)
                elif status == "fail":
                    if file_id in existing_ids or file_id in failed_ids:
                        continue
                    with open(failed_csv, "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDNAMES)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow({
                            "file_identifier": file_id,
                            "caption": result.get("caption", ""),
                            "aesthetic_score": result.get("aesthetic_score", ""),
                            "reason": result.get("reason", ""),
                        })
                    failed_ids.add(file_id)
    else:
        log("[INFO] no new records require rendering")

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
    parser.add_argument("--out-csv-path", default="objaversexl_github/rendered_metadata.csv", help="Output CSV path")
    parser.add_argument("--failed-csv-path", default="objaversexl_github/render_failed_metadata.csv", help="Output CSV path for failed renders")
    parser.add_argument("--download-dir", required=True, help="Directory to store downloaded GLBs")
    parser.add_argument("--out-dir", required=True, help="Output directory for renders")
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
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel worker processes")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    log_main_memory("before_render")

    records = prepare_records(args.csv, args.download_dir)
    
    log_main_memory("After preparing records")

    render_models(records, args)



if __name__ == "__main__":
    main()
