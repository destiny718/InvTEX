import argparse
import base64
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from pygltflib import GLTF2, Image as GLTFImage

def linear_to_srgb_arr(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = 0.055
    srgb = np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * np.power(np.clip(x, 0.0, 1.0) + eps, 1/2.4) - a
    )
    return np.clip(srgb, 0.0, 1.0)

def load_png_as_data_uri(texture_path: str, assume_linear: bool) -> str:
    img = Image.open(texture_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    if assume_linear:
        arr = linear_to_srgb_arr(arr)
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_img = Image.fromarray(arr_u8, mode="RGB")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def ensure_list(x):
    return [] if x is None else x

def add_sampler_trilinear_repeat(gltf: GLTF2) -> int:
    gltf.samplers = ensure_list(gltf.samplers)
    sampler = {
        "magFilter": 9729,   # LINEAR
        "minFilter": 9987,   # LINEAR_MIPMAP_LINEAR
        "wrapS": 10497,      # REPEAT
        "wrapT": 10497       # REPEAT
    }
    gltf.samplers.append(sampler)
    return len(gltf.samplers) - 1

def add_texture_with_image(gltf: GLTF2, data_uri: str, sampler_index: int) -> int:
    gltf.images = ensure_list(gltf.images)
    img = GLTFImage(uri=data_uri)
    gltf.images.append(img)
    image_index = len(gltf.images) - 1

    gltf.textures = ensure_list(gltf.textures)
    tex = {"source": image_index, "sampler": sampler_index}
    gltf.textures.append(tex)
    return len(gltf.textures) - 1

def add_normal_texture_if_any(gltf: GLTF2, normal_png: Optional[str], sampler_index: int) -> Optional[int]:
    if normal_png is None:
        return None
    norm_uri = load_png_as_data_uri(normal_png, assume_linear=False)  # normal为线性域，不做伽马
    gltf.images = ensure_list(gltf.images)
    gltf.images.append(GLTFImage(uri=norm_uri))
    nimg_idx = len(gltf.images) - 1

    gltf.textures = ensure_list(gltf.textures)
    ntex = {"source": nimg_idx, "sampler": sampler_index}
    gltf.textures.append(ntex)
    return len(gltf.textures) - 1

def set_material_pbr(gltf: GLTF2,
                     basecolor_tex_index: int,
                     metallic: float,
                     roughness: float,
                     normal_tex_index: Optional[int]) -> int:
    gltf.materials = ensure_list(gltf.materials)
    mat = {
        "name": "TexturedMaterial",
        "doubleSided": False,
        "alphaMode": "OPAQUE",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": basecolor_tex_index},
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
            # "baseColorFactor": [1,1,1,1]  # 如需整体乘因子，可解注
        }
    }
    if normal_tex_index is not None:
        # normalTexture: index + 可选scale
        mat["normalTexture"] = {"index": normal_tex_index, "scale": 1.0}

    # 确保不是 unlit
    if "extensions" in mat and isinstance(mat["extensions"], dict):
        mat["extensions"].pop("KHR_materials_unlit", None)

    gltf.materials.append(mat)
    return len(gltf.materials) - 1

def apply_material_to_all_primitives(gltf: GLTF2, material_index: int) -> None:
    if not gltf.meshes:
        return
    for mesh in gltf.meshes:
        if not mesh.primitives:
            continue
        for prim in mesh.primitives:
            prim.material = material_index

def add_khr_punctual_light(gltf: GLTF2,
                           light_type: str = "directional",
                           intensity: float = 5000.0,
                           color: Tuple[float,float,float] = (1.0,1.0,1.0),
                           translation: Tuple[float,float,float] = (2.0,3.0,5.0)) -> None:
    # 声明扩展
    if gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if "KHR_lights_punctual" not in gltf.extensionsUsed:
        gltf.extensionsUsed.append("KHR_lights_punctual")

    if gltf.extensions is None:
        gltf.extensions = {}
    if "KHR_lights_punctual" not in gltf.extensions:
        gltf.extensions["KHR_lights_punctual"] = {"lights": []}

    lights = gltf.extensions["KHR_lights_punctual"]["lights"]
    lights.append({
        "type": light_type,
        "color": list(color),
        "intensity": float(intensity)
    })
    light_index = len(lights) - 1

    # 创建光源节点
    gltf.nodes = ensure_list(gltf.nodes)
    light_node = {
        "name": "Auto_Sun_Light",
        "translation": list(translation),
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "extensions": {"KHR_lights_punctual": {"light": light_index}}
    }
    gltf.nodes.append(light_node)
    node_index = len(gltf.nodes) - 1

    # 挂到场景
    if gltf.scenes is not None and len(gltf.scenes) > 0:
        if gltf.scenes[gltf.scene].nodes is None:
            gltf.scenes[gltf.scene].nodes = []
        gltf.scenes[gltf.scene].nodes.append(node_index)
    else:
        gltf.scenes = ensure_list(gltf.scenes)
        gltf.scenes.append({"nodes": [node_index]})
        gltf.scene = len(gltf.scenes) - 1

def combine_glb_with_texture(
    input_glb: str,
    texture_png: str,
    output_glb: str,
    assume_linear: bool = False,
    add_light: bool = False,
    metallic: float = 0.3,
    roughness: float = 0.5,
    normal_png: Optional[str] = None
) -> None:
    gltf = GLTF2().load(input_glb)

    # 纹理图片（必要时 linear->sRGB）嵌入为 data URI
    data_uri = load_png_as_data_uri(texture_png, assume_linear=assume_linear)

    # 采样器（trilinear + repeat）
    sampler_idx = add_sampler_trilinear_repeat(gltf)

    # baseColor texture
    basecolor_tex_idx = add_texture_with_image(gltf, data_uri, sampler_idx)

    # 可选：normal 贴图（线性域，不做伽马）
    normal_tex_idx = add_normal_texture_if_any(gltf, normal_png, sampler_idx)

    # 材质（PBR参数）
    mat_idx = set_material_pbr(
        gltf,
        basecolor_tex_index=basecolor_tex_idx,
        metallic=metallic,
        roughness=roughness,
        normal_tex_index=normal_tex_idx
    )

    # 应用到所有 primitive
    apply_material_to_all_primitives(gltf, mat_idx)

    # 可选：插一盏平行光
    if add_light:
        print("➕ Adding KHR_lights_punctual directional light...")
        add_khr_punctual_light(gltf, light_type="directional", intensity=5000.0)

    gltf.save(output_glb)
    print(f"✅ Done. Wrote: {output_glb}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glb", required=True, help="白膜GLB路径")
    ap.add_argument("--texture", required=True, help="nvdiffrast烘焙PNG路径")
    ap.add_argument("--output-glb", required=True, help="输出GLB路径")
    ap.add_argument("--assume-linear", action="store_true",
                    help="假设输入PNG为线性空间；嵌入前做 linear→sRGB")
    ap.add_argument("--add-light", action="store_true",
                    help="向场景添加 KHR_lights_punctual 平行光")
    ap.add_argument("--metallic", type=float, default=0.3, help="metallicFactor (0~1)")
    ap.add_argument("--roughness", type=float, default=0.5, help="roughnessFactor (0~1)")
    ap.add_argument("--normal", type=str, default=None, help="可选：normal贴图（线性域PNG）")
    args = ap.parse_args()

    combine_glb_with_texture(
        input_glb=args.input_glb,
        texture_png=args.texture,
        output_glb=args.output_glb,
        assume_linear=args.assume_linear,
        add_light=args.add_light,
        metallic=args.metallic,
        roughness=args.roughness,
        normal_png=args.normal
    )

if __name__ == "__main__":
    main()
