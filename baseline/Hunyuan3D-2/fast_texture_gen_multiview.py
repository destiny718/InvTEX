import time

import torch
from PIL import Image
import trimesh

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

images_path = [
    "/data/tianqi/reference_images/0de9252b9afea4833be9662d467c1b71e98600d80edf6745c9e68a3bdb5e8676.png",
]

images = []
for image_path in images_path:
    image = Image.open(image_path)
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images.append(image)

pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    # 'tencent/Hunyuan3D-2',
    '/public/huggingface-models/tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-paint-v2-0-turbo'
)

mesh = trimesh.load('/data/tianqi/test_set/0de9252b9afea4833be9662d467c1b71e98600d80edf6745c9e68a3bdb5e8676.glb')

mesh = pipeline(mesh, image=images)
mesh.export('0de9252b9afea4833be9662d467c1b71e98600d80edf6745c9e68a3bdb5e8676.glb')