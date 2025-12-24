import os
from pygltflib import GLTF2
import base64

def extract_textures(glb_path, out_dir=None):
    # Load GLB
    gltf = GLTF2().load(glb_path)

    # Set output directory
    if out_dir is None:
        out_dir = os.path.splitext(glb_path)[0] + "_textures"
    os.makedirs(out_dir, exist_ok=True)

    # Binary buffer in .glb
    bin_data = gltf.binary_blob()

    print(f"[INFO] Extracting textures from {glb_path}")

    for i, image in enumerate(gltf.images):
        uri = image.uri

        if uri:  # case: external image (rare in GLB)
            if uri.startswith("data:"):  # base64-encoded
                header, data = uri.split(",", 1)
                bytes_data = base64.b64decode(data)
                ext = header.split(";")[0].split("/")[-1]  # png/jpg
            else:  # external file path
                with open(os.path.join(os.path.dirname(glb_path), uri), "rb") as f:
                    bytes_data = f.read()
                ext = uri.split(".")[-1]

        else:
            # case: image stored in bufferView (common in GLB)
            bv = gltf.bufferViews[image.bufferView]
            start = bv.byteOffset or 0
            end = start + (bv.byteLength or 0)
            bytes_data = bin_data[start:end]

            # guess file extension from mimeType
            if image.mimeType:
                ext = image.mimeType.split("/")[-1]
            else:
                ext = "png"

        out_path = os.path.join(out_dir, f"texture_{i}.{ext}")
        with open(out_path, "wb") as f:
            f.write(bytes_data)

        print(f"[OK] Saved: {out_path}")

    print("[DONE] All textures extracted.")


# Example
extract_textures("1b28eef9b0d0e7783d0017b1b14c99e3afaa8ee986b45e4fdced506c0b4465d9.glb")