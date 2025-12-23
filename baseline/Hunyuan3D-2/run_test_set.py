import time
import traceback
from pathlib import Path

import trimesh
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover


def main():
    # 目录配置（相对于当前脚本所在目录）
    base_dir = Path(__file__).resolve().parent
    test_set_dir = base_dir / ".." / "test_set"
    ref_img_dir = base_dir / ".." / "reference_images"
    out_dir = Path("output_test_set")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 运行记录
    success_txt = out_dir / "success.txt"
    failed_txt = out_dir / "failed.txt"

    def _load_logged_names(txt_path: Path):
        names = set()
        if not txt_path.exists():
            return names
        try:
            with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 约定：每行首列为 glb 文件名
                    name = line.split("\t", 1)[0].strip()
                    if name:
                        names.add(name)
        except OSError:
            # 日志不可读时不阻塞主流程
            return set()
        return names

    logged_success = _load_logged_names(success_txt)
    logged_failed = _load_logged_names(failed_txt)

    # 加载 pipeline（只加载一次）
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        # 'tencent/Hunyuan3D-2',
        '/public/huggingface-models/tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0-turbo'
    )

    # 可复用的处理器
    rembg = BackgroundRemover()
    floater_remover = FloaterRemover()
    degenerate_remover = DegenerateFaceRemover()
    face_reducer = FaceReducer()

    glb_files = sorted(test_set_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"未找到 glb：{test_set_dir}")

    with success_txt.open("a", encoding="utf-8") as success_f, failed_txt.open(
        "a", encoding="utf-8"
    ) as failed_f:
        for i, glb_path in enumerate(glb_files, start=1):
            stem = glb_path.stem
            png_path = ref_img_dir / f"{stem}.png"

            # 输出：文件名保持原 glb 名字
            out_path = out_dir / glb_path.name

            # 若结果已存在则跳过，视为成功
            if out_path.exists():
                png_name = png_path.name if png_path.exists() else "-"
                if glb_path.name not in logged_success:
                    success_f.write(
                        f"{glb_path.name}\t{png_name}\t{out_path.as_posix()}\t0.00s\n"
                    )
                    success_f.flush()
                    logged_success.add(glb_path.name)
                print(f"[{i}/{len(glb_files)}] 跳过：已存在输出 {out_path}")
                continue

            if not png_path.exists():
                print(f"[{i}/{len(glb_files)}] 跳过：缺少参考图 {png_path}")
                continue

            t0 = time.time()
            print(f"[{i}/{len(glb_files)}] 处理：{glb_path.name}  <->  {png_path.name}")

            # 读取参考图（保持你的原逻辑：RGB 才做抠图）
            image = Image.open(png_path)
            if image.mode == "RGB":
                image = rembg(image)
            images = [image]

            # 读取并清理 mesh（保持你的原逻辑）
            mesh = trimesh.load(glb_path)
            mesh = floater_remover(mesh)
            mesh = degenerate_remover(mesh)
            mesh = face_reducer(mesh)

            # 贴图生成：增加容错，失败不中断批处理
            try:
                mesh = pipeline(mesh, image=images)
                mesh.export(out_path)

                elapsed = time.time() - t0
                if glb_path.name not in logged_success:
                    success_f.write(
                        f"{glb_path.name}\t{png_path.name}\t{out_path.as_posix()}\t{elapsed:.2f}s\n"
                    )
                    success_f.flush()
                    logged_success.add(glb_path.name)
                print(f"  输出：{out_path}  用时：{elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - t0
                if glb_path.name not in logged_failed:
                    failed_f.write(
                        f"{glb_path.name}\t{png_path.name}\t{elapsed:.2f}s\t{type(e).__name__}: {e}\n"
                    )
                    failed_f.write(traceback.format_exc())
                    failed_f.write("\n" + "-" * 80 + "\n")
                    failed_f.flush()
                    logged_failed.add(glb_path.name)
                print(f"  失败：{type(e).__name__}: {e}  用时：{elapsed:.2f}s")
                continue


if __name__ == "__main__":
    main()