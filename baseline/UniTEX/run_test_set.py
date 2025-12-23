from pathlib import Path
import shutil
import traceback

from pipeline import CustomRGBTextureFullPipeline


def main():
    base_dir = Path(__file__).resolve().parent

    # 目录（按需改成你的实际位置）
    test_set_dir = Path("/home/cjh/test_set")
    ref_img_dir = Path("/home/cjh/reference_images")
    out_dir = base_dir / "output_test_set"
    out_dir.mkdir(parents=True, exist_ok=True)

    success_list_path = out_dir / "success_cases.txt"
    failed_list_path = out_dir / "failed_cases.txt"
    success_cases: list[str] = []
    failed_cases: list[str] = []

    rgb_tfp = CustomRGBTextureFullPipeline(
        pretrain_models="/mnt/disk-2/cjh/unitex_pretrained_models",
        super_resolutions=False,
        seed=63,
    )

    glb_files = sorted(test_set_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"未找到 glb：{test_set_dir}")

    for idx, glb_path in enumerate(glb_files, start=1):
        stem = glb_path.stem
        png_path = ref_img_dir / f"{stem}.png"

        # 若目标目录已存在要生成的 glb（按归档规则：与输入 glb 同名），则跳过
        final_glb_path = out_dir / glb_path.name
        if final_glb_path.exists():
            print(f"[{idx}/{len(glb_files)}] 跳过：已存在输出 {final_glb_path}")
            success_cases.append(glb_path.name)
            continue

        if not png_path.exists():
            print(f"[{idx}/{len(glb_files)}] 跳过：缺少参考图 {png_path}")
            failed_cases.append(glb_path.name)
            continue

        # 每个样本单独一个输出子目录（pipeline 通常会写多文件）
        save_root = out_dir / stem
        save_root.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(glb_files)}] 处理：{glb_path.name} <-> {png_path.name}")
        try:
            rgb_tfp(str(save_root), str(png_path), str(glb_path), clear_cache=False)

            # 可选：把生成的 glb 归档为“原本文件名”，放到 output_test_set 根目录
            produced_glbs = sorted(save_root.rglob("*.glb"), key=lambda p: p.stat().st_mtime)
            if produced_glbs:
                latest_glb = produced_glbs[-1]
                shutil.copy2(latest_glb, final_glb_path)
                print(f"  归档输出：{final_glb_path}  (来自 {latest_glb})")
            else:
                print(f"  提示：{save_root} 下未检测到 glb 输出，已保留原始输出目录")

            success_cases.append(glb_path.name)
        except Exception as e:
            failed_cases.append(glb_path.name)
            print(f"  失败：{glb_path.name}，已跳过。错误：{e}")
            # 仅打印 traceback，便于定位问题；不写入 txt（按需求只写文件名）
            traceback.print_exc()
            continue

    success_list_path.write_text("\n".join(success_cases) + ("\n" if success_cases else ""), encoding="utf-8")
    failed_list_path.write_text("\n".join(failed_cases) + ("\n" if failed_cases else ""), encoding="utf-8")
    print(f"完成：成功 {len(success_cases)} 个，失败 {len(failed_cases)} 个")
    print(f"成功列表：{success_list_path}")
    print(f"失败列表：{failed_list_path}")


if __name__ == "__main__":
    main()