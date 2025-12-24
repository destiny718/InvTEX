import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


NEGATIVE_PROMPT_DEFAULT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass(frozen=True)
class Item:
    row_index: int
    video_dir: str
    prompt: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "读取 test_set.csv，逐行使用 video 目录下的 depth.mp4 + mask.mp4 作为 vace 条件，"
            "并用两张 GPU（cuda:0/cuda:1）并行生成视频输出到指定目录。"
        )
    )
    p.add_argument(
        "--csv_path",
        help="输入CSV路径（需要包含 video 与 prompt 列）",
    )
    p.add_argument(
        "--out_dir",
        help="输出目录（会自动创建）",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV编码（默认utf-8；如乱码可试 utf-8-sig 或 gb18030）",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="生成分辨率高度（同时用于读取 VideoData 的 height）",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="生成分辨率宽度（同时用于读取 VideoData 的 width）",
    )
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--quality", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tiled", action="store_true", default=True)
    p.add_argument(
        "--negative_prompt",
        default=NEGATIVE_PROMPT_DEFAULT,
        help="负向提示词（默认沿用原脚本）",
    )
    p.add_argument(
        "--depth_name",
        default="depth.mp4",
        help="video 目录下深度视频文件名（默认 depth.mp4）",
    )
    p.add_argument(
        "--mask_name",
        default="mask.mp4",
        help="video 目录下 mask 视频文件名（默认 mask.mp4）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前N行（0表示处理全部）",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="从第几行开始处理（0基，默认0）",
    )
    return p.parse_args()


def read_items(csv_path: str, encoding: str, start: int = 0, limit: int = 0) -> list[Item]:
    items: list[Item] = []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV没有表头，无法定位 prompt/video 列。")
        if "video" not in reader.fieldnames:
            raise SystemExit(f"未找到 video 列。现有列：{reader.fieldnames}")
        if "prompt" not in reader.fieldnames:
            raise SystemExit(f"未找到 prompt 列。现有列：{reader.fieldnames}")

        for row_index, row in enumerate(reader):
            if row_index < start:
                continue
            if limit and len(items) >= limit:
                break

            video_dir = row.get("video")
            prompt = row.get("prompt")
            video_dir_str = "" if video_dir is None else str(video_dir).strip()
            prompt_str = "" if prompt is None else str(prompt).strip()
            if not video_dir_str or not prompt_str:
                continue
            items.append(Item(row_index=row_index, video_dir=video_dir_str, prompt=prompt_str))
    return items


def chunks_by_gpu(items: list[Item], gpu_ids: Iterable[int]) -> dict[int, list[Item]]:
    gpu_ids = list(gpu_ids)
    groups: dict[int, list[Item]] = {gid: [] for gid in gpu_ids}
    for i, item in enumerate(items):
        groups[gpu_ids[i % len(gpu_ids)]].append(item)
    return groups


def _build_pipeline(device: str):
    import torch
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                path="/data/models/PAI/Wan2.2-VACE-Fun-A14B/high_noise_model/diffusion_pytorch_model.safetensors",
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                path="/data/models/PAI/Wan2.2-VACE-Fun-A14B/low_noise_model/diffusion_pytorch_model.safetensors",
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                path="/data/models/PAI/Wan2.2-VACE-Fun-A14B/models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                model_id="PAI/Wan2.2-VACE-Fun-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
                path="/data/models/PAI/Wan2.2-VACE-Fun-A14B/Wan2.1_VAE.pth",
            ),
        ],
    )
    pipe.enable_vram_management()
    return pipe


def worker(gpu_id: int, items: list[Item], args: argparse.Namespace) -> None:
    # 重要：在子进程内 import torch / 初始化 cuda
    import torch
    from PIL import Image
    from diffsynth import VideoData, save_video

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    pipe = _build_pipeline(device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        video_dir = Path(item.video_dir.lstrip("/"))
        depth_path = video_dir / args.depth_name
        mask_path = video_dir / args.mask_name

        if not depth_path.exists() or not mask_path.exists():
            print(
                f"[gpu {gpu_id}] skip row={item.row_index}: 缺少文件 depth={depth_path.exists()} mask={mask_path.exists()} dir={video_dir}"
            )
            continue

        vace_video = VideoData(str(depth_path), height=args.height, width=args.width)
        vace_video_mask = VideoData(str(mask_path), height=args.height, width=args.width)

        try:
            video = pipe(
                prompt=item.prompt,
                negative_prompt=args.negative_prompt,
                vace_video=vace_video,
                vace_video_mask=vace_video_mask,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed,
                tiled=args.tiled,
            )
        except Exception as e:
            print(f"[gpu {gpu_id}] error row={item.row_index} dir={video_dir}: {e}")
            continue

        # 输出文件名：行号 + 文件夹名，避免冲突
        safe_dir_name = video_dir.name or "video"
        out_path = out_dir / f"row{item.row_index:06d}_{safe_dir_name}_gpu{gpu_id}.mp4"
        save_video(video, str(out_path), fps=args.fps, quality=args.quality)
        print(f"[gpu {gpu_id}] saved: {out_path}")


def main() -> None:
    args = parse_args()

    items = read_items(args.csv_path, encoding=args.encoding, start=args.start, limit=args.limit)
    if not items:
        raise SystemExit("CSV中没有可处理的行（需要 video 与 prompt 非空）。")

    gpu_ids = [2, 3]

    groups = chunks_by_gpu(items, gpu_ids=gpu_ids)
    for gid in gpu_ids:
        print(f"将 {len(groups[gid])} 条任务分配给 cuda:{gid}")

    # 多进程并行：每张卡一个进程，各自加载 pipeline
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    procs: list[mp.Process] = []
    for gid in gpu_ids:
        p = mp.Process(target=worker, args=(gid, groups[gid], args), daemon=False)
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
