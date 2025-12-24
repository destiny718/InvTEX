import cv2
import os

def video_to_frames(video_path, output_dir="frames", prefix="frame"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读完

        # 输出文件名，带编号补零
        filename = os.path.join(output_dir, f"{prefix}_{frame_idx:04d}.png")
        cv2.imwrite(filename, frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"已保存 {frame_idx} 帧...")

    cap.release()
    print(f"✅ 完成！总共保存 {frame_idx} 帧到 {output_dir}/")

# 示例用法
if __name__ == "__main__":

    video_to_frames("row000013_1b28eef9b0d0e7783d0017b1b14c99e3afaa8ee986b45e4fdced506c0b4465d9_gpu3.mp4", "outputs_1/result", "frame")
