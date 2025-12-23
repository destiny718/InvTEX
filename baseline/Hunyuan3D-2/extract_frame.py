import cv2

def get_total_frames(video_path):
    """返回视频的总帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def save_frame(video_path, frame_number, output_path="frame.jpg"):
    """保存指定帧为图片"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        raise ValueError(f"帧号超出范围 (0 ~ {total_frames-1})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 跳转到指定帧
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取该帧")
    
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"第 {frame_number} 帧已保存为 {output_path}")

if __name__ == "__main__":
    video_file = "example.mp4"  # 你的 mp4 文件路径
    
    # 1. 获取总帧数
    total = get_total_frames(video_file)
    print("视频总帧数:", total)
    
    # 2. 保存某一帧
    frame_to_extract = 19  # 例如提取第 100 帧
    save_frame(video_file, frame_to_extract, "frame100.jpg")
