import os
import time

def combine_videos_ffmpeg(top_video, bottom_video):
    # 修復輸出檔案名稱生成邏輯，避免與輸入檔案重名
    import os
    from pathlib import Path
    
    # 確保輸出檔案名稱不同於輸入檔案
    top_path = Path(top_video)
    output_name = top_path.stem.replace('_45_segment_processed', '_full_video').replace('_45_processed', '_full_video')
    output_video = str(top_path.parent / f"{output_name}.mp4")
    
    # 如果輸出檔案和輸入檔案相同，則添加後綴
    if output_video == top_video:
        output_name = top_path.stem + '_full_video'
        output_video = str(top_path.parent / f"{output_name}.mp4")
    
    cmd = (
        f'ffmpeg -y -hwaccel cuda -i "{top_video}" -i "{bottom_video}" '
        f'-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" '
        f'-c:v h264_nvenc -preset p7 -profile:v high444p -qp 0 -b:v 50000k '
        f'-rc constqp -pix_fmt yuv444p -threads 8 -bf 2 "{output_video}"'
    )
    
    print(f"🎬 合併影片: {Path(top_video).name} + {Path(bottom_video).name} → {Path(output_video).name}")
    result = os.system(cmd)  # 執行 FFmpeg 指令
    
    if result == 0 and Path(output_video).exists():
        return output_video
    else:
        print(f"❌ 影片合併失敗，返回值: {result}")
        return None

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    top_video = "testing__45.mp4"
    bottom_video = "testing__side.mp4"

    print("開始合併影片（超高畫質 + GPU 加速）...")
    combine_videos_ffmpeg(top_video, bottom_video)
    end_time = time.time()  # 記錄結束時間

    elapsed_time = end_time - start_time  # 計算執行時間

    print(f"處理時間: {elapsed_time:.2f} 秒")  # 顯示處理時間
