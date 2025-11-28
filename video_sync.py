import cv2
import json
import numpy as np
import time
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_hit_frame(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for frame_info in data:
        if frame_info.get("tennis_ball_hit", False):
            return frame_info["frame"]
    return None

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 保持float型別
    
    # 防止除零錯誤
    if fps <= 0:
        print(f"⚠️ 警告: {video_path} 的 FPS 值異常 ({fps})，使用預設值 30")
        fps = 30.0
    
    duration = total_frames / fps
    cap.release()
    
    print(f"📹 影片資訊: {video_path}")
    print(f"   總幀數: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   時長: {duration:.2f} 秒")
    
    return total_frames, fps, duration

def process_video(input_path, output_path, start_frame, frames_to_process, dimensions):
    cap = cv2.VideoCapture(input_path)
    width, height = dimensions
    
    # 預設使用 mp4v 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 使用 H.264 編碼器提升效能
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    # 檢查 VideoWriter 是否成功初始化
    if not out.isOpened():
        print(f"❌ VideoWriter 初始化失敗: {output_path}")
        print(f"🔧 嘗試使用不同的編碼器...")
        
        # 嘗試其他編碼器
        codecs_to_try = [
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('H264', cv2.VideoWriter_fourcc(*'H264')),
            ('X264', cv2.VideoWriter_fourcc(*'X264')),
            ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
        ]
        
        for codec_name, codec in codecs_to_try:
            print(f"🔧 嘗試 {codec_name} 編碼器...")
            out.release()  # 釋放失敗的 VideoWriter
            out = cv2.VideoWriter(output_path, codec, cap.get(cv2.CAP_PROP_FPS), (width, height))
            if out.isOpened():
                print(f"✅ {codec_name} 編碼器成功")
                break
        
        # 如果所有編碼器都失敗，嘗試 AVI 格式
        if not out.isOpened():
            print("🔧 嘗試 AVI 格式...")
            output_path_avi = output_path.replace('.mp4', '.avi')
            out.release()
            out = cv2.VideoWriter(output_path_avi, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (width, height))
            
            if not out.isOpened():
                print("❌ 所有影片編碼器都失敗")
                cap.release()
                out.release()
                return False
            else:
                output_path = output_path_avi  # 更新輸出路徑
    
    # 設置讀取緩衝區大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    
    # 直接跳到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 批次讀取和寫入
    batch_size = 32
    frames = []
    
    for i in range(0, frames_to_process, batch_size):
        batch_frames = min(batch_size, frames_to_process - i)
        for _ in range(batch_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # 批次寫入
        for frame in frames:
            out.write(frame)
        frames = []
    
    cap.release()
    out.release()
    
    # 驗證生成的檔案大小
    try:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            if file_size < 10:  # 小於 10KB 可能有問題
                print(f"⚠️ 警告：生成的檔案可能太小 ({file_size:.2f} KB): {output_path}")
                return False
            else:
                print(f"✅ 影片處理成功，檔案大小: {file_size:.2f} KB")
                return True
        else:
            print(f"❌ 檔案未生成: {output_path}")
            return False
    except Exception as e:
        print(f"❌ 檔案驗證失敗: {e}")
        return False

def synchronize_videos(input_path_1, input_path_2, json_path_1, json_path_2):
    trim_length=60

    # 獲取影片資訊
    frames1, fps1, duration1 = get_video_info(input_path_1)
    frames2, fps2, duration2 = get_video_info(input_path_2)
    
    print("\n原始影片資訊:")
    print(f"影片 1: {frames1} 幀, {duration1:.2f} 秒")
    print(f"影片 2: {frames2} 幀, {duration2:.2f} 秒")

    # 獲取擊球幀
    hit_frame_1 = get_hit_frame(json_path_1)
    hit_frame_2 = get_hit_frame(json_path_2)
    
    print(f"\n擊球幀位置:")
    print(f"影片 1: 第 {hit_frame_1} 幀")
    print(f"影片 2: 第 {hit_frame_2} 幀")

    # 計算剪輯範圍
    max_frames_after = min(frames1 - hit_frame_1, frames2 - hit_frame_2)
    max_frames_before = min(hit_frame_1, hit_frame_2)
    
    frames_before = min(trim_length // 2, max_frames_before)
    frames_after = min(trim_length - frames_before, max_frames_after)
    
    start_frame_1 = hit_frame_1 - frames_before
    start_frame_2 = hit_frame_2 - frames_before
    frames_to_process = frames_before + frames_after

    print(f"\n剪輯資訊:")
    print(f"影片 1: 從第 {start_frame_1} 幀到第 {start_frame_1 + frames_to_process} 幀")
    print(f"影片 2: 從第 {start_frame_2} 幀到第 {start_frame_2 + frames_to_process} 幀")

    # 獲取影片尺寸
    cap1 = cv2.VideoCapture(input_path_1)
    cap2 = cv2.VideoCapture(input_path_2)
    dimensions1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimensions2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap1.release()
    cap2.release()

    output_path_1 = input_path_1.replace('_processed.mp4', '_synced.mp4')
    output_path_2 = input_path_2.replace('_processed.mp4', '_synced.mp4')
    
    # 如果沒有找到 _processed，則添加 _synced 後綴
    if output_path_1 == input_path_1:
        output_path_1 = input_path_1.replace('.mp4', '_synced.mp4')
    if output_path_2 == input_path_2:
        output_path_2 = input_path_2.replace('.mp4', '_synced.mp4')

    # 使用線程池並行處理兩個影片
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_video, input_path_1, output_path_1, 
                          start_frame_1, frames_to_process, dimensions1),
            executor.submit(process_video, input_path_2, output_path_2, 
                          start_frame_2, frames_to_process, dimensions2)
        ]
        
        # 等待所有處理完成並檢查結果
        sync_results = []
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                sync_results.append(result)
                if result:
                    print(f"✅ 影片 {i+1} 同步成功")
                else:
                    print(f"❌ 影片 {i+1} 同步失敗")
            except Exception as e:
                print(f"❌ 處理過程中發生錯誤 (線程 {i+1}): {str(e)}")
                sync_results.append(False)
        
        # 如果同步成功，移動檔案回原始路徑
        if all(sync_results):
            try:
                if os.path.exists(output_path_1):
                    shutil.move(output_path_1, input_path_1)
                    print(f"📹 同步影片1已更新: {Path(input_path_1).name}")
                if os.path.exists(output_path_2):
                    shutil.move(output_path_2, input_path_2)
                    print(f"📹 同步影片2已更新: {Path(input_path_2).name}")
            except Exception as e:
                print(f"⚠️ 移動同步檔案失敗: {e}")
        else:
            print("⚠️ 同步失敗，保留原始檔案")

    final_duration = frames_to_process / fps1
    print(f"\n最終影片資訊:")
    print(f"兩個影片都是 {frames_to_process} 幀, {final_duration:.2f} 秒")
    print("\n同步完成!")

if __name__ == "__main__":
    start_time = time.time()
    
    input_video_1 = "pro_1_1_45_temp.mp4"
    input_video_2 = "pro_1_1_side_temp.mp4"
    json_path_1 = "pro_1_1_45_temp(2D_trajectory_smoothed).json"
    json_path_2 = "pro_1_1_side_temp(2D_trajectory_smoothed).json"

    print("開始執行影片同步...")
    synchronize_videos(input_video_1, input_video_2, json_path_1, json_path_2)
    
    print(f"執行時間: {time.time() - start_time:.4f}秒")