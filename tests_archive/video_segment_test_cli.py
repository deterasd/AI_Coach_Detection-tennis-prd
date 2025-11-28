"""
影片自動分割命令行測試工具
簡化版本，用於快速測試功能

使用方法:
python video_segment_test_cli.py input_video.mp4

或者設定參數:
python video_segment_test_cli.py input_video.mp4 --confidence 0.6 --duration 5 --min-interval 3
"""

import cv2
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非互動式後端，避免GUI問題
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: 無法導入 matplotlib，將無法生成圖表")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 無法導入 YOLO，將使用模擬模式")

def create_analysis_visualization(detection_results, ball_entry_times, video_path, confidence_threshold, segment_duration=-2):
    """
    創建分析結果的可視化圖表並保存為圖片
    
    參數:
    - detection_results: 偵測結果列表
    - ball_entry_times: 球進入時間點
    - video_path: 影片路徑 
    - confidence_threshold: 信心度閾值
    - segment_duration: 片段時長（負數表示往前偏移）
    """
    if not MATPLOTLIB_AVAILABLE or not detection_results:
        print("⚠️ matplotlib 不可用或沒有偵測結果，跳過可視化")
        return None
        
    try:
        # 創建圖表 - 3個子圖
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 圖1: 偵測信心度時間序列
        times = [r['time'] for r in detection_results]
        confidences = [r['confidence'] for r in detection_results]
        
        ax1.plot(times, confidences, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=confidence_threshold, color='r', linestyle='--', 
                   label=f'信心度閾值 ({confidence_threshold})')
        
        # 標記球進入時間點
        for entry_time in ball_entry_times:
            ax1.axvline(x=entry_time, color='g', linestyle='-', alpha=0.8, linewidth=2)
            ax1.text(entry_time, ax1.get_ylim()[1]*0.9, f'{entry_time:.1f}s', 
                    rotation=90, ha='right', va='top')
        
        ax1.set_xlabel('時間 (秒)')
        ax1.set_ylabel('偵測信心度')
        ax1.set_title('網球偵測信心度 vs 時間')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 圖2: 偵測狀態與邊緣位置
        detected_states = [1 if r['detected'] else 0 for r in detection_results]
        edge_states = [0.5 if r.get('in_edge', False) else 0 for r in detection_results]
        
        ax2.fill_between(times, detected_states, alpha=0.6, color='orange', label='偵測到球')
        ax2.fill_between(times, edge_states, alpha=0.4, color='purple', label='球在邊緣')
        
        # 標記預計分割區間
        for i, entry_time in enumerate(ball_entry_times):
            start_time = max(0, entry_time + segment_duration)
            end_time = start_time + abs(segment_duration) + 2  # 假設片段長度
            ax2.axvspan(start_time, end_time, alpha=0.3, color='red', 
                       label='分割區間' if i == 0 else '')
            ax2.text(start_time + (end_time - start_time)/2, 0.75, 
                    f'片段{i+1}', ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax2.set_xlabel('時間 (秒)')
        ax2.set_ylabel('偵測狀態')
        ax2.set_title('網球偵測狀態與分割區間 (紫色=邊緣位置)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 圖3: 球的位置軌跡 (如果有位置資訊)
        x_positions = []
        y_positions = []
        valid_times = []
        
        for r in detection_results:
            if r['detected'] and r.get('position'):
                x, y = r['position']
                x_positions.append(x)
                y_positions.append(y)
                valid_times.append(r['time'])
        
        if x_positions:
            # 創建顏色映射表示時間
            scatter = ax3.scatter(x_positions, y_positions, c=valid_times, 
                                cmap='viridis', alpha=0.6, s=20)
            
            # 標記球進入點
            for entry_time in ball_entry_times:
                # 找到最接近進入時間的位置
                closest_idx = min(range(len(valid_times)), 
                                 key=lambda i: abs(valid_times[i] - entry_time))
                if abs(valid_times[closest_idx] - entry_time) < 0.5:  # 0.5秒內
                    ax3.scatter(x_positions[closest_idx], y_positions[closest_idx], 
                              color='red', s=100, marker='*', 
                              label='球進入點' if entry_time == ball_entry_times[0] else '')
            
            ax3.set_xlabel('X 位置 (像素)')
            ax3.set_ylabel('Y 位置 (像素)')
            ax3.set_title('球的位置軌跡 (顏色表示時間，紅星表示進入點)')
            ax3.invert_yaxis()  # 反轉Y軸，因為影像座標系Y軸向下
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 添加色條
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('時間 (秒)')
        else:
            ax3.text(0.5, 0.5, '無位置資訊可顯示', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=16)
            ax3.set_title('球的位置軌跡')
        
        plt.tight_layout()
        
        # 保存圖表
        video_name = Path(video_path).stem
        output_path = Path(video_path).parent / f"{video_name}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # 關閉圖表釋放記憶體
        
        print(f"📊 分析圖表已保存: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ 建立可視化失敗: {str(e)}")
        return None

def generate_simulation_data(video_path, confidence_threshold=0.5, min_interval=2.0):
    """
    生成模擬的球檢測數據，用於測試功能
    """
    import random
    import math
    
    # 模擬影片參數
    duration = 30.0  # 30秒影片
    fps = 30
    total_frames = int(duration * fps)
    
    detection_results = []
    ball_entry_times = []
    
    # 模擬3個球進入的時間點
    entry_scenarios = [5.2, 12.8, 21.5]  # 秒
    
    print(f"🎭 生成模擬數據: {duration}秒影片, {total_frames}幀")
    
    for frame in range(total_frames):
        current_time = frame / fps
        
        # 模擬檢測邏輯
        detected = False
        confidence = 0.0
        position = None
        in_edge = False
        
        # 檢查是否接近球進入時間點
        for entry_time in entry_scenarios:
            time_diff = abs(current_time - entry_time)
            
            if time_diff < 2.0:  # 球進入前後2秒內有檢測
                # 距離球進入時間越近，檢測機率越高
                detection_prob = 1.0 - (time_diff / 2.0)
                
                if random.random() < detection_prob:
                    detected = True
                    confidence = random.uniform(0.3, 0.9)
                    
                    # 模擬位置 - 在邊緣進入時間點附近，球在邊緣
                    if time_diff < 0.5:  # 進入瞬間在邊緣
                        position = (random.uniform(50, 150), random.uniform(100, 300))  # 左邊緣
                        in_edge = True
                    else:  # 其他時間在中央
                        position = (random.uniform(300, 500), random.uniform(200, 400))
                        in_edge = False
        
        # 添加一些隨機噪音檢測
        if not detected and random.random() < 0.05:  # 5%機率誤檢
            detected = True
            confidence = random.uniform(0.2, 0.4)
            position = (random.uniform(100, 600), random.uniform(100, 400))
            in_edge = random.choice([True, False])
        
        detection_results.append({
            'frame': frame,
            'time': current_time,
            'detected': detected,
            'confidence': confidence,
            'position': position,
            'in_edge': in_edge
        })
    
    # 基於檢測結果找出球進入時間點
    previous_detected = False
    
    for result in detection_results:
        current_detected = result['detected'] and result['confidence'] >= confidence_threshold
        
        if current_detected and not previous_detected and result['in_edge']:
            current_time = result['time']
            # 檢查最小間隔
            if not ball_entry_times or (current_time - ball_entry_times[-1]) >= min_interval:
                ball_entry_times.append(current_time)
        
        previous_detected = current_detected
    
    print(f"🎯 模擬找到 {len(ball_entry_times)} 個球進入時間點:")
    for i, time_point in enumerate(ball_entry_times):
        print(f"   {i+1}. {time_point:.2f}秒 (模擬)")
    
    return ball_entry_times, detection_results

def detect_ball_entry_points(video_path, model=None, confidence_threshold=0.5, min_interval=2.0):
    """
    偵測網球進入畫面的時間點
    
    參數:
    - video_path: 影片路徑
    - model: YOLO模型 (如果為None則使用模擬模式)
    - confidence_threshold: 信心度閾值
    - min_interval: 最小間隔時間(秒)
    
    返回:
    - ball_entry_times: 球進入時間點列表
    - detection_results: 詳細偵測結果
    """
    print(f"🎬 開始分析影片: {Path(video_path).name}")
    
    # 如果沒有模型，使用模擬模式
    if model is None:
        print("🎭 使用模擬模式生成測試數據")
        return generate_simulation_data(video_path, confidence_threshold, min_interval)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"無法開啟影片: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 影片資訊: {total_frames} 影格, {fps:.2f} FPS, {duration:.2f} 秒")
    
    ball_entry_times = []
    detection_results = []
    
    previous_ball_detected = False
    previous_ball_position = None
    last_entry_time = -min_interval
    frame_count = 0
    
    # 獲取畫面尺寸資訊
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 定義畫面邊緣區域
    edge_threshold = 0.15  # 邊緣區域佔畫面的比例
    left_edge = frame_width * edge_threshold
    right_edge = frame_width * (1 - edge_threshold)
    top_edge = frame_height * edge_threshold
    bottom_edge = frame_height * (1 - edge_threshold)
    
    print(f"� 畫面尺寸: {frame_width}x{frame_height}")
    print(f"🎯 邊緣偵測區域: 左({left_edge:.0f}), 右({right_edge:.0f}), 上({top_edge:.0f}), 下({bottom_edge:.0f})")
    
    print("�🔍 正在分析影格...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_count / fps
        
        # 偵測網球和位置
        current_ball_detected = False
        max_confidence = 0
        ball_position = None
        ball_in_edge = False
        
        if model and YOLO_AVAILABLE:
            # 使用真實的YOLO模型
            results = model(frame, verbose=False)
            
            if len(results[0].boxes) > 0:
                best_box = None
                best_confidence = 0
                
                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = box
                
                if best_confidence > confidence_threshold:
                    current_ball_detected = True
                    max_confidence = best_confidence
                    
                    # 取得球的位置
                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                    ball_center_x = (x1 + x2) / 2
                    ball_center_y = (y1 + y2) / 2
                    ball_position = (ball_center_x, ball_center_y)
                    
                    # 檢查是否在邊緣
                    ball_in_edge = (ball_center_x < left_edge or ball_center_x > right_edge or
                                   ball_center_y < top_edge or ball_center_y > bottom_edge)
        else:
            # 模擬模式：基於影像變化偵測移動物體
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 簡單的運動偵測模擬
            if frame_count > 0:
                diff = cv2.absdiff(gray, prev_gray)
                motion_pixels = np.sum(diff > 30)
                motion_ratio = motion_pixels / (frame.shape[0] * frame.shape[1])
                
                # 模擬信心度和偵測結果
                max_confidence = min(motion_ratio * 10, 1.0)
                current_ball_detected = motion_ratio > 0.01  # 模擬閾值
                
                if current_ball_detected:
                    # 模擬球的位置（在有運動的區域中心）
                    y_indices, x_indices = np.where(diff > 30)
                    if len(x_indices) > 0:
                        ball_center_x = np.mean(x_indices)
                        ball_center_y = np.mean(y_indices)
                        ball_position = (ball_center_x, ball_center_y)
                        ball_in_edge = (ball_center_x < left_edge or ball_center_x > right_edge or
                                       ball_center_y < top_edge or ball_center_y > bottom_edge)
            else:
                max_confidence = 0
                current_ball_detected = False
                
            if 'prev_gray' not in locals():
                prev_gray = gray.copy()
            else:
                prev_gray = gray.copy()
        
        # 記錄偵測結果
        detection_results.append({
            'frame': frame_count,
            'time': current_time,
            'detected': current_ball_detected,
            'confidence': max_confidence,
            'position': ball_position,
            'in_edge': ball_in_edge
        })
        
        # 判斷球進入畫面的邏輯
        is_ball_entry = False
        entry_reason = ""
        
        if current_ball_detected and ball_position:
            ball_center_x, ball_center_y = ball_position
            
            if current_ball_detected and not previous_ball_detected:
                # 情況1: 球從無到有出現
                if ball_in_edge:
                    # 球出現在邊緣 = 球從畫面外進入
                    is_ball_entry = True
                    entry_reason = f"邊緣進入 (位置: {ball_center_x:.0f}, {ball_center_y:.0f})"
                else:
                    # 球出現在中央 = 可能是擊球瞬間，不算進入
                    entry_reason = f"中央出現 (位置: {ball_center_x:.0f}, {ball_center_y:.0f}) - 忽略"
            
            elif current_ball_detected and previous_ball_detected and previous_ball_position:
                # 情況2: 球持續存在，檢查是否從邊緣移向中央
                prev_x, prev_y = previous_ball_position
                curr_x, curr_y = ball_position
                
                # 檢查球是否從邊緣移到中央 (移動方向分析)
                prev_in_edge = (prev_x < left_edge or prev_x > right_edge or
                               prev_y < top_edge or prev_y > bottom_edge)
                
                if prev_in_edge and not ball_in_edge:
                    # 球從邊緣移到中央區域
                    move_distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                    if move_distance > 20:  # 移動距離閾值
                        is_ball_entry = True
                        entry_reason = f"邊緣移入 (從 {prev_x:.0f},{prev_y:.0f} 到 {curr_x:.0f},{curr_y:.0f})"
        
        # 檢查時間間隔並記錄進入點
        if is_ball_entry and current_time - last_entry_time >= min_interval:
            ball_entry_times.append(current_time)
            last_entry_time = current_time
            print(f"🎾 偵測到球進入: {current_time:.2f}s - {entry_reason} (信心度: {max_confidence:.3f})")
        
        # 更新前一幀的狀態
        previous_ball_detected = current_ball_detected
        previous_ball_position = ball_position
        frame_count += 1
        
        # 顯示進度
        if frame_count % (total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   進度: {progress:.1f}%")
    
    cap.release()
    
    print(f"✅ 分析完成！偵測到 {len(ball_entry_times)} 次球進入畫面")
    if ball_entry_times:
        print(f"🕐 球進入時間點: {[f'{t:.2f}s' for t in ball_entry_times]}")
    
    return ball_entry_times, detection_results

def segment_dual_videos_dynamic(side_video_path, deg45_video_path, output_folder, entry_times, start_offset=-0.5, end_padding=1.0):
    """
    根據進入時間點動態分割兩個角度的影片 (每個片段從一個進入點到下一個進入點)
    
    參數:
    - side_video_path: 側面影片路徑
    - deg45_video_path: 45度影片路徑
    - output_folder: 輸出資料夾
    - entry_times: 球進入時間點列表
    - start_offset: 開始偏移時間 (負數表示提前開始)
    - end_padding: 最後一個片段的額外長度
    """
    if not entry_times:
        print("⚠️ 沒有偵測到球進入時間點，無法進行分割")
        return []

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(side_video_path).stem.replace('_side', '').replace('__side', '')
    segment_info = []
    
    # 獲取影片總長度
    cap = cv2.VideoCapture(str(side_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()
    
    print(f"✂️ 開始動態分割兩個角度的影片到: {output_folder}")
    print(f"📹 側面影片: {Path(side_video_path).name} (總長: {video_duration:.2f}秒)")
    print(f"📹 45度影片: {Path(deg45_video_path).name}")
    print(f"🎯 動態分割邏輯: 每個片段從一個球進入點到下一個球進入點")
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        segment_num = i + 1
        
        # 計算片段結束時間
        if i < len(entry_times) - 1:
            # 不是最後一個片段，結束時間是下一個進入點
            end_time = entry_times[i + 1] + start_offset
        else:
            # 最後一個片段，使用固定長度或影片結尾
            end_time = min(video_duration, entry_time + 4.0 + end_padding)
        
        # 確保片段長度合理
        segment_duration = max(1.0, end_time - start_time)  # 最少1秒
        
        # 側面影片分割
        side_output_name = f"{input_name}_segment_{segment_num:02d}_side.mp4"
        side_output_path = output_folder / side_output_name
        
        # 45度影片分割
        deg45_output_name = f"{input_name}_segment_{segment_num:02d}_45.mp4"
        deg45_output_path = output_folder / deg45_output_name
        
        print(f"📽️ 片段{segment_num}: {start_time:.2f}s - {end_time:.2f}s (時長: {segment_duration:.2f}s)")
        print(f"   球進入時間: {entry_time:.2f}s")
        
        # 分割側面影片
        side_cmd = f'ffmpeg -i "{side_video_path}" -ss {start_time} -t {segment_duration} -c copy "{side_output_path}" -y -loglevel quiet'
        side_result = os.system(side_cmd)
        
        # 分割45度影片
        deg45_cmd = f'ffmpeg -i "{deg45_video_path}" -ss {start_time} -t {segment_duration} -c copy "{deg45_output_path}" -y -loglevel quiet'
        deg45_result = os.system(deg45_cmd)
        
        # 檢查結果
        side_success = side_result == 0 and side_output_path.exists()
        deg45_success = deg45_result == 0 and deg45_output_path.exists()
        
        if side_success and deg45_success:
            side_size = os.path.getsize(side_output_path) / (1024*1024)
            deg45_size = os.path.getsize(deg45_output_path) / (1024*1024)
            
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'side_video': str(side_output_path),
                'deg45_video': str(deg45_output_path),
                'side_size_mb': round(side_size, 2),
                'deg45_size_mb': round(deg45_size, 2),
                'success': True
            })
            print(f"   ✅ 完成 - 側面: {side_size:.1f}MB, 45度: {deg45_size:.1f}MB")
        else:
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'side_video': str(side_output_path),
                'deg45_video': str(deg45_output_path),
                'success': False,
                'error': f"Side: {'OK' if side_success else 'FAIL'}, 45deg: {'OK' if deg45_success else 'FAIL'}"
            })
            print(f"   ❌ 失敗 - 側面: {'成功' if side_success else '失敗'}, 45度: {'成功' if deg45_success else '失敗'}")
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 動態分割完成！成功: {successful}/{len(segment_info)} 個片段組")
    
    return segment_info

def segment_dual_videos(side_video_path, deg45_video_path, output_folder, entry_times, segment_duration=4.0, start_offset=-0.5):
    """
    根據進入時間點同步分割兩個角度的影片 (固定長度版本)
    
    參數:
    - side_video_path: 側面影片路徑
    - deg45_video_path: 45度影片路徑
    - output_folder: 輸出資料夾
    - entry_times: 球進入時間點列表
    - segment_duration: 片段時長
    - start_offset: 開始偏移時間
    """
    if not entry_times:
        print("⚠️ 沒有偵測到球進入時間點，無法進行分割")
        return []

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(side_video_path).stem.replace('_side', '').replace('__side', '')
    segment_info = []
    
    print(f"✂️ 開始同步分割兩個角度的影片到: {output_folder}")
    print(f"📹 側面影片: {Path(side_video_path).name}")
    print(f"📹 45度影片: {Path(deg45_video_path).name}")
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        segment_num = i + 1
        
        # 側面影片分割
        side_output_name = f"{input_name}_segment_{segment_num:02d}_side.mp4"
        side_output_path = output_folder / side_output_name
        
        # 45度影片分割
        deg45_output_name = f"{input_name}_segment_{segment_num:02d}_45.mp4"
        deg45_output_path = output_folder / deg45_output_name
        
        print(f"📽️ 分割片段 {segment_num}: {start_time:.2f}s - {start_time + segment_duration:.2f}s")
        
        # 分割側面影片
        side_cmd = f'ffmpeg -i "{side_video_path}" -ss {start_time} -t {segment_duration} -c copy "{side_output_path}" -y -loglevel quiet'
        side_result = os.system(side_cmd)
        
        # 分割45度影片
        deg45_cmd = f'ffmpeg -i "{deg45_video_path}" -ss {start_time} -t {segment_duration} -c copy "{deg45_output_path}" -y -loglevel quiet'
        deg45_result = os.system(deg45_cmd)
        
        # 檢查結果
        side_success = side_result == 0 and side_output_path.exists()
        deg45_success = deg45_result == 0 and deg45_output_path.exists()
        
        if side_success and deg45_success:
            side_size = os.path.getsize(side_output_path) / (1024*1024)
            deg45_size = os.path.getsize(deg45_output_path) / (1024*1024)
            
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'side_video': str(side_output_path),
                'deg45_video': str(deg45_output_path),
                'side_size_mb': round(side_size, 2),
                'deg45_size_mb': round(deg45_size, 2),
                'success': True
            })
            print(f"   ✅ 完成 - 側面: {side_size:.1f}MB, 45度: {deg45_size:.1f}MB")
        else:
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'side_video': str(side_output_path),
                'deg45_video': str(deg45_output_path),
                'success': False,
                'error': f"Side: {'OK' if side_success else 'FAIL'}, 45deg: {'OK' if deg45_success else 'FAIL'}"
            })
            print(f"   ❌ 失敗 - 側面: {'成功' if side_success else '失敗'}, 45度: {'成功' if deg45_success else '失敗'}")
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 同步分割完成！成功: {successful}/{len(segment_info)} 個片段組")
    
    return segment_info

def segment_video_dynamic(input_path, output_folder, entry_times, start_offset=-0.5, end_padding=1.0):
    """
    根據進入時間點動態分割影片 (每個片段從一個進入點到下一個進入點)
    
    參數:
    - input_path: 輸入影片路徑
    - output_folder: 輸出資料夾
    - entry_times: 球進入時間點列表
    - start_offset: 開始偏移時間 (負數表示提前開始)
    - end_padding: 最後一個片段的額外長度
    """
    if not entry_times:
        print("⚠️ 沒有偵測到球進入時間點，無法進行分割")
        return []
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(input_path).stem
    segment_info = []
    
    # 獲取影片總長度
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()
    
    print(f"✂️ 開始動態分割影片到: {output_folder}")
    print(f"📹 輸入影片: {Path(input_path).name} (總長: {video_duration:.2f}秒)")
    print(f"🎯 動態分割邏輯: 每個片段從一個球進入點到下一個球進入點")
    
    # 檢查 FFmpeg 是否可用
    ffmpeg_available = check_ffmpeg_availability()
    
    if not ffmpeg_available:
        print("⚠️ FFmpeg 不可用，使用 OpenCV 進行分割")
        # 暫時使用固定長度的OpenCV分割，稍後可以實現動態版本
        return segment_video_opencv(input_path, output_folder, entry_times, 4.0, start_offset)
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        segment_num = i + 1
        
        # 計算片段結束時間
        if i < len(entry_times) - 1:
            # 不是最後一個片段，結束時間是下一個進入點
            end_time = entry_times[i + 1] + start_offset
        else:
            # 最後一個片段，使用固定長度或影片結尾
            end_time = min(video_duration, entry_time + 4.0 + end_padding)
        
        # 確保片段長度合理
        segment_duration = max(1.0, end_time - start_time)  # 最少1秒
        
        output_name = f"{input_name}_segment_{segment_num:02d}.mp4"
        output_path = output_folder / output_name
        
        print(f"📽️ 片段{segment_num}: {start_time:.2f}s - {end_time:.2f}s (時長: {segment_duration:.2f}s)")
        print(f"   球進入時間: {entry_time:.2f}s")
        
        try:
            # 使用 subprocess 而非 os.system，提供更好的錯誤處理
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-c', 'copy',
                str(output_path),
                '-y', '-loglevel', 'error'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_path.exists():
                size_mb = os.path.getsize(output_path) / (1024*1024)
                segment_info.append({
                    'segment_id': segment_num,
                    'entry_time': entry_time,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': segment_duration,
                    'output_path': str(output_path),
                    'size_mb': round(size_mb, 2),
                    'success': True
                })
                print(f"   ✅ 完成 - 大小: {size_mb:.1f}MB")
            else:
                error_msg = result.stderr if result.stderr else "未知錯誤"
                segment_info.append({
                    'segment_id': segment_num,
                    'entry_time': entry_time,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': segment_duration,
                    'output_path': str(output_path),
                    'success': False,
                    'error': error_msg
                })
                print(f"   ❌ 失敗 - 錯誤: {error_msg}")
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ 超時 - FFmpeg 處理超過30秒")
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'output_path': str(output_path),
                'success': False,
                'error': "處理超時"
            })
        except Exception as e:
            print(f"   ❌ 錯誤 - {str(e)}")
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'output_path': str(output_path),
                'success': False,
                'error': str(e)
            })
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 動態分割完成！成功: {successful}/{len(segment_info)} 個片段")
    
    return segment_info

def segment_video(input_path, output_folder, entry_times, segment_duration=4.0, start_offset=-0.5):
    """
    根據進入時間點分割影片 (固定長度版本)
    
    參數:
    - input_path: 輸入影片路徑
    - output_folder: 輸出資料夾
    - entry_times: 球進入時間點列表
    - segment_duration: 片段時長
    - start_offset: 開始偏移時間
    """
    if not entry_times:
        print("⚠️ 沒有偵測到球進入時間點，無法進行分割")
        return []
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(input_path).stem
    segment_info = []
    
    print(f"✂️ 開始分割影片到: {output_folder}")
    
    # 檢查 FFmpeg 是否可用
    ffmpeg_available = check_ffmpeg_availability()
    
    if not ffmpeg_available:
        print("⚠️ FFmpeg 不可用，使用 OpenCV 進行分割")
        return segment_video_opencv(input_path, output_folder, entry_times, segment_duration, start_offset)
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        segment_num = i + 1
        
        output_name = f"{input_name}_segment_{segment_num:02d}.mp4"
        output_path = output_folder / output_name
        
        print(f"📽️ 分割片段 {segment_num}: {start_time:.2f}s - {start_time + segment_duration:.2f}s")
        
        try:
            # 使用 subprocess 而非 os.system，提供更好的錯誤處理
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-c', 'copy',
                str(output_path),
                '-y', '-loglevel', 'error'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                file_size = os.path.getsize(output_path) / (1024*1024)  # MB
                segment_info.append({
                    'segment_id': segment_num,
                    'entry_time': entry_time,
                    'start_time': start_time,
                    'duration': segment_duration,
                    'output_file': str(output_path),
                    'file_size_mb': round(file_size, 2),
                    'success': True
                })
                print(f"   ✅ 完成 ({file_size:.1f} MB)")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                segment_info.append({
                    'segment_id': segment_num,
                    'entry_time': entry_time,
                    'start_time': start_time,
                    'duration': segment_duration,
                    'output_file': str(output_path),
                    'success': False,
                    'error': f'FFmpeg failed: {error_msg}'
                })
                print(f"   ❌ 失敗: {error_msg}")
                
        except subprocess.TimeoutExpired:
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'output_file': str(output_path),
                'success': False,
                'error': 'FFmpeg timeout'
            })
            print(f"   ❌ 失敗: 超時")
        except Exception as e:
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'output_file': str(output_path),
                'success': False,
                'error': f'Exception: {str(e)}'
            })
            print(f"   ❌ 失敗: {str(e)}")
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 分割完成！成功: {successful}/{len(entry_times)} 個片段")
    
    return segment_info

def check_ffmpeg_availability():
    """檢查 FFmpeg 是否可用"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

def segment_video_opencv(input_path, output_folder, entry_times, segment_duration=4.0, start_offset=-0.5):
    """
    使用 OpenCV 進行影片分割（備用方案）
    """
    import cv2
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("❌ 無法開啟輸入影片")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 取得影片編解碼器資訊
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🔄 使用 OpenCV 進行影片分割 ({fps:.1f} FPS, {frame_width}x{frame_height})")
    
    input_name = Path(input_path).stem
    segment_info = []
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        segment_num = i + 1
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + segment_duration) * fps)
        
        output_name = f"{input_name}_segment_{segment_num:02d}.mp4"
        output_path = output_folder / output_name
        
        print(f"📽️ 分割片段 {segment_num}: {start_time:.2f}s - {start_time + segment_duration:.2f}s")
        
        # 設置影片位置到開始幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 創建影片寫入器
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        frames_written = 0
        current_frame = start_frame
        
        while current_frame < end_frame and current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frames_written += 1
            current_frame += 1
        
        out.release()
        
        if frames_written > 0 and output_path.exists():
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'output_file': str(output_path),
                'file_size_mb': round(file_size, 2),
                'frames_written': frames_written,
                'success': True
            })
            print(f"   ✅ 完成 ({file_size:.1f} MB, {frames_written} 幀)")
        else:
            segment_info.append({
                'segment_id': segment_num,
                'entry_time': entry_time,
                'start_time': start_time,
                'duration': segment_duration,
                'output_file': str(output_path),
                'success': False,
                'error': 'OpenCV segmentation failed'
            })
            print(f"   ❌ 失敗")
    
    cap.release()
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 OpenCV 分割完成！成功: {successful}/{len(entry_times)} 個片段")
    
    return segment_info
    
    successful = sum(1 for s in segment_info if s['success'])
    print(f"🎬 分割完成！成功: {successful}/{len(segment_info)} 個片段")
    
    return segment_info

def save_results(output_folder, input_video, entry_times, detection_results, segment_info, parameters):
    """儲存分析結果"""
    output_folder = Path(output_folder)
    
    # 儲存JSON結果
    results = {
        'analysis_info': {
            'input_video': str(input_video),
            'analysis_time': datetime.now().isoformat(),
            'total_detections': len(detection_results),
            'ball_entries': len(entry_times)
        },
        'parameters': parameters,
        'ball_entry_times': entry_times,
        'segments': segment_info,
        'detection_summary': {
            'total_frames': len(detection_results),
            'frames_with_ball': sum(1 for r in detection_results if r['detected']),
            'max_confidence': max([r['confidence'] for r in detection_results]) if detection_results else 0,
            'avg_confidence': np.mean([r['confidence'] for r in detection_results]) if detection_results else 0
        }
    }
    
    json_file = output_folder / "analysis_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 結果已儲存: {json_file}")
    
    # 儲存詳細的偵測資料 (CSV格式)
    csv_file = output_folder / "detection_details.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("frame,time_sec,detected,confidence\n")
        for result in detection_results:
            f.write(f"{result['frame']},{result['time']:.3f},{result['detected']},{result['confidence']:.4f}\n")
    
    print(f"📊 詳細資料已儲存: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='影片自動分割測試工具')
    parser.add_argument('input_video', help='輸入影片路徑 (或側面影片路徑)')
    parser.add_argument('--deg45-video', help='45度角影片路徑 (用於同步分割)')
    parser.add_argument('--output', '-o', default='segments_output', help='輸出資料夾路徑')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='偵測信心度閾值')
    parser.add_argument('--duration', '-d', type=float, default=4.0, help='片段時長(秒)')
    parser.add_argument('--min-interval', '-i', type=float, default=2.0, help='最小間隔時間(秒)')
    parser.add_argument('--start-offset', '-s', type=float, default=-0.5, help='開始偏移時間(秒)')
    parser.add_argument('--model', '-m', default='model/tennisball_OD_v1.pt', help='YOLO模型路徑')
    parser.add_argument('--simulate', action='store_true', help='使用模擬模式(不需要YOLO模型)')
    parser.add_argument('--auto-find-pair', action='store_true', help='自動尋找配對影片 (_side 配對 _45)')
    parser.add_argument('--no-visualization', action='store_true', help='不生成分析圖表')
    parser.add_argument('--dynamic', action='store_true', help='動態分割模式: 每個片段從一個球進入點到下一個球進入點')
    parser.add_argument('--end-padding', type=float, default=1.0, help='最後一個片段的額外時長(秒, 僅動態模式)')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案 (模擬模式下跳過檢查)
    if not args.simulate and not os.path.exists(args.input_video):
        print(f"❌ 輸入影片不存在: {args.input_video}")
        return 1
        
    # 自動尋找配對影片 (僅在非模擬模式下)
    deg45_video = args.deg45_video
    if args.auto_find_pair and not deg45_video and not args.simulate:
        input_path = Path(args.input_video)
        if '_side' in input_path.stem:
            # 從 side 影片尋找對應的 45 度影片
            deg45_name = input_path.stem.replace('_side', '_45').replace('__side', '__45') + input_path.suffix
            deg45_video = input_path.parent / deg45_name
            if deg45_video.exists():
                print(f"🔍 自動找到配對影片: {deg45_video}")
            else:
                print(f"⚠️ 未找到配對的45度影片: {deg45_video}")
                deg45_video = None
        elif '_45' in input_path.stem or '__45' in input_path.stem:
            # 從 45 度影片尋找對應的 side 影片
            side_name = input_path.stem.replace('_45', '_side').replace('__45', '__side') + input_path.suffix
            side_video = input_path.parent / side_name
            if side_video.exists():
                print(f"🔍 自動找到配對影片: {side_video}")
                # 交換，以 side 影片為主要分析對象
                deg45_video = args.input_video
                args.input_video = str(side_video)
                print(f"📐 以側面影片為主要分析對象: {side_video}")
            else:
                print(f"⚠️ 未找到配對的側面影片: {side_video}")
                deg45_video = None
    
    # 如果指定了配對影片，檢查其存在性 (模擬模式下跳過)
    if deg45_video and not args.simulate and not os.path.exists(deg45_video):
        print(f"❌ 配對影片不存在: {deg45_video}")
        return 1
    
    # 載入模型
    model = None
    if not args.simulate and YOLO_AVAILABLE:
        if os.path.exists(args.model):
            print(f"🤖 載入YOLO模型: {args.model}")
            try:
                model = YOLO(args.model)
                print("✅ 模型載入成功")
            except Exception as e:
                print(f"❌ 模型載入失敗: {e}")
                print("🔄 切換到模擬模式")
        else:
            print(f"⚠️ 模型檔案不存在: {args.model}")
            print("🔄 使用模擬模式")
    else:
        print("🔄 使用模擬模式")
    
    # 分析影片
    try:
        entry_times, detection_results = detect_ball_entry_points(
            args.input_video, 
            model, 
            args.confidence, 
            args.min_interval
        )
        
        # 分割影片
        if deg45_video:
            # 雙影片同步分割模式
            print(f"🎯 雙影片同步分割模式")
            
            if args.dynamic:
                print(f"🚀 使用動態分割模式")
                segment_info = segment_dual_videos_dynamic(
                    args.input_video,  # 側面影片 (用於分析)
                    deg45_video,       # 45度影片
                    args.output,
                    entry_times,
                    args.start_offset,
                    args.end_padding
                )
            else:
                print(f"📐 使用固定長度分割模式")
                segment_info = segment_dual_videos(
                    args.input_video,  # 側面影片 (用於分析)
                    deg45_video,       # 45度影片
                    args.output,
                    entry_times,
                    args.duration,
                    args.start_offset
                )
        else:
            # 單一影片分割模式
            print(f"🎯 單一影片分割模式")
            
            if args.dynamic:
                print(f"🚀 使用動態分割模式: 每個片段從一個球進入點到下一個球進入點")
                segment_info = segment_video_dynamic(
                    args.input_video,
                    args.output,
                    entry_times,
                    args.start_offset,
                    args.end_padding
                )
            else:
                print(f"📐 使用固定長度分割模式: 每個片段{args.duration}秒")
                segment_info = segment_video(
                    args.input_video,
                    args.output,
                    entry_times,
                    args.duration,
                    args.start_offset
                )
        
        # 儲存結果
        parameters = {
            'confidence_threshold': args.confidence,
            'segment_duration': args.duration,
            'min_interval': args.min_interval,
            'start_offset': args.start_offset,
            'model_path': args.model,
            'simulation_mode': args.simulate or not YOLO_AVAILABLE,
            'dynamic_mode': args.dynamic,
            'end_padding': args.end_padding if args.dynamic else None
        }
        
        save_results(
            args.output,
            args.input_video,
            entry_times,
            detection_results,
            segment_info,
            parameters
        )
        
        # 創建分析可視化圖表
        if not args.no_visualization:
            print("\n📊 建立分析可視化...")
            visualization_path = create_analysis_visualization(
                detection_results,
                entry_times,
                args.input_video,
                args.confidence,
                args.start_offset
            )
        
        print("\n🎉 測試完成！")
        print(f"📁 查看輸出資料夾: {Path(args.output).absolute()}")
        
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())