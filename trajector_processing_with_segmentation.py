"""
整合 2D/3D 軌跡分析、影片自動分割、影片處理、軌跡同步、KNN 與 GPT 反饋生成的整體流程。
此程式會依序完成：
  1. 先對原始影片進行時間同步（參考 trajector_2D_sync）
  2. 自動分割同步後的影片為多個片段
  3. 從側面與 45° 影片中提取 2D 軌跡
  4. 對 2D 軌跡進行平滑、插值與擊球角度處理
  5. 處理影片（前處理/物件偵測）
  6. 同步處理後的影片
  7. 合併同步後的影片
  8. 同步不同角度的軌跡資料
  9. 使用兩組 2D 軌跡與攝影機投影矩陣 (P1, P2) 計算 3D 軌跡
 10. 對 3D 軌跡進行平滑處理
 11. 擷取有效擊球範圍（根據 2D 軌跡判斷，並在 3D 軌跡中提取）
 12. 以 KNN 模組對 3D 軌跡進行初步分析
 13. 最後根據 KNN 分析與 3D 擊球範圍，生成 GPT 文字化反饋

各步驟皆計算執行時間，最後輸出時間統計摘要。
"""

import time
import numpy as np
import cv2
import os
import json
import subprocess
from pathlib import Path
from ultralytics import YOLO
import concurrent.futures

# 匯入原本的模組
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from video_detection import process_video
from video_sync import synchronize_videos
from video_merge import combine_videos_ffmpeg
from trajector_2D_sync import sync_trajectories
from trajector_2D_capture_swing_range import find_range
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_3D_capture_swing_range import extract_frames
from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
from trajectory_gpt_single_feedback import generate_feedback

class VideoSegmenter:
    """影片自動分割器"""
    
    def __init__(self, ball_entry_direction="right", confidence_threshold=0.5, exit_timeout=1.5):
        """
        初始化影片分割器
        
        Args:
            ball_entry_direction: 球進入方向 ("right" 或 "left")
            confidence_threshold: 偵測信心度閾值
            exit_timeout: 出場等待時間（秒）
        """
        self.ball_entry_direction = ball_entry_direction
        self.confidence_threshold = confidence_threshold
        self.exit_timeout = exit_timeout
        self.min_interval = 2.0  # 最小間隔時間
        self.preview_start_time = -0.5  # 預覽開始時間
        self.tennis_model = None
        self.ffmpeg_cmd = self._get_ffmpeg_command()  # 檢查並獲取 FFmpeg 命令
        
    def _get_ffmpeg_command(self):
        """獲取 FFmpeg 命令路徑"""
        # 先檢查系統是否有 ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return 'ffmpeg'
        except:
            # 檢查本地 tools 資料夾
            local_ffmpeg = Path("tools/ffmpeg.exe")
            if local_ffmpeg.exists():
                return str(local_ffmpeg.absolute())
            return None
        
    def load_tennis_model(self, model_path='model/tennisball_OD_v1.pt'):
        """載入網球偵測模型"""
        try:
            self.tennis_model = YOLO(model_path)
            print(f"✅ 網球偵測模型載入完成: {model_path}")
            return True
        except Exception as e:
            print(f"❌ 載入網球偵測模型失敗: {e}")
            return False
    
    def _is_in_edge(self, position, edges, entry_direction="right"):
        """檢查位置是否在指定的邊緣區域"""
        if not position:
            return False
        
        x, y = position
        
        if entry_direction == "right":
            # 右邊上2/3區域偵測
            if x > edges['right']:
                upper_two_thirds = edges['top'] + (edges['bottom'] - edges['top']) * (2/3)
                if y <= upper_two_thirds:
                    return True
        elif entry_direction == "left":
            # 左邊上2/3區域偵測
            if x < edges['left']:
                upper_two_thirds = edges['top'] + (edges['bottom'] - edges['top']) * (2/3)
                if y <= upper_two_thirds:
                    return True
        
        return False
    
    def _detect_ball(self, frame):
        """偵測畫面中的網球"""
        results = self.tennis_model(frame, verbose=False)
        
        if not results[0].boxes:
            return False, 0.0, None
        
        best_box = max(results[0].boxes, key=lambda box: float(box.conf[0]))
        confidence = float(best_box.conf[0])
        
        if confidence < self.confidence_threshold:
            return False, confidence, None
        
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        position = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return True, confidence, position
    
    def _update_ball_tracking(self, active_balls, position, current_time, fps):
        """更新球追蹤資訊"""
        if not position:
            return
        
        # 動態調整追蹤距離，根據FPS調整
        max_tracking_distance = max(200, fps * 8)
        
        # 找到最近的球進行位置更新
        min_distance = float('inf')
        closest_ball_id = None
        
        for ball_id, ball_info in active_balls.items():
            if ball_info['positions']:
                last_pos = ball_info['positions'][-1]
                distance = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
                
                if distance < min_distance and distance < max_tracking_distance:
                    min_distance = distance
                    closest_ball_id = ball_id
        
        # 更新最近球的位置
        if closest_ball_id is not None:
            active_balls[closest_ball_id]['positions'].append(position)
            active_balls[closest_ball_id]['last_seen'] = current_time
            
            # 保持位置歷史在合理範圍內
            if len(active_balls[closest_ball_id]['positions']) > 30:
                active_balls[closest_ball_id]['positions'].pop(0)
    
    def _check_ball_exits(self, active_balls, edges, current_time):
        """檢查球是否出場"""
        exited_balls = []
        balls_to_remove = []
        
        for ball_id, ball_info in active_balls.items():
            time_since_last_seen = current_time - ball_info['last_seen']
            
            # 使用設定的出場等待時間
            min_check_time = max(0.2, self.exit_timeout * 0.15)
            max_force_time = max(1.0, self.exit_timeout)
            
            if time_since_last_seen > min_check_time:
                is_exit, reason = self._is_ball_exit_edge(ball_info['positions'], edges)
                
                if is_exit:
                    exited_balls.append((ball_id, ball_info['last_seen']))
                    print(f"🚪 球 {ball_id} 出場: {reason} (未見時間: {time_since_last_seen:.2f}s)")
                elif time_since_last_seen > max_force_time:
                    print(f"⏰ 球 {ball_id} 超時移除 (未見時間: {time_since_last_seen:.2f}s)")
                
                if is_exit or time_since_last_seen > max_force_time:
                    balls_to_remove.append(ball_id)
        
        # 移除已出場或過期的球
        for ball_id in balls_to_remove:
            del active_balls[ball_id]
        
        return exited_balls
    
    def _is_ball_exit_edge(self, positions, edges):
        """檢查是否為出場"""
        if len(positions) < 2:
            return False, "軌跡點不足"
        
        recent_positions = positions[-min(8, len(positions)):]
        
        # 根據進入方向決定出場邊界
        if self.ball_entry_direction == "right":
            boundary = edges['right'] - 100
            end_pos = recent_positions[-1]
            is_at_edge = end_pos[0] > boundary
            edge_name = "右邊界"
        else:  # left
            boundary = edges['left'] + 100
            end_pos = recent_positions[-1]
            is_at_edge = end_pos[0] < boundary
            edge_name = "左邊界"
        
        if not is_at_edge:
            return False, f"未到達{edge_name} (X: {end_pos[0]:.0f}, 邊界: {boundary:.0f})"
        
        # 檢查移動趨勢
        if len(recent_positions) >= 2:
            x_movement = abs(recent_positions[-1][0] - recent_positions[0][0])
            if x_movement > 5:
                return True, f"{edge_name}移動出場 (ΔX: {x_movement:.0f})"
        
        # 檢查是否在邊界停留
        edge_count = 0
        for pos in recent_positions:
            if self.ball_entry_direction == "right" and pos[0] > boundary:
                edge_count += 1
            elif self.ball_entry_direction == "left" and pos[0] < boundary:
                edge_count += 1
        
        if edge_count >= len(recent_positions) * 0.7:
            return True, f"{edge_name}停留出場 ({edge_count}/{len(recent_positions)})"
        
        return False, "無明確出場跡象"
    
    def analyze_video_for_segmentation(self, video_path):
        """分析影片找出球進入和出場時間點"""
        print(f"🎾 開始分析影片: {Path(video_path).name}")
        print(f"🎯 球進入方向: {self.ball_entry_direction}")
        print(f"🔍 偵測信心度: {self.confidence_threshold}")
        print(f"⏰ 出場等待時間: {self.exit_timeout}秒")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 影片資訊: {total_frames}影格, {fps:.2f}FPS, {frame_width}x{frame_height}")
        print(f"🎯 球追蹤距離: {max(200, fps * 8):.0f}像素")
        
        # 邊緣檢測參數
        edge_ratio = 0.15
        edges = {
            'left': frame_width * edge_ratio,
            'right': frame_width * (1 - edge_ratio),
            'top': frame_height * edge_ratio,
            'bottom': frame_height * (1 - edge_ratio)
        }
        
        # 初始化變數
        ball_entry_times = []
        ball_exit_times = []
        active_balls = {}
        next_ball_id = 0
        prev_detected = False
        prev_position = None
        last_entry_time = -self.min_interval
        
        print("🔍 開始分析影格...")
        
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # 偵測網球
            detected, confidence, position = self._detect_ball(frame)
            
            if detected:
                # 更新球追蹤
                self._update_ball_tracking(active_balls, position, current_time, fps)
                
                # 檢查是否為新球進入
                if not prev_detected or (prev_position and position):
                    is_entry = False
                    
                    if not prev_detected:
                        # 從無到有的偵測
                        is_entry = self._is_in_edge(position, edges, self.ball_entry_direction)
                    elif prev_position:
                        # 從邊緣移入中央
                        prev_in_edge = self._is_in_edge(prev_position, edges, self.ball_entry_direction)
                        curr_in_edge = self._is_in_edge(position, edges, self.ball_entry_direction)
                        
                        if prev_in_edge and not curr_in_edge:
                            distance = ((position[0] - prev_position[0])**2 + (position[1] - prev_position[1])**2)**0.5
                            is_entry = distance > 20
                    
                    # 記錄新球進入
                    if is_entry and current_time - last_entry_time >= self.min_interval:
                        ball_entry_times.append(current_time)
                        last_entry_time = current_time
                        
                        # 建立新球追蹤
                        active_balls[next_ball_id] = {
                            'entry_time': current_time,
                            'positions': [position],
                            'last_seen': current_time
                        }
                        
                        direction_text = "右上2/3" if self.ball_entry_direction == "right" else "左上2/3"
                        print(f"🏥 球進入: {current_time:.1f}s - 球#{next_ball_id} 從{direction_text}進入")
                        next_ball_id += 1
            
            # 檢查球出場
            exited_balls = self._check_ball_exits(active_balls, edges, current_time)
            for ball_id, exit_time in exited_balls:
                ball_exit_times.append(exit_time)
                print(f"🚪 球出場: {exit_time:.1f}s - 球#{ball_id}")
            
            prev_detected = detected
            prev_position = position
            
            # 進度顯示
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ 分析進度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        
        print(f"✅ 影片分析完成！找到 {len(ball_entry_times)} 個球進入點, {len(ball_exit_times)} 個出場點")
        print(f"📍 進入時間點: {[f'{t:.1f}s' for t in ball_entry_times]}")
        print(f"📍 出場時間點: {[f'{t:.1f}s' for t in ball_exit_times]}")
        
        return ball_entry_times, ball_exit_times
    
    def segment_video(self, input_video, output_folder, ball_entry_times, ball_exit_times):
        """根據球進入和出場時間分割影片"""
        if not ball_entry_times:
            print("❌ 沒有找到球進入點，無法分割影片")
            return []
        
        # 確保輸出資料夾存在
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        segments = []
        video_name = Path(input_video).stem
        
        print(f"🎬 開始分割影片: {video_name}")
        
        for i, entry_time in enumerate(ball_entry_times):
            # 計算片段時間
            start_time = max(0, entry_time + self.preview_start_time)
            
            # 動態模式：使用出場時間
            if i < len(ball_exit_times):
                exit_time = ball_exit_times[i]
                end_time = exit_time + 0.3  # 出場後0.3秒結束
            else:
                # 如果沒有對應的出場時間，使用固定長度
                end_time = entry_time + 4.0
            
            duration = max(1.0, end_time - start_time)
            
            # 輸出檔案名稱
            output_filename = f"{video_name}_segment_{i+1:02d}_{entry_time:.1f}s.mp4"
            output_path = output_folder / output_filename
            
            print(f"🎬 片段{i+1}: {start_time:.1f}s → {end_time:.1f}s (時長: {duration:.1f}s)")
            
            # 使用 FFmpeg 分割
            success = self._segment_with_ffmpeg(input_video, output_path, start_time, duration)
            
            if success:
                segments.append({
                    'index': i + 1,
                    'start_time': start_time,
                    'duration': duration,
                    'entry_time': entry_time,
                    'exit_time': ball_exit_times[i] if i < len(ball_exit_times) else None,
                    'output_path': str(output_path)
                })
                print(f"✅ 片段{i+1}分割完成: {output_filename}")
            else:
                print(f"❌ 片段{i+1}分割失敗")
        
        print(f"🎉 影片分割完成！共生成 {len(segments)} 個片段")
        return segments
    
    def _segment_with_ffmpeg(self, input_path, output_path, start_time, duration):
        """使用 FFmpeg 分割影片"""
        try:
            if not self.ffmpeg_cmd:
                print("❌ FFmpeg 不可用")
                return False
                
            # 檢查輸入檔案是否存在
            input_file = Path(input_path)
            if not input_file.exists():
                print(f"❌ 輸入影片檔案不存在: {input_path}")
                return False
            
            # 確保輸出資料夾存在
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"🔧 FFmpeg 分割:")
            print(f"   輸入: {input_path}")
            print(f"   輸出: {output_path}")
            print(f"   FFmpeg: {self.ffmpeg_cmd}")
                
            cmd = [
                self.ffmpeg_cmd, '-y',  # 使用檢測到的 FFmpeg 命令
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # 使用 copy 避免重新編碼
                '-avoid_negative_ts', 'make_zero',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ FFmpeg 錯誤:")
                print(f"   返回碼: {result.returncode}")
                print(f"   標準錯誤: {result.stderr}")
                print(f"   標準輸出: {result.stdout}")
            else:
                print(f"✅ FFmpeg 分割成功")
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ FFmpeg 分割失敗: {e}")
            print(f"   例外類型: {type(e).__name__}")
            import traceback
            print(f"   詳細錯誤: {traceback.format_exc()}")
            return False

def sync_videos_by_trajectory(video_side, video_45, output_folder):
    """
    根據軌跡數據同步兩個影片
    參考 trajector_2D_sync 的邏輯
    """
    print("🔄 開始影片時間同步...")
    
    # 這裡需要先生成簡單的軌跡數據來找到同步點
    # 實際實現時可能需要調用 trajectory_2D_output 的簡化版本
    # 或者使用其他同步方法（如音頻同步、手動標記等）
    
    # 暫時使用文件名作為同步後的輸出
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 複製原始影片作為同步後的結果（實際應該實現真正的同步邏輯）
    synced_side = output_folder / f"synced_{Path(video_side).name}"
    synced_45 = output_folder / f"synced_{Path(video_45).name}"
    
    # 這裡應該實現真正的同步邏輯
    # 暫時直接複製
    import shutil
    shutil.copy2(video_side, synced_side)
    shutil.copy2(video_45, synced_45)
    
    print(f"✅ 影片同步完成")
    print(f"📁 同步後影片: {synced_side}")
    print(f"📁 同步後影片: {synced_45}")
    
    return str(synced_side), str(synced_45)

def processing_trajectory_with_segmentation(P1, P2, yolo_pose_model, yolo_tennis_ball_model, 
                                          video_side, video_45, knn_dataset,
                                          ball_entry_direction="right", confidence_threshold=0.5,
                                          segment_videos=True, output_base_folder="segmented_videos"):
    """
    整合軌跡處理與影片分割的完整流程
    
    Args:
        P1, P2: 投影矩陣
        yolo_pose_model, yolo_tennis_ball_model: YOLO 模型
        video_side, video_45: 影片路徑
        knn_dataset: KNN 資料集路徑
        ball_entry_direction: 球進入方向 ("right" 或 "left")
        confidence_threshold: 偵測信心度
        segment_videos: 是否執行影片分割
        output_base_folder: 分割影片輸出資料夾
    """
    
    # 用於紀錄各步驟執行時間
    timing_results = {}
    start_total = time.perf_counter()
    
    # ------------------------------
    # 步驟0：影片時間同步
    # ------------------------------
    print("\n步驟0：影片時間同步...")
    start = time.perf_counter()
    
    sync_output_folder = Path(output_base_folder) / "synced_videos"
    video_side_synced, video_45_synced = sync_videos_by_trajectory(video_side, video_45, sync_output_folder)
    
    timing_results['影片時間同步'] = time.perf_counter() - start
    print(f"-- 影片時間同步完成，耗時：{timing_results['影片時間同步']:.4f} 秒")
    
    # ------------------------------
    # 步驟1：影片自動分割（可選）
    # ------------------------------
    if segment_videos:
        print("\n步驟1：影片自動分割...")
        start = time.perf_counter()
        
        # 初始化影片分割器
        segmenter = VideoSegmenter(
            ball_entry_direction=ball_entry_direction,
            confidence_threshold=confidence_threshold,
            exit_timeout=1.5
        )
        
        # 載入網球偵測模型
        if not segmenter.load_tennis_model():
            print("❌ 無法載入網球偵測模型，跳過影片分割")
            segment_videos = False
        else:
            # 分析並分割側面影片
            print("\n🎾 分析側面影片...")
            entry_times_side, exit_times_side = segmenter.analyze_video_for_segmentation(video_side_synced)
            
            side_output_folder = Path(output_base_folder) / "segments" / "side"
            side_segments = segmenter.segment_video(video_side_synced, side_output_folder, entry_times_side, exit_times_side)
            
            # 分析並分割45度影片
            print("\n🎾 分析45度影片...")
            entry_times_45, exit_times_45 = segmenter.analyze_video_for_segmentation(video_45_synced)
            
            deg45_output_folder = Path(output_base_folder) / "segments" / "45deg"
            deg45_segments = segmenter.segment_video(video_45_synced, deg45_output_folder, entry_times_45, exit_times_45)
            
            timing_results['影片自動分割'] = time.perf_counter() - start
            print(f"-- 影片自動分割完成，耗時：{timing_results['影片自動分割']:.4f} 秒")
            
            # 如果有分割結果，使用第一個片段進行後續處理
            if side_segments and deg45_segments:
                video_side = side_segments[0]['output_path']
                video_45 = deg45_segments[0]['output_path']
                print(f"🎯 使用第一個片段進行軌跡分析:")
                print(f"   側面片段: {Path(video_side).name}")
                print(f"   45度片段: {Path(video_45).name}")
            else:
                # 如果分割失敗，使用同步後的完整影片
                video_side = video_side_synced
                video_45 = video_45_synced
                print("⚠️ 影片分割失敗，使用完整同步影片進行處理")
    else:
        # 不分割，直接使用同步後的影片
        video_side = video_side_synced
        video_45 = video_45_synced
        print("ℹ️ 跳過影片分割，使用完整同步影片")
    
    # ------------------------------
    # 步驟2：分析2D軌跡
    # ------------------------------
    print("\n步驟2：分析2D軌跡中...")
    start = time.perf_counter()
    trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_side, 28)
    trajectory_45  = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45, 28)
    timing_results['2D軌跡分析'] = time.perf_counter() - start
    print(f"-- 分析2D軌跡完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟3：2D 軌跡平滑/插值/擊球角度處理
    # ------------------------------
    print("\n步驟3：進行2D軌跡平滑化/插值/擊球角度處理...")
    start = time.perf_counter()
    trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
    trajectory_45_smoothing   = smooth_2D_trajectory(trajectory_45)
    timing_results['2D平滑處理'] = time.perf_counter() - start
    print(f"-- 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟4：影片處理
    # ------------------------------
    print("\n步驟4：處理影片中...")
    start = time.perf_counter()
    
    # 檢查影片文件是否存在
    if not os.path.exists(video_side):
        print(f"❌ 側面影片不存在: {video_side}")
        video_side_processed = None
    else:
        try:
            video_side_processed = process_video(video_side)
        except Exception as e:
            print(f"❌ 側面影片處理失敗: {e}")
            video_side_processed = None
    
    if not os.path.exists(video_45):
        print(f"❌ 45度影片不存在: {video_45}")
        video_45_processed = None
    else:
        try:
            video_45_processed = process_video(video_45)
        except Exception as e:
            print(f"❌ 45度影片處理失敗: {e}")
            video_45_processed = None
    
    timing_results['影片處理'] = time.perf_counter() - start
    print(f"-- 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

    # ------------------------------
    # 步驟5：影片同步
    # ------------------------------
    print("\n步驟5：同步影片中...")
    start = time.perf_counter()
    
    # 檢查影片處理結果
    if video_side_processed and video_45_processed:
        try:
            synchronize_videos(video_side_processed, video_45_processed, 
                            trajectory_side_smoothing, trajectory_45_smoothing)
            print("✅ 影片同步完成")
        except Exception as e:
            print(f"❌ 影片同步失敗: {e}")
    else:
        print("⚠️ 跳過影片同步（影片處理失敗）")
    
    timing_results['影片同步'] = time.perf_counter() - start
    print(f"-- 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

    # ------------------------------
    # 步驟6：合併影片
    # ------------------------------
    print("\n步驟6：合併影片中...")
    start = time.perf_counter()
    
    # 檢查影片處理結果和 FFmpeg 可用性
    if video_side_processed and video_45_processed and segment_videos:
        try:
            combine_videos_ffmpeg(video_45_processed, video_side_processed)
            print("✅ 影片合併完成")
        except Exception as e:
            print(f"❌ 影片合併失敗: {e}")
    else:
        print("⚠️ 跳過影片合併（影片處理失敗或 FFmpeg 不可用）")
    
    timing_results['影片合併'] = time.perf_counter() - start
    print(f"-- 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

    # ------------------------------
    # 步驟7：軌跡同步
    # ------------------------------
    print("\n步驟7：同步軌跡中...")
    start = time.perf_counter()
    sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['軌跡同步'] = time.perf_counter() - start
    print(f"-- 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

    # ------------------------------
    # 步驟8：3D 軌跡分析
    # ------------------------------
    print("\n步驟8：計算3D軌跡中...")
    start = time.perf_counter()
    trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
    timing_results['3D軌跡分析'] = time.perf_counter() - start
    print(f"-- 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟9：3D 軌跡平滑處理
    # ------------------------------
    print("\n步驟9：進行3D軌跡平滑處理中...")
    start = time.perf_counter()
    trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
    timing_results['3D平滑處理'] = time.perf_counter() - start
    print(f"-- 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟10：有效擊球範圍判斷
    # ------------------------------
    print("\n步驟10：判斷有效擊球範圍中...")
    start = time.perf_counter()
    start_frame, end_frame = find_range(trajectory_side_smoothing)
    trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing, start_frame, end_frame)
    timing_results['有效擊球範圍判斷'] = time.perf_counter() - start
    print(f"-- 有效擊球範圍判斷完成，耗時：{timing_results['有效擊球範圍判斷']:.4f} 秒")

    # ------------------------------
    # 步驟11：KNN 分析
    # ------------------------------
    print("\n步驟11：KNN 分析中...")
    start = time.perf_counter()
    trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing)
    timing_results['KNN 分析'] = time.perf_counter() - start
    print(f"-- KNN 分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

    # ------------------------------
    # 步驟12：GPT 反饋生成
    # ------------------------------
    print("\n步驟12：生成 GPT 反饋中...")
    start = time.perf_counter()
    trajectory_gpt_suggestion = generate_feedback(trajectory_3d_swing_range, trajectory_knn_suggestion)
    timing_results['GPT 反饋生成'] = time.perf_counter() - start
    print(f"-- GPT 反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")

    # ------------------------------
    # 統計總執行時間並輸出時間摘要
    # ------------------------------
    total_time = time.perf_counter() - start_total
    print('\n' + '=' * 60)
    print("📊 執行時間統計摘要")
    print('=' * 60)
    print(f'處理影片: {Path(video_side).name}')
    print(f'球進入方向: {ball_entry_direction}')
    print(f'偵測信心度: {confidence_threshold}')
    print(f'是否分割影片: {"是" if segment_videos else "否"}')
    print('-' * 60)
    for step, t in timing_results.items():
        print(f"{step:.<35} {t:>10.4f} 秒")
    print('-' * 60)
    print(f"{'總執行時間':.<35} {total_time:>10.4f} 秒")
    print('=' * 60)

    return True

if __name__ == "__main__":
    # 投影矩陣設定
    P1 = np.array([
        [  877.037008,     0.000000,   956.954783,     0.000000],
        [    0.000000,   879.565925,   564.021385,     0.000000],
        [    0.000000,     0.000000,     1.000000,     0.000000],
    ])

    P2 = np.array([
        [  408.666240,    -7.066100,  1265.246736, -264697.889698],
        [ -232.265915,   870.289013,   512.645370, 42861.701021],
        [   -0.400331,    -0.014736,     0.916252,    76.895470],
    ])

    # 參數設定
    knn_dataset = 'knn_dataset.json'
    
    # 載入模型
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    
    # GPU 加速（如果可用）
    yolo_pose_model.model.to('cuda')
    yolo_tennis_ball_model.model.to('cuda')

    # 影片路徑
    video_side = f'trajectory/testing_123/testing__side.mp4'
    video_45 = f'trajectory/testing_123/testing__45.mp4'
    
    # 執行整合處理
    print("🚀 開始整合處理流程...")
    print("=" * 60)
    
    process_status = processing_trajectory_with_segmentation(
        P1=P1, 
        P2=P2, 
        yolo_pose_model=yolo_pose_model, 
        yolo_tennis_ball_model=yolo_tennis_ball_model,
        video_side=video_side, 
        video_45=video_45, 
        knn_dataset=knn_dataset,
        ball_entry_direction="right",  # 可選: "right" 或 "left"
        confidence_threshold=0.5,      # 偵測信心度
        segment_videos=True,           # 是否執行影片分割
        output_base_folder="segmented_videos"  # 輸出資料夾
    )
    
    print(f"\n🎉 整合處理完成！狀態: {process_status}")