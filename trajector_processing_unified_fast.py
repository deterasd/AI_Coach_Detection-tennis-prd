"""
統一輸出管理的軌跡處理流程 - 快速實驗版本
優化策略：
1. 影片預載到 RAM (使用 RAM disk 或記憶體緩存)
2. 跳幀處理 (frame skipping)
3. 批次處理優化
4. 多線程/多進程加速
5. GPU 記憶體管理優化
"""

import json
import cv2
import numpy as np
import shutil
import gc
import os
import time
import psutil
import torch
import subprocess
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# ============================================================================
# 實驗性能優化參數
# ============================================================================
ENABLE_FRAME_SKIP = True        # 啟用跳幀
FRAME_SKIP_RATE = 2             # 跳幀率 (處理每2幀中的1幀)
ENABLE_RAM_CACHE = True         # 啟用 RAM 緩存
ENABLE_BATCH_PROCESSING = True  # 啟用批次處理
BATCH_SIZE_MULTIPLIER = 2       # 批次大小倍數
ENABLE_PARALLEL = True          # 啟用並行處理
MAX_WORKERS = 4                 # 最大工作線程數

print("🚀 快速實驗版本載入")
print(f"   跳幀處理: {'啟用' if ENABLE_FRAME_SKIP else '停用'} (率: 1/{FRAME_SKIP_RATE})")
print(f"   RAM 緩存: {'啟用' if ENABLE_RAM_CACHE else '停用'}")
print(f"   批次處理: {'啟用' if ENABLE_BATCH_PROCESSING else '停用'}")
print(f"   並行處理: {'啟用' if ENABLE_PARALLEL else '停用'}")

# ============================================================================
# RAM 緩存管理
# ============================================================================

class VideoRAMCache:
    """影片 RAM 緩存管理器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def load_video_to_ram(self, video_path):
        """將影片載入 RAM"""
        video_path = str(video_path)
        
        with self.cache_lock:
            if video_path in self.cache:
                print(f"   ✅ 使用快取: {Path(video_path).name}")
                return self.cache[video_path]
        
        print(f"   📥 載入影片到 RAM: {Path(video_path).name}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 跳幀載入以節省記憶體
        skip_rate = FRAME_SKIP_RATE if ENABLE_FRAME_SKIP else 1
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳幀策略
            if frame_idx % skip_rate == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        video_data = {
            'frames': frames,
            'fps': fps,
            'total_frames': len(frames),
            'original_total_frames': total_frames,
            'skip_rate': skip_rate
        }
        
        # 計算記憶體使用
        memory_mb = sum(f.nbytes for f in frames) / 1024 / 1024
        load_time = time.time() - start_time
        
        print(f"      載入完成: {len(frames)} 幀 (原始: {total_frames})")
        print(f"      記憶體使用: {memory_mb:.1f} MB")
        print(f"      載入時間: {load_time:.2f} 秒")
        
        with self.cache_lock:
            self.cache[video_path] = video_data
        
        return video_data
    
    def clear_cache(self, video_path=None):
        """清除快取"""
        with self.cache_lock:
            if video_path:
                if str(video_path) in self.cache:
                    del self.cache[str(video_path)]
                    gc.collect()
            else:
                self.cache.clear()
                gc.collect()

# 全域快取實例
video_cache = VideoRAMCache()

# ============================================================================
# 快速處理函數
# ============================================================================

def analyze_trajectory_with_output_folder_fast(pose_model, ball_model, video_path, batch_size, output_folder):
    """
    快速版本:分析軌跡並將結果保存到指定資料夾
    優化:使用 RAM 緩存 + 跳幀 + 批次處理
    
    注意:為避免並行衝突,每個線程使用獨立的模型實例
    """
    print(f"🚄 快速分析 2D 軌跡: {Path(video_path).name}")
    
    # 為每個線程創建獨立的模型實例(避免並行衝突)
    import threading
    thread_id = threading.current_thread().ident
    
    # 重新載入模型以避免 fuse 衝突
    local_pose_model = YOLO(r'model\yolov8n-pose.pt')
    local_ball_model = YOLO(r'model\tennisball_OD_v1.pt')
    
    # 載入影片到 RAM
    if ENABLE_RAM_CACHE:
        video_data = video_cache.load_video_to_ram(video_path)
        frames = video_data['frames']
        fps = video_data['fps']
        skip_rate = video_data['skip_rate']
    else:
        # 傳統方式
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        skip_rate = 1
    
    # 批次處理優化
    if ENABLE_BATCH_PROCESSING:
        effective_batch_size = batch_size * BATCH_SIZE_MULTIPLIER
    else:
        effective_batch_size = batch_size
    
    trajectory = []
    total_frames = len(frames)
    
    print(f"   處理 {total_frames} 幀 (批次大小: {effective_batch_size})")
    
    for i in range(0, total_frames, effective_batch_size):
        batch_frames = frames[i:i + effective_batch_size]
        
        # 姿態估計 (使用本地模型實例)
        pose_results = local_pose_model(batch_frames, verbose=False)
        
        # 網球偵測 (使用本地模型實例)
        ball_results = local_ball_model(batch_frames, verbose=False)
        
        # 提取數據
        for frame_idx, (pose_res, ball_res) in enumerate(zip(pose_results, ball_results)):
            frame_data = {}
            
            # 提取關節點
            if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                keypoints = pose_res.keypoints[0].xy[0].cpu().numpy()
                keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                
                for idx, name in enumerate(keypoint_names):
                    if idx < len(keypoints):
                        x, y = keypoints[idx]
                        frame_data[name] = {"x": float(x), "y": float(y)}
            
            # 提取網球
            if ball_res.boxes is not None and len(ball_res.boxes) > 0:
                best_box = max(ball_res.boxes, key=lambda box: float(box.conf[0]))
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                frame_data['tennis_ball'] = {
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2)
                }
            
            trajectory.append(frame_data)
        
        # 顯示進度
        if (i // effective_batch_size) % 10 == 0:
            progress = (i + len(batch_frames)) / total_frames * 100
            print(f"      進度: {progress:.1f}%")
    
    # 如果使用了跳幀，進行插值補償
    if skip_rate > 1:
        print(f"   🔄 進行跳幀插值補償 (跳幀率: 1/{skip_rate})")
        trajectory = interpolate_skipped_frames(trajectory, skip_rate)
    
    # 保存結果
    video_name = Path(video_path).stem
    output_path = Path(output_folder) / f"{video_name}(2D_trajectory).json"
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"   ✅ 完成: {len(trajectory)} 幀")
    
    return str(output_path)

def interpolate_skipped_frames(trajectory, skip_rate):
    """
    插值跳過的幀
    使用線性插值填補跳幀造成的空缺
    """
    if skip_rate <= 1:
        return trajectory
    
    # 收集所有可能的關節點名稱(包括 tennis_ball)
    all_keys = set()
    for frame in trajectory:
        all_keys.update(frame.keys())
    
    interpolated = []
    frame_counter = 0  # 幀計數器
    
    for i in range(len(trajectory) - 1):
        # 添加當前幀 (添加 frame 編號)
        current_frame_data = trajectory[i].copy()
        current_frame_data['frame'] = frame_counter
        
        # 確保所有關節點都存在(即使為 None)
        for key in all_keys:
            if key not in current_frame_data:
                current_frame_data[key] = None
        
        interpolated.append(current_frame_data)
        frame_counter += 1
        
        # 插值中間幀
        for j in range(1, skip_rate):
            alpha = j / skip_rate
            interp_frame = {'frame': frame_counter}  # 添加 frame 編號
            
            # 對所有關節點進行插值
            for key in all_keys:
                if key == 'frame':  # 跳過 frame 欄位
                    continue
                
                curr_point = trajectory[i].get(key)
                next_point = trajectory[i + 1].get(key)
                
                # 如果兩個點都存在且不為 None,進行插值
                if curr_point is not None and next_point is not None:
                    interp_frame[key] = {
                        "x": curr_point["x"] * (1 - alpha) + next_point["x"] * alpha,
                        "y": curr_point["y"] * (1 - alpha) + next_point["y"] * alpha
                    }
                else:
                    # 否則設為 None
                    interp_frame[key] = None
            
            interpolated.append(interp_frame)
            frame_counter += 1
    
    # 添加最後一幀 (添加 frame 編號)
    last_frame_data = trajectory[-1].copy()
    last_frame_data['frame'] = frame_counter
    
    # 確保所有關節點都存在(即使為 None)
    for key in all_keys:
        if key not in last_frame_data:
            last_frame_data[key] = None
    
    interpolated.append(last_frame_data)
    
    return interpolated

def smooth_2D_trajectory_with_output_folder_fast(trajectory_path, output_folder):
    """
    快速版本：平滑處理2D軌跡
    """
    from trajector_2D_smoothing import smooth_2D_trajectory
    
    print(f"🚄 快速平滑 2D 軌跡")
    
    # 執行平滑處理
    smoothed_trajectory_path = smooth_2D_trajectory(trajectory_path)
    
    # 移動結果到指定資料夾
    source_path = Path(smoothed_trajectory_path)
    target_path = Path(output_folder) / source_path.name
    
    if source_path.exists() and source_path != target_path:
        shutil.move(str(source_path), str(target_path))
        return str(target_path)
    
    return smoothed_trajectory_path

def process_video_with_output_folder_fast(video_path, output_folder):
    """
    快速版本：處理影片
    優化：使用 RAM 緩存 + 跳幀
    """
    from video_detection import process_video
    
    print(f"🚄 快速處理影片: {Path(video_path).name}")
    
    # 執行影片處理（原函數已經夠快）
    processed_video_path = process_video(video_path)
    
    # 移動結果到指定資料夾
    if processed_video_path and Path(processed_video_path).exists():
        source_path = Path(processed_video_path)
        target_path = Path(output_folder) / source_path.name
        
        if source_path != target_path:
            shutil.move(str(source_path), str(target_path))
            return str(target_path)
    
    return processed_video_path

# ============================================================================
# 保存函數（與原版相同）
# ============================================================================

def save_3d_trajectory_with_output_folder(trajectory_3d, output_folder, name):
    """保存3D軌跡到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory).json"
    
    with open(output_path, 'w') as f:
        json.dump(trajectory_3d, f, indent=2)
    
    return str(output_path)

def save_3d_smoothed_trajectory_with_output_folder(trajectory_3d_smoothing, output_folder, name):
    """保存3D平滑軌跡到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed).json"
    
    with open(output_path, 'w') as f:
        json.dump(trajectory_3d_smoothing, f, indent=2)
    
    return str(output_path)

def save_3d_swing_range_with_output_folder(trajectory_3d_swing_range, output_folder, name):
    """保存3D擊球範圍軌跡到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed)_only_swing.json"
    
    with open(output_path, 'w') as f:
        json.dump(trajectory_3d_swing_range, f, indent=2)
    
    return str(output_path)

def save_knn_feedback_with_output_folder(knn_result, output_folder, name):
    """保存KNN反饋到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment_knn_feedback.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(knn_result)
    
    return str(output_path)

def save_gpt_feedback_with_output_folder(gpt_result, output_folder, name):
    """保存GPT反饋到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment_gpt_feedback.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_result, f, ensure_ascii=False, indent=2)
    
    return str(output_path)

# ============================================================================
# 記憶體管理（與原版相同）
# ============================================================================

def clear_all_memory():
    """清理所有記憶體（GPU + RAM）"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_system_memory():
    """檢查系統記憶體使用情況"""
    memory = psutil.virtual_memory()
    return memory.available > 2 * 1024**3  # 至少需要 2GB 可用

def check_gpu_memory():
    """檢查 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        return (total_memory - cached_memory) > 1.0  # 至少需要 1GB 可用
    return False

# ============================================================================
# 主處理流程（快速版本）
# ============================================================================

def processing_trajectory_unified_fast(P1, P2, yolo_pose_model, yolo_tennis_ball_model, 
                                      video_side, video_45, knn_dataset, name,
                                      ball_entry_direction="right", confidence_threshold=0.5,
                                      output_folder=None, segment_videos=False):
    """
    快速版本：統一輸出管理的完整軌跡處理流程
    優化：RAM 緩存 + 跳幀 + 批次處理 + 並行處理
    """
    
    if output_folder is None:
        output_folder = Path("trajectory") / f"{name}__trajectory"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timing_results = {}
    start_total = time.time()
    
    print(f"🚀 【快速模式】開始 {name} 的完整軌跡分析流程")
    print(f"📁 輸出資料夾: {output_folder}")
    print("=" * 60)
    
    # 檢查系統資源
    print("\n🔍 檢查系統資源...")
    clear_all_memory()
    gpu_ok = check_gpu_memory()
    ram_ok = check_system_memory()
    
    if not ram_ok:
        print("⚠️ 系統記憶體不足，將自動使用 CPU 模式")
    
    try:
        # 預載影片到 RAM
        if ENABLE_RAM_CACHE:
            print("\n📥 預載影片到 RAM...")
            video_cache.load_video_to_ram(video_side)
            video_cache.load_video_to_ram(video_45)
        
        # 處理流程
        success = process_single_video_set_fast(
            P1, P2, yolo_pose_model, yolo_tennis_ball_model,
            video_side, video_45, knn_dataset,
            name, str(output_folder), timing_results
        )
        
        if success:
            total_time = time.time() - start_total
            generate_processing_summary(output_folder, name, timing_results, total_time)
            print(f"\n✅ 處理完成！總耗時: {total_time:.2f} 秒")
        
        # 清除快取
        if ENABLE_RAM_CACHE:
            print("\n🧹 清除 RAM 快取...")
            video_cache.clear_cache()
        
        return success
        
    except Exception as e:
        print(f"\n💥 處理過程發生錯誤: {e}")
        
        # 清除快取
        if ENABLE_RAM_CACHE:
            video_cache.clear_cache()
        
        # 記錄錯誤到日誌
        error_log = output_folder / "logs" / "processing_error.log"
        error_log.parent.mkdir(exist_ok=True)
        
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"錯誤時間: {time.time()}\n")
            f.write(f"處理對象: {name}\n")
            f.write(f"錯誤訊息: {str(e)}\n")
        
        return False

def process_single_video_set_fast(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                                  video_side, video_45, knn_dataset, 
                                  name, output_folder, timing_results):
    """快速版本：處理單組影片的完整流程"""
    try:
        # 匯入處理模組
        from trajector_2D_sync import sync_trajectories
        from trajector_2D_capture_swing_range import find_range
        from trajectory_3D_output import process_trajectories
        from trajector_3D_smoothing import smooth_3D_trajectory
        from trajector_3D_capture_swing_range import extract_frames
        from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
        from trajectory_gpt_single_feedback import generate_feedback_data_only
        
        output_folder = Path(output_folder)
        
        # 步驟1: 2D軌跡提取（快速版本）
        print("\n📹 步驟1: 2D軌跡提取（快速模式）...")
        t1 = time.time()
        
        # 使用並行處理同時處理兩個影片
        if ENABLE_PARALLEL:
            print("   🔄 並行處理兩個視角...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_side = executor.submit(
                    analyze_trajectory_with_output_folder_fast,
                    yolo_pose_model, yolo_tennis_ball_model, video_side, 16, output_folder
                )
                future_45 = executor.submit(
                    analyze_trajectory_with_output_folder_fast,
                    yolo_pose_model, yolo_tennis_ball_model, video_45, 16, output_folder
                )
                
                trajectory_2d_side_path = future_side.result()
                trajectory_2d_45_path = future_45.result()
        else:
            # 序列處理
            trajectory_2d_side_path = analyze_trajectory_with_output_folder_fast(
                yolo_pose_model, yolo_tennis_ball_model, video_side, 16, output_folder
            )
            trajectory_2d_45_path = analyze_trajectory_with_output_folder_fast(
                yolo_pose_model, yolo_tennis_ball_model, video_45, 16, output_folder
            )
        
        timing_results['2d_extraction'] = time.time() - t1
        print(f"   ⏱️  耗時: {timing_results['2d_extraction']:.2f} 秒")
        
        # 步驟2: 2D軌跡平滑（快速版本）
        print("\n🌊 步驟2: 2D軌跡平滑（快速模式）...")
        t2 = time.time()
        
        if ENABLE_PARALLEL:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_side = executor.submit(
                    smooth_2D_trajectory_with_output_folder_fast,
                    trajectory_2d_side_path, output_folder
                )
                future_45 = executor.submit(
                    smooth_2D_trajectory_with_output_folder_fast,
                    trajectory_2d_45_path, output_folder
                )
                
                trajectory_2d_side_smoothed = future_side.result()
                trajectory_2d_45_smoothed = future_45.result()
        else:
            trajectory_2d_side_smoothed = smooth_2D_trajectory_with_output_folder_fast(
                trajectory_2d_side_path, output_folder
            )
            trajectory_2d_45_smoothed = smooth_2D_trajectory_with_output_folder_fast(
                trajectory_2d_45_path, output_folder
            )
        
        timing_results['2d_smoothing'] = time.time() - t2
        print(f"   ⏱️  耗時: {timing_results['2d_smoothing']:.2f} 秒")
        
        # 步驟3: 2D軌跡同步
        print("\n🔄 步驟3: 2D軌跡同步...")
        t3 = time.time()
        
        trajectory_2d_side_synced, trajectory_2d_45_synced = sync_trajectories(
            trajectory_2d_side_smoothed, trajectory_2d_45_smoothed
        )
        
        # 移動同步結果
        for src in [trajectory_2d_side_synced, trajectory_2d_45_synced]:
            if src and Path(src).exists():
                dest = output_folder / Path(src).name
                if Path(src) != dest:
                    shutil.move(src, str(dest))
        
        timing_results['2d_sync'] = time.time() - t3
        print(f"   ⏱️  耗時: {timing_results['2d_sync']:.2f} 秒")
        
        # 步驟4-9: 後續處理（使用原始函數，這些已經夠快）
        print("\n📐 步驟4-9: 3D重建與分析...")
        t4 = time.time()
        
        # 先生成完整的3D軌跡
        trajectory_3d_path = process_trajectories(
            str(output_folder / Path(trajectory_2d_side_synced).name),
            str(output_folder / Path(trajectory_2d_45_synced).name),
            P1, P2
        )
        
        # process_trajectories 已經保存了檔案,移動到正確位置
        if trajectory_3d_path and Path(trajectory_3d_path).exists():
            source = Path(trajectory_3d_path)
            target = output_folder / f"{name}_segment(3D_trajectory).json"
            if source != target:
                shutil.move(str(source), str(target))
                trajectory_3d_path = str(target)
        
        # 3D平滑 (smooth_3D_trajectory 也會自動保存檔案)
        trajectory_3d_smoothed_path = smooth_3D_trajectory(trajectory_3d_path)
        
        # 移動平滑結果到正確位置
        if trajectory_3d_smoothed_path and Path(trajectory_3d_smoothed_path).exists():
            source = Path(trajectory_3d_smoothed_path)
            target = output_folder / f"{name}_segment(3D_trajectory_smoothed).json"
            if source != target:
                shutil.move(str(source), str(target))
                trajectory_3d_smoothed_path = str(target)
        
        # 擊球範圍檢測（使用side視角的平滑數據）
        start_frame, end_frame = find_range(
            str(output_folder / Path(trajectory_2d_side_synced).name)
        )
        
        # 擷取擊球範圍 (extract_frames 會自動保存檔案)
        trajectory_3d_swing_range_path = extract_frames(trajectory_3d_smoothed_path, start_frame, end_frame)
        
        # 移動到正確位置
        if trajectory_3d_swing_range_path and Path(trajectory_3d_swing_range_path).exists():
            source = Path(trajectory_3d_swing_range_path)
            target = output_folder / f"{name}_segment(3D_trajectory_smoothed)_only_swing.json"
            if source != target:
                shutil.move(str(source), str(target))
                trajectory_3d_swing_range_path = str(target)
        
        # KNN 和 GPT 分析（使用原始流程，確保正確性）
        print("\n📊 步驟10-11: KNN 和 GPT 分析...")
        t5 = time.time()
        
        # 確保使用正確的檔案路徑進行 KNN 分析
        print(f"   📁 KNN 輸入檔案: {Path(trajectory_3d_swing_range_path).name}")
        
        # KNN分析
        knn_feedback_path = analyze_trajectory_knn(knn_dataset, trajectory_3d_swing_range_path)
        print(f"   ✅ KNN 分析完成: {Path(knn_feedback_path).name}")
        
        # 讀取並顯示 KNN 結果（用於除錯）
        if Path(knn_feedback_path).exists():
            with open(knn_feedback_path, 'r', encoding='utf-8') as f:
                knn_content = f.read()
            print(f"   📝 KNN 結果預覽: {knn_content[:100]}...")
        
        # 移動 KNN 結果到正確位置
        source = Path(knn_feedback_path)
        target = output_folder / f"{name}_segment_knn_feedback.txt"
        if source != target and source.exists():
            shutil.move(str(source), str(target))
            knn_feedback_path = str(target)
        
        # GPT分析
        print(f"   🤖 GPT 分析中...")
        gpt_result = generate_feedback_data_only(
            trajectory_3d_swing_range_path,
            knn_feedback_path
        )
        save_gpt_feedback_with_output_folder(gpt_result, output_folder, name)
        print(f"   ✅ GPT 分析完成")
        
        timing_results['knn_and_gpt'] = time.time() - t5
        print(f"   ⏱️  耗時: {timing_results['knn_and_gpt']:.2f} 秒")
        
        timing_results['3d_and_analysis'] = time.time() - t4
        print(f"   ⏱️  耗時: {timing_results['3d_and_analysis']:.2f} 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 處理失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_processing_summary(output_folder, name, timing_results, total_time):
    """生成處理摘要檔案"""
    try:
        summary_file = Path(output_folder) / "processing_summary_fast.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"處理摘要 - 快速模式\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"處理對象: {name}\n")
            f.write(f"總耗時: {total_time:.2f} 秒\n\n")
            
            f.write(f"詳細時間:\n")
            for step, duration in timing_results.items():
                f.write(f"  {step}: {duration:.2f} 秒\n")
            
            f.write(f"\n優化設定:\n")
            f.write(f"  跳幀處理: {'啟用' if ENABLE_FRAME_SKIP else '停用'} (率: 1/{FRAME_SKIP_RATE})\n")
            f.write(f"  RAM 緩存: {'啟用' if ENABLE_RAM_CACHE else '停用'}\n")
            f.write(f"  批次處理: {'啟用' if ENABLE_BATCH_PROCESSING else '停用'}\n")
            f.write(f"  並行處理: {'啟用' if ENABLE_PARALLEL else '停用'}\n")
        
        print(f"📄 處理摘要已保存: {summary_file.name}")
    except Exception as e:
        print(f"⚠️ 生成摘要失敗: {e}")
