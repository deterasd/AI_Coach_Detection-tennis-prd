"""
簡易測試版本 - 快速實驗版
使用 trajector_processing_unified_fast 進行加速處理
"""

import time
import numpy as np
import os
import subprocess
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime

# 強制重新載入模組以確保使用最新代碼
if 'trajector_processing_unified_fast' in sys.modules:
    del sys.modules['trajector_processing_unified_fast']

# 匯入快速處理函數
from trajector_processing_unified_fast import (
    processing_trajectory_unified_fast,
    ENABLE_FRAME_SKIP,
    FRAME_SKIP_RATE,
    ENABLE_RAM_CACHE,
    ENABLE_BATCH_PROCESSING,
    ENABLE_PARALLEL
)

# === 輔助函數定義 (替代遺失的 trajector_processing_simple_test) ===

def get_user_info():
    """獲取使用者資訊"""
    print("請輸入使用者資訊:")
    name = input("姓名 (預設: test_user): ").strip() or "test_user"
    height_str = input("身高 (cm) (預設: 175): ").strip() or "175"
    try:
        height = float(height_str)
    except ValueError:
        height = 175.0
        print(f"身高輸入無效，使用預設值: {height}")
    return name, height

def check_and_install_ffmpeg():
    """檢查 FFmpeg 是否安裝"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ FFmpeg 已安裝")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 未檢測到 FFmpeg，請安裝 FFmpeg 以啟用影片分割功能")
        return False

def create_trajectory_folders(name, height):
    """建立輸出資料夾結構"""
    base_folder = Path("trajectory")
    user_folder_name = f"{name}__trajectory"
    user_folder = base_folder / user_folder_name
    
    folders = {
        "root": user_folder,
        "logs": user_folder / "logs",
        "2d_output": user_folder / "2d_output",
        "3d_output": user_folder / "3d_output",
        "segments": user_folder / "segments"
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        
    return user_folder, folders

def check_input_videos(input_folder):
    """檢查輸入影片"""
    input_path = Path(input_folder)
    if not input_path.exists():
        input_path.mkdir(parents=True)
        print(f"已建立輸入資料夾: {input_path}")
        return None, None
        
    videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.MOV")) + list(input_path.glob("*.avi"))
    
    side_video = None
    deg45_video = None
    
    # 簡單的啟發式規則：檔名包含 'side' 或 '45'
    for video in videos:
        if 'side' in video.name.lower() and not side_video:
            side_video = video
        elif '45' in video.name.lower() and not deg45_video:
            deg45_video = video
            
    # 如果找不到，就取前兩個
    if not side_video and len(videos) >= 1:
        side_video = videos[0]
    if not deg45_video and len(videos) >= 2:
        deg45_video = videos[1]
        
    if side_video: print(f"   找到側面影片: {side_video.name}")
    if deg45_video: print(f"   找到45度影片: {deg45_video.name}")
    
    return side_video, deg45_video

def copy_input_videos(side_video, deg45_video, user_folder, name):
    """複製影片到輸出資料夾"""
    if not side_video or not deg45_video:
        return None, None
        
    side_ext = side_video.suffix
    deg45_ext = deg45_video.suffix
    
    new_side_path = user_folder / f"{name}_side{side_ext}"
    new_deg45_path = user_folder / f"{name}_45{deg45_ext}"
    
    shutil.copy2(side_video, new_side_path)
    shutil.copy2(deg45_video, new_deg45_path)
    
    print(f"   已複製: {new_side_path.name}")
    print(f"   已複製: {new_deg45_path.name}")
    
    return new_side_path, new_deg45_path

def create_readme_file(user_folder, name, height, side_video, deg45_video, ball_direction):
    """建立說明檔案"""
    readme_path = user_folder / "README.md"
    content = f"""# {name} 的網球分析報告

## 基本資訊
- **姓名**: {name}
- **身高**: {height} cm
- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **球進入方向**: {ball_direction}

## 輸入影片
- **側面視角**: {side_video.name if side_video else 'N/A'}
- **45度視角**: {deg45_video.name if deg45_video else 'N/A'}

## 輸出檔案說明
- `*_trajectory.json`: 3D 軌跡數據
- `*_feedback.txt`: AI 教練分析報告
- `segments/`: 分割後的擊球片段
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   已建立說明檔: {readme_path.name}")

CONFIG_FILE = "config.json"

def load_config():
    """載入設定"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config):
    """儲存設定"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def interactive_setup():
    """互動式設定"""
    print("\n⚙️ 設定分析參數:")
    
    # 球的方向
    direction = input("   球進入方向 (left/right) [預設: right]: ").strip().lower()
    if direction not in ['left', 'right']:
        direction = 'right'
        
    # 信心度閾值
    try:
        conf = float(input("   偵測信心度 (0.1-0.9) [預設: 0.5]: ").strip() or 0.5)
    except ValueError:
        conf = 0.5
        
    return direction, conf

# === 輔助函數定義結束 ===

def simple_test_pipeline_fast(input_folder="input_videos", 
                              ball_direction="right", 
                              confidence_threshold=0.5):
    """快速版本的測試流程"""
    
    print("🚀 AI網球教練 - 快速實驗版本")
    print("=" * 60)
    print("⚡ 優化功能:")
    print(f"   📥 RAM 緩存: {'啟用' if ENABLE_RAM_CACHE else '停用'}")
    print(f"   ⏩ 跳幀處理: {'啟用' if ENABLE_FRAME_SKIP else '停用'} (率: 1/{FRAME_SKIP_RATE if ENABLE_FRAME_SKIP else 1})")
    print(f"   📦 批次處理: {'啟用' if ENABLE_BATCH_PROCESSING else '停用'}")
    print(f"   🔄 並行處理: {'啟用' if ENABLE_PARALLEL else '停用'}")
    print("=" * 60)
    
    # 步驟1: 獲取使用者資訊
    print("\n👤 步驟1: 輸入使用者資訊...")
    name, height = get_user_info()
    
    # 步驟2: 檢查輸入影片
    print("\n📹 步驟2: 檢查輸入影片...")
    side_video, deg45_video = check_input_videos(input_folder)
    
    if not side_video or not deg45_video:
        print("\n❌ 測試終止：請先準備影片檔案")
        print(f"📁 將影片放入: {Path(input_folder).absolute()}")
        print("🔄 然後重新執行此程式")
        input("\n按 Enter 結束...")
        return False
    
    # 步驟3: 創建使用者資料夾
    print(f"\n📁 步驟3: 創建 {name} 的軌跡資料夾...")
    user_folder, folders = create_trajectory_folders(name, height)
    print(f"📂 使用者資料夾: {user_folder}")
    
    # 步驟4: 複製並重命名輸入影片
    print("\n📋 步驟4: 複製輸入影片到使用者資料夾...")
    side_video_copy, deg45_video_copy = copy_input_videos(
        side_video, deg45_video, user_folder, name
    )
    
    # 步驟5: 創建說明檔案
    create_readme_file(user_folder, name, height, side_video, deg45_video, ball_direction)
    
    # 步驟6: 檢查 FFmpeg
    print(f"\n🎬 步驟6: 檢查 FFmpeg（影片分割功能）...")
    check_and_install_ffmpeg()
    
    # 步驟7: 設定參數
    print(f"\n⚙️ 步驟7: 設定分析參數...")
    print(f"   使用者: {name} ({height} cm)")
    print(f"   球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}")
    print(f"   偵測信心度: {confidence_threshold}")
    print(f"   處理模式: 🚀 快速模式")
    print(f"   影片分割功能: 啟用（優化分割速度）")
    print(f"   輸出資料夾: trajectory/{name}__trajectory")
    
    try:
        # 步驟8: 載入AI模型
        print("\n🤖 步驟8: 載入AI模型...")
        
        # 投影矩陣設定
        P1 = np.array([
            [ 6589.640314,     0.000000,  2376.082461,     0.000000],
            [    0.000000,  5231.039306,  1083.022806,     0.000000],
            [    0.000000,     0.000000,     1.000000,     0.000000],
        ])

        P2 = np.array([
            [-1053.662060,   513.154860,  4035.584006, -19519022.763631],
            [-1201.547422,  3282.802251,   111.083333, 6107286.747928],
            [   -0.936230,     0.075284,     0.343229,  4032.714675],
        ])
        
        # 載入YOLO模型
        print("📦 載入 YOLO 模型...")
        yolo_pose_model = YOLO('model/yolov8n-pose.pt')
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        
        # GPU加速
        print("🔍 檢查 GPU 可用性...")
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   總記憶體: {total_memory:.2f} GB")
                
                torch.cuda.empty_cache()
                
                if total_memory >= 4.0:
                    try:
                        yolo_pose_model.model.to('cuda')
                        yolo_tennis_ball_model.model.to('cuda')
                        print("⚡ GPU 加速已啟用")
                    except Exception as gpu_error:
                        print(f"⚠️ GPU 設置失敗: {gpu_error}")
                        print("💻 回退到 CPU 模式")
                        yolo_pose_model.model.to('cpu')
                        yolo_tennis_ball_model.model.to('cpu')
                else:
                    print("⚠️ GPU 記憶體不足 (需要至少 4GB)")
                    print("💻 使用 CPU 模式")
                    yolo_pose_model.model.to('cpu')
                    yolo_tennis_ball_model.model.to('cpu')
            else:
                print("💻 GPU 不可用，使用 CPU 模式")
                yolo_pose_model.model.to('cpu')
                yolo_tennis_ball_model.model.to('cpu')
        except Exception as e:
            print(f"⚠️ GPU 檢查失敗: {e}")
            print("💻 使用 CPU 模式")
            yolo_pose_model.model.to('cpu')
            yolo_tennis_ball_model.model.to('cpu')
        
        # KNN資料集
        knn_dataset = 'knn_dataset.json'
        
        # 步驟9: 執行快速分析流程
        print("\n🚀 步驟9: 開始快速分析流程...")
        print("⏳ 快速模式：使用 RAM 緩存 + 跳幀處理 + 並行加速")
        print(f"📁 所有結果將保存到: {user_folder}")
        
        # 記錄開始時間
        start_time = time.time()
        
        # 使用快速處理函數
        success = processing_trajectory_unified_fast(
            P1=P1, 
            P2=P2, 
            yolo_pose_model=yolo_pose_model, 
            yolo_tennis_ball_model=yolo_tennis_ball_model,
            video_side=side_video_copy, 
            video_45=deg45_video_copy, 
            knn_dataset=knn_dataset,
            name=name,
            ball_entry_direction=ball_direction,
            confidence_threshold=confidence_threshold,
            output_folder=str(user_folder),
            segment_videos=True  # 啟用影片分割
        )
        
        # 計算總耗時
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 快速分析流程完成！")
            print(f"⏱️  總耗時: {total_time:.2f} 秒")
            print(f"📂 所有結果已保存到: {user_folder}")
            print("\n📋 生成的檔案:")
            
            # 檢查使用者資料夾中的檔案
            if user_folder.exists():
                all_files = list(user_folder.glob("*"))
                video_files = [f for f in all_files if f.suffix.lower() in ['.mp4', '.avi', '.mov']]
                json_files = [f for f in all_files if f.suffix.lower() == '.json']
                other_files = [f for f in all_files if f not in video_files + json_files and f.is_file()]
                
                print(f"   📹 影片檔案: {len(video_files)} 個")
                for video in video_files[:5]:  # 只顯示前5個
                    print(f"      - {video.name}")
                if len(video_files) > 5:
                    print(f"      ... 及其他 {len(video_files)-5} 個")
                
                print(f"   📊 軌跡/分析檔案: {len(json_files)} 個")
                for json_file in json_files[:5]:  # 只顯示前5個
                    print(f"      - {json_file.name}")
                if len(json_files) > 5:
                    print(f"      ... 及其他 {len(json_files)-5} 個")
                
                if other_files:
                    print(f"   📄 其他檔案: {len(other_files)} 個")
            
            print(f"\n📄 詳細說明請查看: {user_folder / 'README.md'}")
            print(f"📈 處理摘要請查看: {user_folder / 'processing_summary_fast.txt'}")
            print(f"🌐 可以在 drawing_3D_three_js.html 中載入 {name} 的分析結果")
            
            # 顯示加速效果預估
            print(f"\n⚡ 加速效果:")
            estimated_normal_time = total_time * (2.0 if ENABLE_FRAME_SKIP else 1.0)
            if ENABLE_RAM_CACHE:
                estimated_normal_time *= 1.2
            if ENABLE_PARALLEL:
                estimated_normal_time *= 1.5
            
            speedup = estimated_normal_time / total_time
            time_saved = estimated_normal_time - total_time
            
            print(f"   預估正常模式耗時: {estimated_normal_time:.2f} 秒")
            print(f"   實際快速模式耗時: {total_time:.2f} 秒")
            print(f"   加速倍率: {speedup:.2f}x")
            print(f"   節省時間: {time_saved:.2f} 秒")
            
            return True
            
        else:
            print("\n❌ 快速分析流程失敗")
            return False
            
    except Exception as e:
        print(f"\n💥 執行過程發生錯誤: {e}")
        print("📝 錯誤詳情已記錄在logs資料夾中")
        
        # 記錄錯誤到日誌
        error_log = folders["logs"] / "error_fast.log"
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"錯誤時間: {datetime.now()}\n")
            f.write(f"使用者: {name} ({height} cm)\n")
            f.write(f"錯誤訊息: {str(e)}\n")
            f.write(f"輸入影片: {side_video}, {deg45_video}\n")
            
            import traceback
            f.write(f"\n詳細錯誤:\n{traceback.format_exc()}\n")
        
        return False

if __name__ == "__main__":
    print("🚀 AI網球教練 - 快速實驗版本啟動")
    print("=" * 50)
    
    # 互動式設定
    ball_direction, confidence_threshold = interactive_setup()
    
    print("\n🚀 開始快速測試流程...")
    print("💡 提示：快速模式會犧牲少量準確度換取更快的處理速度")
    input("按 Enter 繼續...")
    
    # 執行快速測試
    success = simple_test_pipeline_fast(
        input_folder="input_videos",
        ball_direction=ball_direction,
        confidence_threshold=confidence_threshold
    )
    
    if success:
        print("\n✨ 恭喜！快速流程測試成功！")
        print("📊 現在可以在 trajectory/ 資料夾中查看結果")
        print("🔍 請比較快速模式和正常模式的結果差異")
        print("🌐 可以在 drawing_3D_three_js.html 中載入並查看3D軌跡")
    else:
        print("\n😔 處理過程中遇到問題")
        print("🔧 請檢查:")
        print("   1. 影片檔案格式是否正確")
        print("   2. 模型檔案是否存在")
        print("   3. 記憶體是否足夠（快速模式需要更多 RAM）")
    
    print(f"\n📁 結果資料夾位置: trajectory/(姓名)__trajectory/")
    input("\n按 Enter 結束程式...")
