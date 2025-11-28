"""
重新處理已分割的影片 - 基於更新後的球對配置
只執行軌跡分析部分，不重新分割影片
"""

import time
import numpy as np
import os
import json
from pathlib import Path
from ultralytics import YOLO

def reprocess_existing_segments(user_name, user_height=175):
    """
    重新處理已存在的影片片段
    只執行 2D/3D 軌跡分析、KNN 和 GPT，不重新分割影片
    """
    
    print(f"🔄 重新處理 {user_name} 的軌跡分析")
    print("=" * 60)
    
    # 設定路徑
    trajectory_base = Path(f"trajectory/{user_name}__trajectory")
    results_file = trajectory_base / f"{user_name}__segmentation_results.json"
    
    if not results_file.exists():
        print(f"❌ 找不到分割結果: {results_file}")
        return False
    
    # 讀取分割結果
    with open(results_file, 'r', encoding='utf-8') as f:
        segmentation_results = json.load(f)
    
    ball_pairs = segmentation_results.get('ball_pairs', [])
    
    if not ball_pairs:
        print(f"❌ 沒有找到球對資料")
        return False
    
    print(f"📊 找到 {len(ball_pairs)} 個球對")
    
    # 載入模型
    print(f"\n📦 載入 YOLO 模型...")
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    
    # GPU 檢查
    import torch
    if torch.cuda.is_available():
        print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
        try:
            yolo_pose_model.model.to('cuda')
            yolo_tennis_ball_model.model.to('cuda')
            print("✅ GPU 加速已啟用")
        except:
            print("⚠️ GPU 設置失敗，使用 CPU")
            yolo_pose_model.model.to('cpu')
            yolo_tennis_ball_model.model.to('cpu')
    else:
        print("💻 使用 CPU 模式")
        yolo_pose_model.model.to('cpu')
        yolo_tennis_ball_model.model.to('cpu')
    
    # 投影矩陣
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
    
    # KNN 資料集
    knn_dataset = 'knn_dataset.json'
    
    # 匯入處理函數
    from trajector_processing_unified import process_multiple_balls
    
    timing_results = {}
    
    # 取得原始影片路徑（從user_info.json或猜測）
    user_info_file = trajectory_base / "user_info.json"
    if user_info_file.exists():
        with open(user_info_file, 'r', encoding='utf-8') as f:
            user_info = json.load(f)
            user_height = user_info.get('height', user_height)
    
    # 原始影片路徑（猜測）
    video_side = str(trajectory_base / f"{user_name}__1_side.mp4")
    video_45 = str(trajectory_base / f"{user_name}__1_45.mp4")
    
    # 檢查原始影片是否存在
    if not Path(video_side).exists():
        print(f"⚠️ 找不到原始側面影片: {video_side}")
        print(f"   將只使用分割片段進行處理")
    
    if not Path(video_45).exists():
        print(f"⚠️ 找不到原始45度影片: {video_45}")
        print(f"   將只使用分割片段進行處理")
    
    # 執行多球處理
    print(f"\n🚀 開始處理 {len(ball_pairs)} 個球對...")
    
    success = process_multiple_balls(
        P1, P2, yolo_pose_model, yolo_tennis_ball_model,
        video_side, video_45, knn_dataset,
        user_name, trajectory_base, timing_results, segmentation_results
    )
    
    if success:
        print(f"\n✅ 所有球對處理完成！")
        print(f"📁 結果保存在: {trajectory_base}")
        
        # 列出生成的球資料夾
        ball_folders = sorted([f for f in trajectory_base.iterdir() if f.is_dir() and f.name.startswith("trajectory_")])
        print(f"\n📂 生成的球資料夾:")
        for folder in ball_folders:
            json_count = len(list(folder.glob("*.json")))
            print(f"   - {folder.name}: {json_count} 個 JSON 檔案")
        
        return True
    else:
        print(f"\n⚠️ 處理過程中遇到問題")
        return False

if __name__ == "__main__":
    print("🔄 重新處理工具 - 基於更新的球對配置")
    print("=" * 60)
    
    user_name = input("\n請輸入使用者名稱 (例如: TIM82): ").strip()
    
    if not user_name:
        print("❌ 使用者名稱不能為空")
    else:
        print(f"\n⏳ 開始處理 {user_name}...")
        success = reprocess_existing_segments(user_name)
        
        if success:
            print("\n" + "=" * 60)
            print("✨ 重新處理完成！")
            print("🎾 現在所有球都已經分析完成")
        else:
            print("\n" + "=" * 60)
            print("❌ 重新處理失敗")
    
    input("\n按 Enter 結束...")
