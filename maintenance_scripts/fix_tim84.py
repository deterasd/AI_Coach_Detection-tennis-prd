import sys
import importlib
import os
import shutil

# 強制重新載入模組
if 'trajector_processing' in sys.modules:
    del sys.modules['trajector_processing']
if 'trajector_processing_unified' in sys.modules:
    del sys.modules['trajector_processing_unified']

import trajector_processing_unified

print("=" * 80)
print("🔧 tim84 球對修復工具")
print("=" * 80)

username = "tim84"
base_path = r"C:\Users\user\Documents\AI_Coach_Detection-prd"
trajectory_base = os.path.join(base_path, "trajectory")
user_trajectory_path = os.path.join(trajectory_base, f"{username}__trajectory")

# 檢查是否存在
if not os.path.exists(user_trajectory_path):
    print(f"❌ 找不到 {username} 的軌跡資料夾")
    sys.exit(1)

# 讀取原始球數據
side_segments_path = os.path.join(user_trajectory_path, "side_ball_segments.json")
deg45_segments_path = os.path.join(user_trajectory_path, "deg45_ball_segments.json")

if not os.path.exists(side_segments_path) or not os.path.exists(deg45_segments_path):
    print(f"❌ 找不到原始球段數據")
    sys.exit(1)

import json

with open(side_segments_path, 'r', encoding='utf-8') as f:
    side_segments_data = json.load(f)
    
with open(deg45_segments_path, 'r', encoding='utf-8') as f:
    deg45_segments_data = json.load(f)

print(f"\n📊 原始數據:")
print(f"   側面球段: {len(side_segments_data)} 個")
print(f"   45度球段: {len(deg45_segments_data)} 個")

# 轉換為 align_ball_segments 需要的格式
side_ball_data = [(seg["entry_time"], seg["exit_time"], seg["segment_path"]) 
                  for seg in side_segments_data]
deg45_ball_data = [(seg["entry_time"], seg["exit_time"], seg["segment_path"]) 
                   for seg in deg45_segments_data]

# 使用修復後的函數重新配對
print(f"\n🔄 重新配對球...")
ball_pairs = trajector_processing_unified.align_ball_segments(side_ball_data, deg45_ball_data)

print(f"\n📊 新的球對數量: {len(ball_pairs)}")

# 保存新的球對數據
ball_pairs_path = os.path.join(user_trajectory_path, "ball_pairs.json")
with open(ball_pairs_path, 'w', encoding='utf-8') as f:
    json.dump(ball_pairs, f, indent=2, ensure_ascii=False)

print(f"✅ 球對數據已更新: {ball_pairs_path}")

# 刪除舊的 trajectory_n 資料夾
existing_trajectory_folders = [d for d in os.listdir(user_trajectory_path) 
                               if d.startswith("trajectory_") and os.path.isdir(os.path.join(user_trajectory_path, d))]

if existing_trajectory_folders:
    print(f"\n🗑️ 刪除 {len(existing_trajectory_folders)} 個舊的軌跡資料夾...")
    for folder in existing_trajectory_folders:
        folder_path = os.path.join(user_trajectory_path, folder)
        shutil.rmtree(folder_path)
        print(f"   已刪除: {folder}")

print("\n" + "=" * 80)
print("✅ tim84 球對修復完成！")
print("=" * 80)
print(f"\n現在請執行: python reprocess_trajectories.py tim84")
print("來重新處理所有球的軌跡分析")
