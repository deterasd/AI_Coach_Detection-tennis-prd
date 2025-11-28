"""
檢查 TIM82 的球配對狀況
"""

import json
from pathlib import Path

import sys

# 從命令列參數或預設值獲取使用者名稱
user_name = sys.argv[1] if len(sys.argv) > 1 else "TIM82"

results_file = Path(f"trajectory/{user_name}__trajectory/{user_name}__segmentation_results.json")

if not results_file.exists():
    print(f"❌ 檔案不存在: {results_file}")
    exit()

with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 {user_name} 分割結果分析")
print("=" * 60)

print(f"\n側面片段: {len(data['side_segments'])} 個")
for seg in data['side_segments']:
    print(f"  片段 {seg['segment_number']}: 進入={seg['entry_time']:.2f}s, 離開={seg['exit_time']:.2f}s")

print(f"\n45度片段: {len(data['deg45_segments'])} 個")
for seg in data['deg45_segments']:
    print(f"  片段 {seg['segment_number']}: 進入={seg['entry_time']:.2f}s, 離開={seg['exit_time']:.2f}s")

print(f"\n球對數量: {len(data['ball_pairs'])} 對")
for pair in data['ball_pairs']:
    ball_num = pair['ball_number']
    status = pair['status']
    
    side_time = f"{pair['side_data']['entry_time']:.2f}s" if pair['side_data'] else "N/A"
    deg45_time = f"{pair['deg45_data']['entry_time']:.2f}s" if pair['deg45_data'] else "N/A"
    time_diff = f"{pair['time_difference']:.2f}s" if pair['time_difference'] else "N/A"
    
    print(f"  球{ball_num}: 側面={side_time}, 45度={deg45_time}, 差異={time_diff}, 狀態={status}")

# 檢查 trajectory_X 資料夾
print(f"\n📁 檢查球資料夾:")
trajectory_base = Path(f"trajectory/{user_name}__trajectory")
ball_folders = sorted([f for f in trajectory_base.iterdir() if f.is_dir() and f.name.startswith("trajectory_")])

if ball_folders:
    print(f"  找到 {len(ball_folders)} 個球資料夾:")
    for folder in ball_folders:
        print(f"    - {folder.name}")
        
        # 檢查關鍵檔案
        json_files = list(folder.glob("*.json"))
        print(f"      JSON檔案: {len(json_files)} 個")
else:
    print(f"  ❌ 沒有找到球資料夾 (應該有 trajectory_1, trajectory_2, trajectory_3)")

print("\n" + "=" * 60)
