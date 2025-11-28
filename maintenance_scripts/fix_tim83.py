"""
快速修復腳本 - 重新配對並處理 tim83
"""

import json
import sys
from pathlib import Path

# 強制重新載入模組
import importlib
if 'trajector_processing_unified' in sys.modules:
    importlib.reload(sys.modules['trajector_processing_unified'])

from trajector_processing_unified import align_ball_segments

user_name = "tim83"
trajectory_base = Path(f"trajectory/{user_name}__trajectory")
results_file = trajectory_base / f"{user_name}__segmentation_results.json"

print(f"🔧 修復 {user_name} 的球配對")
print("=" * 60)

# 讀取現有結果
with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 原始狀況:")
print(f"   側面片段: {len(data['side_segments'])} 個")
print(f"   45度片段: {len(data['deg45_segments'])} 個")
print(f"   球對數量: {len(data['ball_pairs'])} 對")

# 重新配對
print(f"\n🔄 重新配對...")

side_ball_data = [(seg['entry_time'], seg['exit_time'], seg) for seg in data['side_segments']]
deg45_ball_data = [(seg['entry_time'], seg['exit_time'], seg) for seg in data['deg45_segments']]

new_ball_pairs = align_ball_segments(side_ball_data, deg45_ball_data, user_name)

# 備份
backup_file = results_file.with_suffix('.json.backup')
with open(backup_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n💾 已備份: {backup_file.name}")

# 更新
data['ball_pairs'] = new_ball_pairs

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 配對更新完成!")
print(f"   新球對數量: {len(new_ball_pairs)} 對")

print(f"\n📋 配對詳情:")
for pair in new_ball_pairs:
    ball_num = pair['ball_number']
    if pair['side_data'] and pair['deg45_data']:
        side_time = pair['side_data']['entry_time']
        deg45_time = pair['deg45_data']['entry_time']
        time_diff = pair['time_difference']
        print(f"   球{ball_num}: 側面{side_time:.2f}s ↔ 45度{deg45_time:.2f}s (差異{time_diff:.2f}s)")

print(f"\n" + "=" * 60)
print("✨ 球對配對已修復！")
print(f"📁 結果已保存: {results_file}")
print("\n💡 下一步:")
print(f"   執行: python reprocess_trajectories.py")
print(f"   輸入: {user_name}")
print("   這會重新處理所有球的軌跡分析")
