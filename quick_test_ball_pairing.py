"""
快速測試球對配對功能 - 驗證 align_ball_segments 是否正常運作
"""
import sys
import os

# 強制重新載入模組
if 'trajector_processing_unified' in sys.modules:
    del sys.modules['trajector_processing_unified']

import trajector_processing_unified

print("=" * 80)
print("🧪 快速測試：球對配對功能")
print("=" * 80)

# 模擬 tim84 的實際數據（3個側面球 + 3個45度球）
side_ball_data = [
    (0.23, 1.93, "side_segment_1.mp4"),
    (2.03, 3.73, "side_segment_2.mp4"),
    (3.83, 5.40, "side_segment_3.mp4"),
]

deg45_ball_data = [
    (0.70, 2.70, "deg45_segment_1.mp4"),
    (2.80, 4.80, "deg45_segment_2.mp4"),
    (4.90, 6.30, "deg45_segment_3.mp4"),
]

print(f"\n📊 測試數據:")
print(f"   側面球: {len(side_ball_data)} 個")
print(f"   45度球: {len(deg45_ball_data)} 個")
print(f"\n🔄 執行球對配對...")

# 呼叫函數
ball_pairs = trajector_processing_unified.align_ball_segments(
    side_ball_data, 
    deg45_ball_data, 
    "test_user"
)

print(f"\n📊 配對結果:")
print(f"   球對數量: {len(ball_pairs)} 對")

if len(ball_pairs) == 3:
    print(f"\n✅ 測試通過！成功配對 3 個球")
    for i, pair in enumerate(ball_pairs, 1):
        status = pair['status']
        if pair.get('side_data') and pair.get('deg45_data'):
            side_time = pair['side_data']['entry_time']
            deg45_time = pair['deg45_data']['entry_time']
            time_diff = pair['time_difference']
            print(f"   球{i}: 側面={side_time:.2f}s, 45度={deg45_time:.2f}s, 差異={time_diff:.2f}s ({status})")
else:
    print(f"\n❌ 測試失敗！預期 3 個球對，實際得到 {len(ball_pairs)} 個")
    for i, pair in enumerate(ball_pairs, 1):
        print(f"   球{i}: {pair}")

print("\n" + "=" * 80)
