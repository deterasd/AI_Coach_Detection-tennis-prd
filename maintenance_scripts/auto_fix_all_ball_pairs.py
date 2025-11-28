"""
自動修復所有使用者的球對配對
掃描所有 trajectory 資料夾，找出球對數量 < 片段數量的使用者並自動修復
"""
import sys
import json
import os
import shutil
from pathlib import Path

# 強制重新載入模組
if 'trajector_processing_unified' in sys.modules:
    del sys.modules['trajector_processing_unified']

import trajector_processing_unified

trajectory_base = Path(r"C:\Users\user\Documents\AI_Coach_Detection-prd\trajectory")

print("=" * 80)
print("🔧 自動修復工具 - 掃描並修復所有球對配對問題")
print("=" * 80)

# 掃描所有使用者資料夾
user_folders = [f for f in trajectory_base.iterdir() if f.is_dir() and f.name.endswith("__trajectory")]

fixed_users = []
skipped_users = []
error_users = []

for user_folder in user_folders:
    username = user_folder.name.replace("__trajectory", "")
    segmentation_file = user_folder / f"{username}__segmentation_results.json"
    
    if not segmentation_file.exists():
        print(f"\n⏭️ {username}: 跳過（無分段結果檔案）")
        skipped_users.append(username)
        continue
    
    try:
        # 讀取分段結果
        with open(segmentation_file, 'r', encoding='utf-8') as f:
            seg_data = json.load(f)
        
        side_count = len(seg_data.get('side_segments', []))
        deg45_count = len(seg_data.get('deg45_segments', []))
        ball_pairs_count = len(seg_data.get('ball_pairs', []))
        
        # 判斷是否需要修復
        expected_pairs = max(side_count, deg45_count)
        
        if ball_pairs_count >= expected_pairs:
            print(f"\n✅ {username}: 正常（{ball_pairs_count}/{expected_pairs} 球對）")
            continue
        
        print(f"\n🔧 {username}: 需要修復")
        print(f"   側面: {side_count} 個, 45度: {deg45_count} 個")
        print(f"   目前球對: {ball_pairs_count}, 應有: {expected_pairs}")
        
        # 轉換格式
        side_ball_data = [(seg["entry_time"], seg["exit_time"], seg["file_path"]) 
                          for seg in seg_data['side_segments']]
        deg45_ball_data = [(seg["entry_time"], seg["exit_time"], seg["file_path"]) 
                           for seg in seg_data['deg45_segments']]
        
        # 重新配對
        print(f"   🔄 重新配對中...")
        new_ball_pairs = trajector_processing_unified.align_ball_segments(
            side_ball_data, deg45_ball_data, username
        )
        
        # 更新數據
        seg_data['ball_pairs'] = new_ball_pairs
        
        # 保存
        with open(segmentation_file, 'w', encoding='utf-8') as f:
            json.dump(seg_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 已更新: {ball_pairs_count} → {len(new_ball_pairs)} 球對")
        
        # 刪除多餘的舊 trajectory 資料夾
        for i in range(1, ball_pairs_count + 1):
            traj_folder = user_folder / f"trajectory_{i}"
            if traj_folder.exists():
                shutil.rmtree(traj_folder)
                print(f"   🗑️ 已刪除舊資料夾: trajectory_{i}")
        
        fixed_users.append(username)
        
    except Exception as e:
        print(f"\n❌ {username}: 修復失敗 - {str(e)}")
        error_users.append(username)
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("📊 修復完成摘要")
print("=" * 80)
print(f"✅ 已修復: {len(fixed_users)} 個使用者")
if fixed_users:
    for user in fixed_users:
        print(f"   - {user}")

print(f"\n⏭️ 跳過: {len(skipped_users)} 個使用者")
if skipped_users:
    for user in skipped_users:
        print(f"   - {user}")

if error_users:
    print(f"\n❌ 錯誤: {len(error_users)} 個使用者")
    for user in error_users:
        print(f"   - {user}")

if fixed_users:
    print("\n" + "=" * 80)
    print("📝 後續步驟:")
    print("=" * 80)
    print("對於已修復的使用者，請執行重新處理來生成所有軌跡資料夾：")
    for user in fixed_users:
        print(f"   python reprocess_trajectories.py {user}")

print("\n✅ 所有掃描完成！")
