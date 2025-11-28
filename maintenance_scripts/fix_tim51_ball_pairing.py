"""
修復 TIM51 的球對配對問題
重新分析分割結果並創建正確的球對配對
"""

import json
from pathlib import Path

def fix_tim51_ball_pairing():
    """修復 TIM51 的球對配對"""
    
    segmentation_file = Path("trajectory/TIM51__trajectory/TIM51__segmentation_results.json")
    
    if not segmentation_file.exists():
        print("❌ 找不到分割結果檔案")
        return False
    
    print("🔧 載入分割結果...")
    with open(segmentation_file, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    
    # 提取球的時間資料
    side_segments = seg_data['side_segments']
    deg45_segments = seg_data['deg45_segments']
    
    print(f"📊 側面球: {len(side_segments)} 個")
    for i, seg in enumerate(side_segments):
        print(f"   球{i+1}: {seg['entry_time']:.2f}s")
    
    print(f"📊 45度球: {len(deg45_segments)} 個") 
    for i, seg in enumerate(deg45_segments):
        print(f"   球{i+1}: {seg['entry_time']:.2f}s")
    
    # 重新進行球對配對
    print(f"\n🔄 重新進行球對配對...")
    
    ball_pairs = []
    time_tolerance = 2.0
    used_deg45_indices = set()
    
    for side_idx, side_seg in enumerate(side_segments):
        side_entry = side_seg['entry_time']
        best_match_idx = None
        best_time_diff = float('inf')
        
        # 找最接近的45度球
        for deg45_idx, deg45_seg in enumerate(deg45_segments):
            if deg45_idx in used_deg45_indices:
                continue
                
            deg45_entry = deg45_seg['entry_time']
            time_diff = abs(side_entry - deg45_entry)
            
            if time_diff < best_time_diff and time_diff <= time_tolerance:
                best_time_diff = time_diff
                best_match_idx = deg45_idx
        
        # 創建球對
        ball_number = side_idx + 1
        
        if best_match_idx is not None:
            used_deg45_indices.add(best_match_idx)
            deg45_seg = deg45_segments[best_match_idx]
            
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {
                    "entry_time": side_seg['entry_time'],
                    "exit_time": side_seg['exit_time'],
                    "segment": side_seg
                },
                "deg45_data": {
                    "entry_time": deg45_seg['entry_time'],
                    "exit_time": deg45_seg['exit_time'],
                    "segment": deg45_seg
                },
                "time_difference": best_time_diff,
                "status": "paired"
            }
            
            print(f"   ✅ 球{ball_number}: 側面{side_entry:.2f}s ↔ 45度{deg45_seg['entry_time']:.2f}s (差異{best_time_diff:.2f}s)")
        else:
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {
                    "entry_time": side_seg['entry_time'],
                    "exit_time": side_seg['exit_time'],
                    "segment": side_seg
                },
                "deg45_data": None,
                "time_difference": None,
                "status": "unpaired_side_only"
            }
            
            print(f"   ⚠️ 球{ball_number}: 只有側面{side_entry:.2f}s")
        
        ball_pairs.append(ball_pair)
    
    # 處理未配對的45度球
    for deg45_idx, deg45_seg in enumerate(deg45_segments):
        if deg45_idx not in used_deg45_indices:
            ball_number = len(ball_pairs) + 1
            
            ball_pair = {
                "ball_number": ball_number,
                "side_data": None,
                "deg45_data": {
                    "entry_time": deg45_seg['entry_time'],
                    "exit_time": deg45_seg['exit_time'],
                    "segment": deg45_seg
                },
                "time_difference": None,
                "status": "unpaired_deg45_only"
            }
            
            ball_pairs.append(ball_pair)
            print(f"   ⚠️ 球{ball_number}: 只有45度{deg45_seg['entry_time']:.2f}s")
    
    # 更新分割結果
    seg_data['ball_pairs'] = ball_pairs
    
    # 保存修復後的結果
    print(f"\n💾 保存修復後的結果...")
    with open(segmentation_file, 'w', encoding='utf-8') as f:
        json.dump(seg_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 修復完成！球對數量: {len(ball_pairs)} 對")
    
    # 現在需要創建對應的軌跡資料夾
    print(f"\n📁 創建軌跡資料夾...")
    base_trajectory_folder = Path("trajectory/TIM51__trajectory")
    
    for ball_pair in ball_pairs:
        ball_num = ball_pair['ball_number']
        trajectory_folder = base_trajectory_folder / f"trajectory_{ball_num}"
        trajectory_folder.mkdir(exist_ok=True)
        
        # 複製對應的片段檔案
        if ball_pair['side_data']:
            side_segment_path = Path(ball_pair['side_data']['segment']['file_path'])
            if side_segment_path.exists():
                dest_path = trajectory_folder / f"TIM51__{ball_num}_side_segment.mp4"
                import shutil
                shutil.copy2(side_segment_path, dest_path)
                print(f"   ✅ 側面片段: TIM51__{ball_num}_side_segment.mp4 → trajectory_{ball_num}/")
        
        if ball_pair['deg45_data']:
            deg45_segment_path = Path(ball_pair['deg45_data']['segment']['file_path'])
            if deg45_segment_path.exists():
                dest_path = trajectory_folder / f"TIM51__{ball_num}_45_segment.mp4"
                import shutil
                shutil.copy2(deg45_segment_path, dest_path)
                print(f"   ✅ 45度片段: TIM51__{ball_num}_45_segment.mp4 → trajectory_{ball_num}/")
    
    return True

if __name__ == "__main__":
    print("🔧 TIM51 球對配對修復工具")
    print("=" * 50)
    
    success = fix_tim51_ball_pairing()
    
    if success:
        print(f"\n🎉 修復成功！")
        print(f"💡 現在重新運行處理流程，應該能處理所有球了")
    else:
        print(f"\n❌ 修復失敗")
    
    input("\n按 Enter 結束...")