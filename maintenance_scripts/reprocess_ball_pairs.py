"""
重新配對 TIM82 的球片段
使用最新的球配對邏輯
"""

import json
from pathlib import Path
import sys

# 添加當前目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

# 重新載入模組以確保使用最新版本
import importlib
if 'trajector_processing_unified' in sys.modules:
    importlib.reload(sys.modules['trajector_processing_unified'])

from trajector_processing_unified import align_ball_segments

def reprocess_ball_pairs(user_name):
    """重新處理球配對"""
    
    trajectory_base = Path(f"trajectory/{user_name}__trajectory")
    results_file = trajectory_base / f"{user_name}__segmentation_results.json"
    
    if not results_file.exists():
        print(f"❌ 找不到結果檔案: {results_file}")
        return False
    
    # 讀取現有結果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 {user_name} 原始配對結果:")
    print(f"   側面片段: {len(data['side_segments'])} 個")
    print(f"   45度片段: {len(data['deg45_segments'])} 個")
    print(f"   原始球對: {len(data['ball_pairs'])} 對")
    
    # 重新配對
    print(f"\n🔄 使用最新邏輯重新配對...")
    
    side_ball_data = [
        (seg['entry_time'], seg['exit_time'], seg)
        for seg in data['side_segments']
    ]
    
    deg45_ball_data = [
        (seg['entry_time'], seg['exit_time'], seg)
        for seg in data['deg45_segments']
    ]
    
    # 使用最新的配對函數
    new_ball_pairs = align_ball_segments(side_ball_data, deg45_ball_data, user_name)
    
    # 更新結果
    data['ball_pairs'] = new_ball_pairs
    
    # 保存備份
    backup_file = results_file.with_suffix('.json.backup')
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已備份原始檔案: {backup_file.name}")
    
    # 保存新結果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 配對更新完成!")
    print(f"   新球對數量: {len(new_ball_pairs)} 對")
    print(f"   結果已保存: {results_file.name}")
    
    # 顯示配對詳情
    print(f"\n📋 配對詳情:")
    for pair in new_ball_pairs:
        ball_num = pair['ball_number']
        status = pair['status']
        
        if pair['side_data'] and pair['deg45_data']:
            side_time = pair['side_data']['entry_time']
            deg45_time = pair['deg45_data']['entry_time']
            time_diff = pair['time_difference']
            print(f"   球{ball_num}: 側面{side_time:.2f}s ↔ 45度{deg45_time:.2f}s (差異{time_diff:.2f}s) - {status}")
        elif pair['side_data']:
            side_time = pair['side_data']['entry_time']
            print(f"   球{ball_num}: 僅側面{side_time:.2f}s - {status}")
        else:
            deg45_time = pair['deg45_data']['entry_time']
            print(f"   球{ball_num}: 僅45度{deg45_time:.2f}s - {status}")
    
    return True

if __name__ == "__main__":
    print("🔧 重新配對工具")
    print("=" * 60)
    
    user_name = input("\n請輸入使用者名稱 (例如: TIM82): ").strip()
    
    if not user_name:
        print("❌ 使用者名稱不能為空")
    else:
        success = reprocess_ball_pairs(user_name)
        
        if success:
            print("\n" + "=" * 60)
            print("✨ 重新配對完成！")
            print("💡 提示: 現在可以重新執行處理流程來分析所有球")
        else:
            print("\n" + "=" * 60)
            print("❌ 重新配對失敗")
    
    input("\n按 Enter 結束...")
