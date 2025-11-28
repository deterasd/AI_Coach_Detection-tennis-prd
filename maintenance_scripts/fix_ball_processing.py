"""
一鍵修復工具 - 重新配對並處理所有球
"""

import subprocess
import sys
from pathlib import Path

def fix_ball_pairing_and_process(user_name):
    """
    步驟1: 重新配對球
    步驟2: 重新處理所有球的軌跡分析
    """
    
    print(f"🔧 一鍵修復: {user_name}")
    print("=" * 60)
    
    # 檢查資料夾是否存在
    trajectory_base = Path(f"trajectory/{user_name}__trajectory")
    if not trajectory_base.exists():
        print(f"❌ 找不到資料夾: {trajectory_base}")
        return False
    
    # 步驟1: 重新配對
    print(f"\n步驟1: 重新配對球片段...")
    print("-" * 60)
    
    import json
    import importlib
    
    # 強制重新載入模組
    if 'trajector_processing_unified' in sys.modules:
        importlib.reload(sys.modules['trajector_processing_unified'])
    
    from trajector_processing_unified import align_ball_segments
    
    results_file = trajectory_base / f"{user_name}__segmentation_results.json"
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   原始球對: {len(data['ball_pairs'])} 對")
    
    side_ball_data = [(seg['entry_time'], seg['exit_time'], seg) for seg in data['side_segments']]
    deg45_ball_data = [(seg['entry_time'], seg['exit_time'], seg) for seg in data['deg45_segments']]
    
    new_ball_pairs = align_ball_segments(side_ball_data, deg45_ball_data, user_name)
    
    # 備份並更新
    backup_file = results_file.with_suffix('.json.backup')
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    data['ball_pairs'] = new_ball_pairs
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 更新後球對: {len(new_ball_pairs)} 對")
    print(f"   💾 備份已保存: {backup_file.name}")
    
    # 步驟2: 重新處理軌跡
    print(f"\n步驟2: 重新處理所有球的軌跡分析...")
    print("-" * 60)
    
    # 執行重新處理
    result = subprocess.run(
        [sys.executable, 'reprocess_trajectories.py'],
        input=f"{user_name}\n",
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n✅ 所有步驟完成！")
        return True
    else:
        print(f"\n⚠️ 處理過程中遇到問題")
        return False

if __name__ == "__main__":
    print("🚀 一鍵修復工具")
    print("=" * 60)
    print("此工具會:")
    print("  1. 重新配對球片段 (修復只有1個球對的問題)")
    print("  2. 重新處理所有球的軌跡分析")
    print()
    
    user_name = input("請輸入使用者名稱 (例如: TIM82): ").strip()
    
    if not user_name:
        print("❌ 使用者名稱不能為空")
    else:
        success = fix_ball_pairing_and_process(user_name)
        
        if success:
            print("\n" + "=" * 60)
            print("✨ 修復完成！")
            print(f"📁 查看結果: trajectory/{user_name}__trajectory/")
            print("   - trajectory_1/")
            print("   - trajectory_2/")
            print("   - trajectory_3/")
        else:
            print("\n" + "=" * 60)
            print("❌ 修復失敗，請檢查錯誤訊息")
    
    input("\n按 Enter 結束...")
