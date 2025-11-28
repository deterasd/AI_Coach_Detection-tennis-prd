"""
診斷腳本 - 檢查為什麼只處理第一顆球
"""

import json
from pathlib import Path

def check_trajectory_folders():
    """檢查 trajectory 資料夾中的球處理情況"""
    
    trajectory_base = Path("trajectory")
    
    if not trajectory_base.exists():
        print("❌ trajectory 資料夾不存在")
        return
    
    # 找到所有使用者資料夾
    user_folders = [f for f in trajectory_base.iterdir() if f.is_dir() and "__trajectory" in f.name]
    
    for user_folder in user_folders:
        print(f"\n📁 檢查使用者資料夾: {user_folder.name}")
        print("=" * 60)
        
        # 檢查是否有球資料夾
        ball_folders = sorted([f for f in user_folder.iterdir() if f.is_dir() and f.name.startswith("trajectory_")])
        
        if not ball_folders:
            print("   ⚠️ 沒有找到球資料夾 (trajectory_1, trajectory_2, ...)")
            
            # 檢查是否有分割片段檔案
            segment_files = list(user_folder.glob("*_segment.mp4"))
            if segment_files:
                print(f"   📹 找到 {len(segment_files)} 個分割片段檔案:")
                for seg in sorted(segment_files):
                    print(f"      - {seg.name}")
        else:
            print(f"   ✅ 找到 {len(ball_folders)} 個球資料夾:")
            
            for ball_folder in ball_folders:
                ball_number = ball_folder.name.split("_")[-1]
                print(f"\n   🎾 球 {ball_number} ({ball_folder.name}):")
                
                # 檢查該球資料夾中的檔案
                files = list(ball_folder.glob("*"))
                
                # 分類檔案
                json_files = [f for f in files if f.suffix == '.json']
                mp4_files = [f for f in files if f.suffix == '.mp4']
                txt_files = [f for f in files if f.suffix == '.txt']
                
                print(f"      JSON 檔案: {len(json_files)} 個")
                for jf in sorted(json_files):
                    print(f"         - {jf.name}")
                
                print(f"      影片檔案: {len(mp4_files)} 個")
                for vf in sorted(mp4_files):
                    print(f"         - {vf.name}")
                
                print(f"      文字檔案: {len(txt_files)} 個")
                for tf in sorted(txt_files):
                    print(f"         - {tf.name}")
                
                # 檢查關鍵檔案是否存在
                expected_files = {
                    "2D軌跡": f"*__*_side(2D_trajectory_smoothed).json",
                    "3D軌跡": f"*__*_segment(3D_trajectory_smoothed).json",
                    "KNN分析": f"*__*_segment_trajectory_knn_suggestion.txt",
                    "GPT反饋": f"*__*_segment_gpt_feedback.json"
                }
                
                print(f"\n      關鍵檔案檢查:")
                for desc, pattern in expected_files.items():
                    matches = list(ball_folder.glob(pattern))
                    if matches:
                        print(f"         ✅ {desc}: {matches[0].name}")
                    else:
                        print(f"         ❌ {desc}: 缺少 (模式: {pattern})")

def check_segmentation_log():
    """檢查最近的分割日誌"""
    
    # 查看是否有分割結果的 JSON
    trajectory_base = Path("trajectory")
    
    for user_folder in trajectory_base.glob("*__trajectory"):
        print(f"\n📋 檢查 {user_folder.name} 的分割資訊:")
        
        # 尋找可能的分割資訊檔案
        info_files = list(user_folder.glob("*info*.json")) + list(user_folder.glob("*segment*.json"))
        
        if info_files:
            for info_file in info_files:
                print(f"   📄 {info_file.name}")
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"      內容: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
                except Exception as e:
                    print(f"      ⚠️ 無法讀取: {e}")

def check_video_segments():
    """檢查影片分割片段"""
    
    trajectory_base = Path("trajectory")
    
    for user_folder in trajectory_base.glob("*__trajectory"):
        print(f"\n🎬 檢查 {user_folder.name} 的影片片段:")
        
        # 側面片段
        side_segments = sorted(user_folder.glob("*_side_segment.mp4"))
        print(f"   側面片段: {len(side_segments)} 個")
        for seg in side_segments:
            print(f"      - {seg.name}")
        
        # 45度片段
        deg45_segments = sorted(user_folder.glob("*_45_segment.mp4"))
        print(f"   45度片段: {len(deg45_segments)} 個")
        for seg in deg45_segments:
            print(f"      - {seg.name}")

if __name__ == "__main__":
    print("🔍 開始診斷球處理狀況...")
    print("=" * 60)
    
    check_trajectory_folders()
    check_segmentation_log()
    check_video_segments()
    
    print("\n" + "=" * 60)
    print("✅ 診斷完成")
    input("\n按 Enter 結束...")
