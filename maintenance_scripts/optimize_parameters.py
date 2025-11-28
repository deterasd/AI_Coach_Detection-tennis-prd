#!/usr/bin/env python3
"""
智能參數優化工具
根據檢測結果自動建議和修正分割參數
"""

import json
import sys
from pathlib import Path
import numpy as np
from datetime import timedelta

def analyze_and_suggest_parameters(analysis_file, target_segments=None):
    """
    分析檢測結果並建議優化參數
    
    參數:
    - analysis_file: 分析結果JSON檔案
    - target_segments: 期望的片段數量
    """
    print("🎯 智能參數優化分析")
    print("="*60)
    
    # 載入數據
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entry_times = data.get('ball_entry_times', [])
    current_params = data.get('parameters', {})
    
    current_duration = current_params.get('segment_duration', 4.0)
    current_min_interval = current_params.get('min_interval', 2.0)
    current_start_offset = current_params.get('start_offset', -0.5)
    
    print(f"📁 分析檔案: {analysis_file}")
    print(f"📊 檢測到 {len(entry_times)} 個球進入點")
    
    if len(entry_times) == 0:
        print("❌ 沒有檢測到球進入點，無法進行參數優化")
        return
    
    print(f"\n⚙️ 當前參數:")
    print(f"  片段時長: {current_duration}秒")
    print(f"  最小間隔: {current_min_interval}秒")
    print(f"  開始偏移: {current_start_offset}秒")
    
    # 分析時間間隔
    intervals = []
    if len(entry_times) > 1:
        intervals = [entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)]
        
        print(f"\n📈 間隔分析:")
        print(f"  間隔數量: {len(intervals)}")
        print(f"  最小間隔: {min(intervals):.2f}秒")
        print(f"  最大間隔: {max(intervals):.2f}秒")
        print(f"  平均間隔: {np.mean(intervals):.2f}秒")
        print(f"  中位數間隔: {np.median(intervals):.2f}秒")
        print(f"  標準差: {np.std(intervals):.2f}秒")
        
        # 檢測問題
        print(f"\n🔍 問題檢測:")
        
        # 1. 檢查是否有過短的間隔（可能是重複檢測）
        short_intervals = [i for i in intervals if i < 3.0]
        if short_intervals:
            print(f"  ⚠️ 發現 {len(short_intervals)} 個可疑短間隔 (<3秒)")
            print(f"     可能原因: 同一次擊球被重複檢測")
            print(f"     建議: 增加最小間隔參數到 3-5秒")
        
        # 2. 檢查是否有重疊風險
        risky_intervals = [i for i in intervals if i < current_duration + 1.0]
        if risky_intervals:
            print(f"  ⚠️ 發現 {len(risky_intervals)} 個可能重疊的間隔")
            print(f"     間隔小於片段長度+緩衝: {current_duration + 1.0}秒")
    
    # 智能參數建議
    print(f"\n💡 智能參數建議:")
    
    # 建議片段時長
    if intervals:
        min_interval = min(intervals)
        suggested_duration = min(current_duration, min_interval * 0.7)  # 70%的最小間隔
        
        if suggested_duration < 3.0:
            suggested_duration = 3.0  # 網球最小建議時長
            print(f"  🎾 片段時長: {suggested_duration:.1f}秒 (網球最小建議)")
        elif suggested_duration != current_duration:
            print(f"  🎯 片段時長: {suggested_duration:.1f}秒 (避免重疊)")
        else:
            print(f"  ✅ 片段時長: {current_duration:.1f}秒 (當前參數良好)")
    else:
        suggested_duration = current_duration
        print(f"  📋 片段時長: {suggested_duration:.1f}秒 (保持當前設定)")
    
    # 建議最小間隔
    if intervals:
        # 根據間隔分佈建議
        median_interval = np.median(intervals)
        if median_interval < 5.0:
            # 間隔較短，可能是檢測問題
            suggested_min_interval = max(5.0, suggested_duration + 1.0)
            print(f"  🔧 最小間隔: {suggested_min_interval:.1f}秒 (過濾重複檢測)")
        else:
            suggested_min_interval = max(current_min_interval, suggested_duration + 0.5)
            print(f"  ✅ 最小間隔: {suggested_min_interval:.1f}秒 (確保分離)")
    else:
        suggested_min_interval = max(5.0, suggested_duration + 1.0)
        print(f"  📋 最小間隔: {suggested_min_interval:.1f}秒 (預設建議)")
    
    # 建議開始偏移
    suggested_start_offset = -1.0  # 網球建議
    print(f"  🎾 開始偏移: {suggested_start_offset:.1f}秒 (捕捉準備動作)")
    
    # 生成命令行參數
    print(f"\n🚀 建議的命令行參數:")
    cmd = f"--duration {suggested_duration:.1f} --min-interval {suggested_min_interval:.1f} --start-offset {suggested_start_offset:.1f}"
    print(f"  {cmd}")
    
    # 預測結果
    print(f"\n📊 使用建議參數的預測結果:")
    
    # 根據新參數過濾檢測點
    filtered_times = []
    if entry_times:
        filtered_times.append(entry_times[0])  # 第一個點總是保留
        
        for time_point in entry_times[1:]:
            if time_point - filtered_times[-1] >= suggested_min_interval:
                filtered_times.append(time_point)
    
    print(f"  過濾前檢測點: {len(entry_times)}個")
    print(f"  過濾後檢測點: {len(filtered_times)}個")
    
    if len(filtered_times) != len(entry_times):
        print(f"  🔧 將移除 {len(entry_times) - len(filtered_times)} 個可能重複的檢測點")
    
    # 計算新的片段資訊
    total_segment_duration = len(filtered_times) * suggested_duration
    if len(filtered_times) > 1:
        video_span = filtered_times[-1] - filtered_times[0] + suggested_duration
        coverage = (total_segment_duration / video_span) * 100 if video_span > 0 else 100
        print(f"  總片段時長: {total_segment_duration:.1f}秒")
        print(f"  影片跨度: {video_span:.1f}秒")
        print(f"  覆蓋率: {coverage:.1f}%")
    
    # 與目標比較（如果有提供）
    if target_segments:
        print(f"\n🎯 目標比較:")
        print(f"  期望片段數: {target_segments}")
        print(f"  預測片段數: {len(filtered_times)}")
        
        if len(filtered_times) > target_segments:
            print(f"  ⚠️ 預測片段數過多，可能需要調整檢測參數")
            print(f"     建議: 增加信心度閾值或調整邊緣檢測參數")
        elif len(filtered_times) < target_segments:
            print(f"  ⚠️ 預測片段數過少，可能遺漏了一些擊球")
            print(f"     建議: 降低信心度閾值或檢查檢測參數")
        else:
            print(f"  ✅ 預測片段數符合期望")
    
    # 返回建議參數
    return {
        'suggested_duration': suggested_duration,
        'suggested_min_interval': suggested_min_interval,
        'suggested_start_offset': suggested_start_offset,
        'filtered_entry_times': filtered_times,
        'command_line': cmd
    }

def create_optimized_test_command(analysis_file, video_file=None):
    """生成優化後的測試命令"""
    suggestions = analyze_and_suggest_parameters(analysis_file)
    
    if not suggestions:
        return None
    
    print(f"\n🧪 生成優化測試命令:")
    
    # 基本命令
    base_cmd = "python video_segment_test_cli.py"
    
    if video_file:
        cmd = f"{base_cmd} \"{video_file}\""
    else:
        # 嘗試尋找影片檔案
        result_dir = Path(analysis_file).parent
        video_candidates = list(result_dir.glob("*.mp4"))
        if video_candidates:
            cmd = f"{base_cmd} \"{video_candidates[0]}\""
        else:
            cmd = f"{base_cmd} YOUR_VIDEO_FILE.mp4"
    
    # 添加建議參數
    cmd += f" {suggestions['command_line']}"
    cmd += " --output-dir optimized_test_output"
    cmd += " --visualize"
    
    print(f"  {cmd}")
    print(f"\n💡 複製上述命令來測試優化參數！")
    
    return cmd

def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python optimize_parameters.py <analysis_results.json> [expected_segments]")
        print("  python optimize_parameters.py <analysis_results.json> [expected_segments] [video_file]")
        print("\n範例:")
        print("  python optimize_parameters.py test_enhanced_output/analysis_results.json")
        print("  python optimize_parameters.py test_enhanced_output/analysis_results.json 2")
        print("  python optimize_parameters.py test_enhanced_output/analysis_results.json 2 video.mp4")
        return 1
    
    analysis_file = sys.argv[1]
    expected_segments = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    video_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not Path(analysis_file).exists():
        print(f"❌ 分析檔案不存在: {analysis_file}")
        return 1
    
    # 執行分析
    analyze_and_suggest_parameters(analysis_file, expected_segments)
    
    # 生成測試命令
    create_optimized_test_command(analysis_file, video_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())