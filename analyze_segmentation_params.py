#!/usr/bin/env python3
"""
分析影片分割參數和重疊問題
幫助用戶理解和優化分割參數
"""

import json
import sys
from pathlib import Path
from datetime import timedelta

def analyze_segmentation_parameters(analysis_file=None, entry_times=None, duration=4.0, min_interval=2.0, start_offset=-0.5):
    """
    分析分割參數和可能的重疊問題
    
    參數:
    - analysis_file: 分析結果JSON檔案路徑
    - entry_times: 球進入時間點列表
    - duration: 片段時長
    - min_interval: 最小間隔
    - start_offset: 開始偏移
    """
    print("🔍 影片分割參數分析")
    print("="*50)
    
    # 從檔案讀取數據
    if analysis_file and Path(analysis_file).exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entry_times = data.get('ball_entry_times', [])
            parameters = data.get('parameters', {})
            duration = parameters.get('segment_duration', duration)
            min_interval = parameters.get('min_interval', min_interval)
            start_offset = parameters.get('start_offset', start_offset)
        
        print(f"📁 分析檔案: {analysis_file}")
    elif entry_times:
        print("📊 使用提供的參數進行分析")
    else:
        # 使用範例數據
        entry_times = [5.2, 7.8, 12.4, 18.6, 25.3]
        print("🎭 使用範例數據進行分析")
    
    print(f"\n⚙️ 當前參數:")
    print(f"  片段時長: {duration}秒")
    print(f"  最小間隔: {min_interval}秒")
    print(f"  開始偏移: {start_offset}秒")
    print(f"  球進入點數量: {len(entry_times)}")
    
    if not entry_times:
        print("❌ 沒有球進入時間點數據")
        return
    
    print(f"\n⚾ 球進入時間點:")
    for i, time_point in enumerate(entry_times):
        print(f"  {i+1}. {time_point:.2f}秒")
    
    # 計算實際分割區間
    print(f"\n📽️ 分割片段分析:")
    segments = []
    total_duration = 0
    
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        end_time = start_time + duration
        
        segment = {
            'id': i + 1,
            'entry_time': entry_time,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }
        segments.append(segment)
        total_duration += duration
        
        print(f"  片段{i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration}s)")
    
    # 檢查重疊
    print(f"\n🔄 重疊分析:")
    overlaps = []
    total_overlap_duration = 0
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        if current['end_time'] > next_seg['start_time']:
            overlap_start = next_seg['start_time']
            overlap_end = current['end_time']
            overlap_duration = overlap_end - overlap_start
            
            overlaps.append({
                'segments': (current['id'], next_seg['id']),
                'start': overlap_start,
                'end': overlap_end,
                'duration': overlap_duration
            })
            total_overlap_duration += overlap_duration
            
            print(f"  ⚠️ 片段{current['id']}和{next_seg['id']}重疊: {overlap_duration:.2f}秒 ({overlap_start:.2f}s - {overlap_end:.2f}s)")
    
    if not overlaps:
        print("  ✅ 沒有重疊")
    
    # 計算效率統計
    unique_duration = total_duration - total_overlap_duration
    if len(entry_times) > 1:
        video_span = entry_times[-1] - entry_times[0] + duration
        efficiency = (unique_duration / total_duration) * 100 if total_duration > 0 else 0
    else:
        video_span = duration
        efficiency = 100
    
    print(f"\n📊 統計摘要:")
    print(f"  總片段時長: {total_duration:.2f}秒")
    print(f"  重疊時長: {total_overlap_duration:.2f}秒")
    print(f"  實際獨特內容: {unique_duration:.2f}秒")
    print(f"  效率: {efficiency:.1f}%")
    print(f"  影片跨度: {video_span:.2f}秒")
    
    # 提供建議
    print(f"\n💡 改進建議:")
    
    if overlaps:
        # 計算建議的最小間隔
        max_gap = max([entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)])
        min_gap = min([entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)])
        
        suggested_duration = min(duration, min_gap * 0.8)  # 80%的最小間隔
        suggested_interval = duration + 0.5  # 片段長度 + 0.5秒緩衝
        
        print(f"  🎯 建議片段時長: {suggested_duration:.1f}秒 (避免重疊)")
        print(f"  🎯 建議最小間隔: {suggested_interval:.1f}秒 (確保分離)")
        
        print(f"\n  📋 具體建議:")
        print(f"    --duration {suggested_duration:.1f} --min-interval {suggested_interval:.1f}")
        
        # 檢查球進入點間隔
        print(f"\n  📈 當前球進入點間隔:")
        for i in range(len(entry_times)-1):
            gap = entry_times[i+1] - entry_times[i]
            print(f"    點{i+1}到{i+2}: {gap:.2f}秒")
            if gap < suggested_interval:
                print(f"      ⚠️ 間隔過短，可能需要調整檢測參數")
    
    else:
        print(f"  ✅ 當前參數設定良好，沒有重疊問題")
        
        # 檢查是否可以優化
        if len(entry_times) > 1:
            min_gap = min([entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)])
            if duration < min_gap * 0.9:
                suggested_duration = min(min_gap * 0.9, duration + 1.0)
                print(f"  🚀 可以考慮增加片段時長到 {suggested_duration:.1f}秒")
    
    # 針對網球的特殊建議
    print(f"\n🎾 網球專用建議:")
    print(f"  • 建議片段時長: 3-5秒 (包含準備、擊球、跟進)")
    print(f"  • 建議最小間隔: 5-8秒 (避免連續擊球重疊)")
    print(f"  • 建議開始偏移: -1.0秒 (更好的準備動作)")
    
    return {
        'segments': segments,
        'overlaps': overlaps,
        'statistics': {
            'total_duration': total_duration,
            'overlap_duration': total_overlap_duration,
            'unique_duration': unique_duration,
            'efficiency': efficiency
        }
    }

def main():
    """主函數"""
    if len(sys.argv) > 1:
        analysis_file = sys.argv[1]
        if not Path(analysis_file).exists():
            print(f"❌ 檔案不存在: {analysis_file}")
            return 1
        
        analyze_segmentation_parameters(analysis_file=analysis_file)
    else:
        # 檢查最近的分析結果
        result_candidates = [
            "video_segments_output/analysis_results.json",
            "test_enhanced_output/analysis_results.json",
            "segments_output/analysis_results.json"
        ]
        
        analysis_file = None
        for candidate in result_candidates:
            if Path(candidate).exists():
                analysis_file = candidate
                print(f"📁 找到分析檔案: {candidate}")
                break
        
        if analysis_file:
            analyze_segmentation_parameters(analysis_file=analysis_file)
        else:
            print("🔍 沒有找到分析檔案，使用範例數據:")
            analyze_segmentation_parameters()
            
            print(f"\n💡 使用方式:")
            print(f"  python {Path(__file__).name} [analysis_results.json]")
            print(f"  或直接運行查看範例分析")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())