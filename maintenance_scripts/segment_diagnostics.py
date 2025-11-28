#!/usr/bin/env python3
"""
影片分割診斷工具
檢查檢測結果的準確性和片段內容的正確性
"""

import json
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def load_analysis_results(json_file):
    """載入分析結果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_detection_quality(results):
    """分析檢測品質"""
    print("🔍 檢測品質分析")
    print("="*50)
    
    entry_times = results.get('ball_entry_times', [])
    detection_details = results.get('detection_details', [])
    
    if not entry_times:
        print("❌ 沒有檢測到球進入時間點")
        return
    
    print(f"總檢測點數: {len(entry_times)}")
    
    # 分析間隔分佈
    if len(entry_times) > 1:
        intervals = [entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)]
        print(f"\n📊 時間間隔分析:")
        print(f"  最小間隔: {min(intervals):.2f}秒")
        print(f"  最大間隔: {max(intervals):.2f}秒")
        print(f"  平均間隔: {np.mean(intervals):.2f}秒")
        print(f"  標準差: {np.std(intervals):.2f}秒")
        
        # 檢查可疑的短間隔
        suspicious_intervals = [i for i in intervals if i < 3.0]  # 小於3秒
        if suspicious_intervals:
            print(f"\n⚠️ 可疑的短間隔 (<3秒): {len(suspicious_intervals)}個")
            for i, interval in enumerate(intervals):
                if interval < 3.0:
                    print(f"  點{i+1}到{i+2}: {interval:.2f}秒")
    
    # 分析檢測細節
    if detection_details:
        print(f"\n🎯 檢測細節分析:")
        for i, detail in enumerate(detection_details):
            time_point = detail.get('time', entry_times[i] if i < len(entry_times) else 'N/A')
            confidence = detail.get('confidence', 'N/A')
            position = detail.get('position', 'N/A')
            edge_zone = detail.get('edge_zone', 'N/A')
            
            print(f"  檢測{i+1}: 時間={time_point:.2f}s, 信心度={confidence:.3f}, 邊緣區={edge_zone}")

def check_video_segments(video_path, entry_times, duration=4.0, start_offset=-0.5):
    """檢查實際的影片片段內容"""
    print(f"\n🎬 影片片段內容檢查")
    print("="*50)
    
    if not Path(video_path).exists():
        print(f"❌ 影片檔案不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 無法開啟影片檔案: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    print(f"📹 影片資訊:")
    print(f"  檔案: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  總幀數: {total_frames}")
    print(f"  總時長: {video_duration:.2f}秒")
    
    # 檢查每個片段
    segments_info = []
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        end_time = min(video_duration, start_time + duration)
        actual_duration = end_time - start_time
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        print(f"\n  片段{i+1}:")
        print(f"    球進入時間: {entry_time:.2f}s")
        print(f"    片段範圍: {start_time:.2f}s - {end_time:.2f}s")
        print(f"    實際長度: {actual_duration:.2f}s")
        print(f"    幀範圍: {start_frame} - {end_frame}")
        
        # 取樣檢查關鍵幀
        key_frames = []
        sample_times = [start_time, entry_time, end_time - 0.5]  # 開始、球進入、結束前
        
        for j, sample_time in enumerate(sample_times):
            frame_num = int(sample_time * fps)
            if 0 <= frame_num < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    key_frames.append({
                        'time': sample_time,
                        'frame': frame_num,
                        'description': ['片段開始', '球進入點', '片段結束'][j]
                    })
        
        segments_info.append({
            'id': i + 1,
            'entry_time': entry_time,
            'start_time': start_time,
            'end_time': end_time,
            'duration': actual_duration,
            'key_frames': key_frames
        })
    
    cap.release()
    return segments_info

def create_timeline_visualization(entry_times, duration=4.0, start_offset=-0.5, output_file='segment_timeline.png'):
    """創建時間軸視覺化"""
    print(f"\n📊 創建時間軸視覺化: {output_file}")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 設定顏色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # 繪製片段
    for i, entry_time in enumerate(entry_times):
        start_time = max(0, entry_time + start_offset)
        end_time = start_time + duration
        
        color = colors[i % len(colors)]
        
        # 繪製片段矩形
        rect = Rectangle((start_time, i), duration, 0.8, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # 標記球進入點
        ax.axvline(x=entry_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.text(entry_time, i + 0.4, f'球{i+1}\n{entry_time:.1f}s', 
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 片段標籤
        ax.text(start_time + duration/2, i + 0.9, 
               f'片段{i+1} ({duration}s)', ha='center', va='bottom', fontsize=10)
    
    # 設定圖表
    ax.set_xlim(-1, max(entry_times) + duration + 1)
    ax.set_ylim(-0.5, len(entry_times) + 0.5)
    ax.set_xlabel('時間 (秒)', fontsize=12)
    ax.set_ylabel('片段', fontsize=12)
    ax.set_title('影片分割時間軸', fontsize=14, fontweight='bold')
    
    # 設定Y軸標籤
    ax.set_yticks(range(len(entry_times)))
    ax.set_yticklabels([f'片段{i+1}' for i in range(len(entry_times))])
    
    # 添加網格
    ax.grid(True, alpha=0.3)
    
    # 添加圖例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='影片片段'),
        plt.Line2D([0],[0], color='red', linestyle='--', label='球進入點')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 時間軸視覺化已儲存: {output_file}")
    
    return output_file

def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("使用方法: python segment_diagnostics.py <analysis_results.json> [video_file]")
        print("\n範例:")
        print("  python segment_diagnostics.py test_enhanced_output/analysis_results.json")
        print("  python segment_diagnostics.py test_enhanced_output/analysis_results.json video.mp4")
        return 1
    
    json_file = sys.argv[1]
    video_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(json_file).exists():
        print(f"❌ 分析檔案不存在: {json_file}")
        return 1
    
    # 載入分析結果
    try:
        results = load_analysis_results(json_file)
        print(f"📁 載入分析結果: {json_file}")
    except Exception as e:
        print(f"❌ 載入分析結果失敗: {e}")
        return 1
    
    # 分析檢測品質
    analyze_detection_quality(results)
    
    # 獲取參數
    entry_times = results.get('ball_entry_times', [])
    parameters = results.get('parameters', {})
    duration = parameters.get('segment_duration', 4.0)
    start_offset = parameters.get('start_offset', -0.5)
    
    # 創建時間軸視覺化
    output_dir = Path(json_file).parent
    timeline_file = output_dir / 'segment_timeline_diagnostic.png'
    create_timeline_visualization(entry_times, duration, start_offset, str(timeline_file))
    
    # 檢查影片片段（如果提供影片檔案）
    if video_file:
        if Path(video_file).exists():
            check_video_segments(video_file, entry_times, duration, start_offset)
        else:
            print(f"⚠️ 影片檔案不存在: {video_file}")
            # 嘗試在結果目錄中尋找影片
            video_candidates = list(Path(json_file).parent.glob("*.mp4"))
            if video_candidates:
                video_file = str(video_candidates[0])
                print(f"🔍 找到候選影片: {video_file}")
                check_video_segments(video_file, entry_times, duration, start_offset)
    
    print(f"\n✅ 診斷完成！")
    print(f"📊 時間軸視覺化: {timeline_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())