"""
影片自動分割測試工具
功能：
1. 匯入影片檔案
2. 使用網球偵測模型找出球進入畫面的時間點
3. 自動分割影片為多個片段
4. 匯出結果並生成報告
"""

import cv2
import os
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from datetime import datetime

class VideoSegmentTester:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("影片自動分割測試工具")
        self.root.geometry("800x600")
        
        # 變數
        self.side_video_path = tk.StringVar()
        self.deg45_video_path = tk.StringVar()
        self.output_folder_path = tk.StringVar()
        self.segment_duration = tk.DoubleVar(value=4.0)
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.min_interval = tk.DoubleVar(value=2.0)
        self.preview_start_time = tk.DoubleVar(value=-0.5)
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.end_padding = tk.DoubleVar(value=1.0)
        self.dual_video_mode = tk.BooleanVar(value=False)
        
        # 結果變數
        self.detection_results = []
        self.ball_entry_times = []  # 側面影片的球進入點
        self.deg45_ball_entry_times = []  # 45度影片的球進入點
        self.tennis_model = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """設置用戶界面"""
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title_label = ttk.Label(main_frame, text="影片自動分割測試工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 檔案選擇區
        file_frame = ttk.LabelFrame(main_frame, text="檔案設定", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 雙影片模式開關
        ttk.Checkbutton(file_frame, text="雙影片同步分割模式 (Side + 45度角)", 
                       variable=self.dual_video_mode, 
                       command=self.toggle_dual_video_mode).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # 側面影片選擇
        ttk.Label(file_frame, text="側面影片 (用於分析):").grid(row=1, column=0, sticky=tk.W)
        self.side_entry = ttk.Entry(file_frame, textvariable=self.side_video_path, width=50)
        self.side_entry.grid(row=1, column=1, padx=(5, 5))
        ttk.Button(file_frame, text="瀏覽", command=self.browse_side_video).grid(row=1, column=2)
        
        # 45度影片選擇
        ttk.Label(file_frame, text="45度影片:").grid(row=2, column=0, sticky=tk.W)
        self.deg45_entry = ttk.Entry(file_frame, textvariable=self.deg45_video_path, width=50, state='disabled')
        self.deg45_entry.grid(row=2, column=1, padx=(5, 5))
        self.deg45_button = ttk.Button(file_frame, text="瀏覽", command=self.browse_deg45_video, state='disabled')
        self.deg45_button.grid(row=2, column=2)
        
        # 輸出資料夾選擇
        ttk.Label(file_frame, text="輸出資料夾:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.output_folder_path, width=50).grid(row=3, column=1, padx=(5, 5))
        ttk.Button(file_frame, text="瀏覽", command=self.browse_output_folder).grid(row=3, column=2)
        
        # 參數設定區
        param_frame = ttk.LabelFrame(main_frame, text="參數設定", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 片段時長
        ttk.Label(param_frame, text="片段時長 (秒):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.segment_duration, width=10).grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # 信心度閾值
        ttk.Label(param_frame, text="偵測信心度:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.confidence_threshold, width=10).grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        # 最小間隔
        ttk.Label(param_frame, text="最小間隔 (秒):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.min_interval, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 20))
        
        # 預覽開始時間
        ttk.Label(param_frame, text="預覽開始時間 (秒):").grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.preview_start_time, width=10).grid(row=1, column=3, sticky=tk.W, padx=(5, 0))
        
        # 動態分割模式
        ttk.Checkbutton(param_frame, text="動態分割模式 (每個片段從一個球進入點到下一個球進入點)", 
                       variable=self.dynamic_mode).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # 最後片段額外時長 (僅在動態模式時使用)
        ttk.Label(param_frame, text="最後片段額外時長 (秒):").grid(row=2, column=2, sticky=tk.W, pady=(10, 0))
        ttk.Entry(param_frame, textvariable=self.end_padding, width=10).grid(row=2, column=3, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # 控制按鈕區
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="載入模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="分析影片", command=self.analyze_video_threaded).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="預覽結果", command=self.preview_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="執行分割", command=self.execute_segmentation_threaded).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="匯出報告", command=self.export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="打開輸出資料夾", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        
        # 進度條
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # 狀態顯示
        self.status_text = tk.Text(main_frame, height=15, width=80)
        self.status_text.grid(row=5, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 捲軸
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=5, column=3, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # 設定默認輸出路徑
        default_output = Path.cwd() / "video_segments_output"
        self.output_folder_path.set(str(default_output))
        
    def browse_side_video(self):
        """瀏覽側面影片"""
        filename = filedialog.askopenfilename(
            title="選擇側面影片",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.side_video_path.set(filename)
            self.log(f"選擇側面影片: {filename}")
    
    def browse_deg45_video(self):
        """瀏覽45度影片"""
        filename = filedialog.askopenfilename(
            title="選擇45度影片",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.deg45_video_path.set(filename)
            self.log(f"選擇45度影片: {filename}")
    
    def toggle_dual_video_mode(self):
        """切換雙影片模式的UI狀態"""
        if self.dual_video_mode.get():
            # 啟用雙影片模式
            self.deg45_entry.config(state='normal')
            self.deg45_button.config(state='normal')
            self.log("已啟用雙影片模式")
        else:
            # 停用雙影片模式
            self.deg45_entry.config(state='disabled')
            self.deg45_button.config(state='disabled')
            self.deg45_video_path.set("")  # 清空45度影片路徑
            self.log("已停用雙影片模式")
            
    def browse_input_video(self):
        """瀏覽輸入影片（相容性保留）"""
        filename = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.side_video_path.set(filename)  # 統一使用side_video_path
            self.log(f"選擇輸入影片: {filename}")
            
    def browse_output_folder(self):
        """瀏覽輸出資料夾"""
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_folder_path.set(folder)
            self.log(f"選擇輸出資料夾: {folder}")
    
    def open_output_folder(self):
        """打開輸出資料夾"""
        output_folder = self.output_folder_path.get()
        if not output_folder:
            messagebox.showwarning("警告", "請先設定輸出資料夾")
            return
        
        if not os.path.exists(output_folder):
            messagebox.showerror("錯誤", f"輸出資料夾不存在: {output_folder}")
            return
        
        try:
            import subprocess
            import sys
            
            if sys.platform == "win32":
                # Windows
                os.startfile(output_folder)
                self.log(f"📂 已打開輸出資料夾: {output_folder}")
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", output_folder])
                self.log(f"📂 已打開輸出資料夾: {output_folder}")
            else:
                # Linux
                subprocess.run(["xdg-open", output_folder])
                self.log(f"📂 已打開輸出資料夾: {output_folder}")
                
        except Exception as e:
            self.log(f"❌ 無法打開輸出資料夾: {str(e)}")
            messagebox.showerror("錯誤", f"無法打開資料夾: {str(e)}")
            
    def load_model(self):
        """載入網球偵測模型"""
        try:
            model_path = "model/tennisball_OD_v1.pt"
            if not os.path.exists(model_path):
                messagebox.showerror("錯誤", f"模型檔案不存在: {model_path}")
                return
                
            self.log("正在載入網球偵測模型...")
            self.tennis_model = YOLO(model_path)
            self.log("✅ 網球偵測模型載入完成！")
            
        except Exception as e:
            self.log(f"❌ 模型載入失敗: {str(e)}")
            messagebox.showerror("錯誤", f"模型載入失敗: {str(e)}")
            
    def analyze_video_threaded(self):
        """在新線程中分析影片"""
        if not self.side_video_path.get():
            messagebox.showwarning("警告", "請先選擇側面影片")
            return
            
        if not self.tennis_model:
            messagebox.showwarning("警告", "請先載入模型")
            return
            
        thread = threading.Thread(target=self.analyze_video)
        thread.daemon = True
        thread.start()
        
    def analyze_video(self):
        """分析影片找出球進入時間點"""
        try:
            side_video_path = self.side_video_path.get()
            self.log(f"開始分析側面影片: {Path(side_video_path).name}")
            
            # 分析側面影片
            side_entry_times = self.analyze_single_video(side_video_path, "側面")
            if side_entry_times:
                self.ball_entry_times = side_entry_times
                self.log(f"✅ 側面影片找到 {len(side_entry_times)} 個球進入點")
            else:
                self.log("❌ 側面影片未找到球進入點")
                return
            
            # 如果啟用雙影片模式，分析45度影片
            if self.dual_video_mode.get() and self.deg45_video_path.get():
                deg45_video_path = self.deg45_video_path.get()
                self.log(f"開始分析45度影片: {Path(deg45_video_path).name}")
                
                deg45_entry_times = self.analyze_single_video(deg45_video_path, "45度")
                if deg45_entry_times:
                    self.deg45_ball_entry_times = deg45_entry_times
                    self.log(f"✅ 45度影片找到 {len(deg45_entry_times)} 個球進入點")
                else:
                    self.log("❌ 45度影片未找到球進入點")
                    self.deg45_ball_entry_times = []
            else:
                self.deg45_ball_entry_times = []
            
            # 顯示結果
            self.display_analysis_results()
            
        except Exception as e:
            self.log(f"❌ 分析過程發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def display_analysis_results(self):
        """顯示雙影片分析結果"""
        # 將側面影片結果設為主要顯示結果（用於預覽圖表）
        self.detection_results = []  # 這裡可以保留側面影片的詳細結果供圖表使用
        
        # 顯示結果摘要
        if self.dual_video_mode.get():
            self.log(f"📊 分析結果摘要:")
            self.log(f"   側面影片: {len(self.ball_entry_times)} 個球進入點")
            self.log(f"   45度影片: {len(self.deg45_ball_entry_times)} 個球進入點")
            
            # 比較兩個影片的球進入點
            if len(self.ball_entry_times) != len(self.deg45_ball_entry_times):
                self.log(f"⚠️  注意: 兩個角度檢測到的球進入點數量不同")
        else:
            self.log(f"📊 側面影片分析完成: {len(self.ball_entry_times)} 個球進入點")
    
    def analyze_single_video(self, video_path, video_type):
        """分析單一影片的球進入點"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log(f"❌ 無法開啟{video_type}影片檔案")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.log(f"{video_type}影片資訊: {total_frames} 影格, {fps:.2f} FPS, {duration:.2f} 秒")
            
            # 重置結果變數（針對單一影片分析）
            detection_results = []
            ball_entry_times = []
            
            previous_ball_detected = False
            previous_ball_position = None
            last_entry_time = -self.min_interval.get()
            frame_count = 0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 定義畫面邊緣區域 (用於判斷球是否從邊緣進入)
            edge_threshold = 0.15  # 邊緣區域佔畫面的比例
            left_edge = frame_width * edge_threshold
            right_edge = frame_width * (1 - edge_threshold)
            top_edge = frame_height * edge_threshold
            bottom_edge = frame_height * (1 - edge_threshold)
            
            self.log(f"{video_type}畫面尺寸: {frame_width}x{frame_height}")
            self.log(f"{video_type}邊緣偵測區域: 左({left_edge:.0f}), 右({right_edge:.0f}), 上({top_edge:.0f}), 下({bottom_edge:.0f})")
            
            # 更新進度條
            self.progress['maximum'] = total_frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = frame_count / fps
                
                # 使用 YOLO 偵測網球
                results = self.tennis_model(frame, verbose=False)
                
                # 檢查偵測結果和位置
                current_ball_detected = False
                max_confidence = 0
                ball_position = None
                ball_in_edge = False
                
                if len(results[0].boxes) > 0:
                    best_box = None
                    best_confidence = 0
                    
                    for box in results[0].boxes:
                        confidence = float(box.conf[0])
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_box = box
                    
                    if best_confidence > self.confidence_threshold.get():
                        current_ball_detected = True
                        max_confidence = best_confidence
                        
                        # 取得球的位置 (邊界框中心點)
                        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                        ball_center_x = (x1 + x2) / 2
                        ball_center_y = (y1 + y2) / 2
                        ball_position = (ball_center_x, ball_center_y)
                        
                        # 檢查球是否在畫面邊緣
                        ball_in_edge = (ball_center_x < left_edge or ball_center_x > right_edge or
                                       ball_center_y < top_edge or ball_center_y > bottom_edge)
                
                # 記錄偵測結果
                detection_info = {
                    'frame': frame_count,
                    'time': current_time,
                    'detected': current_ball_detected,
                    'confidence': max_confidence,
                    'position': ball_position,
                    'in_edge': ball_in_edge
                }
                detection_results.append(detection_info)
                
                # 判斷球進入畫面的邏輯
                is_ball_entry = False
                entry_reason = ""
                
                if current_ball_detected and not previous_ball_detected:
                    # 情況1: 球從無到有出現
                    if ball_in_edge:
                        # 球出現在邊緣 = 球從畫面外進入
                        is_ball_entry = True
                        entry_reason = f"邊緣進入 (位置: {ball_center_x:.0f}, {ball_center_y:.0f})"
                    else:
                        # 球出現在中央 = 可能是擊球瞬間，不算進入
                        entry_reason = f"中央出現 (位置: {ball_center_x:.0f}, {ball_center_y:.0f}) - 忽略"
                
                elif current_ball_detected and previous_ball_detected and previous_ball_position:
                    # 情況2: 球持續存在，檢查是否從邊緣移向中央
                    prev_x, prev_y = previous_ball_position
                    curr_x, curr_y = ball_position
                    
                    # 檢查球是否從邊緣移向中央 (移動方向分析)
                    prev_in_edge = (prev_x < left_edge or prev_x > right_edge or
                                   prev_y < top_edge or prev_y > bottom_edge)
                    
                    if prev_in_edge and not ball_in_edge:
                        # 球從邊緣移到中央區域
                        move_distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                        if move_distance > 20:  # 移動距離閾值
                            is_ball_entry = True
                            entry_reason = f"邊緣移入 (從 {prev_x:.0f},{prev_y:.0f} 到 {curr_x:.0f},{curr_y:.0f})"
                
                # 檢查時間間隔並記錄進入點
                if is_ball_entry and current_time - last_entry_time >= self.min_interval.get():
                    ball_entry_times.append(current_time)
                    last_entry_time = current_time
                    self.log(f"🎾 {video_type}球進入: {current_time:.2f}s - {entry_reason} (信心度: {max_confidence:.3f})")
                
                # 更新前一幀的狀態
                previous_ball_detected = current_ball_detected
                previous_ball_position = ball_position
                frame_count += 1
                
                # 更新進度
                if frame_count % 30 == 0:  # 每30幀更新一次
                    self.progress['value'] = frame_count
                    self.root.update_idletasks()
                    
            cap.release()
            self.progress['value'] = total_frames
            
            self.log(f"✅ {video_type}分析完成！偵測到 {len(ball_entry_times)} 次球進入畫面")
            self.log(f"{video_type}球進入時間點: {[f'{t:.2f}s' for t in ball_entry_times]}")
            
            return ball_entry_times
            
        except Exception as e:
            self.log(f"❌ 分析失敗: {str(e)}")
            
    def preview_results(self):
        """預覽分析結果"""
        if not self.detection_results:
            messagebox.showwarning("警告", "請先分析影片")
            return
            
        try:
            # 創建圖表 - 增加到3個子圖
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
            
            # 圖1: 偵測信心度時間序列
            times = [r['time'] for r in self.detection_results]
            confidences = [r['confidence'] for r in self.detection_results]
            
            ax1.plot(times, confidences, 'b-', alpha=0.7, linewidth=1)
            ax1.axhline(y=self.confidence_threshold.get(), color='r', linestyle='--', 
                       label=f'信心度閾值 ({self.confidence_threshold.get()})')
            
            # 標記球進入時間點
            for entry_time in self.ball_entry_times:
                ax1.axvline(x=entry_time, color='g', linestyle='-', alpha=0.8, linewidth=2)
                ax1.text(entry_time, ax1.get_ylim()[1]*0.9, f'{entry_time:.1f}s', 
                        rotation=90, ha='right', va='top')
            
            ax1.set_xlabel('時間 (秒)')
            ax1.set_ylabel('偵測信心度')
            ax1.set_title('網球偵測信心度 vs 時間')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 圖2: 偵測狀態與邊緣位置
            detected_states = [1 if r['detected'] else 0 for r in self.detection_results]
            edge_states = [0.5 if r.get('in_edge', False) else 0 for r in self.detection_results]
            
            ax2.fill_between(times, detected_states, alpha=0.6, color='orange', label='偵測到球')
            ax2.fill_between(times, edge_states, alpha=0.4, color='purple', label='球在邊緣')
            
            # 標記預計分割區間
            for i, entry_time in enumerate(self.ball_entry_times):
                start_time = max(0, entry_time + self.preview_start_time.get())
                
                # 計算片段結束時間
                if self.dynamic_mode.get():
                    # 動態模式
                    if i < len(self.ball_entry_times) - 1:
                        end_time = self.ball_entry_times[i + 1] + self.preview_start_time.get()
                    else:
                        end_time = entry_time + 4.0 + self.end_padding.get()
                    duration = max(1.0, end_time - start_time)
                else:
                    # 固定長度模式
                    duration = self.segment_duration.get()
                    end_time = start_time + duration
                
                color = 'blue' if self.dynamic_mode.get() else 'red'
                ax2.axvspan(start_time, end_time, alpha=0.3, color=color, 
                           label=('動態分割區間' if self.dynamic_mode.get() else '固定分割區間') if i == 0 else '')
                ax2.text(start_time + duration/2, 0.75, 
                        f'片段{i+1}\n({duration:.1f}s)', ha='center', va='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax2.set_xlabel('時間 (秒)')
            ax2.set_ylabel('偵測狀態')
            title = '網球偵測狀態與分割區間 (紫色=邊緣位置) - '
            title += '動態分割模式' if self.dynamic_mode.get() else '固定長度分割模式'
            ax2.set_title(title)
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 圖3: 球的位置軌跡 (如果有位置資訊)
            x_positions = []
            y_positions = []
            valid_times = []
            
            for r in self.detection_results:
                if r['detected'] and r.get('position'):
                    x, y = r['position']
                    x_positions.append(x)
                    y_positions.append(y)
                    valid_times.append(r['time'])
            
            if x_positions:
                # 創建顏色映射表示時間
                scatter = ax3.scatter(x_positions, y_positions, c=valid_times, 
                                    cmap='viridis', alpha=0.6, s=20)
                
                # 標記球進入點
                for entry_time in self.ball_entry_times:
                    # 找到最接近進入時間的位置
                    closest_idx = min(range(len(valid_times)), 
                                     key=lambda i: abs(valid_times[i] - entry_time))
                    if abs(valid_times[closest_idx] - entry_time) < 0.5:  # 0.5秒內
                        ax3.scatter(x_positions[closest_idx], y_positions[closest_idx], 
                                  color='red', s=100, marker='*', 
                                  label='球進入點' if entry_time == self.ball_entry_times[0] else '')
                
                ax3.set_xlabel('X 位置 (像素)')
                ax3.set_ylabel('Y 位置 (像素)')
                ax3.set_title('球的位置軌跡 (顏色表示時間，紅星表示進入點)')
                ax3.invert_yaxis()  # 反轉Y軸，因為影像座標系Y軸向下
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                # 添加色條
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('時間 (秒)')
            else:
                ax3.text(0.5, 0.5, '無位置資訊可顯示', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=16)
                ax3.set_title('球的位置軌跡')
            
            plt.tight_layout()
            plt.show()
            
            self.log("📊 預覽圖表已顯示 (包含位置資訊)")
            
        except Exception as e:
            self.log(f"❌ 預覽失敗: {str(e)}")
            
    def execute_segmentation_threaded(self):
        """在新線程中執行分割"""
        if not self.ball_entry_times:
            messagebox.showwarning("警告", "請先分析影片")
            return
            
        if not self.output_folder_path.get():
            messagebox.showwarning("警告", "請設定輸出資料夾")
            return
            
        thread = threading.Thread(target=self.execute_segmentation)
        thread.daemon = True
        thread.start()
        
    def execute_segmentation(self):
        """執行影片分割"""
        try:
            side_video = self.side_video_path.get()
            deg45_video = self.deg45_video_path.get()
            output_folder = Path(self.output_folder_path.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # 檢查雙影片模式
            if self.dual_video_mode.get():
                if not deg45_video:
                    self.log("❌ 雙影片模式需要同時選擇側面和45度影片")
                    messagebox.showerror("錯誤", "請選擇45度影片檔案")
                    return
                
                self.log(f"🎬 雙影片模式: 側面影片={Path(side_video).name}, 45度影片={Path(deg45_video).name}")
            else:
                self.log(f"📹 單影片模式: {Path(side_video).name}")
            
            # 檢查是否使用動態模式
            if self.dynamic_mode.get():
                self.log(f"🚀 使用動態分割模式: 每個片段從一個球進入點到下一個球進入點")
            else:
                self.log(f"📐 使用固定長度分割模式: 每個片段{self.segment_duration.get()}秒")
            
            self.log(f"開始分割影片到: {output_folder}")
            
            # 檢查 ffmpeg 是否可用
            ffmpeg_available = self.check_ffmpeg_availability()
            if not ffmpeg_available:
                self.log("❌ FFmpeg 不可用，將使用 OpenCV 進行分割")
                self.execute_opencv_segmentation()
                return
            
            # 獲取影片總長度 (動態模式需要)
            video_duration = None
            if self.dynamic_mode.get():
                cap = cv2.VideoCapture(side_video)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps
                    cap.release()
            
            # 執行分割
            if self.dual_video_mode.get() and deg45_video:
                # 雙影片模式：分別使用各自的球進入點進行分割
                self.log(f"🎬 啟動雙影片模式分割")
                self.log(f"   側面球進入點: {len(self.ball_entry_times)} 個")
                self.log(f"   45度球進入點: {len(self.deg45_ball_entry_times)} 個")
                self.execute_dual_video_segmentation(side_video, deg45_video, output_folder, ffmpeg_available)
            else:
                # 單影片模式：使用側面影片的球進入點
                self.log(f"🎬 啟動單影片模式分割")
                self.execute_single_video_segmentation(side_video, output_folder, ffmpeg_available, "側面")
        
        except Exception as e:
            self.log(f"❌ 分割執行失敗: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def execute_single_video_segmentation(self, video_path, output_folder, ffmpeg_available, video_type):
        """執行單一影片分割"""
        try:
            ball_entry_times = self.ball_entry_times if video_type == "側面" else self.deg45_ball_entry_times
            
            # 調試信息：檢查球進入點數據
            self.log(f"🔍 {video_type}影片分割調試信息:")
            self.log(f"   影片路徑: {video_path}")
            self.log(f"   球進入點數量: {len(ball_entry_times)}")
            self.log(f"   球進入時間: {[f'{t:.2f}s' for t in ball_entry_times]}")
            
            if not ball_entry_times:
                if video_type == "45度":
                    # 如果45度影片沒有找到球進入點，使用側面影片的球進入點作為備選
                    self.log(f"⚠️  45度影片沒有球進入點，使用側面影片的球進入點作為備選")
                    ball_entry_times = self.ball_entry_times
                    if not ball_entry_times:
                        self.log(f"❌ 側面影片也沒有球進入點數據，無法分割45度影片")
                        return
                else:
                    self.log(f"❌ {video_type}影片沒有球進入點數據，無法分割")
                    return
            
            # 獲取影片總長度 (動態模式需要)
            video_duration = None
            if self.dynamic_mode.get():
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps
                    cap.release()
            
            # 更新進度條
            self.progress['maximum'] = len(ball_entry_times)
            self.progress['value'] = 0
            
            segment_info = []
            
            for i, entry_time in enumerate(ball_entry_times):
                start_time = max(0, entry_time + self.preview_start_time.get())
                
                # 計算片段時長
                if self.dynamic_mode.get():
                    # 動態模式: 從當前進入點到下一個進入點
                    if i < len(ball_entry_times) - 1:
                        # 不是最後一個片段
                        end_time = ball_entry_times[i + 1] + self.preview_start_time.get()
                    else:
                        # 最後一個片段
                        end_time = entry_time + 4.0 + self.end_padding.get()
                        if video_duration:
                            end_time = min(end_time, video_duration)
                    
                    duration = max(1.0, end_time - start_time)  # 最少1秒
                else:
                    # 固定長度模式
                    duration = self.segment_duration.get()
                
                # 處理影片分割
                input_name = Path(video_path).stem
                output_name = f"{input_name}_segment_{i+1:02d}.mp4"
                output_path = output_folder / output_name
                
                # 使用 ffmpeg 分割影片
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    str(output_path),
                    '-y'  # 覆蓋已存在的檔案
                ]
                
                if self.dynamic_mode.get():
                    self.log(f"{video_type}片段{i+1}: {start_time:.2f}s - {start_time + duration:.2f}s (時長: {duration:.2f}s)")
                    self.log(f"   球進入時間: {entry_time:.2f}s")
                else:
                    self.log(f"{video_type}分割片段 {i+1}: {start_time:.2f}s - {start_time + duration:.2f}s")
                
                # 執行影片分割
                segment_success = True
                try:
                    import subprocess
                    self.log(f"🎬 執行FFmpeg命令: {' '.join(cmd[:3])} ... {cmd[-2:]}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0 and output_path.exists():
                        file_size = os.path.getsize(output_path) / (1024*1024)  # MB
                        self.log(f"✅ {video_type}片段 {i+1} 完成 ({file_size:.1f} MB)")
                        
                        segment_info.append({
                            'segment_id': i + 1,
                            'entry_time': entry_time,
                            'start_time': start_time,
                            'duration': duration,
                            'output_file': output_name,
                            'file_size_mb': round(file_size, 2),
                            'success': True
                        })
                    else:
                        error_msg = result.stderr if result.stderr else "Unknown error"
                        self.log(f"❌ {video_type}片段 {i+1} 失敗:")
                        self.log(f"   FFmpeg返回碼: {result.returncode}")
                        self.log(f"   檔案是否存在: {output_path.exists()}")
                        self.log(f"   錯誤訊息: {error_msg}")
                        segment_success = False
                        
                        segment_info.append({
                            'segment_id': i + 1,
                            'entry_time': entry_time,
                            'start_time': start_time,
                            'duration': duration,
                            'output_file': output_name,
                            'success': False,
                            'error': f'FFmpeg failed: {error_msg}'
                        })
                        
                except Exception as e:
                    self.log(f"❌ {video_type}片段 {i+1} 失敗: {str(e)}")
                    segment_success = False
                    
                    segment_info.append({
                        'segment_id': i + 1,
                        'entry_time': entry_time,
                        'start_time': start_time,
                        'duration': duration,
                        'output_file': output_name,
                        'success': False,
                        'error': f'Exception: {str(e)}'
                    })
                
                # 更新進度
                self.progress['value'] = i + 1
                self.root.update_idletasks()
            
            # 儲存分割資訊
            info_file = output_folder / f"segmentation_info_{video_type}.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'input_video': video_path,
                    'video_type': video_type,
                    'analysis_time': datetime.now().isoformat(),
                    'parameters': {
                        'segment_duration': self.segment_duration.get(),
                        'confidence_threshold': self.confidence_threshold.get(),
                        'min_interval': self.min_interval.get(),
                        'preview_start_time': self.preview_start_time.get(),
                        'dynamic_mode': self.dynamic_mode.get(),
                        'end_padding': self.end_padding.get()
                    },
                    'ball_entry_times': ball_entry_times,
                    'segments': segment_info
                }, f, ensure_ascii=False, indent=2)
            
            successful = sum(1 for s in segment_info if s['success'])
            self.log(f"🎬 {video_type}分割完成！成功: {successful}/{len(segment_info)} 個片段")
            
            # 如果是單影片模式，也提供詳細資訊
            if not self.dual_video_mode.get():
                self.log("="*50)
                self.log(f"📁 輸出資料夾: {output_folder}")
                self.log("📂 產生的檔案:")
                
                video_name = Path(video_path).stem
                video_files = list(Path(output_folder).glob(f"{video_name}_segment_*.mp4"))
                for file_path in sorted(video_files):
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    self.log(f"   ✅ {file_path.name} ({file_size:.1f} MB)")
                
                self.log("")
                self.log("💡 提示: 點擊「打開輸出資料夾」按鈕可以直接訪問所有分割片段")
                self.log("="*50)
                
                # 詢問用戶是否要立即打開輸出資料夾
                if messagebox.askyesno("分割完成", f"{video_type}影片分割已完成！\n\n成功產生: {successful} 個片段\n\n是否要立即打開輸出資料夾？"):
                    self.open_output_folder()
            
        except Exception as e:
            self.log(f"❌ {video_type}分割失敗: {str(e)}")
    
    def execute_dual_video_segmentation(self, side_video, deg45_video, output_folder, ffmpeg_available):
        """執行雙影片分割"""
        try:
            self.log("🎬 執行雙影片分割...")
            
            # 分別分割兩個影片
            self.execute_single_video_segmentation(side_video, output_folder, ffmpeg_available, "側面")
            self.execute_single_video_segmentation(deg45_video, output_folder, ffmpeg_available, "45度")
            
            # 顯示詳細的完成資訊
            self.log("="*60)
            self.log("🎬 雙影片分割完成！")
            self.log(f"📁 輸出資料夾: {output_folder}")
            self.log("")
            self.log("📂 產生的檔案:")
            
            # 檢查並列出側面影片檔案
            side_name = Path(side_video).stem
            side_files = list(Path(output_folder).glob(f"{side_name}_segment_*.mp4"))
            if side_files:
                self.log(f"   側面影片片段: {len(side_files)} 個檔案")
                for file_path in sorted(side_files):
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    self.log(f"   ✅ {file_path.name} ({file_size:.1f} MB)")
            
            # 檢查並列出45度影片檔案
            deg45_name = Path(deg45_video).stem
            self.log(f"🔍 搜尋45度影片片段: {deg45_name}_segment_*.mp4")
            deg45_files = list(Path(output_folder).glob(f"{deg45_name}_segment_*.mp4"))
            if deg45_files:
                self.log(f"   45度影片片段: {len(deg45_files)} 個檔案")
                for file_path in sorted(deg45_files):
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    self.log(f"   ✅ {file_path.name} ({file_size:.1f} MB)")
            else:
                self.log(f"⚠️  未找到45度影片片段檔案")
                # 列出輸出資料夾中的所有MP4檔案進行調試
                all_mp4_files = list(Path(output_folder).glob("*.mp4"))
                self.log(f"   輸出資料夾中的所有MP4檔案: {len(all_mp4_files)} 個")
                for file_path in sorted(all_mp4_files):
                    self.log(f"   📹 {file_path.name}")
            
            self.log("")
            self.log("💡 提示: 點擊「打開輸出資料夾」按鈕可以直接訪問所有分割片段")
            self.log("="*60)
            
            # 詢問用戶是否要立即打開輸出資料夾
            if messagebox.askyesno("分割完成", f"雙影片分割已完成！\n\n側面影片: {len(side_files)} 個片段\n45度影片: {len(deg45_files)} 個片段\n\n是否要立即打開輸出資料夾？"):
                self.open_output_folder()
            
        except Exception as e:
            self.log(f"❌ 雙影片分割失敗: {str(e)}")
    
    def check_ffmpeg_availability(self):
        """檢查 FFmpeg 是否可用"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False
    
    def execute_opencv_segmentation(self):
        """使用 OpenCV 進行影片分割（備用方案）"""
        try:
            side_video = self.side_video_path.get()
            output_folder = Path(self.output_folder_path.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # 檢查側面影片
            cap_side = cv2.VideoCapture(side_video)
            if not cap_side.isOpened():
                self.log("❌ 無法開啟側面影片")
                return
            
            # 取得側面影片資訊
            side_fps = cap_side.get(cv2.CAP_PROP_FPS)
            side_total_frames = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
            side_duration = side_total_frames / side_fps
            side_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            side_width = int(cap_side.get(cv2.CAP_PROP_FRAME_WIDTH))
            side_height = int(cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 檢查是否處於雙影片模式
            is_dual_mode = self.dual_video_mode.get()
            deg45_video = self.deg45_video_path.get() if is_dual_mode else None
            cap_45 = None
            
            # 如果是雙影片模式，則初始化45度影片
            if is_dual_mode:
                if not deg45_video:
                    self.log("⚠️ 雙影片模式已啟用，但未選擇45度影片")
                    return
                
                cap_45 = cv2.VideoCapture(deg45_video)
                if not cap_45.isOpened():
                    self.log("❌ 無法開啟45度影片")
                    cap_side.release()
                    return
                
                # 取得45度影片資訊
                deg45_fps = cap_45.get(cv2.CAP_PROP_FPS)
                deg45_total_frames = int(cap_45.get(cv2.CAP_PROP_FRAME_COUNT))
                deg45_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                deg45_width = int(cap_45.get(cv2.CAP_PROP_FRAME_WIDTH))
                deg45_height = int(cap_45.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 檢查兩個影片的FPS是否相近
                if abs(side_fps - deg45_fps) > 1.0:
                    self.log(f"⚠️ 警告：兩個影片的幀率不同 (側面: {side_fps} fps, 45度: {deg45_fps} fps)")
            
            if self.dynamic_mode.get():
                self.log("🔄 使用 OpenCV 進行動態影片分割...")
            else:
                self.log("🔄 使用 OpenCV 進行固定長度影片分割...")
            
            if is_dual_mode:
                self.log("🔄 雙影片模式啟用：將同時分割側面和45度影片")
                
            segment_info = []
            successful_segments = 0
            
            # 設置進度條最大值
            self.progress['maximum'] = len(self.ball_entry_times)
            self.progress['value'] = 0
            
            for i, entry_time in enumerate(self.ball_entry_times):
                start_time = max(0, entry_time + self.preview_start_time.get())
                
                # 計算片段時長
                if self.dynamic_mode.get():
                    # 動態模式: 從當前進入點到下一個進入點
                    if i < len(self.ball_entry_times) - 1:
                        # 不是最後一個片段
                        end_time = self.ball_entry_times[i + 1] + self.preview_start_time.get()
                    else:
                        # 最後一個片段
                        end_time = min(side_duration, entry_time + 4.0 + self.end_padding.get())
                    
                    duration = max(1.0, end_time - start_time)  # 最少1秒
                else:
                    # 固定長度模式
                    duration = self.segment_duration.get()
                
                # 側面影片處理
                side_start_frame = int(start_time * side_fps)
                side_end_frame = int((start_time + duration) * side_fps)
                if side_end_frame > side_total_frames:
                    side_end_frame = side_total_frames
                
                # 生成側面輸出檔名
                side_input_name = Path(side_video).stem
                side_output_name = f"{side_input_name}_segment_{i+1:02d}.mp4"
                side_output_path = output_folder / side_output_name
                
                # 日誌輸出
                if self.dynamic_mode.get():
                    self.log(f"片段{i+1}: {start_time:.2f}s - {start_time + duration:.2f}s (時長: {duration:.2f}s)")
                    self.log(f"   球進入時間: {entry_time:.2f}s")
                else:
                    self.log(f"分割片段 {i+1}: {start_time:.2f}s - {start_time + duration:.2f}s")
                
                # 設置側面影片位置到開始幀
                cap_side.set(cv2.CAP_PROP_POS_FRAMES, side_start_frame)
                
                # 創建側面影片寫入器
                side_out = cv2.VideoWriter(str(side_output_path), side_fourcc, side_fps, (side_width, side_height))
                
                side_frames_written = 0
                current_frame = side_start_frame
                
                # 處理側面影片
                while current_frame < side_end_frame and current_frame < side_total_frames:
                    ret, frame = cap_side.read()
                    if not ret:
                        break
                    
                    side_out.write(frame)
                    side_frames_written += 1
                    current_frame += 1
                
                side_out.release()
                
                # 45度影片處理 (如果雙影片模式啟用)
                deg45_output_path = None
                deg45_frames_written = 0
                deg45_output_name = None
                deg45_file_size = 0
                
                if is_dual_mode and cap_45 is not None:
                    # 計算45度影片的幀數
                    deg45_start_frame = int(start_time * deg45_fps)
                    deg45_end_frame = int((start_time + duration) * deg45_fps)
                    if deg45_end_frame > deg45_total_frames:
                        deg45_end_frame = deg45_total_frames
                    
                    # 生成45度輸出檔名
                    deg45_input_name = Path(deg45_video).stem
                    deg45_output_name = f"{deg45_input_name}_segment_{i+1:02d}.mp4"
                    deg45_output_path = output_folder / deg45_output_name
                    
                    # 設置45度影片位置到開始幀
                    cap_45.set(cv2.CAP_PROP_POS_FRAMES, deg45_start_frame)
                    
                    # 創建45度影片寫入器
                    deg45_out = cv2.VideoWriter(str(deg45_output_path), deg45_fourcc, deg45_fps, (deg45_width, deg45_height))
                    
                    deg45_frames_written = 0
                    current_frame = deg45_start_frame
                    
                    # 處理45度影片
                    while current_frame < deg45_end_frame and current_frame < deg45_total_frames:
                        ret, frame = cap_45.read()
                        if not ret:
                            break
                        
                        deg45_out.write(frame)
                        deg45_frames_written += 1
                        current_frame += 1
                    
                    deg45_out.release()
                    
                    if deg45_output_path.exists() and deg45_frames_written > 0:
                        deg45_file_size = os.path.getsize(deg45_output_path) / (1024*1024)  # MB
                        self.log(f"✅ 45度片段 {i+1} 完成 ({deg45_file_size:.1f} MB, {deg45_frames_written} 幀)")
                    else:
                        self.log(f"❌ 45度片段 {i+1} 失敗")
                
                # 側面影片段結果記錄
                segment_result = {
                    'segment_id': i + 1,
                    'entry_time': entry_time,
                    'start_time': start_time,
                    'duration': duration
                }
                
                if side_frames_written > 0 and side_output_path.exists():
                    side_file_size = os.path.getsize(side_output_path) / (1024*1024)  # MB
                    segment_result.update({
                        'side_output_file': side_output_name,
                        'side_file_size_mb': round(side_file_size, 2),
                        'side_frames_written': side_frames_written,
                        'side_success': True
                    })
                    successful_segments += 1
                    self.log(f"✅ 側面片段 {i+1} 完成 ({side_file_size:.1f} MB, {side_frames_written} 幀)")
                else:
                    segment_result.update({
                        'side_output_file': side_output_name,
                        'side_success': False,
                        'side_error': 'OpenCV segmentation failed'
                    })
                    self.log(f"❌ 側面片段 {i+1} 失敗")
                
                # 添加45度影片段結果 (如果適用)
                if is_dual_mode:
                    if deg45_frames_written > 0 and deg45_output_path and deg45_output_path.exists():
                        segment_result.update({
                            'deg45_output_file': deg45_output_name,
                            'deg45_file_size_mb': round(deg45_file_size, 2),
                            'deg45_frames_written': deg45_frames_written,
                            'deg45_success': True
                        })
                    else:
                        segment_result.update({
                            'deg45_output_file': deg45_output_name,
                            'deg45_success': False,
                            'deg45_error': 'OpenCV segmentation failed'
                        })
                
                segment_info.append(segment_result)
                
                # 更新進度
                self.progress['value'] = i + 1
                self.root.update_idletasks()
            
            # 釋放資源
            cap_side.release()
            if is_dual_mode and cap_45 is not None:
                cap_45.release()
            
            # 儲存分割資訊
            info_file = output_folder / "segmentation_info.json"
            
            # 準備 JSON 資料
            json_data = {
                'input_videos': {
                    'side': side_video,
                },
                'analysis_time': datetime.now().isoformat(),
                'method': 'OpenCV',
                'parameters': {
                    'segment_duration': self.segment_duration.get(),
                    'confidence_threshold': self.confidence_threshold.get(),
                    'min_interval': self.min_interval.get(),
                    'preview_start_time': self.preview_start_time.get(),
                    'dynamic_mode': self.dynamic_mode.get(),
                    'dual_video_mode': is_dual_mode
                },
                'ball_entry_times': self.ball_entry_times,
                'segments': segment_info
            }
            
            # 如果是雙影片模式，添加45度影片資訊
            if is_dual_mode:
                json_data['input_videos']['deg45'] = deg45_video
                if hasattr(self, 'deg45_ball_entry_times') and self.deg45_ball_entry_times:
                    json_data['deg45_ball_entry_times'] = self.deg45_ball_entry_times
            
            # 寫入 JSON 檔案
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.log(f"🎬 OpenCV 分割完成！成功: {successful_segments}/{len(self.ball_entry_times)} 個片段")
            self.log(f"📁 輸出資料夾: {output_folder}")
            
            # 顯示完成對話框
            mode_text = "雙影片" if is_dual_mode else "單影片"
            if messagebox.askyesno("分割完成", f"影片分割已完成！({mode_text}模式)\n\n成功產生: {successful_segments} 個片段\n\n是否要立即打開輸出資料夾？"):
                self.open_output_folder()
            
        except Exception as e:
            self.log(f"❌ OpenCV 分割失敗: {str(e)}")
            
    def export_report(self):
        """匯出詳細報告"""
        if not self.detection_results:
            messagebox.showwarning("警告", "請先分析影片")
            return
            
        try:
            output_folder = Path(self.output_folder_path.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            report_file = output_folder / "analysis_report.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>影片分析報告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .result-table {{ border-collapse: collapse; width: 100%; }}
                    .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .result-table th {{ background-color: #4CAF50; color: white; }}
                    .success {{ color: green; }}
                    .warning {{ color: orange; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>影片自動分割分析報告</h1>
                    <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>輸入檔案資訊</h2>
                    <p><strong>側面影片:</strong> {Path(self.side_video_path.get()).name}</p>
                    <p><strong>側面檔案路徑:</strong> {self.side_video_path.get()}</p>
                    {f'<p><strong>45度影片:</strong> {Path(self.deg45_video_path.get()).name}</p>' if self.dual_video_mode.get() and self.deg45_video_path.get() else ''}
                    {f'<p><strong>45度檔案路徑:</strong> {self.deg45_video_path.get()}</p>' if self.dual_video_mode.get() and self.deg45_video_path.get() else ''}
                    <p><strong>影片模式:</strong> {'雙影片同步分割' if self.dual_video_mode.get() else '單影片分割'}</p>
                </div>
                
                <div class="section">
                    <h2>分析參數</h2>
                    <p><strong>分割模式:</strong> {'動態分割 (每個片段從一個球進入點到下一個球進入點)' if self.dynamic_mode.get() else '固定長度分割'}</p>
                    <p><strong>片段時長:</strong> {self.segment_duration.get()} 秒 {'(僅固定模式)' if self.dynamic_mode.get() else ''}</p>
                    <p><strong>偵測信心度閾值:</strong> {self.confidence_threshold.get()}</p>
                    <p><strong>最小間隔:</strong> {self.min_interval.get()} 秒</p>
                    <p><strong>預覽開始時間:</strong> {self.preview_start_time.get()} 秒</p>
                    {f'<p><strong>最後片段額外時長:</strong> {self.end_padding.get()} 秒</p>' if self.dynamic_mode.get() else ''}
                </div>
                
                <div class="section">
                    <h2>偵測結果摘要</h2>
                    <p><strong>總影格數:</strong> {len(self.detection_results)}</p>
                    <p><strong>偵測到球的次數:</strong> {len(self.ball_entry_times)}</p>
                    <p><strong>球進入時間點:</strong> {', '.join([f'{t:.2f}s' for t in self.ball_entry_times])}</p>
                </div>
                
                <div class="section">
                    <h2>詳細偵測資料 (前100筆)</h2>
                    <table class="result-table">
                        <tr>
                            <th>影格</th>
                            <th>時間 (秒)</th>
                            <th>偵測到球</th>
                            <th>信心度</th>
                        </tr>
            """
            
            # 只顯示前100筆資料，避免檔案過大
            for result in self.detection_results[:100]:
                detected_text = "是" if result['detected'] else "否"
                html_content += f"""
                        <tr>
                            <td>{result['frame']}</td>
                            <td>{result['time']:.2f}</td>
                            <td class="{'success' if result['detected'] else ''}">{detected_text}</td>
                            <td>{result['confidence']:.3f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.log(f"📋 報告已匯出: {report_file}")
            messagebox.showinfo("完成", f"報告已匯出到:\n{report_file}")
            
        except Exception as e:
            self.log(f"❌ 匯出失敗: {str(e)}")
    
    def log(self, message):
        """記錄訊息到狀態顯示區"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, full_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        print(message)  # 同時輸出到控制台
        
    def run(self):
        """運行應用程式"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoSegmentTester()
    app.run()