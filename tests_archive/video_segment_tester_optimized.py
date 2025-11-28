"""
影片自動分割測試工具 - 優化版本
功能：
1. 匯入影片檔案
2. 使用網球偵測模型找出球進入畫面的時間點
3. 自動分割影片為多個片段
4. 匯出結果並生成報告
"""

import cv2, os, json, subprocess, sys, threading, traceback
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from datetime import datetime

class VideoSegmentTester:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("影片自動分割測試工具")
        self.root.geometry("800x600")
        
        # 初始化變數
        self._init_variables()
        self._setup_ui()
        
    def _init_variables(self):
        """初始化所有變數"""
        # 檔案路徑變數
        self.side_video_path = tk.StringVar()
        self.deg45_video_path = tk.StringVar()
        self.output_folder_path = tk.StringVar(value=str(Path.cwd() / "video_segments_output"))
        
        # 參數變數
        self.segment_duration = tk.DoubleVar(value=4.0)
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.min_interval = tk.DoubleVar(value=2.0)
        self.preview_start_time = tk.DoubleVar(value=-0.5)
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.end_padding = tk.DoubleVar(value=1.0)
        self.dual_video_mode = tk.BooleanVar(value=False)
        self.detection_area = tk.StringVar(value="right_upper_two_thirds")  # 新增：偵測範圍
        self.enable_bounce_filter = tk.BooleanVar(value=True)  # 新增：反彈球過濾
        self.bounce_detection_frames = tk.IntVar(value=15)     # 新增：反彈偵測幀數
        self.enable_exit_detection = tk.BooleanVar(value=True)  # 新增：球出場偵測
        self.exit_timeout = tk.DoubleVar(value=1.5)            # 調整：出場等待時間從2.0改為1.5秒
        
        # 結果變數
        self.detection_results = []
        self.ball_entry_times = []
        self.ball_exit_times = []        # 新增：球出場時間
        self.deg45_ball_entry_times = []
        self.deg45_ball_exit_times = []  # 新增：45度影片球出場時間
        self.tennis_model = None
        
        # GPU加速狀態
        self.gpu_available = self._check_gpu_acceleration()
        self.processing_stats = {
            'total_segments': 0,
            'successful_segments': 0,
            'gpu_accelerated': 0,
            'cpu_fallback': 0
        }
        
    def _setup_ui(self):
        """設置用戶界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        ttk.Label(main_frame, text="影片自動分割測試工具", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 檔案選擇區
        self._create_file_frame(main_frame)
        
        # 參數設定區
        self._create_param_frame(main_frame)
        
        # 控制按鈕
        self._create_control_frame(main_frame)
        
        # 進度條和狀態顯示
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.status_text = tk.Text(main_frame, height=15, width=80)
        self.status_text.grid(row=5, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=5, column=3, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
    def _create_file_frame(self, parent):
        """創建檔案選擇區域"""
        file_frame = ttk.LabelFrame(parent, text="檔案設定", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 雙影片模式開關
        ttk.Checkbutton(file_frame, text="雙影片同步分割模式 (Side + 45度角)", 
                       variable=self.dual_video_mode, command=self._toggle_dual_mode).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # 檔案選擇
        file_configs = [
            ("側面影片 (用於分析):", self.side_video_path, self._browse_side_video, None),
            ("45度影片:", self.deg45_video_path, self._browse_deg45_video, 'disabled'),
            ("輸出資料夾:", self.output_folder_path, self._browse_output_folder, None)
        ]
        
        for i, (label, var, command, state) in enumerate(file_configs, 1):
            ttk.Label(file_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            entry = ttk.Entry(file_frame, textvariable=var, width=50, state=state or 'normal')
            entry.grid(row=i, column=1, padx=(5, 5))
            button = ttk.Button(file_frame, text="瀏覽", command=command, state=state or 'normal')
            button.grid(row=i, column=2)
            
            if i == 2:  # 45度影片
                self.deg45_entry, self.deg45_button = entry, button
                
    def _create_param_frame(self, parent):
        """創建參數設定區域"""
        param_frame = ttk.LabelFrame(parent, text="參數設定", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 配置列權重以避免重疊
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(2, weight=1)
        param_frame.columnconfigure(3, weight=1)
        param_frame.columnconfigure(4, weight=1)
        
        # 參數配置
        params = [
            [("片段時長 (秒):", self.segment_duration, 10), ("偵測信心度:", self.confidence_threshold, 10)],
            [("最小間隔 (秒):", self.min_interval, 10), ("預覽開始時間 (秒):", self.preview_start_time, 10)]
        ]
        
        for row, param_row in enumerate(params):
            for col, (label, var, width) in enumerate(param_row):
                ttk.Label(param_frame, text=label).grid(row=row, column=col*2, sticky=tk.W, padx=(0 if col==0 else 20, 0))
                ttk.Entry(param_frame, textvariable=var, width=width).grid(row=row, column=col*2+1, sticky=tk.W, padx=(5, 0))
        
        # 球進入偵測範圍選擇
        ttk.Label(param_frame, text="球進入偵測範圍:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(15, 5))
        
        # 偵測範圍選項
        detection_options = [
            ("右邊上2/3 (發球機專用)", "right_upper_two_thirds"),
            ("只偵測右邊 (避免地板球干擾)", "right_only"),
            ("只偵測上方", "top_only"),
            ("右邊 + 上方", "right_top"),
            ("全部邊緣 (原始模式)", "all_edges")
        ]
        
        for i, (text, value) in enumerate(detection_options):
            ttk.Radiobutton(param_frame, text=text, variable=self.detection_area, value=value).grid(
                row=3+i//2, column=(i%2)*2, columnspan=2, sticky=tk.W, padx=(20 if i%2==1 else 0, 0))
        
        # 反彈球過濾設定
        ttk.Label(param_frame, text="反彈球過濾:", font=("Arial", 9, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Checkbutton(param_frame, text="啟用反彈球過濾 (避免撞牆反彈重複偵測)", 
                       variable=self.enable_bounce_filter).grid(row=6, column=0, columnspan=3, sticky=tk.W)
        
        ttk.Label(param_frame, text="反彈偵測範圍 (幀數):").grid(row=6, column=3, sticky=tk.W, padx=(20, 0))
        ttk.Entry(param_frame, textvariable=self.bounce_detection_frames, width=8).grid(row=6, column=4, sticky=tk.W, padx=(5, 0))
        
        # 球出場偵測設定
        ttk.Label(param_frame, text="球出場偵測:", font=("Arial", 9, "bold")).grid(row=7, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Checkbutton(param_frame, text="啟用球出場偵測 (片段在球離開後結束)", 
                       variable=self.enable_exit_detection).grid(row=8, column=0, columnspan=3, sticky=tk.W)
        
        ttk.Label(param_frame, text="出場等待時間 (秒):").grid(row=8, column=3, sticky=tk.W, padx=(20, 0))
        exit_entry = ttk.Entry(param_frame, textvariable=self.exit_timeout, width=8)
        exit_entry.grid(row=8, column=4, sticky=tk.W, padx=(5, 0))
        
        # 添加說明提示
        ttk.Label(param_frame, text="(球消失後等待確認出場的時間)", 
                 font=('Arial', 7), foreground='gray').grid(row=9, column=3, columnspan=2, sticky=tk.W, padx=(20, 0))
        
        # 動態分割模式
        ttk.Checkbutton(param_frame, text="動態分割模式 (每個片段從一個球進入點到下一個球進入點)", 
                       variable=self.dynamic_mode).grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        
        ttk.Label(param_frame, text="最後片段額外時長 (秒):").grid(row=10, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(param_frame, textvariable=self.end_padding, width=10).grid(row=10, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
    def _create_control_frame(self, parent):
        """創建控制按鈕區域"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        buttons = [
            ("載入模型", self.load_model),
            ("分析影片", self._analyze_threaded),
            ("預覽結果", self.preview_results),
            ("執行分割", self._segment_threaded),
            ("匯出報告", self.export_report),
            ("打開輸出資料夾", self.open_output_folder)
        ]
        
        for text, command in buttons:
            ttk.Button(control_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        
        # GPU狀態顯示
        gpu_frame = ttk.Frame(parent)
        gpu_frame.grid(row=3, column=0, columnspan=3, pady=(40, 5), sticky=(tk.W, tk.E))
        
        gpu_info = self._get_gpu_info()
        gpu_color = "green" if self.gpu_available else "orange"
        self.gpu_status_label = ttk.Label(gpu_frame, text=gpu_info, foreground=gpu_color)
        self.gpu_status_label.pack(side=tk.LEFT)
        
        # 處理統計
        self.stats_label = ttk.Label(gpu_frame, text="等待處理...")
        self.stats_label.pack(side=tk.RIGHT)
    
    def _browse_side_video(self):
        self._browse_file("選擇側面影片", self.side_video_path, "側面影片")
    
    def _browse_deg45_video(self):
        self._browse_file("選擇45度影片", self.deg45_video_path, "45度影片")
    
    def _browse_output_folder(self):
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_folder_path.set(folder)
            self.log(f"選擇輸出資料夾: {folder}")
    
    def _browse_file(self, title, var, desc):
        """通用檔案瀏覽方法"""
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
            self.log(f"選擇{desc}: {filename}")
    
    def _toggle_dual_mode(self):
        """切換雙影片模式"""
        enabled = self.dual_video_mode.get()
        state = 'normal' if enabled else 'disabled'
        self.deg45_entry.config(state=state)
        self.deg45_button.config(state=state)
        
        if not enabled:
            self.deg45_video_path.set("")
        
        self.log(f"{'已啟用' if enabled else '已停用'}雙影片模式")
    
    def open_output_folder(self):
        """打開輸出資料夾"""
        folder = self.output_folder_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("錯誤", "輸出資料夾不存在")
            return
        
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
            self.log(f"📂 已打開輸出資料夾: {folder}")
        except Exception as e:
            self.log(f"❌ 無法打開資料夾: {e}")
    
    def load_model(self):
        """載入網球偵測模型"""
        try:
            model_path = "model/tennisball_OD_v1.pt"
            if not os.path.exists(model_path):
                messagebox.showerror("錯誤", f"模型檔案不存在: {model_path}")
                return
            
            self.log("正在載入網球偵測模型...")
            self.tennis_model = YOLO(model_path)
            self.log("✅ 模型載入完成！")
        except Exception as e:
            self.log(f"❌ 模型載入失敗: {e}")
            messagebox.showerror("錯誤", f"模型載入失敗: {e}")
    
    def _analyze_threaded(self):
        """在新線程中分析影片"""
        if not self.side_video_path.get() or not self.tennis_model:
            messagebox.showwarning("警告", "請先選擇側面影片並載入模型")
            return
        threading.Thread(target=self.analyze_video, daemon=True).start()
    
    def analyze_video(self):
        """分析影片找出球進入時間點"""
        try:
            # 分析側面影片
            self.ball_entry_times = self._analyze_single_video(self.side_video_path.get(), "側面")
            
            # 分析45度影片（如果啟用）
            if self.dual_video_mode.get() and self.deg45_video_path.get():
                self.deg45_ball_entry_times = self._analyze_single_video(self.deg45_video_path.get(), "45度")
            else:
                self.deg45_ball_entry_times = []
            
            self._display_results()
        except Exception as e:
            self.log(f"❌ 分析失敗: {e}")
            traceback.print_exc()
    
    def _analyze_single_video(self, video_path, video_type):
        """分析單一影片的球進入點"""
        self.log(f"開始分析{video_type}影片: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"❌ 無法開啟{video_type}影片")
            return []
        
        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 保存當前FPS供球追蹤使用
        self.current_fps = fps
        
        self.log(f"{video_type}影片: {total_frames}影格, {fps:.2f}FPS, {frame_width}x{frame_height}")
        self.log(f"🎯 球追蹤距離: {max(200, fps * 8):.0f}像素 (根據{fps:.1f}FPS調整)")
        
        # 邊緣檢測參數
        edge_ratio = 0.15
        edges = {
            'left': frame_width * edge_ratio,
            'right': frame_width * (1 - edge_ratio),
            'top': frame_height * edge_ratio,
            'bottom': frame_height * (1 - edge_ratio)
        }
        
        # 顯示偵測範圍資訊
        detection_mode = self.detection_area.get()
        detection_info = {
            "right_only": f"右邊緣 (X > {edges['right']:.0f})",
            "top_only": f"上邊緣 (Y < {edges['top']:.0f})",
            "right_top": f"右邊緣 (X > {edges['right']:.0f}) 或 上邊緣 (Y < {edges['top']:.0f})",
            "all_edges": f"全邊緣 (左{edges['left']:.0f}, 右{edges['right']:.0f}, 上{edges['top']:.0f}, 下{edges['bottom']:.0f})"
        }
        self.log(f"🎯 偵測範圍: {detection_info.get(detection_mode, '右邊緣')}")
        
        # 初始化變數
        ball_entry_times = []
        ball_exit_times = []    # 球出場時間記錄
        active_balls = {}       # 新增：活躍球追蹤 {ball_id: {'entry_time': float, 'positions': []}}
        next_ball_id = 0        # 新增：下一個球的ID
        prev_detected = False
        prev_position = None
        last_entry_time = -self.min_interval.get()
        last_detected_time = 0  # 最後偵測到球的時間
        position_history = []  # 用於反彈檢測的位置歷史
        
        self.progress['maximum'] = total_frames
        
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            ball_detected, confidence, position = self._detect_ball(frame)
            
            # 更新最後偵測時間（用於出場偵測）
            if ball_detected:
                last_detected_time = current_time
                
                # 更新活躍球的位置追蹤
                self._update_ball_tracking(active_balls, position, current_time)
            
            # 球進入偵測（使用右上2/3範圍）
            is_entry = ball_detected and self._is_in_edge(position, edges)
            
            if is_entry and current_time - last_entry_time >= self.min_interval.get():
                if not self._is_bounce_ball(position_history, current_time, ball_entry_times):
                    # 新球進入，建立追蹤
                    ball_id = next_ball_id
                    next_ball_id += 1
                    
                    active_balls[ball_id] = {
                        'entry_time': current_time,
                        'positions': [position],
                        'last_seen': current_time
                    }
                    
                    ball_entry_times.append(current_time)
                    last_entry_time = current_time
                    self.log(f"🏥 {video_type}球進入: {current_time:.1f}s - 開始追蹤球#{ball_id}")
                else:
                    self.log(f"🔄 {video_type}反彈球過濾: {current_time:.1f}s - 已忽略撞牆反彈")
            
            # 球出場偵測（使用整個右邊範圍）
            if self.enable_exit_detection.get():
                exited_balls = self._check_ball_exits(active_balls, edges, current_time)
                for ball_id, exit_time in exited_balls:
                    ball_exit_times.append(exit_time)
                    self.log(f"🚪 {video_type}球出場: {exit_time:.1f}s - 球#{ball_id}由中央向右邊移動")
            
            # 更新位置歷史（用於反彈檢測）
            position_history.append(position if ball_detected else None)
            if len(position_history) > self.bounce_detection_frames.get() * 2:  # 保持合理的歷史長度
                position_history.pop(0)
            
            # 記錄檢測結果（僅用於預覽）
            if video_type == "側面":
                in_edge = self._is_in_edge(position, edges) if position else False
                self.detection_results.append({
                    'frame': frame_count,
                    'time': current_time,
                    'detected': ball_detected,
                    'confidence': confidence,
                    'position': position,
                    'in_edge': in_edge
                })
            
            # 判斷球進入
            if self._is_ball_entry(ball_detected, prev_detected, position, prev_position, edges):
                if current_time - last_entry_time >= self.min_interval.get():
                    # 檢查是否為反彈球
                    is_bounce = self._is_bounce_ball(position_history, current_time, ball_entry_times)
                    
                    if not is_bounce:
                        ball_entry_times.append(current_time)
                        last_entry_time = current_time
                        reason = self._get_entry_reason(position, prev_position, edges)
                        self.log(f"🎾 {video_type}球進入: {current_time:.2f}s - {reason} (信心度: {confidence:.3f})")
                    else:
                        self.log(f"🔄 {video_type}反彈球過濾: {current_time:.2f}s - 已忽略撞牆反彈")
            
            prev_detected = ball_detected
            prev_position = position
            
            if frame_count % 30 == 0:
                self.progress['value'] = frame_count
                self.root.update_idletasks()
        
        cap.release()
        self.progress['value'] = total_frames
        
        # 儲存出場時間到實例變數
        if video_type == "側面":
            self.ball_exit_times = ball_exit_times
        else:
            self.deg45_ball_exit_times = ball_exit_times
        
        self.log(f"✅ {video_type}分析完成！找到 {len(ball_entry_times)} 個球進入點, {len(ball_exit_times)} 個出場點")
        return ball_entry_times
    
    def _detect_ball(self, frame):
        """偵測畫面中的網球"""
        results = self.tennis_model(frame, verbose=False)
        
        if not results[0].boxes:
            return False, 0, None
        
        best_box = max(results[0].boxes, key=lambda box: float(box.conf[0]))
        confidence = float(best_box.conf[0])
        
        if confidence < self.confidence_threshold.get():
            return False, confidence, None
        
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        position = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return True, confidence, position
    
    def _is_in_edge(self, position, edges):
        """檢查位置是否在指定的邊緣區域"""
        if not position:
            return False
        
        x, y = position
        detection_mode = self.detection_area.get()
        
        if detection_mode == "right_upper_two_thirds":
            # 只檢查右邊緣的上2/3區域（發球機專用）
            if x > edges['right']:
                # 計算右邊緣的上2/3範圍
                # 將右邊緣高度分成3段，只偵測上面2段
                height_range = edges['bottom'] - edges['top']
                upper_two_thirds_bottom = edges['top'] + (height_range * 2 / 3)
                return y <= upper_two_thirds_bottom
            return False
        
        elif detection_mode == "right_only":
            # 只檢查右邊緣
            return x > edges['right']
        
        elif detection_mode == "top_only":
            # 只檢查上邊緣
            return y < edges['top']
        
        elif detection_mode == "right_top":
            # 檢查右邊緣或上邊緣
            return x > edges['right'] or y < edges['top']
        
        elif detection_mode == "all_edges":
            # 原始模式：檢查所有邊緣
            return (x < edges['left'] or x > edges['right'] or 
                    y < edges['top'] or y > edges['bottom'])
        
        # 預設使用右上2/3模式
        if x > edges['right']:
            height_range = edges['bottom'] - edges['top']
            upper_two_thirds_bottom = edges['top'] + (height_range * 2 / 3)
            return y <= upper_two_thirds_bottom
        return False
    
    def _update_ball_tracking(self, active_balls, position, current_time):
        """更新球追蹤資訊"""
        if not position:
            return
        
        # 動態調整追蹤距離，根據FPS調整
        fps = getattr(self, 'current_fps', 30)  # 使用當前FPS
        max_tracking_distance = max(200, fps * 8)  # 高FPS需要更大的追蹤距離
        
        # 找到最近的球進行位置更新
        min_distance = float('inf')
        closest_ball_id = None
        
        for ball_id, ball_info in active_balls.items():
            if ball_info['positions']:
                last_pos = ball_info['positions'][-1]
                distance = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
                
                if distance < min_distance and distance < max_tracking_distance:
                    min_distance = distance
                    closest_ball_id = ball_id
        
        # 更新最近球的位置
        if closest_ball_id is not None:
            active_balls[closest_ball_id]['positions'].append(position)
            active_balls[closest_ball_id]['last_seen'] = current_time
            
            # 保持位置歷史在合理範圍內
            if len(active_balls[closest_ball_id]['positions']) > 30:
                active_balls[closest_ball_id]['positions'].pop(0)
    
    def _check_ball_exits(self, active_balls, edges, current_time):
        """檢查球是否出場（使用整個右邊範圍）"""
        exited_balls = []
        balls_to_remove = []
        
        for ball_id, ball_info in active_balls.items():
            time_since_last_seen = current_time - ball_info['last_seen']
            
            # 使用GUI設定的出場等待時間
            exit_timeout = self.exit_timeout.get()
            min_check_time = max(0.2, exit_timeout * 0.15)  # 最短檢查時間為設定值的15%
            max_force_time = max(1.0, exit_timeout)         # 強制移除時間為設定值或1秒
            
            if time_since_last_seen > min_check_time:
                
                # 檢查是否為真正的右邊出場
                is_exit, reason = self._is_ball_exit_right_edge(ball_info['positions'], edges)
                
                # 增加詳細調試輸出
                if time_since_last_seen > min_check_time and time_since_last_seen < max_force_time:
                    last_pos = ball_info['positions'][-1] if ball_info['positions'] else None
                    self.log(f"🔍 球 {ball_id} 檢查中: 未見{time_since_last_seen:.2f}s, 最後位置{last_pos}, 出場判斷: {is_exit} - {reason}")
                
                if is_exit:
                    exited_balls.append((ball_id, ball_info['last_seen']))
                    self.log(f"🚪 球 {ball_id} 出場: {reason} (未見時間: {time_since_last_seen:.2f}s)")
                elif time_since_last_seen > max_force_time:
                    # 超時前顯示最後的軌跡分析
                    last_positions = ball_info['positions'][-5:] if len(ball_info['positions']) >= 5 else ball_info['positions']
                    self.log(f"⏰ 球 {ball_id} 超時移除 - 最後5個位置: {last_positions}")
                    self.log(f"   未見時間: {time_since_last_seen:.2f}s, 閾值: {max_force_time:.1f}s, 最終判斷: {reason}")
                
                if is_exit or time_since_last_seen > max_force_time:
                    balls_to_remove.append(ball_id)
        
        # 移除已出場或過期的球
        for ball_id in balls_to_remove:
            del active_balls[ball_id]
        
        return exited_balls
    
    def _is_ball_exit_right_edge(self, positions, edges):
        """檢查是否為右邊出場（靈活偵測邏輯）"""
        if len(positions) < 2:
            return False, "軌跡點不足"
        
        # 分析最近的軌跡
        recent_positions = positions[-min(8, len(positions)):]
        
        # 檢查最終位置是否在右邊範圍
        end_pos = recent_positions[-1]
        right_boundary = edges['right'] - 100  # 右邊界緩衝區
        
        is_at_right_edge = end_pos[0] > right_boundary
        
        if not is_at_right_edge:
            return False, f"未到達右邊界 (X: {end_pos[0]:.0f}, 邊界: {right_boundary:.0f})"
        
        # 分析移動趨勢
        movement_analysis = self._analyze_movement_trend(recent_positions, edges)
        
        # 多種出場情況判斷
        exit_reasons = []
        
        # 1. 向右移動出場（最常見）
        if movement_analysis['moving_right']:
            exit_reasons.append(f"向右移動 (ΔX: {movement_analysis['x_trend']:.0f})")
        
        # 2. 從中央區域出場
        if movement_analysis['from_center']:
            exit_reasons.append("從中央區域出場")
        
        # 3. 持續在右邊緣移動
        if movement_analysis['consistently_right']:
            exit_reasons.append("持續在右邊緣")
        
        # 4. 任何明顯的向外移動
        if movement_analysis['moving_outward']:
            exit_reasons.append("向邊緣移動")
        
        # 5. 新增：接近右邊界且有移動（放寬條件）
        if is_at_right_edge and len(recent_positions) >= 2:
            x_movement = abs(recent_positions[-1][0] - recent_positions[0][0])
            if x_movement > 5:  # 任何小幅移動
                exit_reasons.append(f"右邊界移動 (ΔX: {x_movement:.0f})")
        
        # 6. 新增：在右邊界停留一段時間
        right_edge_count = sum(1 for pos in recent_positions if pos[0] > right_boundary)
        if right_edge_count >= len(recent_positions) * 0.7:  # 70%時間在右邊界
            exit_reasons.append(f"右邊界停留 ({right_edge_count}/{len(recent_positions)})")
        
        # 判斷出場
        is_exit = len(exit_reasons) > 0
        reason = "; ".join(exit_reasons) if exit_reasons else "無明確出場跡象"
        
        return is_exit, reason
    
    def _analyze_movement_trend(self, positions, edges):
        """分析球的移動趨勢"""
        if len(positions) < 2:
            return {'moving_right': False, 'from_center': False, 
                   'consistently_right': False, 'moving_outward': False, 'x_trend': 0}
        
        # 計算畫面區域
        width = edges['right'] - edges['left']
        center_x_min = edges['left'] + width * 0.25
        center_x_max = edges['right'] - width * 0.25
        right_zone = edges['right'] - width * 0.3
        
        # X方向移動趨勢
        x_start = positions[0][0]
        x_end = positions[-1][0]
        x_trend = x_end - x_start
        
        # 檢查是否從中央開始
        from_center = center_x_min <= x_start <= center_x_max
        
        # 檢查是否向右移動
        moving_right = x_trend > 10  # 進一步降低閾值從15到10
        
        # 檢查是否持續在右邊
        consistently_right = all(pos[0] > right_zone for pos in positions[-min(3, len(positions)):])
        
        # 檢查是否向外移動（任何方向）- 放寬條件
        moving_outward = (abs(x_trend) > 15 or      # 降低總移動閾值
                         moving_right or 
                         consistently_right or
                         x_trend > 8)              # 新增：小幅向右移動也算
        
        return {
            'moving_right': moving_right,
            'from_center': from_center,
            'consistently_right': consistently_right,
            'moving_outward': moving_outward,
            'x_trend': x_trend
        }
    
    def _is_ball_exit(self, position_history, edges, last_detected_time, current_time):
        """檢查是否為真正的球出場（由中央移向邊緣）"""
        
        # 檢查時間間隔
        if current_time - last_detected_time < 0.3:  # 至少 0.3秒未偵測到
            return False
        
        # 獲取最近的有效位置歷史
        valid_positions = [pos for pos in position_history if pos is not None]
        
        if len(valid_positions) < 3:  # 需要至少 3 個位置點來判斷趋勢
            return False
        
        # 分析最近 5-10 個位置的移動趋勢
        recent_positions = valid_positions[-min(10, len(valid_positions)):]
        
        # 計算畫面中心區域
        center_x_min = edges['left'] + (edges['right'] - edges['left']) * 0.3
        center_x_max = edges['right'] - (edges['right'] - edges['left']) * 0.3
        center_y_min = edges['top'] + (edges['bottom'] - edges['top']) * 0.3
        center_y_max = edges['bottom'] - (edges['bottom'] - edges['top']) * 0.3
        
        # 檢查是否從中心區域開始移動
        start_in_center = False
        end_near_edge = False
        
        # 檢查起始位置是否在中心區域
        if len(recent_positions) >= 3:
            start_pos = recent_positions[0]
            if (center_x_min <= start_pos[0] <= center_x_max and 
                center_y_min <= start_pos[1] <= center_y_max):
                start_in_center = True
        
        # 檢查結束位置是否接近邊緣（特別是右邊）
        end_pos = recent_positions[-1]
        
        # 右邊出場（最常見）
        if end_pos[0] > edges['right'] - 100:  # 接近右邊緣
            end_near_edge = True
        
        # 其他邊緣出場
        elif (end_pos[0] < edges['left'] + 100 or    # 左邊
              end_pos[1] < edges['top'] + 100 or     # 上邊  
              end_pos[1] > edges['bottom'] - 100):   # 下邊
            end_near_edge = True
        
        # 檢查移動方向趋勢
        direction_towards_edge = False
        
        if len(recent_positions) >= 3:
            # 計算最後幾個位置的移動方向
            last_3_positions = recent_positions[-3:]
            
            # X 方向趋勢（向右移動為正）
            x_trend = last_3_positions[-1][0] - last_3_positions[0][0]
            
            # Y 方向趋勢（向下移動為正）
            y_trend = last_3_positions[-1][1] - last_3_positions[0][1]
            
            # 判斷是否向邊緣移動
            if (x_trend > 50 or      # 向右移動
                x_trend < -50 or     # 向左移動  
                y_trend < -50 or     # 向上移動
                y_trend > 50):       # 向下移動
                direction_towards_edge = True
        
        # 綜合判斷：從中心開始 + 向邊緣移動 + 結束於邊緣附近
        is_exit = (start_in_center or direction_towards_edge) and end_near_edge
        
        return is_exit
    
    def _is_bounce_ball(self, position_history, current_time, entry_times):
        """檢測是否為反彈球"""
        if not self.enable_bounce_filter.get():
            return False
            
        if len(position_history) < self.bounce_detection_frames.get():
            return False
            
        # 檢查最近是否有球進入記錄（避免真正的新球被誤判）
        recent_entries = [t for t in entry_times if current_time - t <= 3.0]
        if len(recent_entries) == 0:
            return False  # 沒有最近的進入記錄，可能是真正的新球
            
        # 分析最近的軌跡
        recent_positions = position_history[-self.bounce_detection_frames.get():]
        
        # 計算移動方向變化
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(recent_positions)):
            if recent_positions[i] and recent_positions[i-1]:
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                
                if abs(dx) > 5 or abs(dy) > 5:  # 有明顯移動
                    current_direction = (1 if dx > 0 else -1, 1 if dy > 0 else -1)
                    
                    if prev_direction and current_direction != prev_direction:
                        direction_changes += 1
                    
                    prev_direction = current_direction
        
        # 如果方向變化過多，可能是反彈球
        if direction_changes >= 2:
            return True
            
        # 檢查速度變化（反彈球通常會有急劇的速度變化）
        speeds = []
        for i in range(1, len(recent_positions)):
            if recent_positions[i] and recent_positions[i-1]:
                distance = ((recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                           (recent_positions[i][1] - recent_positions[i-1][1])**2)**0.5
                speeds.append(distance)
        
        if len(speeds) >= 3:
            # 檢查是否有急劇的速度變化
            speed_changes = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
            avg_speed_change = sum(speed_changes) / len(speed_changes) if speed_changes else 0
            
            # 如果平均速度變化很大，可能是反彈
            if avg_speed_change > 10:
                return True
                
        return False
    
    def _is_ball_entry(self, current_detected, prev_detected, current_pos, prev_pos, edges):
        """判斷是否為球進入畫面"""
        if current_detected and not prev_detected:
            return self._is_in_edge(current_pos, edges)
        
        if current_detected and prev_detected and prev_pos and current_pos:
            prev_in_edge = self._is_in_edge(prev_pos, edges)
            curr_in_edge = self._is_in_edge(current_pos, edges)
            
            if prev_in_edge and not curr_in_edge:
                distance = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)**0.5
                return distance > 20
        
        return False
    
    def _get_entry_reason(self, position, prev_position, edges):
        """獲取進入原因描述"""
        if not position:
            return "偵測到球進入"
        
        detection_mode = self.detection_area.get()
        x, y = position
        
        if detection_mode == "right_only":
            return f"右邊緣進入 (X: {x:.0f})"
        elif detection_mode == "top_only":
            return f"上邊緣進入 (Y: {y:.0f})"
        elif detection_mode == "right_top":
            if x > edges['right']:
                return f"右邊緣進入 (X: {x:.0f})"
            else:
                return f"上邊緣進入 (Y: {y:.0f})"
        else:
            if not prev_position:
                return f"邊緣進入 (位置: {x:.0f}, {y:.0f})"
            return "從邊緣移向中央"
    
    def _display_results(self):
        """顯示分析結果"""
        if self.dual_video_mode.get():
            self.log(f"📊 分析結果: 側面{len(self.ball_entry_times)}個, 45度{len(self.deg45_ball_entry_times)}個球進入點")
            if len(self.ball_entry_times) != len(self.deg45_ball_entry_times):
                self.log("⚠️ 兩個角度檢測到的球進入點數量不同")
        else:
            self.log(f"📊 側面影片分析完成: {len(self.ball_entry_times)} 個球進入點")
    
    def preview_results(self):
        """預覽分析結果"""
        if not self.detection_results:
            messagebox.showwarning("警告", "請先分析影片")
            return
        
        try:
            self._create_preview_charts()
        except Exception as e:
            self.log(f"❌ 預覽失敗: {e}")
    
    def _create_preview_charts(self):
        """創建預覽圖表"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        times = [r['time'] for r in self.detection_results]
        confidences = [r['confidence'] for r in self.detection_results]
        
        # 圖1: 信心度時間序列
        ax1.plot(times, confidences, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=self.confidence_threshold.get(), color='r', linestyle='--', label=f'信心度閾值')
        
        for entry_time in self.ball_entry_times:
            ax1.axvline(x=entry_time, color='g', linestyle='-', alpha=0.8, linewidth=2)
            ax1.text(entry_time, ax1.get_ylim()[1]*0.9, f'{entry_time:.1f}s', rotation=90, ha='right', va='top')
        
        ax1.set_xlabel('時間 (秒)')
        ax1.set_ylabel('偵測信心度')
        ax1.set_title('網球偵測信心度 vs 時間')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 圖2: 偵測狀態
        detected_states = [1 if r['detected'] else 0 for r in self.detection_results]
        edge_states = [0.5 if r.get('in_edge', False) else 0 for r in self.detection_results]
        
        ax2.fill_between(times, detected_states, alpha=0.6, color='orange', label='偵測到球')
        ax2.fill_between(times, edge_states, alpha=0.4, color='purple', label='球在邊緣')
        
        # 標記分割區間
        for i, entry_time in enumerate(self.ball_entry_times):
            start_time, duration = self._calculate_segment_time(i, entry_time)
            color = 'blue' if self.dynamic_mode.get() else 'red'
            ax2.axvspan(start_time, start_time + duration, alpha=0.3, color=color)
            ax2.text(start_time + duration/2, 0.75, f'片段{i+1}\n({duration:.1f}s)', ha='center', va='center')
        
        ax2.set_title('偵測狀態與分割區間')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 圖3: 位置軌跡
        self._plot_trajectory(ax3)
        
        plt.tight_layout()
        plt.show()
        self.log("📊 預覽圖表已顯示")
    
    def _calculate_segment_time(self, index, entry_time):
        """計算片段時間"""
        start_time = max(0, entry_time + self.preview_start_time.get())
        
        if self.dynamic_mode.get():
            # 動態模式：嘗試使用球出場時間
            if (self.enable_exit_detection.get() and 
                hasattr(self, 'ball_exit_times') and 
                index < len(self.ball_exit_times)):
                
                # 使用對應的出場時間 + 小量緩衝時間（0.3秒）
                exit_time = self.ball_exit_times[index]
                end_time = exit_time + 0.3  # 固定0.3秒緩衝，不使用出場等待時間
                self.log(f"🎬 片段{index+1}: 進入{entry_time:.1f}s → 出場{exit_time:.1f}s → 結束{end_time:.1f}s")
                
            else:
                # 傳統動態模式：到下一個進入點
                if index < len(self.ball_entry_times) - 1:
                    end_time = self.ball_entry_times[index + 1] + self.preview_start_time.get()
                else:
                    end_time = entry_time + 4.0 + self.end_padding.get()
            
            duration = max(1.0, end_time - start_time)
        else:
            duration = self.segment_duration.get()
        
        return start_time, duration
    
    def _plot_trajectory(self, ax):
        """繪製球的位置軌跡"""
        positions = [r for r in self.detection_results if r['detected'] and r.get('position')]
        
        if not positions:
            ax.text(0.5, 0.5, '無位置資訊', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('球的位置軌跡')
            return
        
        x_pos = [r['position'][0] for r in positions]
        y_pos = [r['position'][1] for r in positions]
        times = [r['time'] for r in positions]
        
        scatter = ax.scatter(x_pos, y_pos, c=times, cmap='viridis', alpha=0.6, s=20)
        
        # 標記進入點
        for entry_time in self.ball_entry_times:
            closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - entry_time))
            if abs(times[closest_idx] - entry_time) < 0.5:
                ax.scatter(x_pos[closest_idx], y_pos[closest_idx], color='red', s=100, marker='*')
        
        ax.set_xlabel('X 位置')
        ax.set_ylabel('Y 位置')
        ax.set_title('球的位置軌跡')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='時間 (秒)')
    
    def _segment_threaded(self):
        """在新線程中執行分割"""
        if not self.ball_entry_times:
            messagebox.showwarning("警告", "請先分析影片")
            return
        if not self.output_folder_path.get():
            messagebox.showwarning("警告", "請設定輸出資料夾")
            return
        threading.Thread(target=self.execute_segmentation, daemon=True).start()
    
    def execute_segmentation(self):
        """執行影片分割"""
        try:
            # 重置處理統計
            self.processing_stats = {
                'total_segments': 0,
                'successful_segments': 0,
                'gpu_accelerated': 0,
                'cpu_fallback': 0
            }
            self._update_stats_display()
            
            base_output_folder = Path(self.output_folder_path.get())
            base_output_folder.mkdir(parents=True, exist_ok=True)
            
            # 檢查FFmpeg可用性，如果沒有則嘗試安裝
            use_ffmpeg = self._check_ffmpeg()
            if not use_ffmpeg:
                self.log("⚠️  未檢測到 FFmpeg，正在嘗試自動安裝...")
                if self._install_ffmpeg():
                    use_ffmpeg = self._check_ffmpeg()
                else:
                    self.log("❌ FFmpeg 安裝失敗，將使用 OpenCV (較慢)")
            
            method = "FFmpeg + GPU加速" if use_ffmpeg else "OpenCV (CPU)"
            self.log(f"🎬 使用 {method} 進行分割")
            
            if self.dual_video_mode.get() and self.deg45_video_path.get():
                self._segment_dual_videos(base_output_folder, use_ffmpeg)
            else:
                # 單影片模式：為影片創建專屬資料夾
                video_name = Path(self.side_video_path.get()).stem
                video_folder = self._create_unique_folder(base_output_folder, video_name)
                self.log(f"📁 影片輸出資料夾: {video_folder.name}")
                
                self._segment_single_video(self.side_video_path.get(), video_folder, use_ffmpeg, "側面")
                
                # 顯示完成資訊
                video_files = list(video_folder.glob(f"{video_name}_segment_*.mp4"))
                self.log("="*50)
                self.log(f"🎬 影片分割完成！")
                self.log(f"📁 輸出資料夾: {video_folder}")
                self.log(f"   - 產生 {len(video_files)} 個片段")
                for video_file in sorted(video_files):
                    file_size = os.path.getsize(video_file) / (1024*1024)
                    self.log(f"   ✅ {video_file.name} ({file_size:.1f} MB)")
                self.log("="*50)
            
        except Exception as e:
            self.log(f"❌ 分割失敗: {e}")
            traceback.print_exc()
    
    def _check_ffmpeg(self):
        """檢查FFmpeg可用性"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return True
        except:
            # 檢查本地tools資料夾是否有ffmpeg
            local_ffmpeg = Path("tools/ffmpeg.exe")
            if local_ffmpeg.exists():
                return True
            return False
    
    def _check_gpu_acceleration(self):
        """檢查GPU加速可用性"""
        try:
            ffmpeg_cmd = self._get_ffmpeg_command()
            if not ffmpeg_cmd:
                return False
                
            # 檢查FFmpeg是否支援NVIDIA編碼器
            result = subprocess.run([ffmpeg_cmd, '-encoders'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'h264_nvenc' in output:
                    # 進一步檢查GPU是否真的可用
                    try:
                        gpu_test = subprocess.run(['nvidia-smi'], 
                                                capture_output=True, timeout=5)
                        return gpu_test.returncode == 0
                    except:
                        return False
            return False
        except:
            return False
    
    def _get_gpu_info(self):
        """獲取GPU資訊"""
        if not self.gpu_available:
            return "GPU加速: 不可用"
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 2:
                    name = gpu_info[0].strip()
                    memory = gpu_info[1].strip()
                    return f"GPU加速: {name} ({memory}MB VRAM)"
            return "GPU加速: NVIDIA GPU已檢測"
        except:
            return "GPU加速: 可用但無法獲取詳細資訊"
    
    def _install_ffmpeg(self):
        """自動安裝FFmpeg"""
        try:
            import requests
            import zipfile
            
            self.log("🔄 正在自動下載和安裝 FFmpeg...")
            self.log("⚠️  首次安裝需要下載約100MB，請稍等...")
            
            # 創建tools資料夾
            tools_dir = Path("tools")
            tools_dir.mkdir(exist_ok=True)
            
            # FFmpeg下載URL (Windows essentials版本)
            ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            
            # 下載FFmpeg
            self.log("📥 正在下載 FFmpeg...")
            response = requests.get(ffmpeg_url, stream=True)
            response.raise_for_status()
            
            zip_path = tools_dir / "ffmpeg.zip"
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        self.log(f"📥 下載進度: {progress:.1f}%")
            
            # 解壓縮
            self.log("📦 正在解壓縮 FFmpeg...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tools_dir)
            
            # 找到ffmpeg.exe並複製到tools根目錄
            ffmpeg_found = False
            for item in tools_dir.glob("ffmpeg-*"):
                if item.is_dir():
                    ffmpeg_exe = item / "bin" / "ffmpeg.exe"
                    if ffmpeg_exe.exists():
                        target_path = tools_dir / "ffmpeg.exe"
                        import shutil
                        shutil.copy2(ffmpeg_exe, target_path)
                        ffmpeg_found = True
                        break
            
            # 清理下載檔案
            zip_path.unlink(missing_ok=True)
            
            if ffmpeg_found:
                self.log("✅ FFmpeg 安裝成功！")
                return True
            else:
                self.log("❌ FFmpeg 安裝失敗：找不到執行檔")
                return False
                
        except Exception as e:
            self.log(f"❌ FFmpeg 自動安裝失敗: {e}")
            return False
    
    def _create_unique_folder(self, base_path, video_name):
        """創建唯一的資料夾名稱"""
        base_folder = base_path / video_name
        
        # 如果資料夾不存在，直接使用原名稱
        if not base_folder.exists():
            base_folder.mkdir(parents=True, exist_ok=True)
            return base_folder
        
        # 如果存在，則添加編號
        counter = 1
        while True:
            numbered_folder = base_path / f"{video_name}_{counter}"
            if not numbered_folder.exists():
                numbered_folder.mkdir(parents=True, exist_ok=True)
                return numbered_folder
            counter += 1
    
    def _segment_dual_videos(self, output_folder, use_ffmpeg):
        """分割雙影片"""
        self.log("🎬 雙影片模式分割")
        
        # 為側面影片創建專屬資料夾
        side_video_name = Path(self.side_video_path.get()).stem
        side_folder = self._create_unique_folder(output_folder, f"{side_video_name}_側面")
        self.log(f"📁 側面影片輸出資料夾: {side_folder.name}")
        
        # 為45度影片創建專屬資料夾
        deg45_video_name = Path(self.deg45_video_path.get()).stem
        deg45_folder = self._create_unique_folder(output_folder, f"{deg45_video_name}_45度")
        self.log(f"📁 45度影片輸出資料夾: {deg45_folder.name}")
        
        # 分別分割到各自的資料夾
        self._segment_single_video(self.side_video_path.get(), side_folder, use_ffmpeg, "側面")
        self._segment_single_video(self.deg45_video_path.get(), deg45_folder, use_ffmpeg, "45度")
        
        # 統計結果
        side_files = list(side_folder.glob(f"{side_video_name}_segment_*.mp4"))
        deg45_files = list(deg45_folder.glob(f"{deg45_video_name}_segment_*.mp4"))
        
        self.log("="*60)
        self.log("🎬 雙影片分割完成！")
        self.log(f"📁 側面影片資料夾: {side_folder}")
        self.log(f"   - 產生 {len(side_files)} 個片段")
        self.log(f"📁 45度影片資料夾: {deg45_folder}")
        self.log(f"   - 產生 {len(deg45_files)} 個片段")
        self.log("="*60)
        
        if messagebox.askyesno("完成", f"雙影片分割完成！\n\n側面影片: {len(side_files)} 個片段\n45度影片: {len(deg45_files)} 個片段\n\n各影片已分別放入專屬資料夾\n\n打開主輸出資料夾？"):
            self.open_output_folder()
    
    def _segment_single_video(self, video_path, output_folder, use_ffmpeg, video_type):
        """分割單一影片"""
        ball_times = self.ball_entry_times if video_type == "側面" else (self.deg45_ball_entry_times or self.ball_entry_times)
        
        if not ball_times:
            self.log(f"❌ {video_type}影片無球進入點數據")
            return
        
        self.progress['maximum'] = len(ball_times)
        segments = []
        
        for i, entry_time in enumerate(ball_times):
            start_time, duration = self._calculate_segment_time(i, entry_time)
            
            output_name = f"{Path(video_path).stem}_segment_{i+1:02d}.mp4"
            output_path = output_folder / output_name
            
            success = self._segment_video_clip(video_path, output_path, start_time, duration, use_ffmpeg)
            
            # 更新統計
            self.processing_stats['total_segments'] += 1
            if success:
                self.processing_stats['successful_segments'] += 1
            
            if success and output_path.exists():
                file_size = os.path.getsize(output_path) / (1024*1024)
                segments.append({'segment_id': i+1, 'success': True, 'file_size_mb': round(file_size, 2)})
                self.log(f"✅ {video_type}片段 {i+1} 完成 ({file_size:.1f} MB)")
            else:
                segments.append({'segment_id': i+1, 'success': False})
                self.log(f"❌ {video_type}片段 {i+1} 失敗")
            
            # 更新統計顯示
            self._update_stats_display()
            
            self.progress['value'] = i + 1
            self.root.update_idletasks()
        
        # 保存分割資訊
        self._save_segmentation_info(output_folder, video_path, video_type, ball_times, segments)
        
        successful = sum(1 for s in segments if s['success'])
        self.log(f"🎬 {video_type}分割完成！成功: {successful}/{len(segments)} 個片段")
        
        # 只在非雙影片模式下顯示完成對話框（雙影片模式會在上層統一顯示）
        if not self.dual_video_mode.get():
            if messagebox.askyesno("完成", f"{video_type}影片分割完成！\n成功: {successful} 個片段\n\n影片已放入專屬資料夾\n\n打開輸出資料夾？"):
                # 打開影片專屬資料夾，而不是主輸出資料夾
                try:
                    if sys.platform == "win32":
                        os.startfile(str(output_folder))
                    elif sys.platform == "darwin":
                        subprocess.run(["open", str(output_folder)])
                    else:
                        subprocess.run(["xdg-open", str(output_folder)])
                    self.log(f"📂 已打開影片資料夾: {output_folder}")
                except Exception as e:
                    self.log(f"❌ 無法打開資料夾: {e}")
                    # 備選方案：打開主輸出資料夾
                    self.open_output_folder()
    
    def _segment_video_clip(self, input_path, output_path, start_time, duration, use_ffmpeg):
        """分割影片片段"""
        if use_ffmpeg:
            return self._segment_with_ffmpeg(input_path, output_path, start_time, duration)
        else:
            return self._segment_with_opencv(input_path, output_path, start_time, duration)
    
    def _get_ffmpeg_command(self):
        """獲取FFmpeg命令路徑"""
        # 先檢查系統是否有ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return 'ffmpeg'
        except:
            # 檢查本地tools資料夾
            local_ffmpeg = Path("tools/ffmpeg.exe")
            if local_ffmpeg.exists():
                return str(local_ffmpeg.absolute())
            return None
    
    def _segment_with_ffmpeg(self, input_path, output_path, start_time, duration):
        """使用FFmpeg分割（支援RTX 3060 GPU加速）"""
        try:
            ffmpeg_cmd = self._get_ffmpeg_command()
            if not ffmpeg_cmd:
                return False
            
            # 首先嘗試GPU加速模式 (針對RTX 3060優化)
            gpu_cmd = [
                ffmpeg_cmd,
                '-hwaccel', 'cuda',              # CUDA硬體加速
                '-hwaccel_output_format', 'cuda', # GPU記憶體格式
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'h264_nvenc',            # NVIDIA硬體編碼器
                '-preset', 'p4',                 # RTX 3060最佳預設
                '-tune', 'hq',                   # 高品質模式
                '-rc', 'vbr',                    # 可變位元率
                '-cq', '19',                     # 品質設定 (低數值=高品質)
                '-b:v', '0',                     # 讓CQ控制位元率
                '-maxrate:v', '20M',             # 最大位元率
                '-bufsize:v', '40M',             # 緩衝區大小
                '-c:a', 'aac',                   # 音訊編碼
                '-b:a', '192k',                  # 音訊位元率
                '-avoid_negative_ts', 'make_zero',
                '-movflags', '+faststart',       # 優化串流
                str(output_path),
                '-y'
            ]
            
            self.log(f"🚀 使用RTX 3060 GPU加速分割: {Path(output_path).name}")
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log(f"✅ GPU加速分割成功")
                self.processing_stats['gpu_accelerated'] += 1
                return True
            else:
                self.log(f"⚠️  GPU加速失敗，錯誤: {result.stderr[:200]}...")
                self.log("🔄 回退到CPU快速複製模式")
                
                # CPU備用模式 - 快速複製
                cpu_cmd = [
                    ffmpeg_cmd,
                    '-i', input_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',                # 直接複製，最快速度
                    '-avoid_negative_ts', 'make_zero',
                    str(output_path),
                    '-y'
                ]
                result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    self.log(f"✅ CPU複製模式分割成功")
                    self.processing_stats['cpu_fallback'] += 1
                    return True
                else:
                    self.log(f"❌ CPU分割也失敗: {result.stderr[:200]}...")
                    return False
            
            return result.returncode == 0
        except Exception as e:
            self.log(f"❌ FFmpeg分割失敗: {e}")
            return False
    
    def _update_stats_display(self):
        """更新統計顯示"""
        try:
            stats = self.processing_stats
            if stats['total_segments'] > 0:
                success_rate = (stats['successful_segments'] / stats['total_segments']) * 100
                gpu_rate = (stats['gpu_accelerated'] / stats['total_segments']) * 100 if stats['total_segments'] > 0 else 0
                
                stats_text = (f"處理統計: {stats['successful_segments']}/{stats['total_segments']} 成功 "
                             f"({success_rate:.1f}%) | GPU加速: {gpu_rate:.1f}% | "
                             f"CPU備用: {stats['cpu_fallback']}")
                
                if hasattr(self, 'stats_label'):
                    self.stats_label.config(text=stats_text)
        except Exception as e:
            pass  # 忽略統計更新錯誤
    
    def _segment_with_opencv(self, input_path, output_path, start_time, duration):
        """使用OpenCV分割（優化版本）"""
        try:
            # 嘗試使用GPU加速的OpenCV後端
            cap = cv2.VideoCapture(input_path)
            
            # 嘗試設定GPU後端
            try:
                cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_DSHOW)  # Windows優化
            except:
                pass
            
            if not cap.isOpened():
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 嘗試使用硬體編碼器
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # 硬體友好格式
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 嘗試硬體編碼，失敗則回退
            try:
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 回退到軟體編碼
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frames_written = 0
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            cap.release()
            out.release()
            
            return frames_written > 0
        except:
            return False
    
    def _save_segmentation_info(self, output_folder, video_path, video_type, ball_times, segments):
        """保存分割資訊"""
        info_file = output_folder / f"segmentation_info_{video_type}.json"
        data = {
            'input_video': video_path,
            'video_type': video_type,
            'analysis_time': datetime.now().isoformat(),
            'method': 'FFmpeg/OpenCV',
            'parameters': {
                'segment_duration': self.segment_duration.get(),
                'confidence_threshold': self.confidence_threshold.get(),
                'min_interval': self.min_interval.get(),
                'preview_start_time': self.preview_start_time.get(),
                'dynamic_mode': self.dynamic_mode.get(),
                'dual_video_mode': self.dual_video_mode.get(),
                'detection_area': self.detection_area.get(),  # 偵測範圍設定
                'enable_bounce_filter': self.enable_bounce_filter.get(),  # 反彈球過濾
                'bounce_detection_frames': self.bounce_detection_frames.get()  # 反彈偵測幀數
            },
            'ball_entry_times': ball_times,
            'segments': segments
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_report(self):
        """匯出HTML報告"""
        if not self.detection_results:
            messagebox.showwarning("警告", "請先分析影片")
            return
        
        try:
            output_folder = Path(self.output_folder_path.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            report_file = output_folder / "analysis_report.html"
            html_content = self._generate_html_report()
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.log(f"📋 報告已匯出: {report_file}")
            messagebox.showinfo("完成", f"報告已匯出到:\n{report_file}")
        except Exception as e:
            self.log(f"❌ 匯出失敗: {e}")
    
    def _generate_html_report(self):
        """生成HTML報告內容"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>影片分析報告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .success {{ color: green; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>影片自動分割分析報告</h1>
        <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>影片資訊</h2>
        <p><strong>側面影片:</strong> {Path(self.side_video_path.get()).name if self.side_video_path.get() else '未選擇'}</p>
        {f'<p><strong>45度影片:</strong> {Path(self.deg45_video_path.get()).name}</p>' if self.dual_video_mode.get() and self.deg45_video_path.get() else ''}
        <p><strong>模式:</strong> {'雙影片同步分割' if self.dual_video_mode.get() else '單影片分割'}</p>
        <p><strong>分割模式:</strong> {'動態分割' if self.dynamic_mode.get() else '固定長度分割'}</p>
    </div>
    
    <div class="section">
        <h2>檢測結果</h2>
        <p><strong>總影格數:</strong> {len(self.detection_results)}</p>
        <p><strong>球進入次數:</strong> {len(self.ball_entry_times)}</p>
        <p><strong>球進入時間:</strong> {', '.join([f'{t:.2f}s' for t in self.ball_entry_times])}</p>
    </div>
    
    <div class="section">
        <h2>參數設定</h2>
        <p><strong>信心度閾值:</strong> {self.confidence_threshold.get()}</p>
        <p><strong>最小間隔:</strong> {self.min_interval.get()} 秒</p>
        <p><strong>片段時長:</strong> {self.segment_duration.get()} 秒</p>
    </div>
</body>
</html>
        """
    
    def log(self, message):
        """記錄訊息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, full_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        print(message)
    
    def run(self):
        """運行應用程式"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoSegmentTester()
    app.run()