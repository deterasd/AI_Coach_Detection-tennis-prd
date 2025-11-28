import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import gc
import os

# COCO 預設 17 個關節的名稱，可視需求調整/增加
body_parts_list = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def resize_frame(frame, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=inter)

def process_video(
    video_path,
    ball_model_path='model/tennisball_OD_v1.pt',
    pose_model_path='model/yolov8n-pose.pt',
    OUTPUT_WIDTH=1280,
    OUTPUT_HEIGHT=720,
    skip_frames=1,
    yolo_batch_size=4,
    ball_conf_threshold=0.8
):
    start_time = time.time()
    print(f"📹 開始處理影片: {video_path}")
    print(f"   參數: 寬度={OUTPUT_WIDTH}, 高度={OUTPUT_HEIGHT}, 批次大小={yolo_batch_size}")
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   使用設備: {device_str}")

    try:
        # 只在此處載入模型一次，並移至指定裝置
        print("🤖 載入AI模型...")
        ball_model = YOLO(ball_model_path)
        pose_model = YOLO(pose_model_path)
        ball_model.model.to(device_str)
        pose_model.model.to(device_str)
        print("✅ 模型載入完成")

        print("📖 讀取影片幀...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 無法讀取影片: {video_path}")
            return None

        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   影片資訊: {total_video_frames} 幀, {original_fps} FPS")

        frames_for_output = []
        frames_for_infer = []
        infer_indices = []

        frame_idx = 0
        read_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            try:
                resized_frame = resize_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                frames_for_output.append(resized_frame)

                if frame_idx % skip_frames == 0:
                    frames_for_infer.append(resized_frame)
                    infer_indices.append(frame_idx)
            except Exception as e:
                print(f"⚠️ 處理第 {frame_idx} 幀失敗: {e}")
                continue
            
            # 進度更新
            if frame_idx % 100 == 0:
                elapsed = time.time() - read_start
                print(f"   讀取進度: {frame_idx}/{total_video_frames} 幀 ({elapsed:.1f}s)")

        cap.release()
        
        total_frames = len(frames_for_output)
        if total_frames == 0:
            print("❌ 沒有成功讀取任何影片幀")
            return None
        
        print(f"✅ 影片讀取完成: {total_frames} 幀 ({len(frames_for_infer)} 幀用於推理)")

        # AI推理
        print("🧠 開始AI推理...")
        inference_start = time.time()
        
        # 推論時加入 no_grad 以減少記憶體佔用
        with torch.no_grad():
            print(f"   姿態檢測: {len(frames_for_infer)} 幀")
            pose_results_batch = pose_model.predict(
                frames_for_infer,
                verbose=False,
                device=device_str,
                batch=yolo_batch_size
            )
            
            print(f"   網球檢測: {len(frames_for_infer)} 幀")
            ball_results_batch = ball_model.predict(
                frames_for_infer,
                verbose=False,
                device=device_str,
                batch=yolo_batch_size
            )
        
        inference_time = time.time() - inference_start
        print(f"✅ AI推理完成 ({inference_time:.2f}s)")

        # 處理結果
        print("📊 處理AI結果...")
        ball_positions = [None] * total_frames
        ball_confidences = [None] * total_frames
        keypoints_per_frame = [None] * total_frames

        for i, fidx in enumerate(infer_indices):
            try:
                pose_result = pose_results_batch[i]
                ball_result = ball_results_batch[i]

                # 處理姿態檢測結果
                if (pose_result.keypoints is not None and 
                    len(pose_result.keypoints) > 0 and 
                    len(pose_result.keypoints.xy) > 0):
                    kpts = pose_result.keypoints.xy[0]  # shape (17,2)
                    kpts_xy = [(int(x), int(y)) for x, y in kpts]
                else:
                    kpts_xy = None

                # 處理網球檢測結果
                boxes = ball_result.boxes
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    if float(box.conf[0]) >= ball_conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        ball_pos = (cx, cy)
                        ball_conf = float(box.conf[0])
                    else:
                        ball_pos = None
                        ball_conf = None
                else:
                    ball_pos = None
                    ball_conf = None

                idx_in_list = fidx - 1
                ball_positions[idx_in_list] = ball_pos
                ball_confidences[idx_in_list] = ball_conf
                keypoints_per_frame[idx_in_list] = kpts_xy
                
            except Exception as e:
                print(f"⚠️ 處理第 {i} 個推理結果失敗: {e}")
                continue

        print("🔄 補全缺失數據...")
        # 補全缺失資料：若當前幀資料缺失則使用前一幀補上
        last_ball = None
        last_conf = None
        last_kpts = None
        for i in range(total_frames):
            if ball_positions[i] is None:
                ball_positions[i] = last_ball
                ball_confidences[i] = last_conf
            else:
                last_ball = ball_positions[i]
                last_conf = ball_confidences[i]

            if keypoints_per_frame[i] is None:
                keypoints_per_frame[i] = last_kpts
            else:
                last_kpts = keypoints_per_frame[i]

        print("🎥 生成輸出影片...")
        output_path = video_path.replace('.mp4', '_processed.mp4')
        info_panel_width = 400
        output_width = OUTPUT_WIDTH + info_panel_width
        output_height = OUTPUT_HEIGHT

        # 嘗試創建VideoWriter
        video_writer_created = False
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (output_width, output_height))
        
        # 檢查 VideoWriter 是否成功初始化
        if not out.isOpened():
            print(f"❌ VideoWriter 初始化失敗: {output_path}")
            print(f"🔧 嘗試使用不同的編碼器...")
            
            # 嘗試其他編碼器
            codecs_to_try = [
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('H264', cv2.VideoWriter_fourcc(*'H264')),
                ('X264', cv2.VideoWriter_fourcc(*'X264')),
                ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
            ]
            
            for codec_name, codec in codecs_to_try:
                print(f"🔧 嘗試 {codec_name} 編碼器...")
                out = cv2.VideoWriter(output_path, codec, original_fps, (output_width, output_height))
                if out.isOpened():
                    print(f"✅ {codec_name} 編碼器成功")
                    video_writer_created = True
                    break
                else:
                    out.release()
            
            # 如果所有編碼器都失敗，嘗試 AVI 格式
            if not video_writer_created:
                print("🔧 嘗試 AVI 格式...")
                output_path = output_path.replace('.mp4', '.avi')
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), original_fps, (output_width, output_height))
                
                if out.isOpened():
                    video_writer_created = True
                    print("✅ AVI 格式成功")
                else:
                    print("❌ 所有影片編碼器都失敗，返回原始檔案")
                    return video_path
        else:
            video_writer_created = True
            print("✅ MP4V 編碼器成功")

        if not video_writer_created:
            print("❌ 無法創建影片寫入器")
            return None

        TRACKED_KEYPOINTS = [10]
        keypoint_trails = {kp: [] for kp in TRACKED_KEYPOINTS}
        ball_trail = []

        print("✍️ 寫入影片幀...")
        write_start = time.time()
        
        for i in range(total_frames):
            try:
                frame = frames_for_output[i].copy()
                ball_pos = ball_positions[i]
                ball_conf = ball_confidences[i]
                kpts = keypoints_per_frame[i]

                ball_trail.append(ball_pos)
                if kpts is not None:
                    for kp_idx in TRACKED_KEYPOINTS:
                        if kp_idx < len(kpts):
                            keypoint_trails[kp_idx].append(kpts[kp_idx])
                        else:
                            keypoint_trails[kp_idx].append(None)
                else:
                    for kp_idx in TRACKED_KEYPOINTS:
                        keypoint_trails[kp_idx].append(None)

                # 視覺化處理（簡化版）
                combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                combined_frame[:, :OUTPUT_WIDTH] = frame
                
                # 添加基本資訊到右側面板
                info_text = f"Frame: {i+1}/{total_frames}"
                cv2.putText(combined_frame, info_text, (OUTPUT_WIDTH + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if ball_pos:
                    ball_text = f"Ball: ({ball_pos[0]}, {ball_pos[1]})"
                    cv2.putText(combined_frame, ball_text, (OUTPUT_WIDTH + 10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.circle(frame, ball_pos, 8, (0, 255, 255), -1)
                
                # 繪製關鍵點
                if kpts is not None:
                    for j, (x, y) in enumerate(kpts):
                        if x > 0 and y > 0:
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # 更新合併幀
                combined_frame[:, :OUTPUT_WIDTH] = frame
                
                # 寫入影片幀
                out.write(combined_frame)
                
                # 進度更新
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - write_start
                    progress = (i + 1) / total_frames * 100
                    print(f"   寫入進度: {progress:.1f}% ({i+1}/{total_frames}) - {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"⚠️ 處理第 {i} 幀失敗: {e}")
                # 寫入黑色幀作為替代
                black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                out.write(black_frame)
                continue

        out.release()
        write_time = time.time() - write_start
        print(f"✅ 影片寫入完成 ({write_time:.2f}s)")
        
        # 檢查輸出檔案
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # 檔案小於 1KB 表示可能有問題
                print(f"⚠️ 輸出檔案可能有問題: {output_path} (大小: {file_size} bytes)")
            else:
                print(f"✅ 影片處理完成: {output_path} (大小: {file_size} bytes)")
        else:
            print(f"❌ 輸出檔案未創建: {output_path}")
            return None

        # 清理記憶體
        print("🧹 清理記憶體...")
        del frames_for_output, frames_for_infer, ball_positions, ball_confidences, keypoints_per_frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"🎉 影片處理總耗時: {total_time:.2f}秒")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 影片處理發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 清理記憶體
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None
        else:
            for kp_idx in TRACKED_KEYPOINTS:
                keypoint_trails[kp_idx].append(None)

        valid_ball_positions = [p for p in ball_trail if p is not None]
        for b in range(1, len(valid_ball_positions)):
            p1 = valid_ball_positions[b - 1]
            p2 = valid_ball_positions[b]
            if p1 and p2:
                progress = b / len(valid_ball_positions)
                color = (0, int(255*(1 - progress)), int(255*progress))
                cv2.line(frame, p1, p2, color, 4)

        for kp_idx, trail in keypoint_trails.items():
            valid_trail = [p for p in trail if p is not None]
            for t in range(1, len(valid_trail)):
                p1 = valid_trail[t-1]
                p2 = valid_trail[t]
                progress = t / len(valid_trail)
                color = (int(255*(1 - progress)), int(255*progress), 0)
                cv2.line(frame, p1, p2, color, 4)

        if kpts is not None:
            for idx, (xx, yy) in enumerate(kpts):
                color = (0, 0, 255) if idx in TRACKED_KEYPOINTS else (0, 255, 0)
                cv2.circle(frame, (xx, yy), 5, color, -1)
                cv2.putText(frame, str(idx), (xx+5, yy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        info_panel = np.ones((output_height, info_panel_width, 3), dtype=np.uint8) * 40
        header_height = 50
        cv2.rectangle(info_panel, (0, 0), (info_panel_width, header_height), (0, 150, 0), -1)
        cv2.putText(info_panel, "Tennis Ball Detection", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        y_text = header_height + 30
        if ball_pos is not None:
            cv2.putText(info_panel, "Ball Status: Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(info_panel, "Ball Status: Not Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_text += 30

        if ball_pos is not None:
            cx, cy = ball_pos
            cv2.putText(info_panel, f"Position: ({cx}, {cy})", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            y_text += 30
            if ball_conf is not None:
                cv2.putText(info_panel, f"Confidence: {ball_conf:.2f}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                y_text += 30

        pose_header_height = 40
        pose_header_top = y_text
        pose_header_bottom = pose_header_top + pose_header_height
        cv2.rectangle(info_panel,
                      (0, pose_header_top),
                      (info_panel_width, pose_header_bottom),
                      (255, 100, 0),
                      -1)
        cv2.putText(info_panel, "Pose Estimation",
                    (10, pose_header_top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        y_text = pose_header_bottom + 30
        if kpts is not None:
            for idx, part_name in enumerate(body_parts_list):
                if idx < len(kpts):
                    xx, yy = kpts[idx]
                    text_line = f"{part_name:<15}: ({xx}, {yy})"
                else:
                    text_line = f"{part_name:<15}: ( -, - )"
                cv2.putText(info_panel, text_line, (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                y_text += 22
                if y_text >= output_height - 10:
                    break
        else:
            cv2.putText(info_panel, "No keypoints found",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if frame.shape[0] != output_height:
            frame = cv2.resize(frame, (OUTPUT_WIDTH, output_height))
        
        # 確保影片幀維度正確
        if frame.shape != (output_height, OUTPUT_WIDTH, 3):
            frame = cv2.resize(frame, (OUTPUT_WIDTH, output_height))
        
        combined_frame = np.hstack((frame, info_panel))
        
        # 檢查合併後的幀維度
        if combined_frame.shape != (output_height, output_width, 3):
            print(f"⚠️ 幀 {i}: 維度不匹配 {combined_frame.shape} != ({output_height}, {output_width}, 3)")
            combined_frame = cv2.resize(combined_frame, (output_width, output_height))
        
        # 寫入影片幀
        out.write(combined_frame)
        # 注意：OpenCV 的 write() 返回值不可靠，不檢查返回值

    out.release()
    
    # 檢查輸出檔案是否成功創建
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size < 1000:  # 檔案小於 1KB 表示可能有問題
            print(f"⚠️ 輸出檔案可能有問題: {output_path} (大小: {file_size} bytes)")
        else:
            print(f"✅ 影片處理完成: {output_path} (大小: {file_size} bytes)")
    else:
        print(f"❌ 輸出檔案未創建: {output_path}")

    # 清理中間資料，避免累積
    del frames_for_output, frames_for_infer, ball_positions, ball_confidences, keypoints_per_frame
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path

if __name__ == "__main__":
    total_start = time.time()
    video_path = '測試2__1_45_compressed.mp4'
    output_path = process_video(video_path)
    total_end = time.time()
    print(f"===== 程式總耗時: {total_end - total_start:.2f} 秒 =====")
