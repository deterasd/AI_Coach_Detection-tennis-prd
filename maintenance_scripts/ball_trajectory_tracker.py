"""
完整球軌跡追蹤系統
基於連續偵測和追蹤，而非邊緣檢測

核心邏輯：
1. 球進入：從邊緣區域首次出現
2. 追蹤階段：持續追蹤球的位置
3. 球離開：球離開畫面或長時間未偵測到
"""

import cv2
import numpy as np
from pathlib import Path

class BallState:
    """球的狀態"""
    NOT_PRESENT = "not_present"      # 球未出現
    ENTERING = "entering"             # 球進入中
    TRACKING = "tracking"             # 追蹤中
    LEAVING = "leaving"               # 球離開中
    EXITED = "exited"                 # 球已離開

class Ball:
    """球的追蹤資訊"""
    def __init__(self, ball_id, first_position, first_time, first_frame):
        self.ball_id = ball_id
        self.state = BallState.ENTERING
        self.positions = [first_position]  # 所有位置記錄
        self.times = [first_time]          # 對應的時間
        self.frames = [first_frame]        # 對應的幀號
        self.entry_time = first_time
        self.entry_frame = first_frame
        self.exit_time = None
        self.exit_frame = None
        self.last_seen_time = first_time
        self.last_seen_frame = first_frame
        self.disappeared_count = 0  # 連續未偵測到的幀數
        
    def update_position(self, position, time, frame):
        """更新球的位置"""
        self.positions.append(position)
        self.times.append(time)
        self.frames.append(frame)
        self.last_seen_time = time
        self.last_seen_frame = frame
        self.disappeared_count = 0
        
        # 更新狀態
        if self.state == BallState.ENTERING:
            self.state = BallState.TRACKING
    
    def mark_disappeared(self):
        """標記球未被偵測到"""
        self.disappeared_count += 1
        
        # 如果持續未偵測到，標記為離開
        if self.disappeared_count > 15:  # 超過15幀未偵測到
            if self.state != BallState.EXITED:
                self.state = BallState.LEAVING
    
    def mark_exited(self, exit_time, exit_frame):
        """標記球已離開"""
        self.state = BallState.EXITED
        self.exit_time = exit_time
        self.exit_frame = exit_frame
    
    def get_trajectory_duration(self):
        """獲取軌跡持續時間"""
        if self.exit_time is not None:
            return self.exit_time - self.entry_time
        else:
            return self.last_seen_time - self.entry_time
    
    def get_trajectory_info(self):
        """獲取軌跡資訊"""
        return {
            "ball_id": self.ball_id,
            "entry_time": self.entry_time,
            "entry_frame": self.entry_frame,
            "exit_time": self.exit_time or self.last_seen_time,
            "exit_frame": self.exit_frame or self.last_seen_frame,
            "duration": self.get_trajectory_duration(),
            "total_positions": len(self.positions),
            "state": self.state
        }

class BallTrajectoryTracker:
    """完整球軌跡追蹤器"""
    
    def __init__(self, confidence_threshold=0.5, ball_entry_direction="right"):
        self.confidence_threshold = confidence_threshold
        self.ball_entry_direction = ball_entry_direction
        self.active_balls = {}  # 當前活躍的球
        self.completed_balls = []  # 已完成的球軌跡
        self.next_ball_id = 0
        self.max_tracking_distance = 240  # 最大追蹤距離（像素）
        self.max_disappeared_frames = 30  # 最大消失幀數
        
    def detect_ball_in_frame(self, frame, model):
        """在單幀中偵測球"""
        results = model(frame, verbose=False)
        
        detected_balls = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence >= self.confidence_threshold:
                        detected_balls.append({
                            'position': (center_x, center_y),
                            'confidence': confidence
                        })
        
        return detected_balls
    
    def is_in_entry_zone(self, position, frame_width, frame_height):
        """檢查是否在進入區域"""
        x, y = position
        edge_ratio = 0.15
        
        edges = {
            'left': frame_width * edge_ratio,
            'right': frame_width * (1 - edge_ratio),
            'top': frame_height * edge_ratio,
            'bottom': frame_height * (1 - edge_ratio)
        }
        
        if self.ball_entry_direction == "right":
            # 右邊緣上2/3 + 上邊緣右半邊
            right_edge_y_threshold = frame_height * (2/3)
            right_edge_in_zone = (x > edges['right'] and y < right_edge_y_threshold)
            
            top_edge_x_threshold = frame_width * 0.5
            top_edge_in_zone = (y < edges['top'] and x > top_edge_x_threshold)
            
            return right_edge_in_zone or top_edge_in_zone
        else:
            # 左邊緣上2/3 + 上邊緣左半邊
            left_edge_y_threshold = frame_height * (2/3)
            left_edge_in_zone = (x < edges['left'] and y < left_edge_y_threshold)
            
            top_edge_x_threshold = frame_width * 0.5
            top_edge_in_zone = (y < edges['top'] and x < top_edge_x_threshold)
            
            return left_edge_in_zone or top_edge_in_zone
    
    def is_leaving_frame(self, position, frame_width, frame_height):
        """檢查球是否正在離開畫面"""
        x, y = position
        margin = 50  # 邊緣容差
        
        # 檢查是否接近邊界
        near_left = x < margin
        near_right = x > (frame_width - margin)
        near_top = y < margin
        near_bottom = y > (frame_height - margin)
        
        return near_left or near_right or near_top or near_bottom
    
    def match_detection_to_ball(self, detection, current_time, current_frame):
        """將偵測結果匹配到現有的球"""
        position = detection['position']
        
        # 尋找最接近的活躍球
        best_match_id = None
        min_distance = float('inf')
        
        for ball_id, ball in self.active_balls.items():
            if ball.state == BallState.EXITED:
                continue
                
            last_pos = ball.positions[-1]
            distance = np.sqrt((position[0] - last_pos[0])**2 + 
                             (position[1] - last_pos[1])**2)
            
            if distance < min_distance and distance < self.max_tracking_distance:
                min_distance = distance
                best_match_id = ball_id
        
        return best_match_id
    
    def process_frame(self, frame, model, frame_number, current_time, frame_width, frame_height):
        """處理單一幀"""
        # 偵測所有球
        detected_balls = self.detect_ball_in_frame(frame, model)
        
        # 標記所有活躍球為未偵測
        for ball in self.active_balls.values():
            if ball.state != BallState.EXITED:
                ball.mark_disappeared()
        
        matched_ball_ids = set()
        
        # 處理每個偵測到的球
        for detection in detected_balls:
            position = detection['position']
            
            # 嘗試匹配到現有的球
            matched_id = self.match_detection_to_ball(detection, current_time, frame_number)
            
            if matched_id is not None:
                # 更新現有球的位置
                ball = self.active_balls[matched_id]
                ball.update_position(position, current_time, frame_number)
                matched_ball_ids.add(matched_id)
                
                # 檢查球是否正在離開
                if self.is_leaving_frame(position, frame_width, frame_height):
                    ball.state = BallState.LEAVING
                    
            elif self.is_in_entry_zone(position, frame_width, frame_height):
                # 新球進入
                new_ball = Ball(self.next_ball_id, position, current_time, frame_number)
                self.active_balls[self.next_ball_id] = new_ball
                matched_ball_ids.add(self.next_ball_id)
                print(f"   ⚾ 新球進入 (球#{self.next_ball_id}): {current_time:.2f}s")
                self.next_ball_id += 1
        
        # 處理消失的球
        balls_to_complete = []
        for ball_id, ball in self.active_balls.items():
            if ball.state == BallState.LEAVING and ball.disappeared_count > 10:
                # 球已經離開
                ball.mark_exited(current_time, frame_number)
                balls_to_complete.append(ball_id)
                print(f"   🎯 球離開 (球#{ball_id}): {current_time:.2f}s (持續{ball.get_trajectory_duration():.2f}s)")
            elif ball.disappeared_count > self.max_disappeared_frames:
                # 球長時間未偵測到，標記為已離開
                ball.mark_exited(ball.last_seen_time, ball.last_seen_frame)
                balls_to_complete.append(ball_id)
                print(f"   ⚠️ 球消失 (球#{ball_id}): {ball.last_seen_time:.2f}s")
        
        # 將完成的球移到已完成列表
        for ball_id in balls_to_complete:
            self.completed_balls.append(self.active_balls[ball_id])
            del self.active_balls[ball_id]
    
    def finalize_tracking(self, video_duration, total_frames):
        """結束追蹤，處理剩餘的活躍球"""
        for ball_id, ball in self.active_balls.items():
            if ball.state != BallState.EXITED:
                ball.mark_exited(video_duration, total_frames - 1)
                self.completed_balls.append(ball)
                print(f"   🎬 影片結束，球#{ball_id}標記為完成")
        
        self.active_balls.clear()
    
    def get_all_ball_trajectories(self):
        """獲取所有球的軌跡資訊"""
        return [ball.get_trajectory_info() for ball in self.completed_balls]

def track_ball_trajectories(video_path, model, confidence_threshold=0.5, 
                           ball_entry_direction="right"):
    """
    完整球軌跡追蹤主函數
    
    Returns:
        list: 所有球的軌跡資訊
    """
    print(f"🔍 開始完整球軌跡追蹤: {Path(video_path).name}")
    print(f"   追蹤模式: 完整軌跡追蹤（進入→移動→擊球→離開）")
    print(f"   球進入方向: {'右邊' if ball_entry_direction == 'right' else '左邊'}")
    print(f"   信心度閾值: {confidence_threshold}")
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    
    print(f"   影片資訊: {total_frames} 幀, {fps:.2f} FPS, {frame_width}x{frame_height}")
    
    # 創建追蹤器
    tracker = BallTrajectoryTracker(confidence_threshold, ball_entry_direction)
    
    # 處理每一幀
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # 處理當前幀
        tracker.process_frame(frame, model, frame_count, current_time, 
                            frame_width, frame_height)
        
        # 顯示進度
        if frame_count % int(fps * 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   進度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        frame_count += 1
    
    # 結束追蹤
    tracker.finalize_tracking(video_duration, total_frames)
    
    cap.release()
    
    # 獲取所有軌跡
    trajectories = tracker.get_all_ball_trajectories()
    
    print(f"✅ 追蹤完成: 找到 {len(trajectories)} 個完整球軌跡")
    for i, traj in enumerate(trajectories, 1):
        print(f"   球{i}: {traj['entry_time']:.2f}s → {traj['exit_time']:.2f}s "
              f"(持續{traj['duration']:.2f}s, {traj['total_positions']}個位置)")
    
    return trajectories

if __name__ == "__main__":
    print("🎾 完整球軌跡追蹤系統測試")
    print("=" * 60)
    print("此模組提供基於完整軌跡追蹤的球偵測系統")
    print("不再依賴邊緣檢測，而是追蹤球的完整生命週期")
