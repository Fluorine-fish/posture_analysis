import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
from collections import deque

os.environ['GLOG_minloglevel'] = '2'

# 配置参数
OCCLUSION_FRAMES_THRESHOLD = 4
CLEAR_FRAMES_THRESHOLD = 3
VISIBILITY_THRESHOLD = 0.5
HEAD_ANGLE_THRESHOLD = 50
SET_MIN_DETECTION_CONFIDENCE = 0.8
SET_MIN_TRACKING_CONFIDENCE = 0.2

# 性能监控参数
FPS_WINDOW_SIZE = 10  # 帧率计算窗口大小

def check_occlusion(landmarks):
    """检测面部和肩部遮挡情况"""
    LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    NOSE = mp.solutions.pose.PoseLandmark.NOSE
    LEFT_EYE = mp.solutions.pose.PoseLandmark.LEFT_EYE
    RIGHT_EYE = mp.solutions.pose.PoseLandmark.RIGHT_EYE

    try:
        # 检查肩膀可见性
        shoulder_occluded = (
            landmarks[LEFT_SHOULDER.value].visibility < VISIBILITY_THRESHOLD or
            landmarks[RIGHT_SHOULDER.value].visibility < VISIBILITY_THRESHOLD
        )
        
        # 检查面部关键点
        face_occluded = any(
            landmarks[point.value].visibility < VISIBILITY_THRESHOLD
            for point in [NOSE, LEFT_EYE, RIGHT_EYE]
        )

        if shoulder_occluded and face_occluded:
            return True, "Full Occlusion"
        elif shoulder_occluded:
            return True, "Shoulder Occluded"
        elif face_occluded:
            return True, "Face Occluded"
        return False, "Clear"
    
    except (IndexError, AttributeError):
        return True, "Detection Failed"

def calculate_head_angle(landmarks, frame_shape):
    """计算头部前倾角度"""
    LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    NOSE = mp.solutions.pose.PoseLandmark.NOSE
    
    try:
        h, w = frame_shape[:2]
        ls = landmarks[LEFT_SHOULDER.value]
        rs = landmarks[RIGHT_SHOULDER.value]
        nose = landmarks[NOSE.value]
        
        mid_shoulder = np.array([(ls.x + rs.x)/2 * w, (ls.y + rs.y)/2 * h])
        nose_point = np.array([nose.x * w, nose.y * h])
        vector = nose_point - mid_shoulder
        
        angle_rad = math.atan2(vector[0], -vector[1])  # y轴向下故取反
        angle_deg = math.degrees(angle_rad)
        abs_angle = abs(angle_deg)
        
        is_forward = abs_angle > HEAD_ANGLE_THRESHOLD
        return abs_angle, is_forward, {
            'mid_shoulder': mid_shoulder.astype(int),
            'nose': nose_point.astype(int)
        }
        
    except (IndexError, AttributeError, ValueError):
        return None, False, {}

def initialize_camera():
    """自动检测可用摄像头"""
    for i in range(10):  # 尝试0-9号摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"找到摄像头设备: /dev/video{i}")
            # 配置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    return None

def realtime_pose_estimation():
    # 初始化摄像头
    cap = initialize_camera()
    if not cap:
        print("错误：未找到可用摄像头")
        return
    
    # 初始化MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=SET_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=SET_MIN_TRACKING_CONFIDENCE
    )

    # 性能监控变量
    frame_times = deque(maxlen=FPS_WINDOW_SIZE)  # 总帧时间队列
    process_times = deque(maxlen=FPS_WINDOW_SIZE) # 处理时间队列
    last_frame_time = time.time()
    
    # 状态跟踪变量
    occlusion_counter = 0
    clear_counter = 0
    final_occlusion = False
    last_valid_angle = None
    
    try:
        while cap.isOpened():
            # 帧接收开始时间
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("视频流中断")
                break

            # 帧接收结束时间
            frame_receive_time = time.time() - frame_start_time
            
            # 处理开始时间
            process_start_time = time.time()

            # 调整分辨率并转换颜色空间
            resized_frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # 执行姿势检测
            results = pose.process(rgb_frame)
            
            current_occluded = False
            occlusion_status = "Clear"
            display_text = "Initializing..."
            status_color = (255, 255, 255)

            if results.pose_landmarks:
                # 检测当前帧遮挡状态
                current_occluded, occlusion_status = check_occlusion(
                    results.pose_landmarks.landmark
                )
                
                # 更新状态计数器
                if current_occluded:
                    occlusion_counter = min(occlusion_counter + 1, OCCLUSION_FRAMES_THRESHOLD)
                    clear_counter = max(0, clear_counter - 1)
                else:
                    clear_counter = min(clear_counter + 1, CLEAR_FRAMES_THRESHOLD)
                    occlusion_counter = max(0, occlusion_counter - 1)
                
                # 判断最终遮挡状态
                final_occlusion = occlusion_counter >= OCCLUSION_FRAMES_THRESHOLD
                is_valid_detection = clear_counter >= CLEAR_FRAMES_THRESHOLD
                
                if is_valid_detection and not final_occlusion:
                    # 绘制姿势骨架
                    mp.solutions.drawing_utils.draw_landmarks(
                        resized_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # 计算头部角度
                    angle, is_forward, points = calculate_head_angle(
                        results.pose_landmarks.landmark,
                        resized_frame.shape
                    )
                    
                    if angle is not None:
                        last_valid_angle = angle
                        # 绘制参考线
                        cv2.line(resized_frame, 
                                tuple(points['mid_shoulder']), 
                                tuple(points['nose']), 
                                (0, 255, 0), 2)
                        
                        # 更新显示文本
                        status_color = (0, 0, 255) if is_forward else (0, 255, 0)
                        display_text = f"Angle: {angle:.1f}° {'[BAD]' if is_forward else '[GOOD]'}"
                    else:
                        display_text = "Angle calculation failed"
                        status_color = (0, 255, 255)  # 黄色警告
                else:
                    # 显示遮挡信息
                    status_color = (0, 0, 255)
                    display_text = f"OCCLUSION: {occlusion_status}"
                    if last_valid_angle is not None:
                        display_text += f" | Last: {last_valid_angle:.1f}°"

                # 记录处理时间
                process_time = time.time() - process_start_time
                process_times.append(process_time)
                frame_times.append(time.time() - last_frame_time)
                last_frame_time = time.time()

                # 绘制状态信息
                state_text = f"State: {'Occluded' if final_occlusion else 'Tracking'}"
                cv2.putText(resized_frame, state_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                            (0, 0, 255) if final_occlusion else (0, 255, 0), 2)
                cv2.putText(resized_frame, display_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                # 无人检测状态
                cv2.putText(resized_frame, "No Person Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 计算性能指标
            camera_fps = 1/np.mean(frame_times) if frame_times else 0
            process_fps = 1/np.mean(process_times) if process_times else 0
            avg_receive_time = np.mean(frame_times)*1000 if frame_times else 0
            avg_process_time = np.mean(process_times)*1000 if process_times else 0

            # 绘制性能监控面板
            perf_y_start = 90
            cv2.putText(resized_frame, f"Camera FPS: {camera_fps:.1f}", (10, perf_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(resized_frame, f"Process FPS: {process_fps:.1f}", (10, perf_y_start+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(resized_frame, f"Frame Receive: {frame_receive_time*1000:.1f}ms", (10, perf_y_start+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(resized_frame, f"Frame Process: {avg_process_time:.1f}ms", (10, perf_y_start+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

            # 显示调试信息
            debug_info = f"Occ: {occlusion_counter}/{OCCLUSION_FRAMES_THRESHOLD} Clear: {clear_counter}/{CLEAR_FRAMES_THRESHOLD}"
            cv2.putText(resized_frame, debug_info, (10, resized_frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow('Posture Monitor', resized_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    realtime_pose_estimation()