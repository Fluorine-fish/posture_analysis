"""
实时姿势与面部情绪监测系统
功能：同时监测头部姿势和基础面部表情，包含防遮挡检测功能
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
from collections import deque
from enum import Enum

# 设置日志级别（必须放在所有import之前）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭TensorFlow信息日志(0-3级，2=WARNING)
os.environ['GLOG_minloglevel'] = '2'      # 关闭mediapipe的GLOG日志

class EmotionState(Enum):
    """面部情绪状态枚举"""
    NEUTRAL = 0    # 中性
    HAPPY = 1      # 高兴（检测到明显笑容）
    ANGRY = 3      # 生气（检测到皱眉）
    SAD = 4        # 悲伤（检测到闭眼）

# === 系统参数配置 ===
# 摄像头参数
CAMERA_WIDTH = 1280   # 摄像头采集原始分辨率宽度
CAMERA_HEIGHT = 720   # 摄像头采集原始分辨率高度
PROCESS_WIDTH = 640    # 实际处理时的缩放宽度（降低分辨率提升性能）
PROCESS_HEIGHT = 360   # 实际处理时的缩放高度

# 姿势检测参数
OCCLUSION_FRAMES_THRESHOLD = 4    # 连续多少帧检测到遮挡视为有效遮挡
CLEAR_FRAMES_THRESHOLD = 3        # 连续多少帧无遮挡视为恢复追踪
VISIBILITY_THRESHOLD = 0.5        # 关键点可见度阈值（0-1，越大要求可见度越高）
HEAD_ANGLE_THRESHOLD = 45         # 头部偏转角度阈值（超过视为姿势不良）

# 情绪检测参数
EMOTION_SMOOTHING_WINDOW = 7      # 情绪状态平滑窗口（基于多少帧做多数投票）
MOUTH_OPEN_RATIO_THRESHOLD = 0.45 # 嘴部开合比阈值（越大需要更明显的笑容）
EYE_OPEN_RATIO_THRESHOLD = 0.25   # 眼睛开合比阈值（小于此值视为闭眼）
BROW_DOWN_THRESHOLD = 0.04        # 眉毛下压阈值（负值越大表示皱眉越明显）

# MediaPipe初始化
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def check_occlusion(landmarks):
    """
    检测身体遮挡状态
    参数：
        landmarks - 姿势关键点列表
    返回：
        (是否被遮挡, 遮挡类型描述)
    """
    # 定义需要检测的关键点
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
    NOSE = mp_pose.PoseLandmark.NOSE
    LEFT_EYE = mp_pose.PoseLandmark.LEFT_EYE
    RIGHT_EYE = mp_pose.PoseLandmark.RIGHT_EYE

    try:
        # 双肩可见性检测
        shoulder_occluded = (
            landmarks[LEFT_SHOULDER.value].visibility < VISIBILITY_THRESHOLD or
            landmarks[RIGHT_SHOULDER.value].visibility < VISIBILITY_THRESHOLD
        )
        
        # 面部关键点可见性检测
        face_occluded = any(
            landmarks[point.value].visibility < VISIBILITY_THRESHOLD
            for point in [NOSE, LEFT_EYE, RIGHT_EYE]
        )

        # 综合判断遮挡类型
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
    """
    计算头部偏转角度
    参数：
        landmarks - 姿势关键点列表
        frame_shape - 帧图像尺寸
    返回：
        (角度值, 是否超过阈值, 绘制用关键点坐标)
    """
    try:
        h, w = frame_shape[:2]
        
        # 获取双肩和鼻子坐标
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        # 计算双肩中点坐标
        mid_shoulder = np.array([(ls.x + rs.x)/2 * w, (ls.y + rs.y)/2 * h])
        # 获取鼻子坐标
        nose_point = np.array([nose.x * w, nose.y * h])
        
        # 计算向量（鼻子相对于双肩中点的位置）
        vector = nose_point - mid_shoulder
        
        # 计算偏转角度（将向量转换为与垂直线的夹角）
        angle_rad = math.atan2(vector[0], -vector[1])
        abs_angle = abs(math.degrees(angle_rad))
        
        return abs_angle, abs_angle > HEAD_ANGLE_THRESHOLD, {
            'mid_shoulder': mid_shoulder.astype(int),
            'nose': nose_point.astype(int)
        }
    except (IndexError, AttributeError, ValueError):
        return None, False, {}

class EmotionAnalyzer:
    """面部情绪分析器"""
    def __init__(self):
        # 初始化Face Mesh模型
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,          # 最大检测人脸数
            refine_landmarks=True,    # 使用精细模式（包含虹膜检测）
            min_detection_confidence=0.7,  # 检测置信度阈值
            min_tracking_confidence=0.5     # 跟踪置信度阈值
        )
        self.emotion_history = deque(maxlen=EMOTION_SMOOTHING_WINDOW)
        
        # 面部特征点索引配置
        self.LIPS = [61, 291, 78, 308]    # 上下唇和嘴角点（61:上唇，291:下唇，78/308:嘴角）
        self.LEFT_EYE = [33, 160, 158, 133]  # 左眼特征点（上下眼睑）
        self.RIGHT_EYE = [362, 385, 386, 263] # 右眼特征点
        self.LEFT_BROW = [70, 63, 105, 66]   # 左眉毛特征点
        self.RIGHT_BROW = [300, 293, 334, 296] # 右眉毛特征点

    def analyze(self, frame):
        """分析当前帧面部情绪"""
        start_time = time.time()
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_time = time.time() - start_time
        
        if not results.multi_face_landmarks:
            return EmotionState.NEUTRAL, None, process_time
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # 计算各部位特征参数
        mouth_ratio = self._mouth_open_ratio(landmarks, h, w)
        eye_ratio = self._eye_open_ratio(landmarks, h, w)
        brow_pos = self._brow_position(landmarks, h, w)
        
        # 判断当前情绪
        emotion = self._determine_emotion(mouth_ratio, eye_ratio, brow_pos)
        self.emotion_history.append(emotion)
        
        return self._smooth_emotion(), results.multi_face_landmarks[0], process_time

    def _mouth_open_ratio(self, landmarks, h, w):
        """计算嘴部开合比例（垂直距离/水平宽度）"""
        upper = landmarks[self.LIPS[0]].y * h  # 上唇点Y坐标
        lower = landmarks[self.LIPS[1]].y * h  # 下唇点Y坐标
        width = abs(landmarks[self.LIPS[2]].x - landmarks[self.LIPS[3]].x) * w  # 嘴角间距
        return abs(upper - lower) / width if width != 0 else 0

    def _eye_open_ratio(self, landmarks, h, w):
        """计算眼睛开合比例（垂直距离/水平宽度）"""
        # 使用左眼作为代表（可改为左右眼平均值）
        vert = abs(landmarks[self.LEFT_EYE[1]].y - landmarks[self.LEFT_EYE[3]].y) * h
        horiz = abs(landmarks[self.LEFT_EYE[0]].x - landmarks[self.LEFT_EYE[2]].x) * w
        return vert / horiz if horiz != 0 else 0

    def _brow_position(self, landmarks, h, w):
        """计算眉毛平均位置（相对于面部的位置）"""
        left = np.mean([landmarks[i].y for i in self.LEFT_BROW]) * h
        right = np.mean([landmarks[i].y for i in self.RIGHT_BROW]) * h
        return (left + right) / 2

    def _determine_emotion(self, mouth, eye, brow):
        """根据特征参数判断情绪"""
        # 空值保护
        if any(v is None for v in [mouth, eye, brow]):
            return EmotionState.NEUTRAL
            
        # 高兴判断：嘴部开合度达标且眼睛保持睁开
        if mouth > MOUTH_OPEN_RATIO_THRESHOLD and eye > EYE_OPEN_RATIO_THRESHOLD*1.3:
            return EmotionState.HAPPY
        # 悲伤判断：眼睛闭合
        if eye < EYE_OPEN_RATIO_THRESHOLD*0.8:
            return EmotionState.SAD
        # 生气判断：眉毛下压
        if brow < -BROW_DOWN_THRESHOLD:
            return EmotionState.ANGRY
        return EmotionState.NEUTRAL

    def _smooth_emotion(self):
        """基于历史帧的多数投票平滑情绪状态"""
        counts = {e:0 for e in EmotionState}
        for e in self.emotion_history:
            counts[e] += 1
        return max(counts, key=counts.get)

class PostureMonitor:
    """姿势监测主程序"""
    def __init__(self):
        self.cap = self._init_camera()
        self.pose = mp_pose.Pose(
            static_image_mode=False,    # 视频流模式
            model_complexity=1,        # 模型复杂度（0-2）
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.emotion_analyzer = EmotionAnalyzer()
        self.occlusion_counter = 0     # 遮挡连续计数
        self.clear_counter = 0         # 清晰连续计数
        self.last_valid_angle = None   # 最后有效角度值
        self.pose_times = deque(maxlen=30)  # 姿势检测时间队列（用于计算FPS）
        self.face_times = deque(maxlen=30)  # 面部检测时间队列

    def _init_camera(self):
        """初始化摄像头设备"""
        for i in range(10):  # 尝试前10个摄像头设备
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"摄像头初始化成功: /dev/video{i}")
                return cap
            cap.release()
        raise RuntimeError("未检测到可用摄像头设备")

    def _process_pose(self, frame):
        """处理姿势检测帧"""
        start_time = time.time()
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_time = time.time() - start_time
        self.pose_times.append(process_time)

        if not results.pose_landmarks:
            return frame, None, None, process_time

        try:
            # 遮挡检测
            is_occluded, status = check_occlusion(results.pose_landmarks.landmark)
            self._update_occlusion_counters(is_occluded)
            
            # 头部角度计算
            angle_info = calculate_head_angle(results.pose_landmarks.landmark, frame.shape)
            if angle_info[0] is not None:
                self.last_valid_angle = angle_info[0]

            # 绘制姿势关键点
            display_frame = frame.copy()
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            return display_frame, is_occluded, angle_info, process_time
        except Exception as e:
            print(f"姿势处理异常: {str(e)}")
            return frame, None, None, process_time

    def _update_occlusion_counters(self, is_occluded):
        """更新遮挡状态计数器"""
        if is_occluded:
            self.occlusion_counter = min(self.occlusion_counter + 1, OCCLUSION_FRAMES_THRESHOLD)
            self.clear_counter = max(0, self.clear_counter - 1)
        else:
            self.clear_counter = min(self.clear_counter + 1, CLEAR_FRAMES_THRESHOLD)
            self.occlusion_counter = max(0, self.occlusion_counter - 1)

    def _draw_pose_info(self, frame, is_occluded, angle_info, process_time):
        """在姿势画面绘制监测信息"""
        final_occlusion = self.occlusion_counter >= OCCLUSION_FRAMES_THRESHOLD
        valid_detection = self.clear_counter >= CLEAR_FRAMES_THRESHOLD

        # 绘制遮挡状态
        state_text = f"State: {'Occluded' if final_occlusion else 'Tracking'}"
        color = (0, 0, 255) if final_occlusion else (0, 255, 0)
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制角度信息
        try:
            if angle_info and valid_detection and not final_occlusion:
                angle, is_forward, points = angle_info
                status_color = (0, 0, 255) if is_forward else (0, 255, 0)
                text = f"Angle: {angle:.1f}° {'[BAD]' if is_forward else '[GOOD]'}"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.line(frame, tuple(points['mid_shoulder']), tuple(points['nose']), (0, 255, 0), 2)
            elif final_occlusion and self.last_valid_angle:
                text = f"Occluded | Last: {self.last_valid_angle:.1f}°"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except:
            pass

        # 绘制性能信息
        try:
            fps = 1 / np.mean(self.pose_times) if self.pose_times else 0
            time_text = f"Pose: {process_time*1000:.1f}ms ({fps:.1f}FPS)"
            cv2.putText(frame, time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        except:
            pass

        return frame

    def run(self):
        """主运行循环"""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 预处理帧
                processed_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
                pose_frame = processed_frame.copy()
                emotion_frame = processed_frame.copy()

                # 并行处理姿势和情绪检测
                pose_result, is_occluded, angle_info, pose_time = self._process_pose(pose_frame)
                emotion_state, face_landmarks, face_time = self.emotion_analyzer.analyze(emotion_frame)
                self.face_times.append(face_time)

                # 绘制姿势界面
                pose_display = self._draw_pose_info(pose_result, is_occluded, angle_info, pose_time)

                # 绘制情绪界面
                try:
                    if face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=emotion_frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    
                    # 计算并显示性能指标
                    face_fps = 1 / np.mean(self.face_times) if self.face_times else 0
                    time_text = f"Face: {face_time*1000:.1f}ms ({face_fps:.1f}FPS)"
                    cv2.putText(emotion_frame, time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                    
                    # 显示当前情绪状态
                    emotion_text = f"Emotion: {emotion_state.name}"
                    cv2.putText(emotion_frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 250), 2)
                except Exception as e:
                    print(f"情绪界面绘制异常: {str(e)}")

                # 显示窗口
                cv2.imshow('Posture Monitor', pose_display)
                cv2.imshow('Emotion Analysis', emotion_frame)

                if cv2.waitKey(5) == 27:  # ESC退出
                    break
        except Exception as e:
            print(f"运行时异常: {str(e)}")
        finally:
            # 资源释放
            self.cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            self.emotion_analyzer.face_mesh.close()

if __name__ == "__main__":
    monitor = PostureMonitor()
    monitor.run()
