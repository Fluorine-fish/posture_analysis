import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化MediaPipe姿势识别
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 视频处理函数
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_world_landmarks = []
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 转换颜色空间并进行姿势检测
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # 收集世界坐标
            if results.pose_world_landmarks:
                all_world_landmarks.append(results.pose_world_landmarks.landmark)
                
                # 计算头部角度
                ls = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                nose = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                
                mid_shoulder = np.array([(ls.x + rs.x)/2, (ls.y + rs.y)/2, (ls.z + rs.z)/2])
                nose_point = np.array([nose.x, nose.y, nose.z])
                
                vector = nose_point - mid_shoulder
                magnitude = np.linalg.norm(vector)
                if magnitude > 0:
                    cos_theta = vector[1] / magnitude  # Y轴分量
                    angle = math.degrees(math.acos(cos_theta))
                    
                    # 姿势提醒逻辑
                    alert_text = ""
                    if angle > 25:  # 可调整阈值
                        alert_text = "Bad Posture! (Angle: {:.1f}°)".format(angle)
                    else:
                        alert_text = "Good Posture (Angle: {:.1f}°)".format(angle)

            # 绘制2D关键点
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2))
                
                # 显示角度信息
                cv2.putText(image, alert_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            out.write(image)
            cv2.imshow('Posture Analysis', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return all_world_landmarks

# 3D可视化函数
def visualize_3d(landmarks_list):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取关键点数据
    nose_x = [lm[mp_pose.PoseLandmark.NOSE.value].x for lm in landmarks_list]
    nose_y = [lm[mp_pose.PoseLandmark.NOSE.value].y for lm in landmarks_list]
    nose_z = [lm[mp_pose.PoseLandmark.NOSE.value].z for lm in landmarks_list]
    
    # 绘制轨迹
    ax.plot(nose_x, nose_z, nose_y, 'r', label='Head Movement')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Head Movement Visualization')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output.mp4"
    
    # 处理视频并获取3D数据
    world_landmarks = process_video(input_video, output_video)
    
    # 显示3D可视化
    if world_landmarks:
        visualize_3d(world_landmarks)