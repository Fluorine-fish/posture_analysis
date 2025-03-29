# 实时姿势监测与健康提醒系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)

本系统是基于计算机视觉的实时姿势监测工具，主要用于检测头部前倾角度和身体遮挡状态，适用于办公健康提醒、康复训练监测等场景。

## 功能特性

- 🎯 实时头部姿势角度计算（精确到0.1度）
- 🛡️ 多部位遮挡检测（肩部/面部/完全遮挡）
- 📷 自动摄像头检测（支持0-9号设备）
- 📊 智能状态跟踪（连续帧验证机制）
- 🚨 可视化警报系统（颜色编码提示）
- ⚙️ 高度可配置参数（阈值/灵敏度调节）

## 安装步骤

### 依赖安装
```bash
pip install opencv-python mediapipe numpy
```

Linux系统摄像头权限配置
``` bash
sudo usermod -aG video $USER  # 将当前用户加入video组
sudo reboot  # 重启生效
```
### 使用说明
快速启动
```bash 
python posture_monitor.py
```
### 界面说明

界面示意图

    状态栏：显示系统检测状态（跟踪/遮挡）

    角度显示：实时头部倾斜角度与健康评级

    骨架视图：人体姿势骨架可视化

    参考线：头部与肩部中点的连线

    调试信息：帧计数器与系统状态

### 配置参数
| 参数名称 |	默认值 |	说明 |
|-----|-----|-----|
|OCCLUSION_FRAMES_THRESHOLD	| 4 | 触发遮挡报警的连续帧数 |
|CLEAR_FRAMES_THRESHOLD	| 3	| 恢复正常检测的连续帧数 |
|HEAD_ANGLE_THRESHOLD |	50°	| 头部前倾报警阈值 |
|VISIBILITY_THRESHOLD |	0.5	| 关键点可见性阈值 |
|SET_MIN_DETECTION_CONFIDENCE | 0.8 | 姿势检测置信度 |
|SET_MIN_TRACKING_CONFIDENCE | 0.2 | 姿势跟踪置信度 |
### 技术细节 核心算法

    头部角度计算：基于肩部中点与鼻尖的向量夹角
```    math

    θ = arctan(Δx/Δy) × (180/π)
```
    遮挡检测：综合评估以下关键点可见性：

        双肩（LEFT_SHOULDER, RIGHT_SHOULDER）

        面部（NOSE, LEFT_EYE, RIGHT_EYE）

### 性能优化

    多分辨率处理（1280×720输入 → 640×360输出）

    MediaPipe GPU加速（自动启用）

    异步状态跟踪机制

### 常见问题
摄像头无法识别

    检查设备连接状态

    尝试手动指定摄像头索引：
``` python
    cap = cv2.VideoCapture(0)  # 修改0为实际设备号
```

### 检测不准确

    调整HEAD_ANGLE_THRESHOLD参数

    确保检测环境光照充足

    保持与摄像头1-2米的距离

### 依赖安装失败

使用清华镜像源加速安装
``` bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package_name]
```
