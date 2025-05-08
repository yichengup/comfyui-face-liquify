import numpy as np
import cv2

class FaceUtils:
    @staticmethod
    def calculate_face_center(landmarks):
        """计算人脸中心点"""
        return np.mean(landmarks, axis=0)
    
    @staticmethod
    def calculate_face_size(landmarks):
        """计算人脸大小"""
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        return np.array([x_max - x_min, y_max - y_min])
    
    @staticmethod
    def get_face_regions(landmarks):
        """获取面部不同区域的关键点"""
        regions = {
            'jaw': landmarks[0:17],    # 下巴轮廓
            'right_eyebrow': landmarks[17:22],  # 右眉毛
            'left_eyebrow': landmarks[22:27],   # 左眉毛
            'nose_bridge': landmarks[27:31],     # 鼻梁
            'nose_tip': landmarks[31:36],        # 鼻尖
            'right_eye': landmarks[36:42],       # 右眼
            'left_eye': landmarks[42:48],        # 左眼
            'outer_lip': landmarks[48:60],       # 外唇
            'inner_lip': landmarks[60:68]        # 内唇
        }
        return regions
    
    @staticmethod
    def smooth_transition(start_points, end_points, factor):
        """平滑过渡between两组点"""
        return start_points + (end_points - start_points) * factor
    
    @staticmethod
    def create_mask(shape, points, radius):
        """创建基于点的mask"""
        mask = np.zeros(shape[:2], dtype=np.float32)
        for point in points:
            cv2.circle(mask, (int(point[0]), int(point[1])), radius, 1.0, -1)
        return cv2.GaussianBlur(mask, (0, 0), radius/3)
    
    @staticmethod
    def blend_images(img1, img2, mask):
        """基于mask混合两张图片"""
        return img1 * (1 - mask) + img2 * mask 