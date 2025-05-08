import torch
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from .utils import FaceUtils

class FaceLiquifyNode:
    """视频人脸变形节点 - 支持整体脸部变形效果和批量处理"""
    
    # 添加英文到中文的映射
    EN_TO_CN_EFFECT = {
        "Fat Face": "胖脸",
        "Thin Face": "大眼",
        "Big Face": "尖脸",
        "Small Face": "小脸"
    }
    
    # 添加英文到中文的排序映射
    EN_TO_CN_ORDER = {
        "Large to Small": "大到小",
        "Small to Large": "小到大",
        "Left to Right": "从左到右",
        "Right to Left": "从右到左"
    }
    
    def __init__(self):
        self.face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.face_utils = FaceUtils()
        self.last_face_infos = None  # 修改为存储多个人脸信息
        self.frame_cache = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "effect_type": (["Fat Face", "Thin Face", "Big Face", "Small Face"],),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eye_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smooth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "area_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "frame_blend": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_order": (["Large to Small", "Small to Large", "Left to Right", "Right to Left"],),
                "face_indices": ("STRING", {"default": "all", "placeholder": "all or 0,1,2..."})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_face_liquify"
    CATEGORY = "Face Liquify/image"

    def get_face_areas(self, img, frame_idx=None):
        """获取所有人脸区域信息"""
        faces = self.face_analyzer.get(img)
        
        if not faces:
            # 如果检测失败，使用上一帧的信息
            if self.last_face_infos is not None:
                return self.last_face_infos
            return None
        
        face_infos = []
        for face in faces:
            # 获取人脸框和关键点
            bbox = face.bbox
            landmarks = face.landmark_2d_106
            
            # 计算人脸中心和大小
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            
            face_info = {
                'center': (center_x, center_y),
                'size': (face_width, face_height),
                'bbox': bbox,
                'landmarks': landmarks,
                'area': face_width * face_height,
                'position_x': center_x
            }
            face_infos.append(face_info)
        
        # 根据选择的排序方式排序
        if self.face_order == "Large to Small":
            face_infos.sort(key=lambda x: x['area'], reverse=True)
        elif self.face_order == "Small to Large":
            face_infos.sort(key=lambda x: x['area'])
        elif self.face_order == "Left to Right":
            face_infos.sort(key=lambda x: x['position_x'])
        elif self.face_order == "Right to Left":
            face_infos.sort(key=lambda x: x['position_x'], reverse=True)
        
        # 如果是视频处理，进行帧间平滑
        if frame_idx is not None and self.last_face_infos is not None:
            face_infos = self.smooth_face_infos(self.last_face_infos, face_infos)
        
        self.last_face_infos = face_infos
        return face_infos

    def smooth_face_infos(self, last_infos, current_infos, smooth_factor=0.3):
        """平滑多个人脸信息"""
        if len(last_infos) != len(current_infos):
            return current_infos
            
        smoothed_infos = []
        for last, current in zip(last_infos, current_infos):
            # 平滑中心点
            last_center = np.array(last['center'])
            current_center = np.array(current['center'])
            smoothed_center = last_center + (current_center - last_center) * smooth_factor
            
            # 平滑大小
            last_size = np.array(last['size'])
            current_size = np.array(current['size'])
            smoothed_size = last_size + (current_size - last_size) * smooth_factor
            
            smoothed_info = {
                'center': tuple(smoothed_center),
                'size': tuple(smoothed_size),
                'bbox': current['bbox'],  # 保持原始检测框
                'landmarks': current['landmarks'],  # 保持原始关键点
                'area': current['area'],
                'position_x': current['position_x']
            }
            smoothed_infos.append(smoothed_info)
            
        return smoothed_infos

    def filter_face_indices(self, face_infos, indices_str):
        """根据索引过滤人脸"""
        if indices_str.lower() == "all":
            return face_infos
            
        try:
            indices = [int(i.strip()) for i in indices_str.split(",")]
            filtered_infos = []
            for idx in indices:
                if 0 <= idx < len(face_infos):
                    filtered_infos.append(face_infos[idx])
            return filtered_infos if filtered_infos else face_infos
        except:
            return face_infos

    def apply_eye_enlargement(self, img, landmarks, strength):
        """应用大眼效果"""
        height, width = img.shape[:2]
        result = img.copy()
        
        # 获取左右眼关键点
        left_eye = landmarks[60:68]  # 左眼轮廓点
        right_eye = landmarks[68:76]  # 右眼轮廓点
        
        # 处理每只眼睛
        for eye_points in [left_eye, right_eye]:
            # 计算眼睛中心
            center_x = np.mean(eye_points[:, 0])
            center_y = np.mean(eye_points[:, 1])
            
            # 计算眼睛大小
            eye_width = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
            eye_height = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
            
            # 设置变形半径（略大于眼睛尺寸）
            radius = max(eye_width, eye_height) * 1.5
            
            # 应用局部放大效果
            result = self.liquify_effect(
                result,
                int(center_x),
                int(center_y),
                int(radius),
                strength,
                "PUSH",
                1.2
            )
        
        return result

    def apply_face_effect(self, img, face_infos, effect_type, strength, eye_scale, area_scale, frame_idx=None):
        """应用整体脸部变形效果，支持多人脸"""
        height, width = img.shape[:2]
        result = img.copy()
        
        if face_infos is None:
            return result
            
        # 过滤要处理的人脸
        face_infos = self.filter_face_indices(face_infos, self.face_indices)
        
        # 处理每个人脸
        for face_info in face_infos:
            center_x, center_y = face_info['center']
            face_width, face_height = face_info['size']
            landmarks = face_info['landmarks']
            
            # 计算变形半径（基于脸部大小）
            radius = max(face_width, face_height) * area_scale
            
            # 根据效果类型设置变形模式和参数
            if effect_type == "Fat Face":
                result = self.liquify_effect(
                    result,
                    int(center_x),
                    int(center_y),
                    int(radius),
                    strength,
                    "PULL",
                    0.8
                )
            elif effect_type == "Thin Face":
                result = self.apply_eye_enlargement(result, landmarks, eye_scale)
            elif effect_type == "Big Face":
                result = self.liquify_effect(
                    result,
                    int(center_x),
                    int(center_y),
                    int(radius),
                    strength,
                    "PINCH",
                    0.9
                )
            elif effect_type == "Small Face":
                result = self.liquify_effect(
                    result,
                    int(center_x),
                    int(center_y),
                    int(radius),
                    strength * 0.8,
                    "PUSH",
                    1.2
                )
        
        return result

    def liquify_effect(self, img, center_x, center_y, radius, strength, mode, feather):
        """液化变形效果实现"""
        height, width = img.shape[:2]
        
        y, x = np.indices((height, width))
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 使用更平滑的影响力曲线
        influence = np.clip(1.0 - distance / (radius * feather), 0, 1)
        influence = influence * influence * (3 - 2 * influence)
        
        if mode == "PUSH":
            scale = 1.0 + strength * influence
        elif mode == "PULL":
            scale = 1.0 - strength * influence
        elif mode == "PINCH":
            scale = 1.0 - strength * influence * (distance / radius)
        else:
            return img
            
        x_offset = dx * (scale - 1)
        y_offset = dy * (scale - 1)
        
        x_new = x + x_offset
        y_new = y + y_offset
        
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        return cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def process_batch(self, images, effect_type, strength, eye_scale, smooth, area_scale, frame_blend):
        """处理图像批次（视频帧）"""
        batch_size = images.shape[0]
        processed_frames = []
        
        for i in range(batch_size):
            # 转换当前帧格式
            current_frame = (images[i] * 255).astype(np.uint8)
            
            # 获取人脸区域信息（带帧间平滑）
            face_infos = self.get_face_areas(current_frame, frame_idx=i)
            
            if face_infos is None:
                processed_frames.append(images[i])
                continue
            
            # 应用变形效果
            result = self.apply_face_effect(
                current_frame, 
                face_infos, 
                effect_type, 
                strength, 
                eye_scale,
                area_scale,
                frame_idx=i
            )
            
            # 平滑处理
            if smooth > 0:
                result = cv2.GaussianBlur(result, (0, 0), smooth * 3)
            
            # 帧间混合
            if i > 0 and frame_blend > 0:
                prev_frame = processed_frames[-1]
                result = cv2.addWeighted(
                    result, 
                    1 - frame_blend,
                    (prev_frame * 255).astype(np.uint8),
                    frame_blend,
                    0
                )
            
            # 转换回tensor格式
            result = result.astype(np.float32) / 255.0
            processed_frames.append(result)
        
        # 清理缓存
        if len(self.frame_cache) > 100:
            self.frame_cache.clear()
        
        return torch.stack([torch.from_numpy(frame) for frame in processed_frames])

    def apply_face_liquify(self, image, effect_type, strength, eye_scale, smooth, area_scale, frame_blend, face_order="Large to Small", face_indices="all"):
        try:
            # 保存排序方式和索引选择
            self.face_order = face_order
            self.face_indices = face_indices
            
            # 检查是否是批量输入（视频帧）
            is_batch = len(image.shape) == 4
            
            if is_batch:
                # 批量处理视频帧
                result = self.process_batch(
                    image.cpu().numpy(),
                    effect_type,
                    strength,
                    eye_scale,
                    smooth,
                    area_scale,
                    frame_blend
                )
            else:
                # 单帧处理
                img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
                face_infos = self.get_face_areas(img)
                
                if face_infos is None:
                    return (image,)
                    
                # 应用效果
                result = self.apply_face_effect(
                    img,
                    face_infos,
                    effect_type,
                    strength,
                    eye_scale,
                    area_scale
                )
                
                # 平滑处理
                if smooth > 0:
                    result = cv2.GaussianBlur(result, (0, 0), smooth * 3)
                
                # 转换回tensor格式
                result = torch.from_numpy(result.astype(np.float32) / 255.0)
                result = result.unsqueeze(0)
                
            return (result,)
            
        except Exception as e:
            print(f"Error in face liquify: {str(e)}")
            return (image,) 