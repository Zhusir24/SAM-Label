import os

import cv2
import torch
from ultralytics import FastSAM

from app.conf import config


class SAMInit:
    """SAM初始化模块"""
    def __init__(self):
        self.model_path = os.getenv('FASTSAM_MODEL_PATH', config.fastsam_model_path)
        self.model = FastSAM(self.model_path)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.conf = 0.4
        self.iou = 0.9

    def model_parameter(self, conf:float, iou:float):
        """
        模型推理参数更新
        :param conf:
        :param iou:
        :return:
        """
        self.conf = conf
        self.iou = iou

    def inference_with_point(self, points:[list], image_path):
        """
        通过点进行推理
        :param points:
        :param image_path:
        :return:
        """
        results = self.model(source=image_path, device=self.device, points=points, labels=[1], conf=self.conf, iou=self.iou)
        return results

if __name__ == "__main__":
    os.environ['FASTSAM_MODEL_PATH'] = '/Users/zhusir/Documents/Application/Python/SAM_Label/app/model/FastSAM-s.pt'
    sam = SAMInit()
    image_path = '/Users/zhusir/Documents/Application/Python/SAM_Label/app/tmp/bus.jpg'
    results = sam.inference_with_point(points=[[800,800]], image_path=image_path)
    results[0].plot()
    for result in results:
        img = result.orig_img.copy()  # 原始图像 (BGR格式)
        masks = result.masks.data.cpu().numpy()  # 掩码 [N, H, W]

        for mask in masks:
            # 调整掩码尺寸（如果 FastSAM 返回的掩码是缩放的）
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # 将掩码转为二值化（0或255）
            mask_uint8 = (mask * 255).astype("uint8")  # [H, W], 0或255

            # 生成彩色掩码（仅用于着色区域）
            colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)  # [H, W, 3]

            # --- 关键步骤：仅对掩码区域着色 ---
            # 1. 提取掩码区域的原图像素
            masked_region = cv2.bitwise_and(img, img, mask=mask_uint8)  # 原图在掩码区域的部分

            # 2. 提取掩码区域的彩色效果
            colored_region = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_uint8)

            # 3. 将彩色效果与原图结合（加权混合）
            blended_region = cv2.addWeighted(masked_region, 0.3, colored_region, 0.7, 0)

            # 4. 将处理后的掩码区域放回原图
            inverse_mask = cv2.bitwise_not(mask_uint8)
            background = cv2.bitwise_and(img, img, mask=inverse_mask)  # 非掩码区域保持原图
            img = cv2.add(background, blended_region)  # 合并

        # 显示结果
        cv2.imshow("Masked Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()