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

    def inference_with_point(self, points:[list], image_path, conf:float, iou:float):
        """
        通过点进行推理
        :param points:
        :param image_path:
        :return:
        """
        results = self.model(source=image_path, device=self.device, points=points, labels=[0], conf=conf, iou=iou)

        return self._postprocess_with_point(results=results,image_path=image_path)

    def _postprocess_with_point(self, results, image_path):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        boxes = []
        masks = []
        for result in results:
            # 获取边界框 (xyxy格式)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
            boxes.extend(boxes_xyxy)

            # 获取掩码并转换为二值图像
            if result.masks is not None:
                for mask in result.masks:
                    # 将掩码转换为与原图相同尺寸的二值图像
                    mask_np = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h))
                    mask_binary = (mask_resized > 0).astype('uint8') * 255
                    masks.append(mask_binary)

        return [boxes[0]], [masks[0]]


if __name__ == "__main__":
    os.environ['FASTSAM_MODEL_PATH'] = '/home/ohos/Desktop/SAM-Label/app/model/FastSAM-s.pt'
    sam = SAMInit()
    image_path = '/home/ohos/Desktop/SAM-Label/app/tmp/bus.jpg'

    # 执行推理并获取boxes和masks
    boxes, masks = sam.inference_with_point(points=[[800, 800]], image_path=image_path)

    # 读取原始图像
    img = cv2.imread(image_path)

    # 绘制检测框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 处理每个掩码
    for mask in masks:
        # 生成彩色掩码（仅用于着色区域）
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # [H, W, 3]

        # --- 关键步骤：仅对掩码区域着色 ---
        # 1. 提取掩码区域的原图像素
        masked_region = cv2.bitwise_and(img, img, mask=mask)  # 原图在掩码区域的部分

        # 2. 提取掩码区域的彩色效果
        colored_region = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)

        # 3. 将彩色效果与原图结合（加权混合）
        blended_region = cv2.addWeighted(masked_region, 0.3, colored_region, 0.7, 0)

        # 4. 将处理后的掩码区域放回原图
        inverse_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(img, img, mask=inverse_mask)  # 非掩码区域保持原图
        img = cv2.add(background, blended_region)  # 合并

    # 显示结果
    cv2.imshow("Detection and Segmentation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()