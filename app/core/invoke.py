import cv2
import random
import numpy as np
from collections import defaultdict
from app.core.fastsam.fastsam import FastSAM


class InvokeModel:
    def __init__(self):
        self.sam_model = FastSAM()
        self.history = defaultdict(list)

    def get_random_color(self):
        return (0, 0, 255)  # 固定红色用于测试

    def draw_masks(self, base_img, mask, color):
        """ 通用掩码绘制方法 """
        mask = mask.astype(np.uint8)
        color_layer = np.zeros_like(base_img)
        color_layer[mask == 255] = color

        blended = cv2.addWeighted(color_layer, 0.3, base_img, 0.7, 0)

        mask_inv = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
        foreground = cv2.bitwise_and(blended, blended, mask=mask)
        return cv2.add(background, foreground)

    def process_image(self, points, image_path, conf, iou):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        h, w = img.shape[:2]
        processed_img = img.copy()

        # 处理历史记录
        for entry in self.history[image_path]:
            boxes, masks, color = entry['boxes'], entry['masks'], entry['color']
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)
            for mask in masks:
                processed_img = self.draw_masks(processed_img, mask, color)

        # 新检测
        new_boxes, new_masks = self.sam_model.inference_with_point([points], image_path, conf, iou)
        if new_boxes or new_masks:
            new_color = self.get_random_color()
            resized_masks = []

            for mask in new_masks:
                _, binary_mask = cv2.threshold(mask.astype(np.float32), 0.5, 1.0, cv2.THRESH_BINARY)
                resized_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                resized_masks.append(resized_mask)
                # cv2.imwrite('debug_new_mask.jpg', resized_mask)  # 保存新掩码

            self.history[image_path].append({
                'boxes': new_boxes,
                'masks': resized_masks,
                'color': new_color
            })

            for box in new_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), new_color, 2)
                # cv2.imwrite('debug_box.jpg', processed_img)  # 调试框颜色

            for mask in resized_masks:
                processed_img = self.draw_masks(processed_img, mask, new_color)

        # cv2.imwrite('debug_final.jpg', processed_img)  # 保存最终结果
        return processed_img