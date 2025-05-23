import cv2
import random
import numpy as np

from collections import defaultdict

from app.core.fastsam.fastsam import FastSAM


class InvokeModel:
    def __init__(self):
        self.sam_model = FastSAM()
        # 使用字典存储不同图片的历史检测结果
        self.history = defaultdict(list)

    def get_random_color(self):
        """生成随机的BGR颜色"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def process_image(self, points, image_path, conf, iou):
        # 获取当前图片的历史记录
        history_entries = self.history[image_path]

        # 读取原始图像
        img = cv2.imread(image_path)

        # 处理历史记录
        for entry in history_entries:
            boxes, masks = entry['boxes'], entry['masks']
            color = entry['color']

            # 绘制历史检测框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 处理历史掩码
            for mask in masks:
                # 创建彩色掩码
                colored_mask = np.zeros_like(img)
                colored_mask[:] = color

                # 仅对掩码区域着色
                masked_region = cv2.bitwise_and(img, img, mask=mask)
                colored_region = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)

                # 混合效果
                blended_region = cv2.addWeighted(masked_region, 0.3, colored_region, 0.7, 0)

                # 将处理后的区域放回原图
                inverse_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(img, img, mask=inverse_mask)
                img = cv2.add(background, blended_region)

        # 进行新的检测
        new_boxes, new_masks = self.sam_model.inference_with_point([points], image_path, conf, iou)

        if len(new_boxes) > 0 or len(new_masks) > 0:
            # 为本次检测生成随机颜色
            new_color = self.get_random_color()

            # 保存到历史记录
            self.history[image_path].append({
                'boxes': new_boxes,
                'masks': new_masks,
                'color': new_color
            })

            # 绘制新检测框
            for box in new_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), new_color, 2)

            # 处理新掩码
            for mask in new_masks:
                # 创建彩色掩码
                colored_mask = np.zeros_like(img)
                colored_mask[:] = new_color

                # 仅对掩码区域着色
                masked_region = cv2.bitwise_and(img, img, mask=mask)
                colored_region = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)

                # 混合效果
                blended_region = cv2.addWeighted(masked_region, 0.3, colored_region, 0.7, 0)

                # 将处理后的区域放回原图
                inverse_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(img, img, mask=inverse_mask)
                img = cv2.add(background, blended_region)

        print(self.history)
        return img