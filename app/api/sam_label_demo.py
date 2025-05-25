import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import warnings

from app.core.invoke import InvokeModel

warnings.filterwarnings("ignore", category=UserWarning, module="gradio.helpers")

invoke_model = InvokeModel()


def save_uploaded_files(files):
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    for idx, file in enumerate(files):
        ext = os.path.splitext(file.name)[1][1:]
        temp_path = os.path.join(temp_dir, f"uploaded_{idx}.{ext}")
        with open(temp_path, "wb") as f:
            f.write(open(file.name, "rb").read())
        saved_paths.append(temp_path)
    return saved_paths


def image_click_handler(img, evt: gr.SelectData):
    if img is None or evt is None:
        return None, None
    try:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.ellipse([(evt.index[0] - 5, evt.index[1] - 5),
                      (evt.index[0] + 5, evt.index[1] + 5)],
                     fill="yellow")
        return np.array(pil_img), evt.index
    except Exception as e:
        print(f"Error in click handler: {e}")
        return None, None


def demo():
    current_image = gr.State()
    click_pos = gr.State()
    click_enabled = gr.State(True)
    image_paths = gr.State([])
    has_clicked = gr.State(False)
    last_result = gr.State(None)
    last_click_pos = gr.State(None)
    new_click_occurred = gr.State(False)  # 新增状态：标记是否有新点击

    with gr.Blocks() as demo:
        alert = gr.Textbox(visible=False, label="提示信息")

        with gr.Row():
            with gr.Column(scale=1):
                upload_button = gr.File(
                    file_count="multiple",
                    file_types=["image", ".jpg", ".png"],
                    label="上传图片"
                )
                gallery = gr.Gallery(
                    label="图片列表",
                    columns=2,
                    preview=False,
                    object_fit="scale-down"
                )

                with gr.Accordion("检测参数", open=True):
                    conf_slider = gr.Slider(0, 1, value=0.4, label="Confidence")
                    iou_slider = gr.Slider(0, 1, value=0.9, label="IOU")

            with gr.Column(scale=3):
                main_image = gr.Image(
                    label="标注区域",
                    interactive=True,
                    type="numpy",
                )
                detect_btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=3):
                result_image = gr.Image(label="检测结果")

        def update_gallery(files):
            if not files:
                return None, None
            saved_paths = save_uploaded_files(files)
            return saved_paths, saved_paths

        upload_button.upload(
            fn=update_gallery,
            inputs=[upload_button],
            outputs=[gallery, image_paths]
        )

        def select_image(evt: gr.SelectData, paths):
            if not paths or evt.index >= len(paths):
                return None, None
            try:
                img = Image.open(paths[evt.index]).convert("RGB")
                return np.array(img), paths[evt.index]
            except Exception as e:
                print(f"Error loading image: {e}")
                return None, None

        gallery.select(
            fn=select_image,
            inputs=[image_paths],
            outputs=[main_image, current_image]
        )

        def handle_click(img, evt: gr.SelectData):
            result_img, pos = image_click_handler(img, evt)
            if result_img is None or pos is None:
                return img, None, False, False, None, False
            return result_img, pos, True, True, pos, True  # 新增：标记有新点击

        main_image.select(
            fn=handle_click,
            inputs=[main_image],
            outputs=[main_image, click_pos, click_enabled, has_clicked, last_click_pos, new_click_occurred]
        )

        def run_detection(image_path, pos, conf, iou, clicked, last_pos, last_res, new_click):
            # 如果没有点击过图片
            if not clicked:
                return last_res if last_res is not None else np.zeros((100, 100, 3),
                                                                      dtype=np.uint8), False, False, last_pos, "请先点击图片选择检测位置！", last_res, False

            # 如果没有新点击且已有结果
            if not new_click and last_res is not None:
                return last_res, False, False, last_pos, "请点击新位置后再检测！", last_res, False

            # 正常检测流程
            if not image_path or pos is None:
                return last_res if last_res is not None else np.zeros((100, 100, 3),
                                                                      dtype=np.uint8), False, False, last_pos, "请先点击图片选择检测位置！", last_res, False

            try:
                result = invoke_model.process_image(image_path=image_path, points=pos, conf=conf, iou=iou)
                if result.shape[-1] == 3:  # 确保是BGR格式
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    result_rgb = result.copy()
                if result_rgb.dtype != np.uint8:
                    result_rgb = (result_rgb * 255).astype(np.uint8)
                return result_rgb, True, False, pos, "", result, False  # 重置新点击标记
            except Exception as e:
                print(f"Detection error: {e}")
                return last_res if last_res is not None else np.zeros((100, 100, 3),
                                                                      dtype=np.uint8), True, False, last_pos, f"检测出错: {str(e)}", last_res, False

        detect_btn.click(
            fn=run_detection,
            inputs=[current_image, click_pos, conf_slider, iou_slider, has_clicked, last_click_pos, last_result,
                    new_click_occurred],
            outputs=[result_image, click_enabled, has_clicked, last_click_pos, alert, last_result, new_click_occurred]
        )

    return demo


if __name__ == "__main__":
    demo().launch()