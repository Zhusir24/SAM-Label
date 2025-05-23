from importlib import import_module
import gradio as gr

# 动态导入所有页面模块（这里以sam_label_demo为例）
pages = {
    "SAM标注演示": import_module("app.api.sam_label_demo").demo
}

with gr.Blocks(title="智能标注系统") as app:
    gr.Markdown("# 智能标注系统")
    with gr.Tabs():
        for page_name, page_content in pages.items():
            with gr.Tab(page_name):
                page_content()

if __name__ == "__main__":
    app.launch()