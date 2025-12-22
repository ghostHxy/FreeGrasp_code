import os
import cv2
import trimesh
import numpy as np
import gradio as gr
from pathlib import Path

from run import compute_grasp_pose
from models.FGC_graspnet.utils.data_utils import CameraInfo

TMP_DIR = os.path.join(os.getcwd(), 'data', 'demo')
EXAMPLES_DIR = os.path.join(os.getcwd(), 'data', 'real_examples', 'hard')
os.makedirs(TMP_DIR, exist_ok=True)


def read_file(path):
    if path is None:
        return "文件未找到"
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"读取文件错误: {str(e)}"


def exists(label, ext):
    file = os.path.join(TMP_DIR, f"{label}.{ext}")
    if os.path.exists(file):
        return file
    return None


def get_grasp_pose(text_prompt, rgb_image, depth_file, _fx, _fy, _cx, _cy, _scale):
    tmp = Path(TMP_DIR)
    for file in tmp.iterdir():
        if file.is_file():
            file.unlink()

    if depth_file is None:
        raise ValueError("请上传深度图文件")

    with np.load(depth_file.name) as data:
        depth = data['depth']

    np.savez_compressed(f'{TMP_DIR}/depth.npz', depth=depth)

    rgb_image = np.array(rgb_image)
    _height, _width, _ = rgb_image.shape
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{TMP_DIR}/image.png', rgb_image)

    prompt_path = f"{TMP_DIR}/task.txt"
    with open(prompt_path, "w") as file:
        file.write(text_prompt)

    camera = CameraInfo(width=_width, height=_height, fx=_fx,
                        fy=_fy, cx=_cx, cy=_cy, scale=_scale)

    try:
        grasp_dict = compute_grasp_pose(TMP_DIR, camera)
    except Exception as e:
        error_msg = f"计算抓取位姿出错: {str(e)}"
        print(f"⚠️ {error_msg}")
        grasp_dict = {"error": error_msg}

    molmo = exists("molmo_label", "png")

    gpt_path = exists("log", "txt")
    gpt = read_file(gpt_path)

    mask = exists("image_mask_1", "png")
    pcd = create_pcd()

    return gr.update(value=molmo), gpt, gr.update(value=mask), gr.update(value=pcd), grasp_dict


def create_pcd():
    glbscene = trimesh.Scene()

    cloud_file_path = exists("cloud", "ply")
    grasp_file_path = exists("grasp", "obj")

    if grasp_file_path != None:
        point_cloud = trimesh.load(cloud_file_path)
        glbscene.add_geometry(point_cloud)

    if grasp_file_path != None:
        grasp_pose = trimesh.load(grasp_file_path)
        glbscene.add_geometry(grasp_pose)

    if len(glbscene.geometry) == 0:
        return None

    glb_path = os.path.join(TMP_DIR, f'visualization.glb')
    glbscene.export(glb_path)

    return glb_path


def interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 🦾 FreeGrasp: 自由形式语言驱动的机器人推理与抓取系统")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 输入")
                input_text = gr.Textbox(label="任务描述", placeholder="例如：拿起红色的杯子")
                rgb_input = gr.Image(label="RGB 图像", type="pil")
                depth_input = gr.File(
                    label="深度图 (.npz)", file_types=[".npz"])

                gr.Markdown("## 相机内参")
                with gr.Group():
                    with gr.Row(equal_height=True):
                        fx = gr.Number(label="fx (焦距x)", value=912.481)
                        cx = gr.Number(label="cx (主点x)", value=644.943)

                    with gr.Row(equal_height=True):
                        fy = gr.Number(label="fy (焦距y)", value=910.785)
                        cy = gr.Number(label="cy (主点y)", value=353.497)

                    with gr.Row():
                        scale = gr.Number(label="scale (深度缩放)", value=1000.0)

                submit_btn = gr.Button("开始处理", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 处理流程")
                molmo_output = gr.Image(label="Molmo 目标检测")
                gpt_output = gr.Textbox(label="Qwen 推理结果", lines=1)
                mask_output = gr.Image(label="LangSAM 分割结果")

            with gr.Column(scale=2):
                gr.Markdown("## 输出结果")
                grasp_output = gr.Model3D(label="GraspNet 抓取姿态")
                grasp_json_output = gr.JSON(label="抓取位姿数据")

        submit_btn.click(fn=get_grasp_pose, inputs=[input_text, rgb_input, depth_input, fx, fy, cx, cy, scale],
                         outputs=[molmo_output, gpt_output, mask_output, grasp_output, grasp_json_output])
        
        examples = []

        for i in range(1, 4):
            example = []
            tmp_path = os.path.join(EXAMPLES_DIR, str(i))

            task = read_file(os.path.join(tmp_path, "task.txt"))
            example.append(task)

            example.append(os.path.join(tmp_path, "image.png"))
            example.append(os.path.join(tmp_path, "depth.npz"))
            examples.append(example)

        gr.Examples(
            examples=examples,
            inputs=[input_text, rgb_input, depth_input, fx, fy, cx, cy, scale],
            cache_examples=False,  
        )

    demo.launch()


if __name__ == "__main__":
    interface()