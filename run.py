import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image

from grasp_model import grasp_model
from models.langsam import langsamutils
from models.FGC_graspnet.utils.data_utils import CameraInfo
from molmo_eval import process_and_send_to_gpt
from utils.utils import *
from utils.config import *
from utils.graspnet_utils import get_correct_pose

# 可选的调试可视化模块（导入失败不影响主流程）
try:
    from utils.debug_visualizer import DebugVisualizer
    DEBUG_VIS_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Debug visualizer not available: {e}")
    DEBUG_VIS_AVAILABLE = False
    DebugVisualizer = None


def compute_grasp_pose(path, camera_info, debug_vis=True):
    """
    Compute grasp pose for a given scene folder.
    Expects image.png, depth.npz, task.txt inside `path`.

    Args:
        path: 场景文件夹路径
        camera_info: 相机参数
        debug_vis: 是否启用调试可视化（默认True）
    """
    parser = argparse.ArgumentParser('RUN an experiment with real data', parents=[get_args_parser()])
    args = parser.parse_args()

    # 初始化调试可视化器（安全初始化，失败不影响主流程）
    visualizer = None
    if debug_vis and DEBUG_VIS_AVAILABLE:
        try:
            visualizer = DebugVisualizer(output_dir=path, enabled=True)
        except Exception as e:
            print(f"[Warning] Failed to init DebugVisualizer: {e}")

    try:
        # --------------------------
        # 路径处理
        # --------------------------
        path = str(path)
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)
        path = os.path.normpath(path)

        image_path = os.path.join(path, "image.png")
        depth_path = os.path.join(path, "depth.npz")
        text_path = os.path.join(path, "task.txt")

        # --------------------------
        # 文件存在检查
        # --------------------------
        for fpath, ftype in [(image_path, "image"), (depth_path, "depth"), (text_path, "task")]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"{ftype.capitalize()} file not found: {fpath}")

        # --------------------------
        # 图片加载检查
        # --------------------------
        image = Image.open(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = image.convert("RGB")
        print(f"Loaded image: {image_path}, size: {image.size}")

        # --------------------------
        # Molmo 处理
        # --------------------------
        prompt = "Point out the objects in the red rectangle on the table."
        print(f"[DEBUG] Starting Molmo processing for image: {image_path}")
        base64_labeled_image, labeled_text = process_and_send_to_gpt(image_path, prompt, path)
        print(f"[DEBUG] Molmo returned: base64_labeled_image type={type(base64_labeled_image)}, labeled_text type={type(labeled_text)}")
        if base64_labeled_image is None or labeled_text is None:
            raise ValueError(f"Molmo processing failed for image: {image_path}")

        # --------------------------
        # cv2 读取原图
        # --------------------------
        img_ori = cv2.imread(image_path)
        if img_ori is None:
            raise ValueError(f"cv2 failed to load image: {image_path}")
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        # --------------------------
        # 文本读取
        # --------------------------
        with open(text_path, 'r') as f:
            input_text = f.read()

        # --------------------------
        # 深度读取
        # --------------------------
        depth_data = np.load(depth_path)
        if "depth" not in depth_data:
            raise ValueError(f"No 'depth' in {depth_path}")
        depth_ori = depth_data["depth"]

        # --------------------------
        # PIL 图片，用于 LangSAM
        # --------------------------
        image_pil = langsamutils.load_image(image_path)
        print(f"[DEBUG] Depth data shape: {depth_ori.shape}, type: {type(depth_ori)}")
        print(f"[DEBUG] PIL image size: {image_pil.size}, type: {type(image_pil)}")

        # --------------------------
        # GPT 推理部分
        # --------------------------
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robotic system for bin picking, using a parallel gripper. "
                    "I labeled all objects id in the image.\n"
                    "You have two possible actions:\n"
                    "1. remove obstacle, object_id: This action moves the specified object out of the way so it does not interfere with grasping the desired target object. This action can only be performed if the specified object is free of obstacles.\n"
                    "2. pick object, object_id: This action picks up the specified object. It can only be performed if the object is free of obstacles.\n"
                    "An object is considered an obstacle if it occludes another object.\n"
                    "Task: Given a target object description as input, determine the first object that needs to be grasped to enable picking the target object. "
                    "If the target object is free of obstacles, return the target object ID itself. "
                    "Otherwise, identify an object that is occluding the target and is itself free of obstacles. "
                    "If multiple objects could be removed, return any one valid option.\n"
                    "Output Format: The output should only be the object ID of the first object to grasp, formatted as: [object_id, color class_name]\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_labeled_image}"}}
                ]
            }
        ]

        from utils.config import QWEN_MODEL

        # [Debug Vis 1] 保存 Qwen prompts
        if visualizer:
            visualizer.save_qwen_prompts(messages, input_text)

        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=713,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = response.choices[0].message.content
        print(f"[DEBUG] Qwen output type: {type(output)}, content: {output}")
        result = process_grasping_result(output, input_text)
        print(f"[DEBUG] process_grasping_result: {result}")

        # [Debug Vis 2] 保存 Qwen output
        if visualizer:
            visualizer.save_qwen_output(output, result)

        goal = result['class_name']
        goal_id = result['selected_object_id']
        print(f"[DEBUG] labeled_text type: {type(labeled_text)}, content: {labeled_text[:200] if isinstance(labeled_text, str) else labeled_text}")
        # Get image dimensions for coordinate conversion
        image_w, image_h = image.size
        goal_coor = get_coordinates(labeled_text, goal_id, image_w, image_h)
        print(f"[DEBUG] goal_coor for goal_id={goal_id}: {goal_coor}")
        if goal_coor is None:
            raise ValueError(f"Failed to find coordinates for goal_id={goal_id}")

        with open(f"{path}/log.txt", "a") as f:
            f.write(f"I have to remove the object with id = {str(goal_id)}, named {goal}\n")

        # --------------------------
        # LangSAM Actor
        # --------------------------
        print(f"base64_labeled_image type: {type(base64_labeled_image)}")
        print(f"labeled_text type: {type(labeled_text)}")
        print(f"image_pil type: {type(image_pil)}")
        print(f"[DEBUG] Starting LangSAM Actor prediction for goal: {goal}")
        masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)
        print(f"masks: {masks}, boxes: {boxes}")
        print(f"[DEBUG] LangSAM returned: masks={masks}, boxes={boxes}")
        if masks is None or boxes is None or len(masks) == 0:
            raise ValueError(f"LangSAM prediction failed for {image_path}, goal={goal}")

        # [Debug Vis 3] 保存 LangSAM 所有掩码可视化
        if visualizer:
            visualizer.save_langsam_all_masks(image_pil, masks, boxes, logits, goal)

        goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
        print(f"[DEBUG] goal_mask type={type(goal_mask)}, mask_index={mask_index}")
        if goal_mask is None:
            raise ValueError(f"Failed to get goal mask for {image_path}, goal_coor={goal_coor}")

        # [Debug Vis 4] 保存坐标匹配可视化
        if visualizer:
            visualizer.save_coordinate_matching(image_pil, masks, goal_coor, mask_index)

        goal_bbox = boxes[mask_index].cpu().numpy()
        cropping_box = create_cropping_box_from_boxes(goal_bbox, (img_ori.shape[1], img_ori.shape[0]))

        # [Debug Vis 5] 保存最终选中的掩码
        if visualizer:
            visualizer.save_final_selected_mask(image_pil, goal_mask, goal_bbox)

        goal_mask = goal_mask.unsqueeze(0)
        print(f"LangSAM masks: {len(masks)}, boxes: {len(boxes)}")
        print(f"Goal coordinates: {goal_coor}, mask_index: {mask_index}")

        if args.viz:
            visualize_cropping_box(img_ori, cropping_box)

        langsam_actor.save(masks, boxes, phrases, logits, image_pil, path, viz=args.viz)

        # --------------------------
        # GraspNet
        # --------------------------
        # [Debug Vis 6 & 7] 保存 RGB + 掩码 + 点云可视化（完全隔离，不影响主流程）
        if visualizer:
            try:
                visualizer.save_rgb_with_mask_for_pointcloud(img_ori, goal_mask, cropping_box)
                # 生成完整场景点云用于可视化
                from models.FGC_graspnet.utils.data_utils import create_point_cloud_from_depth_image
                cloud_full = create_point_cloud_from_depth_image(depth_ori, camera_info, organized=True)
                color_full = img_ori.astype(np.float32) / 255.0
                visualizer.save_scene_pointcloud(cloud_full, color_full, goal_mask.squeeze(0), depth_ori, camera_info)
            except Exception as e:
                print(f"[Warning] Point cloud visualization failed: {e}")

        endpoint, pcd = get_and_process_data(cropping_box, img_ori, depth_ori, camera_info, viz=args.viz)
        grasp_net = grasp_model(args=args, device="cuda", image=img_ori, mask=goal_mask, camera_info=camera_info)
        gg, _ = grasp_net.forward(endpoint, pcd, path)

        if len(gg) == 0:
            data = {}
        else:
            R, t, w = get_correct_pose(gg[0], path, args.viz)
            data = {
                'translation': t.tolist(),
                'rotation': R.tolist(),
                'width': w
            }
            with open(os.path.join(path, "grasp_pose.json"), 'w') as f:
                json.dump(data, f, indent=4)

        # [Debug Vis] 保存运行摘要
        if visualizer:
            visualizer.save_summary({
                'task_input': input_text,
                'qwen_output': output,
                'parsed_result': result,
                'goal_class': goal,
                'goal_id': goal_id,
                'goal_coordinates': goal_coor,
                'num_langsam_masks': len(masks),
                'selected_mask_index': mask_index,
                'cropping_box': cropping_box,
                'num_grasps_found': len(gg),
                'grasp_pose': data
            })

        return data

    except Exception as e:
        print(f"⚠️ Error processing {path}: {e}")
        return None


# --------------------------
# Main 批量处理
# --------------------------
if __name__ == "__main__":
    # 相机参数
    camera = CameraInfo(width=1280, height=720, fx=912.481, fy=910.785, cx=644.943, cy=353.497, scale=1000.0)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dirs = [
        os.path.join(base_dir, "data/real_examples/hard/1"),
        os.path.join(base_dir, "data/real_examples/hard/2"),
        os.path.join(base_dir, "data/real_examples/hard/3"),
    ]

    for i, image_path in enumerate(image_dirs, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {i}: {image_path}")
        print(f"{'='*60}")
        result = compute_grasp_pose(image_path, camera)
        if result is None:
            print(f"Skipping image {i} due to error.\n")
            continue
        print(f"Result: {result}")
