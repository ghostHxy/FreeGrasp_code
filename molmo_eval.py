import os
import io
import re
import base64
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# ==========================
# Load Molmo Model
# ==========================
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
model.eval()  # 设置评估模式

# ==========================
# Extract (x, y) coordinates
# ==========================
def extract_points(molmo_output, image_w, image_h):
    points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        x, y = float(match.group(1)), float(match.group(2))
        if x <= 100 and y <= 100:
            pixel_x = int((x / 100) * image_w)
            pixel_y = int((y / 100) * image_h)
            points.append((pixel_x, pixel_y))
    return points

# ==========================
# Run Molmo inference on an image
# ==========================
def run_local_inference(image, prompt, max_tokens=500, temperature=0.2):
    """
    Run Molmo for prediction on a single image.
    Returns: list of tuples (Molmo_ID, x_pixel, y_pixel)
    Fully compatible with Transformers 4.57+ and Molmo 7B.
    """
    # 验证输入
    if image is None or not hasattr(image, 'size'):
        raise ValueError(f"Invalid image input: {image}")

    image_w, image_h = image.size

    if processor is None or model is None:
        raise ValueError("Processor or model not initialized!")

    # 模型评估模式
    model.eval()

    # 处理输入
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # 删除可能存在的 past_key_values
    inputs.pop('past_key_values', None)

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        try:
            # 尝试 generate_from_batch
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    stop_strings=["<|endoftext|>"],
                    use_cache=False  # 禁用缓存
                ),
                tokenizer=processor.tokenizer
            )
        except Exception as e:
            print(f"⚠️ generate_from_batch failed: {e}. Using standard generate...")
            # 使用标准 generate 回退
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=False,
                    pad_token_id=getattr(processor.tokenizer, "pad_token_id", processor.tokenizer.eos_token_id)
                )

    # 解析输出文本
    if isinstance(output, torch.Tensor):
        generated_tokens = output[0, inputs['input_ids'].size(1):]
    else:
        # generate 返回 list of tensors
        generated_tokens = output[0][inputs['input_ids'].size(1):]

    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated Output: {generated_text}")

    # 提取坐标
    points = extract_points(generated_text, image_w, image_h)
    points_with_ids = [(i + 1, x, y) for i, (x, y) in enumerate(points)]

    return points_with_ids


# ==========================
# Map Molmo ID to GT
# ==========================
def map_molmo_id_to_gt(points_with_ids, semantic_file):
    semantic_data = np.load(semantic_file)
    instance_objects = semantic_data["instances_objects"].astype(int)
    molmo_to_gt_map = {}
    for molmo_id, x, y in points_with_ids:
        if 0 <= y < instance_objects.shape[0] and 0 <= x < instance_objects.shape[1]:
            gt_id = instance_objects[y, x]
            molmo_to_gt_map[molmo_id] = gt_id if gt_id > 0 else -1
        else:
            molmo_to_gt_map[molmo_id] = -1
    return molmo_to_gt_map

# ==========================
# Save annotated image & TXT
# ==========================
def save_results(scene_id, image, points_with_ids, molmo_to_gt_map, output_dir):
    output_folder = os.path.join(output_dir, f"scene{scene_id}")
    os.makedirs(output_folder, exist_ok=True)

    # Save labeled image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    for obj_id, x, y in points_with_ids:
        plt.text(x, y, obj_id, color="yellow", fontsize=8, fontweight="bold",
                 ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
    plt.axis("off")
    output_image_path = os.path.join(output_folder, f"{scene_id}.png")
    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # Save TXT
    output_text_path = os.path.join(output_folder, f"{scene_id}_id.txt")
    with open(output_text_path, "w") as f:
        f.write("Molmo_ID X Y GT_ID\n")
        for obj_id, x, y in points_with_ids:
            f.write(f"{obj_id} {x} {y} {molmo_to_gt_map.get(obj_id, -1)}\n")

    print(f"Saved labeled image: {output_image_path}")
    print(f"Saved TXT file: {output_text_path}")

# ==========================
# Batch processing
# ==========================
def batch_process_molmo(df, npz_dir, output_dir):
    scene_ids = df["sceneId"].unique()
    for scene_id in scene_ids:
        output_folder = os.path.join(output_dir, f"scene{scene_id}")
        if os.path.exists(output_folder):
            print(f"Skipping scene {scene_id} (already processed)")
            continue

        scene_data = df[df["sceneId"] == scene_id].iloc[0]
        image_bytes = scene_data["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        semantic_file = os.path.join(npz_dir, f"{scene_id}.npz")
        if not os.path.exists(semantic_file):
            print(f"⚠️ Missing NPZ for scene {scene_id}")
            continue

        prompt = "Point out all objects in the green tray"
        points_with_ids = run_local_inference(image, prompt)
        molmo_to_gt_map = map_molmo_id_to_gt(points_with_ids, semantic_file)
        save_results(scene_id, image, points_with_ids, molmo_to_gt_map, output_dir)

    print("🎉 Batch processing complete!")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    PARQUET_FILES = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet"
    ]
    NPZ_DIR = "data/npz_file"
    OUTPUT_DIR = "data/output/molmo_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.concat([pd.read_parquet(p) for p in PARQUET_FILES])
    batch_process_molmo(df, NPZ_DIR, OUTPUT_DIR)
