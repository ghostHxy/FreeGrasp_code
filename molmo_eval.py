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


# **Upload Molmo Model**
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


# Extract (x, y) coordinates from Molmo output
def extract_points(molmo_output, image_w, image_h):
    points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        x, y = float(match.group(1)), float(match.group(2))
        if x <= 100 and y <= 100:
            pixel_x = int((x / 100) * image_w)
            pixel_y = int((y / 100) * image_h)
            points.append((pixel_x, pixel_y))
    return points


# Run Molmo inference on an image
def run_molmo_inference(image, prompt):
    image_w, image_h = image.size

    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=500,
                do_sample=True,
                temperature=0.2,
                stop_strings=["<|endoftext|>"]
            ),
            tokenizer=processor.tokenizer
        )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated Output: {generated_text}")

    points = extract_points(generated_text, image_w, image_h)
    points_with_ids = [(i + 1, x, y) for i, (x, y) in enumerate(points)]
    return points_with_ids


# Map Molmo output ID to Ground Truth ID
# The output of this step will only used for evaluation
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

# Save annotated output
def save_results(scene_id, image, points_with_ids, molmo_to_gt_map):
    output_folder = os.path.join(OUTPUT_DIR, f"scene{scene_id}")
    os.makedirs(output_folder, exist_ok=True)

    # **file path**
    output_image_path = os.path.join(output_folder, f"{scene_id}.png")
    output_text_path = os.path.join(output_folder, f"{scene_id}_id.txt")


    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    # **Generate png annotation image**
    for obj_id, x, y in points_with_ids:
        plt.text(
            x, y, obj_id,
            color="yellow", fontsize=8, fontweight="bold",
            ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )

    plt.axis("off")
    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"Labeled image saved in: {output_image_path}")

    # **Save txt result**
    with open(output_text_path, "w") as f:
        f.write("Molmo_ID X Y GT_ID\n")
        for obj_id, x, y in points_with_ids:
            f.write(f"{obj_id} {x} {y} {molmo_to_gt_map.get(obj_id, -1)}\n")
    print(f"Predict ID、coordinates & GT ID were saved: {output_text_path}")


def run_local_inference(image, prompt):
    """
    Run Molmo for prediction and return a list of (random ID, coordinates) pairs.
    """
    # 添加参数验证
    if image is None:
        raise ValueError("run_local_inference: image parameter is None!")
    if not hasattr(image, 'size'):
        raise ValueError(f"run_local_inference: image has no 'size' attribute. Type: {type(image)}")
    
    image_w, image_h = image.size
    
    # 验证 processor 和 model 是否已加载
    if processor is None:
        raise ValueError("Processor is not initialized!")
    if model is None:
        raise ValueError("Model is not initialized!")
    
    # 设置模型为评估模式
    model.eval()
    
    # Process input
    inputs = processor.process(images=[image], text=prompt)
    # 🔥 关键修复：将输入转移到 GPU 并添加 batch 维度（这行之前缺失了！）
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # 确保没有 past_key_values 相关的键
    if 'past_key_values' in inputs:
        del inputs['past_key_values']
    
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        try:
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.2,
                    stop_strings=["<|endoftext|>"],
                    use_cache=False  # 🔥 禁用缓存避免 past_key_values 问题
                ),
                tokenizer=processor.tokenizer
            )
        except Exception as e:
            # 如果 generate_from_batch 失败，尝试使用标准 generate 方法
            print(f"⚠️ generate_from_batch failed: {e}. Trying standard generate method...")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.2,
                    use_cache=False,
                    pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
                )
    
    # Parse output text
    if isinstance(output, torch.Tensor):
        generated_tokens = output[0, inputs['input_ids'].size(1):]
    else:
        generated_tokens = output[0][inputs['input_ids'].size(1):]
    
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract point coordinates
    points = extract_points(generated_text, image_w, image_h)
    points_with_ids = [(i + 1, x, y) for i, (x, y) in enumerate(points)]
    
    return points_with_ids


def process_image(image_path, prompt, output_folder):
    """
    Process an image and return label image and text content.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 添加调试信息
    print(f"📷 正在加载图像: {image_path}")
    
    # Load image with error handling
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✅ 图像加载成功: {image.size if image else 'None'}")
        
        # 验证图像成功加载
        if image is None:
            raise ValueError(f"Image.open() returned None for {image_path}")
        if not hasattr(image, 'size'):
            raise ValueError(f"Loaded image has no 'size' attribute: {type(image)}")
            
        print(f"✅ 图像验证通过: size={image.size}")
        
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        raise Exception(f"Error loading image from {image_path}: {e}")
    
    # 调用 Molmo 推理生成点坐标
    print(f"🤖 开始 Molmo 推理...")
    try:
        points_with_ids = run_local_inference(image, prompt)
        print(f"✅ Molmo 推理完成，找到 {len(points_with_ids)} 个点")
    except Exception as e:
        print(f"❌ Molmo 推理失败: {e}")
        print(f"   image 类型: {type(image)}, image 是否为 None: {image is None}")
        raise
    
    # Generate labeled image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    output_image_path = os.path.join(output_folder, "molmo_label.png")
    output_text_path = os.path.join(output_folder, "molmo_id.txt")

    for obj_id, x, y in points_with_ids:
        plt.text(
            x, y, obj_id,
            color="yellow", fontsize=8, fontweight="bold",
            ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )
    
    plt.title("Molmo ID Mapping")
    plt.axis("off")
    plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Generate TXT result file
    text_content = "Molmo_ID X Y\n"
    text_content += "\n".join(f"{obj_id} {x} {y}" for obj_id, x, y in points_with_ids)
    
    with open(output_text_path, "w") as f:
        f.write(text_content)
    
    return output_image_path, text_content


def process_and_send_to_gpt(image_path, prompt, output_folder):
    """
    Process image with Molmo and send labeled image to GPT for further reasoning.
    """
    labeled_image_path, text_content = process_image(image_path, prompt, output_folder)
    
    # Convert labeled image to base64
    with open(labeled_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    return base64_image, text_content


# Main batch processing
def batch_process_molmo():
    print("Starting Molmo batch processing...")
    # scene_folders = [f for f in os.listdir(SCENES_DIR) if os.path.isdir(os.path.join(SCENES_DIR, f)) and f.startswith("scene")]
    scene_ids = df["sceneId"].unique()

    for scene_id in scene_ids:
        output_folder = os.path.join(OUTPUT_DIR,f"scene{scene_id}")
        if os.path.exists(output_folder):
            print(f"Skipping scene {scene_id} (already processed)")
            continue
        
        scene_data = df[df["sceneId"] == scene_id].iloc[0]
        
        # Extract image from Parquet
        image_bytes = scene_data["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Locate corresponding npz file
        semantic_file = os.path.join(NPZ_DIR, f"{scene_id}.npz")
        print(semantic_file)
        if not os.path.exists(semantic_file):
            print(f"⚠️ Missing NPZ for scene {scene_id}")
            continue

        print(f"🚀 Processing scene {scene_id}...")

        # Run Molmo
        prompt = "Point out all objects in the green tray"
        points_with_ids = run_molmo_inference(image, prompt)

        # Map to Ground Truth
        molmo_to_gt_map = map_molmo_id_to_gt(points_with_ids, semantic_file)

        # Save outputs
        save_results(scene_id, image, points_with_ids, molmo_to_gt_map)


    print("🎉 Batch processing complete!")


if __name__ == "__main__":
    # Path of dataset
    PARQUET_FILES = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet"
    ]
    NPZ_DIR = "data/npz_file"
    
    OUTPUT_DIR = "data/output/molmo_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the dataset
    df = pd.concat([pd.read_parquet(p) for p in PARQUET_FILES])
    batch_process_molmo()