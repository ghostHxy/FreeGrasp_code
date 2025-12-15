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
    image_w, image_h = image.size

    # Process input
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

    # Parse output text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
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
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    points_with_ids = run_local_inference(image, prompt)
    
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


def process_and_send_to_gpt(image_path, prompt, save_path):
    """
    离线加载 Molmo 模型，处理图片生成标注和文本。
    返回：
        base64_labeled_image: PIL -> base64 编码的图像
        labeled_text: 模型输出的标注文本
    """
    # ------------------------------
    # 本地 Molmo 模型路径
    # ------------------------------
    molmo_path = os.path.expanduser("~/.cache/huggingface/hub/models--allenai--Molmo-7B-D-0924")
    print(f"[DEBUG] Loading Molmo from local path: {molmo_path}")

    # ------------------------------
    # 加载模型（优先本地，失败则从 HuggingFace 下载）
    # ------------------------------
    try:
        # 先尝试从本地加载
        print("[INFO] Attempting to load Molmo from local cache...")
        processor = AutoProcessor.from_pretrained(
            molmo_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            molmo_path, 
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        print("[INFO] Successfully loaded Molmo from local cache")
    except Exception as e:
        print(f"[WARNING] Failed to load from local cache: {e}")
        print("[INFO] Downloading Molmo from HuggingFace...")
        # 如果本地加载失败，从 HuggingFace 下载
        processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        print("[INFO] Successfully downloaded and loaded Molmo from HuggingFace")

    # ------------------------------
    # 图片加载
    # ------------------------------
    image = Image.open(image_path).convert("RGB")
    print(f"[DEBUG] Loaded image for Molmo: {image_path}, size: {image.size}")

    # ------------------------------
    # 模型推理（使用与 run_local_inference 相同的调用方式）
    # ------------------------------
    # 使用 processor.process 方法处理输入
    inputs = processor.process(images=[image], text=prompt)
    
    # 添加调试信息和错误检查
    print(f"[DEBUG] processor.process returned keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'Not a dict'}")
    print(f"[DEBUG] inputs type: {type(inputs)}")
    
    # 检查 inputs 是否是字典且包含必要的键
    if not isinstance(inputs, dict):
        raise ValueError(f"[ERROR] processor.process returned non-dict: {type(inputs)}")
    
    if 'input_ids' not in inputs:
        raise ValueError(f"[ERROR] 'input_ids' not in inputs. Available keys: {list(inputs.keys())}")
    
    # 检查 input_ids 是否为 None
    if inputs['input_ids'] is None:
        raise ValueError("[ERROR] inputs['input_ids'] is None")
    
    # 将输入移动到设备并添加 batch 维度
    inputs = {k: v.to(model.device).unsqueeze(0) if v is not None else None for k, v in inputs.items()}
    
    # 再次验证 input_ids 不为 None（处理后的检查）
    if inputs.get('input_ids') is None:
        raise ValueError("[ERROR] inputs['input_ids'] is None after processing")
    
    print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
    
    try:
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            print(f"[DEBUG] Calling model.generate_from_batch...")
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                temperature=0.2,
                use_cache=False,
            
        )
        print(f"[DEBUG] model.generate completed")
    except Exception as e:
        print(f"[ERROR] Exception in model.generate: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    # 检查 output 是否为 None
    if output is None:
        raise ValueError("[ERROR] model.generate_from_batch returned None")
    
    print(f"[DEBUG] output type: {type(output)}, has shape attr: {hasattr(output, 'shape')}")
    if hasattr(output, 'shape'):
        print(f"[DEBUG] output shape: {output.shape}")
    else:
        print(f"[DEBUG] output does not have shape attribute")
    
    # 解析输出文本 - 使用更安全的方式获取 input_ids 的长度
    input_ids_tensor = inputs.get('input_ids')
    if input_ids_tensor is None:
        raise ValueError("[ERROR] inputs['input_ids'] is None when trying to extract tokens")
    
    if len(input_ids_tensor.shape) >= 2:
        input_ids_len = input_ids_tensor.shape[1]
    elif len(input_ids_tensor.shape) == 1:
        input_ids_len = input_ids_tensor.shape[0]
    else:
        raise ValueError(f"[ERROR] Unexpected input_ids shape: {input_ids_tensor.shape}")
    
    print(f"[DEBUG] input_ids_len: {input_ids_len}, output shape: {output.shape if hasattr(output, 'shape') else 'no shape'}")
    
    try:
        generated_tokens = output[0, input_ids_len:]
        print(f"[DEBUG] generated_tokens shape: {generated_tokens.shape if hasattr(generated_tokens, 'shape') else 'no shape'}")
    except Exception as e:
        print(f"[ERROR] Failed to slice output: {type(e).__name__}: {e}")
        print(f"[ERROR] output type: {type(output)}, output value: {output}")
        raise
    labeled_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if labeled_text is None or len(labeled_text.strip()) == 0:
        raise ValueError("[ERROR] Molmo returned empty text!")
    
    print(f"[DEBUG] Molmo generated text: {labeled_text}")

    # ------------------------------
    # 将标注图像保存为 base64
    # ------------------------------
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_labeled_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    if base64_labeled_image is None:
        raise ValueError("[ERROR] Failed to convert image to base64!")

    print(f"[DEBUG] Molmo processing done. labeled_text length: {len(labeled_text)}")
    return base64_labeled_image, labeled_text


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
