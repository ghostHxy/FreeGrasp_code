#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify Molmo model works correctly.
Run with: python3 test_molmo.py

This script only loads Molmo (not LangSAM or other models) to avoid OOM.
"""

import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def patch_molmo_model(model):
    """
    Patch the Molmo model to fix the KV cache bug where past_key_values[0][0] is None.
    """
    original_forward = model.model.forward

    def patched_forward(*args, **kwargs):
        # Fix: Handle None values in past_key_values
        past_key_values = kwargs.get('past_key_values', None)
        if past_key_values is not None:
            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                if past_key_values[0] is None or (isinstance(past_key_values[0], (list, tuple)) and past_key_values[0][0] is None):
                    kwargs['past_key_values'] = None

        return original_forward(*args, **kwargs)

    model.model.forward = patched_forward
    return model


def main():
    print("=" * 60)
    print("Molmo Test Script")
    print("=" * 60)

    print("\n[1/4] Loading Molmo processor...")
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    print("[2/4] Loading Molmo model...")
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    print("[3/4] Applying KV cache patch...")
    model = patch_molmo_model(model)

    # Find a test image
    image_paths = [
        "data/demo/image.png",
        "data/real_examples/hard/1/image.png",
        "data/real_examples/hard/2/image.png",
    ]

    image_path = None
    for p in image_paths:
        if os.path.exists(p):
            image_path = p
            break

    if image_path is None:
        print("ERROR: No test image found!")
        return

    print(f"[4/4] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"      Image size: {image.size}")

    # Process input
    prompt = "Point out the objects in the red rectangle on the table."
    print(f"\nPrompt: {prompt}")

    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Generate
    print("\nGenerating output (this may take a minute)...")
    try:
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.2,
                    stop_strings=["<|endoftext|>"],
                    use_cache=True
                ),
                tokenizer=processor.tokenizer,
                prefill_chunk_size=256
            )

        # Decode output
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("\n" + "=" * 60)
        print("SUCCESS! Molmo generated output:")
        print("=" * 60)
        print(generated_text)
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
