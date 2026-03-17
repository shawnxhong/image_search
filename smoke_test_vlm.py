"""Smoke test for Qwen2.5-VL-3B-Instruct with OpenVINO GenAI VLMPipeline.

Tests:
1. Basic captioning (unconditional)
2. Conditional captioning with person names
3. Multiple images
"""

import time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

MODEL_PATH = "C:/Users/53422/Documents/image_search/Qwen2.5-VL-3B-Instruct/INT4"
DEVICE = "GPU"
DATASET = Path("C:/Users/53422/Documents/image_search/dataset")


MAX_IMAGE_PIXELS = 1024 * 1024  # ~1 megapixel limit to avoid GPU OOM


def load_image_as_tensor(image_path: str) -> ov.Tensor:
    """Load an image file and convert to OpenVINO Tensor (NHWC uint8).

    Large images are resized to stay within MAX_IMAGE_PIXELS to avoid
    exceeding GPU memory allocation limits.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    pixels = w * h
    if pixels > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized {w}x{h} -> {new_w}x{new_h}")
    arr = np.array(img)  # HWC uint8
    # VLMPipeline expects NHWC: add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return ov.Tensor(arr)


def main():
    print(f"Loading VLMPipeline from {MODEL_PATH} on {DEVICE}...")
    t0 = time.time()
    pipe = ov_genai.VLMPipeline(MODEL_PATH, DEVICE)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.do_sample = False

    # --- Test 1: Basic captioning ---
    print("\n=== Test 1: Basic captioning ===")
    img_path = str(DATASET / "Colin_Powell_0043.jpg")
    print(f"Image: {img_path}")
    image_tensor = load_image_as_tensor(img_path)

    prompt = "Describe this image in one sentence."
    print(f"Prompt: {prompt}")

    t0 = time.time()
    result = pipe.generate(prompt, images=[image_tensor], generation_config=config)
    elapsed = time.time() - t0
    text = str(result)
    print(f"Result ({elapsed:.1f}s): {text}")

    # --- Test 2: Conditional captioning with person name ---
    print("\n=== Test 2: Conditional captioning with name ===")
    prompt2 = "This photo contains Colin Powell. Describe what Colin Powell is doing in this image in one sentence."
    print(f"Prompt: {prompt2}")

    t0 = time.time()
    result2 = pipe.generate(prompt2, images=[image_tensor], generation_config=config)
    elapsed = time.time() - t0
    text2 = str(result2)
    print(f"Result ({elapsed:.1f}s): {text2}")
    assert "powell" in text2.lower() or "colin" in text2.lower(), \
        f"Expected 'Powell' or 'Colin' in caption, got: {text2}"

    # --- Test 3: Another image (kitchen/gas stove) ---
    print("\n=== Test 3: Different image ===")
    img_files = list(DATASET.glob("IMG_*.jpeg"))
    if img_files:
        img_path3 = str(img_files[0])
        print(f"Image: {img_path3}")
        image_tensor3 = load_image_as_tensor(img_path3)

        prompt3 = "Describe this image in one sentence."
        t0 = time.time()
        result3 = pipe.generate(prompt3, images=[image_tensor3], generation_config=config)
        elapsed = time.time() - t0
        text3 = str(result3)
        print(f"Result ({elapsed:.1f}s): {text3}")
    else:
        print("No IMG_*.jpeg files found, skipping")

    # --- Test 4: Caption generation prompt (simulating what the app will use) ---
    print("\n=== Test 4: App-style caption prompt ===")
    app_prompt = (
        "Describe what is happening in this photo in one sentence. "
        "The people in the photo are: Colin Powell. "
        "Use their names instead of generic terms like 'a man' or 'a person'."
    )
    print(f"Prompt: {app_prompt}")
    image_tensor4 = load_image_as_tensor(str(DATASET / "Colin_Powell_0043.jpg"))

    t0 = time.time()
    result4 = pipe.generate(app_prompt, images=[image_tensor4], generation_config=config)
    elapsed = time.time() - t0
    text4 = str(result4)
    print(f"Result ({elapsed:.1f}s): {text4}")

    # --- Test 5: Two-person captioning ---
    print("\n=== Test 5: Two-person conditional caption ===")
    img_path5 = str(DATASET / "IMG_0085.jpeg")
    image_tensor5 = load_image_as_tensor(img_path5)
    prompt5 = (
        "Describe what is happening in this photo in one sentence. "
        "The people in the photo are: Alice, Bob. "
        "Use their names instead of generic terms like 'a man' or 'a person'."
    )
    print(f"Prompt: {prompt5}")
    t0 = time.time()
    result5 = pipe.generate(prompt5, images=[image_tensor5], generation_config=config)
    elapsed = time.time() - t0
    text5 = str(result5)
    print(f"Result ({elapsed:.1f}s): {text5}")

    # --- Test 6: Unconditional caption (no names, for comparison) ---
    print("\n=== Test 6: Unconditional caption ===")
    prompt6 = "Describe this image in one sentence."
    t0 = time.time()
    result6 = pipe.generate(prompt6, images=[image_tensor5], generation_config=config)
    elapsed = time.time() - t0
    text6 = str(result6)
    print(f"Result ({elapsed:.1f}s): {text6}")

    print("\n=== All smoke tests passed! ===")


if __name__ == "__main__":
    main()
