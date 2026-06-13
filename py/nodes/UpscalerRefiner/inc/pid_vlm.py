import copy
import math
import re
import time

import comfy.utils

from ....vendor.ComfyUI_QwenVL.AILab_QwenVL import QwenVLBase
from ....vendor.ComfyUI_Unload_Models_main.py.unload_one_model import UnloadOneModelNode


PID_VLM_MODEL = "Qwen3-VL-2B-Instruct"
PID_VLM_DEFAULT_PROMPT = (
    "Describe this image tile as a concise image generation prompt. "
    "Focus on visible subjects, materials, lighting, colors, and important local details. "
    "Do not mention that this is a tile, crop, image, assistant response, or analysis."
)


def clean_pid_vlm_prompt(text):
    if not text:
        return ""
    remove_words = (
        "assistant",
        "helpful",
        "vision",
        "Thedescription",
        "TheUser",
        "The description",
        "The User",
        "It can assist",
        "natural language",
        "It can understand",
    )
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    kept = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(word.lower() in lowered for word in remove_words):
            continue
        kept.append(sentence.strip())
    return " ".join(part for part in kept if part).strip()


def _vlm_image(tile):
    image = copy.copy(tile)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    samples = image.movedim(-1, 1)
    target_pixels = 1024 * 1024
    scale_by = math.sqrt(target_pixels / max(1, samples.shape[3] * samples.shape[2]))
    width = max(1, round(samples.shape[3] * scale_by))
    height = max(1, round(samples.shape[2] * scale_by))
    scaled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return scaled.movedim(1, -1)[:, :, :, :3]


def make_pid_vlm_prompt_fn(seed=0, prompt_text="", model_name=PID_VLM_MODEL, quantization="None (FP16)"):
    qwen_vl = QwenVLBase()
    custom_prompt = (prompt_text or "").strip() or PID_VLM_DEFAULT_PROMPT

    def prompt_tile(tile, index=0, total=None, node_id=None, iteration="PID VLM"):
        start = time.perf_counter()
        prefix = "[TBG PID VLM]"
        if node_id is not None:
            prefix = f"TBG[Node {node_id}] PID VLM"
        if total is None:
            print(f"{prefix} tile {int(index) + 1} - generating Qwen prompt...")
        else:
            print(f"{prefix} tile {int(index) + 1}/{int(total)} - generating Qwen prompt...")
        prompt = qwen_vl.run(
            model_name,
            quantization,
            preset_prompt="Detailed Analysis",
            custom_prompt=custom_prompt,
            image=_vlm_image(tile),
            video=None,
            frame_count=16,
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.2,
            seed=int(seed or 0) + int(index or 0),
            keep_model_loaded=True,
            attention_mode="auto",
            use_torch_compile=True,
            device="auto",
        )[0]
        prompt = clean_pid_vlm_prompt(prompt)
        elapsed = time.perf_counter() - start
        if total is None:
            print(f"{prefix} tile {int(index) + 1} - [tile prompt] {prompt} ({elapsed:.2f}s)")
        else:
            print(f"{prefix} tile {int(index) + 1}/{int(total)} - [tile prompt] {prompt} ({elapsed:.2f}s)")
        return prompt

    def cleanup():
        try:
            UnloadOneModelNode.route(qwen_vl)
        except Exception as exc:
            print(f"[TBG PID VLM] Qwen unload skipped: {exc}")

    return prompt_tile, cleanup
