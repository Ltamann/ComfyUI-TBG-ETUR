import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid
from .response_cleanup import strip_thinking


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class Qwen2VL_TBG:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def inference(

        self,
        text,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        video_path=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)

        if model.startswith("Qwen"):
            model_id = f"qwen/{model}"
        else:
            model_id = f"Skywork/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            if video_path:
                print("deal video_path", video_path)
                # 使用FFmpeg处理视频
                unique_id = uuid.uuid4().hex  # 生成唯一标识符
                processed_video_path = f"/tmp/processed_video_{unique_id}.mp4"  # 临时文件路径
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", "fps=1,scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    processed_video_path
                ]
                subprocess.run(ffmpeg_command, check=True)

                # 添加处理后的视频信息到消息
                messages[0]["content"].insert(0, {
                    "type": "video",
                    "video": processed_video_path,
                })

            # 处理图像输入
            else:
                print("deal image")
                pil_image = tensor_to_pil(image)
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": pil_image,
                })

            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("deal messages", messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # 推理
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
                result = [strip_thinking(text, "[QwenVL]") for text in result]
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # 删除临时视频文件
            if video_path:
                os.remove(processed_video_path)

            return result

