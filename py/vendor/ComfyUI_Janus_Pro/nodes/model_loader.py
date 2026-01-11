import os
from huggingface_hub import snapshot_download  # auto-download Janus weights

class JanusModelLoader:
    def __init__(self):
        pass

    def load_model(self, model_name):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install git+https://github.com/deepseek-ai/Janus.git'")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer bfloat16 on GPUs that support it, otherwise fallback to float16
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # ComfyUI root
        comfy_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(__file__)
                            )
                        )
                    )
                )
            )
        )

        # Local models root: <ComfyRoot>/models/Janus-Pro
        models_root = os.path.join(comfy_path, "models", "Janus-Pro")
        os.makedirs(models_root, exist_ok=True)

        # We expect model_name like "deepseek-ai/Janus-Pro-1B"
        local_dir_name = os.path.basename(model_name)  # -> "Janus-Pro-1B"
        model_dir = os.path.join(models_root, local_dir_name)

        # Auto-download from Hugging Face if missing
        if not os.path.exists(model_dir):
            try:
                print(
                    f"[Janus] Local model not found at {model_dir}. "
                    f"Downloading from Hugging Face repo '{model_name}'..."
                )
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", ".git*"],
                )
                print(f"[Janus] Download completed. Using local path: {model_dir}")
            except Exception as exc:
                raise ValueError(
                    f"Failed to download Janus model '{model_name}' into '{model_dir}'. "
                    f"Error: {exc}"
                )

        # Load processor and model from local directory
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir, use_fast=True)

        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        vl_gpt = vl_gpt.to(dtype).to(device).eval()

        return (vl_gpt, vl_chat_processor)
