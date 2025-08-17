# ComfyUI + TBG ETUR on RunPod

This guide explains how to set up **ComfyUI** with custom nodes (TBG ETUR, Manager, TeaCache) on **RunPod**.

---

## Prerequisites

- A **RunPod account**
- Minimum **60 GB disk space** for the pod
- A **Hugging Face API token** ([create here](https://huggingface.co/docs/hub/security-tokens))

---

## Step 1: Create a Pod

1. Open [RunPod](https://www.runpod.io/).
2. Create a new pod with at least **60 GB disk space**.
3. Open the pod terminal.

---

## Step 2: Export Hugging Face Token

In the terminal, run:

```bash
export HF_TOKEN=your_huggingface_api_token_here

## Step 3: Run the Setup Script

Copy and paste the following command in your pod terminal.

Note: This is a single-line setup script that installs dependencies, clones repositories, downloads models, and creates a start script.

apt update && apt install -y git python3-venv wget && \
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
cd /workspace/ComfyUI && python3 -m venv venv && source venv/bin/activate && \
pip install --upgrade pip && pip install -r requirements.txt && \
git clone https://github.com/Ltamann/ComfyUI-TBG-ETUR.git ./custom_nodes/ComfyUI-TBG-ETUR && \
git clone https://github.com/Comfy-Org/ComfyUI-Manager.git ./custom_nodes/ComfyUI-Manager && \
git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git ./custom_nodes/ComfyUI-TeaCache && \
pip install --no-cache-dir -r ./custom_nodes/ComfyUI-TBG-ETUR/requirements.txt && \
pip install --no-cache-dir -r ./custom_nodes/ComfyUI-Manager/requirements.txt && \
pip install --no-cache-dir -r ./custom_nodes/ComfyUI-TeaCache/requirements.txt && \
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 && \
pip install --no-cache-dir xformers==0.0.31 && \
mkdir -p models/style_models models/unet models/vae models/clip_vision models/controlnet models/clip models/loras models/checkpoints models/upscale_models && \
wget --header="Authorization: Bearer $HF_TOKEN" -O models/style_models/flux1-redux-dev.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors?download=true" && \
wget -O models/unet/flux1-dev-fp8.safetensors "https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors?download=true" && \
wget --header="Authorization: Bearer $HF_TOKEN" -O models/vae/ae.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true" && \
wget -O models/vae/fluxae.safetensors "https://huggingface.co/StableDiffusionVN/Flux/resolve/main/Vae/flux_vae.safetensors?download=true" && \
wget -O models/clip_vision/sigclip_vision_patch14_384.safetensors "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors?download=true" && \
wget -O models/controlnet/FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/resolve/main/diffusion_pytorch_model.safetensors?download=true" && \
wget -O models/controlnet/Flux.1-dev-Controlnet-Upscaler.safetensors "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors?download=true" && \
wget -O models/clip/clip-vit-large-patch14.safetensors "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors?download=true" && \
wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" && \
wget -O models/loras/FLUX.1-Turbo-Alpha.safetensors "https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors?download=true" && \
wget -O models/checkpoints/JuggernautXLRagnarok.safetensors "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16" && \
wget -O models/loras/sdxl_lightning_8step_lora.safetensors "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true" && \
wget -O models/controlnet/xinsir_sdxl_tile_controlnet.safetensors "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true" && \
git clone --depth 1 https://huggingface.co/uwg/upscaler /tmp/upscaler && mv /tmp/upscaler/ESRGAN/* models/upscale_models/ && rm -rf /tmp/upscaler && \
echo '#!/bin/bash\ncd /workspace/ComfyUI || exit 1\nsource /workspace/ComfyUI/venv/bin/activate\npython /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 7777' > start.sh && chmod +x start.sh

## Step 4: Configure Pod Settings

Container Start Command
Add the start script:

/workspace/ComfyUI/start.sh


Expose HTTP Ports

8888, 7777

## Step 5: Run the Pod

Click Run to start your ComfyUI pod.
Once running, access the interface at:
Click Connect and open the HTTP Link
