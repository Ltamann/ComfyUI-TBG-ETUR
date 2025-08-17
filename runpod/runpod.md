
Open runPod

create a pod with min 60 GB of disk space

open Terminal

paste:

export HF_TOKEN= your huggingface api token from https://huggingface.co/docs/hub/security-tokens

 apt update && apt install -y git python3-venv && git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && cd /workspace/ComfyUI && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && git clone https://github.com/Ltamann/ComfyUI-TBG-ETUR.git ./custom_nodes/ComfyUI-TBG-ETUR && git clone https://github.com/Comfy-Org/ComfyUI-Manager.git ./custom_nodes/ComfyUI-Manager && git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git ./custom_nodes/ComfyUI-TeaCache && python3 -m pip install --no-cache-dir -r ./custom_nodes/ComfyUI-TBG-ETUR/requirements.txt && python3 -m pip install --no-cache-dir -r ./custom_nodes/ComfyUI-Manager/requirements.txt && python3 -m pip install --no-cache-dir -r ./custom_nodes/ComfyUI-TeaCache/requirements.txt && wget --header="Authorization: Bearer $HF_TOKEN" -O /workspace/ComfyUI/models/style_models/flux1-redux-dev.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors?download=true" && wget -O /workspace/ComfyUI/models/unet/flux1-dev-fp8.safetensors "https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors?download=true" && wget --header="Authorization:  Bearer $HF_TOKEN" -O /workspace/ComfyUI/models/vae/ae.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true" && wget -O /workspace/ComfyUI/models/vae/fluxae.safetensors "https://huggingface.co/StableDiffusionVN/Flux/resolve/main/Vae/flux_vae.safetensors?download=true" && wget -O /workspace/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors?download=true" && wget -O /workspace/ComfyUI/models/controlnet/FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/resolve/main/diffusion_pytorch_model.safetensors?download=true" && wget -O /workspace/ComfyUI/models/controlnet/Flux.1-dev-Controlnet-Upscaler.safetensors "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors?download=true" && wget -O /workspace/ComfyUI/models/clip/clip-vit-large-patch14.safetensors "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors?download=true" && wget -O /workspace/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" && wget -O /workspace/ComfyUI/models/loras/FLUX.1-Turbo-Alpha.safetensors "https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors?download=true" && wget -O /workspace/ComfyUI/models/checkpoints/JuggernautXLRagnarok.safetensors "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16" && wget -O /workspace/ComfyUI/models/loras/sdxl_lightning_8step_lora.safetensors "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true" && wget -O /workspace/ComfyUI/models/controlnet/xinsir_sdxl_tile_controlnet.safetensors "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true" && git clone --depth 1 https://huggingface.co/uwg/upscaler /tmp/upscaler && mv /tmp/upscaler/ESRGAN/* /workspace/ComfyUI/models/upscale_models/ && rm -rf /tmp/upscaler && echo '#!/bin/bash\ncd /workspace/ComfyUI || exit 1\nsource /workspace/ComfyUI/venv/bin/activate\npython /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 7777' > /workspace/start.sh && chmod +x /workspace/start.sh


 create start file:
 
cd ComfuUI

 echo '#!/bin/bash
cd /workspace/ComfyUI || exit 1
source /workspace/ComfyUI/venv/bin/activate
python /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 7777' > start.sh && chmod +x start.sh


 edit pod settings
 Container Start Command
 add
 /workspace/ComfyUI/start.sh

 Expose HTTP Ports (Max 10)
 8888,7777

 Run the Pod
 
