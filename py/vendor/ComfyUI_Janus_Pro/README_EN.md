# ComfyUI-Janus-Pro

English | [简体中文](README.md)

ComfyUI nodes for Janus-Pro, a unified multimodal understanding and generation framework.

![alt text](<workflow/ComfyUI Janus-Pro-workflow.png>)

## Installation

### Method 1: Install via ComfyUI Manager (Recommended)
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "Janus-Pro" in the Manager
3. Click install

### Method 2: Manual Installation
1. Clone this repository to your ComfyUI's custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro
```

2. Install the required dependencies:

For Windows:
```bash
# If you're using ComfyUI portable version
cd ComfyUI-Janus-Pro
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

# If you're using your own Python environment
cd ComfyUI-Janus-Pro
path\to\your\python.exe -m pip install -r requirements.txt
```

For Linux/Mac:
```bash
# Using ComfyUI's Python environment
cd ComfyUI-Janus-Pro
../../python_embeded/bin/python -m pip install -r requirements.txt

# Or using your own environment
cd ComfyUI-Janus-Pro
python -m pip install -r requirements.txt
```

Note: If you encounter any installation issues:
- Make sure you have git installed
- Try updating pip: `python -m pip install --upgrade pip`
- If you're behind a proxy, make sure git can access GitHub
- Make sure you're using the same Python environment as ComfyUI


## Model Download

Place your models in the `ComfyUI/models/Janus-Pro` folder:
1. Create a `Janus-Pro` folder in your ComfyUI's models directory
2. Download the models from Hugging Face:
   - [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
   - [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
3. Extract the models into their respective folders:
   ```
   ComfyUI/models/Janus-Pro/Janus-Pro-1B/
   ComfyUI/models/Janus-Pro/Janus-Pro-7B/
   ```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=CY-CHENYUE/ComfyUI-Janus-Pro&type=Date)](https://star-history.com/#CY-CHENYUE/ComfyUI-Janus-Pro&Date)

## Contact Me

- X (Twitter): [@cychenyue](https://x.com/cychenyue)
- TikTok: [@cychenyue](https://www.tiktok.com/@cychenyue)
- YouTube: [@CY-CHENYUE](https://www.youtube.com/@CY-CHENYUE)
- BiliBili: [@CY-CHENYUE](https://space.bilibili.com/402808950)
- Xiaohongshu: [@CY-CHENYUE](https://www.xiaohongshu.com/user/profile/6360e61f000000001f01bda0)

---
<div align="center">
    If you find this project helpful, please give it a Star ⭐️
</div> 