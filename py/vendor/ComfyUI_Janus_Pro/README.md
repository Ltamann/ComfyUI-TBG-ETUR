# ComfyUI-Janus-Pro

[English](README_EN.md) | 简体中文

ComfyUI 的 Janus-Pro 节点，一个统一的多模态理解和生成框架。

![alt text](<workflow/ComfyUI Janus-Pro-workflow.png>)


## 安装方法

### 方法一：通过 ComfyUI Manager 安装（推荐）
1. 安装 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 在管理器中搜索 "Janus-Pro"
3. 点击安装

### 方法二：手动安装
1. 将此仓库克隆到你的 ComfyUI 的 custom_nodes 文件夹中：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro
```

2. 安装所需依赖：

Windows系统：
```bash
# 如果你使用ComfyUI便携版
cd ComfyUI-Janus-Pro
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

# 如果你使用自己的Python环境
cd ComfyUI-Janus-Pro
path\to\your\python.exe -m pip install -r requirements.txt
```

Linux/Mac系统：
```bash
# 使用ComfyUI的Python环境
cd ComfyUI-Janus-Pro
../../python_embeded/bin/python -m pip install -r requirements.txt

# 或者使用你的环境
cd ComfyUI-Janus-Pro
python -m pip install -r requirements.txt
```

注意：如果你遇到安装问题：
- 确保已安装 git
- 尝试更新 pip：`python -m pip install --upgrade pip`
- 如果你使用代理，确保 git 可以访问 GitHub
- 确保使用的是与 ComfyUI 相同的 Python 环境


## 模型下载

将模型文件放在 `ComfyUI/models/Janus-Pro` 文件夹中：
1. 在你的 ComfyUI 的 models 目录下创建 `Janus-Pro` 文件夹
2. 从 Hugging Face 下载模型：
   - [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
   - [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
3. 将模型解压到各自的文件夹中：
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
- 小红书: [@CY-CHENYUE](https://www.xiaohongshu.com/user/profile/6360e61f000000001f01bda0)

---
<div align="center">
    如果这个项目对你有帮助，请给它一个 Star ⭐️
</div>