# ComfyUI-TBG-ETUR: 100MP Enhanced Tiled Upscaler & Refiner FLUX Pro. Enhance Your Images with TBG's Upscaler
**TBG_Enhanced Tiled Upscaler & Refiner FLUX PRO**

## Table of Contents
- [Overview](#overview)
- [Early_Access](#Early_Access)
- [Features](#features)
- [Installation](#installation)
- [API_Access](#API_Access)
- [Usage](#usage)
- [TBG_Magnific_Magnifier_Node](#TBG_Magnific_Magnifier_Node)
- [Update 1.04 alfa](#Update)

## Overview

Welcome to **ComfyUI-TBG-ETUR** repository! **TBG Enhanced Tiled Upscaler and Refiner Pro!**
We at **TBG Think. Build. Generate. AI upscaling & image enrichment** are
excited to make our TBG Enhanced Tiled Upscaler and Refiner Pro available to you.

# Alpha Testing of PRO Features Now Available

We’re excited to announce that alpha testing of the PRO version is now live
for our Patreon supporters and free community members!

Please help us to find all Bugs and open an issue here !!!

You can download the latest version of our software [here]([https://github.com/Ltamann/ComfyUI-TBG-ETUR](https://github.com/Ltamann/ComfyUI-TBG-ETUR). 
!!! Note: This code is updated daily, so stay tuned for bug fixes !!!

## Early_Access
Get early access by joining us at:
https://www.patreon.com/TB_LAAR

The CE (Community Edition) nodes are free and standalone, suitable for any type of workflow. For access to some PRO features, a TB_LAAR Patreon membership is required. The free version is sufficient for testing and experimenting. Once you become a member, you can obtain your API key:
https://api.ylab.es/login.php 

## Features

**TBG Enhanced Tiled Upscaler and Refiner Pro** is an advanced, modular enhancement suite for **tiled image generation and refinement** in ComfyUI. It introduces neuro generative tile fusion, interactive tile-based editing, and multi-path processing pipelines designed for extreme resolution workflows up to but not limitet to 100MP, high fidelity, and adaptive post-generation control.

- **AI Image Enhancemen**t: Applies advanced algorithms to improve visual quality.
- **High-Resolution Generation**: Creates images with greater clarity and fine detail.
- **Image Polishing**: Enhances visuals for a smooth, finished appearance.
- **Neuro-Generative Tile Fusion**: Seamlessly fuses image tiles into a coherent whole.
- **User-Friendly Interface**: CE version built for straightforward and intuitive use, Pro version offers advanced customization.

## 1. New Fusion Techniques

TBG Enhanced Tiled Upscaler and Refiner Pro introduces tree next-generation tile fusion processes that go far beyond traditional blending:

### Smart Merge CE
Choose between adaptive blending strategies compatible with common upscalers like USDU,and others. It intelligently handles tile overlaps, avoiding ghosting and seams.

### PRO Tile Diffusion  
A novel approach where each tile is generated while considering surrounding tile information — perfect for mid-denoise levels. This enables:
- Context-aware detail consistency  
- Smooth transitions across tile borders  
- Better handling of textures and patterns  

### PRO Neuro-Generative Tile Fusion (NGTF)  
An advanced generative system that remembers newly generated surroundings and adapts subsequent sampling steps accordingly. This makes high-denoise tile refinement possible while maintaining:
- Global consistency  
- Sharp, coherent details  
- Memory of contextual relationships across tiles  

## 2. Design Suite

TBG_Enhanced is not only about quality — it's about flexible, editable workflows over time:

### One-Tile Preview & Smart Presets  
Quickly generate single-tile previews to fine-tune the right settings before running the full job. Presets adapt intelligently to image dimensions and desired resolution.

### Post-Sampling Tile Correction  
You can resample only the tiles you don’t like — no need to regenerate the full image. This allows:
- Precision editing  
- Fixing small errors without full reprocessing  

### Resume Tile Refinement  
Modify or refine individual tiles days later while keeping the original input image and final result. The system supports:
- Saved tile maps and settings  
- Re-injection of noise and conditioning  
- Fully restorable editing workflows  

## 3. Pipeline Structure

TBG_Enhanced is powered by a flexible pipeline architecture built around tile-aware processing paths:

### TBG Tile Prompter Pipe  
Access prompt and denoise settings per tile. Enables:
- Per-region storytelling  
- Adaptive text-to-image behavior  
- Denoising strength by tile  

### TBG Tile Enrichment Pipe  
Control multiple sampling and model-level features:
- Model-side: Use DemonTools for deep model manipulation  
- Sampler-side: Inject custom noise at specific steps, or apply sigma curves to selected steps  
- Sampler-internal: Enable per-step sampler-side noise injection  
- Built-in noise reduction  
- Optional tile up/downscaling during sampling  

### ControlNet Pipe  
Tiled generation now supports unlimited ControlNet inputs per tile, unlocking:
- High-resolution conditioning  
- Fine-grained control for large images  
- Targeted structure, depth, edge, pose, or segmentation maps for each region  


## Status and Roadmap

- Fusion Techniques: Smart Merge, Tile Diffusion, NGTF  
- Post-editing Tools: One-tile preview, correction, resume-refine  
- Pipelines: Prompt / Enrichment / ControlNet Pipes  
- Upcoming: Full integration with custom samplers and DreamEdit compatibility  


## Installation
To install the TBG Enhanced Tiled Upscaler and Refiner Pro, follow these steps:

**Install from Comfyui Manager**:

 - Open Manager
 - select Install missing custom nodes
 - select Install via Git URL
 - copy and paste this url:   [https://github.com/Ltamann/ComfyUI-TBG-ETUR](https://github.com/Ltamann/ComfyUI-TBG-ETUR)
 - Install and Restart ComfyUI
 - Download LLM models from https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
   Janus-Pro-1B	and or Janus-Pro-7B
   copy  Janus-Pro-1B to ...ComfyUI_344\ComfyUI_windows_portable\ComfyUI\models\Janus-Pro\Janus-Pro-1B\
   copy  Janus-Pro-7B to ...ComfyUI_344\ComfyUI_windows_portable\ComfyUI\models\Janus-Pro\Janus-Pro-7B\ 
   
**Manual Install**:

 - Download the Software:  [https://github.com/Ltamann/ComfyUI-TBG-ETUR](https://github.com/Ltamann/ComfyUI-TBG-ETUR)
 - Unpack and Copy to folder: ..\ComfyUI\custom_nodes\
 - Install requirements: ..\ComfyUI\custom_nodes\ComfyUI-TBG-ETUR\pip install -r requirements.txt
 - Download LLM models from https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
   Janus-Pro-1B	and or Janus-Pro-7B
   copy  Janus-Pro-1B to ...ComfyUI_344\ComfyUI_windows_portable\ComfyUI\models\Janus-Pro\Janus-Pro-1B\
   copy  Janus-Pro-7B to ...ComfyUI_344\ComfyUI_windows_portable\ComfyUI\models\Janus-Pro\Janus-Pro-7B\ 

## API_Access
If you like to test the PRO feachers get early access by joining us at:
 - joining us at: https://www.patreon.com/TB_LAAR
 - get your API key from here:  https://api.ylab.es/login.php
 - add environment variables using TBG_ETUR_API_KEY = your api key
    ![environment_var.png](img/environment_var.png)
   
You can paste the API key directly into the TBG Tiler node, but keep in mind that doing so will embed it into your images and workflow metadata, making it easy to share unintentionally.
For better security, consider adding the key to your environment variables using TBG_ETUR_API_KEY instead.

   
--------------------------------------------------------------
## Usage
check https://youtu.be/fyQRj5nv1IE

Recommended Workflow:
Do your setup and testing with PRO features turned OFF,
and only enable them for the final steps of your workflow.

Thank you for your support and happy tiling!

## Topics
- ai-image-processing
- ai-upscaler
- comfyui
- comfyui-node
- comfyui-nodes
- high-resolution
- image-refinement
- image-upscaling
- neuro-generative-tile-fusion
- stable-diffusion
- tbg-enhanced-tiled-upscaler-and-refiner-pro
- tiled-upscaling

![TBG Magnific Magnifier Example](img/TBG_magnific_upscalser_still.jpg)  
## TBG_Magnific_Magnifier_Node

The **TBG Magnific Magnifier** node is an advanced yet simple-to-use enhancement tool designed for tiled image generation and refinement within ComfyUI. It uses cutting-edge neuro-generative tile fusion and multi-path processing pipelines tailored for extreme resolution workflows — supporting images up to (but not limited to) 100 megapixels with high fidelity or high freedom.

The TBG Magnific Magnifier **enables multi-step automatic upscaling** with very simple, easy-to-use settings like **Creativity, Reassemblance, Fractality,** and **Inventivity**, giving users powerful control without complexity.

## Key Features

- **AI Image Enhancement**  
  Applies advanced algorithms to significantly improve image visual quality.

- **High-Resolution Generation**  
  Produces images with exceptional clarity and fine detail, even at ultra-high resolutions.

- **Image Polishing**  
  Smooths and refines visuals for a clean, professional finish.

- **Neuro-Generative Tile Fusion**  
  Seamlessly fuses tiled images into a coherent, artifact-free whole.

- **User-Friendly Interface**  
  The CE (Community Edition) offers straightforward and intuitive controls, while the PRO version provides extensive customization options for advanced users.

---

## Parameter Overview
![TBG Magnific Magnifier Example](img/Node.png)  

### 🔹 **Fractality**

> A tile size multiplier controlling detail randomness during upscaling.

- **Lower values**: introduce more randomness and abstract patterns.  
- **Higher values**: retain more of the original image’s structure.  

**Impact:**  
Fractality determines how much fractal-like detail is generated.  
⚠️ **Higher values require exponentially more GPU memory!**

> My test settings require around **28GB VRAM** during the refinement pass.  
If you're not using an RTX 5090, consider lowering Fractality, turning off the refinement pass, or using GGUF/optimization techniques to reduce VRAM usage.

---

### 🔹 **Creativity**

> Controls the balance between faithful detail preservation and creative freedom.

- **Lower values**: yield consistent, refined results that closely follow the input.  
- **Higher values**: allow for reinterpretation, abstract variations, and novel detail generation.

**Impact:**  
Affects how "safe" or "imaginative" the upscale becomes.  
Creativity is automatically **scaled down** during **multi-step upscaling** to maintain control.

---

### 🔹 **Inventivity**

> Boosts micro-detail creation and enhances or invents textures.

- Adds realistic or artistic **small-scale details** that may not exist in the original.  
- Particularly useful in **low-resolution** areas or flat surfaces.

**Impact:**  
Inventivity enhances perceived richness in skin, fabric, nature, and more.  
🧠 **Tip:** Prompt phrasing significantly affects Inventivity. In some cases, you may need to **lower flux guidance** to retain control.

---

### 🔹 **Reassemblance (Anchor)**

> Connects structure consistency to **ControlNet**, **Redux**, and **IPAdapter** features.

- **High values**: strongly preserve original layout and mega-structures.  
- **Low values**: allow for more freedom and restructuring.

**Impact:**  
Guides structural consistency during multi-step upscaling.  
Reassemblance dynamically adapts to Redux/ControlNet inputs.

---

### 🔹 **Refinement Pass**

> An optional final step that enhances fine details after the upscale.

- Produces **sharper, cleaner, more complete** results.  
- ⚠️ **Requires significantly more GPU memory** than standard steps.

**Use with caution** if you're on limited VRAM. Consider disabling it or optimizing other parameters to save memory.

**The TBG ETUR Magnific Magnifier requires a Pro, Premium, or Unlimited membership.** 
However, you can achieve the same results using the TBG Enhanced Upscaler and Refiner PRO with manual settings and a Free membership.



## Update
1.04 alfa
New Feature: FLUX KONTEXT Integration in CNETpipe
The biggest highlight is the introduction of FLUX KONTEXT, offering a fully adaptive context pipeline that chains and stiches with ControlNetPipe Depth and Canny.

Mask Attention Completely Overhauled
Mask Attention has been rewritten from the ground up. It now works across both Pro and Community editions. A new border margin setting lets you fine-tune the mask's outer influence zone, and the segment sampling is now fully neuro-generative fusion capable, providing smoother, more organic blending between segments.

Fusion Fixes & Tile Improvements
We've patched several bugs in the Neuro Generative Fusion and Tile Fusion systems. Some users experienced glitches when tile sizes and border areas overlapped teh fusion area - that’s now resolved. Expect cleaner results and better alignment.

Color Glitch Compensation
Color artifacts caused by latent encode/decode are now handled much better. A new feature compensates for these VAE latent color glitches. It was generating faint lines along the tile joints when using high denoise settings.

LLM Switching: Janus 1B and Janus 7B Now Supported
You can now choose your preferred LLM directly from the node interface. Currently supported: Janus 1B and Janus/B — with more coming soon.

Other Fixes and Tweaks
Many small bugs and glitches reported by the community have been addressed — thank you for your feedback! This includes improvements to node stability, UI tweaks, and general robustness across the board.

Give it a spin and let me know how it works for you. Your feedback always helps shape the next version so don’t hesitate to reach out!

New Tile Cache
We've updated the cache function for Soft Merge and Tile Fusion. You can now toggle the cache on or off at any time and choose between two modes:

Full Cache Mode: Uses cached images for everything — including input tiles — allowing you to refine over previous refinements.

Fusion-Only Cache Mode: Uses the original input tile but applies cached tiles only for surrounding areas during fusion.

Due to the complex structure of Neuro Generative Tile Fusion (NGTF), the cache feature isn't yet fully effective for repairing individual tiles in NGTF. This is because overlapping areas from one tile don't directly affect the surrounding tiles without recalculating them too, especially when using higher denoise settings, resulting in imperfect blending.

We're actively working to improve this behavior and aim to address it in the next release.

And we add new WORKFLOWS for 1.04 alfa
