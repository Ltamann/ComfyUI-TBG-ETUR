# TBG_Enhanced Tiled Upscaler & Refiner FLUX PRO


Welcome to **TBG Enhanced Tiled Upscaler and Refiner Pro!**
We at **TBG Think. Build. Generate. AI upscaling & image enrichment** are
excited to make our TBG Enhanced Tiled Upscaler and Refiner Pro available to you.

# Alpha Testing of PRO Features Now Available

We’re excited to announce that alpha testing of the PRO version is now live
for our Patreon supporters and free community members!

Please help us to find all Bugs and open an issue here !!!


Get early access by joining us at:
https://www.patreon.com/TB_LAAR

The CE (Community Edition) nodes are free and standalone, suitable for any type of workflow. For access to some PRO features, a TB_LAAR Patreon membership is required. The free version is sufficient for testing and experimenting. Once you become a member, you can obtain your API key:
https://api.ylab.es/login.php 


**TBG Enhanced Tiled Upscaler and Refiner Pro** is an advanced, modular enhancement suite for **tiled image generation and refinement** in ComfyUI. It introduces neuro generative tile fusion, interactive tile-based editing, and multi-path processing pipelines designed for extreme resolution workflows up to but not limitet to 100MP, high fidelity, and adaptive post-generation control.

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



Instructions for Installing TBG_upscaler_Alfa_1.03 for ComfyUI
--------------------------------------------------------------
1. Install this git from the Manager:
   
   [https://github.com/Ltamann/ComfyUI-TBG-ETUR](https://github.com/Ltamann/ComfyUI-TBG-ETUR)

or Copy the folder 'ComfyUI-TBG-ETUR' into your ComfyUI custom_nodes directory:
   ..\ComfyUI\custom_nodes\
   
   and then:

   pip install -r requirements.txt


2. If you like to test teh PRO feachers get early access by joining us at:
   https://www.patreon.com/TB_LAAR

   You must be a TB_LAAR Patreon supporter or free member to get an API key which you can get here:
   https://api.ylab.es/login.php 
   Paste it directly into the TBG Tiler node
   
--------------------------------------------------------------

Important:
To use PRO features, you must stay connected to the internet.
If you're offline, you can still use all the Community Edition features for free.

Recommended Workflow:
Do your setup and testing with PRO features turned OFF,
and only enable them for the final steps of your workflow.

Thank you for your support and happy tiling!

Just to clarifying the Originality of the TBG Tiled Upscaler CE and PRO Node, this node is not a mere bundle of third-party features—it embodies a completely new tiling concept developed, refined, and rigorously tested over countless hours of engineering and experimentation. Any external code it leverages is strictly minimal and used only where it made sense to avoid reinventing low-level utilities. Here’s exactly what was borrowed and why:
- USDS Crop-Conditioning Node
   Used solely for its crop-conditioning routine to generate per-tile conditioning masks.
   All integration and per-tile looping logic around it was handcrafted.
  https://github.com/ssitu/ComfyUI_UltimateSDUpscale
- LLM Unload Model Function
   Incorporated to correctly unload large language models between tile runs, preventing memory bloat.
   The orchestration and tile-level invocation logic are original to this node.
  https://github.com/willblaschko/ComfyUI-Unload-Models
- MaraScott’s Tile Promter
   With MaraScott’s explicit permission, their loader and setup code served as a starting scaffold.
   Every tiling, scheduling, and upscaling mechanism was rewritten or heavily refactored.
  https://github.com/MaraScott/ComfyUI_MaraScott_Nodes
- KJ Nodes Color-Match Function
   Employed for its robust color-matching algorithm.
   Coded into the different tile to tile blendings and not as an one time final process so it had to stay inside the code.
  https://github.com/kijai/ComfyUI-KJNodes
- ImpactPack Segment Conversion
   Utilized only to convert internal segment formats to the mask_to_seg format that any ImpactPack node can handel.
   All tile segmentation, merging, and fusion logic is proprietary.
  https://github.com/ltdrdata/ComfyUI-Impact-Pack
- EnrichmentPipe Extension - FluxTopas RectifyFlowInversion
   Brought in for its flow inversion routine; the per-tile applicator and boundary handling are homegrown.
  https://github.com/logtd/ComfyUI-Fluxtapoz
- EnrichmentPipe Extension - DeamonToles Detail Daemon Node
   Integrated only for its detail_daemon_node implementation; And names if is.
  https://github.com/daemontools/daemontools
- ControlNet-Aux Code
   Borrowed snippets to enable a per-tile ControlNet branch.
   Crucially, the preprocessor and ControlNet model call within the tile loop are original and make per-tile CNet a much better experience.
  https://github.com/Fannovel16/comfyui_controlnet_aux


