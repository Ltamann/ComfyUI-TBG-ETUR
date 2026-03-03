"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________

"""
import comfy
import comfy.latent_formats
import comfy.model_sampling
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models
import folder_paths

from ..UpscalerRefiner.TBG_Refiner import TBG_Refiner_v1
from ..UpscalerRefiner.TBG_Tiler import TBG_Upscaler_v1
from ...vendor.ComfyUI_Impact_Pack.masktoseg import MaskToSEGS



class TBG_ETUR_Upscaler_and_Tile_Generator_PRO():


    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    SEGMENTS = {}
    SIZE = {}
    LLM = {}

    PRESETS  = [
    'NONE',
    'Full size Image no Tiles',
    ]

    DIFFUSION_MODES = [
        'Soft Merge',
        'Neuro_Generative_Tile_Fusion',
    ]
    ROUND_METHODS = [
        'Disabled',
       # 'Enabled',
       # 'Enabled_XL',
    ]
    UPSCALE_TYPE = [
        'NONE',
        'Upscale Image By',
        'Upscale Image By (using Model)',
        'Upscale Image (using Model)',

    ]
    UPSCALE_METHODS = [
        "area",
        "bicubic",
        "bilinear",
        "bislerp",
        "lanczos",
        "nearest-exact",
        "Tiled-SeedVR2 Fast",
        "Tiled-SeedVR2 Standard",
        "Tiled-SeedVR2 High",
        "Tiled-SeedVR2 Ultra",
        "FlashVSR-v1.1 Small 8GB",
        "FlashVSR-v1.1 Big 18GB",
    ]

    from nodes import NODE_CLASS_MAPPINGS

    node_name = "FastVLM7BNode"

    if node_name in NODE_CLASS_MAPPINGS:
        NodeAClass = NODE_CLASS_MAPPINGS[node_name]
        VLM = [
            "NONE",
            "Janus-Pro-1B",
            "Janus-Pro-7B",
            "SkyCaptioner-V1",
            "SkyCaptioner-V1_8bit",
            "SkyCaptioner-V1_4bit",
            "Qwen3-VL-2B-Instruct",
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-2B-Instruct-FP8",
            "Qwen3-VL-2B-Thinking-FP8",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-4B-Instruct-FP8",
            "Qwen3-VL-4B-Thinking-FP8",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-VL-8B-Thinking",
            "Qwen3-VL-8B-Instruct-FP8",
            "Qwen3-VL-8B-Thinking-FP8",
            "Qwen3-VL-32B-Instruct",
            "Qwen3-VL-32B-Thinking",
            "Qwen3-VL-32B-Instruct-FP8",
            "Qwen3-VL-32B-Thinking-FP8",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "OpenAI-Compatible (Labs Server)",
            "GGUF\\Qwen3-VL-4B-Instruct-F16",
            "GGUF\\Qwen3-VL-4B-Instruct-Q4_K_M",
            "GGUF\\Qwen3-VL-4B-Instruct-Q8_0",
            "GGUF\\Qwen3-VL-8B-Instruct-F16",
            "GGUF\\Qwen3-VL-8B-Instruct-Q4_K_M",
            "GGUF\\Qwen3-VL-8B-Instruct-Q8_0",
            "microsoft/Florence-2-base",
            "microsoft/Florence-2-base-ft",
            "microsoft/Florence-2-large",
            "microsoft/Florence-2-large-ft",
            "HuggingFaceM4/Florence-2-DocVQA",
            "thwri/CogFlorence-2.1-Large",
            "thwri/CogFlorence-2.2-Large",
            "gokaygokay/Florence-2-SD3-Captioner",
            "gokaygokay/Florence-2-Flux-Large",
            "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
            "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
            "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
            "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
            "PJMixers-Images/Florence-2-base-Castollux-v0.5",
            "Apple FastVLM 7B Research use only",
        ]
    else:
        VLM = [
            "NONE",
            "Janus-Pro-1B",
            "Janus-Pro-7B",
            "SkyCaptioner-V1",
            "SkyCaptioner-V1_8bit",
            "SkyCaptioner-V1_4bit",
            "Qwen3-VL-2B-Instruct",
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-2B-Instruct-FP8",
            "Qwen3-VL-2B-Thinking-FP8",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-4B-Instruct-FP8",
            "Qwen3-VL-4B-Thinking-FP8",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-VL-8B-Thinking",
            "Qwen3-VL-8B-Instruct-FP8",
            "Qwen3-VL-8B-Thinking-FP8",
            "Qwen3-VL-32B-Instruct",
            "Qwen3-VL-32B-Thinking",
            "Qwen3-VL-32B-Instruct-FP8",
            "Qwen3-VL-32B-Thinking-FP8",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "OpenAI-Compatible (Labs Server)",
            "GGUF\\Qwen3-VL-4B-Instruct-F16",
            "GGUF\\Qwen3-VL-4B-Instruct-Q4_K_M",
            "GGUF\\Qwen3-VL-4B-Instruct-Q8_0",
            "GGUF\\Qwen3-VL-8B-Instruct-F16",
            "GGUF\\Qwen3-VL-8B-Instruct-Q4_K_M",
            "GGUF\\Qwen3-VL-8B-Instruct-Q8_0",
            "microsoft/Florence-2-base",
            "microsoft/Florence-2-base-ft",
            "microsoft/Florence-2-large",
            "microsoft/Florence-2-large-ft",
            "HuggingFaceM4/Florence-2-DocVQA",
            "thwri/CogFlorence-2.1-Large",
            "thwri/CogFlorence-2.2-Large",
            "gokaygokay/Florence-2-SD3-Captioner",
            "gokaygokay/Florence-2-Flux-Large",
            "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
            "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
            "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
            "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
            "PJMixers-Images/Florence-2-base-Castollux-v0.5"
        ]

    VLM_quant = [
        "4-bit (VRAM-friendly)",
        "8-bit (Balanced)",
        "None (FP16)"
    ]
    # Get the list of filenames
    upscale_models = folder_paths.get_filename_list("upscale_models")

    # Add "None" at the beginning (or end)
    upscale_models = ["NONE",
                        "FAST/_area",
                        "FAST/_bicubic",
                        "FAST/_bilinear",
                        "FAST/_bislerp",
                        "FAST/_lanczos",
                        "FAST/_nearest-exact",
                        "SuperResolution/Tiled-SeedVR2 Fast",
                        "SuperResolution/Tiled-SeedVR2 Standard",
                        "SuperResolution/Tiled-SeedVR2 High",
                        "SuperResolution/Tiled-SeedVR2 Ultra",
                        "SuperResolution/FlashVSR-v1.1 Small 8GB",
                        "SuperResolution/FlashVSR-v1.1 Big 18GB",
                        "Waifu/art",
                        "Waifu/art noise 1",
                        "Waifu/art noise 2",
                        "Waifu/art noise 3",
                        "Waifu/photo",
                        "Waifu/photo noise 1",
                        "Waifu/photo noise 2",
                        "Waifu/photo noise 3"] + upscale_models  # or upscale_models + ["None"]
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "Optimize_Tile_Size": (self.ROUND_METHODS, {"label": "Optimize_Tile_Size", "default": "Disabled"}),
            },

            "required": {
                "image": ("IMAGE", {"label": "Image"}),
                "presets": (self.PRESETS, {"label": "presets", "default": "NONE"}),
                "Fragmentation":("FLOAT",{"label": "Fragmentation", "default": 1, "min": 0.5, "max": 4, "step": 0.01, "round": 0.01}),
                "tile_size_w": ("INT",{"label": "Tile Size height", "default": 1024, "min": 320, "max": 8192, "step": 64}),
                "tile_size_h": ("INT",{"label": "Tile Size width", "default": 1024, "min": 320, "max": 8192, "step": 64}),
                "upscale_model": (self.upscale_models, {"label": "Upscale Model","default":"NONE"}),
                "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 8, "step": 0.05, "round": 0.01}),
                #"upscaler_method": (self.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'bilinear'}),
                "VLM_Model": (
                    self.VLM,
                    {
                        "label": "VLM_Model",
                        "default": "NONE",
                        "tooltip": (
                            "Check the license for all models. "
                            "Apple’s model is for research use only and the files are not included in this custom node. "
                            "It must be installed separately by the user. "
                            "Use 'OpenAI-Compatible (Labs Server)' to route VLM requests to the server configured in the Labs Upscaler node."
                        )
                    }
                ),
                "VLM_Quantization": (self.VLM_quant, {"default": "None (FP16)", "tooltip":"Only for QWEN 3 models"
                        }),
                "VLM_Prompt": ("STRING", {"multiline": True, "label": "LLMPrompt Prompt",
                                                "default": "Provide a highly detailed description of the image, emphasizing materials and textures. Enhance every visual detail, including accurate colors, lighting, and stylistic elements. Also describe the artistic or photographic style, such as film type, camera style, era, or overall aesthetic."}),
                "VLM_Selected_Tiles_Only": ("BOOLEAN", {"label": "Process_selected_Tiles_only", "default": False, "label_on": "Generate Selected Tiles Only", "label_off": "Disabled"}),
                "VLM_Selected_Tiles_By_Numbers": ("STRING", {"label": "Selected_Tiles_Index_Numbers to process", "default": '',
                                                         "tooltip": "You can set a list of selected tiles to process like 1,2,3,6 and activate Selected_Tiles_Only"}),
                "VLM_seed": ("INT", {"label": "Seed", "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,"fixed": True  }),

                #"PRO_activate": ("BOOLEAN", {"label": "api_activate_pro", "default": True, "label_on": "ETUR PRO","label_off": "ETUR"}),
                #PRO_Tile_Fusion_Mode":
                "Fusion Mode": (self.DIFFUSION_MODES, {"label": "PRO_Tile_Fusion_Mode", "default": "Neuro_Generative_Tile_Fusion"}),
                #PRO_Tile_Fusion_margin
                "Fusion Margin": ("INT",{"label": "PRO_Tile_Fusion_blur_margin","default": 64,"min": 0, "max": 128, "step": 8}),
                #PRO_Fusion_Space_Denoise"
                "Fusion Strength": ("FLOAT",
                                             {"label": "Fusion Strength", "default": 0.95, "min": 0,
                                              "max": 1, "step": 0.01, "round": 0.01,
                                              "tooltip": "Controls the fusion mask. If you see light color shifts on tile borders, lower this a little. Progressively reduces the fusion edge effect around tile borders: 0 = Off, 1 = Full."}),
                "Feather Mask": ("INT",
                                          {"tooltip": "Feather mask is a gradient used only in final image compositing to smoothly blend tiles together using neuro-generative tile fusion.", "default": 16, "min": 0,
                                           "max": 128, "step": 8}),
            },
            "optional": {
                "Segment_Mask": ("MASK",),
                "PRO_api_token": ("STRING", {"default": ""}),
                "PRO_segs": ("SEGS",),
                "labs_upscaler": ("labs_upscaler", {"label": "TBG ETUR Labs", "tooltips": "TBG ETUR experimental tools"}),
            }

        }

    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    RETURN_TYPES = (
        "TBG_Pipe",
        "IMAGE"
    )

    RETURN_NAMES = (
        "TBG_Pipe",
        "Upscaled Image"
    )

    OUTPUT_IS_LIST = (
        False,
        False,
    )

    OUTPUT_NODE = True
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'Upscaler and Tiler to split you images into tiles for TBG ETUR'
    FUNCTION = "fn"


    @classmethod
    def fn(self, **kwargs):
        # Read an environment variable
        TBG_Upscaler_v1i = TBG_Upscaler_v1()
        result =  TBG_Upscaler_v1i.fn(**kwargs)
        current_credits = 0
        _,_,_,_,_,_,_,_,userinfo,_,_,infos= result[0]
        return {
            "ui": {"value": [f"{userinfo}", infos]},
            "result": result
        }

class TBG_ETUR_Refiner_PRO():

    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'Sampler and Refiner for TBG ETUR'


    SIZE = None
    SEGMENTS = None
    OUTPUTS = None
    KSAMPLER = None
    INPUTS = None
    PARAMS = None


    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX2': 2048,
        'FLUX1 Kontext': 1024,
        'Qwen Image': 1328,
        'Qwen Image Edit': 1328,
        'Z-Image': 1024,
        'Others': 1024,
    }


    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())

    DENOISE_METHODS = [
        'default',
        'normalized advanced',
        'default short',
    ]

    COLOR_MATCH_METHODS = [
        'none',
        'lab color match+detail preservation',
        'lab full color match',
        'wavelet',
        'wavelet_adaptive',
        'hsv',
        'adain',
        'mkl',
        'hm',
        'reinhard',
        'mvgd',
        'hm-mvgd-hm',
        'hm-mkl-hm',
    ]
    DIFFUSION_MODES = [
        'From TGB Tiler Node',
        'Neuro_Generative_Tile_Fusion',
    ]




    @classmethod
    def INPUT_TYPES(self):
        # def INPUT_TYPES(cls):

        return {

            "optional": {

                #"Dual_Model_General_Prompt_Positive_low": ("STRING", {"tooltip": "Dual_Model_General_Prompt_Positive_low", "multiline": True, "label": "Dual Model low General Prompt for all Tiles", "default": ""}),
                #"Dual_Model_General_Prompt_Negative_low": ("STRING", {"tooltip": "Dual_Model_General_Prompt_Negative_low", "multiline": True, "label": "Dual Model low General Prompt for all Tiles",
                #                                                      "default": "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理"}),
                #"Dual_Model_model_low": ("MODEL", {"label": "Model",
                #                                   "tooltip": "Dual Model are models working in pair in the same Latent space (same VAE) - like Flux and Z-image, or Qwen and Wan, low model has no cnets support)"}),
                #"Dual_Model_clip_low": ("CLIP", {"label": "Clip"}),
                #"Dual_Model_vae_low": ("VAE", {"label": "VAE"}),
                #"Dual_Model_cfg_low": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                #"Dual_Model_steps_low": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),
                #"Dual_Model_high_low_swap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "OnlY for Dual Model / % where model swap recommend 0.4"}),
                #"Dual_Model_low_refiner": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "OnlY for Dual Model / shifts the sigma up and done < 1= more noise >1 softer"}),

                "Controlnet_Pipe": ("Controlnet_Pipe", {"label": "TBG ControlNet Pipe"}),
                "Enrichment_Pipe": ("Enrichment_Pipe", {"label": "TBG enrichment Pipe"}),

                "denoise_mask": ("MASK",),
                "Redux_Style_Model": ("STYLE_MODEL", {"label": "Redux_Style_Model"}),
                "Redux_Clip_Vision": ("CLIP_VISION", {"label": "Redux_Clip_Vision"}),
                "labs_refiner": ("labs_refiner", {"label": "TBG ETUR Labs", "tooltips": "TBG ETUR experimental tools"}),
            },
            "required": {
                "model_type": (self.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),

                "model": ("MODEL", {"label": "Model"}),
                "clip": ("CLIP", {"label": "Clip"}),
                "vae": ("VAE", {"label": "VAE"}),
                "cfg": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),


                "TBG_Pipe": ("TBG_Pipe", {"label": "TBG Pipe"}),


                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "fixed": True}),
                "Flux_Guidance": ("FLOAT",{"label": "Flux Guidance for Tiles", "default": 3.5, "min": -100.0, "max": 100.0,"step": 0.1, "round": 0.01,  "tooltip": "All Fusion Modes benefit from high Guidance, so if you notice that certain areas aren't blending well, try increasing the Guidance value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),


                "vae_encode": ("BOOLEAN", {"label": "VAE Encode type", "default": True, "label_on": "tiled slow","label_off": "tbg Color-preserving fast",  "tooltip": ""}),
                "tile_size_vae": ("INT",{"label": "Tile Size (VAE)", "default": 1024, "min": 256, "max": 4096, "step": 64}),
                "General_Prompt_Positive": ("STRING", {"tooltip": "General_Prompt_Positive", "multiline": True, "label": "General Positive Prompt for all Tiles", "default": ""}),
                "General_Prompt_Negative": ("STRING",  {"tooltip": "General_Prompt_Negative", "multiline": True, "label": "General Negative Prompt for all Tiles",
                                                        "default": "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理"}),

                "denoise": ("FLOAT", {"label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_method": (self.DENOISE_METHODS, {"label": "DENOISE_METHODS", "default": 'normalized advanced',
                                    "tooltip":"Default: splits sigmas by step percentage and interpolates back to full steps | Normalize Advanced: splits by sigma noise values and interpolates back to full steps, best for sampler-independent creative control | Default Short: splits sigmas by step percentage and keeps only low-noise steps for efficient img2img denoising."}),
                "Per_Pixel_Denoise_Mask_Strength": ("FLOAT", {"display": "slider", "label": "Per Pixel Denoise Mask Strength", "default": 1, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                  "tooltip": "Changes the influence of the Per_Pixel_Denoise_Mask"}),
                #PRO_Fusion_Complexity_Mask_Strength
                "Image Stabilizer": ("FLOAT", {"display": "slider", "label": "Image Stabilizer", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                  "tooltip": "0=OFF. Applys adaptive denoising based on local image complexity. Flat regions are stabilized to prevent color shifts, while detailed areas allow stronger creative changes. Useful for light backgrounds and large uniform areas and as alternative to cnets."}),
                "Smoother - Sharper": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "display": "slider",
                                               "tooltip": "0=OFF. Dual-stage adaptive sharpening. At high sigma (early steps), adds structured noise for detail invention. At low sigma (late steps), applies high-pass edge sharpening. Positive values sharpen and add details. Negative values soften and blur. Zero disables sharpening. Higher absolute values create stronger effects."}),
                "Detail_Enhancer": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "display": "slider",
                                              "tooltip": "0=OFF. Substep evaluation for detail control. Positive values (0.1-1.0): lookahead to next sigma, adds coherent details and refinement, reduces variation. Negative values (-0.1 to -1.0): lookback to previous sigma, adds creative variation and texture complexity. Zero = disabled (single pass, fastest). Performance cost: 2x slower on affected steps."}),
                "Redux_strength": ("FLOAT", {"display": "slider", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001,
                                             "tooltip": "0=OFF. It's a Redux multiplier value applied uniformly to all Tiles"}),
                "Controlnet_Pipe_strength": ("FLOAT", {"display": "slider", "label": "Controlnet_Pipe_strength", "default": 1.00, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                       "tooltip": "0=OFF. It's a multiplier value applied uniformly to all ControlNets from CnetPipe, scaling their combined influence."}),
                "Color_Match": (self.COLOR_MATCH_METHODS, {"label": "Color Match Method", "default": 'none'}),
                "Fast_1_Tile_Preview": ("BOOLEAN", {"label": "Fast_1_Tile_Preview", "default": False, "label_on": "Preview Single Tile", "label_off": "Disabled",
                                                    "tooltip": "The first Selected_Tiles_By_Number are processed at full scale as a preview, allowing a quick check of settings before processing the entire set."}),
                "Selected_Tiles_Only": ("BOOLEAN", {"label": "Process_selected_Tiles_only", "default": False, "label_on": "Generate Selected Tiles Only", "label_off": "Disabled"}),
                "Selected_Tiles_By_Numbers": ("STRING", {"label": "Selected_Tiles_Index_Numbers to process", "default": '',
                                                         "tooltip": "You can set a list of selected tiles to process like 1,2,3,6 and activate Selected_Tiles_Only"}),

                "VRAM_Profile": (
                    [
                        "Fast Cache (Max Speed)",
                        "Low VRAM Cache (Unload Models)",
                        "Ultra Low Memory (Per-Tile Streaming)",
                    ],
                    {
                        "label": "VRAM Profile",
                        "default": "Low VRAM Cache (Unload Models)",
                        "tooltip": (
                            "Fast Cache (Max Speed): Precomputes full tile conditioning (text + Redux + ControlNet) "
                            "for all tiles and keeps models loaded. Fastest sampling, highest RAM/VRAM usage. "
                            "Low VRAM Cache (Unload Models): Same full precompute, then unloads models to reduce VRAM. "
                            "RAM can still be high with many tiles. "
                            "Ultra Low Memory (Per-Tile Streaming): Caches repeated text conditioning only; Redux/ControlNet "
                            "are rebuilt per tile and released immediately. Also unloads/reloads models between steps/tiles "
                            "for minimum VRAM. Slowest mode; best for very low-spec systems."
                        ),
                    },
                ),

            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "Debug_Grid_Overlay": ("BOOLEAN", {"label": "Debug_Grid_Overlay", "default": False,"label_on": "Show Grid","label_off": "Disabled"}),
                "contrast": ("INT", {"label": "contrast", "default": 0, "min": 0, "max": 100.0, "step": 1}),
                "highpass": ("FLOAT",{"highpass": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "Enhanced_Laplacian_Blending ": ("BOOLEAN", {"label": "Laplacian Pyramid Blending", "default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Work in progress"}),

            },

        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE"
    )

    RETURN_NAMES = (
        "Refined",
        "Refined without Segs",
        "Refined without ColorCorrection",
        "Original Upscaled",
        "Original",
        "Tiles"
    )

    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)

    OUTPUT_NODE = True
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        return {
            "ui": {"value": [f"{kwargs.get('seed', None)}"]},
            "result": (TBG_Refiner_v1.fn(**kwargs))
        }

class TBG_ETUR_Labs_Refiner():

    OUTPUT_NODE = True
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "TBG's home for ETUR experimental tools"
    FUNCTION = "fn"


    STICHTYPE = [
        'gpupyramid',
        'cpupyramid',
    ]

    LABS = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "Custom_Sigmas_!DENOISE=1": ("SIGMAS", {"label": "Sigmas with denoise 1", "tooltip": "Insert your full custom sigma noise curve (not denoised), as denoising is performed per tile by the node."}),
                "Sampler": ("SAMPLER", {"label": "Overwrites the Sampler Name field"}),
                "cropped_positive": ("CONDITIONING",),
                "cropped_negative": ("CONDITIONING",),

            },
            "required": {

               "Tile_Fusion_Blend": ("FLOAT", {"label": "Tile_Fusion_Blend", "default": 0.5, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                "tooltip": "Fusion margin with neighboring tiles (default: 0.5 = same value as Fusion Margin), lower values reduce risk of overlapping artifacts; higher values create more diffused blending but may risk visible seams."}),
                "Fusion_end": ("INT", {"display": "slider", "default": 0, "min": -50, "max": 0,
                                        "tooltip": "Step number from the end after which fusion is skipped. For example, with 20 total steps, setting -10 means fusion runs only from step 1 to 10."}),

                "Fusion_conditioning": ("BOOLEAN", {"label": "inpaint_conditioning", "default": True,
                                                     "tooltip":"If true, fusion works with both conditioning and noise_mask; if false, it works only with noise_mask."}),
                "LanPaint": ("BOOLEAN", {"label": "LanPaint", "default": True,
                                         "tooltip": "LanPaint: Universal Inpainting Sampler with Think Mode"}),
                "Differential_Diffusion": ("BOOLEAN", {"label": "Differential_Diffusion", "default": True,
                                         "tooltip": "Differential_Diffusion: ON OFF"}),
                #"point_grid_image_stabilizer_experimental"
                "Color & Structure Stabilizer": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "display": "slider",
                                                           "tooltip": "Uses a grid of anchor points to stabilize color and structure, reducing drift and preserving spatial coherence. Experimental feature."}),
                "Save_Tiles_in_Temp_Folder": ("BOOLEAN", {"label": "Preview_Tiles_in_Temp_Folder", "default": False, "label_on": "Save Tiles to /temp/TBG/", "label_off": "Disabled",
                                              "tooltip": "For full-resolution in-process previews, tiles are saved to temp/TBG/compareTiles/"}),
                "stitch_blending": (cls.STICHTYPE, {"label": "stich_blending", "default": "gpupyramid"}),
                "max_upscale_size_segment": ("INT", {"label": "max_upscale_size_segment", "default": 2048, "min": 256,
                                                     "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("labs_refiner",)

    @classmethod
    def fn(self, **kwargs):
        return (kwargs,)


class TBG_ETUR_Labs_Upscaler():

    OUTPUT_NODE = True
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "TBG's home for ETUR experimental tools"
    FUNCTION = "fn"
    LABS = None
    WORKER_CLOSE = [
        'ShutDown after each run',
        'ShutDown after each run delayed',
        'Close with Comfyui',
        'Keep Running (not recommended)',
    ]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TBG_APP_ShutDown": (
                    cls.WORKER_CLOSE,
                    {
                        "label": "TBG_APP_ShutDown",
                        "default": "Close with Comfyui",
                        "tooltip": (
                            "Control how TBG_APP manages memory after each run:\n"
                            "startup could easily take ~1–5 seconds\n"
                            "- 'ShutDown after each run': Frees memory immediately after Refiner is finished but restarting takes a few seconds.\n"
                            "- 'ShutDown after each run delayed': Frees memory after 1 minutes after Refiner is finished, balancing speed and memory usage (recommended).\n"
                            "- 'Keep Running (not recommended)': Keeps the app running 1h after Work is finished to avoid delays and to keep memory after Comfyui Restart.\n"
                        )
                    }
                ),
            },
            "optional": {
                "Tiler_Upscale_Cache": ("BOOLEAN", {"label": "Activate Upscale Cache", "default": True, "label_on": "ON", "label_off": "OFF"}),
                "Only_Upscale": ("BOOLEAN",
                                        {"label": "Only Upscale", "default": False, "label_on": "ON",
                                         "label_off": "OFF"}),
                "VLM_Server_Base_URL": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080/v1",
                        "tooltip": (
                            "OpenAI-compatible VLM server base URL used when PRO VLM_Model is set to "
                            "'OpenAI-Compatible (Labs Server)'. ETUR normalizes this to /v1 and calls /v1/chat/completions. "
                            "API key must be provided via environment variable: TBG_ETUR_OPENAI_API_KEY."
                        ),
                    },
                ),
                "VLM_Server_Model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Model name passed to /v1/chat/completions when PRO VLM_Model is "
                            "'OpenAI-Compatible (Labs Server)'. API key is env-only: TBG_ETUR_OPENAI_API_KEY."
                        ),
                    },
                ),
                "SEEDVR2_DIT": ("SEEDVR2_DIT",),
                "SEEDVR2_VAE": ("SEEDVR2_VAE",),
                "PRO_Tile_Fusion_shift_in_out": ("INT",{"label": "PRO_Tile_Fusion_shift_in_out", "default": 0, "min": -128, "max": 128,"step": 8}),
                "PRO_Tile_Fusion_shift_top_left": ("INT",{"label": "PRO_Tile_Fusion_shift_top_left", "default": 0, "min": -128, "max": 128,"step": 8}),
                "PRO_Tile_Fusion_border_margin": ("INT", {"label": "shift_mask", "default": 16, "min": 0, "max": 128,"step": 8}),
            }
        }

    RETURN_TYPES = ("labs_upscaler",)

    @classmethod
    def fn(self, **kwargs):
        return (kwargs,)





