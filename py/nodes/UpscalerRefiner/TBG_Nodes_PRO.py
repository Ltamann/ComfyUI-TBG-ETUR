"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________

"""
import comfy
import comfy.latent_formats
import comfy.model_sampling
import comfy.patcher_extension
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models
import folder_paths
import torch
import torch.nn.functional as F

from ..UpscalerRefiner.TBG_Refiner import TBG_Refiner_v1
from ..UpscalerRefiner.TBG_Tiler import TBG_Upscaler_v1
from ..UpscalerRefiner.inc.image import TBG_Image
from ..UpscalerRefiner.inc.sift_drift import DRIFT_CORRECTION_MODES, apply_sift_drift_correction
from ..UpscalerRefiner.inc.tbg_pid import (
    PID_UPSCALE_OPTIONS,
    PID_VAE_COMPATIBLE_MODEL_TYPES,
    get_pid_upscale_options,
)
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
                        *PID_UPSCALE_OPTIONS,
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
        local_upscale_models = folder_paths.get_filename_list("upscale_models")
        self.upscale_models = ["NONE",
                        "FAST/_area",
                        "FAST/_bicubic",
                        "FAST/_bilinear",
                        "FAST/_bislerp",
                        "FAST/_lanczos",
                        "FAST/_nearest-exact",
                        *get_pid_upscale_options(refresh=True),
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
                        "Waifu/photo noise 3"] + local_upscale_models
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
                "Optimize_Upscale_Factor_For_Tile_Use": ("BOOLEAN", {
                    "label": "Optimize upscale factor for optimal tile use",
                    "default": False,
                    "tooltip": "Keeps the tile grid from the requested upscale factor, then lowers the effective scale to the largest size that uses only the required Soft Merge or NGTF overlap."
                }),
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
                "VLM_Process_Selected_Tiles_Only": ("BOOLEAN", {"label": "VLM Process Selected Tiles Only", "default": False, "label_on": "Generate VLM Prompts For Selected Tiles Only", "label_off": "Disabled"}),
                "VLM_Selected_Tiles_Index_Numbers": ("STRING", {"label": "VLM Selected Tiles Index Numbers", "default": '',
                                                         "tooltip": "You can set a list of selected tiles for VLM prompt generation like 1,2,3,6 and activate VLM Process Selected Tiles Only"}),
                "VLM_seed": ("INT", {"label": "Seed", "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,"fixed": True  }),

                #"PRO_activate": ("BOOLEAN", {"label": "api_activate_pro", "default": True, "label_on": "ETUR PRO","label_off": "ETUR"}),
                #PRO_Tile_Fusion_Mode":
                "Fusion Mode": (self.DIFFUSION_MODES, {"label": "PRO_Tile_Fusion_Mode", "default": "Neuro_Generative_Tile_Fusion"}),
                #PRO_Tile_Fusion_margin
                "Fusion Reference Margin": ("INT",{"label": "PRO_Tile_Fusion_reference_margin","default": 0,"min": 0, "max": 256, "step": 8,
                                                      "tooltip": "Reference border content kept around each NGTF tile and cropped out when stitching the image back together."}),
                "Fusion Margin": ("INT",{"label": "Fusion Blur","default": 64,"min": 0, "max": 128, "step": 8,
                                           "tooltip": "Fusion blur width used for NGTF tile blending."}),
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
        'Ideogram4': 2048,
        'FLUX1 Kontext': 1024,
        'Qwen Image': 1328,
        'Qwen Image Edit': 1328,
        'SDXL': 1024,
        'SD3': 1024,
        'Z-Image': 1024,
        'Others': 1024,
    }


    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())

    DENOISE_METHODS = [
        'default',
        'normalized advanced',
        'default short',
    ]

    VAE_ENCODE_TYPES = [
        'tiled slow',
        'tbg Color-preserving fast',
        'Nvidia PiD 4x',
    ]

    VAE_ENCODE_TOOLTIP = (
        "tiled slow is the standard ComfyUI tiled VAE path. "
        "tbg Color-preserving fast is the faster TBG VAE path that preserves colors. "
        "Nvidia PiD 4x changes VAE decoding to the PiD model; 1024x1024 uses the native fast path, while other tile or segment sizes use tiled PiD latent decode. It works with FLUX1, FLUX2, Qwen Image, Qwen Image Edit, SDXL, SD3, and Z-Image."
    )

    COLOR_MATCH_METHODS = [
        'none',
        TBG_Refiner_v1.TILE_STABILIZER_METHOD,
        'lab color match+detail preservation',
        'lab full color match',
        'wavelet',
        'wavelet_adaptive',
        'hsv',
        'adain',
        'mkl',
        'hm',
        'reinhard',
        'reinhard_lab_gpu',
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
                "RF_UntwistingRoPE": ("RF_UntwistingRoPE_Pipe", {"label": "TBG RF UntwistingRoPE"}),

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


                "vae_encode": (self.VAE_ENCODE_TYPES, {"label": "VAE Encode type", "default": "tiled slow", "tooltip": self.VAE_ENCODE_TOOLTIP}),
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
                "Color_Match": (self.COLOR_MATCH_METHODS, {
                    "label": "Color Match Method",
                    "default": TBG_Refiner_v1.TILE_STABILIZER_METHOD,
                    "tooltip": "Tile-aware detail-preserving color stabilization with feathered seam blending into generated neighbor tiles. If tile-aware conditions are unavailable, it falls back to TBG Detail-Preserving Color Stabilizer.",
                }),
                "Scale-Invariant Feature Transform": ("BOOLEAN", {
                    "label": "Geometry Drift Correction",
                    "default": True,
                    "label_on": "On",
                    "label_off": "Off",
                    "tooltip": "After a tile is generated, aligns it back to the reference tile. Off disables it. On uses the strongest x4 drift-correction path.",
                }),
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

class TBG_ETUR_ColorMatch_Debug():
    OUTPUT_NODE = True
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "Developer gates for isolating ETUR ColorMatch/global color stages."
    FUNCTION = "fn"

    SWITCHES = {
        "03_Flux2_PID_NormalVAE_Reference": "Reference only. Flux2 PiD normal tiles only. Creates a normal-VAE reference from sampled latent when that optional reference path is active.",
        "04_Flux2_PID_AfterPiDVAE_ColorMatch": "After PiD VAE decode. Flux2 PiD normal tiles only. Applies detail-preserving global RGB/luma correction; Color_Match only gates on/off.",
        "05_Flux2_PID_PostTone_ColorMatch": "After PiD VAE decode. Flux2 PiD normal tiles only. Applies post-tone/seam low-frequency stabilization without per-pixel ColorMatch.",
        "08_Segment_PostVAE_ColorMatch": "After segment sampling and segment PiD/VAE decode. Segments only. TBG Detail-Preserving Color Stabilizer uses global RGB/luma + 16x16 grid; eligible tile-aware runs use segment/restore-mask smooth grid correction.",
        "10_Final_TileOnly_ColorCorrection": "After normal tiles are stitched, before final segment rebuild. Applies non-structural global RGB/luma using original/upscaled input as reference.",
        "11_Final_SegmentAware_ColorBase": "Before final PiD color match. Builds final reference from original/upscaled input, optionally replacing selected segment areas with PiD segment crops.",
        "12_PID_Final_ColorMatch_4x": "Near final output. PiD final rebuild container. Color_Match=TBG Detail-Preserving Color Stabilizer routes to stage 14 unless protected tile/segment overrides require stage 13.",
        "13_Final_PerArea_SegmentOverrides": "Final protected/per-area path. Runs when a normal Color_Match method is selected, or when protected tile/segment overrides substitute the global RGB/luma pass. Protect/origin/off masks override global settings.",
        "14_Final_Global_ColorMode": "Final full/global path. Runs only when Color_Match is a TBG detail-preserving stabilizer and no protected tile/segment overrides exist. Uses global RGB/luma, not an old ColorMatch model.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "Override_Normal_Gates": ("BOOLEAN", {
                "label": "Override Normal Gates",
                "default": False,
                "tooltip": "OFF: switches only disable stages that normal ETUR settings already activated. ON: switches force stages on/off, including over tile/segment color gates. The TBG detail-preserving stabilizers use global RGB/luma + 16x16 grid; tile-aware still requires API member access and Neuro Generative Tile Fusion.",
            }),
        }
        for key, tooltip in cls.SWITCHES.items():
            required[key] = ("BOOLEAN", {
                "label": key,
                "default": True,
                "tooltip": tooltip,
            })
        return {"required": required}

    RETURN_TYPES = ("tbg_colormatch_debug",)

    @classmethod
    def fn(cls, **kwargs):
        values = {"_connected": True}
        values["Override_Normal_Gates"] = bool(kwargs.get("Override_Normal_Gates", False))
        for key in cls.SWITCHES:
            values[key] = bool(kwargs.get(key, True))
        return (values,)


class TBG_ETUR_ColorCorrection():
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "Standalone TBG color correction. TBG Detail-Preserving Color Stabilizer is global RGB/luma with a low-frequency 16x16 whole-image grid."
    FUNCTION = "fn"

    COLOR_MATCH_METHODS = [
        method for method in TBG_ETUR_Refiner_PRO.COLOR_MATCH_METHODS
        if str(method).strip().lower() not in TBG_Refiner_v1.TILE_STABILIZER_ALIASES
    ]
    if TBG_Refiner_v1.COLOR_STABILIZER_METHOD not in COLOR_MATCH_METHODS:
        COLOR_MATCH_METHODS.insert(1, TBG_Refiner_v1.COLOR_STABILIZER_METHOD)

    RGB_LUMA_ALIASES = {
        *TBG_Refiner_v1.COLOR_STABILIZER_ALIASES,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"label": "Image"}),
                "reference": ("IMAGE", {"label": "Reference Image"}),
                "Color_Match": (cls.COLOR_MATCH_METHODS, {
                    "label": "Color Match Method",
                    "default": TBG_Refiner_v1.COLOR_STABILIZER_METHOD,
                    "tooltip": "Detail-preserving global RGB/luma color stabilization with a low-frequency 16x16 whole-image grid. Tile-aware feathered seam blending is only available inside the ETUR refiner.",
                }),
                "Geometry_Drift_Correction": ("BOOLEAN", {
                    "label": "Geometry Drift Correction",
                    "default": True,
                    "label_on": "On",
                    "label_off": "Off",
                    "tooltip": "Optional alignment before color correction. Off disables it. On uses the strongest x4 drift-correction path.",
                }),
                "Color_Match_Str": ("FLOAT", {
                    "label": "Color Match Strength",
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

    @staticmethod
    def _ensure_bhwc(image):
        if not torch.is_tensor(image):
            return None
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            return None
        return image.clamp(0.0, 1.0)

    @classmethod
    def _resize_like(cls, image, target):
        image = cls._ensure_bhwc(image)
        target = cls._ensure_bhwc(target)
        if image is None or target is None:
            return image
        if int(image.shape[1]) == int(target.shape[1]) and int(image.shape[2]) == int(target.shape[2]):
            return image
        return torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),
            size=(int(target.shape[1]), int(target.shape[2])),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)

    @classmethod
    def _mask_like(cls, mask, target):
        if mask is None or not torch.is_tensor(mask):
            return None
        target = cls._ensure_bhwc(target)
        if target is None:
            return None
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.ndim != 3:
            return None
        mask = mask.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0).unsqueeze(1)
        if int(mask.shape[2]) != int(target.shape[1]) or int(mask.shape[3]) != int(target.shape[2]):
            mask = torch.nn.functional.interpolate(
                mask,
                size=(int(target.shape[1]), int(target.shape[2])),
                mode="bilinear",
                align_corners=False,
            )
        return mask.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)

    @staticmethod
    def _masked_mean(image, mask=None):
        if mask is None:
            return image.mean(dim=(0, 1, 2), keepdim=True)
        weight = mask.clamp(0.0, 1.0)
        denom = weight.sum(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
        return (image * weight).sum(dim=(0, 1, 2), keepdim=True) / denom

    @classmethod
    def _global_rgb_luma_match(cls, reference, target, strength=1.0, mask=None):
        ref = cls._resize_like(reference, target)
        target = cls._ensure_bhwc(target)
        if ref is None or target is None:
            return target
        ref = ref.to(device=target.device, dtype=torch.float32).clamp(0.0, 1.0)
        work = target.to(torch.float32).clamp(0.0, 1.0)
        mask = cls._mask_like(mask, work)
        if mask is not None:
            mask = mask.to(device=work.device, dtype=work.dtype)
            if mask.numel() <= 0 or float(mask.max().detach().cpu()) <= 0.001:
                mask = None
        strength = max(0.0, min(2.0, float(strength)))

        rgb_shift = (cls._masked_mean(ref, mask) - cls._masked_mean(work, mask)).clamp(-0.25, 0.25) * strength
        corrected = (work + rgb_shift).clamp(0.0, 1.0)

        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=work.device, dtype=work.dtype).view(1, 1, 1, 3)
        ref_luma = cls._masked_mean((ref * luma_weights).sum(dim=-1, keepdim=True), mask)
        corrected_luma = cls._masked_mean((corrected * luma_weights).sum(dim=-1, keepdim=True), mask)
        corrected = (corrected + (ref_luma - corrected_luma).clamp(-0.08, 0.08) * min(1.0, strength)).clamp(0.0, 1.0)
        try:
            delta_bchw = (ref - corrected).permute(0, 3, 1, 2).contiguous().clamp(-48.0 / 255.0, 48.0 / 255.0)
            grid = torch.nn.functional.interpolate(delta_bchw, size=(16, 16), mode="area")
            for _ in range(2):
                grid = cls._box_blur_bchw(grid, 3).clamp(-32.0 / 255.0, 32.0 / 255.0)
            field = torch.nn.functional.interpolate(
                grid,
                size=(int(corrected.shape[1]), int(corrected.shape[2])),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous().clamp(-32.0 / 255.0, 32.0 / 255.0)
            corrected = (corrected + field * min(1.0, strength) * 0.45).clamp(0.0, 1.0)
        except Exception:
            pass

        if mask is not None:
            corrected = (corrected * mask + work * (1.0 - mask)).clamp(0.0, 1.0)
        return corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

    @staticmethod
    def _box_blur_bchw(value, kernel):
        kernel = int(kernel)
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        return F.avg_pool2d(
            F.pad(value, (pad, pad, pad, pad), mode="reflect"),
            kernel,
            stride=1,
        )

    @staticmethod
    def _edge_zone_weights_bchw(height, width, device, dtype):
        y = torch.linspace(0.0, 1.0, int(height), device=device, dtype=dtype).view(1, 1, int(height), 1)
        x = torch.linspace(0.0, 1.0, int(width), device=device, dtype=dtype).view(1, 1, 1, int(width))
        top = (1.0 - y).clamp(0.0, 1.0).pow(3.0)
        bottom = y.clamp(0.0, 1.0).pow(3.0)
        left = (1.0 - x).clamp(0.0, 1.0).pow(3.0)
        right = x.clamp(0.0, 1.0).pow(3.0)
        centers = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        radius = 1.0 / 3.0
        x_segments = [(1.0 - (x - center).abs() / radius).clamp(0.0, 1.0).pow(2.0) for center in centers]
        y_segments = [(1.0 - (y - center).abs() / radius).clamp(0.0, 1.0).pow(2.0) for center in centers]
        raw = []
        raw.extend(top * segment for segment in x_segments)
        raw.extend(bottom * segment for segment in x_segments)
        raw.extend(left * segment for segment in y_segments)
        raw.extend(right * segment for segment in y_segments)
        total = torch.stack(raw, dim=0).sum(dim=0).clamp_min(1.0e-6)
        return [zone / total for zone in raw]

    @classmethod
    def _zoned_edge_field_bchw(cls, delta_bhwc, mask_bhwc, zones, kernel, clamp_value):
        delta_bchw = (delta_bhwc * mask_bhwc).permute(0, 3, 1, 2).contiguous()
        mask_bchw = mask_bhwc.permute(0, 3, 1, 2).contiguous()
        field_sum = torch.zeros_like(delta_bchw)
        weight_sum = torch.zeros_like(mask_bchw)
        for zone in zones:
            zone_mask = mask_bchw * zone
            blurred_weight = cls._box_blur_bchw(zone_mask, kernel)
            zone_field = cls._box_blur_bchw(delta_bchw * zone, kernel) / blurred_weight.clamp_min(1.0e-5)
            field_sum = field_sum + zone_field * zone
            weight_sum = weight_sum + zone
        return (field_sum / weight_sum.clamp_min(1.0e-6)).clamp(-clamp_value, clamp_value)

    @classmethod
    def _edge_rgb_luma_joint_alignment(cls, reference, target, strength=1.0, mask=None):
        ref = cls._resize_like(reference, target)
        target = cls._ensure_bhwc(target)
        if ref is None or target is None:
            return target
        ref = ref.to(device=target.device, dtype=torch.float32).clamp(0.0, 1.0)
        work = target.to(torch.float32).clamp(0.0, 1.0)
        mask = cls._mask_like(mask, work)
        if mask is None or mask.numel() <= 0:
            return cls._global_rgb_luma_match(reference, target, strength=strength, mask=None)
        input_mask = mask.to(device=work.device, dtype=torch.float32).clamp(0.0, 1.0)
        edge_mask = (1.0 - input_mask).clamp(0.0, 1.0)
        if float(edge_mask.max().detach().cpu()) <= 0.001:
            return cls._global_rgb_luma_match(reference, target, strength=strength, mask=None)
        strength = max(0.0, min(2.0, float(strength)))
        if strength <= 0.0:
            return target
        bchw_mask = edge_mask.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = bchw_mask.shape
        expand_kernel = max(25, min(65, int(round(min(height, width) * 0.055))))
        if expand_kernel % 2 == 0:
            expand_kernel += 1
        expanded_edge = cls._box_blur_bchw(bchw_mask.clamp(0.0, 1.0).pow(0.55), expand_kernel).clamp(0.0, 1.0)
        bchw_mask = (bchw_mask + expanded_edge * 0.85).clamp(0.0, 1.0)
        corrected = work
        inner_bchw = (1.0 - bchw_mask).clamp(0.0, 1.0)
        ref_bchw = ref.permute(0, 3, 1, 2).contiguous()

        def weighted_delta(ref_bchw, cur_bchw, zone, clamp_value):
            zone = zone.to(device=cur_bchw.device, dtype=torch.float32).clamp(0.0, 1.0)
            denom = zone.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-5)
            value = ((ref_bchw - cur_bchw) * zone).sum(dim=(2, 3), keepdim=True) / denom
            return value.clamp(-clamp_value, clamp_value)

        inner_apply_kernel = max(17, min(33, int(round(min(height, width) * 0.025))))
        if inner_apply_kernel % 2 == 0:
            inner_apply_kernel += 1
        inner_apply = cls._box_blur_bchw(inner_bchw, inner_apply_kernel).clamp(0.0, 1.0)
        cur_bchw = corrected.permute(0, 3, 1, 2).contiguous()
        inner_delta = weighted_delta(ref_bchw, cur_bchw, inner_bchw, 90.0 / 255.0)
        corrected = (
            corrected
            + (inner_delta * inner_apply * strength).permute(0, 2, 3, 1).contiguous()
        ).clamp(0.0, 1.0)

        edge_kernel = max(31, min(97, int(round(min(height, width) * 0.065))))
        if edge_kernel % 2 == 0:
            edge_kernel += 1
        cur_bchw = corrected.permute(0, 3, 1, 2).contiguous()
        yy = torch.linspace(0.0, 1.0, height, device=cur_bchw.device, dtype=torch.float32).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, width, device=cur_bchw.device, dtype=torch.float32).view(1, 1, 1, width)
        side_defs = (
            ((1.0 - yy).pow(2.0), xx),
            (yy.pow(2.0), xx),
            ((1.0 - xx).pow(2.0), yy),
            (xx.pow(2.0), yy),
        )
        field_sum = torch.zeros_like(cur_bchw)
        weight_sum = torch.zeros_like(bchw_mask)
        for side_weight, axis in side_defs:
            for zone_index in range(4):
                lo = zone_index / 4.0
                hi = (zone_index + 1) / 4.0
                axis_zone = ((axis >= lo) & (axis <= hi)).to(torch.float32)
                zone = (bchw_mask * side_weight * axis_zone).clamp(0.0, 1.0)
                if float(zone.sum().detach().cpu()) <= 1.0e-5:
                    continue
                zone_apply = cls._box_blur_bchw(zone.pow(0.45), edge_kernel).clamp(0.0, 1.0)
                zone_delta = weighted_delta(ref_bchw, cur_bchw, zone, 80.0 / 255.0)
                field_sum = field_sum + zone_delta * zone_apply
                weight_sum = weight_sum + zone_apply
        edge_field = (field_sum / weight_sum.clamp_min(1.0e-5)).clamp(-80.0 / 255.0, 80.0 / 255.0)
        edge_apply = cls._box_blur_bchw(bchw_mask.clamp(0.0, 1.0).pow(0.45), edge_kernel).clamp(0.0, 1.0)
        corrected = (
            corrected
            + (edge_field * edge_apply * (0.95 * strength)).permute(0, 2, 3, 1).contiguous()
        ).clamp(0.0, 1.0)

        # Center corrected edge pixels in their eventual 8-bit PNG bucket; Comfy's saver floors floats.
        corrected = (corrected + edge_mask * (0.25 / 255.0) * min(1.0, strength)).clamp(0.0, 1.0)
        return corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

    @classmethod
    def fn(cls, image, reference, Color_Match=None, Color_Match_Str=1.0, Geometry_Drift_Correction=True, mask=None):
        if Color_Match is None:
            Color_Match = TBG_Refiner_v1.COLOR_STABILIZER_METHOD
        target = cls._ensure_bhwc(image)
        if target is None:
            return (image,)
        reference = cls._resize_like(reference, target)
        if reference is None:
            return (target,)
        try:
            target, _ = apply_sift_drift_correction(reference, target, mode=Geometry_Drift_Correction)
            target = cls._ensure_bhwc(target).to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        except Exception:
            target = cls._ensure_bhwc(target)
        strength = float(Color_Match_Str if Color_Match_Str is not None else 1.0)
        color_match_key = str(Color_Match or "").strip().lower()
        if color_match_key in cls.RGB_LUMA_ALIASES:
            return (cls._global_rgb_luma_match(reference, target, strength=strength, mask=None),)
        if Color_Match is None or str(Color_Match).lower() == "none":
            return (target,)

        corrected = TBG_Image.colormatch(reference, target, Color_Match, strength)[0]
        corrected = cls._ensure_bhwc(corrected).to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        return (corrected,)


class TBG_Model_Agnostic_Color_Anchor():
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "ETUR-owned model-agnostic source latent mean preservation hook"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "How strongly to preserve the sampler input latent per-channel spatial mean. The sampler must receive the primary VAE-encoded source/input image latent. Do not use this with an Empty Latent if you expect source-image color preservation. 0 disables."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Linear sampler-step progress where the mean lock starts. 0.0 starts at the first sampler step."
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Linear sampler-step progress where the mean lock ends. 1.0 ends at the last sampler step."
                }),
                "ramp_curve": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.05,
                    "round": 0.01,
                    "tooltip": "Progress curve for applying the mean lock over sampler steps. Higher values make the effect arrive earlier."
                }),
            }
        }

    @staticmethod
    def _sigma_value(sigma):
        try:
            if torch.is_tensor(sigma):
                return float(sigma.flatten()[0].detach().cpu())
            return float(sigma)
        except Exception:
            return 0.0

    @staticmethod
    def _sampler_step_count(sigmas):
        try:
            if torch.is_tensor(sigmas):
                count = int(sigmas.numel())
            else:
                count = len(sigmas)
        except Exception:
            count = 0
        return max(1, count - 1 if count > 1 else count)

    @staticmethod
    def _capture_ref_means_from_latent(latent_image):
        if not torch.is_tensor(latent_image) or latent_image.ndim != 4 or int(latent_image.shape[1]) <= 0:
            return None
        return latent_image.detach().float().mean(dim=(-2, -1), keepdim=True).to(device="cpu")

    @staticmethod
    def _rf_state_from_model_chain(model):
        seen = set()
        current = model
        for _ in range(12):
            if current is None:
                return None
            ident = id(current)
            if ident in seen:
                return None
            seen.add(ident)
            state = getattr(current, "_untwisting_rope_rf_state", None)
            if isinstance(state, dict):
                return state
            current = getattr(current, "parent", None)
        return None

    @classmethod
    def patch(cls, model, strength=0.0, start_percent=0.0, end_percent=1.0, ramp_curve=1.5, debug=False):
        debug = False
        try:
            strength = max(0.0, min(1.0, float(strength or 0.0)))
        except Exception:
            strength = 0.0
        if strength <= 0.0:
            if debug:
                print("[TBG ModelAgnosticColorAnchor] disabled: strength=0.")
            return (model,)

        try:
            start_percent = max(0.0, min(1.0, float(start_percent or 0.0)))
        except Exception:
            start_percent = 0.0
        try:
            end_percent = max(0.0, min(1.0, float(end_percent if end_percent is not None else 1.0)))
        except Exception:
            end_percent = 1.0
        if end_percent < start_percent:
            start_percent, end_percent = end_percent, start_percent

        try:
            ramp_curve = max(0.1, float(ramp_curve or 1.5))
        except Exception:
            ramp_curve = 1.5

        state = {
            "ref_means": None,
            "ref_shape": None,
            "step": 0,
            "total_steps": 1,
            "first_sigma": None,
            "last_sigma_logged": None,
            "shape_skip_logged": False,
            "missing_logged": False,
            "rf_skip_logged": False,
            "window_skip_logged": False,
        }

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            state["ref_means"] = cls._capture_ref_means_from_latent(latent_image)
            state["ref_shape"] = tuple(latent_image.shape) if torch.is_tensor(latent_image) else None
            state["step"] = 0
            state["total_steps"] = cls._sampler_step_count(sigmas)
            state["first_sigma"] = None
            state["last_sigma_logged"] = None
            state["shape_skip_logged"] = False
            state["missing_logged"] = False
            state["rf_skip_logged"] = False
            state["window_skip_logged"] = False
            if debug:
                if state["ref_means"] is None:
                    print(f"[TBG ModelAgnosticColorAnchor] skipped: sampler latent_image shape={state['ref_shape']}.")
                else:
                    print(
                        f"[TBG ModelAgnosticColorAnchor] captured sampler latent_image "
                        f"shape={state['ref_shape']} strength={strength:.3f} "
                        f"start={start_percent:.3f} end={end_percent:.3f} "
                        f"ramp_curve={ramp_curve:.3f} total_steps={state['total_steps']}."
                    )
            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        def post_cfg_fn(args):
            denoised = args.get("denoised", None)
            if not torch.is_tensor(denoised):
                return denoised
            rf_state = cls._rf_state_from_model_chain(args.get("model", None))
            if isinstance(rf_state, dict) and (
                bool(rf_state.get("sigma_probe_active", False)) or not bool(rf_state.get("schedule_built", False))
            ):
                if debug and not state["rf_skip_logged"]:
                    print(
                        "[TBG ModelAgnosticColorAnchor] gated off during RF inversion/probe; "
                        "active only for final sampler steps."
                    )
                    state["rf_skip_logged"] = True
                return denoised
            ref_means = state["ref_means"]
            if ref_means is None:
                if debug and not state["missing_logged"]:
                    print("[TBG ModelAgnosticColorAnchor] skipped: sampler input latent was not available.")
                    state["missing_logged"] = True
                return denoised

            state["step"] += 1
            current_step = int(state["step"])
            total_steps = max(1, int(state.get("total_steps", 1) or 1))
            linear_progress = (current_step - 1) / max(total_steps - 1, 1)
            if linear_progress < start_percent or linear_progress > end_percent:
                if debug and not state["window_skip_logged"]:
                    print(
                        f"[TBG ModelAgnosticColorAnchor] timing gate skip: "
                        f"step={current_step}/{total_steps} linear={linear_progress:.3f} "
                        f"start={start_percent:.3f} end={end_percent:.3f}."
                    )
                    state["window_skip_logged"] = True
                return denoised

            if denoised.ndim != 4 or int(denoised.shape[1]) != int(ref_means.shape[1]):
                if debug and not state["shape_skip_logged"]:
                    print(
                        f"[TBG ModelAgnosticColorAnchor] skipped: "
                        f"denoised_shape={tuple(denoised.shape)} ref_shape={tuple(ref_means.shape)}."
                    )
                    state["shape_skip_logged"] = True
                return denoised

            sigma_value = cls._sigma_value(args.get("sigma", None))
            if state["first_sigma"] is None:
                state["first_sigma"] = max(abs(sigma_value), 1.0e-6)

            sigma_progress = 1.0 - max(0.0, min(1.0, abs(sigma_value) / max(float(state["first_sigma"]), 1.0e-6)))
            step_progress = 1.0 - 0.5 ** state["step"]
            progress = max(sigma_progress, step_progress)
            effective = strength * (progress ** (1.0 / ramp_curve))
            if effective < 1.0e-5:
                return denoised

            ref = ref_means.to(device=denoised.device, dtype=denoised.dtype)
            if int(ref.shape[0]) != int(denoised.shape[0]):
                ref = ref.mean(dim=0, keepdim=True).expand(int(denoised.shape[0]), -1, -1, -1)
            cur = denoised.mean(dim=(-2, -1), keepdim=True)
            correction = ref - cur
            corrected = denoised + correction * effective

            if debug and sigma_value != state["last_sigma_logged"]:
                state["last_sigma_logged"] = sigma_value
                mean_drift = (ref - cur).abs().mean().item()
                applied = (correction * effective).abs().mean().item()
                print(
                    f"[TBG ModelAgnosticColorAnchor] step={state['step']} "
                    f"linear={linear_progress:.3f} sigma={sigma_value:.4f} "
                    f"progress={progress:.3f} effective={effective:.3f} "
                    f"mean_drift={mean_drift:.5f} applied={applied:.5f}"
                )
            return corrected

        patched = model.clone()
        patched.model_options = patched.model_options.copy()
        patched.model_options["sampler_post_cfg_function"] = list(
            patched.model_options.get("sampler_post_cfg_function", [])
        )
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            sampler_sample_wrapper,
            patched.model_options,
            is_model_options=True,
        )
        patched.model_options["sampler_post_cfg_function"].append(post_cfg_fn)
        if debug:
            print(
                f"[TBG ModelAgnosticColorAnchor] active: strength={strength:.3f} "
                f"start={start_percent:.3f} end={end_percent:.3f} "
                f"ramp_curve={ramp_curve:.3f}; sampler input latent will be captured at runtime."
            )
        return (patched,)


class TBG_Model_Agnostic_Latent_Anchor():
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = "ETUR-owned model-agnostic source latent preservation hook"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "round": 0.01,
                    "tooltip": "How strongly to pull the denoised latent toward the sampler input latent. This uses the sampler's primary VAE-encoded source/input image latent as the anchor. Empty Latent is not a real source-image anchor. High values can preserve or ghost structure, texture, style, and color. 0 disables."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Linear sampler-step progress where the latent anchor starts. 0.0 starts at the first sampler step."
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Linear sampler-step progress where the latent anchor ends. 1.0 ends at the last sampler step."
                }),
                "ramp_curve": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Progress curve for applying the full latent anchor over sampler steps. Formula: progress^(1/curve). Higher values make the effect arrive earlier."
                }),
            }
        }

    @staticmethod
    def _sigma_value(sigma):
        return TBG_Model_Agnostic_Color_Anchor._sigma_value(sigma)

    @staticmethod
    def _sampler_step_count(sigmas):
        return TBG_Model_Agnostic_Color_Anchor._sampler_step_count(sigmas)

    @staticmethod
    def _rf_state_from_model_chain(model):
        return TBG_Model_Agnostic_Color_Anchor._rf_state_from_model_chain(model)

    @staticmethod
    def _capture_ref_latent(latent_image):
        if not torch.is_tensor(latent_image) or latent_image.ndim != 4 or int(latent_image.shape[1]) <= 0:
            return None
        ref = latent_image.detach()
        try:
            if torch.cuda.is_available() and not ref.is_cuda:
                ref = ref.to(device=torch.device("cuda"))
            else:
                ref = ref.to(device=ref.device)
        except Exception:
            pass
        return ref.clone()

    @staticmethod
    def _match_ref_latent_shape(reference_latent, denoised):
        ref = reference_latent.to(device=denoised.device, dtype=denoised.dtype)

        if int(ref.shape[0]) != int(denoised.shape[0]):
            ref = ref[:1].expand(int(denoised.shape[0]), -1, -1, -1)

        if ref.shape[2:] != denoised.shape[2:]:
            ref = F.interpolate(
                ref,
                size=denoised.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if int(ref.shape[1]) != int(denoised.shape[1]):
            if int(ref.shape[1]) > int(denoised.shape[1]):
                ref = ref[:, :int(denoised.shape[1])]
            else:
                ref = F.pad(ref, (0, 0, 0, 0, 0, int(denoised.shape[1]) - int(ref.shape[1])))

        return ref

    @classmethod
    def patch(cls, model, strength=0.5, start_percent=0.0, end_percent=1.0, ramp_curve=1.5, debug=False):
        debug = False
        try:
            strength = max(0.0, min(1.0, float(strength or 0.0)))
        except Exception:
            strength = 0.0
        if strength <= 0.0:
            if debug:
                print("[TBG ModelAgnosticLatentAnchor] disabled: strength=0.")
            return (model,)

        try:
            start_percent = max(0.0, min(1.0, float(start_percent or 0.0)))
        except Exception:
            start_percent = 0.0
        try:
            end_percent = max(0.0, min(1.0, float(end_percent if end_percent is not None else 1.0)))
        except Exception:
            end_percent = 1.0
        if end_percent < start_percent:
            start_percent, end_percent = end_percent, start_percent

        try:
            ramp_curve = max(0.5, float(ramp_curve or 1.5))
        except Exception:
            ramp_curve = 1.5

        state = {
            "ref_latent": None,
            "ref_shape": None,
            "step": 0,
            "total_steps": 1,
            "first_sigma": None,
            "last_sigma_logged": None,
            "shape_skip_logged": False,
            "missing_logged": False,
            "rf_skip_logged": False,
            "window_skip_logged": False,
        }

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            state["ref_latent"] = cls._capture_ref_latent(latent_image)
            state["ref_shape"] = tuple(latent_image.shape) if torch.is_tensor(latent_image) else None
            state["step"] = 0
            state["total_steps"] = cls._sampler_step_count(sigmas)
            state["first_sigma"] = None
            state["last_sigma_logged"] = None
            state["shape_skip_logged"] = False
            state["missing_logged"] = False
            state["rf_skip_logged"] = False
            state["window_skip_logged"] = False
            if debug:
                if state["ref_latent"] is None:
                    print(f"[TBG ModelAgnosticLatentAnchor] skipped: sampler latent_image shape={state['ref_shape']}.")
                else:
                    print(
                        f"[TBG ModelAgnosticLatentAnchor] captured sampler latent_image "
                        f"shape={state['ref_shape']} stored_device={state['ref_latent'].device} "
                        f"strength={strength:.3f} start={start_percent:.3f} end={end_percent:.3f} "
                        f"ramp_curve={ramp_curve:.3f} total_steps={state['total_steps']}. "
                        "This is a real source-image anchor only when the sampler latent came from the primary VAE-encoded input image, not Empty Latent."
                    )
            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        def post_cfg_fn(args):
            denoised = args.get("denoised", None)
            if not torch.is_tensor(denoised):
                return denoised
            rf_state = cls._rf_state_from_model_chain(args.get("model", None))
            if isinstance(rf_state, dict) and (
                bool(rf_state.get("sigma_probe_active", False)) or not bool(rf_state.get("schedule_built", False))
            ):
                if debug and not state["rf_skip_logged"]:
                    print(
                        "[TBG ModelAgnosticLatentAnchor] gated off during RF inversion/probe; "
                        "active only for final sampler steps."
                    )
                    state["rf_skip_logged"] = True
                return denoised

            ref_latent = state["ref_latent"]
            if ref_latent is None:
                if debug and not state["missing_logged"]:
                    print("[TBG ModelAgnosticLatentAnchor] skipped: sampler input latent was not available.")
                    state["missing_logged"] = True
                return denoised

            state["step"] += 1
            current_step = int(state["step"])
            total_steps = max(1, int(state.get("total_steps", 1) or 1))
            linear_progress = (current_step - 1) / max(total_steps - 1, 1)
            if linear_progress < start_percent or linear_progress > end_percent:
                if debug and not state["window_skip_logged"]:
                    print(
                        f"[TBG ModelAgnosticLatentAnchor] timing gate skip: "
                        f"step={current_step}/{total_steps} linear={linear_progress:.3f} "
                        f"start={start_percent:.3f} end={end_percent:.3f}."
                    )
                    state["window_skip_logged"] = True
                return denoised

            if denoised.ndim != 4 or ref_latent.ndim != 4:
                if debug and not state["shape_skip_logged"]:
                    print(
                        f"[TBG ModelAgnosticLatentAnchor] skipped: "
                        f"denoised_shape={tuple(denoised.shape)} ref_shape={tuple(ref_latent.shape)}."
                    )
                    state["shape_skip_logged"] = True
                return denoised

            sigma_value = cls._sigma_value(args.get("sigma", None))
            if state["first_sigma"] is None:
                state["first_sigma"] = max(abs(sigma_value), 1.0e-6)

            sigma_progress = 1.0 - max(0.0, min(1.0, abs(sigma_value) / max(float(state["first_sigma"]), 1.0e-6)))
            step_progress = 1.0 - 0.5 ** state["step"]
            progress = max(sigma_progress, step_progress)
            effective = strength * (progress ** (1.0 / ramp_curve))
            if effective < 1.0e-5:
                return denoised

            ref = cls._match_ref_latent_shape(ref_latent, denoised)
            correction = ref - denoised
            corrected = denoised + correction * effective

            if debug and sigma_value != state["last_sigma_logged"]:
                state["last_sigma_logged"] = sigma_value
                mean_delta = correction.abs().mean().item()
                applied = (correction * effective).abs().mean().item()
                print(
                    f"[TBG ModelAgnosticLatentAnchor] step={state['step']} "
                    f"linear={linear_progress:.3f} sigma={sigma_value:.4f} "
                    f"progress={progress:.3f} effective={effective:.3f} "
                    f"mean_delta={mean_delta:.5f} applied={applied:.5f}"
                )
            return corrected

        patched = model.clone()
        patched.model_options = patched.model_options.copy()
        patched.model_options["sampler_post_cfg_function"] = list(
            patched.model_options.get("sampler_post_cfg_function", [])
        )
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            sampler_sample_wrapper,
            patched.model_options,
            is_model_options=True,
        )
        patched.model_options["sampler_post_cfg_function"].append(post_cfg_fn)
        if debug:
            print(
                f"[TBG ModelAgnosticLatentAnchor] active: strength={strength:.3f} "
                f"start={start_percent:.3f} end={end_percent:.3f} "
                f"ramp_curve={ramp_curve:.3f}; sampler input latent will be captured at runtime."
            )
        return (patched,)


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
                "PID_Model": ("MODEL", {
                    "label": "PID Diffusion Model",
                    "tooltip": "PiD models are auto-loaded from Hugging Face when available. This optional input lets you override that auto-selected PiD model with your own PiD model compatible with the main model and VAE used by the refiner."
                }),
                "ColorMatch_Debug": ("tbg_colormatch_debug", {"label": "ColorMatch Debug Gates", "tooltip": "Optional developer pipe for isolating individual ETUR ColorMatch stages."}),
                "Custom_Sigmas_!DENOISE=1": ("SIGMAS", {"label": "Sigmas with denoise 1", "tooltip": "Insert your full custom sigma noise curve (not denoised), as denoising is performed per tile by the node."}),
                "Sampler": ("SAMPLER", {
                    "label": "Overwrites the Sampler Name field",
                    "tooltip": "Plug in a custom sampler or ClownSampler here to override the sampler selected in the refiner node."
                }),
                "Ideogram4_Guider": ("GUIDER", {"label": "Ideogram4 Guider"}),
                "cropped_positive": ("CONDITIONING", {
                    "tooltip": "Optional full-image positive conditioning for crop-aware tile sampling. Provide conditioning for the full reference/source image; ETUR crops the matching region for each tile and appends it to the tile's normal positive prompt/reference conditioning. It does not overwrite the normal conditioning. If you do not want to use the ETUR ControlNet pipe, you can use cropped conditioning in a similar way to Ultimate SD Upscale, including conditioning that already contains ReferenceLatent inputs."
                }),
                "cropped_negative": ("CONDITIONING", {
                    "tooltip": "Optional full-image negative conditioning for crop-aware tile sampling. Provide conditioning for the full reference/source image; ETUR crops the matching region for each tile and appends it to the tile's normal negative conditioning. It does not overwrite the normal conditioning. If you do not want to use the ETUR ControlNet pipe, you can use cropped conditioning in a similar way to Ultimate SD Upscale, including conditioning that already contains ReferenceLatent inputs."
                }),
                "Segment_Background_Harmonization": ("BOOLEAN", {"label": "Segment Background Harmonization", "default": True,
                                         "label_on": "Protect Object / Match Background", "label_off": "Disabled",
                                         "tooltip": "Low-frequency color/tint correction for generated segment surroundings. Protects the object/core mask while matching the segment background to the finished tile-only canvas."}),

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
                "Flux2_Tile_Color_Correction": ("BOOLEAN", {"label": "Tile Color Correction", "default": True,
                                         "tooltip": "Enable the selected Color Match correction for all refiner model types. If this Labs node is not connected, the refiner keeps this enabled by default."}),
                "Flux2_Sampler_Hook": ("BOOLEAN", {"label": "Flux2 Sampler Hook", "default": False,
                                         "label_on": "ON", "label_off": "OFF",
                                         "tooltip": "OFF: use normal VAE encode plus ComfyUI Set Latent Noise Mask. ON: use the legacy/private Flux2 differential sampler hook."}),
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
        if kwargs.get("stitch_blending") not in self.STICHTYPE:
            shifted_max = kwargs.get("stitch_blending", None)
            kwargs["stitch_blending"] = "gpupyramid"
            if "max_upscale_size_segment" not in kwargs and shifted_max is not None:
                kwargs["max_upscale_size_segment"] = shifted_max
        kwargs.setdefault("Segment_Background_Harmonization", True)
        kwargs.setdefault("Flux2_Sampler_Hook", False)
        if "ColorMatch_Debug" in kwargs and kwargs["ColorMatch_Debug"] is not None:
            kwargs["ColorMatch_Debug"] = dict(kwargs["ColorMatch_Debug"])
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
    DEV_DEBUG_MODES = [
        'dev debug on',
        'dev debug off',
        'dev free user test',
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
                "Dev_Debug_Mode": (
                    cls.DEV_DEBUG_MODES,
                    {
                        "label": "Dev Debug Mode",
                        "default": "dev debug on",
                        "tooltip": (
                            "Developer-only local runtime switch. Non-dev accounts ignore this. "
                            "'dev debug off' suppresses dev debug/temp saves. "
                            "'dev free user test' runs this job with Free-user limits without changing the account."
                        ),
                    },
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
                "PID_Model": ("MODEL", {"label": "PID Diffusion Model"}),
                "PID_VAE_Compatible_Model": (
                    PID_VAE_COMPATIBLE_MODEL_TYPES,
                    {"label": "PiD VAE Compatible Model", "default": "FLUX1"},
                ),
                "PID_CLIP": ("CLIP", {"label": "PID CLIP"}),
                "PID_Source_VAE": ("VAE", {"label": "PID Source VAE"}),
                "PID_degrade_sigma": ("FLOAT", {"label": "PID degrade_sigma", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "PRO_Tile_Fusion_shift_in_out": ("INT",{"label": "PRO_Tile_Fusion_shift_in_out", "default": 0, "min": -128, "max": 128,"step": 8}),
                "PRO_Tile_Fusion_shift_top_left": ("INT",{"label": "PRO_Tile_Fusion_shift_top_left", "default": 0, "min": -128, "max": 128,"step": 8}),
            }
        }

    RETURN_TYPES = ("labs_upscaler",)

    @classmethod
    def fn(self, **kwargs):
        return (kwargs,)





