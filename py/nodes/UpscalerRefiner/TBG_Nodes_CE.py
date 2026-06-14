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



class TBG_ETUR_Upscaler_and_Tile_Generator_CE():

    VLM = [
        "NONE",
        "SkyCaptioner-V1",
        "SkyCaptioner-V1_8bit",
        "SkyCaptioner-V1_4bit",
        "Qwen3-VL-4B-Instruct-FP8",
        "Qwen3-VL-4B-Thinking-FP8",
        "Qwen3-VL-8B-Instruct-FP8",
        "Qwen3-VL-8B-Thinking-FP8",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "GGUF\\Qwen3-VL-4B-Instruct-F16",
        "GGUF\\Qwen3-VL-4B-Instruct-Q4_K_M",
        "GGUF\\Qwen3-VL-4B-Instruct-Q8_0",
        "GGUF\\Qwen3-VL-8B-Instruct-F16",
        "GGUF\\Qwen3-VL-8B-Instruct-Q4_K_M",
        "GGUF\\Qwen3-VL-8B-Instruct-Q8_0",
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    ]
    VLM_Quantization = "None (FP16)"

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
                        "SuperResolution/Tiled-SeedVR2 Standard",
                        "SuperResolution/FlashVSR-v1.1 Small 8GB",
                        "Waifu/art",
                        "Waifu/photo"] + upscale_models  # or upscale_models + ["None"]
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },

            "required": {
                "image": ("IMAGE", {"label": "Image"}),
                "tile_size": ("INT",{"label": "Tile Size height", "default": 1024, "min": 320, "max": 8192, "step": 64}),
                "upscale_model": (self.upscale_models, {"label": "Upscale Model","default":"NONE"}),
                "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 8, "step": 0.05, "round": 0.01}),
                "Optimize_Upscale_Factor_For_Tile_Use": ("BOOLEAN", {
                    "label": "Optimize upscale factor for optimal tile use",
                    "default": False,
                    "tooltip": "Keeps the tile grid from the requested upscale factor, then lowers the effective scale to the largest size that uses only the required tile overlap."
                }),
                "VLM_Model": (
                    self.VLM,
                    {
                        "label": "VLM_Model",
                        "default": "NONE",
                        "tooltip": (
                            "Check the license for all models. "
                            "Apple’s model is for research use only and the files are not included in this custom node. "
                            "It must be installed separately by the user."
                        )
                    }
                ),
                "VLM_Prompt": ("STRING", {"multiline": True, "label": "LLMPrompt Prompt",
                                                "default": "Provide a highly detailed description of the image, emphasizing materials and textures. Enhance every visual detail, including accurate colors, lighting, and stylistic elements. Also describe the artistic or photographic style, such as film type, camera style, era, or overall aesthetic."}),
                "VLM_seed": ("INT", {"label": "Seed", "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,"fixed": True  }),

                "Fusion Reference Margin": ("INT",{"label": "Reference margin","default": 0,"min": 0, "max": 256, "step": 8,
                                                     "tooltip": "Reference border content kept around each Soft Merge tile and cropped out when stitching the image back together."}),
                "Feather Mask": ("INT",
                                          {"tooltip": "Feather mask is a gradient used only in final image compositing to smoothly blend tiles together using Soft Merge.", "default": 16, "min": 0,
                                           "max": 128, "step": 8}),

            },
            "optional": {
                "Segment_Mask": ("MASK",),
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
        kwargs["tile_size_w"] = kwargs.get("tile_size")
        kwargs["tile_size_h"] = kwargs.get("tile_size")
        kwargs["Fusion Reference Margin"] = kwargs.get("Fusion Reference Margin", kwargs.get("Reference Margin", 0))
        TBG_Upscaler_v1i = TBG_Upscaler_v1()
        result =  TBG_Upscaler_v1i.fn(**kwargs)
        _,_,_,_,_,_,_,_,userinfo,_,_,infos= result[0]

        return {
            "ui": {"value": [f"{userinfo}", infos]},
            "result": result
        }

class TBG_ETUR_Refiner_CE():

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



    COLOR_MATCH_METHODS = [
        'none',
        TBG_Refiner_v1.TILE_STABILIZER_METHOD,
        'lab full color match',
        'wavelet',
        'reinhard_lab_gpu',
        'hm-mkl-hm',
    ]

    VAE_ENCODE_TOOLTIP = (
        "tiled slow is the standard ComfyUI tiled VAE path. "
        "tbg Color-preserving fast is the faster TBG VAE path that preserves colors. "
        "Nvidia PiD 4x changes VAE decoding to the PiD model; 1024x1024 uses the native fast path, while other tile or segment sizes use tiled PiD latent decode. It works with FLUX1, FLUX2, Qwen Image, Qwen Image Edit, SDXL, SD3, and Z-Image."
    )


    @classmethod
    def INPUT_TYPES(self):
        # def INPUT_TYPES(cls):

        return {

            "optional": {
                "Controlnet_Pipe": ("Controlnet_Pipe", {"label": "TBG ControlNet Pipe"}),
                "Enrichment_Pipe": ("Enrichment_Pipe", {"label": "TBG enrichment Pipe"}),
                "Redux_Style_Model": ("STYLE_MODEL", {"label": "Redux_Style_Model"}),
                "Redux_Clip_Vision": ("CLIP_VISION", {"label": "Redux_Clip_Vision"}),
            },
            "required": {
                "model_type": (self.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),

                "model": ("MODEL", {"label": "Model"}),
                "clip": ("CLIP", {"label": "Clip"}),
                "vae": ("VAE", {"label": "VAE"}),
                "cfg": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),


                "TBG_Pipe": ("TBG_Pipe", {"label": "TBG Pipe"}),


                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "Flux_Guidance": ("FLOAT",{"label": "Flux Guidance for Tiles", "default": 3.5, "min": -100.0, "max": 100.0,"step": 0.1, "round": 0.01,  "tooltip": "All Fusion Modes benefit from high Guidance, so if you notice that certain areas aren't blending well, try increasing the Guidance value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),


                "vae_encode": ("BOOLEAN", {"label": "VAE Encode type", "default": True, "label_on": "tiled slow","label_off": "tbg Color-preserving fast",  "tooltip": self.VAE_ENCODE_TOOLTIP}),
                "tile_size_vae": ("INT",{"label": "Tile Size (VAE)", "default": 1024, "min": 256, "max": 4096, "step": 64}),
                "General_Prompt_Positive": ("STRING", {"tooltip": "General_Prompt_Positive", "multiline": True, "label": "General Positive Prompt for all Tiles", "default": ""}),
                "General_Prompt_Negative": ("STRING",  {"tooltip": "General_Prompt_Negative", "multiline": True, "label": "General Negative Prompt for all Tiles",
                                                        "default": "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理"}),

                "denoise": ("FLOAT", {"label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
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
                "Redux_strength": ("FLOAT", {"display": "slider", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),

                "Fast_1_Tile_Preview": ("BOOLEAN", {"label": "Fast_1_Tile_Preview", "default": False, "label_on": "Preview Single Tile", "label_off": "Disabled",
                                                    "tooltip": "The first Selected_Tiles_By_Number are processed at full scale as a preview, allowing a quick check of settings before processing the entire set."}),
                "Selected_Tiles_Only": ("BOOLEAN", {"label": "Process_selected_Tiles_only", "default": False, "label_on": "Generate Selected Tiles Only", "label_off": "Disabled"}),
                "Selected_Tiles_By_Numbers": ("STRING", {"label": "Selected_Tiles_Index_Numbers to process", "default": '',
                                                         "tooltip": "You can set a list of selected tiles to process like 1,2,3,6 and activate Selected_Tiles_Only"}),
            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },

        }

    RETURN_TYPES = (
        "IMAGE",
    )

    RETURN_NAMES = (
        "Refined",
    )

    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)

    OUTPUT_NODE = True
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        kwargs["VRAM_Profile"] = "Ultra Low Memory (Per-Tile Streaming)"
        return {
            "ui": {"value": [f"{kwargs.get('seed', None)}"]},
            "result": (TBG_Refiner_v1.fn(**kwargs))
        }

