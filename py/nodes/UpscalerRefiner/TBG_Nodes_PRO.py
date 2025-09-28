"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________

"""
import math
import os


import comfy
import comfy.latent_formats
import comfy.model_sampling
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models
import cv2
import folder_paths
import numpy as np
import torch
from ..UpscalerRefiner.TBG_Refiner import TBG_Refiner_v1
from ..UpscalerRefiner.TBG_Tiler import TBG_Upscaler_v1
from ...vendor.ComfyUI_Impact_Pack.masktoseg import MaskToSEGS
from ....TBG_presets import PRESETS_PRO, get_presets


class TBG_Upscaler_v1_pro():
    NAME = "TBG Enhanced Tiled Upscaler FLUX PRO"

    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    SEGMENTS = {}
    SIZE = {}
    LLM = {}

    PRESETS = PRESETS_PRO

    DIFFUSION_MODES = [
        'Soft Merge',
        'Tile_Fusion',
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
        "nearest-exact"
    ]

    from nodes import NODE_CLASS_MAPPINGS

    node_name = "FastVLM7BNode"

    if node_name in NODE_CLASS_MAPPINGS:
        NodeAClass = NODE_CLASS_MAPPINGS[node_name]
        LLM = [
            "NONE",
            "Janus-Pro-1B",
            "Janus-Pro-7B",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "SkyCaptioner-V1",
            "Qwen2.5-VL-3B-Instruct_8bit",
            "Qwen2.5-VL-7B-Instruct_8bit",
            "SkyCaptioner-V1_8bit",
            "Qwen2.5-VL-3B-Instruct_4bit",
            "Qwen2.5-VL-7B-Instruct_4bit",
            "SkyCaptioner-V1_4bit",
            "Apple FastVLM 7B Research use only",
        ]
    else:
        LLM = [
            "NONE",
            "Janus-Pro-1B",
            "Janus-Pro-7B",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "SkyCaptioner-V1",
            "Qwen2.5-VL-3B-Instruct_8bit",
            "Qwen2.5-VL-7B-Instruct_8bit",
            "SkyCaptioner-V1_8bit",
            "Qwen2.5-VL-3B-Instruct_4bit",
            "Qwen2.5-VL-7B-Instruct_4bit",
            "SkyCaptioner-V1_4bit"
        ]




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
                "Fragmentation":("FLOAT",{"label": "inpaint_max", "default": 1, "min": 0.5, "max": 4, "step": 0.01, "round": 0.01}),
                "tile_size_w": ("INT",{"label": "Tile Size height", "default": 1024, "min": 320, "max": 8192, "step": 16}),
                "tile_size_h": ("INT",{"label": "Tile Size width", "default": 1024, "min": 320, "max": 8192, "step": 16}),
                "max_upscale_size_segment": ("INT", {"label": "max_upscale_size_segment","default": 2048, "min": 256, "max": 4096, "step": 8}),
                "upscaler": (self.UPSCALE_TYPE, {"label": "Upscale Type", "default": "NONE"}),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"), {"label": "Upscale Model"}),
                "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "round": 0.01}),
                "upscaler_method": (self.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'lanczos'}),
                "LLMPrompt": (
                    self.LLM,
                    {
                        "label": "Upscale Type",
                        "default": "NONE",
                        "tooltip": (
                            "Check the license for all models. "
                            "Apple’s model is for research use only and the files are not included in this custom node. "
                            "It must be installed separately by the user."
                        )
                    }
                ),
                "LLMPrompt_Prompt": ("STRING", {"multiline": True, "label": "LLMPrompt Prompt",
                                                "default": "Provide a highly detailed description of the image, emphasizing materials and textures. Enhance every visual detail, including accurate colors, lighting, and stylistic elements. Also describe the artistic or photographic style, such as film type, camera style, era, or overall aesthetic."}),

                "compositing_mask_blur": ("INT", {"label": "Manual Feather Mask for Tile Overlapping", "default": 16,"min": 0, "max": 128, "step": 8}),
                "PRO_activate": ("BOOLEAN", {"label": "api_activate_pro", "default": True, "label_on": "ETUR PRO","label_off": "ETUR"}),
                "PRO_Tile_Fusion_Mode": (self.DIFFUSION_MODES, {"label": "PRO_Tile_Fusion_Mode", "default": "Neuro_Generative_Tile_Fusion"}),
                "PRO_Tile_Fusion_blur_margin": ("INT",{"label": "PRO_Tile_Fusion_blur_margin","default": 48,"min": 0, "max": 128, "step": 8}),
                "PRO_Tile_Fusion_shift_in_out": ("INT",{"label": "PRO_Tile_Fusion_shift_in_out", "default": 0, "min": -128, "max": 128,"step": 8}),
                "PRO_Tile_Fusion_shift_top_left": ("INT",{"label": "PRO_Tile_Fusion_shift_top_left", "default": 0, "min": -128, "max": 128,"step": 8}),
                "PRO_Tile_Fusion_border_margin": ("INT", {"label": "shift_mask", "default": 16, "min": 0, "max": 128,"step": 8}),
                #"PRO_Fusion_Space_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 1, "min": 0, "max": 2, "step": 0.01, "round": 0.01}),
            },
            "optional": {
                "PRO_segs": ("SEGS",),
                "PRO_api_token": ("STRING", {"default": ""}),

            }

        }

    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    RETURN_TYPES = (
        "IMAGE",
        "TBG_Pipe",
        "Tile_Prompt_Pipe",  # self.OUTPUTS.grid_prompts ,output_tiles
        "STRING",
        "IMAGE"

    )

    RETURN_NAMES = (
        "Mask Overlay Preview",
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "Info",
        "Upscaled Image"
    )

    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
    )

    OUTPUT_NODE = True
    CATEGORY = "TBG/Enhanced Upscaler PRO"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'Upscaler and Tiler to split you images into tiles for TBG ETUR'
    FUNCTION = "fn"


    @classmethod
    def fn(self, **kwargs):


        kwargs["tile_size"] = kwargs.get("tile_size_h")
        if kwargs["Fragmentation"] and  kwargs["Fragmentation"] != 0:
            kwargs["tile_size_w"] = int(kwargs.get("tile_size_w", 1024)*kwargs["Fragmentation"])
            kwargs["tile_size"] = int(kwargs.get("tile_size_h", 1024)*kwargs["Fragmentation"])

        if kwargs["presets"] == 'Add Yours → TBG_presets.py':
            kwargs["presets"] = 'None'

        if kwargs["PRO_Tile_Fusion_Mode"] != 'NONE' or kwargs["PRO_Tile_Fusion_Mode"] != 'Soft Merge':
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
        else:
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = False

        if kwargs["PRO_Tile_Fusion_Mode"] == 'Soft Merge': # soft merge is only compositing mask blur so we set this to none
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = False
            kwargs["PRO_Tile_Fusion_Mode"] = 'NONE'
            kwargs["PRO_Tile_Fusion_blur_margin"] = 0
            kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = 0
            kwargs["PRO_Tile_Fusion_border_margin"] = 0

        if kwargs["PRO_Tile_Fusion_Mode"] != 'NONE' or kwargs["PRO_Tile_Fusion_Mode"] != 'Soft Merge':
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
        else:
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = False

        if kwargs["PRO_Tile_Fusion_Mode"] == 'NGTF_FLUX_Kontext':
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
            #FluxKontextImageScale()
            #aspect_ratio = kwargs["tile_size_w"] / kwargs["tile_size"]
            # _, kwargs["tile_size_w"], kwargs["tile_size"] = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)

        min_tile_size = min(kwargs["tile_size"], kwargs["tile_size_w"])
        kwargs = get_presets(min_tile_size, **kwargs)
        # Read an environment variable

        result =  TBG_Upscaler_v1.fn(**kwargs)
        current_credits = 0
        _, _, _, _, _,current_credits = result
        return {
            "ui": {"value": [f"{current_credits}"]},
            "result": result
        }

class TBG_Refiner_v1_pro():
    SIZE = None
    SEGMENTS = None
    OUTPUTS = None
    KSAMPLER = None
    INPUTS = None
    PARAMS = None
    NAME = "TBG Enhanced Refiner FLUX PRO"

    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX1 Kontext': 1024,
        'Qwen Image': 1328,
        'Qwen Image Edit': 1328,
        'Others': 1024,
    }


    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())

    DENOISE_METHODS = [
        'default',
        'normalized',
        'normalized advanced',
        'multiplied',
        'multiplied normalized',
        'default short ',
    ]

    COLOR_MATCH_METHODS = [
        'none',
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

    CACHE = [
        'OFF',
        'use Cached Tiles as Input',
        'use Cached Tiles only for Fusion',
    ]



    @classmethod
    def INPUT_TYPES(self):
        # def INPUT_TYPES(cls):

        return {

            "optional": {
                "Controlnet_Pipe": ("Controlnet_Pipe", {"label": "TBG ControlNet Pipe"}),
                "Tile_Prompt_Pipe": ("TILE_Prompt_PIPE_OUT", {"label": "Tile Prompt Pipe"}),
                "Enrichment_Pipe": ("Enrichment_Pipe", {"label": "TBG enrichment Pipe"}),
                "Custom_Sigmas_!DENOISE=1": ("SIGMAS", {"label": "Sigmas with denoise 1","tooltip": "Insert your full custom sigma noise curve (not denoised), as denoising is performed per tile by the node."}),
                "Sampler": ("SAMPLER",{"label": "Overwrites the Sampler Name field"}),
                "Resume_Tiled_Refinement_Image": ("IMAGE", {"label": "Presampled_Background_Image"}),
                "cropped_positive": ("CONDITIONING",),
                "cropped_negative": ("CONDITIONING",),
                "denoise_mask": ("MASK",),
                "Redux_Style_Model": ("STYLE_MODEL", {"label": "Redux_Style_Model"}),
                "Redux_Clip_Vision": ("CLIP_VISION", {"label": "Redux_Clip_Vision"}),
                "Redux_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "round": 0.001}),

            },
            "required": {
                "model": ("MODEL", {"label": "Model"}),
                "clip": ("CLIP", {"label": "Clip"}),
                "vae": ("VAE", {"label": "VAE"}),
                "TBG_Pipe": ("TBG_Pipe", {"label": "TBG Pipe"}),

                "model_type": (self.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),
                "Tile_Fusion_Mode": (self.DIFFUSION_MODES,{"label": "Tile_Fusion_Mode", "default": "From TGB Tiler Node"}),
                "Tile_Fusion_Blend": ("FLOAT",{"label": "Tile_Fusion_Blend", "default": 0.5, "min": 0, "max": 1,"step": 0.01,"round": 0.01}),
                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "Flux_Guidance": ("FLOAT",{"label": "Flux Guidance for Tiles", "default": 3.5, "min": -100.0, "max": 100.0,"step": 0.1, "round": 0.01,  "tooltip": "All Fusion Modes benefit from high Guidance, so if you notice that certain areas aren't blending well, try increasing the Guidance value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),
                "denoise": ("FLOAT", {"label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_method": (self.DENOISE_METHODS, {"label": "DENOISE_METHODS", "default": 'default'}),

                "vae_encode": ("BOOLEAN", {"label": "VAE Encode type", "default": True, "label_on": "tiled","label_off": "standard",  "tooltip": ""}),
                "tile_size_vae": ("INT",{"label": "Tile Size (VAE)", "default": 1024, "min": 256, "max": 4096, "step": 64}),
                "General_Prompt_Positive": ("STRING", {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "General_Prompt_Negative": ("STRING",  {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "Save_Tiles_in_Temp_Folder": ("BOOLEAN", {"label": "Preview_Tiles_in_Temp_Folder", "default": False,"label_on": "Save Tiles to /temp/TBG/","label_off": "Disabled"}),
                "Fast_1_Tile_Preview": ("BOOLEAN", {"label": "Fast_1_Tile_Preview", "default": False, "label_on": "Preview Single Tile", "label_off": "Disabled", "tooltip": "The first Selected_Tiles_By_Number are processed at full scale as a preview, allowing a quick check of settings before processing the entire set."}),
                "Selected_Tiles_Only": ("BOOLEAN", {"label": "Process_selected_Tiles_only", "default": False, "label_on": "Generate Selected Tiles Only", "label_off": "Disabled"}),
                "Selected_Tiles_By_Numbers": ("STRING", {"label": "Selected_Tiles_Index_Numbers to process", "default": '',
                                                         "tooltip": "You can set a list of selected tiles to process like 1,2,3,6 and activate Selected_Tiles_Only"}),

                "Color_Match": (self.COLOR_MATCH_METHODS, {"label": "Color Match Method", "default": 'none'}),
                "Controlnet_Pipe_strength": ("FLOAT",{"label": "Controlnet_Pipe_strength", "default": 1.00, "min": 0, "max": 100, "step": 0.01,"round": 0.01, "tooltip": "It's a multiplier value applied uniformly to all ControlNets from CnetPipe, scaling their combined influence."}),
                # PRO_Fusion_Space_Denoise max could be 2 but this could produce seams so i crop it out

                "PRO_Tile_Cache": (self.CACHE, {"label": "Tile_Cache", "default": "OFF",
                                                "tooltip": "Cached the generated tiles to enable generative tile fusion to insert or correct individual images after the final render is complete."}),
                "PRO_Resume_Tiled_Refinement": ("BOOLEAN", {"label": "PRO_Resume_Tiled_Refinement", "default": False, "label_on": "Resume", "label_off": "Disabled",
                                                            "tooltip": "Provide the original input image along with the completed tiled upscaling result to resume the refinement process. This allows continuing enhancement based on the initial image and the existing finished output."}),

                "PRO_Fusion_Space_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 2, "step": 0.01, "round": 0.01,
                                                       "tooltip": "0=Off, Controls the Fusion mask. Values between 0 and 1 behave like denoising, while values from 1 to 2 progressively reduce the fusion edge effect, with 2 disabling fusion entirely"}),

                "PRO_Fusion_Complexity_Mask_Strength": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                       "tooltip": "Generates a map based on image complexity to apply adaptive denoising. For example, areas with uniform color backgrounds receive lower denoise values, while more detailed or crowded regions can receive higher values, and vice versa."}),

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
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
    )

    RETURN_NAMES = (
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "Refined Image",
        "Refined Image without Segments",
        "Refined Image without ColorCorrection",
        "Original Image",
        "Image Tiles",
        "Tile Overlay Grid",
        "Prompt List",
    )

    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)

    OUTPUT_NODE = True
    CATEGORY = "TBG/Enhanced Upscaler PRO"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'Sampler and Refiner for TBG ETUR'
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        return {
            "ui": {"value": [f"{kwargs.get('seed', None)}"]},
            "result": (TBG_Refiner_v1.fn(**kwargs))
        }

        #return TBG_Refiner_v1.fn(**kwargs)

class EdgePadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # BHWC tensor
                "top": ("INT", {"default": 10, "min": 0}),
                "bottom": ("INT", {"default": 10, "min": 0}),
                "left": ("INT", {"default": 10, "min": 0}),
                "right": ("INT", {"default": 10, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad_image"

    def pad_image(self, image, top, bottom, left, right):
        image_tensor = image  # shape: [B, H, W, C]

        batch = []
        for i in range(image_tensor.shape[0]):
            img = image_tensor[i].cpu().numpy()  # HWC
            img = (img * 255).astype(np.uint8)

            padded = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                borderType=cv2.BORDER_REPLICATE
            )

            padded = padded.astype(np.float32) / 255.0
            batch.append(torch.from_numpy(padded))

        padded_batch = torch.stack(batch)  # shape: [B, H+pad, W+pad, C]
        return (padded_batch,)




class TBG_masked_attention:
    NAME = "TBG Masked Attention"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",),
                         "border_margin": ("FLOAT", {"label": "border_margin", "default": 1.5, "min": 1, "max": 5, "step": 0.1, "round": 0.1}),
                         },


        }

    CATEGORY = "TBG/Enhanced Upscaler PRO"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'Converts Mask to Tiles for TBG ETUR'

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "load_image"
    def load_image(self, mask, border_margin):
        combined = False
        crop_factor = border_margin
        bbox_fill = False
        drop_size = 8
        contour_fill = False
        SEG = MaskToSEGS(mask, combined, crop_factor, bbox_fill, drop_size, contour_fill)
        return (SEG,)





NODE_CLASS_MAPPINGS = {
    "EdgePadNode": EdgePadNode,
    "TBG_masked_attention": TBG_masked_attention,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgePadNode": "Pad Image with Border Pixels",
    "TBG_masked_attention": "TBG_masked_attention",

}



