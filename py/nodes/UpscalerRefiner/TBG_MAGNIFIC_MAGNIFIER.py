"""
TBG_magnific_ETUR: Enhanced Tiled Upscaler and Refiner (FLUX PRO)
"""
import copy
import os
import nodes
import comfy
import comfy.latent_formats
import comfy.model_sampling
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models

from .inc.api import PatreonAuthNative
from ..UpscalerRefiner.TBG_Refiner import TBG_Refiner_v1
from ..UpscalerRefiner.TBG_Tiler import TBG_Upscaler_v1
from ...vendor.ComfyUI_MaraScott_Nodes.py.utils.constants import get_category
from ....TBG_presets import PRESETS_PRO, get_presets




class TBG_magnific_ETUR ():
    NAME = "TBG Magnific Magnifier PRO"

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
    LLM = [
        "NONE",
        "Janus-Pro-1B",
        "Janus-Pro-7B"
    ]

    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX1 Kontext': 1024,
        'HiDream in next version4': 1024,
        'SD1 not tested': 512,
        'SDXL': 1024,
        'SD3 not tested': 1024,
        'SVD not tested': 1024,
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
        return {
            "hidden": {
                "id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },

            "required": {
                "model": ("MODEL", {"label": "Model"}),
                "clip": ("CLIP", {"label": "Clip"}),
                "vae": ("VAE", {"label": "VAE"}),
                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "Flux_Guidance": ("FLOAT",{"label": "Flux Guidance for Tiles", "default": 3.5, "min": -100.0, "max": 100.0,"step": 0.1, "round": 0.01,  "tooltip": "All Fusion Modes benefit from high Guidance, so if you notice that certain areas aren't blending well, try increasing the Guidance value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),
                "image": ("IMAGE", {"label": "Image"}),
                "General_Prompt_Positive": ("STRING", {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "General_Prompt_Negative": ("STRING", {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "model_type": (self.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),
                "Fractality": ("FLOAT", {"label": "inpaint_max", "default": 1, "min": 0.5, "max": 4, "step": 0.01}),
                "Creativity": ("FLOAT", {"label": "inpaint_max", "default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "Inventivity": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 1, "step": 0.01}),
                "Resemblance": ("FLOAT", {"label": "inpaint_max", "default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "Scale_Factor_per_Step": ("FLOAT", {"default": 1, "min": 0.05, "max": 4, "step": 0.05}),
                "Upscale_Steps": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "Add_Refinement_passes": ("BOOLEAN", {"default": False}),
                "Save_Steps_to_Temp": ("BOOLEAN", {"label": "Save_Steps_to_Temp", "default": False}),

            },
            "optional": {
                "Redux_Style_Model": ("STYLE_MODEL", {"label": "Redux_Style_Model"}),
                "Redux_Clip_Vision": ("CLIP_VISION", {"label": "Redux_Clip_Vision"}),
                "PRO_segs": ("SEGS",),
                "PRO_api_token": ("STRING", {"default": ""}),
                "Controlnet_Pipe": ("Controlnet_Pipe", {"label": "TBG ControlNet Pipe"}),
                "cropped_positive": ("CONDITIONING",),
                "cropped_negative": ("CONDITIONING",),
                "Info:": ("SEGS",),
            }

        }

    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE"
    )

    RETURN_NAMES = (
        "STEP1 Refined Image",
        "STEP2 Refined Image",
        "STEP3 Refined Image",
        "STEP4 Refined Image",
        "Refinement or Final Image"
    )

    OUTPUT_NODE = True
    CATEGORY = get_category("Upscaling")
    DESCRIPTION = "An \"IMAGE TO TILE \" Node"
    FUNCTION = "fn"


    @classmethod
    def fn(self, **kwargs):
        # API login
        Enrichment_Pipe = []

        kwargs["Enrichment_Pipe"] = Enrichment_Pipe
        kwargs["Fragmentation"] = kwargs["Fractality"]
        kwargs["tile_size"] = 1024
        kwargs["tile_size_w"] = 1024
        if kwargs["Fragmentation"] and  kwargs["Fragmentation"] != 0:
            kwargs["tile_size_w"] = int(kwargs.get("tile_size_w", 1024)*kwargs["Fragmentation"])
            kwargs["tile_size"] = int(kwargs.get("tile_size", 1024)*kwargs["Fragmentation"])


        kwargs["denoise"] =  kwargs["Creativity"]


        kwargs["Controlnet_Pipe_strength"] =  kwargs["Resemblance"]
        kwargs["Redux_strength"] = kwargs["Resemblance"]*1.5
        kwargs["Redux_strength"] = min(kwargs["Redux_strength"], 1.0)

        if kwargs["Scale_Factor_per_Step"] and kwargs["Scale_Factor_per_Step"] != 1:
            kwargs["upscaler"] = "Upscale Image By"
            kwargs["upscale_by"] = kwargs["Scale_Factor_per_Step"]

        #"Upscale_Steps": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
        #"Add Refinment pass": ("BOOLEAN", {"default": False}),

        kwargs["PRO_api_token"] =  kwargs.get("PRO_api_token", None)
        if kwargs["PRO_api_token"] == "" or kwargs["PRO_api_token"] == None:
            if  os.environ["TBG_ETUR_API_KEY"]:
                kwargs["PRO_api_token"] = os.environ["TBG_ETUR_API_KEY"]
                print("TBG API uses the TBG_ETUR_API_KEY environment variable for authentication")
            else:
                print("TBG API: No token found. Pro features disabled.")
        else:
                print("TBG API uses your comfyui TBG API_KEY for authentication")

        kwargs["PRO_api_info"], kwargs["PRO_api_status"], kwargs[
            "PRO_api_creditsleft"], current_credits = PatreonAuthNative.check_status(0, kwargs["PRO_api_token"])
        # return result

        kwargs["PRO_Tile_Fusion_Mode"] = 'Neuro_Generative_Tile_Fusion'
        kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
        kwargs["Optimize_Tile_Size"] =  "Disabled"
        kwargs["max_upscale_size_segment"] = 2048
        kwargs["upscale_model"] = None
        kwargs["upscaler_method"] ='lanczos'
        kwargs["LLMPrompt"] = "NONE"
        kwargs["LLMPrompt_Prompt"] = "Provide a highly detailed description of the image, emphasizing materials and textures. Enhance every visual detail, including accurate colors, lighting, and stylistic elements. Include a comprehensive list of all visible objects with precise and vivid descriptions. Write the result as a Flux image generation prompt, without any introductory."
        kwargs["compositing_mask_blur"] = 32
        kwargs["PRO_activate"] = True
        kwargs["PRO_Tile_Fusion_Mode"] = "Neuro_Generative_Tile_Fusion"
        kwargs["PRO_Tile_Fusion_blur_margin"] = 48
        kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
        kwargs["PRO_Tile_Fusion_shift_top_left"] = 0
        kwargs["PRO_Tile_Fusion_border_margin"] = 32
        kwargs["presets"] = "NONE"
        kwargs["Tile_Fusion_Mode"] = "From TGB Tiler Node"
        kwargs["Tile_Fusion_Blend"] = 0.5

        kwargs["denoise_method"] ='normalized advanced'
        kwargs["vae_encode"] = True
        kwargs["tile_size_vae"] = 1024
        kwargs["Save_Tiles_in_Temp_Folder"] = False
        kwargs["Fast_1_Tile_Preview"] =False
        kwargs["Selected_Tiles_Only"] =False
        kwargs["Selected_Tiles_By_Numbers"] =''
        kwargs["Color_Match"] = 'hm-mvgd-hm'
        kwargs["Controlnet_Pipe_strength"] = 1.00

        kwargs["PRO_Fusion_Space_Denoise"] = 0
        kwargs["PRO_Tile_Cache"] = 'OFF'
        kwargs["PRO_Resume_Tiled_Refinement"] = False
        kwargs["Enrichment_Pipe"] = None
        kwargs["Custom_Sigmas_!DENOISE=1"] = None
        kwargs["Resume_Tiled_Refinement_Image"] = None

        kwargs["Debug_Grid_Overlay"] =  False


        if kwargs["PRO_Tile_Fusion_Mode"] == 'NGTF_FLUX_Kontext':
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True

        min_tile_size = min(kwargs["tile_size"], kwargs["tile_size_w"])
        kwargs = get_presets(min_tile_size, **kwargs)
        # Read an environment variable
        #result =  TBG_Upscaler_v1.fn(**kwargs)
        #(overlay_masks_image, (self.INPUTS, self.PARAMS, self.KSAMPLER, self.OUTPUTS, self.SEGMENTS, self.SIZE, self.API,), (self.OUTPUTS.grid_prompts, output_tiles, self.SEGMENTS.segment_tiles,), info,)
        kwargs["Enrichment_Pipe"] =  self.Enrichment_Pipe()
        kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "none"
        if kwargs["Inventivity"]:
            kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = True
            kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = kwargs["Inventivity"]
            kwargs["Enrichment_Pipe"][0]["eta"] = kwargs["Inventivity"] / 10

        if kwargs["Inventivity"] > 0.9:
            kwargs["Enrichment_Pipe"][0]["SplitSteps"] = True
            kwargs["Enrichment_Pipe"][0]["SplitSteps_noise"] = kwargs["Inventivity"] -0.5
            kwargs["Enrichment_Pipe"][0]["SplitSteps_steps"] = int(kwargs["steps"] / 4)

# change borders depending an denoise
        # maby add cnet too
        kwargs["PRO_Tile_Fusion_blur_margin"] = 16 + int(kwargs["denoise"] * 96)  # 102 divide the tile by 5 so min 1 stayes to render 1024 = 204
        kwargs["PRO_Tile_Fusion_border_margin"] = 8 + int(kwargs["denoise"] * 80) # 88
        kwargs["PRO_Tile_Fusion_blur_margin"] = (kwargs["PRO_Tile_Fusion_blur_margin"] // 8) * 8
        kwargs["PRO_Tile_Fusion_border_margin"] = (kwargs["PRO_Tile_Fusion_border_margin"] // 8) * 8
        kwargs["PRO_Tile_Fusion_blur_margin"] = max(kwargs["PRO_Tile_Fusion_blur_margin"],48)
        kwargs["PRO_Tile_Fusion_border_margin"] =  max(kwargs["PRO_Tile_Fusion_border_margin"],32)
        kwargs["compositing_mask_blur"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 16)

        finalimages = []
        kwargs["Tile_Fusion_Blend"] = kwargs["denoise"]
        if kwargs["Upscale_Steps"] == 1:
            _, tbg_pipe, _, _ , _ = TBG_Upscaler_v1.fn(**kwargs)
            kwargs["TBG_Pipe"] = tbg_pipe
            result = TBG_Refiner_v1.fn(**kwargs)
            finalimages.append(result[2])
            refined_image = result[2]

            if kwargs["Add_Refinement_passes"] == True:
                kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "finer details"
                kwargs["PRO_Tile_Cache"] = 'use Cached Tiles as Input'
                result = TBG_Refiner_v1.fn(**kwargs)
                refined_image = result[2]

        else:

            # Calculate per-step scale factor
            # Compute scale per step (multiplicative)

            newimages = None
            origdenoise = copy.copy( kwargs["denoise"] )
            for i in range(0, kwargs["Upscale_Steps"]):
                mi = i+1

                if newimages is not None:
                    kwargs["image"] = newimages

                # Tiler
                mask_overlay_preview, tbg_pipe, tile_prompt_pipe, info, _ = TBG_Upscaler_v1.fn(**kwargs)
                kwargs["TBG_Pipe"] = tbg_pipe

                #Refiner

                # for the second+ step we reduce the step count - low denoise dont need hight stepcounts
                if kwargs["steps"] > 6 and kwargs["denoise"] < 0.5 and i > 0:
                    kwargs["steps"] = int(kwargs["steps"] * 0.8)

                # Reduce denoise on each step
                kwargs["denoise"] = max(origdenoise / mi, 0)
                kwargs["denoise"] = min(kwargs["denoise"], 1)

                # Calculate Fusion parameter related to denoise settings
                kwargs["PRO_Tile_Fusion_blur_margin"] = 16 + int(kwargs["denoise"] * 102)  # divide the tile by 5 so min 1 stays to render 1024 = 204
                kwargs["PRO_Tile_Fusion_border_margin"] = 8 + int(kwargs["denoise"] * 88)
                kwargs["PRO_Tile_Fusion_blur_margin"] = (kwargs["PRO_Tile_Fusion_blur_margin"] // 8) * 8
                kwargs["PRO_Tile_Fusion_border_margin"] = (kwargs["PRO_Tile_Fusion_border_margin"] // 8) * 8
                kwargs["PRO_Tile_Fusion_blur_margin"] = max(kwargs["PRO_Tile_Fusion_blur_margin"], 48)
                kwargs["PRO_Tile_Fusion_border_margin"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 32)
                kwargs["compositing_mask_blur"] = max(kwargs["PRO_Tile_Fusion_border_margin"],16)

                # Reduce Guidance on each upscale
                if  kwargs["Flux_Guidance"] > 1.5:
                    kwargs["Flux_Guidance"] = min(kwargs["Flux_Guidance"] - 0.5, 1)

                kwargs["Tile_Fusion_Blend"] =  0.5# kwargs["denoise"]

                # Set Cnet and Redux Strength
                kwargs["Controlnet_Pipe_strength"] = kwargs["Resemblance"]
                kwargs["Redux_strength"] = kwargs["Resemblance"]

                # Set noise injection eta - add split step noise injection on high value
                kwargs["Inventivity"] = kwargs["Inventivity"] / mi
                if kwargs["Inventivity"]:
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = "True"
                    kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = kwargs["Inventivity"] / mi
                    kwargs["Enrichment_Pipe"][0]["eta"] = kwargs["Inventivity"] / 10 / mi
                if kwargs["Inventivity"] > 0.9:
                    kwargs["Enrichment_Pipe"][0]["SplitSteps"] = True
                    kwargs["Enrichment_Pipe"][0]["SplitSteps_noise"] = kwargs["Inventivity"] - 0.5
                    kwargs["Enrichment_Pipe"][0]["SplitSteps_steps"] = int(kwargs["steps"] / 4)

                result = TBG_Refiner_v1.fn(**kwargs)
                finalimages.append(result[2])
                refined_image = result[2]
                # update input image:

                newimages = result[2]

                if  kwargs["Save_Steps_to_Temp"]:
                   filename_prefix = f"TBG_MM/STep{i}"
                   preview = nodes.PreviewImage()
                   _ = preview.save_images(newimages, filename_prefix, None, None)['ui']['images']



            if kwargs["Add_Refinement_passes"] == True:
                kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "finer details"
                kwargs["PRO_Tile_Cache"] = 'use Cached Tiles as Input'
                result = TBG_Refiner_v1.fn(**kwargs)
                refined_image = result[2]
                if kwargs["Save_Steps_to_Temp"]:
                    filename_prefix = f"TBG_MM/Refinement_Step{i}"
                    preview = nodes.PreviewImage()
                    _ = preview.save_images(newimages, filename_prefix, None, None)['ui']['images']


        needed_indices = [0, 1, 2, 3]
        for i in needed_indices:
            # Extend the list with None if it's too short
            while len(finalimages) <= i:
                finalimages.append(None)

            # If the position is empty, fill it with result[2]
            if finalimages[i] is None:
                finalimages[i] =  kwargs["image"]
        return {
            "ui": {"value": [f"{current_credits}"]},
            "result": (finalimages[0],finalimages[1],finalimages[2],finalimages[3],refined_image)
        }
    @staticmethod
    def Enrichment_Pipe():
            Enrichment_Pipe = []
            Enrichment_Pipe.append({
                "detail_daemon_active": False,
                "detail_amount": 0,
                "detail_daemon_start": 0.12,
                "detail_daemon_end": 0.3,
                "detail_daemon_bias": 0.5,
                "detail_daemon_exponent": 0.8,
                "detail_daemon_start_offset": 0,
                "detail_daemon_end_offset": 0,
                "detail_daemon_fade": 0,
                "detail_daemon_smooth": False,
                "detail_daemon_cfg_scale": 0,
                "latentupscale": False,
                "latentupscale_noise": 0,
                "latentupscale_steps": 0,
                "latentupscale_denoise": 0,
                "SplitSteps": False,
                "SplitSteps_noise": 0,
                "SplitSteps_steps": 0,
                "SplitStepsSigmas": 0,
                "SplitStepsMultiplyer": 0,
                "SplitStepsSigmasCurve": 0,
                "SplitStepsStart": 0,
                "SplitStepsEnd": 0,
                "eta": 0,  # kwargs["Inventivity"]/100,
                "RF_inversion": 0,
                "tile_upscale_plus": "none",
                "upscaler_method_inpainting": 'lanczos',
                "upscale_model_inpainting": None,
                "upscale_tiles_by": 1.5,
                "upscale_segments_by": 1.5,
            })


            return Enrichment_Pipe


