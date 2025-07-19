"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX CE Community Edition_________________________________________________________

"""
import math

# Application-specific imports
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
from ...vendor.ComfyUI_MaraScott_Nodes.py.utils.constants import get_category
from ....TBG_presets import PRESETS_CE, get_presets


class TBG_Tiled_Upscaler_CE():
    NAME = "TBG Tiled Upscaler CE"

    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    SEGMENTS = {}
    SIZE = {}
    LLM = {}

    PRESETS = PRESETS_CE

    DIFFUSION_MODES = [
        'NONE',
        'Neuro_Generative_Tile_Fusion',
    ]
    ROUND_METHODS = [
        'Disabled',
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
    ]

    LLM = [
        "NONE",
        "Janus-Pro-1B",
        "Janus-Pro-7B"
    ]

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {

                "PRO_activate": ("BOOLEAN", {"label": "api_activate_pro", "default": False, "label_on": "ETUR PRO","label_off": "ETUR"}),
                "PRO_Tile_Fusion_Mode": (self.DIFFUSION_MODES, {"label": "PRO_Tile_Fusion_Mode", "default": "NONE"}),
                "PRO_Tile_Fusion_blur_margin": ("INT",{"label": "PRO_Tile_Fusion_blur_margin","default": 0,"min": 0, "max": 1024, "step": 8}),
                "PRO_Tile_Fusion_shift_in_out": ("INT",{"label": "PRO_Tile_Fusion_shift_in_out", "default": 0, "min": -1024,"max": 1024,"step": 8}),
                "PRO_Tile_Fusion_shift_top_left": ("INT",{"label": "PRO_Tile_Fusion_shift_top_left", "default": 0, "min": -1024,"max": 1024,"step": 8}),
                "PRO_Tile_Fusion_border_margin": ("INT", {"label": "shift_mask", "default": 0, "min": -1024, "max": 1024,"step": 8}),
                "PRO_Fusion_Space_Denoise": ("FLOAT",{"label": "inpaint_max", "default": 0.00, "min": -10, "max": 10, "step": 0.01}),
                "tile_size_w": ("INT",{"label": "Tile Size width", "default": 1024, "min": 320, "max": 8192, "step": 8}),
                "max_upscale_size_segment": ("INT", {"label": "max_upscale_size_segment","default": 2048, "min": 256, "max": 4096, "step": 8}),
                "Optimize_Tile_Size": (self.ROUND_METHODS, {"label": "Optimize_Tile_Size", "default": "Disabled"}),
                "upscaler": (self.UPSCALE_TYPE, {"label": "Upscale Type", "default": "NONE"}),
                "PRO_api_token": ("STRING", {"default": ""}),
                "id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"},

            "required": {
                "image": ("IMAGE", {"label": "Image"}),
                "presets": (self.PRESETS, {"label": "presets", "default": "NONE"}),
                "Fragmentation": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": -5, "max": 5, "step": 0.01}),
                "tile_size": ("INT",{"label": "Tile Size height", "default": 1024, "min": 320, "max": 8192, "step": 8}),
                "upscaler": (self.UPSCALE_TYPE, {"label": "Upscale Type", "default": "NONE"}),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"), {"label": "Upscale Model"}),
                "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05}),
                "upscaler_method": (self.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'lanczos'}),
                "LLMPrompt": (self.LLM, {"label": "Upscale Type", "default": "NONE"}),
                "LLMPrompt_Prompt": ("STRING",{"multiline": True, "label": "LLMPrompt Prompt","default": "Provide a highly detailed description of the image, emphasizing materials and textures. Enhance every visual detail, including accurate colors, lighting, and stylistic elements. Include a comprehensive list of all visible objects with precise and vivid descriptions. Write the result as a Flux image generation prompt, without any introductory."}),
                "compositing_mask_blur": ("INT", {"label": "Manual Feather Mask for Tile Overlapping", "default": 32, "min": 0, "max": 1024, "step": 8}),

            },
            "optional": {
                  "PRO_segs": ("SEGS",),
            }
        }

    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    RETURN_TYPES = (
        "IMAGE",
        "TBG_Pipe",
        "Tile_Prompt_Pipe",  # self.OUTPUTS.grid_prompts ,output_tiles
        "STRING"

    )

    RETURN_NAMES = (
        "Mask Overlay Preview",
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "Info"
    )

    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
    )

    OUTPUT_NODE = True
    CATEGORY = get_category("Upscaling")
    DESCRIPTION = "An \"IMAGE TO TILE \" Node"
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):

        if kwargs["Fragmentation"] and  kwargs["Fragmentation"] != 0:
            kwargs["tile_size_w"] = int(kwargs.get("tile_size_w", 1024)*kwargs["Fragmentation"])
            kwargs["tile_size"] = int(kwargs.get("tile_size", 1024)*kwargs["Fragmentation"])


        # set square Tile
        kwargs["CE_segs"] = kwargs.get("segs_Alpha1_CE", None)
        kwargs["inpaint_max"] = 0
        kwargs["upscaler_method"] = 'lanczos'
        kwargs["max_upscale_size_segment_inpainting"] = int(2*kwargs.get("tile_size", 1024))
        # auto square tile
        kwargs["tile_size_w"] = kwargs.get("tile_size", 1024)
        kwargs["PRO_Neuro_Generative_Tile_Fusion"] = False

        min_tile_size = min(kwargs["tile_size"], kwargs["tile_size_w"])
        kwargs = get_presets(min_tile_size, **kwargs)
        result = TBG_Upscaler_v1.fn(**kwargs)

        # For output nodes, return UI separately
        return {
            "result": result
        }



class TBG_Refiner_CE():
    SIZE = None
    SEGMENTS = None
    OUTPUTS = None
    KSAMPLER = None
    INPUTS = None
    PARAMS = None
    NAME = "TBG Refiner CE"

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
        'normalized advanced',
    ]

    COLOR_MATCH_METHODS = [
        'none',
        'hm-mvgd-hm',
    ]

    DIFFUSION_MODES = [
        'From TGB Tiler Node',
        'Tile_Fusion',
        'Neuro_Generative_Tile_Fusion',
    ]

    @classmethod
    def INPUT_TYPES(self):
        # def INPUT_TYPES(cls):

        return {


            "optional": {
                "Controlnet_Pipe": ("Controlnet_Pipe", {"label": "TBG ControlNet Pipe"}),
                "Tile_Prompt_Pipe": ("TILE_Prompt_PIPE_OUT", {"label": "Tile Prompt Pipe"}),
                "Enrichment_Pipe": ("Enrichment_Pipe", {"label": "TBG enrichment Pipe"}),
                "cropped_positive": ("CONDITIONING",),
                "cropped_negative": ("CONDITIONING",),
                "Redux_Style_Model": ("STYLE_MODEL", {"label": "Redux_Style_Model"}),
                "Redux_Clip_Vision": ("CLIP_VISION", {"label": "Redux_Clip_Vision"}),
                "Redux_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),

            },
            "required": {
                "model": ("MODEL", {"label": "Model"}),
                "clip": ("CLIP", {"label": "Clip"}),
                "vae": ("VAE", {"label": "VAE"}),
                "TBG_Pipe": ("TBG_Pipe", {"label": "TBG Pipe"}),

                "model_type": (self.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),
                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"label": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "Flux_Guidance": ("FLOAT", {"label": "Flux Guidance for Tiles", "default": 4.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                            "tooltip": "All Fusion Modes benefit from high Guidance, so if you notice that certain areas aren't blending well, try increasing the Guidance value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),
                "denoise": ("FLOAT", {"label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),

                "vae_encode": ("BOOLEAN", {"label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard", "tooltip": ""}),
                "tile_size_vae": ("INT", {"label": "Tile Size (VAE)", "default": 1024, "min": 256, "max": 4096, "step": 64}),
                "General_Prompt_Positive": ("STRING", {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "General_Prompt_Negative": ("STRING", {"multiline": True, "label": "General Prompt for all Tiles", "default": ""}),
                "Save_Tiles_in_Temp_Folder": ("BOOLEAN", {"label": "Preview_Tiles_in_Temp_Folder", "default": False, "label_on": "Save Tiles to /temp/TBG/", "label_off": "Disabled"}),
                "Fast_1_Tile_Preview": ("BOOLEAN", {"label": "Fast_1_Tile_Preview", "default": False, "label_on": "Preview Single Tile", "label_off": "Disabled",
                                                    "tooltip": "The first Selected_Tiles_By_Number are processed at full scale as a preview, allowing a quick check of settings before processing the entire set."}),
                "Selected_Tiles_Only": ("BOOLEAN", {"label": "Process_selected_Tiles_only", "default": False, "label_on": "Generate Selected Tiles Only", "label_off": "Disabled"}),
                "Selected_Tiles_By_Numbers": ("STRING", {"label": "Selected_Tiles_Index_Numbers to process", "default": '',
                                                         "tooltip": "You can set a list of selected tiles to process like 1,2,3,6 and activate Selected_Tiles_Only"}),

                "Color_Match": (self.COLOR_MATCH_METHODS, {"label": "Color Match Method", "default": 'none'}),
                "Controlnet_Pipe_strength": ("FLOAT", {"label": "Controlnet_Pipe_strength", "default": 1.00, "min": 0, "max": 100, "step": 0.01, "round": 0.01,
                                                       "tooltip": "It's a multiplier value applied uniformly to all ControlNets from CnetPipe, scaling their combined influence."}),

            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "Debug_Grid_Overlay": ("BOOLEAN", {"label": "Debug_Grid_Overlay", "default": False, "label_on": "Show Grid", "label_off": "Disabled"}),
                "contrast": ("INT", {"label": "contrast", "default": 0, "min": 0, "max": 100.0, "step": 1}),
                "highpass": ("FLOAT", {"highpass": "CFG", "default": 1, "min": -10, "max": 100.0, "step": 0.1, "round": 0.01}),
                "PRO_Fusion_Space_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 2, "step": 0.01, "round": 0.01,
                                                       "tooltip": "If set to 0, it inherits the value from the Tiled Upscaler. Use it as a denoiser in mask space: 0–1 affects white areas, 1–2 affects black areas. For previews, adjust it in the TBG Enhanced Tiled Upscaler FLUX PRO node — it's duplicated there to allow refinement without rerunning all nodes."}),
                "PRO_Tile_Cache": ("BOOLEAN", {"label": "Tile_Cache", "default": False, "label_on": "keep generated", "label_off": "Disabled",
                                               "tooltip": "Cached the generated tiles to enable generative tile fusion to insert or correct individual images after the final render is complete."}),
                "PRO_Resume_Tiled_Refinement": ("BOOLEAN", {"label": "PRO_Resume_Tiled_Refinement", "default": False, "label_on": "Resume", "label_off": "Disabled",
                                                            "tooltip": "Provide the original input image along with the completed tiled upscaling result to resume the refinement process. This allows continuing enhancement based on the initial image and the existing finished output."}),
                "Enhanced_Laplacian_Blending ": ("BOOLEAN", {"label": "Laplacian Pyramid Blending", "default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Work in progress"}),
                "denoise_method": (self.DENOISE_METHODS, {"label": "DENOISE_METHODS", "default": 'default'}),
                "Tile_Fusion_Mode": (self.DIFFUSION_MODES, {"label": "Tile_Fusion_Mode", "default": "From TGB Tiler Node"}),
                "Tile_Fusion_Blend": ("FLOAT", {"label": "Tile_Fusion_Blend", "default": 0.5, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "Custom_Sigmas_!DENOISE=1": ("SIGMAS", {"label": "Sigmas with denoise 1", "tooltip": "Insert your full custom sigma noise curve (not denoised), as denoising is performed per tile by the node."}),
                "Resume_Tiled_Refinement_Image": ("IMAGE", {"label": "Presampled_Background_Image"}),

            },

        }

    RETURN_TYPES = (
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",

    )

    RETURN_NAMES = (
        "TBG_Pipe",
        "Tile_Prompt_Pipe",
        "Refined Image",
        "Refined Image without Segments",
        "Refined Image without ColorCorrection",
        "Original Image",
    )

    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)

    OUTPUT_NODE = True
    CATEGORY = get_category("Refiner")
    DESCRIPTION = "A \"Tile Refiner\" Node"
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        if  kwargs["model_type"] == 'FLUX1 Kontext':
            self.PARAMS.FLUX_Kontext = True
        kwargs["Enhanced_Laplacian_Blending"] = False
        kwargs["PRO_Resume_Tiled_Refinement"] = False
        kwargs["PRO_Tile_Cache"] = False
        kwargs["PRO_Fusion_Space_Denoise"] = 0
        kwargs["Debug_Grid_Overlay"] = False
        kwargs["denoise_method"] = "default"
        kwargs["Tile_Fusion_Mode"] = "From TGB Tiler Node"
        kwargs["Tile_Fusion_Blend"] = 0.5
        kwargs["Resume_Tiled_Refinement_Image"] = None
        kwargs["Custom_Sigmas_!DENOISE=1"] = None
        kwargs["cnet_multiply"] =kwargs["Controlnet_Pipe_strength"]


        return TBG_Refiner_v1.fn(**kwargs)
