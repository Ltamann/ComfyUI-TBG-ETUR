"""
TBG_magnific_ETUR: Enhanced Tiled Upscaler and Refiner (FLUX PRO)
Cleaned-up, commented version – logic and kwargs flow preserved 1-to-1.
"""

# ─────────────────────────────── Imports ──────────────────────────────── #
from __future__ import annotations

import nodes
import comfy
import comfy.latent_formats
import comfy.model_sampling
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models


from ..UpscalerRefiner.TBG_Refiner import TBG_Refiner_v1
from ..UpscalerRefiner.TBG_Tiler import TBG_Upscaler_v1
from ....TBG_presets import PRESETS_PRO

# ────────────────────────────── Main Class ────────────────────────────── #
class TBG_magnific_ETUR:
    """
    Magnific FLUX PRO *image-to-tile* upscaler / refiner.
    """

    # ──────────────────────────── Metadata ────────────────────────────── #
    NAME = "TBG Magnific Magnifier PRO"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    OUTPUT_NODE = True
    CATEGORY = "TBG/Magnific Magnifier Upscaler"
    DESCRIPTION = 'An multistep upscaler and refiner Node'
    FUNCTION = "fn"

    # ────────────────────────── Static Tables ─────────────────────────── #
    # Placeholder attributes for compatibility
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    SEGMENTS = {}
    SIZE = {}

    PRESETS = PRESETS_PRO
    DIFFUSION_MODES = ["Soft Merge", "Tile_Fusion", "Neuro_Generative_Tile_Fusion"]
    ROUND_METHODS = ["Disabled"]        # XL / other variants reserved
    UPSCALE_TYPE = [
        "NONE",
        "Upscale Image By",
        "Upscale Image By (using Model)",
        "Upscale Image (using Model)",
    ]
    UPSCALE_METHODS = ["area", "bicubic", "bilinear", "bislerp", "lanczos", "nearest-exact"]
    LLM = ["NONE", "Janus-Pro-1B", "Janus-Pro-7B"]

    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX1 Kontext': 1024,
        'Qwen Image': 1328,
        'Qwen Image Edit': 1328,
        'Others': 1024,
    }
    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())
    DENOISE_METHODS = [
        "default",
        "normalized",
        "normalized advanced",
        "multiplied",
        "multiplied normalized",
        "default short ",
    ]
    COLOR_MATCH_METHODS = [
        "none",
        "mkl",
        "hm",
        "reinhard",
        "mvgd",
        "hm-mvgd-hm",
        "hm-mkl-hm",
    ]
    CACHE = ["OFF", "use Cached Tiles as Input", "use Cached Tiles only for Fusion"]

    # ─────────────────────────── UI Schemas ───────────────────────────── #
    @classmethod
    def INPUT_TYPES(cls):
        """
        Node input specification for ComfyUI. *All labels, defaults,
        value ranges, and tool-tips are preserved verbatim.*
        """
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
                "seed": (
                    "INT",
                    {"label": "Seed", "default": 4, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"label": "Steps", "default": 30, "min": 1, "max": 10_000}),
                "cfg": (
                    "FLOAT",
                    {
                        "label": "CFG",
                        "default": 1,
                        "min": -10,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "Flux_Guidance": (
                    "FLOAT",
                    {
                        "label": "Flux Guidance for Tiles",
                        "default": 3.5,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": (
                            "All Fusion Modes benefit from high Guidance; "
                            "if areas fail to blend, increase this value."
                        ),
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"label": "Sampler Name"}),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"label": "Basic Scheduler"}),
                "image": ("IMAGE", {"label": "Image"}),
                "General_Prompt_Positive": (
                    "STRING",
                    {"multiline": True, "label": "General Prompt for all Tiles", "default": ""},
                ),
                "General_Prompt_Negative": (
                    "STRING",
                    {"multiline": True, "label": "General Prompt for all Tiles", "default": ""},
                ),
                "model_type": (cls.MODEL_TYPES, {"label": "Model Type", "default": "FLUX1"}),
                "Tiled_Sampler": (
                    "BOOLEAN",
                    {
                        "label": "Tiled_Sampler",
                        "default": True,
                        "label_on": "On",
                        "label_off": "Off",
                        "tooltip": "For refining images without tiling",
                    },
                ),
                # Creative control sliders
                "Fractality": ("FLOAT", {"label": "inpaint_max", "default": 1, "min": 0.1, "max": 4, "step": 0.01, "round": 0.01}),
                "Creativity": ("FLOAT", {"label": "inpaint_max", "default": 0.5, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "Resemblance": ("FLOAT", {"label": "inpaint_max", "default": 0.5, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "Add_Details": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "Resharpener": ("FLOAT", {"label": "inpaint_max", "default": 0.0, "min": -1, "max": 1, "step": 0.01, "round": 0.01}),
                "Scale_Factor_per_Step": ("FLOAT", {"default": 1, "min": 1, "max": 4, "step": 0.01, "round": 0.01}),
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
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "STEP1 Refined Image",
        "STEP2 Refined Image",
        "STEP3 Refined Image",
        "STEP4 Refined Image",
        "Refinement or Final Image",
    )

    # ─────────────────────────── Public API ───────────────────────────── #
    @classmethod
    def fn(cls, **kwargs):
        """
        Main execution entry for ComfyUI.
        CRITICAL: Preserves exact kwargs flow - some must be set AFTER Tiler calls.
        """
        # ── Initial kwargs setup (BEFORE any Tiler calls) ──────────────── #
        if kwargs["Tiled_Sampler"]:
            kwargs["presets"] = "NONE"
        else:
            kwargs["presets"] = 'Full size Image no Tiles'

        Enrichment_Pipe = []

        # Renaming and aliasing
        kwargs["Enrichment_Pipe"] = Enrichment_Pipe
        kwargs["Fragmentation"] = kwargs["Fractality"]
        kwargs["denoise"] = kwargs["Creativity"]
        origdenoise = kwargs["Creativity"]  # Store original denoise value
        kwargs["Fragmentation"] = kwargs["Fractality"]  # Duplicate assignment from original
        kwargs["Controlnet_Pipe_strength"] = min(kwargs["Resemblance"], 1)
        kwargs["Redux_strength"] = min(kwargs["Resemblance"], 1)
        kwargs["tile_size"] = 1024
        kwargs["tile_size_w"] = 1024

        # Resize Tiles based on Fragmentation
        if "Fragmentation" in kwargs and kwargs["Fragmentation"] not in (0.0, 1.0):
            kwargs["tile_size_w"] = int(kwargs.get("tile_size_w", 1024) * kwargs["Fragmentation"])
            kwargs["tile_size"] = int(kwargs.get("tile_size", 1024) * kwargs["Fragmentation"])

        # Set upscale factor for Tiles
        if kwargs["Scale_Factor_per_Step"] and (kwargs["Scale_Factor_per_Step"] != 1 or kwargs["Scale_Factor_per_Step"] != 0.0):
            kwargs["upscaler"] = "Upscale Image By"
            kwargs["upscaler_method"] = 'lanczos'
            kwargs["upscale_by"] = kwargs["Scale_Factor_per_Step"]
        else:
            kwargs["upscaler"] = "NONE"
            kwargs["upscaler_method"] = 'lanczos'

        # PRO configuration - must be set before Tiler calls
        kwargs["PRO_activate"] = True
        kwargs["PRO_Tile_Fusion_Mode"] = 'Neuro_Generative_Tile_Fusion'
        kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
        kwargs["Optimize_Tile_Size"] = "Disabled"
        kwargs["max_upscale_size_segment"] = 2048
        kwargs["LLMPrompt"] = "NONE"
        kwargs["LLMPrompt_Prompt"] = ""

        # Initial fusion and compositing settings
        kwargs["compositing_mask_blur"] = 16
        kwargs["PRO_Tile_Fusion_blur_margin"] = 48
        kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
        kwargs["PRO_Tile_Fusion_shift_top_left"] = 48+16
        kwargs["PRO_Tile_Fusion_border_margin"] = 16
        kwargs["Tile_Fusion_Mode"] = "From TGB Tiler Node"
        kwargs["Tile_Fusion_Blend"] = 0
        kwargs["denoise_method"] = 'normalized advanced'
        kwargs["vae_encode"] = True
        kwargs["tile_size_vae"] = 1024
        kwargs["Save_Tiles_in_Temp_Folder"] = False
        kwargs["Fast_1_Tile_Preview"] = False
        kwargs["Selected_Tiles_Only"] = False
        kwargs["Selected_Tiles_By_Numbers"] = ''
        kwargs["Color_Match"] = 'hm-mvgd-hm'
        kwargs["PRO_Fusion_Space_Denoise"] = 0
        kwargs["PRO_Tile_Cache"] = 'OFF'
        kwargs["PRO_Resume_Tiled_Refinement"] = False
        kwargs["Enrichment_Pipe"] = None
        kwargs["Custom_Sigmas_!DENOISE=1"] = None
        kwargs["Resume_Tiled_Refinement_Image"] = None
        kwargs["Debug_Grid_Overlay"] = False

        # Handle Kontext mode
        if kwargs["PRO_Tile_Fusion_Mode"] == 'NGTF_FLUX_Kontext':
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True

        # Pipeline initialization
        newimages = None
        finalimages = []
        current_credits = 0  # Placeholder for UI info text
        kwargs["Enrichment_Pipe"] = cls.Enrichment_Pipe()

        # Set borders and scale based on denoise strength (max total recommended 204)
        #kwargs["PRO_Tile_Fusion_blur_margin"] = 16 + int(kwargs["denoise"] * 96)  # 102 divide the tile by 5 so min 1 stays to render 1024 = 204
        #kwargs["PRO_Tile_Fusion_border_margin"] = 8 + int(kwargs["denoise"] * 8)  # 88
        #kwargs["PRO_Tile_Fusion_blur_margin"] = (kwargs["PRO_Tile_Fusion_blur_margin"] // 8) * 8
        #kwargs["PRO_Tile_Fusion_border_margin"] = (kwargs["PRO_Tile_Fusion_border_margin"] // 8) * 8
        #kwargs["PRO_Tile_Fusion_blur_margin"] = max(kwargs["PRO_Tile_Fusion_blur_margin"], 48)
        #kwargs["PRO_Tile_Fusion_border_margin"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 32)
        #kwargs["compositing_mask_blur"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 16)
        # kwargs["PRO_Tile_Fusion_shift_top_left"] =   kwargs["PRO_Tile_Fusion_border_margin"] + kwargs["PRO_Tile_Fusion_blur_margin"]

        # Creative control variables for loop
        detail_daemon_active = True
        detail_amount_add = kwargs["Add_Details"]
        eta = kwargs["Add_Details"] / 2
        detail_daemon_end = max(0.4, 0.2)
        Sigmas_creative_tail_active = True
        Sigmas_creative_tail_strength= kwargs["Add_Details"]

        Resharpener = kwargs["Resharpener"]
        kwargs['Color_Match_Str'] = 1 - kwargs["denoise"]
        kwargs["Enrichment_Pipe"][0]["eta"] = kwargs["Add_Details"]
        kwargs["Enrichment_Pipe"][0]["eta_starta"] = 0
        kwargs["Enrichment_Pipe"][0]["eta_end"] = 0.5
        kwargs["Enrichment_Pipe"][0]["resharpen_start"] = 0
        kwargs["Enrichment_Pipe"][0]["resharpen_end"] = 0.2
        i = 0
        refined_image = None
        # ── Main upscale / refine loop ───────────────────────────────── #
        for i in range(0, kwargs["Upscale_Steps"]):
            # Reset Enrichment_Pipe for each iteration
            kwargs["Enrichment_Pipe"] = cls.Enrichment_Pipe()

            if i == 0:  # First pass
                # ── BEFORE TILER: Set initial Enrichment_Pipe ─────────── #
                kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "none"

                # ── CALL TILER (may overwrite some kwargs) ────────────── #

                _, tbg_pipe, _, _, _,current_credits = TBG_Upscaler_v1.fn(**kwargs)
                kwargs["TBG_Pipe"] = tbg_pipe

                # ── AFTER TILER: Configure kwargs for Refiner ─────────── #
                kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "none"

                if kwargs["Add_Details"] != 0:
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_active"] = Sigmas_creative_tail_active
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_strength"] = Sigmas_creative_tail_strength
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = detail_daemon_active
                    kwargs["Enrichment_Pipe"][0]["detail_amount"] = detail_amount_add
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_end"] = detail_daemon_end
                    kwargs["Enrichment_Pipe"][0]["eta_active"] = True
                    kwargs["Enrichment_Pipe"][0]["eta"] = kwargs["Add_Details"] / 2#cls.compute_eta(kwargs["Add_Details"], kwargs["denoise"])
                else:
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = False
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_active"] = False
                    kwargs["Enrichment_Pipe"][0]["eta_active"] =False

                if kwargs["Resharpener"] != 0:
                    kwargs["Enrichment_Pipe"][0]["Resharpener_active"] = True
                    kwargs["Enrichment_Pipe"][0]["Resharpener_strength"] = Resharpener
                    kwargs["Enrichment_Pipe"][0]["resharpen_end"] = max(0.2, min(1.0, (1 - kwargs["denoise"])))
                else:
                    kwargs["Enrichment_Pipe"][0]["Resharpener_active"] = False
                kwargs["Enrichment_Pipe"][0]["PRO_Fusion_Complexity_min_Denoise"] = 0
                kwargs["Enrichment_Pipe"][0]["PRO_Fusion_Complexity_max_Denoise"] = 1
                kwargs["Enrichment_Pipe"][0]["PRO_Fusion_Complexity_Mask_Blur"] = 0

                # Has to be set last before calling refiner
                kwargs["Controlnet_Pipe_strength"] = kwargs["Resemblance"]
                kwargs["Redux_strength"] = kwargs["Resemblance"]

                # ── CALL REFINER ──────────────────────────────────────── #
                result = TBG_Refiner_v1.fn(**kwargs)
                finalimages.append(result[2])
                refined_image = result[2]
                newimages = result[2]

                # Optional save to temp
                if kwargs["Save_Steps_to_Temp"]:
                    filename_prefix = f"TBG_MM/STep{i}"
                    preview = nodes.PreviewImage()
                    _ = preview.save_images(newimages, filename_prefix, None, None)['ui']['images']

            else:  # Subsequent passes (i >= 1)
                kwargs["seed"] = kwargs["seed"] + i

                # Reset denoise - add visual effect to noise injection
                kwargs["denoise"] = min(origdenoise + (kwargs["Add_Details"] / 30), 1)

                # Calculate step 2-3-4 - multiplicative scale per step
                mi = i + 1

                # Use previous result as input
                kwargs["image"] = newimages

                # ── CALL TILER (may overwrite some kwargs) ────────────── #
                _, tbg_pipe, _, _, _, current_credits = TBG_Upscaler_v1.fn(**kwargs)
                mask_overlay_preview, tbg_pipe, tile_prompt_pipe, info, _,current_credits = TBG_Upscaler_v1.fn(**kwargs)
                kwargs["TBG_Pipe"] = tbg_pipe

                # ── AFTER TILER: Adjust kwargs for Refiner ─────────────── #
                # Reduce step count for second+ steps with low denoise
                if kwargs["steps"] > 8 and kwargs["denoise"] < 0.5 and i > 0:
                    kwargs["steps"] = int(kwargs["steps"] * 0.8)

                # Reduce denoise on each step
                kwargs["denoise"] = max(origdenoise / (mi * 0.8), 0.2)
                kwargs["denoise"] = min(kwargs["denoise"], 1)

                # Reset and configure Enrichment_Pipe
                kwargs["Enrichment_Pipe"] = cls.Enrichment_Pipe()
                kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "none"

                # Reduce Guidance on each upscale
                if kwargs["Flux_Guidance"] > 1.5:
                    kwargs["Flux_Guidance"] = max(kwargs["Flux_Guidance"] - 0.10, 1)

                # Set noise injection eta - add split step noise injection on high value
                if kwargs["Add_Details"] != 0:
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_active"] = True
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_strength"] = Sigmas_creative_tail_strength / mi  # High sigma tail
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = True
                    kwargs["Enrichment_Pipe"][0]["detail_amount"] = detail_amount_add / mi
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_end"] = detail_daemon_end
                    kwargs["Enrichment_Pipe"][0]["eta_active"] = True
                    kwargs["Enrichment_Pipe"][0]["eta"] = kwargs["Add_Details"] / 2#max(cls.compute_eta(kwargs["Add_Details"], kwargs["denoise"]) / mi, 0)

                    if mi > 2:  # Step 3+4 without noise from here
                        kwargs["Enrichment_Pipe"][0]["eta_active"] = False
                        kwargs["Enrichment_Pipe"][0]["eta"] = 0
                        kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = False


                else:
                    kwargs["Enrichment_Pipe"][0]["eta_active"] = False
                    kwargs["Enrichment_Pipe"][0]["eta"] = 0
                    kwargs["Enrichment_Pipe"][0]["detail_daemon_active"] = False
                    kwargs["Enrichment_Pipe"][0]["Sigmas_creative_tail_active"] = False


                if kwargs["Resharpener"] != 0:
                    kwargs["Enrichment_Pipe"][0]["Resharpener_active"] = True
                    kwargs["Enrichment_Pipe"][0]["Resharpener_strength"] =  Resharpener / mi
                    kwargs["Enrichment_Pipe"][0]["resharpen_end"]  = max(0.2, min(1.0, (1-kwargs["denoise"])))
                else:
                    kwargs["Enrichment_Pipe"][0]["Resharpener_active"] = False

                    # Set ControlNet and Redux Strength - has to be set last before calling refiner
                kwargs["Controlnet_Pipe_strength"] = kwargs["Resemblance"]
                kwargs["Redux_strength"] = kwargs["Resemblance"]

                # Calculate Fusion parameters related to denoise settings
                #kwargs["PRO_Tile_Fusion_blur_margin"] = 16 + int(kwargs["denoise"] * 102)  # Divide tile by 5 so min 1 stays to render 1024 = 204
                #kwargs["PRO_Tile_Fusion_border_margin"] = 8 + int(kwargs["denoise"] * 88)

                #kwargs["PRO_Tile_Fusion_blur_margin"] = (kwargs["PRO_Tile_Fusion_blur_margin"] // 8) * 8
                #kwargs["PRO_Tile_Fusion_border_margin"] = (kwargs["PRO_Tile_Fusion_border_margin"] // 8) * 8

                #kwargs["PRO_Tile_Fusion_blur_margin"] = max(kwargs["PRO_Tile_Fusion_blur_margin"], 48)
                #kwargs["PRO_Tile_Fusion_border_margin"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 16)
                #kwargs["PRO_Tile_Fusion_shift_top_left"] = max(kwargs["PRO_Tile_Fusion_shift_top_left"], 16)

                #kwargs["compositing_mask_blur"] = max(kwargs["PRO_Tile_Fusion_border_margin"], 16)

                # ── CALL REFINER ──────────────────────────────────────── #
                result = TBG_Refiner_v1.fn(**kwargs)
                finalimages.append(result[2])
                refined_image = result[2]
                newimages = result[2]

                # Optional save to temp
                if kwargs["Save_Steps_to_Temp"]:
                    filename_prefix = f"TBG_MM/STep{i}"
                    preview = nodes.PreviewImage()
                    _ = preview.save_images(newimages, filename_prefix, None, None)['ui']['images']

                print("Step", mi, " steps=", kwargs["steps"], " denoise=", kwargs["denoise"],
                      " Controlnet_Pipe_strength=", kwargs["Controlnet_Pipe_strength"],
                      " Redux_strength=", kwargs["Redux_strength"],
                      " Flux_Guidance=", kwargs["Flux_Guidance"])

        # ── Optional refinement pass ─────────────────────────────────── #
        if kwargs["Add_Refinement_passes"] == True:
            kwargs["Enrichment_Pipe"][0]["tile_upscale_plus"] = "finer details"
            kwargs["PRO_Tile_Cache"] = 'use Cached Tiles as Input'
            result = TBG_Refiner_v1.fn(**kwargs)
            refined_image = result[2]
            kwargs["PRO_Tile_Cache"] = 'OFF'

            if kwargs["Save_Steps_to_Temp"]:
                filename_prefix = f"TBG_MM/Refinement_Step{i}"
                preview = nodes.PreviewImage()
                _ = preview.save_images(newimages, filename_prefix, None, None)['ui']['images']

        # Ensure exactly four intermediary images
        needed_indices = [0, 1, 2, 3]
        for i in needed_indices:
            # Extend the list with None if it's too short
            while len(finalimages) <= i:
                finalimages.append(None)
            # If the position is empty, fill it with original image
            if finalimages[i] is None:
                finalimages[i] = kwargs["image"]

        # ── Return to ComfyUI ─────────────────────────────────────────── #
        return {
            "ui": {"value": [f"{current_credits}"]},
            "result": (finalimages[0], finalimages[1], finalimages[2], finalimages[3], refined_image)
        }

    # ──────────────────────────── Helper Methods ─────────────────────── #
    @staticmethod
    def compute_eta(inventivity: float, denoise: float) -> float:
        """
        Computes the eta value based on inventivity and denoise levels.

        Args:
            inventivity (float): Value between 0 and 1 controlling core eta shape.
            denoise (float): Value between 0 and 1 scaling eta strength.

        Returns:
            float: Final eta value (clamped between 0 and 1).
        """
        # Step 1: Base eta from Inventivity
        if inventivity <= 0.5:
            eta_base = 0
        elif 0.5 < inventivity <= 0.75:
            eta_base = 0.1 + (inventivity - 0.5) / 0.25 * (1.0 - 0.1)  # 0.1 to 1.0
        elif inventivity <= 1.0:
            eta_base = 1.0 - (inventivity - 0.75) / 0.25 * 1.0  # 1.0 to 0
        else:
            eta_base = 0.0  # Optional clamp for values > 1

        # Step 2: Scale by denoise
        if denoise >= 0.9:
            denoise_scale = 0
        elif denoise <= 0.75:
            denoise_scale = 1.0
        else:
            denoise_scale = 1.0 - (denoise - 0.5) / 0.4 * (1.0 - 0.1)  # 1.0 to 0.1

        # Final scaled eta
        eta_final = eta_base * denoise_scale
        return max(0.0, min(eta_final, 1.0))  # Clamp to [0, 1]

    @staticmethod
    def Enrichment_Pipe():
        """
        Factory for a single-element enrichment-pipe list.
        Returns the same structure and defaults as the original.
        """
        Enrichment_Pipe = []
        Enrichment_Pipe.append({
            "PRO_Fusion_Complexity_min_Denoise" : 0,
            "PRO_Fusion_Complexity_max_Denoise": 1,
            "PRO_Fusion_Complexity_Mask_Blur": 0,
            "eta_start": 0,
            "eta_end": 0.5,
            "resharpen_start": 0,
            "resharpen_end": 0.2,
            "eta_active": False,
            "eta": 0,
            "Sigmas_creative_tail_active" : False,
            "Sigmas_creative_tail_strength" : 0,
            "Resharpener_strength" : 0,
            "Resharpener_active" : False,
            "detail_daemon_active": False,
            "detail_amount": 0,
            "detail_daemon_start": 0.3,
            "detail_daemon_end": 0.5,
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
            "RF_inversion": 0,
            "tile_upscale_plus": "none",
            "upscaler_method_inpainting": 'lanczos',
            "upscale_model_inpainting": None,
            "upscale_tiles_by": 1.5,
            "upscale_segments_by": 1.5,
        })
        return Enrichment_Pipe
