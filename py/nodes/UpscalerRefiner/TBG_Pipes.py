# Standard library imports
from types import SimpleNamespace
import json
import aiohttp
import PIL
from PIL import Image
import os
import glob

MAX_SIZE_THUMPNAIL = (256, 256)  # max width, height
MAX_SIZE_TILESIZESAVED = (4096, 4096)  # max width, height
import time
# Third-party imports
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 592515344
from PIL import Image
from aiohttp import web

# Application-specific imports
import folder_paths

from server import PromptServer


from ...vendor.ComfyUI_MaraScott_Nodes.py.inc.lib.cache import MS_Cache
from ...utils.log import log

# Relative imports - local module
from .inc.tileprompter_helpers import Node as NodePrompt

# Relative imports - root constants
from .... import root_dir
SERVER = os.getenv("COMFY_HOST", "127.0.0.1:8188")

class TBG_ControlNetPipeline:
    NAME = "TBG Tile ControlNet Pipe"
    PREPROCESSOR_OPTIONS = {
        'None': "None",
        'DepthAnythingV2': "DepthAnythingV2",
        'Canny Edge': "Canny Edge",
        'Canny': "Canny",
    }

    KONTEXT_OPTIONS = {
        'NONE': "NONE",
        'Stitched': "Stitched",
        'Chained': "Chained",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),  # Strength of ControlNet
                "start": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),  # Start influence
                "end": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),  # End influence
                "canny_low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),  # Low threshold
                "canny_high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),  # High threshold
                "preprocessor": (list(cls.PREPROCESSOR_OPTIONS.keys()),),  # Now required
                "patch_for_Flux_Kontext": (list(cls.KONTEXT_OPTIONS.keys()), {"label": "patch_for_Flux_Kontext", "default": 'NONE', }),  # Now required

            },
            "optional": {
                "controlnet": ("CONTROL_NET",),  # ControlNet model
                "Controlnet_Pipe": ("Controlnet_Pipe",),  # Incoming pipeline (empty or existing list)
                "custom_controlnet_image": ("IMAGE", {"label": "custom_controlnet_image"}),
            }
        }


    RETURN_TYPES = ("Controlnet_Pipe", "STRING")  # Added STRING for debugging
    RETURN_NAMES = ("Controlnet_Pipe", "INFO")  # Added STRING for debugging
    FUNCTION = "update_pipe"
    CATEGORY = "TBG/Enhanced Upscaler PRO"
    NAME = "TBG ETUR controlnet pipline"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An tile space controlnet wrapper for TBG ETUR'

    def update_pipe(
            self,
            controlnet=None,  # <- make it optional
            strength=0.5,
            start=0.0,
            end=0.5,
            canny_low_threshold=100,
            canny_high_threshold=150,
            preprocessor='None',
            patch_for_Flux_Kontext='NONE',
            custom_controlnet_image=None,
            Controlnet_Pipe=None
    ):        # Ensure pipe is a list
        if Controlnet_Pipe is None or not isinstance(Controlnet_Pipe, list):
            Controlnet_Pipe = []

        # Append new ControlNet settings
        Controlnet_Pipe.append({
            "controlnet": controlnet,
            "preprocessor": preprocessor,  # Now always required
            "strength": strength,
            "start": start,
            "end": end,
            "canny_low_threshold": canny_low_threshold,
            "canny_high_threshold": canny_high_threshold,
            "noise_image": custom_controlnet_image,
            "patch_for_Flux_Kontext" : patch_for_Flux_Kontext,
        })

        # Convert pipe to string for debugging
        pipe_str = str(Controlnet_Pipe)

        return Controlnet_Pipe, pipe_str


class TBG_enrichment_pipe:
    NAME = "TBG Tile Enrichment Pipe"

    INNERUPSCALE_METHODS = [
        'none',
        'finer details',
        'finer details + grain removal',
    ]

    UPSCALE_METHODS = [
        "area",
        "bicubic",
        "bilinear",
        "bislerp",
        "lanczos",
        "nearest-exact",
        "with model",
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Resharpener_active": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "Resharpener_strength": ("FLOAT", {"default": 0.5, "min": -5, "max": 5, "round": 0.01}),
                "resharpen_start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "round": 0.01}),
                "resharpen_end": ("FLOAT", {"default": 1, "min": 0, "max": 1, "round": 0.01}),
                "eta_active": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "eta": ("FLOAT", {"default": 0.60, "min": 0, "max": 1.20, "step": 0.01, "round": 0.01}),
                "eta_start": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "eta_end": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),


                "Sigmas_creative_tail_active": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "Sigmas_creative_tail_strength": ("FLOAT", {"default": 0, "min": -5.0, "max": 5.0, "step": 0.01, "round": 0.01}),

                "detail_daemon_active": ("BOOLEAN", {"label": "Use Detail Daemon", "default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "detail_amount": ("FLOAT", {"default": 1.17, "min": -5.0, "max": 5.0, "step": 0.01, "round": 0.01, "label": "Detail Amount"}),
                "detail_daemon_start": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "label": "Start Detail"}),
                "detail_daemon_end": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "label": "End Detail"}),
                "detail_daemon_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "detail_daemon_exponent": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "detail_daemon_start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "detail_daemon_end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "round": 0.01},),
                "detail_daemon_fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "detail_daemon_smooth": ("BOOLEAN", {"default": True}),
                "detail_daemon_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),






                "tile_upscale_plus": (cls.INNERUPSCALE_METHODS, {"label": "tile_upscale_plus", "default": 'none'}),
                "upscaler_method_inpainting": (cls.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'lanczos'}),
                "upscale_model_inpainting": (folder_paths.get_filename_list("upscale_models"),
                                             {"label": "Upscale Model"}),
                "upscale_tiles_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 2, "step": 0.1,
                                               "round": 0.1}),
                "upscale_segments_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 2, "step": 0.1,
                                                  "round": 0.1}),
                "PRO_Fusion_Complexity_min_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 0.3, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the min denoise level mainly on low-complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),
                "PRO_Fusion_Complexity_max_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 1, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the mx denoise level mainly on high complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),

                "PRO_Fusion_Complexity_Mask_Blur": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                              "tooltip": "apply blur and adjust the sensitivity of the Complexity Mask"}),

            },
            "optional": {
            },
            "hidden": {
                 "Sampler_side_noise_injection": ("FLOAT", {"label": "ETA", "default": 0.00, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                 "SplitStepsSigmasCurve": ("SIGMAS", {"label": "SplitStepsSigmasCurve"}),
                 "RF_inversion": ("FLOAT", {"label": "RF_inversion", "default": 0, "min": 0, "max": 1, "step": 0.1, "round": 0.1}),
            }
        }


    RETURN_TYPES = ("Enrichment_Pipe", "STRING")  # Added STRING for debugging
    RETURN_NAMES = ("Enrichment_Pipe", "INFO")
    FUNCTION = "update_pipe"
    CATEGORY = "TBG/Enhanced Upscaler PRO"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An detail enhancer for for TBG ETUR'

    def update_pipe(self, detail_daemon_active, detail_amount, detail_daemon_start, detail_daemon_end, detail_daemon_bias,
                    detail_daemon_exponent, detail_daemon_start_offset, detail_daemon_end_offset,
                    detail_daemon_fade, detail_daemon_smooth, detail_daemon_cfg_scale,
                    tile_upscale_plus,upscaler_method_inpainting,upscale_model_inpainting, upscale_tiles_by,upscale_segments_by,
                    Resharpener_strength, Resharpener_active, Sigmas_creative_tail_active, Sigmas_creative_tail_strength,eta_active,eta,
                    eta_start,eta_end,resharpen_start,resharpen_end,PRO_Fusion_Complexity_Mask_Blur,PRO_Fusion_Complexity_max_Denoise,PRO_Fusion_Complexity_min_Denoise
                    ):


        Enrichment_Pipe = []
        SplitStepsSigmasCurve = None
        latentupscale = False
        latentupscale_noise = 0
        latentupscale_steps = 10
        latentupscale_denoise = 0
        SplitSteps =  False
        SplitSteps_noise = 0.7
        SplitSteps_steps = 5
        SplitStepsSigmas = False
        SplitStepsMultiplyer = 0.7
        SplitStepsStart = 5
        SplitStepsEnd = 5
        RF_inversion = 0

        if Resharpener_active == False:
            Resharpener_strength = 0
        if eta_active == False:
            eta = 0

        # Append new ControlNet settings
        Enrichment_Pipe.append({
            "eta_start": eta_start,
            "eta_end": eta_end,
            "resharpen_start": resharpen_start,
            "resharpen_end": resharpen_end,
            "Sigmas_creative_tail_active":Sigmas_creative_tail_active,
            "Sigmas_creative_tail_strength":Sigmas_creative_tail_strength,
            "Resharpener_strength": Resharpener_strength,
            "Resharpener_active": Resharpener_active,
            "detail_daemon_active": detail_daemon_active,
            "detail_amount": detail_amount,
            "detail_daemon_start": detail_daemon_start,
            "detail_daemon_end": detail_daemon_end,
            "detail_daemon_bias": detail_daemon_bias,
            "detail_daemon_exponent": detail_daemon_exponent,
            "detail_daemon_start_offset": detail_daemon_start_offset,
            "detail_daemon_end_offset": detail_daemon_end_offset,
            "detail_daemon_fade": detail_daemon_fade,
            "detail_daemon_smooth": detail_daemon_smooth,
            "detail_daemon_cfg_scale": detail_daemon_cfg_scale,
            "latentupscale": latentupscale,
            "latentupscale_noise": latentupscale_noise,
            "latentupscale_steps": latentupscale_steps,
            "latentupscale_denoise": latentupscale_denoise,
            "SplitSteps": SplitSteps,
            "SplitSteps_noise": SplitSteps_noise,
            "SplitSteps_steps": SplitSteps_steps,
            "SplitStepsSigmas": SplitStepsSigmas,
            "SplitStepsMultiplyer": SplitStepsMultiplyer,
            "SplitStepsSigmasCurve": SplitStepsSigmasCurve,
            "SplitStepsStart": SplitStepsStart,
            "SplitStepsEnd": SplitStepsEnd,
            "eta_active": eta_active,
            "eta": eta,
            "RF_inversion": RF_inversion,
            "tile_upscale_plus": tile_upscale_plus,
            "upscaler_method_inpainting": upscaler_method_inpainting,
            "upscale_model_inpainting": upscale_model_inpainting,
            "upscale_tiles_by": upscale_tiles_by,
            "upscale_segments_by": upscale_segments_by,
            "PRO_Fusion_Complexity_Mask_Blur":PRO_Fusion_Complexity_Mask_Blur,
            "PRO_Fusion_Complexity_max_Denoise":PRO_Fusion_Complexity_max_Denoise,
            "PRO_Fusion_Complexity_min_Denoise":PRO_Fusion_Complexity_min_Denoise


        })
        pipe_str = str(Enrichment_Pipe)


        return Enrichment_Pipe, pipe_str  # Return both the structure and the string version



"""
 /*
 * McBoaty Tile-Prompter – rebuilt class TBG_TilePrompter_v1(): version for TGB enhanced upscaler and refiner pro
 *
 * Copyright (c) 2025 Tobias Laarmann
 * Copyright (c) 2024 David Asquiedge
 *
 * This file is a derivative of the original “McBoaty_v5.js” from
 * https://github.com/MaraScott/ComfyUI_MaraScott_Nodes. Copyright (c) 2024 David Asquiedge
 *
 * Released under the MIT License;
 *
 *  MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
**Attribution is required. The use of this software must be accompanied by proper
credit to the original author.**

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
// Attribution complies with the MIT licence by retaining this header.
"""

class TBG_TilePrompter_v1():
    NAME = "TBG Tile Prompt Pipe"

    @classmethod
    def INPUT_TYPES(self):
        # Minimal hidden schema: unique id + one JSON blob carrying all edits
        hidden_entries = {
            "id": "UNIQUE_ID",
            **NodePrompt.ENTRIES_CONFIG,  # tile_edits_json
        }
        return {
            "hidden": hidden_entries,
            "required": {
                "Tile_Prompt_Pipe": ("Tile_Prompt_Pipe", {"label": "Tile Prompt Pipe"}),
            },
            "optional": {
                "requeue": ("INT", {
                    "label": "requeue (automatic or manual)",
                    "default": 0, "min": 0, "max": 99999999999, "step": 1
                }),
            },
        }

    RETURN_TYPES = ("TILE_Prompt_PIPE_OUT",)
    RETURN_NAMES = ("Tile_Prompt_Pipe",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True
    CATEGORY = "TBG/Enhanced Upscaler"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An note to set per tile sampler settings for TBG ETUR'
    FUNCTION = "fn"
    #EXECUTION_CACHING = False

    @classmethod
    async def _fetch_tile_edits_json(node_id: int):
        url = f"http://{SERVER}/TBG/McBoaty/v5/get_tile_edits_json?node={node_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("tile_edits_json")
        except Exception as e:
            print("fetch tile_edits_json failed:", e)
        return None

    @classmethod
    def init(self, **kwargs):
        self.INFO = SimpleNamespace(
            id=kwargs.get('id', 0),
        )
        self.CACHE = SimpleNamespace(
            prompt=f'input_prompts_{self.INFO.id}',
            prompt_edited=None,
            denoise=f'input_denoises_{self.INFO.id}',
            denoise_edited=None,
            seeds=f'input_seeds_{self.INFO.id}',
            seeds_edited=None,
            cnet=f'input_cnet_strength_{self.INFO.id}',
            cnet_edited=None,
        )
        self.CACHE.prompt_edited = f'{self.CACHE.prompt}_edited'
        self.CACHE.denoise_edited = f'{self.CACHE.denoise}_edited'
        self.CACHE.seeds_edited = f'{self.CACHE.seeds}_edited'
        self.CACHE.cnet_edited = f'{self.CACHE.cnet}_edited'
        self.output_dir = folder_paths.get_temp_directory()
        log(self.output_dir, None, None, f"Node {self.INFO.id}")


    def fn(self, **kwargs):


        # Unpack incoming pipe (Tile_Prompt_Pipe carries tuples: (prompts, tiles, segment_tiles))
        input_prompts, input_tiles, segment_tiles = kwargs.get('Tile_Prompt_Pipe', (None, None))

        base_len = len(input_prompts) if input_prompts is not None else 0
        length = base_len
        # Init node state
        self.init(**kwargs)


        # First get all Values from Workflow
            # we can try async
            # we can try MS_Cache Json

        # --- BEGIN: Only-prompts merge into tile_edits_json so UI shows new prompts ---
        try:
            node_id = self.INFO.id
            cache_key = f"tile_edits_json_{node_id}"

            # Load existing JSON (keep non-prompt fields)
            existing = MS_Cache.get(cache_key, None)

            if isinstance(existing, str) and existing.strip():
                try:
                    obj = json.loads(existing)
                except Exception:
                    obj = {}
            else:
                obj = {}

            den_old = obj.get("denoises") or []
            seeds_old = obj.get("seeds") or []
            cnet_old = obj.get("cnet_strength") or []
            prev_prompts = obj.get("prompts") or []

            # Use your computed tile count `length` and your final prompts `input_prompts`
            def _normalize(arr, n):
                arr = list(arr or [])
                if len(arr) < n:
                    arr.extend([""] * (n - len(arr)))
                elif len(arr) > n:
                    arr = arr[:n]
                return arr

            new_prompts = _normalize(input_prompts, length)
            den_old = _normalize(den_old, length)
            seeds_old = _normalize(seeds_old, length)
            cnet_old = _normalize(cnet_old, length)
            prev_prompts_norm = _normalize(prev_prompts, length)

            prompts_changed = any((prev_prompts_norm[i] or "") != (new_prompts[i] or "") for i in range(length))

            # Only write if something actually changed (and if there is at least one non-empty prompt)
            if prompts_changed and any((p or "").strip() for p in new_prompts):

                obj["prompts"] = new_prompts
                obj["denoises"] = den_old
                obj["seeds"] = seeds_old
                obj["cnet_strength"] = cnet_old
                MS_Cache.set(cache_key, json.dumps(obj))


        except Exception as e:
            print("TilePrompter only-prompts merge failed:", e)
        # --- END: Only-prompts merge ---




        log("TBG (TBG Tile Prompt Pipe) starting", None, None, f"Node {self.INFO.id}")

        # Loader for tile_edits_json (kwargs -> MS_Cache), returns parsed dict or {}
        def _load_tile_edits_json(_kwargs, node_id):
            # 1) Hidden input (kwargs)
            data_json = _kwargs.get("tile_edits_json")
            if isinstance(data_json, str) and data_json.strip():
                try:
                    return json.loads(data_json), "KWARGS"
                except Exception as e:
                    print("tile_edits_json kwargs parse error:", e, data_json[:120])
            # 2) MS_Cache mirror
            try:
                cached = MS_Cache.get(f"tile_edits_json_{node_id}", None)
            except Exception as e:
                print("tile_edits_json cache read error:", e)
                cached = None
            if isinstance(cached, str) and cached.strip():
                try:
                    return json.loads(cached), "CACHE"
                except Exception as e:
                    print("tile_edits_json cache parse error:", e, cached[:120])
            # Nothing usable
            return {}, None

        # Use it
        json_obj, src = _load_tile_edits_json(kwargs, self.INFO.id)


        #if src is None:
        #    print("Workflow TBG settings not available yet (first run likely). Proceeding with empty edits. Hit run twice first tu update")
        #else:
        #    print(f"Workflow TBG settings loaded via {src}")

        # Extract arrays with defaults
        prompts_js = json_obj.get("prompts") or []
        denoises_js = json_obj.get("denoises") or []
        seeds_js = json_obj.get("seeds") or []
        cnet_js = json_obj.get("cnet_strength") or []

        # Decide length from tiles: SINGLE SOURCE OF TRUTH
        tiles_len = 0
        if input_tiles:
            try:
                tiles_len += len([t[0] for t in input_tiles])
            except Exception:
                # If input_tiles is already a flat list of tensors/images
                tiles_len += len(input_tiles)
        if segment_tiles:
            tiles_len += len(segment_tiles)

        # Fallback if no tiles are present yet
        if tiles_len <= 0:
            tiles_len = max(
                base_len,
                len(prompts_js),
                len(denoises_js),
                len(seeds_js),
                len(cnet_js),
            )

        length = tiles_len

        # Normalization helper
        def _normalize(arr, n):
            arr = list(arr or [])
            if len(arr) < n:
                arr = arr + [""] * (n - len(arr))
            elif len(arr) > n:
                arr = arr[:n]
            return arr

        # Start from inputs (preserve), normalize to tile length
        edited_prompts = _normalize(input_prompts or [], length)
        edited_denoises = _normalize([], length)  # no incoming denoise in pipe
        edited_seeds = _normalize([], length)
        edited_cnet = _normalize([], length)

        # Safe getter and override policy
        def _safe_get(arr, idx):
            try:
                return arr[idx]
            except Exception:
                return None

        def _has_value(x):
            # Policy: treat empty string as "no override"? Set to False if not.
            # If empty string should override, return (x is not None)
            return x is not None and x != ""

        # Normalize JS arrays to simplify overlay
        prompts_js = _normalize(prompts_js, length)
        denoises_js = _normalize(denoises_js, length)
        seeds_js = _normalize(seeds_js, length)
        cnet_js = _normalize(cnet_js, length)

        # Overlay JS edits safely
        for i in range(length):
            v = _safe_get(prompts_js, i)
            if _has_value(v):
                edited_prompts[i] = v

            v = _safe_get(denoises_js, i)
            if _has_value(v):
                edited_denoises[i] = v

            v = _safe_get(seeds_js, i)
            if _has_value(v):
                edited_seeds[i] = v

            v = _safe_get(cnet_js, i)
            if _has_value(v):
                edited_cnet[i] = v

        # Echo arrays for UI
        output_prompts = edited_prompts
        output_denoises = edited_denoises
        output_seeds_js = edited_seeds
        output_cnet_js = edited_cnet

        input_prompts_js = _normalize(input_prompts or [], length)
        input_denoises_js = _normalize([], length)
        input_seeds_js = _normalize([], length)
        input_cnet_js = _normalize([], length)
# test to add promt
        MS_Cache.set(self.CACHE.prompt, tuple(input_prompts_js))



        # Save tile previews
        results = []
        filename_prefix = f"TBG_temp_tilePrompter_id_{self.INFO.id}"
        tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
        pattern = os.path.join(tbg_temp_dir, filename_prefix + "*")

        # Clean previous temp files for this node
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

        # Save input tiles
        if input_tiles:
            # Flatten if needed
            try:
                flat_tiles = [t[0] for t in input_tiles]
            except Exception:
                flat_tiles = input_tiles

            for idx, torchtile in enumerate(flat_tiles):
                # Expect Comfy tensors in CHW in [0..1]
                try:
                    arr = torchtile.squeeze(0).cpu().numpy() * 255
                except Exception:
                    # If it's already a numpy array or PIL.Image-like
                    try:
                        arr = np.asarray(torchtile)  # fallback
                    except Exception:
                        continue
                """
                full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                    f"TBG/thumbnail_{self.INFO.id}/{filename_prefix}", self.output_dir, arr.shape[1], arr.shape
                )
                fname = f"{base}_{idx:05}.png"
                os.makedirs(full_folder, exist_ok=True)
                path = os.path.join(full_folder, fname)
                if not os.path.exists(path):
                    img = Image.fromarray(arr.clip(0, 255).astype('uint8'))
                    img.thumbnail(MAX_SIZE_THUMPNAIL, Image.Resampling.LANCZOS)
                    img.save(path, compress_level=4)
                results.append({"filename": fname, "subfolder": sub, "type": "temp"})


                full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                    f"TBG/tiles_{self.INFO.id}/{filename_prefix}", self.output_dir, arr.shape[1], arr.shape
                )
                fname = f"{base}_{idx:05}.png"
                os.makedirs(full_folder, exist_ok=True)
                path = os.path.join(full_folder, fname)
                if not os.path.exists(path):
                    img = Image.fromarray(arr.clip(0, 255).astype('uint8'))
                    img.thumbnail(MAX_SIZE_TILESIZESAVED, Image.Resampling.LANCZOS)
                    img.save(path, compress_level=4)
                """
                full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                    f"TBG/thumbnail_{self.INFO.id}/{filename_prefix}", self.output_dir, arr.shape[1], arr.shape
                )

                # Delete old files before saving
                if idx == 0:  # only clear once per batch
                    for f in glob.glob(os.path.join(full_folder, "*.png")):
                        os.remove(f)

                os.makedirs(full_folder, exist_ok=True)

                # Create unique filename (timestamp or random suffix)
                unique_suffix = int(time.time() * 1000)  # milliseconds
                fname = f"{base}_{idx:05}_{unique_suffix}.png"
                path = os.path.join(full_folder, fname)

                img = Image.fromarray(arr.clip(0, 255).astype('uint8'))
                img.thumbnail(MAX_SIZE_THUMPNAIL, Image.Resampling.LANCZOS)
                img.save(path, compress_level=4)

                results.append({"filename": fname, "subfolder": sub, "type": "temp"})

                # --- Second block: tiles ---
                full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                    f"TBG/tiles_{self.INFO.id}/{filename_prefix}", self.output_dir, arr.shape[1], arr.shape
                )

                if idx == 0:
                    for f in glob.glob(os.path.join(full_folder, "*.png")):
                        os.remove(f)

                os.makedirs(full_folder, exist_ok=True)

                unique_suffix = int(time.time() * 1000)
                fname = f"{base}_{idx:05}_{unique_suffix}.png"
                path = os.path.join(full_folder, fname)

                img = Image.fromarray(arr.clip(0, 255).astype('uint8'))
                img.thumbnail(MAX_SIZE_TILESIZESAVED, Image.Resampling.LANCZOS)
                img.save(path, compress_level=4)

        # Save segment tiles
        if segment_tiles:
            for j, torchtile in enumerate(segment_tiles):
                try:
                    arr = torchtile.squeeze(0).cpu().numpy() * 255
                except Exception:
                    try:
                        arr = np.asarray(torchtile)
                    except Exception:
                        continue

                idx2 = (len(input_tiles) if input_tiles else 0) + j
                full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                    f"TBG/{filename_prefix}", self.output_dir, arr.shape[1], arr.shape
                )
                fname = f"{base}_{idx2:05}.png"
                os.makedirs(full_folder, exist_ok=True)
                path = os.path.join(full_folder, fname)
                if not os.path.exists(path):
                    img = Image.fromarray(arr.clip(0, 255).astype('uint8'))
                    img.thumbnail(MAX_SIZE_THUMPNAIL, Image.Resampling.LANCZOS)
                    img.save(path, compress_level=4)
                    #Image.fromarray(arr.clip(0, 255).astype('uint8')).save(path, compress_level=4)
                results.append({"filename": fname, "subfolder": sub, "type": "temp"})

        log("TBG (TBG Tile Prompt Pipe) done", None, None, f"Node {self.INFO.id}")



        return {
            "ui": {
                "prompts_out": output_prompts,
                "prompts_in": input_prompts_js,
                "denoises_out": output_denoises,
                "denoises_in": input_denoises_js,
                "seeds_out": output_seeds_js,
                "seeds_in": input_seeds_js,
                "cnet_strength_out": output_cnet_js,
                "cnet_strength_in": input_cnet_js,
                "tiles": results,
            },
            "result": ((output_prompts, output_denoises, output_seeds_js, output_cnet_js),)
        }


# --- Routes (keep these once in the file; do not duplicate) ---

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_tile_edits_json")
async def get_tile_edits_json(request):
    node_id = request.query.get("node")
    if not node_id:
        return web.json_response({"tile_edits_json": None})
    try:
        cached = MS_Cache.get(f"tile_edits_json_{node_id}", None)
        return web.json_response({"tile_edits_json": cached})
    except Exception as e:
        return web.json_response({"tile_edits_json": None, "error": str(e)})

@PromptServer.instance.routes.post("/TBG/McBoaty/v5/set_tile_edits_json")
async def set_tile_edits_json(request):
    try:
        payload = await request.json()
    except:
        payload = {}
    node_id = str(payload.get("node") or "")
    data = payload.get("tile_edits_json")
    if not node_id:
        return web.json_response({"ok": False, "error": "missing node"})
    try:
        MS_Cache.set(f"tile_edits_json_{node_id}", data)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_seeds")
async def get_input_seeds(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_seeds_{nodeId}'
    seeds = MS_Cache.get(cache_name, [])
    return web.json_response({"seeds_in": seeds})

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_cnet_strength")
async def get_input_cnet_strength(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_cnet_strength_{nodeId}'
    cnet = MS_Cache.get(cache_name, [])
    return web.json_response({"cnet_strength_in": cnet})

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = MS_Cache.get(cache_name, [])
    return web.json_response({"prompts_in": input_prompts})

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = MS_Cache.get(cache_name, [])
    return web.json_response({"denoises_in": input_denoises})

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_prompt")
async def set_prompt(request):
    prompt = request.query.get("prompt", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_prompts = MS_Cache.get(cache_name, [])
    _input_prompts_edited = MS_Cache.get(cache_name_edited, _input_prompts)
    if _input_prompts_edited and 0 <= index < len(_input_prompts_edited):
        lst = list(_input_prompts_edited)
        lst[index] = prompt
        MS_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} prompt has been updated :{prompt}")

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_denoise")
async def set_denoise(request):
    denoise = request.query.get("denoise", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_denoises = MS_Cache.get(cache_name, [])
    _input_denoises_edited = MS_Cache.get(cache_name_edited, _input_denoises)
    if _input_denoises_edited and 0 <= index < len(_input_denoises_edited):
        lst = list(_input_denoises_edited)
        lst[index] = denoise
        MS_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} denoise has been updated: {denoise}")

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_seed")
async def set_seed(request):
    seed = request.query.get("seed", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_seeds_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _seeds = MS_Cache.get(cache_name, [])
    _seeds_edited = MS_Cache.get(cache_name_edited, _seeds)
    if _seeds_edited and 0 <= index < len(_seeds_edited):
        lst = list(_seeds_edited)
        lst[index] = seed
        MS_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} seed has been updated: {seed}")

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_cnet_strength")
async def set_cnet_strength(request):
    cnet_strength = request.query.get("cnet_strength", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_cnet_strength_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _cnet = MS_Cache.get(cache_name, [])
    _cnet_edited = MS_Cache.get(cache_name_edited, _cnet)
    if _cnet_edited and 0 <= index < len(_cnet_edited):
        lst = list(_cnet_edited)
        lst[index] = cnet_strength
        MS_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} cnet_strength has been updated: {cnet_strength}")

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/tile_prompt")
async def tile_prompt(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)
    _type = request.query.get("type", "output")
    if _type not in ["output", "input", "temp"]:
        return web.Response(status=400)
    target_dir = os.path.join(root_dir, _type)
    image_path = os.path.abspath(os.path.join(
        target_dir,
        request.query.get("subfolder", ""),
        request.query["filename"]
    ))

    c = os.path.commonpath((image_path, target_dir))
    if c != target_dir:
        return web.Response(status=403)
    if not os.path.isfile(image_path):
        return web.Response(status=404)
    return web.json_response(f"here is the prompt \n{image_path}")

