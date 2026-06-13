# Standard library imports
from types import SimpleNamespace
import json
import aiohttp
import PIL
from PIL import Image
import os
import glob
import requests
import json
import copy
import shutil
import hashlib
import torch

MAX_SIZE_THUMBNAIL = (256, 256)  # max width, height
MAX_SIZE_TILESIZESAVED = (4096, 4096)  # max width, height
TILE_MODEL_OVERRIDE_REGISTRY = {}
TILE_CNETPIPE_OVERRIDE_REGISTRY = {}
# Third-party imports
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = 592515344
from PIL import Image
from aiohttp import web
import folder_paths
from .inc.tp_cache import Tile_Prompter_Cache
from .inc.batch import attach_tile_overrides_to_batch_pipe, is_batch_pipe
from ...utils.log import log
from .inc.tileprompter_helpers import Node as NodePrompt
from .... import root_dir

# WORKER
from ....TBG.SERVERS.WORKER_server import WORKER
from ....TBG.CALLBACKS.constants import TBGState, get_tbg
from server import PromptServer
SERVER = os.getenv("COMFY_HOST", "127.0.0.1:8188")
"""
from server import PromptServer
def get_comfy_server_url():
    server = PromptServer.instance
    if not hasattr(server, "address"):
        return None  # server not ready yet

    host = server.address
    if host == "0.0.0.0":
        host = "127.0.0.1"

    return (host,f"{host}:{server.port}") # without http
"""
class TBG_ControlNetPipeline:

    PREPROCESSOR_OPTIONS = {
        'None': "None",
        'DepthAnythingV2': "DepthAnythingV2",
        'Canny Edge': "Canny Edge",
        'Canny': "Canny",
        'ControlNetInpaintingAliMama': "ControlNetInpaintingAliMama",
    }

    CONTROLNET_MODE_OPTIONS = {
        'ControlNet': "ControlNet",
        'Reference_Image': "Reference_Image",
        'Input Tile CFG Hook': "Input Tile CFG Hook",
        '42lux-Hildegard Ref.Img': "42lux-Hildegard Ref.Img",
        '42lux-Hildegard Ref.Img + CFG Hook': "42lux-Hildegard Ref.Img + CFG Hook",
        'Model_Patch': "Model_Patch",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "tooltip": "ControlNet strength for ControlNet modes. For Input Tile CFG Hook, controls the post-CFG pull toward the sampler input tile latent. Ignored by Reference_Image mode."}),
                "start": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "tooltip": "Start percentage for ControlNet or Input Tile CFG Hook influence. Ignored by Reference_Image mode."}),
                "end": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "tooltip": "End percentage for ControlNet or Input Tile CFG Hook influence. Ignored by Reference_Image mode."}),
                "canny_low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "canny_high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),
                "preprocessor": (list(cls.PREPROCESSOR_OPTIONS.keys()),),
            },
            "optional": {
                "controlnet_mode": (list(cls.CONTROLNET_MODE_OPTIONS.keys()), {"label": "controlnet_mode", "default": 'ControlNet'}),
                "controlnet": ("CONTROL_NET",),
                "model_patch": ("MODEL_PATCH",),
                "Controlnet_Pipe": ("Controlnet_Pipe",),
                "custom_controlnet_image": ("IMAGE", {"label": "custom_controlnet_image"}),
            }
        }

    RETURN_TYPES = ("Controlnet_Pipe", "STRING")
    RETURN_NAMES = ("Controlnet_Pipe", "INFO")
    FUNCTION = "update_pipe"
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"

    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An tile space controlnet wrapper for TBG ETUR'

    def update_pipe(
        self,
        controlnet=None,
        strength=0.5,
        start=0.0,
        end=0.5,
        canny_low_threshold=100,
        canny_high_threshold=150,
        preprocessor='None',
        controlnet_mode='NONE',
        custom_controlnet_image=None,
        model_patch=None,
        Controlnet_Pipe=None
    ):
        if Controlnet_Pipe is None or not isinstance(Controlnet_Pipe, list):
            Controlnet_Pipe = []

        # Keep the legacy key populated so older refiner paths keep working.
        legacy_kontext_mode = "Chained" if controlnet_mode == "Reference_Image" else "NONE"
        Controlnet_Pipe.append({
            "controlnet": controlnet,
            "model_patch": model_patch,
            "controlnet_mode": controlnet_mode,
            "mode": controlnet_mode,
            "preprocessor": preprocessor,
            "strength": strength,
            "start": start,
            "end": end,
            "canny_low_threshold": canny_low_threshold,
            "canny_high_threshold": canny_high_threshold,
            "noise_image": custom_controlnet_image,
            "patch_for_Flux_Kontext": legacy_kontext_mode,
        })

        pipe_str = str(Controlnet_Pipe)
        return Controlnet_Pipe, pipe_str
class TBG_enrichment_pipe:


    INNERUPSCALE_METHODS = [
        'none',
        'finer details',
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Resharpener_active": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "Resharpener_strength": ("FLOAT", {"default": 0.5, "min": -5, "max": 5, "round": 0.01}),
                "resharpen_start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "round": 0.01}),
                "resharpen_end": ("FLOAT", {"default": 1, "min": 0, "max": 1, "round": 0.01}),
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
                "upscaler_method_inpainting": (cls.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'bilinear'}),
                "upscale_model_inpainting": (folder_paths.get_filename_list("upscale_models"),
                                             {"label": "Upscale Model"}),
                "upscale_tiles_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 4, "step": 0.1,
                                               "round": 0.1}),
                "upscale_segments_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 4, "step": 0.1,
                                                  "round": 0.1}),
                "PRO_Fusion_Complexity_min_Denoise": ("FLOAT", {"label": "Complexity Min Denoise", "default": 0.2, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the min denoise level mainly on low-complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),
                "PRO_Fusion_Complexity_max_Denoise": ("FLOAT", {"label": "Complexity Max Denoise", "default": 1, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the mx denoise level mainly on high complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),
                "PRO_Fusion_Complexity_Mask_Blur": ("FLOAT", {"label": "Complexity Mask Blur", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                              "tooltip": "apply blur and adjust the sensitivity of the Complexity Mask"}),

            },
            "optional": {
            },
            "hidden": {
            }
        }


    RETURN_TYPES = ("Enrichment_Pipe", "STRING")  # Added STRING for debugging
    RETURN_NAMES = ("Enrichment_Pipe", "INFO")
    FUNCTION = "fn"
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An detail enhancer for for TBG ETUR'

    @classmethod
    def fn(cls, **kwargs):
        return (kwargs,)


class TBG_RF_UntwistingRoPE_Pipe:
    RF_MODES = [
        "linear",
        "rf_gamma",
        "rf_gamma_rk2",
        "rf_solver_2",
        "endpoint_heun",
        "fireflow",
        "flowturbo_pc",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": "Default off. When enabled, ETUR applies its bundled RF Inversion + UntwistingRoPE runtime per tile/segment.",
                }),
                "rf_mode": (cls.RF_MODES, {"default": "rf_gamma_rk2"}),
                "gamma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pmi_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "otip_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "otip_clip_norm": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 50.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "high_scale_start": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 8.0, "step": 0.01}),
                "high_scale_end": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 8.0, "step": 0.01}),
                "low_scale_start": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 8.0, "step": 0.01}),
                "low_scale_end": ("FLOAT", {"default": 3.0, "min": -4.0, "max": 8.0, "step": 0.01}),
                "adain_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blocks": ("STRING", {"default": "0-999"}),
                "rf_verbose": ("BOOLEAN", {"default": False}),
                "untwisting_verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "unofficial_extensions": ("UNTWISTING_ROPE_EXTENSIONS",),
            },
        }

    RETURN_TYPES = ("RF_UntwistingRoPE_Pipe", "STRING")
    RETURN_NAMES = ("RF_UntwistingRoPE", "INFO")
    FUNCTION = "fn"
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    DESCRIPTION = "Optional per-tile adapter settings for ETUR's bundled RF Inversion + UntwistingRoPE runtime."

    @classmethod
    def fn(cls, **kwargs):
        pipe = dict(kwargs)
        pipe["enabled"] = bool(pipe.get("enabled", False))
        return pipe, str({k: v for k, v in pipe.items() if k != "unofficial_extensions"})


"""
 /*
 * TBG_TilePrompter_v1(): version for TGB enhanced upscaler and refiner pro
 *
 * Copyright (c) 2025 Tobias Laarmann
 *
 * This class TBG_TilePrompter_v1() is a derivative of the original “McBoaty_v5.js” from
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
    TILE_EDIT_JSON_KEYS = (
        "prompts",
        "denoises",
        "seeds",
        "cnet_strength",
        "cfg_overrides",
        "model_overrides",
        "cnetpipe_overrides",
        "color_match_overrides",
        "ignore_general_prompts",
    )
    IS_CHANGED_RUNTIME_KEYS = {
        "id",
        "TBG_Pipe",
        "Model_1",
        "Model_2",
        "Model_3",
        "CnetPipe_1",
        "CnetPipe_2",
        "CnetPipe_3",
    }

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
                "TBG_Pipe": ("TBG_Pipe", {"label": "Tile Prompt Pipe"}),
                "Rebuild_only_modified_tiles": ("BOOLEAN", {"label": "rebuild only modified tiles", "default": True, "label_on": "On",
                                           "label_off": "Off", "tooltip": "Experimental: Detects user input changes and reprocesses only affected tiles instead of the full image. Typical speed gains up to 90%"}),

            },
            "optional": {
                "Model_1": ("MODEL", {"label": "model 1 override"}),
                "Model_2": ("MODEL", {"label": "model 2 override"}),
                "Model_3": ("MODEL", {"label": "model 3 override"}),
                "CnetPipe_1": ("Controlnet_Pipe", {"label": "cnetpipe 1 override"}),
                "CnetPipe_2": ("Controlnet_Pipe", {"label": "cnetpipe 2 override"}),
                "CnetPipe_3": ("Controlnet_Pipe", {"label": "cnetpipe 3 override"}),
                "requeue": ("INT", {
                    "label": "requeue (automatic or manual)",
                    "default": 0, "min": 0, "max": 99999999999, "step": 1
                }),
            },
        }



    RETURN_TYPES = ("TBG_Pipe",)
    RETURN_NAMES = ("TBG_Pipe",)
    OUTPUT_IS_LIST =(False,)
    OUTPUT_NODE = True
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://www.patreon.com/c/TB_LAAR"
    DESCRIPTION = 'An note to set per tile sampler settings for TBG ETUR'
    FUNCTION = "fn"

    @staticmethod
    def _json_value_has_content(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip() != ""

    @classmethod
    def _tile_edits_has_content(cls, obj):
        if not isinstance(obj, dict):
            return False
        return any(
            any(cls._json_value_has_content(v) for v in (obj.get(key) or []))
            for key in cls.TILE_EDIT_JSON_KEYS
        )

    @classmethod
    def _canonical_tile_edits_json(cls, data_json):
        if not isinstance(data_json, str) or not data_json.strip():
            return ""
        try:
            obj = json.loads(data_json)
        except Exception:
            return data_json.strip()
        if not isinstance(obj, dict):
            return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        obj = dict(obj)
        obj.pop("tiles", None)
        if not cls._tile_edits_has_content(obj):
            return ""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def _is_fingerprint_primitive(cls, value):
        return isinstance(value, (str, int, float, bool)) or value is None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint = {}
        for key, value in sorted(kwargs.items()):
            if key in cls.IS_CHANGED_RUNTIME_KEYS:
                continue
            if key == "tile_edits_json":
                fingerprint[key] = cls._canonical_tile_edits_json(value)
            elif cls._is_fingerprint_primitive(value):
                fingerprint[key] = value
        payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    async def _fetch_tile_edits_json(node_id: int):
        #HOST, SERVER = get_comfy_server_url()
        url = f"http://{SERVER}/TBG/get_tile_edits_json?node={node_id}"
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
        global tbg, tiler_id
        tiler_id = kwargs.get('TBG_Pipe')[10]
        tbg = get_tbg(tiler_id)

        tbg.INFO.id=kwargs.get('id', 0)
        tbg.CACHE.prompt=f'input_prompts_{tbg.INFO.id}'
        tbg.CACHE.prompt_edited=None,
        tbg.CACHE.denoise=f'input_denoises_{tbg.INFO.id}'
        tbg.CACHE.denoise_edited=None,
        tbg.CACHE.seeds=f'input_seeds_{tbg.INFO.id}'
        tbg.CACHE.seeds_edited=None,
        tbg.CACHE.cnet=f'input_cnet_strength_{tbg.INFO.id}'
        tbg.CACHE.cnet_edited=None
        tbg.CACHE.prompt_edited = f'{tbg.CACHE.prompt}_edited'
        tbg.CACHE.denoise_edited = f'{tbg.CACHE.denoise}_edited'
        tbg.CACHE.seeds_edited = f'{tbg.CACHE.seeds}_edited'
        tbg.CACHE.cnet_edited = f'{tbg.CACHE.cnet}_edited'
        tbg.temp_output_dir = folder_paths.get_temp_directory()


    def fn(self, **kwargs):
        pipe = kwargs.get('TBG_Pipe', (None, None))
        if is_batch_pipe(pipe):
            batch_kwargs = dict(kwargs)
            if not batch_kwargs.get("tile_edits_json"):
                try:
                    cached_json = Tile_Prompter_Cache.get(f"tile_edits_json_{kwargs.get('id', None)}", None)
                    if cached_json:
                        batch_kwargs["tile_edits_json"] = cached_json
                except Exception:
                    pass
            batch_pipe = attach_tile_overrides_to_batch_pipe(pipe, batch_kwargs, kwargs.get("id", None))
            log(
                "TBG Tile Overrides batch mode: storing override config without generating tile previews",
                None,
                None,
                f"Node {kwargs.get('id', None)}",
            )
            return {
                "ui": {
                    "prompts_out": [],
                    "prompts_in": [],
                    "denoises_out": [],
                    "denoises_in": [],
                    "seeds_out": [],
                    "seeds_in": [],
                    "cnet_strength_out": [],
                    "cnet_strength_in": [],
                    "cfg_overrides_out": [],
                    "cfg_overrides_in": [],
                    "model_overrides_out": [],
                    "model_overrides_in": [],
                    "cnetpipe_overrides_out": [],
                    "cnetpipe_overrides_in": [],
                    "color_match_overrides_out": [],
                    "color_match_overrides_in": [],
                    "ignore_general_prompts_out": [],
                    "ignore_general_prompts_in": [],
                    "tiles": [],
                    "tiler_context_key": "batch_mode",
                },
                "result": (batch_pipe,)
            }

        # Unpack incoming pipe
        INPUTS, PARAMS, KSAMPLER, OUTPUTS, SEGMENTS, SIZE, API, PROMPTER, current_credits, node_id, tiler_id, info_url = pipe
        input_prompts = PROMPTER.tiler_prompts

        input_tiles = list(OUTPUTS.orig_grid_images_all or [])



        self.init(**kwargs)
        log("TBG Tile Overrides starting", None, None, f"Node {tbg.INFO.id}")

        tiler_context_key = self._get_tiler_context_key(PARAMS, OUTPUTS, SEGMENTS)

        # Load JSON data once
        json_obj, src = self._load_tile_edits_json(kwargs, tbg.INFO.id, tiler_context_key)

        # Determine final length from tiles
        tiles_len = self._get_tile_length(input_tiles, json_obj, input_prompts)

        # Normalize all arrays to final length
        base_prompts = self._normalize(input_prompts or [], tiles_len)
        json_prompts = self._normalize(json_obj.get("prompts") or [], tiles_len)
        json_denoises = self._normalize(json_obj.get("denoises") or [], tiles_len)
        json_seeds = self._normalize(json_obj.get("seeds") or [], tiles_len)
        json_cnet = self._normalize(json_obj.get("cnet_strength") or [], tiles_len)
        json_cfg = self._normalize(json_obj.get("cfg_overrides") or [], tiles_len)
        json_models = self._normalize(json_obj.get("model_overrides") or [], tiles_len)
        json_cnetpipes = self._normalize(json_obj.get("cnetpipe_overrides") or [], tiles_len)
        json_color_match = self._normalize(json_obj.get("color_match_overrides") or [], tiles_len)
        json_ignore_general = [
            self._to_bool(v) for v in self._normalize(json_obj.get("ignore_general_prompts") or [], tiles_len)
        ]

        tiler_prompts = None,

        # Apply JSON overrides to base inputs
        final_prompts = self._apply_overrides(base_prompts, json_prompts)
        final_denoises = self._apply_overrides([""] * tiles_len, json_denoises)
        final_seeds = self._apply_overrides([""] * tiles_len, json_seeds)
        final_cnet = self._apply_overrides([""] * tiles_len, json_cnet)
        final_cfg = self._apply_overrides([""] * tiles_len, json_cfg)
        final_models = self._apply_overrides([""] * tiles_len, json_models)
        final_cnetpipes = self._apply_overrides([""] * tiles_len, json_cnetpipes)
        final_color_match = self._apply_overrides([""] * tiles_len, json_color_match)
        final_ignore_general = json_ignore_general

        # Update cache if prompts changed
        self._update_cache_if_needed(tbg.INFO.id, final_prompts, json_obj, base_prompts, tiler_context_key)

        # Save tiles and build results
        results = self._save_all_tiles(input_tiles, kwargs=kwargs, pipe_outputs=OUTPUTS, pipe_params=PARAMS, pipe_segments=SEGMENTS)

        # Cache prompts for test
        Tile_Prompter_Cache.set(tbg.CACHE.prompt, tuple(base_prompts))

        log("TBG Tile Overrides done", None, None, f"Node {tbg.INFO.id}")

        # Update PROMPTER
        PROMPTER.output_prompts = final_prompts
        PROMPTER.output_denoises = final_denoises
        PROMPTER.output_seeds_js = final_seeds
        PROMPTER.output_cnet_js = final_cnet
        PROMPTER.output_cfg_js = final_cfg
        PROMPTER.output_model_js = final_models
        PROMPTER.output_cnetpipe_js = final_cnetpipes
        PROMPTER.output_color_match_js = final_color_match
        PROMPTER.output_ignore_general_prompt_js = final_ignore_general
        model_override_key = str(tbg.INFO.id)
        TILE_MODEL_OVERRIDE_REGISTRY[model_override_key] = [
            kwargs.get("Model_1", None),
            kwargs.get("Model_2", None),
            kwargs.get("Model_3", None),
        ]
        TILE_CNETPIPE_OVERRIDE_REGISTRY[model_override_key] = [
            kwargs.get("CnetPipe_1", None),
            kwargs.get("CnetPipe_2", None),
            kwargs.get("CnetPipe_3", None),
        ]
        PROMPTER.model_override_key = model_override_key
        PROMPTER.cnetpipe_override_key = model_override_key
        PROMPTER.model_overrides = None
        PROMPTER.cnetpipe_overrides = None
        PROMPTER.cache_key = f"tile_edits_json_{tbg.INFO.id}"
        PROMPTER.tiler_context_key = tiler_context_key
        PROMPTER.Rebuild_only_modified_tiles = kwargs.get("Rebuild_only_modified_tiles")

        id  = copy.copy(tbg.INFO.id)

        return {
            "ui": {
                "prompts_out": final_prompts,
                "prompts_in": base_prompts,
                "denoises_out": final_denoises,
                "denoises_in": [""] * tiles_len,
                "seeds_out": final_seeds,
                "seeds_in": [""] * tiles_len,
                "cnet_strength_out": final_cnet,
                "cnet_strength_in": [""] * tiles_len,
                "cfg_overrides_out": final_cfg,
                "cfg_overrides_in": [""] * tiles_len,
                "model_overrides_out": final_models,
                "model_overrides_in": [""] * tiles_len,
                "cnetpipe_overrides_out": final_cnetpipes,
                "cnetpipe_overrides_in": [""] * tiles_len,
                "color_match_overrides_out": final_color_match,
                "color_match_overrides_in": [""] * tiles_len,
                "ignore_general_prompts_out": final_ignore_general,
                "ignore_general_prompts_in": [False] * tiles_len,
                "tiles": results,
                "tiler_context_key": tiler_context_key,
            },
            "result": ((INPUTS, PARAMS, KSAMPLER, OUTPUTS, SEGMENTS, SIZE, API, PROMPTER, current_credits,id,tiler_id, API.info_url),)
        }

    # Extracted helper methods (add to your class)

    def _normalize(self, arr, n):
        """Pad or truncate array to length n."""
        arr = list(arr or [])
        if len(arr) < n:
            arr.extend([""] * (n - len(arr)))
        elif len(arr) > n:
            arr = arr[:n]
        return arr

    def _get_tiler_context_key(self, params=None, outputs=None, segments=None):
        """Fingerprint upstream tiler geometry so stale UI JSON cannot cross runs."""
        try:
            key = getattr(params, "tiler_cache_key", None)
            if key:
                return str(key)
        except Exception:
            pass
        try:
            grid_count = len(getattr(outputs, "grid_images_all", []) or [])
            segment_count = len(getattr(segments, "segment_sampling_transforms", []) or [])
            image = getattr(outputs, "upscaled_image", None)
            image_shape = tuple(int(v) for v in getattr(image, "shape", ()) or ())
            grid_specs = getattr(params, "grid_specs", None)
            payload = {
                "grid_count": grid_count,
                "segment_count": segment_count,
                "image_shape": image_shape,
                "grid_specs": grid_specs,
                "upscale_by": getattr(params, "upscale_by", None),
                "upscale_model": getattr(params, "upscale_model_name", None),
                "fusion_mode": getattr(params, "Tile_Fusion_Mode", None),
            }
            return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        except Exception:
            return None

    def _json_matches_tiler_context(self, obj, tiler_context_key):
        if not tiler_context_key:
            return True
        stored_key = obj.get("_tbg_tiler_context_key")
        return bool(stored_key) and str(stored_key) == str(tiler_context_key)

    def _sanitize_tile_edits_json(self, obj):
        if not isinstance(obj, dict):
            return {}
        obj = dict(obj)
        obj.pop("tiles", None)
        has_edits = any(
            any(self._has_value(v) for v in (obj.get(key) or []))
            for key in self.TILE_EDIT_JSON_KEYS
        )
        return obj if has_edits else {}

    def _to_bool(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "on", "checked", "enabled"}

    def _load_tile_edits_json(self, _kwargs, node_id, tiler_context_key=None):
        """Load tile edits JSON from kwargs or cache."""
        # 1) Hidden input (kwargs)
        data_json = _kwargs.get("tile_edits_json")
        if isinstance(data_json, str) and data_json.strip():
            try:
                obj = json.loads(data_json)
                obj = self._sanitize_tile_edits_json(obj)
                if self._json_matches_tiler_context(obj, tiler_context_key):
                    return obj, "KWARGS"
                print(f"[TBG] Ignored stale tile_edits_json for node {node_id}: upstream tiler context changed")
            except Exception as e:
                print("tile_edits_json kwargs parse error:", e, data_json[:120])
        # 2) Tile_Prompter_Cache mirror
        try:
            cached = Tile_Prompter_Cache.get(f"tile_edits_json_{node_id}", None)
        except Exception as e:
            print("tile_edits_json cache read error:", e)
            cached = None
        if isinstance(cached, str) and cached.strip():
            try:
                obj = json.loads(cached)
                obj = self._sanitize_tile_edits_json(obj)
                if self._json_matches_tiler_context(obj, tiler_context_key):
                    return obj, "CACHE"
                print(f"[TBG] Ignored stale cached tile_edits_json for node {node_id}: upstream tiler context changed")
            except Exception as e:
                print("tile_edits_json cache parse error:", e, cached[:120])
        # Nothing usable
        return {}, None

    def _get_tile_length(self, input_tiles, json_obj, input_prompts):
        """Calculate final tile length from available data."""
        tiles_len = 0
        if input_tiles:
            try:
                tiles_len += len([t[0] for t in input_tiles])
            except Exception:
                tiles_len += len(input_tiles)


        # Fallback if no tiles are present yet
        if tiles_len <= 0:
            base_len = len(input_prompts) if input_prompts is not None else 0
            tiles_len = max(
                base_len,
                len(json_obj.get("prompts") or []),
                len(json_obj.get("denoises") or []),
                len(json_obj.get("seeds") or []),
                len(json_obj.get("cnet_strength") or []),
                len(json_obj.get("cfg_overrides") or []),
                len(json_obj.get("model_overrides") or []),
                len(json_obj.get("cnetpipe_overrides") or []),
                len(json_obj.get("color_match_overrides") or []),
                len(json_obj.get("ignore_general_prompts") or []),
            )
        return tiles_len

    def _apply_overrides(self, base_array, override_array):
        """Apply override values where they exist."""
        result = base_array[:]
        for i, v in enumerate(override_array):
            if self._has_value(v):
                result[i] = v
        return result

    def _update_cache_if_needed(self, node_id, final_prompts, json_obj, base_prompts, tiler_context_key=None):
        """Update cache if prompts have changed."""

        try:
            cache_key = f"tile_edits_json_{node_id}"

            existing = Tile_Prompter_Cache.get(cache_key, None)

            if isinstance(existing, str) and existing.strip():
                try:
                    obj = json.loads(existing)
                except Exception:
                    obj = {}
            else:
                obj = {}
            if tiler_context_key and str(obj.get("_tbg_tiler_context_key") or "") != str(tiler_context_key):
                obj = {}

            prev_prompts = obj.get("prompts") or []
            prev_prompts_norm = self._normalize(prev_prompts, len(final_prompts))

            prompts_changed = any(
                (prev_prompts_norm[i] or "") != (final_prompts[i] or "") for i in range(len(final_prompts)))

            if prompts_changed and any((p or "").strip() for p in final_prompts):

                # Keep non-prompt fields, update prompts
                obj["prompts"] = final_prompts
                obj["denoises"] = self._normalize(obj.get("denoises") or [], len(final_prompts))
                obj["seeds"] = self._normalize(obj.get("seeds") or [], len(final_prompts))
                obj["cnet_strength"] = self._normalize(obj.get("cnet_strength") or [], len(final_prompts))
                obj["cfg_overrides"] = self._normalize(obj.get("cfg_overrides") or [], len(final_prompts))
                obj["model_overrides"] = self._normalize(obj.get("model_overrides") or [], len(final_prompts))
                obj["cnetpipe_overrides"] = self._normalize(obj.get("cnetpipe_overrides") or [], len(final_prompts))
                obj["color_match_overrides"] = self._normalize(obj.get("color_match_overrides") or [], len(final_prompts))
                obj["ignore_general_prompts"] = [
                    self._to_bool(v) for v in self._normalize(obj.get("ignore_general_prompts") or [], len(final_prompts))
                ]
                if tiler_context_key:
                    obj["_tbg_tiler_context_key"] = tiler_context_key

                Tile_Prompter_Cache.set(cache_key, json.dumps(obj))
        except Exception as e:
            print("TilePrompter only-prompts merge failed:", e)

    def _tile_preview_enabled(self, kwargs):
        explicit = kwargs.get("Enable_Tile_Preview", None)
        if explicit is not None:
            return bool(explicit)
        preview_setting = str(os.getenv("TBG_TILEPROMPTER_PREVIEW", "1")).strip().lower()
        if preview_setting in {"0", "false", "no", "off"}:
            return False
        env_enabled = preview_setting in {"1", "true", "yes", "on"}
        is_dev = str(getattr(getattr(tbg, "API", None), "status", "")).lower() == "dev"
        return env_enabled or is_dev

    def _save_all_tiles(self, input_tiles, kwargs=None, pipe_outputs=None, pipe_params=None, pipe_segments=None):
        """Save all tiles and return results list."""
        kwargs = kwargs or {}
        if not self._tile_preview_enabled(kwargs):
            return []
        results = []
        filename_prefix = f"TBG_temp_tilePrompter_id_{tbg.INFO.id}"

        # Cleanup old files
        tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
        cleanup_targets = [
            os.path.join(tbg_temp_dir, f"thumbnail_{tbg.INFO.id}"),
            os.path.join(tbg_temp_dir, f"tiles_{tbg.INFO.id}"),
        ]
        cleanup_pattern = os.path.join(tbg_temp_dir, filename_prefix + "*")
        for f in [*cleanup_targets, *glob.glob(cleanup_pattern)]:
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
            except Exception:
                pass

        segment_masks = self._segment_preview_masks(pipe_params=pipe_params, pipe_segments=pipe_segments)
        segment_preview_tiles = self._segment_preview_tiles(pipe_outputs=pipe_outputs, pipe_params=pipe_params, pipe_segments=pipe_segments)
        segment_indices = self._segment_preview_indices(pipe_params=pipe_params, pipe_segments=pipe_segments)

        # Save input tiles
        if input_tiles:
            try:
                flat_tiles = [t[0] for t in input_tiles]
            except Exception:
                flat_tiles = input_tiles

            for idx, torchtile in enumerate(flat_tiles):
                arr = self._process_tile_tensor(torchtile)
                if arr is None:
                    continue
                if idx in segment_indices:
                    clean_arr = segment_preview_tiles.get(idx)
                    if clean_arr is None:
                        print(f"[TBG] Segment thumbnail skipped: missing full-image preview tile for index={idx}")
                        continue
                else:
                    clean_arr = arr
                thumb_arr = clean_arr

                mask = segment_masks.get(idx)
                if mask is not None:
                    thumb_arr = self._composite_segment_thumbnail(clean_arr, mask)

                # Save thumbnail
                fname, sub = self._save_tile_image(
                    thumb_arr, f"TBG/thumbnail_{tbg.INFO.id}/{filename_prefix}",
                    idx, MAX_SIZE_THUMBNAIL
                )
                row = {"filename": fname, "subfolder": sub, "type": "temp"}

                if mask is not None:
                    row["segment"] = True

                results.append(row)

                # Save tile
                self._save_tile_image(
                    clean_arr, f"TBG/tiles_{tbg.INFO.id}/{filename_prefix}",
                    idx, MAX_SIZE_TILESIZESAVED
                )


        return results

    def _segment_preview_indices(self, pipe_params=None, pipe_segments=None):
        try:
            params = pipe_params or tbg.PARAMS
            segments = pipe_segments or tbg.SEGMENTS
            len_grid_images = int(getattr(params, "len_grid_images", 0) or 0)
            transforms = getattr(segments, "segment_sampling_transforms", None)
            if isinstance(transforms, (list, tuple)):
                return {len_grid_images + i for i in range(len(transforms))}
        except Exception:
            pass
        return set()

    def _segment_preview_tiles(self, pipe_outputs=None, pipe_params=None, pipe_segments=None):
        tiles = {}
        try:
            params = pipe_params or tbg.PARAMS
            segments = pipe_segments or tbg.SEGMENTS
            outputs = pipe_outputs or tbg.OUTPUTS
            len_grid_images = int(getattr(params, "len_grid_images", 0) or 0)
            transforms = getattr(segments, "segment_sampling_transforms", None)
            full_image = getattr(outputs, "upscaled_image", None)
            if not isinstance(transforms, (list, tuple)) or not isinstance(full_image, torch.Tensor):
                return tiles

            image = full_image.detach().to(torch.float32)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                return tiles
            if image.shape[-1] not in (1, 3, 4) and image.shape[1] in (1, 3, 4):
                image = image.permute(0, 2, 3, 1).contiguous()
            full_h = int(image.shape[1])
            full_w = int(image.shape[2])

            for seg_idx, transform in enumerate(transforms):
                if not isinstance(transform, dict):
                    continue
                crop = transform.get("sampling_crop_region")
                tile_size = transform.get("sampling_tile_size")
                if not crop or not tile_size:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in crop]
                tile_w, tile_h = [int(round(float(v))) for v in tile_size]
                tile = self._crop_full_image_preview_tile(image, full_w, full_h, x1, y1, x2, y2, tile_w, tile_h)
                if tile is None:
                    continue
                tiles[len_grid_images + seg_idx] = tile
                if getattr(tbg.API, "status", None) == "Dev":
                    bbox = transform.get("bbox_in_sampling_tile", None)
                    margins = transform.get("sampling_margins", None)
                    print(
                        f"[TBG] Segment thumbnail source rebuilt from full image: "
                        f"segment={seg_idx + 1} index={len_grid_images + seg_idx} "
                        f"full={full_w}x{full_h} crop=({x1},{y1},{x2},{y2}) "
                        f"tile={tile_w}x{tile_h} bbox_in_tile={bbox} margins={margins}"
                    )
        except Exception as e:
            print(f"[TBG] Segment preview tile rebuild failed: {e}")
        return tiles

    def _crop_full_image_preview_tile(self, image, full_w, full_h, x1, y1, x2, y2, tile_w, tile_h):
        try:
            x1 = max(0, min(int(full_w), int(x1)))
            y1 = max(0, min(int(full_h), int(y1)))
            x2 = max(x1 + 1, min(int(full_w), int(x2)))
            y2 = max(y1 + 1, min(int(full_h), int(y2)))
            tile_w = max(1, int(tile_w))
            tile_h = max(1, int(tile_h))
            crop_tensor = image[:, y1:y2, x1:x2, :]
            if int(crop_tensor.shape[1]) != tile_h or int(crop_tensor.shape[2]) != tile_w:
                crop_tensor = torch.nn.functional.interpolate(
                    crop_tensor.permute(0, 3, 1, 2),
                    size=(tile_h, tile_w),
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
            return self._process_tile_tensor(crop_tensor.clamp(0.0, 1.0))
        except Exception:
            return None

    def _segment_preview_masks(self, pipe_params=None, pipe_segments=None):
        masks = {}
        try:
            params = pipe_params or tbg.PARAMS
            segments = pipe_segments or tbg.SEGMENTS
            len_grid_images = int(getattr(params, "len_grid_images", 0) or 0)
            mode = getattr(params, "Tile_Fusion_Mode", None)
            if mode == "Soft Merge":
                mask_sources = ("compositing_mask", "segms_cropped_masks")
            elif mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion"):
                mask_sources = ("inpainting_mask", "segms_cropped_masks", "compositing_mask")
            else:
                mask_sources = ("compositing_mask", "segms_cropped_masks", "inpainting_mask")

            mask_source = None
            values = None
            for candidate in mask_sources:
                candidate_values = getattr(segments, candidate, None)
                if isinstance(candidate_values, (list, tuple)):
                    mask_source = candidate
                    values = candidate_values
                    break
            transforms = getattr(segments, "segment_sampling_transforms", None)
            if not isinstance(values, (list, tuple)):
                return masks
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"[TBG] Segment thumbnail overlay uses {mask_source} for mode {mode}")
            for seg_idx, mask in enumerate(values):
                if mask is not None:
                    if isinstance(transforms, (list, tuple)) and seg_idx < len(transforms):
                        mask = self._transform_segment_preview_mask(mask, transforms[seg_idx])
                    masks[len_grid_images + seg_idx] = mask
        except Exception as e:
            print(f"[TBG] Segment preview mask lookup failed: {e}")
        return masks

    def _segment_preview_mask_from_transform(self, transform, mode):
        if not isinstance(transform, dict):
            return None
        try:
            tile_size = transform.get("sampling_tile_size")
            bbox = transform.get("bbox_in_sampling_tile")
            if not tile_size or not bbox:
                return None
            tile_w, tile_h = [max(1, int(round(float(v)))) for v in tile_size]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            margins = transform.get("sampling_margins") or {}
            if mode == "Soft Merge":
                margin = int(max(0, round(float(margins.get("feather", 0) or 0))))
            elif mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion"):
                margin = int(max(0, round(float(
                    (margins.get("reference", 0) or 0) + (margins.get("fusion", 0) or 0)
                ))))
            else:
                margin = int(max(0, round(float(transform.get("sampling_margin", 0) or 0))))

            x1 = max(0, min(tile_w, x1 - margin))
            y1 = max(0, min(tile_h, y1 - margin))
            x2 = max(x1 + 1, min(tile_w, x2 + margin))
            y2 = max(y1 + 1, min(tile_h, y2 + margin))

            # Thumbnail overlay convention: 0 means active red overlay, 1 means no overlay.
            preview_mask = torch.ones((1, tile_h, tile_w), dtype=torch.float32)
            preview_mask[:, y1:y2, x1:x2] = 0.0
            if getattr(tbg.API, "status", None) == "Dev":
                print(
                    f"[TBG] Segment thumbnail mask rebuilt from transform: "
                    f"tile={tile_w}x{tile_h} active=({x1},{y1},{x2},{y2}) margin={margin}"
                )
            return preview_mask
        except Exception as e:
            print(f"[TBG] Segment thumbnail transform mask failed: {e}")
            return None

    def _transform_segment_preview_mask(self, mask, transform):
        if not isinstance(transform, dict) or not isinstance(mask, torch.Tensor):
            return mask
        try:
            crop_region = transform.get("native_crop_region")
            sampling_crop = transform.get("sampling_crop_region")
            tile_size = transform.get("sampling_tile_size")
            if not crop_region or not sampling_crop or not tile_size:
                return mask
            sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in sampling_crop]
            cx1, cy1, cx2, cy2 = [int(round(float(v))) for v in crop_region]
            tile_w, tile_h = [int(round(float(v))) for v in tile_size]
            crop_w = max(1, sx2 - sx1)
            crop_h = max(1, sy2 - sy1)

            source = mask.detach().to(torch.float32)
            if source.ndim == 2:
                source = source.unsqueeze(0)
            elif source.ndim == 4:
                if source.shape[1] == 1:
                    source = source[:, 0]
                else:
                    source = source[..., 0]
            if source.ndim != 3:
                return mask

            expected_h = max(1, cy2 - cy1)
            expected_w = max(1, cx2 - cx1)
            if int(source.shape[-2]) != expected_h or int(source.shape[-1]) != expected_w:
                source = torch.nn.functional.interpolate(
                    source.unsqueeze(1),
                    size=(expected_h, expected_w),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]

            out = torch.zeros((source.shape[0], crop_h, crop_w), dtype=source.dtype, device=source.device)
            ox1 = max(sx1, cx1)
            oy1 = max(sy1, cy1)
            ox2 = min(sx2, cx2)
            oy2 = min(sy2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                return out

            src_x1 = ox1 - cx1
            src_y1 = oy1 - cy1
            dst_x1 = ox1 - sx1
            dst_y1 = oy1 - sy1
            width = ox2 - ox1
            height = oy2 - oy1
            out[:, dst_y1:dst_y1 + height, dst_x1:dst_x1 + width] = source[:, src_y1:src_y1 + height, src_x1:src_x1 + width]
            out = torch.nn.functional.interpolate(
                out.unsqueeze(1),
                size=(tile_h, tile_w),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            return out.clamp(0.0, 1.0)
        except Exception as e:
            print(f"[TBG] Segment preview mask transform failed: {e}")
            return mask

    def _process_tile_tensor(self, torchtile):
        """Convert tile tensor to numpy array."""
        try:
            return torchtile.squeeze(0).cpu().numpy() * 255
        except Exception:
            try:
                return np.asarray(torchtile)
            except Exception:
                return None

    def _process_mask_tensor(self, mask):
        """Convert mask tensor to RGB uint8 preview array."""
        try:
            arr = mask.detach().squeeze().cpu().numpy()
        except Exception:
            try:
                arr = np.asarray(mask).squeeze()
            except Exception:
                return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    def _composite_segment_thumbnail(self, arr, mask):
        """Bake the segment mask into the thumbnail image; no sidecar mask file needed."""
        mask_arr = self._process_mask_tensor(mask)
        if mask_arr is None:
            return arr
        try:
            img_arr = arr
            if img_arr.dtype != np.float32 and img_arr.dtype != np.float64:
                img_arr = img_arr.astype(np.float32)
            if img_arr.max() <= 1.0:
                img_arr = img_arr * 255.0
            img_arr = np.clip(img_arr, 0, 255).astype(np.float32)

            mask_gray = 1.0 - (mask_arr[..., 0].astype(np.float32) / 255.0)
            if mask_gray.shape[:2] != img_arr.shape[:2]:
                mask_img = Image.fromarray(np.clip(mask_gray * 255.0, 0, 255).astype(np.uint8))
                mask_img = mask_img.resize((int(img_arr.shape[1]), int(img_arr.shape[0])), Image.BILINEAR)
                mask_gray = np.asarray(mask_img).astype(np.float32) / 255.0

            overlay = np.zeros_like(img_arr)
            overlay[..., 0] = 255.0
            overlay[..., 1] = 64.0
            overlay[..., 2] = 64.0
            alpha = (mask_gray[..., None] * 0.5).clip(0.0, 0.5)
            return (img_arr * (1.0 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[TBG] Segment thumbnail composite failed: {e}")
            return arr

    def _save_tile_image(self, arr, path_prefix, idx, max_size):
        """Save a single tile image and return filename info."""

        # ---- hard guards ----
        if arr is None:
            return None, None

        if not hasattr(arr, "shape") or arr.ndim < 2:
            return None, None

        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return None, None

        if max_size is None or max_size[0] <= 0 or max_size[1] <= 0:
            return None, None

        try:
            full_folder, base, _, sub, _ = folder_paths.get_save_image_path(
                path_prefix,
                tbg.temp_output_dir,
                w,
                h,
            )
            os.makedirs(full_folder, exist_ok=True)

            fname = f"{base}_{idx:05d}.png"
            path = os.path.join(full_folder, fname)

            # ---- sanitize image ----
            img_arr = arr
            if img_arr.dtype != np.uint8:
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_arr)

            # thumbnail() will never upscale — safe for previews
            img.thumbnail(max_size, Image.Resampling.BILINEAR)
            img.save(path, compress_level=4)

            return fname, sub

        except Exception as e:
            print(f"[TBG] Tile save failed (idx={idx}): {e}")
            return None, None

    def _has_value(self, x):
        """Check if value is meaningful for override."""
        if isinstance(x, bool):
            return x
        return x is not None and x != ""



def _route_exists(method: str, path: str) -> bool:
    try:
        for route in PromptServer.instance.routes:
            route_method = getattr(route, "method", "")
            resource = getattr(route, "resource", None)
            canonical = getattr(resource, "canonical", None)
            if route_method == method and canonical == path:
                return True
    except Exception:
        return False
    return False


def _safe_get(path: str):
    def _decorator(fn):
        if _route_exists("GET", path):
            return fn
        return PromptServer.instance.routes.get(path)(fn)
    return _decorator


def _safe_post(path: str):
    def _decorator(fn):
        if _route_exists("POST", path):
            return fn
        return PromptServer.instance.routes.post(path)(fn)
    return _decorator
# --- Routes (keep these once in the file; do not duplicate) ---

@_safe_get("/TBG/get_tile_edits_json")
async def get_tile_edits_json(request):
    node_id = request.query.get("node")
    if not node_id:
        return web.json_response({"tile_edits_json": None})
    try:
        cached = Tile_Prompter_Cache.get(f"tile_edits_json_{node_id}", None)
        if isinstance(cached, str) and cached.strip():
            try:
                obj = json.loads(cached)
                obj.pop("tiles", None)
                has_edits = any(
                    any(TBG_TilePrompter_v1._json_value_has_content(v) for v in (obj.get(key) or []))
                    for key in TBG_TilePrompter_v1.TILE_EDIT_JSON_KEYS
                )
                cached = json.dumps(obj) if has_edits else None
            except Exception:
                cached = None
        return web.json_response({"tile_edits_json": cached})
    except Exception as e:
        return web.json_response({"tile_edits_json": None, "error": str(e)})

@_safe_post("/TBG/set_tile_edits_json")
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
        if isinstance(data, str) and data.strip():
            try:
                obj = json.loads(data)
                obj.pop("tiles", None)
                has_edits = any(
                    any(TBG_TilePrompter_v1._json_value_has_content(v) for v in (obj.get(key) or []))
                    for key in TBG_TilePrompter_v1.TILE_EDIT_JSON_KEYS
                )
                if not has_edits:
                    Tile_Prompter_Cache.cache_delete(f"tile_edits_json_{node_id}")
                    return web.json_response({"ok": True, "cleared": True})
                data = json.dumps(obj)
            except Exception:
                pass
        elif not data:
            Tile_Prompter_Cache.cache_delete(f"tile_edits_json_{node_id}")
            return web.json_response({"ok": True, "cleared": True})
        Tile_Prompter_Cache.set(f"tile_edits_json_{node_id}", data)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})

@_safe_get("/TBG/get_input_seeds")
async def get_input_seeds(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_seeds_{nodeId}'
    seeds = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"seeds_in": seeds})

@_safe_get("/TBG/get_input_cnet_strength")
async def get_input_cnet_strength(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_cnet_strength_{nodeId}'
    cnet = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"cnet_strength_in": cnet})

@_safe_get("/TBG/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"prompts_in": input_prompts})

@_safe_get("/TBG/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"denoises_in": input_denoises})

@_safe_get("/TBG/set_prompt")
async def set_prompt(request):
    prompt = request.query.get("prompt", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_prompts = Tile_Prompter_Cache.get(cache_name, [])
    _input_prompts_edited = Tile_Prompter_Cache.get(cache_name_edited, _input_prompts)
    if _input_prompts_edited and 0 <= index < len(_input_prompts_edited):
        lst = list(_input_prompts_edited)
        lst[index] = prompt
        Tile_Prompter_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} prompt has been updated :{prompt}")

@_safe_get("/TBG/set_denoise")
async def set_denoise(request):
    denoise = request.query.get("denoise", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_denoises = Tile_Prompter_Cache.get(cache_name, [])
    _input_denoises_edited = Tile_Prompter_Cache.get(cache_name_edited, _input_denoises)
    if _input_denoises_edited and 0 <= index < len(_input_denoises_edited):
        lst = list(_input_denoises_edited)
        lst[index] = denoise
        Tile_Prompter_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} denoise has been updated: {denoise}")

@_safe_get("/TBG/set_seed")
async def set_seed(request):
    seed = request.query.get("seed", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_seeds_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _seeds = Tile_Prompter_Cache.get(cache_name, [])
    _seeds_edited = Tile_Prompter_Cache.get(cache_name_edited, _seeds)
    if _seeds_edited and 0 <= index < len(_seeds_edited):
        lst = list(_seeds_edited)
        lst[index] = seed
        Tile_Prompter_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} seed has been updated: {seed}")

@_safe_get("/TBG/set_cnet_strength")
async def set_cnet_strength(request):
    cnet_strength = request.query.get("cnet_strength", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_cnet_strength_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _cnet = Tile_Prompter_Cache.get(cache_name, [])
    _cnet_edited = Tile_Prompter_Cache.get(cache_name_edited, _cnet)
    if _cnet_edited and 0 <= index < len(_cnet_edited):
        lst = list(_cnet_edited)
        lst[index] = cnet_strength
        Tile_Prompter_Cache.set(cache_name_edited, tuple(lst))
    return web.json_response(f"Tile {index} cnet_strength has been updated: {cnet_strength}")

@_safe_get("/TBG/tile_prompt")
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







