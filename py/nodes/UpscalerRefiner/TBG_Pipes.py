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
import time
import copy
from nacl import hashlib

MAX_SIZE_THUMBNAIL = (256, 256)  # max width, height
MAX_SIZE_TILESIZESAVED = (4096, 4096)  # max width, height
# Third-party imports
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = 592515344
from PIL import Image
from aiohttp import web
import folder_paths
from .inc.tp_cache import Tile_Prompter_Cache
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
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"

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
                "PRO_Fusion_Complexity_min_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 0.2, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the min denoise level mainly on low-complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),
                "PRO_Fusion_Complexity_max_Denoise": ("FLOAT", {"label": "inpaint_max", "default": 1, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
                                                                "tooltip": "(crops Complexity denoise min) Sets the mx denoise level mainly on high complexity areas. If the minimum value is greater than the maximum, the map is inverted."}),
                "PRO_Fusion_Complexity_Mask_Blur": ("FLOAT", {"label": "inpaint_max", "default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.01,
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
    EXECUTION_CACHING = "NODE"

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
        # Unpack incoming pipe
        INPUTS, PARAMS, KSAMPLER, OUTPUTS, SEGMENTS, SIZE, API, PROMPTER, current_credits, node_id, tiler_id, info_url= kwargs.get('TBG_Pipe',
                                                                                                       (None, None))
        input_prompts = PROMPTER.tiler_prompts

        input_tiles = copy.deepcopy(OUTPUTS.orig_grid_images_all)



        self.init(**kwargs)
        log("TBG Tile Overrides starting", None, None, f"Node {tbg.INFO.id}")

        # Load JSON data once
        json_obj, src = self._load_tile_edits_json(kwargs, tbg.INFO.id)

        # Determine final length from tiles
        tiles_len = self._get_tile_length(input_tiles, json_obj, input_prompts)

        # Normalize all arrays to final length
        base_prompts = self._normalize(input_prompts or [], tiles_len)
        json_prompts = self._normalize(json_obj.get("prompts") or [], tiles_len)
        json_denoises = self._normalize(json_obj.get("denoises") or [], tiles_len)
        json_seeds = self._normalize(json_obj.get("seeds") or [], tiles_len)
        json_cnet = self._normalize(json_obj.get("cnet_strength") or [], tiles_len)

        tiler_prompts = None,

        # Apply JSON overrides to base inputs
        final_prompts = self._apply_overrides(base_prompts, json_prompts)
        final_denoises = self._apply_overrides([""] * tiles_len, json_denoises)
        final_seeds = self._apply_overrides([""] * tiles_len, json_seeds)
        final_cnet = self._apply_overrides([""] * tiles_len, json_cnet)

        # Update cache if prompts changed
        self._update_cache_if_needed(tbg.INFO.id, final_prompts, json_obj, base_prompts)

        # Save tiles and build results
        results = self._save_all_tiles(input_tiles)

        # Cache prompts for test
        Tile_Prompter_Cache.set(tbg.CACHE.prompt, tuple(base_prompts))

        log("TBG Tile Overrides done", None, None, f"Node {tbg.INFO.id}")

        # Update PROMPTER
        PROMPTER.output_prompts = final_prompts
        PROMPTER.output_denoises = final_denoises
        PROMPTER.output_seeds_js = final_seeds
        PROMPTER.output_cnet_js = final_cnet
        PROMPTER.cache_key = f"tile_edits_json_{tbg.INFO.id}"
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
                "tiles": results,
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

    def _load_tile_edits_json(self, _kwargs, node_id):
        """Load tile edits JSON from kwargs or cache."""
        # 1) Hidden input (kwargs)
        data_json = _kwargs.get("tile_edits_json")
        if isinstance(data_json, str) and data_json.strip():
            try:
                return json.loads(data_json), "KWARGS"
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
                return json.loads(cached), "CACHE"
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
            )
        return tiles_len

    def _apply_overrides(self, base_array, override_array):
        """Apply override values where they exist."""
        result = base_array[:]
        for i, v in enumerate(override_array):
            if self._has_value(v):
                result[i] = v
        return result

    def _update_cache_if_needed(self, node_id, final_prompts, json_obj, base_prompts):
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

                Tile_Prompter_Cache.set(cache_key, json.dumps(obj))
        except Exception as e:
            print("TilePrompter only-prompts merge failed:", e)

    def _save_all_tiles(self, input_tiles):
        """Save all tiles and return results list."""
        results = []
        filename_prefix = f"TBG_temp_tilePrompter_id_{tbg.INFO.id}"

        # Cleanup old files
        tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
        cleanup_pattern = os.path.join(tbg_temp_dir, filename_prefix + "*")
        for f in glob.glob(cleanup_pattern):
            try:
                os.remove(f)
            except Exception:
                pass

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

                # Save thumbnail
                fname, sub = self._save_tile_image(
                    arr, f"TBG/thumbnail_{tbg.INFO.id}/{filename_prefix}",
                    idx, MAX_SIZE_THUMBNAIL
                )
                results.append({"filename": fname, "subfolder": sub, "type": "temp"})

                # Save tile
                self._save_tile_image(
                    arr, f"TBG/tiles_{tbg.INFO.id}/{filename_prefix}",
                    idx, MAX_SIZE_TILESIZESAVED
                )


        return results

    def _process_tile_tensor(self, torchtile):
        """Convert tile tensor to numpy array."""
        try:
            return torchtile.squeeze(0).cpu().numpy() * 255
        except Exception:
            try:
                return np.asarray(torchtile)
            except Exception:
                return None

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
        return x is not None and x != ""


# --- Routes (keep these once in the file; do not duplicate) ---

@PromptServer.instance.routes.get("/TBG/get_tile_edits_json")
async def get_tile_edits_json(request):
    node_id = request.query.get("node")
    if not node_id:
        return web.json_response({"tile_edits_json": None})
    try:
        cached = Tile_Prompter_Cache.get(f"tile_edits_json_{node_id}", None)
        return web.json_response({"tile_edits_json": cached})
    except Exception as e:
        return web.json_response({"tile_edits_json": None, "error": str(e)})

@PromptServer.instance.routes.post("/TBG/set_tile_edits_json")
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
        Tile_Prompter_Cache.set(f"tile_edits_json_{node_id}", data)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})

@PromptServer.instance.routes.get("/TBG/get_input_seeds")
async def get_input_seeds(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_seeds_{nodeId}'
    seeds = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"seeds_in": seeds})

@PromptServer.instance.routes.get("/TBG/get_input_cnet_strength")
async def get_input_cnet_strength(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_cnet_strength_{nodeId}'
    cnet = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"cnet_strength_in": cnet})

@PromptServer.instance.routes.get("/TBG/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"prompts_in": input_prompts})

@PromptServer.instance.routes.get("/TBG/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = Tile_Prompter_Cache.get(cache_name, [])
    return web.json_response({"denoises_in": input_denoises})

@PromptServer.instance.routes.get("/TBG/set_prompt")
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

@PromptServer.instance.routes.get("/TBG/set_denoise")
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

@PromptServer.instance.routes.get("/TBG/set_seed")
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

@PromptServer.instance.routes.get("/TBG/set_cnet_strength")
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

@PromptServer.instance.routes.get("/TBG/tile_prompt")
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

