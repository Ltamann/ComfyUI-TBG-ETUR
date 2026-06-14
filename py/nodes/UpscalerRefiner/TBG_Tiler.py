"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________
_______________________________________________________________________________________________________________________________________________

TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO
Nodes:
TBG_Enhanced Tiled Upscaler FLUX PRO
TBG_Enhanced Refiner FLUX PRO

⚈ TBG Tile Enrichment Pipe
⚈ TBG Tile Prompt Pipe
⚈ TBG Tile ControlNet Pipe

____TBG_Flux Enhanced Tiled Upscaler and Refiner PRO ______________________________________________________________________________

"""
import threading
import math
import copy
import comfy.utils
import re
import comfy.text_encoders.llama
import comfy.model_management as mm
import torch
import time
import PIL
import nodes
import hashlib
import json
from functools import wraps
from PIL import Image
from comfy_extras import nodes_upscale_model
from ...utils.log import log
from .inc.image import TBG_Image
from .inc.batch import is_batch_image, make_batch_pipe
from .inc.tbg_pid import (
    PID_DEFAULT_COLOR_MATCH,
    PID_DEFAULT_OVERLAP,
    PID_DEFAULT_STITCH_BLUR,
    PID_DEFAULT_STITCH_FEATHER,
    PID_SCALE,
    is_pid_upscale_model,
    run_pid_tiled_upscale,
)
from .inc.pid_vlm import PID_VLM_MODEL, make_pid_vlm_prompt_fn
from .inc.prosegs import TBG_Segms
from ...vendor.ComfyUI_Unload_Models_main.py.unload_one_model import UnloadOneModelNode
from ...vendor.ComfyUI_QwenVL.AILab_QwenVL import QwenVLBase
from ...vendor.ComfyUI_QwenVL.AILab_QwenVL_GGUF import QwenVLGGUFBase
from ...vendor.ComfyUI_QwenVL.response_cleanup import strip_thinking
from ...vendor.ComfyUI_QwenVL.TBG_QwenVL_Server import QwenVLServerClient
from ...vendor.ComfyUI_QwenVL.TBG_QwenVL_ServerManager import get_server_manager
from ...vendor.ComfyUI_QwenVL.nodes import Qwen2VL_TBG
from ...vendor.ComfyUI_Florence2.nodes import DownloadAndLoadFlorence2Model,Florence2Run
from ...vendor.ComfyUI_Impact_Pack.masktoseg import MaskToSEGS, combine_segs
PIL.Image.MAX_IMAGE_PIXELS = 592515344
# WORKER
from ....TBG.SERVERS.WORKER_server import WORKER,TBG_Controller
from .TBG_Refiner import TBG_Refiner_v1

from ....TBG.CALLBACKS.constants import get_tbg , TBGState, reset_tbg
from ...vendor.seedvr2_videoupscaler.src.interfaces.video_upscaler import TBG_SeedVR2VideoUpscaler
from ...vendor.flashvsr_ultra_fast.nodes import flashvr
Tiler_Upscale_Cache = True
Only_Upscale = False
OPENAI_COMPAT_MODEL = "OpenAI-Compatible (Labs Server)"


def _normalize_vlm_model(value):
    if value is None:
        return "NONE"
    text = str(value).strip()
    if not text:
        return "NONE"
    if text.upper() == "NONE" or text.lower() == "none":
        return "NONE"
    if text.lower() == "none (fp16)":
        return "NONE"
    return text


def _runtime_device_string():
    try:
        dev = mm.get_torch_device()
        if hasattr(dev, "type"):
            if dev.type == "cuda":
                idx = dev.index if dev.index is not None else 0
                return f"cuda:{idx}"
            return str(dev)
        return str(dev)
    except Exception:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
nodes_upscale_model_UpscaleModelLoader = nodes_upscale_model.UpscaleModelLoader
if hasattr(nodes_upscale_model_UpscaleModelLoader, "execute"):
    nodes_upscale_model_UpscaleModelLoader_execute = nodes_upscale_model_UpscaleModelLoader.execute
elif hasattr(nodes_upscale_model_UpscaleModelLoader, "load_model"):
    nodes_upscale_model_UpscaleModelLoader_execute = nodes_upscale_model_UpscaleModelLoader.load_model


nodes_upscale_model_ImageUpscaleWithModel = nodes_upscale_model.ImageUpscaleWithModel
if hasattr(nodes_upscale_model_ImageUpscaleWithModel, "execute"):
    nodes_upscale_model_ImageUpscaleWithModel_execute = nodes_upscale_model_ImageUpscaleWithModel.execute
elif hasattr(nodes_upscale_model_ImageUpscaleWithModel, "upscale"):
    nodes_upscale_model_ImageUpscaleWithModel_execute = nodes_upscale_model_ImageUpscaleWithModel.upscale



# Cache protects the tiler for reruns if only the promt or image unrelevant settings are changed
import hashlib
import json
from functools import wraps

def upscale_cache_comfy_method(maxsize=1, enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            cache_name = f"_{func.__name__}_cache"

            # Evaluate enabled at runtime if it's callable
            is_enabled = enabled() if callable(enabled) else enabled

            if not is_enabled:
                if hasattr(cls, cache_name):
                    getattr(cls, cache_name).clear()
                return func(cls, *args, **kwargs)

            if not hasattr(cls, cache_name):
                setattr(cls, cache_name, {})
            cache = getattr(cls, cache_name)

            cache_key = None
            try:
                # 1) Take image from args (input_image), not from tbg
                image = None
                if len(args) > 0:
                    image = args[0]  # for @classmethod: input_image

                image_hash = _hash_tiler_tensor(image) if image is not None else "none"

                # 2) Cache only when all inputs that can change the upscaled pixels match.
                cache_kwargs = _input_upscale_cache_payload(image, kwargs)
                kwargs_hash = _fingerprint_hash(cache_kwargs)

                cache_key = f"{func.__name__}_{image_hash}_{kwargs_hash}"
            except Exception as e:
                print(f"Cache key generation failed: {e}. Running without cache.")

            # If key exists -> cache hit
            if cache_key and cache_key in cache:
                log("Using Cache for Upscaled Image", None, None, f"Node {tbg.INFO.id}")
                return cache[cache_key]

            # Otherwise run the function
            log("Upscaled Imag", None, None, f"Node {tbg.INFO.id}")
            result = func(cls, *args, **kwargs)

            # Store in cache with size limit
            if cache_key:
                if len(cache) >= maxsize:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                cache[cache_key] = result

            return result

        return wrapper
    return decorator

import hashlib
import json
from functools import wraps


import hashlib, json

def _hash_tiler_tensor(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        try:
            tensor = value.detach().to("cpu").contiguous()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            h = hashlib.md5()
            h.update(str(tuple(tensor.shape)).encode())
            h.update(str(tensor.dtype).encode())
            h.update(tensor.numpy().tobytes())
            return h.hexdigest()
        except Exception as e:
            return f"tensor-hash-error:{tuple(value.shape)}:{value.dtype}:{value.device}:{e}"

    if hasattr(value, "tobytes"):
        h = hashlib.md5()
        h.update(str(getattr(value, "shape", "")).encode())
        h.update(str(getattr(value, "dtype", "")).encode())
        h.update(value.tobytes())
        return h.hexdigest()

    return hashlib.md5(str(value).encode()).hexdigest()

def _stable_tiler_value(value, depth=0):
    if depth > 8:
        return f"<max-depth:{type(value).__name__}>"

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if torch.is_tensor(value) or hasattr(value, "tobytes"):
        return {
            "type": type(value).__name__,
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")),
            "hash": _hash_tiler_tensor(value),
        }

    if isinstance(value, dict):
        return {
            str(k): _stable_tiler_value(v, depth + 1)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple, set)):
        return [_stable_tiler_value(v, depth + 1) for v in list(value)]

    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(value))
    return {"type": type(value).__name__, "repr": text}


def _fingerprint_hash(payload):
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _filtered_labs_upscaler_config(labs_upscaler_dict):
    if not isinstance(labs_upscaler_dict, dict):
        return None

    relevant_names = {
        "Tiler_Upscale_Cache",
        "Only_Upscale",
        "PID_degrade_sigma",
        "PID_Model",
        "PID_VAE_Compatible_Model",
        "PID_CLIP",
        "PID_Source_VAE",
    }
    relevant_prefixes = ("SEEDVR2_", "PID_", "Flash", "Waifu", "Upscale", "upscale")
    filtered = {}
    for key, value in labs_upscaler_dict.items():
        key_text = str(key)
        if key_text in relevant_names or key_text.startswith(relevant_prefixes):
            filtered[key_text] = _stable_tiler_value(value)
    return filtered


def _input_upscale_cache_payload(image, kwargs):
    return {
        "input_image": _stable_tiler_value(image),
        "upscale_model": kwargs.get("upscale_model", "NONE"),
        "upscale_by": kwargs.get("upscale_by", 1),
        "effective_upscale_by": getattr(tbg.PARAMS, "effective_upscale_by", kwargs.get("upscale_by", 1)),
        "optimize_upscale_factor_for_tile_use": bool(
            getattr(
                tbg.PARAMS,
                "optimize_upscale_factor_for_tile_use",
                kwargs.get("Optimize_Upscale_Factor_For_Tile_Use", False),
            )
        ),
        "labs_upscaler": _filtered_labs_upscaler_config(kwargs.get("labs_upscaler", None)),
    }


def _segment_fingerprint(segms):
    if segms is None:
        return {"count": 0, "value": None}
    try:
        count = len(segms)
    except Exception:
        count = None
    return {
        "count": count,
        "value": _stable_tiler_value(segms),
    }


def make_tiler_refresh_fingerprint(tbg, kwargs, labs_upscaler_dict):
    input_image = getattr(tbg.INPUTS, "image", None)
    upscaled_image = getattr(tbg.OUTPUTS, "upscaled_image", None)
    seg_mask = getattr(tbg.SEGMENTS, "Segment_Mask", None)

    cfg = {
        "input_image": _stable_tiler_value(input_image),
        "upscaled_image": _stable_tiler_value(upscaled_image),
        "preset": getattr(tbg.PARAMS, "preset", "NONE"),
        "fragmentation": getattr(tbg.PARAMS, "fragmentation", 1),
        "tile_size_w": getattr(tbg.SIZE, "fullW", kwargs.get("tile_size_w", 1024)),
        "tile_size_h": getattr(tbg.SIZE, "fullH", kwargs.get("tile_size_h", 1024)),
        "upscale_model": getattr(tbg.PARAMS, "upscale_model_name", kwargs.get("upscale_model", "NONE")),
        "upscale_by": getattr(tbg.PARAMS, "upscale_by", kwargs.get("upscale_by", 1)),
        "requested_upscale_by": getattr(tbg.PARAMS, "requested_upscale_by", kwargs.get("upscale_by", 1)),
        "effective_upscale_by": getattr(tbg.PARAMS, "effective_upscale_by", getattr(tbg.PARAMS, "upscale_by", kwargs.get("upscale_by", 1))),
        "optimize_upscale_factor_for_tile_use": bool(
            getattr(
                tbg.PARAMS,
                "optimize_upscale_factor_for_tile_use",
                kwargs.get("Optimize_Upscale_Factor_For_Tile_Use", False),
            )
        ),
        "fusion_mode": getattr(tbg.PARAMS, "Tile_Fusion_Mode", kwargs.get("Fusion Mode", "NONE")),
        "fusion_reference_margin": getattr(tbg.SIZE, "Fusion_reference_margin", kwargs.get("Fusion Reference Margin", 0)),
        "fusion_margin": getattr(tbg.SIZE, "inpaint_blur_margin", kwargs.get("Fusion Margin", 64)),
        "feather_mask": getattr(tbg.SIZE, "composite_blur_margin", kwargs.get("Feather Mask", 16)),
        "fusion_strength": getattr(tbg.SIZE, "inpaint_max", kwargs.get("Fusion Strength", 0.05)),
        "segment_mask": {
            "present": seg_mask is not None,
            "value": _stable_tiler_value(seg_mask),
        },
        "pro_segs": _segment_fingerprint(getattr(tbg.SEGMENTS, "segms", None)),
        "labs_upscaler": _filtered_labs_upscaler_config(labs_upscaler_dict),
    }
    return _fingerprint_hash(cfg), cfg


def _changed_fingerprint_keys(previous, current):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return ["initial"]
    keys = sorted(set(previous.keys()) | set(current.keys()))
    return [key for key in keys if previous.get(key) != current.get(key)]


def make_tiler_config(tbg, labs_upscaler_dict):
    key, _ = make_tiler_refresh_fingerprint(tbg, {}, labs_upscaler_dict)
    return key



from ....TBG.SERVERS.COMFYUI_server import register_main_class
from comfy.utils import ProgressBar
@register_main_class
class TBG_Upscaler_v1():
    _pbar = None

    @classmethod
    def _classify_segment_sampling_case(cls, bbox_w, bbox_h, crop_w, crop_h, tile_w, tile_h, scale_x, scale_y, guard_applied=False, ratio_adjustment="none"):
        scale_x = float(scale_x)
        scale_y = float(scale_y)
        min_scale = min(scale_x, scale_y)
        max_scale = max(scale_x, scale_y)
        ratio_adjustment = str(ratio_adjustment or "none")

        if bool(guard_applied):
            case = "large_context_guarded_restore"
            reason = "Flux2/large segment guard changed sampling tile size; restore through transform only"
        elif min_scale > 1.05:
            case = "small_segment_upscaled_bbox_restore"
            reason = "segment is intentionally upscaled in sampling tile for detail, then restored to native bbox/crop"
        elif max_scale < 0.95:
            case = "large_segment_downscaled_context_restore"
            reason = "segment/context is downscaled for sampling, then restored to native crop"
        else:
            case = "native_scale_segment_restore"
            reason = "segment sampling scale is close to native"
        if ratio_adjustment != "none":
            case += "_ratio_adjusted"
            reason += f"; crop ratio adjusted={ratio_adjustment}"
        return case, reason

    @classmethod
    def _simulate_grid_for_size(cls, image_h, image_w):
        size = copy.copy(tbg.SIZE)
        mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
        size_unit = 64
        size.fullW = int(math.floor(int(size.fullW) / size_unit) * size_unit)
        size.fullH = int(math.floor(int(size.fullH) / size_unit) * size_unit)
        size.inpaint_border_margin = max(int(getattr(size, "inpaint_border_margin", 0) or 0), 0)
        size.composite_blur_margin = max(int(getattr(size, "composite_blur_margin", 0) or 0), 0)

        if mode == "Tile_Fusion":
            outer_mask_area = max(
                0,
                int(getattr(size, "inpaint_border_margin", 0) or 0)
                + int(getattr(size, "inpaint_blur_margin", 0) or 0),
            )
            overlay_between_tiles = 2 * outer_mask_area
            crop_margin = outer_mask_area - int(getattr(size, "shift", 0) or 0)
        elif mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext"):
            outer_mask_area = max(0, int(getattr(size, "Fusion_reference_margin", 0) or 0))
            overlay_between_tiles = (2 * outer_mask_area) + size.composite_blur_margin
            crop_margin = outer_mask_area
        elif mode == "Soft Merge":
            outer_mask_area = max(0, int(getattr(size, "Fusion_reference_margin", 0) or 0))
            overlay_between_tiles = (2 * outer_mask_area) + size.composite_blur_margin
            crop_margin = outer_mask_area
        else:
            outer_mask_area = size.composite_blur_margin
            overlay_between_tiles = size.composite_blur_margin
            crop_margin = 0

        tile_grid_w = max(1, int(size.fullW) - int(overlay_between_tiles))
        tile_grid_h = max(1, int(size.fullH) - int(overlay_between_tiles))
        if mode == "Soft Merge":
            rows_qty = math.ceil(int(image_h) / tile_grid_h)
            cols_qty = math.ceil(int(image_w) / tile_grid_w)
        else:
            rows_qty = math.ceil((int(image_h) - outer_mask_area) / tile_grid_h)
            cols_qty = math.ceil((int(image_w) - outer_mask_area) / tile_grid_w)

        if ((rows_qty - 2) * tile_grid_h) >= (int(image_h) - int(size.fullH)):
            rows_qty -= 1
        if ((cols_qty - 2) * tile_grid_w) >= (int(image_w) - int(size.fullW)):
            cols_qty -= 1

        size.crop_margin = crop_margin
        size.tile_grid_W = tile_grid_w
        size.tile_grid_H = tile_grid_h
        size.rows_qty = max(1, int(rows_qty))
        size.cols_qty = max(1, int(cols_qty))
        size.outer_mask_area = outer_mask_area
        size.overlay_between_tiles = overlay_between_tiles
        size.actual_inner_tile_sizeW = int(size.fullW) - 2 * outer_mask_area
        size.actual_inner_tile_sizeH = int(size.fullH) - 2 * outer_mask_area
        return int(size.rows_qty), int(size.cols_qty), size

    @classmethod
    def _optimize_upscale_factor_for_tile_use(cls, input_image):
        if input_image is None or not torch.is_tensor(input_image):
            return
        requested = float(getattr(tbg.PARAMS, "requested_upscale_by", getattr(tbg.PARAMS, "upscale_by", 1.0)) or 1.0)
        if requested <= 0:
            requested = 1.0
        tbg.PARAMS.effective_upscale_by = requested
        if not bool(getattr(tbg.PARAMS, "optimize_upscale_factor_for_tile_use", False)):
            return
        if getattr(tbg.PARAMS, "upscale_model_name", "NONE") == "NONE":
            tbg.PARAMS.effective_upscale_by = 1.0
            tbg.PARAMS.upscale_by = 1.0
            return

        source_h = int(input_image.shape[1])
        source_w = int(input_image.shape[2])
        if source_h <= 0 or source_w <= 0:
            return

        requested_w = max(1, int(round(source_w * requested)))
        requested_h = max(1, int(round(source_h * requested)))
        previous_h = getattr(tbg.SIZE, "UpscaledInputImageH", None)
        previous_w = getattr(tbg.SIZE, "UpscaledInputImageW", None)
        try:
            tbg.SIZE.UpscaledInputImageH = requested_h
            tbg.SIZE.UpscaledInputImageW = requested_w
            rows, cols, size = cls._simulate_grid_for_size(requested_h, requested_w)
        finally:
            tbg.SIZE.UpscaledInputImageH = previous_h
            tbg.SIZE.UpscaledInputImageW = previous_w

        if rows <= 1 and cols <= 1:
            return

        tile_w = int(size.fullW)
        tile_h = int(size.fullH)
        step_w = max(1, int(getattr(size, "tile_grid_W", tile_w)))
        step_h = max(1, int(getattr(size, "tile_grid_H", tile_h)))
        mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
        max_w = tile_w + max(0, cols - 1) * step_w
        max_h = tile_h + max(0, rows - 1) * step_h

        # Keep one scalar scale: the image aspect ratio never changes.
        optimized = min(requested, max_w / source_w, max_h / source_h)
        optimized = max(0.05, float(optimized))
        tbg.PARAMS.effective_upscale_by = optimized
        tbg.PARAMS.upscale_by = optimized
        tbg.PARAMS.optimized_upscale_tile_grid = f"{cols}x{rows}"
        tbg.PARAMS.optimized_upscale_target_size = (
            int(round(source_w * optimized)),
            int(round(source_h * optimized)),
        )

        log(
            "Optimized upscale factor for tile use: "
            f"requested={requested:.4f} effective={optimized:.4f} "
            f"grid={cols}x{rows} target={tbg.PARAMS.optimized_upscale_target_size[0]}x{tbg.PARAMS.optimized_upscale_target_size[1]} "
            f"mode={mode} step={step_w}x{step_h}",
            "BRIGHT_BLUE",
            "BRIGHT_CYAN",
            f"Node {tbg.INFO.id}",
        )


    @classmethod
    def tbg_mark_worker_job_started(cls):
        """
        Call at the very start of any job (tiler or refiner).

        Delegates to TBG_Controller so that the worker lifetime
        is tracked globally across all tilers.
        """
        TBG_Controller.mark_job_started()

    @classmethod
    def tbg_schedule_worker_shutdown(cls, delay):
        """
        Schedule a conditional shutdown for the shared worker process.

        Uses a single global timer/last-activity in TBG_Controller so
        all tilers/refiners share the same shutdown logic.
        """
        TBG_Controller.schedule_worker_shutdown(delay)

    @classmethod
    def _apply_dev_debug_mode(cls, labs_upscaler_dict):
        mode = "dev debug on"
        if isinstance(labs_upscaler_dict, dict):
            mode = str(labs_upscaler_dict.get("Dev_Debug_Mode", mode) or mode)
        if mode not in ("dev debug on", "dev debug off", "dev free user test"):
            mode = "dev debug on"

        real_status = getattr(tbg.API, "status", "Guest")
        is_real_dev = real_status == "Dev"
        tbg.API.real_status = real_status
        tbg.API.dev_debug_mode = mode if is_real_dev else "dev debug on"
        tbg.API.dev_debug_enabled = bool(is_real_dev and mode == "dev debug on")
        tbg.API.dev_free_user_test = bool(is_real_dev and mode == "dev free user test")

        tbg.PARAMS.Dev_Debug_Mode = tbg.API.dev_debug_mode
        tbg.PARAMS.Dev_Debug_Enabled = tbg.API.dev_debug_enabled
        tbg.PARAMS.Dev_Free_User_Test = tbg.API.dev_free_user_test

        if tbg.API.dev_free_user_test:
            tbg.API.status = "Free"

    @classmethod
    def _dev_debug_enabled(cls):
        return bool(
            getattr(tbg.API, "status", None) == "Dev"
            and getattr(tbg.API, "dev_debug_enabled", True)
        )

    # Apply decorator directly to instance method
    @classmethod
    def fn(cls, **kwargs):


        # INIT TBG_APP WORKER
        global tbg, tiler_id
        tiler_id = kwargs.get('id', None)
        #reset_tbg(tiler_id)
        tbg = get_tbg(tiler_id)
        tbg.INFO.id = tiler_id
        tbg.INFO.tiler_id = tiler_id
        tbg.start_time = time.time()
        # SET WORKER TIMER for Shutdown
        if not hasattr(tbg, "WORKER_shutdown_timer"):
            tbg.WORKER_shutdown_timer = None  # threading.Timer or None
        if not hasattr(tbg, "WORKER_last_activity"):
            tbg.WORKER_last_activity = 0.0  # timestamp
        # new: mark that a job is running for this tbg
        cls.tbg_mark_worker_job_started()



        # INIT upscale_by could be changes in upscale
        tbg.PARAMS.upscale_model_name = kwargs.get('upscale_model', "NONE")
        tbg.PARAMS.requested_upscale_by = kwargs.get('upscale_by', 1)
        tbg.PARAMS.effective_upscale_by = tbg.PARAMS.requested_upscale_by
        tbg.PARAMS.optimize_upscale_factor_for_tile_use = bool(kwargs.get('Optimize_Upscale_Factor_For_Tile_Use', False))
        tbg.PARAMS.upscale_by = tbg.PARAMS.effective_upscale_by
        cls.clear_stale_refiner_params()
        if tbg.PARAMS.upscale_model_name == "NONE":
            tbg.PARAMS.upscale_by = 1
            tbg.PARAMS.effective_upscale_by = 1



        tbg.API.token = kwargs.get("PRO_api_token", None)
        WORKER.id(tiler_id).TBG_api.update(tbg.API.token,_tbg_send_images=False, )

        end_time = time.time()
        elapsed = end_time - tbg.start_time
        def get_api_infos():
            info,status,creditsleft,current_credits, info_url = WORKER.id(tiler_id).TBG_api.check_status(tbg.API.token, True,_tbg_send_images=False, )
            return  info,status,creditsleft,current_credits,info_url
        tbg.API.info, tbg.API.status, tbg.API.creditsleft, tbg.API.current_credits, tbg.API.info_url = get_api_infos()
        cls._apply_dev_debug_mode(kwargs.get('labs_upscaler', None))


        end_time = time.time()
        elapsed = end_time - tbg.start_time

        tbg.PARAMS.TBG_APP_ShutDown = kwargs.get('TBG_APP_ShutDown', 'Close with Comfyui')

        # INIT SEGMENTS
        tbg.SEGMENTS.Segment_Mask = kwargs.get('Segment_Mask', None)
        tbg.SEGMENTS.segms = kwargs.get('PRO_segs', None)

        # ADD THIS BLOCK:
        if tbg.SEGMENTS.Segment_Mask is None and tbg.SEGMENTS.segms is None:
            # No segment inputs → clear everything
            tbg.SEGMENTS.upscale_factor = None
            tbg.SEGMENTS.pad_offset = None
            tbg.SEGMENTS.segment_tiles = None
            tbg.SEGMENTS.orig_segment_tiles = None
            tbg.SEGMENTS.segms_scale = None
            tbg.SEGMENTS.segms_cropped_masks = None
            tbg.SEGMENTS.segment_binary_masks = None
            tbg.SEGMENTS.segms_crop_regions = None
            tbg.SEGMENTS.segment_sampling_transforms = None
            tbg.SEGMENTS.segms_new = None
            tbg.SEGMENTS.inpainting_mask = None
            tbg.SEGMENTS.compositing_mask = None
            tbg.SEGMENTS.h = None
            tbg.SEGMENTS.w = None

        if tbg.SEGMENTS.Segment_Mask is not None:
            SEG = MaskToSEGS(tbg.SEGMENTS.Segment_Mask, False, 2, False, 0, False)
            if tbg.SEGMENTS.segms is not None:
                tbg.SEGMENTS.segms = combine_segs(tbg.SEGMENTS.segms, SEG)
            else:
                tbg.SEGMENTS.segms = SEG
            print(f"TBG[Node {tbg.INFO.id}] Converted {len(SEG)} segment{'s' if len(SEG) != 1 else ''} to Tiles.")

        cls.init(**kwargs)
        if is_batch_image(tbg.INPUTS.image) and not Only_Upscale:
            batch_size = int(tbg.INPUTS.image.shape[0])
            log(
                f"ETUR batch mode detected: deferring tiling for {batch_size} images to the refiner",
                None,
                None,
                f"Node {tbg.INFO.id}",
            )
            batch_pipe = make_batch_pipe(tbg, kwargs, copy.copy(tbg.INFO.id), tiler_id)
            return (batch_pipe, tbg.INPUTS.image)

        # Upscale - SR upscales are overwriting the tbg so we mantain the original images and pass it later again into the tbg
        # st jetzt sauber getrennt
        #orig_image = copy.deepcopy(tbg.INPUTS.image)

        cls._optimize_upscale_factor_for_tile_use(tbg.INPUTS.image)
        tbg.OUTPUTS.upscaled_image = cls.upscale_full_input_image(tbg.INPUTS.image, **kwargs) # goes to cache wrapper with kwargs

        if Only_Upscale:
            return ((0,0,0,0,0,0,0,0,"Only Upscale",0,0,tbg.API.info_url), tbg.OUTPUTS.upscaled_image)

        #tbg.INPUTS.image = orig_image
        if tbg.OUTPUTS.upscaled_image is not None:
            tbg.SIZE.UpscaledInputImageH = tbg.OUTPUTS.upscaled_image.shape[1]
            tbg.SIZE.UpscaledInputImageW = tbg.OUTPUTS.upscaled_image.shape[2]

        # in TBG_Tiler.start_tiles or just after init()
        labs_upscaler_dict = kwargs.get('labs_upscaler', None)
        new_key, fingerprint_debug = make_tiler_refresh_fingerprint(tbg, kwargs, labs_upscaler_dict)
        tbg.PARAMS.tiler_cache_key = new_key
        last_key = getattr(tbg.TEMP, "last_tiler_config", None)
        last_debug = getattr(tbg.TEMP, "last_tiler_config_debug", None)

        if last_key == new_key:
            tbg.TEMP.skip_tiler = True
        else:
            tbg.TEMP.skip_tiler = False
            tbg.TEMP.last_tiler_config = new_key
            tbg.TEMP.last_tiler_config_debug = copy.deepcopy(fingerprint_debug)
            if cls._dev_debug_enabled():
                changed_keys = _changed_fingerprint_keys(last_debug, fingerprint_debug)
                log(
                    f"[TBG Tiler Cache] refresh fingerprint changed; rerunning tiler. changed={changed_keys}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}",
                )

        return  cls.start_tiles(**kwargs)

    @classmethod
    def start_tiles(cls, **kwargs):
        # Get padding offset from format_2_divby8
        tbg.OUTPUTS.upscaled_image, image_width, image_height, _, tbg.SEGMENTS.pad_offset = TBG_Image().format_2_divby8(tbg.OUTPUTS.upscaled_image)
        # Apply upscaling and get upscale factor
        tbg.SEGMENTS.upscale_factor = tbg.PARAMS.upscale_by if tbg.PARAMS.upscale_by else 1.0
        # Transform segments with both padding offset and upscale factor
        if tbg.SEGMENTS.segms:
            tbg.SEGMENTS.segms = TBG_Image().transform_segment_coordinates(
                tbg.SEGMENTS.segms, tbg.SEGMENTS.pad_offset, tbg.SEGMENTS.upscale_factor
            )

        # TILING
        if getattr(tbg.TEMP, "skip_tiler", False):
            # Geometry unchanged: do NOT call upscale_full_input_image or tiler
            log("Tiler is using cached tiled - input images was the same as before", None, None, f"Node {tbg.INFO.id}")
        else:
            log("Tiling", None, None, f"Node {tbg.INFO.id}")
            cls.tiler(tbg.OUTPUTS.upscaled_image, "Tiling")  # goes direct

        #cls.tiler(tbg.OUTPUTS.upscaled_image, "Tiling")  # goes direct
        # PROMPTING

        cls.prompter("Tile Overrides")


        if tbg.SEGMENTS.segment_tiles is None:
            tbg.SEGMENTS.orig_segment_tiles = None
        else:
            tbg.SEGMENTS.orig_segment_tiles = copy.deepcopy( tbg.SEGMENTS.segment_tiles)



        tbg.PARAMS.timestamp = time.time() # for cache cleaning in refiner
        node_id = copy.copy(tbg.INFO.id)
        import os
        mode = getattr(tbg.PARAMS, "TBG_APP_ShutDown", "ShutDown after each run")
        if mode == 'ShutDown after each run':
            delay = 0
            cls.tbg_schedule_worker_shutdown(delay)
            os.environ["TBG_MAIN_WATCHDOG_INTERVAL"] = "10"  # check if comfy is alive
        elif mode == 'ShutDown after each run delayed':
            delay = 60
            cls.tbg_schedule_worker_shutdown(delay)
            os.environ["TBG_MAIN_WATCHDOG_INTERVAL"] = "10"  # check if comfy is alive
        elif mode == 'Keep Running (not recommended)':
            delay = 3600  # 1 hour idle timeout as default
            cls.tbg_schedule_worker_shutdown(delay)
            os.environ["TBG_MAIN_WATCHDOG_INTERVAL"] = "0"  # check if comfy is alive
        elif mode == 'Close with Comfyui':
            os.environ["TBG_MAIN_WATCHDOG_INTERVAL"] = "10"  # check if comfy is alive
        else:
            os.environ["TBG_MAIN_WATCHDOG_INTERVAL"] = "10"  # check if comfy is alive
        

        end_time = time.time()
        elapsed = end_time - tbg.start_time
        log(
            f"Upscaler and Tile Generator completed in {elapsed:.2f} seconds",
            None,
            None,
            f"Node {tbg.INFO.id}"
        )
        tbg.PROMPTER.cache_key = None
        return ((tbg.INPUTS, tbg.PARAMS, tbg.KSAMPLER, tbg.OUTPUTS, tbg.SEGMENTS, tbg.SIZE, tbg.API, tbg.PROMPTER, tbg.API.current_credits, node_id, tiler_id, tbg.API.info_url), tbg.OUTPUTS.upscaled_image,)

    @classmethod
    def init(cls, **kwargs):
        try:
            import flash_attn
            attention_mode = 'flash_attn'
        except ImportError:
            attention_mode = 'sdpa'
        global Tiler_Upscale_Cache,Only_Upscale
        runtime_device = _runtime_device_string()
        tbg.PARAMS.Prompt_seed =  kwargs.get('VLM_seed', None)
        labs_upscaler_dict = kwargs.get('labs_upscaler', None)
        if labs_upscaler_dict:
            # Optional
            Tiler_Upscale_Cache = labs_upscaler_dict.get('Tiler_Upscale_Cache', True)
            Only_Upscale = labs_upscaler_dict.get('Only_Upscale', True)
            tbg.PARAMS.VLM_Server_Base_URL = labs_upscaler_dict.get('VLM_Server_Base_URL', "http://127.0.0.1:8080/v1")
            tbg.PARAMS.VLM_Server_Model = labs_upscaler_dict.get('VLM_Server_Model', "")
            tbg.PARAMS.PID_degrade_sigma = labs_upscaler_dict.get('PID_degrade_sigma', 0.1)
            # Keep these internal defaults for backwards compatibility in GGUF server paths.
            tbg.PARAMS.VLM_Server_API_Key_Env = "TBG_ETUR_OPENAI_API_KEY"
            tbg.PARAMS.VLM_Server_Launch_Mode = "external_only"
            tbg.PARAMS.VLM_Server_Command = ""
            tbg.PARAMS.VLM_Server_Args = ""
            tbg.PARAMS.VLM_Server_Health_Endpoint = "/v1/models"
            tbg.PARAMS.VLM_Server_Startup_Timeout_s = 30
            tbg.PARAMS.VLM_Server_Request_Timeout_s = 90
            tbg.PARAMS.VLM_Server_Stop_On_Refiner_Start = True
            tbg.PARAMS.VLM_Server_Stop_On_Run_End = True
            tbg.PARAMS.SEEDVR2_VAE = labs_upscaler_dict.get('SEEDVR2_VAE', {'model': 'ema_vae_fp16.safetensors', 'device': runtime_device, 'offload_device': 'none',
                           'cache_model': True, 'encode_tiled': True, 'encode_tile_size': 512,
                           'encode_tile_overlap': 64,
                           'decode_tiled': True,
                           'decode_tile_size': 512, 'decode_tile_overlap': 64, 'tile_debug': 'false',
                           'torch_compile_args': 'reduce-overhead', 'node_id': '9'})
            tbg.PARAMS.SEEDVR2_DIT = labs_upscaler_dict.get('SEEDVR2_DIT',  {'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                           'device': runtime_device, 'offload_device': 'none', 'cache_model': True,
                           'blocks_to_swap': 0,
                           'swap_io_components': False,
                           'attention_mode': attention_mode,
                           'torch_compile_args': 'reduce-overhead', 'node_id': '8'})
            tbg.PARAMS.SEEDVR2_DIT_low = labs_upscaler_dict.get('SEEDVR2_DIT_low',  {'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                           'device': runtime_device, 'offload_device': 'none', 'cache_model': True,
                           'blocks_to_swap': 0,
                           'swap_io_components': False,
                           'attention_mode': attention_mode,
                           'torch_compile_args': 'reduce-overhead', 'node_id': '8'})
        else:
            Tiler_Upscale_Cache = False
            if tbg.API.status in ["Free", "Pro", "Premium", "Unlimited", "Dev"]:
                Tiler_Upscale_Cache = True
            Only_Upscale = False
            tbg.PARAMS.VLM_Server_Base_URL = "http://127.0.0.1:8080/v1"
            tbg.PARAMS.VLM_Server_Model = ""
            tbg.PARAMS.PID_degrade_sigma = 0.1
            tbg.PARAMS.VLM_Server_API_Key_Env = "TBG_ETUR_OPENAI_API_KEY"
            tbg.PARAMS.VLM_Server_Launch_Mode = "external_only"
            tbg.PARAMS.VLM_Server_Command = ""
            tbg.PARAMS.VLM_Server_Args = ""
            tbg.PARAMS.VLM_Server_Health_Endpoint = "/v1/models"
            tbg.PARAMS.VLM_Server_Startup_Timeout_s = 30
            tbg.PARAMS.VLM_Server_Request_Timeout_s = 90
            tbg.PARAMS.VLM_Server_Stop_On_Refiner_Start = True
            tbg.PARAMS.VLM_Server_Stop_On_Run_End = True
        tbg.PARAMS.Inventivity=kwargs.get('Inventivity', None)
        tbg.PARAMS.Resemblance=kwargs.get('Resemblance', None)
        tbg.PARAMS.Fractality=kwargs.get('Fractality', None)
        tbg.PARAMS.Creativity=kwargs.get('Creativity', None)
        tbg.INFO.id = kwargs.get('id', None)
        tbg.INPUTS.image = kwargs.get('image', None)
        tbg.PROMPTER.Prompt_Selected_Tiles_Only = kwargs.get(
            'VLM_Process_Selected_Tiles_Only',
            kwargs.get('VLM_Selected_Tiles_Only', False),
        )
        tbg.PROMPTER.Prompt_Selected_Tiles_By_Numbers = kwargs.get(
            'VLM_Selected_Tiles_Index_Numbers',
            kwargs.get('VLM_Selected_Tiles_By_Numbers', " "),
        )
        tbg.LLM.prompt = kwargs.get('VLM_Prompt', " ")
        tbg.LLM.quantization =   kwargs.get('VLM_Quantization', "None (FP16)")
        tbg.LLM.model = _normalize_vlm_model(kwargs.get('VLM_Model', 'NONE'))
        tbg.PARAMS.VLM_Model = tbg.LLM.model
        tbg.PARAMS.VLM_Server_Active = False
        tbg.PARAMS.MODEL_TYPE_SIZES = kwargs.get('MODEL_TYPE_SIZES', False)
        tbg.PARAMS.Tile_Fusion_Mode = kwargs.get('Fusion Mode', 'NONE')
        tbg.PARAMS.upscale_model_name=kwargs.get('upscale_model', "NONE")
        tbg.PARAMS.preset = kwargs.get('presets', "NONE")
        tbg.PARAMS.fragmentation = kwargs.get('Fragmentation', 1)
        tbg.PARAMS.requested_upscale_by = getattr(tbg.PARAMS, "requested_upscale_by", kwargs.get('upscale_by', 1))
        tbg.PARAMS.effective_upscale_by = getattr(tbg.PARAMS, "effective_upscale_by", tbg.PARAMS.requested_upscale_by)
        tbg.PARAMS.optimize_upscale_factor_for_tile_use = bool(
            getattr(
                tbg.PARAMS,
                "optimize_upscale_factor_for_tile_use",
                kwargs.get('Optimize_Upscale_Factor_For_Tile_Use', False),
            )
        )
        tbg.PARAMS.upscale_by = tbg.PARAMS.effective_upscale_by

        tbg.SIZE.fullH = kwargs.get('tile_size_h', 1024)
        tbg.SIZE.fullW = kwargs.get('tile_size_w', 1024)

        if tbg.PARAMS.fragmentation and tbg.PARAMS.fragmentation != 0 or tbg.PARAMS.fragmentation != 1:
            tbg.SIZE.fullW =tbg.SIZE.fullH * tbg.PARAMS.fragmentation
            tbg.SIZE.fullH =tbg.SIZE.fullW * tbg.PARAMS.fragmentation


        tbg.SIZE.Fusion_reference_margin = kwargs.get('Fusion Reference Margin', 0)
        tbg.SIZE.Fusion_margin = kwargs.get('Fusion Margin', 64)
        if is_pid_upscale_model(tbg.PARAMS.upscale_model_name):
            if tbg.PARAMS.Tile_Fusion_Mode in (None, "NONE"):
                tbg.PARAMS.Tile_Fusion_Mode = "Neuro_Generative_Tile_Fusion"
            tbg.PARAMS.Flux2_Tile_Color_Correction = True
            tbg.PARAMS.color_match_method = PID_DEFAULT_COLOR_MATCH
            tbg.PARAMS.color_match_str = 1.0
            if tbg.SIZE.Fusion_reference_margin == 0:
                tbg.SIZE.Fusion_reference_margin = PID_DEFAULT_OVERLAP
            if tbg.SIZE.Fusion_margin == 64:
                tbg.SIZE.Fusion_margin = PID_DEFAULT_STITCH_BLUR
        # auto fusion border calculation
        tbg.SIZE.inpaint_blur_margin = tbg.SIZE.Fusion_margin
        tbg.SIZE.shift = 0
        tbg.SIZE.composite_blur_margin = kwargs.get('Feather Mask', 16)
        if is_pid_upscale_model(tbg.PARAMS.upscale_model_name) and tbg.SIZE.composite_blur_margin == 16:
            tbg.SIZE.composite_blur_margin = PID_DEFAULT_STITCH_FEATHER
        tbg.SIZE.inpaint_border_margin = tbg.SIZE.Fusion_reference_margin
        tbg.SIZE.shifttl = 0
        tbg.SIZE.inpaint_max = kwargs.get('Fusion Strength', 0.05)

    @classmethod
    @upscale_cache_comfy_method(maxsize=1, enabled=lambda: Tiler_Upscale_Cache)
    def upscale_full_input_image(cls, input_image, **kwargs):

        if input_image is None:
            raise ValueError(f"TBG Enhanced Tiled Generator id {tbg.INFO.id}: No image provided")

        log("TBG Enhanced Tiled Generator is starting", None, None, f"Node {tbg.INFO.id}")
        # log(f"Starting Upscaling  upscale_type", None, None, f"Node {tbg.INFO.id}")

        is_pid_upscale = is_pid_upscale_model(tbg.PARAMS.upscale_model_name)
        if tbg.PARAMS.upscale_model_name != "NONE" and (tbg.PARAMS.upscale_by not in (0,1) or is_pid_upscale):
            upscaled_image = copy.copy(input_image)

            if is_pid_upscale:
                requested_upscale_by = float(tbg.PARAMS.upscale_by or 1.0)
                pid_overlap = PID_DEFAULT_OVERLAP
                pid_prompt_fn = None
                pid_prompt_cleanup = None
                if tbg.LLM.model == PID_VLM_MODEL:
                    print(f"TBG[Node {tbg.INFO.id}] PID VLM prompt generation explicitly enabled by VLM_Model={tbg.LLM.model}")
                    pid_prompt_fn, pid_prompt_cleanup = make_pid_vlm_prompt_fn(
                        seed=getattr(tbg.PARAMS, "Prompt_seed", 0),
                        prompt_text=getattr(tbg.LLM, "prompt", ""),
                        model_name=PID_VLM_MODEL,
                        quantization=getattr(tbg.LLM, "quantization", "None (FP16)"),
                    )
                else:
                    print(f"TBG[Node {tbg.INFO.id}] PID VLM disabled by user")
                def rebuild_pid_tiles(pid_tiles, grid_specs, reference_image, target_width, target_height):
                    pid_tiles = [tile.detach().to("cpu", copy=True).contiguous() if torch.is_tensor(tile) else tile for tile in pid_tiles]
                    if torch.is_tensor(reference_image):
                        reference_image = reference_image.detach().to("cpu", copy=True).contiguous()
                    previous_grid_specs = getattr(tbg.PARAMS, "grid_specs", None)
                    previous_grid_prompts = getattr(tbg.PARAMS, "grid_prompts", None)
                    previous_output_prompts = getattr(tbg.PROMPTER, "output_prompts", None)
                    previous_stitch_blending = getattr(tbg.PARAMS, "stitch_blending", None)
                    previous_overlay_between_tiles = getattr(tbg.SIZE, "overlay_between_tiles", None)
                    previous_composite_blur_margin = getattr(tbg.SIZE, "composite_blur_margin", None)
                    previous_inpaint_blur_margin = getattr(tbg.SIZE, "inpaint_blur_margin", None)
                    previous_inpaint_border_margin = getattr(tbg.SIZE, "inpaint_border_margin", None)
                    try:
                        tbg.PARAMS.grid_specs = grid_specs
                        tbg.PARAMS.grid_prompts = [""] * len(grid_specs)
                        tbg.PROMPTER.output_prompts = [""] * len(grid_specs)
                        tbg.PARAMS.stitch_blending = "gpupyramid"
                        tbg.SIZE.overlay_between_tiles = pid_overlap * PID_SCALE
                        tbg.SIZE.inpaint_blur_margin = PID_DEFAULT_STITCH_BLUR
                        tbg.SIZE.composite_blur_margin = PID_DEFAULT_STITCH_FEATHER
                        tbg.SIZE.inpaint_border_margin = pid_overlap * PID_SCALE
                        worker_image = WORKER.id(tiler_id).TBG_Image
                        rebuilt, _, _ = worker_image.rebuild_final_image(
                            list(pid_tiles),
                            reference_image,
                            [],
                            [],
                            True,
                            None,
                            _tbg_send_images=False,
                        )
                        if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
                            rebuilt = rebuilt.unsqueeze(0)
                        if torch.is_tensor(rebuilt):
                            rebuilt = rebuilt[:, :target_height, :target_width, :]
                        return rebuilt
                    finally:
                        tbg.PARAMS.grid_specs = previous_grid_specs
                        tbg.PARAMS.grid_prompts = previous_grid_prompts
                        tbg.PROMPTER.output_prompts = previous_output_prompts
                        tbg.PARAMS.stitch_blending = previous_stitch_blending
                        tbg.SIZE.overlay_between_tiles = previous_overlay_between_tiles
                        tbg.SIZE.composite_blur_margin = previous_composite_blur_margin
                        tbg.SIZE.inpaint_blur_margin = previous_inpaint_blur_margin
                        tbg.SIZE.inpaint_border_margin = previous_inpaint_border_margin

                def color_match_pid_tile(reference_tile, pid_tile, method):
                    if hasattr(TBG_Image, "detail_preserving_colormatch"):
                        return TBG_Image.detail_preserving_colormatch(
                            reference_tile,
                            pid_tile,
                            method,
                            1.0,
                            label="tiler_pid_tile_color",
                        )[0]
                    return TBG_Image.colormatch(reference_tile, pid_tile, method, 1.0)[0]

                try:
                    upscaled_image, pid_meta = run_pid_tiled_upscale(
                        upscaled_image,
                        tbg.PARAMS.upscale_model_name,
                        prompt_text=getattr(tbg.LLM, "prompt", ""),
                        seed=getattr(tbg.PARAMS, "Prompt_seed", 0),
                        rebuild_fn=rebuild_pid_tiles,
                        pid_model=labs_upscaler_dict.get("PID_Model", None) if labs_upscaler_dict else None,
                        pid_model_type=labs_upscaler_dict.get("PID_VAE_Compatible_Model", None) if labs_upscaler_dict else None,
                        clip=labs_upscaler_dict.get("PID_CLIP", None) if labs_upscaler_dict else None,
                        source_vae=labs_upscaler_dict.get("PID_Source_VAE", None) if labs_upscaler_dict else None,
                        overlap=pid_overlap,
                        degrade_sigma=getattr(tbg.PARAMS, "PID_degrade_sigma", 0.1),
                        color_match_fn=color_match_pid_tile,
                        color_match_method=PID_DEFAULT_COLOR_MATCH,
                        prompt_fn=pid_prompt_fn,
                    )
                finally:
                    if pid_prompt_cleanup is not None:
                        pid_prompt_cleanup()
                final_width = max(1, int(round(input_image.shape[2] * requested_upscale_by)))
                final_height = max(1, int(round(input_image.shape[1] * requested_upscale_by)))
                if int(upscaled_image.shape[2]) != final_width or int(upscaled_image.shape[1]) != final_height:
                    upscaled_image = nodes.ImageScale().upscale(
                        upscaled_image,
                        "lanczos",
                        final_width,
                        final_height,
                        False,
                    )[0]
                tbg.PARAMS.upscale_by = requested_upscale_by
                tbg.PARAMS.pid_internal_scale_x = float(pid_meta.get("effective_scale_x", requested_upscale_by))
                tbg.PARAMS.pid_internal_scale_y = float(pid_meta.get("effective_scale_y", requested_upscale_by))
                tbg.PARAMS.pid_tile_count = int(pid_meta.get("tile_count", 0))
                tbg.PARAMS.pid_overlap = int(pid_meta.get("overlap", pid_overlap))
                tbg.PARAMS.pid_degrade_sigma = float(pid_meta.get("degrade_sigma", getattr(tbg.PARAMS, "PID_degrade_sigma", 0.1)))
                tbg.PARAMS.pid_color_match_method = pid_meta.get("color_match_method", PID_DEFAULT_COLOR_MATCH)

            #GAN MODELS
            elif tbg.PARAMS.upscale_model_name != None and tbg.PARAMS.upscale_model_name not in tbg.upscale_models:
                if hasattr(nodes_upscale_model_UpscaleModelLoader, "execute"):
                    tbg.PARAMS.upscale_model = \
                    nodes_upscale_model_UpscaleModelLoader_execute(tbg.PARAMS.upscale_model_name)[0]
                elif hasattr(nodes_upscale_model_UpscaleModelLoader, "load_model"):
                    tbg.PARAMS.upscale_model = \
                    nodes_upscale_model_UpscaleModelLoader_execute(0, tbg.PARAMS.upscale_model_name)[0]
                if hasattr(nodes_upscale_model_ImageUpscaleWithModel, "execute"):
                    upscaled_image = nodes_upscale_model_ImageUpscaleWithModel_execute(tbg.PARAMS.upscale_model, upscaled_image)[0]
                elif hasattr(nodes_upscale_model_ImageUpscaleWithModel, "upscale"):
                    upscaled_image = nodes_upscale_model_ImageUpscaleWithModel_execute(0,tbg.PARAMS.upscale_model, upscaled_image)[0]
                upscaled_image = nodes.ImageScale().upscale(upscaled_image, "bilinear",
                                                            int(input_image.shape[2] * tbg.PARAMS.upscale_by),
                                                            int(input_image.shape[1] * tbg.PARAMS.upscale_by),
                                                            False)[0]
            # SuperResolution MODELS SeedVR2
            elif tbg.PARAMS.upscale_model_name.startswith("SuperResolution/Tiled-SeedVR2"):
                upscaled_image = nodes.ImageScale().upscale(upscaled_image, "bilinear",
                                                            int(input_image.shape[2] * tbg.PARAMS.upscale_by),
                                                            int(input_image.shape[1] * tbg.PARAMS.upscale_by),
                                                            False)[0]
                #upscaled_image = cls.min_final_imagesize(upscaled_image, 1024)
                def get_upscaled_image(upscaled_image):
                    upscaled_image = WORKER.id(tiler_id).TBG_SuperResolution.seed_vr2_upscale(upscaled_image,tbg.PARAMS.upscale_model_name)
                    return upscaled_image
                upscaled_image = get_upscaled_image(upscaled_image)

            # SuperResolution MODELS FlashVSR
            elif tbg.PARAMS.upscale_model_name.startswith("SuperResolution/FlashVSR-v1.1"):

                upscaled_image = nodes.ImageScale().upscale(upscaled_image, "bilinear",
                                                            int(input_image.shape[2] * tbg.PARAMS.upscale_by),
                                                            int(input_image.shape[1] * tbg.PARAMS.upscale_by),
                                                            False)[0]
                #upscaled_image = cls.min_final_imagesize(upscaled_image, 1536)
                def get_upscaled_image(upscaled_image):
                    upscaled_image = WORKER.id(tiler_id).TBG_SuperResolution.flash_vsr_upscale(upscaled_image, tbg.PARAMS.upscale_model_name, tbg.PARAMS.upscale_by)
                    return upscaled_image
                upscaled_image = get_upscaled_image(upscaled_image)
            elif tbg.PARAMS.upscale_model_name.startswith("Waifu"):
                print(tbg.PARAMS.upscale_model_name,"tbg.PARAMS.upscale_model_name")
                upscaled_image = nodes.ImageScale().upscale(upscaled_image, "bilinear",
                                                            int(input_image.shape[2] * tbg.PARAMS.upscale_by),
                                                            int(input_image.shape[1] * tbg.PARAMS.upscale_by),
                                                            False)[0]
                #upscaled_image = cls.min_final_imagesize(upscaled_image, 1024)
                def get_upscaled_image(upscaled_image):
                    upscaled_image = WORKER.id(tiler_id).TBG_SuperResolution.Waifu_upscale(upscaled_image,tbg.PARAMS.upscale_model_name,tbg.PARAMS.upscale_by)
                    return upscaled_image

                upscaled_image = get_upscaled_image(upscaled_image)

            elif tbg.PARAMS.upscale_model_name in ("FAST/_area","FAST/_bicubic","FAST/_bilinear","FAST/_bislerp","FAST/_lanczos","FAST/_nearest-exact"):
                    tbg.PARAMS.upscaler_method = tbg.PARAMS.upscale_model_name.replace("FAST/_", "").strip()
                    upscaled_image = nodes.ImageScale().upscale(upscaled_image, tbg.PARAMS.upscaler_method,
                                                                int(input_image.shape[2] * tbg.PARAMS.upscale_by),
                                                                int(input_image.shape[1] * tbg.PARAMS.upscale_by),
                                                                False)[0]

        else:
            upscaled_image = copy.copy(input_image)

        return upscaled_image

    @classmethod
    def clear_stale_refiner_params(cls):
        for name in (
            "denoise_mask",
            "Redux_Style_Model",
            "Redux_Clip_Vision",
            "upscale_model",
            "upscale_model_inpainting",
        ):
            if hasattr(tbg.PARAMS, name):
                setattr(tbg.PARAMS, name, None)

    @classmethod
    def _worker_tiler_init_args(cls):
        params = copy.copy(tbg.PARAMS)
        size = copy.copy(tbg.SIZE)
        cls.clear_stale_refiner_params()
        for name in (
            "denoise_mask",
            "Redux_Style_Model",
            "Redux_Clip_Vision",
            "upscale_model",
            "upscale_model_inpainting",
        ):
            if hasattr(params, name):
                setattr(params, name, None)
        return (
            TBG_Refiner_v1._worker_cpu_value(params),
            TBG_Refiner_v1._worker_cpu_value(size),
        )

    @classmethod
    def _segment_sampling_transform(cls, seg, index_seg, full_image):
        if torch.is_tensor(full_image) and full_image.dim() == 3:
            full_image = full_image.unsqueeze(0)
        base_tile_w = max(8, int(getattr(tbg.SIZE, "fullW", 1024) or 1024))
        base_tile_h = max(8, int(getattr(tbg.SIZE, "fullH", 1024) or 1024))
        max_segment_side = max(
            8,
            int(getattr(tbg.PARAMS, "max_upscale_size_segment_inpainting", 2048) or 2048),
        )
        max_segment_side = max(max_segment_side, base_tile_w, base_tile_h)
        image_w = int(full_image.shape[2])
        image_h = int(full_image.shape[1])

        bx1, by1, bx2, by2 = [int(round(float(v))) for v in seg.bbox]
        bx1 = max(0, min(image_w, bx1))
        by1 = max(0, min(image_h, by1))
        bx2 = max(bx1 + 1, min(image_w, bx2))
        by2 = max(by1 + 1, min(image_h, by2))
        bbox_w = max(1, bx2 - bx1)
        bbox_h = max(1, by2 - by1)

        margin_info = cls._segment_sampling_margins()
        reserved_margin = int(margin_info.get("reserved_margin", 0))
        required_w = bbox_w + (reserved_margin * 2)
        required_h = bbox_h + (reserved_margin * 2)

        usable_w = max(1, max_segment_side - (reserved_margin * 2))
        usable_h = max(1, max_segment_side - (reserved_margin * 2))
        scale = max(0.01, min(1.5, usable_w / float(bbox_w), usable_h / float(bbox_h)))
        tile_w = cls._align_segment_size(int(round((bbox_w * scale) + (reserved_margin * 2))), base_tile_w, max_segment_side)
        tile_h = cls._align_segment_size(int(round((bbox_h * scale) + (reserved_margin * 2))), base_tile_h, max_segment_side)
        original_tile_w = int(tile_w)
        original_tile_h = int(tile_h)
        flux2_segment_size_guard = False
        model_markers = [
            getattr(tbg.PARAMS, "model_type", None),
            getattr(tbg.KSAMPLER, "model_type", None),
            getattr(tbg.KSAMPLER, "model_name", None),
            getattr(tbg.PARAMS, "checkpoint", None),
            getattr(tbg.PARAMS, "unet_name", None),
            getattr(tbg.PARAMS, "clip_name", None),
            getattr(tbg.PARAMS, "text_encoder", None),
        ]
        model_marker_text = " ".join(str(value) for value in model_markers if value is not None).upper()
        is_flux2_segment = (
            "FLUX2" in model_marker_text
            or "FLUX 2" in model_marker_text
            or bool(getattr(tbg.PARAMS, "Flux2_Tile_Color_Correction", False))
        )
        needs_flux2_segment_size_guard = (
            is_flux2_segment
            and (
                tile_w > base_tile_w
                or tile_h > base_tile_h
                or tile_w % 64 != 0
                or tile_h % 64 != 0
            )
        )
        if needs_flux2_segment_size_guard:
            tile_w, tile_h = cls._align_flux2_oversized_segment_size(
                tile_w,
                tile_h,
                max_segment_side,
                max_pixels=4194304,
            )
            flux2_segment_size_guard = (tile_w != original_tile_w or tile_h != original_tile_h)
        crop_w = max(8, min(image_w, max(required_w, int(round(tile_w / scale)))))
        crop_h = max(8, min(image_h, max(required_h, int(round(tile_h / scale)))))
        crop_w, crop_h, crop_ratio_adjustment = cls._adjust_segment_crop_to_tile_ratio(
            crop_w,
            crop_h,
            tile_w,
            tile_h,
            min(required_w, image_w),
            min(required_h, image_h),
            image_w,
            image_h,
        )
        requested_crop_w = int(crop_w)
        requested_crop_h = int(crop_h)

        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        crop_x1 = int(round(cx - crop_w / 2.0))
        crop_y1 = int(round(cy - crop_h / 2.0))
        crop_x1 = max(0, min(max(0, image_w - crop_w), crop_x1))
        crop_y1 = max(0, min(max(0, image_h - crop_h), crop_y1))
        crop_x2 = min(image_w, crop_x1 + crop_w)
        crop_y2 = min(image_h, crop_y1 + crop_h)
        crop_w = max(1, crop_x2 - crop_x1)
        crop_h = max(1, crop_y2 - crop_y1)

        scale_x = tile_w / float(crop_w)
        scale_y = tile_h / float(crop_h)
        stretch_warning = abs(scale_x - scale_y) > 0.001
        segment_case, segment_case_reason = cls._classify_segment_sampling_case(
            bbox_w,
            bbox_h,
            crop_w,
            crop_h,
            tile_w,
            tile_h,
            scale_x,
            scale_y,
            guard_applied=flux2_segment_size_guard,
            ratio_adjustment=crop_ratio_adjustment,
        )
        bbox_tile_x1 = int(round((bx1 - crop_x1) * scale_x))
        bbox_tile_y1 = int(round((by1 - crop_y1) * scale_y))
        bbox_tile_x2 = int(round((bx2 - crop_x1) * scale_x))
        bbox_tile_y2 = int(round((by2 - crop_y1) * scale_y))
        bbox_tile_x1 = max(0, min(tile_w - 1, bbox_tile_x1))
        bbox_tile_y1 = max(0, min(tile_h - 1, bbox_tile_y1))
        bbox_tile_x2 = max(bbox_tile_x1 + 1, min(tile_w, bbox_tile_x2))
        bbox_tile_y2 = max(bbox_tile_y1 + 1, min(tile_h, bbox_tile_y2))

        context_crop = full_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
        context_tile = nodes.ImageScale().upscale(context_crop, "bicubic", tile_w, tile_h, False)[0]

        transform = {
            "native_bbox": (bx1, by1, bx2, by2),
            "native_bbox_size": (bbox_w, bbox_h),
            "native_crop_region": tuple(int(round(float(v))) for v in seg.crop_region),
            "sampling_crop_region": (crop_x1, crop_y1, crop_x2, crop_y2),
            "sampling_tile_size": (tile_w, tile_h),
            "full_image_size": (image_w, image_h),
            "sampling_scale": min(scale_x, scale_y),
            "sampling_scale_xy": (scale_x, scale_y),
            "bbox_in_sampling_tile": (bbox_tile_x1, bbox_tile_y1, bbox_tile_x2, bbox_tile_y2),
            "sampling_margin": reserved_margin,
            "sampling_margins": margin_info,
            "max_segment_side": max_segment_side,
            "flux2_oversized_guard": flux2_segment_size_guard,
            "flux2_segment_size_guard": flux2_segment_size_guard,
            "requested_sampling_tile_size": (original_tile_w, original_tile_h),
            "requested_sampling_crop_size": (requested_crop_w, requested_crop_h),
            "crop_ratio_adjustment": crop_ratio_adjustment,
            "stretch_warning": stretch_warning,
            "segment_case": segment_case,
            "segment_case_reason": segment_case_reason,
        }

        if cls._dev_debug_enabled():
            if needs_flux2_segment_size_guard:
                print(
                    f"[TBG SegmentSampling] segment size guard for Flux2 {index_seg + 1}: "
                    f"user_tile={base_tile_w}x{base_tile_h} "
                    f"requested={original_tile_w}x{original_tile_h} "
                    f"adjusted={tile_w}x{tile_h} expanded_crop={requested_crop_w}x{requested_crop_h} "
                    f"actual_crop={crop_w}x{crop_h} pixels={tile_w * tile_h} "
                    f"guard_applied={flux2_segment_size_guard}"
                )
                if stretch_warning:
                    print(
                        f"[TBG SegmentSampling] WARNING Flux2 segment {index_seg + 1}: "
                        f"image borders prevented exact uniform scale; "
                        f"scale_x={scale_x:.6f} scale_y={scale_y:.6f}"
                    )
            print(
                f"[TBG SegmentSampling] tiler context crop segment {index_seg + 1}: "
                f"bbox={bbox_w}x{bbox_h} crop={crop_w}x{crop_h} tile={tile_w}x{tile_h} "
                f"scale={min(scale_x, scale_y):.4f} margin={reserved_margin} max={max_segment_side} "
                f"actual=({scale_x:.4f},{scale_y:.4f}) ratio_adjust={crop_ratio_adjustment} "
                f"case={segment_case}"
            )
            print(
                f"[TBG SegmentSampling] segment case {index_seg + 1}: "
                f"{segment_case} - {segment_case_reason}"
            )
        return context_tile, transform

    @classmethod
    def _align_segment_size(cls, value, minimum, maximum):
        value = max(int(minimum), int(value))
        value = min(int(maximum), value)
        aligned = int(math.ceil(value / 8.0) * 8)
        if aligned > int(maximum):
            aligned = int(maximum) - (int(maximum) % 8)
        return max(8, aligned)

    @classmethod
    def _adjust_segment_crop_to_tile_ratio(cls, crop_w, crop_h, tile_w, tile_h, min_w, min_h, image_w, image_h):
        crop_w = max(1, min(int(image_w), int(round(float(crop_w)))))
        crop_h = max(1, min(int(image_h), int(round(float(crop_h)))))
        min_w = max(1, min(int(image_w), int(round(float(min_w)))))
        min_h = max(1, min(int(image_h), int(round(float(min_h)))))
        tile_w = max(1, int(round(float(tile_w))))
        tile_h = max(1, int(round(float(tile_h))))

        target_ratio = tile_w / float(tile_h)
        current_ratio = crop_w / float(crop_h)
        if abs(current_ratio - target_ratio) <= 0.001:
            return crop_w, crop_h, "none"

        adjustment = "none"
        if current_ratio < target_ratio:
            grown_w = max(crop_w, int(round(crop_h * target_ratio)))
            if grown_w <= image_w:
                crop_w = grown_w
                adjustment = "grow_width"
            else:
                reduced_h = int(math.floor(crop_w / target_ratio))
                crop_h = max(min_h, min(crop_h, reduced_h))
                adjustment = "reduce_height_border_limited"
        else:
            grown_h = max(crop_h, int(round(crop_w / target_ratio)))
            if grown_h <= image_h:
                crop_h = grown_h
                adjustment = "grow_height"
            else:
                reduced_w = int(math.floor(crop_h * target_ratio))
                crop_w = max(min_w, min(crop_w, reduced_w))
                adjustment = "reduce_width_border_limited"

        crop_w = max(min_w, min(int(image_w), int(crop_w)))
        crop_h = max(min_h, min(int(image_h), int(crop_h)))
        return crop_w, crop_h, adjustment

    @classmethod
    def _align_flux2_oversized_segment_size(cls, width, height, max_side, max_pixels=4194304):
        width = max(64, int(round(float(width))))
        height = max(64, int(round(float(height))))
        _ = max(64, int(round(float(max_side))))
        max_pixels = max(64 * 64, int(max_pixels))

        rounded_w = max(64, int(math.ceil(width / 64.0) * 64))
        rounded_h = max(64, int(math.ceil(height / 64.0) * 64))
        if rounded_w * rounded_h <= max_pixels:
            return rounded_w, rounded_h

        scale = min(
            1.0,
            math.sqrt(max_pixels / float(max(width * height, 1))),
        )
        width = max(64, int(math.floor((width * scale) / 64.0) * 64))
        height = max(64, int(math.floor((height * scale) / 64.0) * 64))

        while width * height > max_pixels and (width > 64 or height > 64):
            if width >= height and width > 64:
                width -= 64
            elif height > 64:
                height -= 64
            else:
                break

        return max(64, width), max(64, height)

    @classmethod
    def _segment_sampling_margins(cls):
        reference = min(16, int(max(
            0,
            round(float(max(
                getattr(tbg.SIZE, "Fusion_reference_margin", 0) or 0,
                getattr(tbg.SIZE, "inpaint_border_margin", 0) or 0,
            )))
        )))
        fusion = min(64, int(max(
            0,
            round(float(max(
                getattr(tbg.SIZE, "Fusion_margin", 0) or 0,
                getattr(tbg.SIZE, "inpaint_blur_margin", 0) or 0,
            )))
        )))
        feather = min(24, int(max(
            0,
            round(float(max(
                getattr(tbg.SIZE, "composite_blur_margin", 0) or 0,
                getattr(tbg.SIZE, "Feather_Mask", 0) or 0,
            )))
        )))
        mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
        if mode == "Soft Merge":
            reserved = feather
        elif mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion"):
            reserved = reference + fusion + 8
        else:
            reserved = 0
        return {
            "mode": mode,
            "reference": reference,
            "fusion": fusion,
            "feather": feather,
            "safety": 8 if reserved > 0 else 0,
            "reserved_margin": reserved,
        }

    @classmethod
    #@tiler_cache_comfy_method(maxsize=1)
    def tiler(cls, full_upscaled_image, iteration):

        tbg.OUTPUTS.upscaled_image = full_upscaled_image
        def get_tiler_init():
            worker_params, worker_size = cls._worker_tiler_init_args()
            PARAMS, SIZE = WORKER.id(tiler_id).ETUR.tiler_init(worker_params, worker_size)
            return PARAMS, SIZE
        tbg.PARAMS, tbg.SIZE = get_tiler_init()

        grid_images = TBG_Image().gridspecs_get_grid_images(tbg.OUTPUTS.upscaled_image, tbg.PARAMS.grid_specs)

        if (tbg.SEGMENTS.segms and len(tbg.SEGMENTS.segms[0]) != 0):
            # FIRST: Apply coordinate transformation for padding/upscaling
            if hasattr(cls, 'transform_segment_coordinates'):
                # Get the same padding/upscaling parameters used for the image
                pad_offset = tbg.SEGMENTS.pad_offset  # From your image processing
                upscale_factor = tbg.SEGMENTS.upscale_factor  # From your image processing

                transformed_segments = tbg.transform_segment_coordinates(
                    tbg.SEGMENTS.segms, pad_offset, upscale_factor
                )
            else:
                transformed_segments = tbg.SEGMENTS.segms

            # THEN: Apply existing div8 processing



            upscaled_segments = TBG_Segms.upscale_segm_to_match_div8_and_upscalebysettings(
                transformed_segments, tbg.OUTPUTS.upscaled_image
            )
            accepted_segments = []
            segment_tiles = []
            segment_sampling_transforms = []
            for i, seg in enumerate(upscaled_segments[1]):
                segment_tile, segment_transform = cls._segment_sampling_transform(seg, i, tbg.OUTPUTS.upscaled_image)
                if segment_tile is None or segment_transform is None:
                    continue
                accepted_segments.append(seg)
                segment_tiles.append(segment_tile)
                segment_sampling_transforms.append(segment_transform)

            # get updated grid_specs

            crop_regions = [seg.crop_region for seg in accepted_segments]

            h, w = upscaled_segments[0]
            def get_workers_grid_specs(h, w, crop_regions):
                grid_specs = WORKER.id(tiler_id).TBG_Segms.create_grid_specs_for_segments(h, w, crop_regions,
                                                                                        tbg.PARAMS.grid_specs,
                                                                                        tbg.SIZE.rows_qty,
                                                                                        tbg.SIZE.cols_qty,_tbg_send_images=False, )
                return grid_specs

            tbg.PARAMS.grid_specs = get_workers_grid_specs(h, w, crop_regions) if crop_regions else tbg.PARAMS.grid_specs

            # create array of tiles and mask

            segs_cropped_masks = []
            segment_binary_masks = []
            segs_scale = []
            segment_inpainting_mask = []
            segment_compositing_mask = []
            for seg, segment_transform in zip(accepted_segments, segment_sampling_transforms):
                segs_cropped_masks.append(seg.cropped_mask)  # composite mask - upscaled mask
                segment_binary_masks.append(getattr(seg, "binary_mask", None))
                segment_inpainting_mask.append(seg.inpainting_mask)
                segment_compositing_mask.append(seg.compositing_mask)
                segs_scale.append(segment_transform.get("sampling_scale", 1.0))
            tbg.PARAMS.segs_scale = segs_scale
            tbg.OUTPUTS.grid_images_all = (grid_images or []) + (segment_tiles or [])
            tbg.OUTPUTS.orig_grid_images_all = copy.copy(tbg.OUTPUTS.grid_images_all)

            tbg.SEGMENTS.segment_tiles = segment_tiles
            tbg.PARAMS.len_segments = len(segment_tiles)
            tbg.SEGMENTS.segms_scale = segs_scale
            tbg.SEGMENTS.segms_cropped_masks = segs_cropped_masks
            tbg.SEGMENTS.segment_binary_masks = segment_binary_masks
            tbg.SEGMENTS.segms_new = (upscaled_segments[0], accepted_segments)
            tbg.SEGMENTS.segms_crop_regions = crop_regions
            tbg.SEGMENTS.segment_sampling_transforms = segment_sampling_transforms
            tbg.SEGMENTS.inpainting_mask = segment_inpainting_mask
            tbg.SEGMENTS.compositing_mask = segment_compositing_mask
            if cls._dev_debug_enabled():
                print(
                    f"[TBG SegmentIndex] normal_tiles={len(grid_images)} "
                    f"segments={len(segment_tiles)} total={len(tbg.OUTPUTS.grid_images_all)} "
                    f"first_segment_index={len(grid_images) if segment_tiles else 'none'}"
                )

        else:
            tbg.OUTPUTS.grid_images_all = grid_images + [] # + to capy so we can clear grid_image
            tbg.OUTPUTS.orig_grid_images_all = copy.copy(grid_images)
            tbg.SEGMENTS.segment_tiles  = None
            tbg.PARAMS.len_segments = 0
            tbg.PARAMS.segs_scale = None
            tbg.SEGMENTS.segms_cropped_masks = None
            tbg.SEGMENTS.segment_binary_masks = None
            tbg.SEGMENTS.segms_new = None
            tbg.SEGMENTS.segms_crop_regions = None
            tbg.SEGMENTS.segment_sampling_transforms = None
            tbg.SEGMENTS.inpainting_mask = None
            tbg.SEGMENTS.compositing_mask = None

        # reset to save memory
        tbg.PARAMS.len_grid_images = len(grid_images)
        grid_images.clear()


    @classmethod
    def prompter(cls, iteration):
        print("len(tbg.OUTPUTS.grid_images_all)",len(tbg.OUTPUTS.grid_images_all))
        grid_prompts = []
        qwen = None
        QwenVL = None
        QwenVLGGUF = None
        OpenAIServer = None
        openai_server_model = ""
        florence_loader = None
        if tbg.LLM.model.startswith("GGUF\\"):
            print(f"[QwenVL-GGUF] Selected backend=gguf model='{tbg.LLM.model}'")
        elif tbg.LLM.model.startswith("Qwen"):
            print(f"[QwenVL] Selected backend=transformers model='{tbg.LLM.model}'")
        if not tbg.LLM.model == "NONE":
            if tbg.LLM.model == OPENAI_COMPAT_MODEL:
                base_url = str(getattr(tbg.PARAMS, "VLM_Server_Base_URL", "") or "").strip()
                openai_server_model = str(getattr(tbg.PARAMS, "VLM_Server_Model", "") or "").strip()
                if not base_url:
                    raise ValueError(
                        "[OpenAI-VLM] VLM_Model is 'OpenAI-Compatible (Labs Server)' but Labs VLM_Server_Base_URL is empty."
                    )
                if not openai_server_model:
                    raise ValueError(
                        "[OpenAI-VLM] VLM_Model is 'OpenAI-Compatible (Labs Server)' but Labs VLM_Server_Model is empty."
                    )
                OpenAIServer = QwenVLServerClient(
                    {
                        "base_url": base_url,
                        "api_key_env": "TBG_ETUR_OPENAI_API_KEY",
                        "health_endpoint": "/v1/models",
                        "request_timeout_s": float(getattr(tbg.PARAMS, "VLM_Server_Request_Timeout_s", 90)),
                    }
                )
                try:
                    status, _ = OpenAIServer.health_check()
                except Exception as e:
                    raise RuntimeError(
                        f"[OpenAI-VLM] OpenAI-compatible server is not reachable at '{base_url}': {e}"
                    ) from e
                if status < 200 or status >= 300:
                    raise RuntimeError(
                        f"[OpenAI-VLM] OpenAI-compatible server health check failed at '{base_url}' status={status}"
                    )
                try:
                    available_models = OpenAIServer.list_models()
                except Exception as models_err:
                    raise RuntimeError(
                        f"[OpenAI-VLM] Connected but failed to read models from '{base_url}'. {models_err}"
                    ) from models_err
                if openai_server_model not in available_models:
                    preview = ", ".join(available_models[:25])
                    if len(available_models) > 25:
                        preview = f"{preview}, ... (+{len(available_models) - 25} more)"
                    raise RuntimeError(
                        "[OpenAI-VLM] OpenAI-compatible VLM model not found.\n"
                        f"Requested model: '{openai_server_model}'\n"
                        f"Server URL: '{base_url}'\n"
                        f"Available models: {preview or '(none returned)'}"
                    )
                print(
                    f"[OpenAI-VLM] Selected backend=openai model='{openai_server_model}' base_url='{base_url}'"
                )
            if tbg.LLM.model.startswith("GGUF\\"):
                QwenVLGGUF = QwenVLGGUFBase()
            if "sky" in tbg.LLM.model.lower():
                qwen = Qwen2VL_TBG()
            if tbg.LLM.model.startswith("Qwen"):
                QwenVL = QwenVLBase()
            if "florence" in tbg.LLM.model.lower():
                florence_loader = DownloadAndLoadFlorence2Model()
                florence_run = Florence2Run()
                if "large" in tbg.LLM.model.lower():
                    florence_task = "prompt_gen_mixed_caption"
                else:
                    florence_task = "detailed_caption"

        def get_prompt_tile(grid_image):
            # rescale image for LLM to 1 megapixel and multibyte 28
            image = copy.copy(grid_image)
            samples = image.movedim(-1, 1)

            # target total pixels ~ 1 megapixel
            target_pixels = 1024 * 1024

            scale_by = math.sqrt(target_pixels / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            # upscale
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]

            vlm_prompt = tbg.LLM.prompt
            llm_model = tbg.LLM.model  # modelname

            if llm_model.startswith("GGUF\\"):
                seed = tbg.PARAMS.Prompt_seed
                print(f"[QwenVL-GGUF] VLM_Quantization is ignored for GGUF models: '{tbg.LLM.quantization}'")
                return QwenVLGGUF.run(
                    llm_model,
                    tbg.LLM.quantization,
                    preset_prompt="🖼️ Detailed Analysis",
                    custom_prompt=vlm_prompt,
                    image=images[0],
                    video=None,
                    frame_count=16,
                    max_tokens=1024,
                    temperature=0.6,
                    top_p=0.9,
                    num_beams=1,
                    repetition_penalty=1.2,
                    seed=seed,
                    keep_model_loaded=True,
                    attention_mode="auto",
                    use_torch_compile=False,
                    device="auto",
                )[0]
            elif llm_model == OPENAI_COMPAT_MODEL:
                seed = tbg.PARAMS.Prompt_seed
                return OpenAIServer.run_with_model(
                    model_name=openai_server_model,
                    custom_prompt=vlm_prompt,
                    image=images[0],
                    seed=seed,
                )[0]

            elif tbg.LLM.model.startswith("Qwen"):


                if tbg.LLM.model.endswith("-FP8"):
                    tbg.LLM.quantization = "None (FP16)"

                seed =  tbg.PARAMS.Prompt_seed
                return QwenVL.run(
                    tbg.LLM.model,
                    tbg.LLM.quantization,
                    preset_prompt="🖼️ Detailed Analysis",
                    custom_prompt=vlm_prompt,
                    image=images[0],
                    video=None,
                    frame_count=16,
                    max_tokens=1024,
                    temperature=0.6,
                    top_p=0.9,
                    num_beams=1,
                    repetition_penalty=1.2,
                    seed=seed,
                    keep_model_loaded=True,
                    attention_mode="auto",
                    use_torch_compile=True,
                    device="auto",
                )[0]

            elif "florence" in tbg.LLM.model.lower():
                try:
                    import flash_attn
                    florence_attention = 'flash_attention_2'
                except ImportError:
                    florence_attention = 'sdpa'
                florence2_model = florence_loader.loadmodel(tbg.LLM.model, 'fp16', florence_attention, lora=None,
                                                            convert_to_safetensors=False)[0]
                seed =  tbg.PARAMS.Prompt_seed
                return florence_run.encode(images[0], "", florence2_model, florence_task, fill_mask=False,
                                           keep_model_loaded=True,
                                           num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="",
                                           seed=seed)[2]
            elif llm_model == "SkyCaptioner-V1":
                return \
                qwen.inference(vlm_prompt, "SkyCaptioner-V1", "none", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "SkyCaptioner-V1_4bit":
                return \
                qwen.inference(vlm_prompt, "SkyCaptioner-V1", "4bit", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "SkyCaptioner-V1_8bit":
                return \
                qwen.inference(vlm_prompt, "SkyCaptioner-V1", "8bit", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "Apple FastVLM 7B Research use only":
                from nodes import NODE_CLASS_MAPPINGS
                node_name = "FastVLM7BNode"
                if node_name in NODE_CLASS_MAPPINGS:
                    FastVLM7BNode = NODE_CLASS_MAPPINGS[node_name]
                    return FastVLM7BNode.inference(0, images, vlm_prompt, 200)[0]
            else:
                raise ValueError(f"Unsupported VLM Model: {llm_model}")

        # prompt_context = llm.vision_llm.generate_prompt(image)
        prompt_context = ""
        total = len(tbg.OUTPUTS.grid_images_all)

        def get_worker_tiles_to_process():
            tiles_to_process = WORKER.id(tiler_id).TBG_Image.set_tiles_to_process(
                tbg.PROMPTER.Prompt_Selected_Tiles_Only,
                len(tbg.OUTPUTS.grid_images_all),
                tbg.PROMPTER.Prompt_Selected_Tiles_By_Numbers,
                True,
                _tbg_send_images=False,  # NEW: scalar-only call
            )
            return tiles_to_process

        tbg.PROMPTER.tiles_to_process = get_worker_tiles_to_process()
        vlm_selected_only = bool(tbg.PROMPTER.Prompt_Selected_Tiles_Only)
        vlm_selected_tiles = list(tbg.PROMPTER.tiles_to_process or [])
        vlm_selected_tiles_human = [i + 1 for i in vlm_selected_tiles if isinstance(i, int) and i >= 0]

        if tbg.LLM.model == "NONE":
            print(f"TBG[Node {tbg.INFO.id}] VLM disabled by user")
        else:
            print(
                f"TBG[Node {tbg.INFO.id}] VLM {tbg.LLM.model} working  of {len(tbg.OUTPUTS.grid_images_all)}")
            log(
                f"[TBG VLM Selection] selected_only={vlm_selected_only} "
                f"total_tiles={len(tbg.OUTPUTS.grid_images_all)} "
                f"selected_tiles={vlm_selected_tiles_human}",
                None,
                None,
                f"Node {tbg.INFO.id}",
            )
            if vlm_selected_only and not vlm_selected_tiles:
                log(
                    "[TBG VLM Selection] selected-only is enabled but no valid VLM tile indexes were parsed; "
                    "VLM prompt generation will skip all tiles.",
                    "BRIGHT_YELLOW",
                    "YELLOW",
                    f"Node {tbg.INFO.id}",
                )
        try:
            for index, tile_to_process in enumerate(tbg.OUTPUTS.grid_images_all):
                if vlm_selected_only and index not in vlm_selected_tiles:
                    prompt_tile = ""
                    grid_prompts.append(prompt_tile)
                    continue


                prompt_tile = prompt_context
                if tbg.LLM.model != "NONE":
                    print(f"TBG[Node {tbg.INFO.id}] VLM {tbg.LLM.model} working on Tile {index + 1} of {len(tbg.OUTPUTS.grid_images_all)}")
                    prompt_tile = get_prompt_tile(tile_to_process)
                    prompt_tile = strip_thinking(
                        prompt_tile,
                        f"[QwenVL] tile {index + 1}/{total}",
                    )
                    sentences_with_words_to_remove = [
                        "assistant", "helpful", "vision", "Thedescription", "TheUser", "The睿", "The description",
                        "The User",
                        "It can assist", "natural language", "It can understand"
                    ]
                    prompt_tile = cls.remove_sentences_with_words(prompt_tile, sentences_with_words_to_remove)

                    log(f"tile {index + 1}/{total} - [tile answer] {prompt_tile}", "BRIGHT_BLUE", "BRIGHT_CYAN",
                        f"Node {tbg.INFO.id} - Prompting {iteration}")
                grid_prompts.append(prompt_tile)

            tbg.PROMPTER.tiler_prompts = grid_prompts

            # if the input comes direct from tiler ignore and rebuild empty values from prompter
            tbg.PROMPTER.output_prompts = tbg.PROMPTER.tiler_prompts
            tiles_len = len(tbg.PROMPTER.tiler_prompts)
            tbg.PROMPTER.output_denoises = cls._normalize([], tiles_len)
            tbg.PROMPTER.output_seeds_js = cls._normalize([], tiles_len)
            tbg.PROMPTER.output_cnet_js = cls._normalize([], tiles_len)
            tbg.PROMPTER.output_model_js = cls._normalize([], tiles_len)
            tbg.PROMPTER.output_color_match_js = cls._normalize([], tiles_len)
        finally:
            if tbg.LLM.model != "NONE":
                if "sky" in tbg.LLM.model.lower() and qwen is not None:
                    UnloadOneModelNode.route(qwen)
                if tbg.LLM.model.startswith("GGUF\\") and QwenVLGGUF is not None:
                    print("[QwenVL-GGUF] unloading cached GGUF model after prompting")
                    QwenVLGGUF.clear()
                    try:
                        launch_mode = str(getattr(tbg.PARAMS, "VLM_Server_Launch_Mode", "managed_local") or "managed_local")
                        if launch_mode == "managed_local":
                            stopped = get_server_manager().stop(reason="prompt_end", timeout_s=0)
                            if stopped:
                                print("[OpenAI-VLM] managed server stopped after prompting")
                            tbg.PARAMS.VLM_Server_Active = False
                    except Exception as stop_err:
                        print(f"[OpenAI-VLM] post-prompt stop skipped: {stop_err}")
                if tbg.LLM.model.startswith("Qwen") and QwenVL is not None:
                    UnloadOneModelNode.route(QwenVL)
                if "florence" in tbg.LLM.model.lower() and florence_loader is not None:
                    UnloadOneModelNode.route(florence_loader)

    @classmethod
    def min_final_imagesize(cls, full_upscaled_image, tile_h, tile_w):
        """
        Ensure the input image is large enough to fully fit a tile of size (tile_h, tile_w).
        If the image is smaller than the tile in either dimension, it is upscaled
        using the larger scale factor to preserve aspect ratio.
        """
        img_h, img_w = full_upscaled_image.shape[1], full_upscaled_image.shape[2]  # H, W

        # Calculate scaling factors for H and W
        scale_h = tile_h / img_h if img_h < tile_h else 1.0
        scale_w = tile_w / img_w if img_w < tile_w else 1.0

        # Use the larger factor to ensure both dimensions fit
        scale_factor = max(scale_h, scale_w)

        if scale_factor > 1.0:
            new_h = int(img_h * scale_factor)
            new_w = int(img_w * scale_factor)
            full_upscaled_image = nodes.ImageScale().upscale(
                full_upscaled_image, "bilinear", new_w, new_h, False
            )[0]
            tbg.PARAMS.upscale_by = scale_factor
            print(f"TBG[Node {tbg.INFO.id}] Your upscaled images is smaller than your Tile-size we scale it so tiles fit")

        return full_upscaled_image

    @staticmethod
    def remove_sentences_with_words(text, words):

        # Remove { and }
        text = text.replace("{", "").replace("}", "")
        # Remove parentheses
        text = text.replace("(", "").replace(")", "")
        # Remove "Flux": (with optional surrounding spaces)
        text = re.sub(r'"\s*Flux\s*":\s*', "", text)
        # Remove all remaining double quotes
        text = text.replace('"', "")
        # Optional: clean up extra spaces
        text = re.sub(r'\s+', " ", text).strip()
    
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
        # Build a regex pattern to match any of the words, word-boundary delimited, case-insensitive
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    
        # Keep sentences that do NOT contain any of the words
        filtered_sentences = [s for s in sentences if not pattern.search(s)]
    
        return " ".join(filtered_sentences)

    @classmethod
    def start_progress(cls):
        # 100 "steps" = 0–100%
        cls._pbar = ProgressBar(100)
        cls._current = 0

    @classmethod
    def progressbar(cls, percent: float):
        """Called from TBG_APP with a percentage 0–100."""
        if cls._pbar is None:
            return

        # clamp to 0–100
        if percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100

        target = int(percent)

        # compute delta relative to our own counter, not pbar.value
        delta = target - cls._current
        if delta > 0:
            cls._pbar.update(delta)
            cls._current = target

    @classmethod
    def progressbar_finish(cls, percent: float | None = None):
        """Optional: ignore percent, just force to 100% and reset."""
        if cls._pbar is not None:
            # ensure we hit 100
            delta = 100 - cls._current
            if delta > 0:
                cls._pbar.update(delta)
            cls._pbar = None
            cls._current = 0

    @classmethod
    def _normalize(cls, arr, n):
        """Pad or truncate array to length n."""
        arr = list(arr or [])
        if len(arr) < n:
            arr.extend([""] * (n - len(arr)))
        elif len(arr) > n:
            arr = arr[:n]
        return arr

