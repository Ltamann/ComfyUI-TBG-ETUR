"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________
__________________________________________________a ToBi´s Gen production_____________________________________________________________________
"""
import time
import threading
import time
import comfy
import os
import json
import inspect
import folder_paths
#import numpy as np
import PIL
from PIL import Image
import copy
import nodes
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_mask import MaskToImage, ImageToMask
from .inc.TBG_split_aware_lanpaint_sampler  import TBG_DualModelSampler_COPY as TBG_DualModelSampler_lanpaint, TBG_KSamplerAdvancedSplitAware_Copy as TBG_KSamplerAdvancedSplitAware_lanpaint
from .inc.TBG_sampler_split_aware import TBG_DualModelSampler_COPY as TBG_DualModelSampler_normal, TBG_KSamplerAdvancedSplitAware_Copy as TBG_KSamplerAdvancedSplitAware_normal
from .inc import flux2_differential
from .inc.flux2_direct_sampler import sample_flux2_direct
from .inc.flux2_sampler_registry import TBG_FLUX2_SAMPLER_NAME
from ...vendor.ComfyUI_UltimateSDUpscale.utils import  crop_cond
from ...vendor.ComfyUI_Detail_Daemon.detail_daemon_node import DetailDaemonSamplerNode
from ....TBG.SERVERS.WORKER_server import WORKER,TBG_Controller

from ...vendor.comfyui_resharpen_main.tbgresharpen import disable_resharpen, TBG_DetailEnhancer
from ...utils.log import log
from .inc.image import TBG_Image
from .inc.batch import is_batch_pipe, run_batch_refiner
from .inc.sift_drift import (
    DRIFT_CORRECTION_MODES,
    apply_sift_drift_correction,
    normalize_drift_correction_mode,
    sift_tensor_to_uint8_rgb,
)
from .inc.tbg_pid import (
    PID_SCALE,
    PID_UPSCALE_SPECS,
    gpu_pid_tile_rebuild,
    load_pid_refiner_runtime,
    pid_gpu_final_rebuild_enabled,
    run_pid_refiner_latent_decode,
    select_pid_refiner_model,
    unload_pid_refiner_runtime,
)
from .inc.sigmas import get_sigmas
from .inc.sigmas import denoise_sigmas_tgb
from .inc.cnet import apply_reference_mode_hooks, get_Kontext_stiched_o_chained_cond, get_qwen_stiched_o_chained_cond, normalize_controlnet_mode
import comfy.model_management as mm
from types import SimpleNamespace

from .inc.vram_optimizing import VRAMOptimizer
import torch
from ....TBG.CALLBACKS.constants import get_tbg , TBGState
PIL.Image.MAX_IMAGE_PIXELS = 592515344
persistent_storage = {}
# V3 scheme name changes
if hasattr(MaskToImage, "execute"):
    # execute is a @classmethod; no instance needed
    def MaskToImage_execute(mask):
        return MaskToImage.execute(mask)
elif hasattr(MaskToImage, "composite"):
    # composite is an instance method; needs an instance
    def MaskToImage_execute(mask):
        node = MaskToImage()  # or cache a global instance if you prefer
        return node.composite(mask)




if hasattr(ImageToMask, "execute"):
    ImageToMask_execute = ImageToMask.execute
elif hasattr(ImageToMask, "image_to_mask"):
    ImageToMask_execute = ImageToMask.image_to_mask

from comfy_extras.nodes_mask import ImageCompositeMasked

if hasattr(ImageCompositeMasked, "execute"):
    # execute is a @classmethod; no instance needed
    def ImageCompositeMasked_execute(destination, source, x, y, resize_source, mask=None):
        return ImageCompositeMasked.execute(destination, source, x, y, resize_source, mask)
elif hasattr(ImageCompositeMasked, "composite"):
    # composite is an instance method; needs an instance
    def ImageCompositeMasked_execute(destination, source, x, y, resize_source, mask=None):
        node = ImageCompositeMasked()  # or cache a global instance if you prefer
        return node.composite(destination, source, x, y, resize_source, mask)

from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
if hasattr(TextEncodeQwenImageEdit, "execute"):
    TextEncodeQwenImageEdit_execute = TextEncodeQwenImageEdit.execute
elif hasattr(TextEncodeQwenImageEdit, "encode"):
    TextEncodeQwenImageEdit_execute = TextEncodeQwenImageEdit.encode

from comfy_extras.nodes_flux import CLIPTextEncodeFlux, FluxGuidance
CLIPTextEncodeFlux_instance = CLIPTextEncodeFlux()
if hasattr(CLIPTextEncodeFlux, "execute"):
    CLIPTextEncodeFlux_execute = CLIPTextEncodeFlux_instance.execute
elif hasattr(CLIPTextEncodeFlux, "encode"):
    CLIPTextEncodeFlux_execute = CLIPTextEncodeFlux_instance.encode

if hasattr(FluxGuidance, "execute"):
    FluxGuidance_execute = FluxGuidance.execute
elif hasattr(FluxGuidance, "append"):
    FluxGuidance_execute = FluxGuidance.append

from comfy_extras.nodes_custom_sampler import KSamplerSelect
ksampler_instance = KSamplerSelect()
if hasattr(KSamplerSelect, "execute"):
    KSamplerSelect_execute = ksampler_instance.execute
elif hasattr(KSamplerSelect, "get_sampler"):
    KSamplerSelect_execute = ksampler_instance.get_sampler

from comfy_extras.nodes_edit_model import ReferenceLatent
if hasattr(ReferenceLatent, "execute"):
    ReferenceLatent_execute = ReferenceLatent.execute
elif hasattr(ReferenceLatent, "append"):
    ReferenceLatent_execute = ReferenceLatent.append

def append_reference_latent(conditioning, latent):
    try:
        return ReferenceLatent_execute(conditioning, latent)[0]
    except TypeError:
        return ReferenceLatent_execute(0, conditioning, latent)[0]

from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

# Safely get the executor (class or instance-based)
def get_diffusion_executor():
    # Try class-level execute (new Comfy schema nodes)
    if hasattr(DifferentialDiffusion, "execute") and callable(DifferentialDiffusion.execute):
        return DifferentialDiffusion.execute

    # Try legacy instance-based methods
    diff_node = DifferentialDiffusion()
    for name in ("execute", "apply", "__call__"):
        fn = getattr(diff_node, name, None)
        if callable(fn):
            return fn

    # Fallback safeguard
    def _not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "DifferentialDiffusion node has no callable interface (execute/apply/__call__)."
        )

    return _not_implemented

DifferentialDiffusion_execute = get_diffusion_executor()
# Global worker shutdown state
DEBUG = True
from ....TBG.SERVERS.COMFYUI_server import register_main_class

@register_main_class
class TBG_Refiner_v1():
    NAME = "TBG Enhanced Refiner FLUX PRO 1"
    DRIFT_CORRECTION_MODES = DRIFT_CORRECTION_MODES
    COLOR_STABILIZER_METHOD = "TBG Detail-Preserving Color Stabilizer"
    TILE_STABILIZER_METHOD = "TBG ETUR Detail-Preserving Tile Stabilizer"
    COLOR_STABILIZER_ALIASES = {
        "tbg non-structural",
        "tbg detail-preserving color stabilizer",
    }
    TILE_STABILIZER_ALIASES = {
        "tbg non-structural tile-aware",
        "tbg etur detail-preserving tile stabilizer",
    }
    VRAM_OPTIMIZER = None
    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX2': 2048,
        'Ideogram4': 2048,
        'FLUX1 Kontext': 1024,
        'Qwen Image': 1328,
        'Qwen Image Edit': 1328,
        'SDXL': 1024,
        'SD3': 1024,
        'Others': 1024,
    }

    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())

    @classmethod
    def _parse_max_segment_size(cls, value):
        try:
            text = str(value).strip()
            parsed = int(float(text)) if text else 2048
        except Exception:
            parsed = 2048
        return max(256, min(4096, parsed))

    @classmethod
    def _sigma_trace_enabled(cls):
        return bool(getattr(tbg.API, "dev_debug_enabled", getattr(tbg.API, "status", None) == "Dev"))

    @classmethod
    def _sigma_trace_text(cls, sigmas):
        if sigmas is None:
            return "count=0 min=None max=None head=[] tail=[] full=[]"
        try:
            if torch.is_tensor(sigmas):
                sigma_tensor = sigmas.detach().float().cpu().view(-1)
                sigma_values = sigma_tensor.tolist()
            else:
                sigma_values = [float(v) for v in list(sigmas)]
        except Exception as exc:
            return f"unavailable error={exc}"

        count = len(sigma_values)
        if count == 0:
            return "count=0 min=None max=None head=[] tail=[] full=[]"

        rounded = [round(v, 6) for v in sigma_values]
        head = rounded[:6]
        tail = rounded[-6:] if count > 6 else rounded[:]
        full_values = rounded if count <= 64 else f"omitted(count={count})"
        return (
            f"count={count} min={min(rounded):.6f} max={max(rounded):.6f} "
            f"head={head} tail={tail} full={full_values}"
        )

    @classmethod
    def _log_sigma_trace(cls, stage, sigmas, **metadata):
        if not cls._sigma_trace_enabled():
            return
        meta_parts = [f"stage={stage}"]
        for key, value in metadata.items():
            meta_parts.append(f"{key}={value}")
        log(
            f"[TBG SigmaTrace] {' '.join(meta_parts)} {cls._sigma_trace_text(sigmas)}",
            None,
            None,
            f"Node {tbg.INFO.id}",
        )

    DENOISE_METHODS = [
        'default',
        'normalized',
        'normalized advanced',
        'multiplied',
        'multiplied normalized',
        'default short ',
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
        TILE_STABILIZER_METHOD,
        'mkl',
        'hm',
        'reinhard',
        'reinhard_lab_gpu',
        'mvgd',
        'hm-mvgd-hm',
        'hm-mkl-hm',
    ]

    PID_VAE_ALLOWED_MODEL_TYPES = {"FLUX1", "FLUX2", "FLUX1 Kontext", "Qwen Image", "Qwen Image Edit", "SDXL", "SD3", "Z-Image"}
    _rf_untwisting_missing_warned = False
    _rf_untwisting_error_warned = False

    @staticmethod
    def is_tbg_non_structural(method):
        key = str(method or "").strip().lower()
        return key in TBG_Refiner_v1.COLOR_STABILIZER_ALIASES or (
            key in TBG_Refiner_v1.TILE_STABILIZER_ALIASES and not TBG_Refiner_v1.is_tbg_tile_aware(method)
        )

    @staticmethod
    def is_tbg_tile_aware(method):
        if str(method or "").strip().lower() not in TBG_Refiner_v1.TILE_STABILIZER_ALIASES:
            return False
        mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
        if str(mode or "").strip().lower() != "neuro_generative_tile_fusion":
            return False
        status = getattr(tbg.API, "status", None)
        real_status = getattr(tbg.API, "real_status", status)
        member_status = str(real_status or status) in ("Pro", "Premium", "Unlimited", "Dev")
        return bool(member_status and getattr(tbg.API, "activate_pro", False))

    DIFFUSION_MODES = [
        'From TGB Tiler Node',
        'Tile_Fusion',
        'Neuro_Generative_Tile_Fusion',

    ]

    @classmethod
    def tbg_mark_worker_job_started(cls):
        """
        Call at the very start of any job (tiler or refiner).

        Delegates to TBG_Controller so worker lifetime is shared
        across all tilers/refiners.
        """
        TBG_Controller.mark_job_started()

    @classmethod
    def tbg_schedule_worker_shutdown(cls, delay):
        """
        Schedule a conditional shutdown for the shared worker process.

        The shutdown will only happen if no newer job has started since
        this schedule call.
        """
        TBG_Controller.schedule_worker_shutdown(delay)

    @classmethod
    def fn(cls, **kwargs):
        start_time = time.time()
        if is_batch_pipe(kwargs.get('TBG_Pipe', ())):
            from .TBG_Tiler import TBG_Upscaler_v1
            from .TBG_Pipes import TBG_TilePrompter_v1
            return run_batch_refiner(cls, TBG_Upscaler_v1, TBG_TilePrompter_v1, **kwargs)

        # Init TBG
        global tbg, tiler_id

        # --- mark new job started: cancel any pending shutdown ---

        tiler_id = kwargs.get('TBG_Pipe')[10]
        tbg = get_tbg(tiler_id)
        tbg.INFO.tiler_id = tiler_id  # why this is not set from the tiler ?
        tbg.INFO.id = kwargs.get('id', None)

        # SET WORKER TIMER for Shutdown
        if not hasattr(tbg, "WORKER_shutdown_timer"):
            tbg.WORKER_shutdown_timer = None  # threading.Timer or None
        if not hasattr(tbg, "WORKER_last_activity"):
            tbg.WORKER_last_activity = 0.0  # timestamp
        # new: mark that a job is running for this tbg
        cls.tbg_mark_worker_job_started()

        log("ETUR Refiner PRO is starting", None, None, f"Node {tbg.INFO.id}")
        cls.init(**kwargs)
        cls._sanitize_tile_override_model_registry()
        cls._sanitize_tile_override_cnetpipe_registry()

        # === NEW: Early exit if upscaler ran in "Only Upscale" mode ===
        tbg_pipe = kwargs.get('TBG_Pipe', ())
        if len(tbg_pipe) > 8 and tbg_pipe[8] == 'Only Upscale':
            log("Refiner skipping: Upscaler ran in 'Only Upscale' mode (no tiles generated)",
                None, None, f"Node {tbg.INFO.id}")
            return {
                "ui": {
                    "value": [
                        "⚠️  REFINE SKIPPED – NO TILES TO PROCESS",
                        "",
                        "❌ **TBG ETUR Labs Upscaler** is set to **'Only Upscale'** mode.",
                        "   This generates no tiles for the refiner.",
                        "",
                        "✅ **FIX:** Deactivate 'Only Upscale' in the TBG ETUR Labs Upscaler node.",
                        "   Connect it again and run to generate tiles first.",
                    ]
                },
                "result": (None,) * 6  # Match expected return shape with None
            }
        # === END NEW BLOCK ===




        output_image, output_image_only_tiles, output_image_noCC = cls.refine(tbg.OUTPUTS.upscaled_image, "Refiner")
        # Close WORKER on end of refinment


        end_time = time.time()
        elapsed = end_time - start_time
        log(
            f"ETUR Refiner PRO completed in {elapsed:.2f} seconds",
            None,
            None,
            f"Node {tbg.INFO.id}"
        )


        #Convert images of different scale to correct format for PIL --- if all are the same size output_tiles = torch.cat(output_tiles)
        pid_output_tiles = getattr(tbg.TEMP, "pid_grid_images_4x", None)
        if getattr(tbg.KSAMPLER, "pid_vae_decode", False) and pid_output_tiles:
            output_tiles = list(copy.deepcopy(pid_output_tiles))
        else:
            output_tiles = list(copy.deepcopy(tbg.OUTPUTS.grid_images_all))
        input_tiles = list(tbg.OUTPUTS.orig_grid_images_all or [])

        # Squeeze batch dim where tile exists
        for index, torchtile in enumerate(output_tiles):  # BHWC
            if torchtile is not None:
                output_tiles[index] = torchtile.squeeze(0)  # [1,H,W,3] → [H,W,3]

        # Fill missing or non-existent output tiles from input tiles
        for index, input_tile in enumerate(input_tiles):  # BHWC
            if index >= len(output_tiles) or output_tiles[index] is None:
                fallback_tile = copy.deepcopy(input_tile)
                if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
                    fallback_tile = nodes.ImageScale().upscale(
                        fallback_tile,
                        "lanczos",
                        int(fallback_tile.shape[2]) * PID_SCALE,
                        int(fallback_tile.shape[1]) * PID_SCALE,
                        False,
                    )[0]
                output_tiles[index] = fallback_tile.squeeze(0)

        output_tiles = tuple(output_tiles)

        return (
            output_image,
            output_image_only_tiles,
            output_image_noCC,
            tbg.OUTPUTS.upscaled_image, #upscaled
            tbg.INPUTS.image,
            output_tiles,
        )
    @classmethod
    def _validate_pid_vae_requirements(cls):
        model_type = str(getattr(tbg.KSAMPLER, "model_type", "") or "")

        problems = []
        if model_type not in cls.PID_VAE_ALLOWED_MODEL_TYPES:
            problems.append(
                f"selected model type is '{model_type or 'None'}'; supported: "
                "FLUX1, FLUX2, FLUX1 Kontext, Qwen Image, Qwen Image Edit, SDXL, SD3, Z-Image"
            )

        if problems:
            raise ValueError(
                "Nvidia PiD 4x VAE requires a compatible "
                "model type (FLUX1, FLUX2, FLUX1 Kontext, Qwen Image, Qwen Image Edit, SDXL, SD3, or Z-Image).\n"
                "Tiles and Segment-to-Tile crops may have arbitrary sizes; non-1024 inputs "
                "are handled by the tiled PiD latent decode path.\n"
                "Fix these settings, then run again.\n"
                "Current problem(s): " + "; ".join(problems)
            )

    @classmethod
    def _call_external_node_method(cls, node_obj, method_name, call_kwargs):
        method = getattr(node_obj, method_name)
        try:
            sig = inspect.signature(method)
            params = sig.parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return method(**call_kwargs)
            filtered_kwargs = {key: value for key, value in call_kwargs.items() if key in params}
            return method(**filtered_kwargs)
        except ValueError:
            return method(**call_kwargs)

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
    def _apply_rf_untwisting_rope_for_tile(cls, model, latent_image, positive, index, latent_source="unknown"):
        pipe = getattr(tbg.KSAMPLER, "RF_UntwistingRoPE", None)
        if not isinstance(pipe, dict) or not bool(pipe.get("enabled", False)):
            return model
        if bool(getattr(tbg.PARAMS, "RF_UntwistingRoPE_runtime_disabled", False)):
            return model
        if not isinstance(latent_image, dict) or latent_image.get("samples", None) is None:
            if not cls._rf_untwisting_error_warned:
                print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE disabled: tile latent is missing or invalid.")
                cls._rf_untwisting_error_warned = True
            tbg.PARAMS.RF_UntwistingRoPE_runtime_disabled = True
            return model
        if positive is None:
            if not cls._rf_untwisting_error_warned:
                print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE disabled: tile conditioning is missing.")
                cls._rf_untwisting_error_warned = True
            tbg.PARAMS.RF_UntwistingRoPE_runtime_disabled = True
            return model
        samples = latent_image.get("samples", None)
        latent_channels = int(samples.shape[1]) if torch.is_tensor(samples) and samples.ndim >= 2 else "unknown"
        if getattr(tbg.API, "status", "") == "Dev":
            print(
                f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE tile {index + 1}: "
                f"channels={latent_channels} "
                f"latent_source={latent_source} "
                f"vae_encode={getattr(tbg.KSAMPLER, 'vae_encode_type', None)} "
                f"model_type={getattr(tbg.KSAMPLER, 'model_type', None)}"
            )

        try:
            from ...vendor.tbg_untwisting_rope_runtime import RFInversion as rf_cls
            from ...vendor.tbg_untwisting_rope_runtime import UntwistingRoPE as untwisting_cls
        except Exception as exc:
            if not cls._rf_untwisting_missing_warned:
                print(
                    f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE disabled: ETUR vendored "
                    f"runtime could not be imported ({type(exc).__name__}: {exc})."
                )
                cls._rf_untwisting_missing_warned = True
            tbg.PARAMS.RF_UntwistingRoPE_runtime_disabled = True
            return model

        try:
            rf_node = rf_cls()
            rf_result = cls._call_external_node_method(
                rf_node,
                "build",
                {
                    "model": model,
                    "reference_latent": latent_image,
                    "ref_conditioning": positive,
                    "rf_mode": pipe.get("rf_mode", "rf_gamma_rk2"),
                    "gamma": float(pipe.get("gamma", 0.5)),
                    "pmi_alpha": float(pipe.get("pmi_alpha", 0.0)),
                    "otip_strength": float(pipe.get("otip_strength", 0.0)),
                    "otip_clip_norm": float(pipe.get("otip_clip_norm", 10.0)),
                    "verbose": bool(pipe.get("rf_verbose", pipe.get("verbose", False))),
                },
            )
            rf_model, rf_inversion = rf_result[0], rf_result[1]

            untwisting_kwargs = {
                "model": rf_model,
                "rf_inversion": rf_inversion,
                "beta": float(pipe.get("beta", 50.0)),
                "high_scale_start": float(pipe.get("high_scale_start", 1.0)),
                "high_scale_end": float(pipe.get("high_scale_end", 0.0)),
                "low_scale_start": float(pipe.get("low_scale_start", 1.0)),
                "low_scale_end": float(pipe.get("low_scale_end", 3.0)),
                "adain_strength": float(pipe.get("adain_strength", 0.5)),
                "blocks": pipe.get("blocks", "0-999"),
                "verbose": bool(pipe.get("untwisting_verbose", pipe.get("verbose", False))),
            }
            if pipe.get("unofficial_extensions", None) is not None:
                untwisting_kwargs["unofficial_extensions"] = pipe.get("unofficial_extensions")

            untwisting_node = untwisting_cls()
            patched_result = cls._call_external_node_method(
                untwisting_node,
                "patch",
                untwisting_kwargs,
            )
            if getattr(tbg.API, "status", "") == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE active for tile {index + 1}.")
            return patched_result[0]
        except Exception as exc:
            if not cls._rf_untwisting_error_warned:
                print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE disabled after error: {exc}")
                cls._rf_untwisting_error_warned = True
            tbg.PARAMS.RF_UntwistingRoPE_runtime_disabled = True
            return model

    @classmethod
    def init(cls, **kwargs):

        tbg.storage_key = f"TBG_Refiner_{tbg.INFO.id}"
        attribute_names = ('INPUTS', 'PARAMS', 'KSAMPLER', 'OUTPUTS', 'SEGMENTS', 'SIZE', 'API', 'PROMPTER')
        pipe = kwargs.get('TBG_Pipe', (None,) * len(attribute_names))
        for name, value in zip(attribute_names, pipe):
            setattr(tbg, name, value)

        current_credits = kwargs.get('TBG_Pipe')[8]
        TBG_PIPE_input_ID = kwargs.get('TBG_Pipe')[9]
        TBG_Tiler_input_ID = kwargs.get('TBG_Pipe')[10]
        #from promter "result": ((INPUTS, PARAMS, KSAMPLER, OUTPUTS, SEGMENTS, SIZE, API, PROMPTER, current_credits, id, tiler_id, API.info_url),)
        #from timer "result": (tbg.INPUTS, tbg.PARAMS, tbg.KSAMPLER, tbg.OUTPUTS, tbg.SEGMENTS, tbg.SIZE, tbg.API, tbg.PROMPTER,tbg.API.current_credits, node_id, tiler_id, tbg.API.info_url)

        labs_refiner_dict = kwargs.get('labs_refiner', None)
        #labs_refiner_dict = labs_refiner[0]
        if tbg.SEGMENTS.Segment_Mask is None and tbg.SEGMENTS.segms is None:
            # No segment inputs → clear everything
            tbg.SEGMENTS.segms = None,
            tbg.SEGMENTS.upscale_factor = None,
            tbg.SEGMENTS.pad_offset = None,
            tbg.SEGMENTS.segment_tiles = None,
            tbg.SEGMENTS.orig_segment_tiles = None,
            tbg.SEGMENTS.segms_scale = None,
            tbg.SEGMENTS.segms_cropped_masks = None,
            tbg.SEGMENTS.segment_binary_masks = None,
            tbg.SEGMENTS.segms_crop_regions = None,
            tbg.SEGMENTS.segms_new = None,
            tbg.SEGMENTS.Segment_Mask = None,
            tbg.SEGMENTS.inpainting_mask = None,
            tbg.SEGMENTS.compositing_mask = None,
            tbg.SEGMENTS.h = None,
            tbg.SEGMENTS.w = None,

        if labs_refiner_dict is not None:
            # Optional
            tbg.PARAMS.Differential_Diffusion =  labs_refiner_dict.get('Differential_Diffusion', True)
            tbg.PARAMS.Flux2_Tile_Color_Correction = labs_refiner_dict.get('Flux2_Tile_Color_Correction', True)
            tbg.PARAMS.Flux2_Sampler_Hook = labs_refiner_dict.get('Flux2_Sampler_Hook', False)
            tbg.PARAMS.Segment_Background_Harmonization = labs_refiner_dict.get('Segment_Background_Harmonization', True)
            tbg.PARAMS.ColorMatch_Debug_Switches = labs_refiner_dict.get('ColorMatch_Debug', None)
            tbg.KSAMPLER.custom_sigmas = labs_refiner_dict.get('Custom_Sigmas_!DENOISE=1', None)
            cls._log_sigma_trace(
                "labs_refiner_input",
                tbg.KSAMPLER.custom_sigmas,
                source="labs_refiner",
                model_type=kwargs.get('model_type', None),
            )
            tbg.PARAMS.Alternative_Image = labs_refiner_dict.get('Resume_Tiled_Refinement_Image', None)
            # Requiered
            tbg.PARAMS.Tile_Fusion_Blend = labs_refiner_dict.get('Tile_Fusion_Blend', 0.5)
            tbg.PARAMS.inpaint_conditioning = labs_refiner_dict.get('Fusion_conditioning', True)
            tbg.PARAMS.point_grid_image_stabilizer_experimental = labs_refiner_dict.get('Color & Structure Stabilizer', 0)
            tbg.PARAMS.memorize = labs_refiner_dict.get('PRO_Tile_Cache', 'OFF')

            tbg.PARAMS.LanPaint = labs_refiner_dict.get('LanPaint', True)
            inpaint_end = labs_refiner_dict.get('Fusion_end', 0)
            tbg.PARAMS.Preview_Tiles_in_Temp_Folder = labs_refiner_dict.get('Save_Tiles_in_Temp_Folder', False)
            if not bool(getattr(tbg.API, "dev_debug_enabled", getattr(tbg.API, "status", None) == "Dev")):
                tbg.PARAMS.Preview_Tiles_in_Temp_Folder = False
            tbg.KSAMPLER.sampler_input = labs_refiner_dict.get('Sampler', None)
            tbg.KSAMPLER.ideogram4_guider = labs_refiner_dict.get('Ideogram4_Guider', None)
            tbg.KSAMPLER.pid_model = labs_refiner_dict.get('PID_Model', None)
            tbg.KSAMPLER.cropped_positive = labs_refiner_dict.get('cropped_positive', None)
            tbg.KSAMPLER.cropped_negative = labs_refiner_dict.get('cropped_negative', None)
            tbg.PARAMS.stitch_blending = labs_refiner_dict.get('stitch_blending', "gpupyramid")
            tbg.PARAMS.max_upscale_size_segment_inpainting = cls._parse_max_segment_size(
                labs_refiner_dict.get('max_upscale_size_segment', 2048)
            )


        else:

            tbg.KSAMPLER.custom_sigmas = None
            tbg.KSAMPLER.pid_model = None
            tbg.PARAMS.Alternative_Image = None
            tbg.PARAMS.Differential_Diffusion = True
            tbg.PARAMS.Flux2_Tile_Color_Correction = True
            tbg.PARAMS.Flux2_Sampler_Hook = False
            tbg.PARAMS.Segment_Background_Harmonization = False
            tbg.PARAMS.ColorMatch_Debug_Switches = None
            tbg.PARAMS.inpaint_conditioning = True
            tbg.PARAMS.point_grid_image_stabilizer_experimental = 0
            tbg.PARAMS.memorize = 'OFF'
            tbg.SIZE.inpaint_max = 0.05

            tbg.PARAMS.LanPaint = True
            tbg.PARAMS.Preview_Tiles_in_Temp_Folder = False
            inpaint_end = 0
            tbg.KSAMPLER.sampler_input = None
            tbg.KSAMPLER.ideogram4_guider = None
            tbg.KSAMPLER.cropped_positive = None
            tbg.KSAMPLER.cropped_negative = None
            tbg.PARAMS.stitch_blending = "gpupyramid"
            tbg.PARAMS.max_upscale_size_segment_inpainting = 2048



        tbg.DUALMODEL.steps=kwargs.get('Dual_Model_steps_low', None)
        tbg.DUALMODEL.cfg=kwargs.get('Dual_Model_cfg_low', None)
        tbg.DUALMODEL.model=kwargs.get('Dual_Model_model_low', None)
        tbg.DUALMODEL.clip=kwargs.get('Dual_Model_clip_low', None)
        tbg.DUALMODEL.vae=kwargs.get('Dual_Model_vae_low', None)
        tbg.DUALMODEL.high_low_swap=kwargs.get('Dual_Model_high_low_swap', 0.5)
        tbg.DUALMODEL.low_refiner=kwargs.get('Dual_Model_low_refiner', 1)
        tbg.DUALMODEL.General_Prompt=kwargs.get('Dual_Model_General_Prompt_Positive_low', "")
        tbg.DUALMODEL.General_Prompt_Negative=kwargs.get('Dual_Model_General_Prompt_Negative_low'
                                               "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理")
        tbg.DUALMODEL.model_crossover_sigma_strength=kwargs.get('model_crossover_sigma_strength', "")
        tbg.DUALMODEL.inpaint_end=inpaint_end
        tbg.DUALMODEL.smoother_sharper=kwargs.get('Smoother - Sharper', 0)
        tbg.DUALMODEL.detail_enhancer=kwargs.get('Detail Enhancer', 0)


        vram_profile = kwargs.get("VRAM_Profile", "Low VRAM Cache (Unload Models)")
        tbg.KSAMPLER.vram_profile = vram_profile
        tbg.lowvram = vram_profile != "Fast Cache (Max Speed)"



        tbg.PARAMS.PRO_Fusion_Complexity_Mask_Strength = kwargs.get('Image Stabilizer', 0)
        tbg.PARAMS.PRO_Per_Pixel_Denoise_Mask_Strength = kwargs.get('Per_Pixel_Denoise_Mask_Strength', 0)
        tbg.PARAMS.denoise_mask = kwargs.get('denoise_mask', None)
        tbg.PARAMS.Redux_strength = kwargs.get('Redux_strength', 0)
        tbg.PARAMS.contrast = kwargs.get('contrast', None)
        tbg.PARAMS.highpass = kwargs.get('highpass', None)
        DENOISE_METHODS = [
            'default',
            'normalized advanced',
            'default short',
        ]


        tbg.PARAMS.denoise_method =  kwargs.get('denoise_method', 'default')
        tbg.PARAMS.Fast_1_Tile_Preview =  kwargs.get('Fast_1_Tile_Preview', False)
        tbg.PARAMS.Redux_Style_Model =  kwargs.get('Redux_Style_Model', None)
        tbg.PARAMS.Redux_Clip_Vision =  kwargs.get('Redux_Clip_Vision', None)
        tbg.PARAMS.Laplacian_Pyramid_Blending =  kwargs.get('Enhanced_Laplacian_Blending', True)
        color_match_method = kwargs.get('Color_Match', cls.TILE_STABILIZER_METHOD)
        tbg.PARAMS.color_match_method = color_match_method
        tbg.PARAMS.color_match_str = kwargs.get('Color_Match_Str', 1)
        tbg.PARAMS.rgb_luma_nonstructural = cls.is_tbg_non_structural(color_match_method)
        tbg.PARAMS.model_type = kwargs.get('model_type', None)
        tbg.PARAMS.tiles_to_process_active = kwargs.get('Selected_Tiles_Only', False)
        tbg.PARAMS.Selected_Tiles_By_Numbers =  kwargs.get('Selected_Tiles_By_Numbers', '')
        tbg.PARAMS.force_fresh_refiner_background = False
        drift_mode = normalize_drift_correction_mode(kwargs.get('Scale-Invariant Feature Transform', True))
        tbg.PARAMS.sift_drift_correction_mode = drift_mode
        tbg.PARAMS.sift_drift_correction = drift_mode != "Off"

        parse_selected_tiles = tbg.PARAMS.tiles_to_process_active or tbg.PARAMS.Fast_1_Tile_Preview
        tbg.PARAMS.tiles_to_process = WORKER.id(tiler_id).TBG_Image.set_tiles_to_process(parse_selected_tiles,
                                              len(tbg.OUTPUTS.grid_images_all),
                                              tbg.PARAMS.Selected_Tiles_By_Numbers, False, _tbg_send_images = False)
        if parse_selected_tiles and not tbg.PARAMS.tiles_to_process:
            tbg.PARAMS.tiles_to_process = [0]
        # tiles_to_process and Fast_1_Tile_Preview
        if tbg.PARAMS.Fast_1_Tile_Preview:
            tbg.PARAMS.tiles_to_process_active = True  # Get First from list


        tbg.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        tbg.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        tbg.KSAMPLER.steps = kwargs.get('steps', None)
        tbg.KSAMPLER.cfg = kwargs.get('cfg', None)
        tbg.KSAMPLER.denoise = kwargs.get('denoise', None)
        cnet_multiply = kwargs.get(
            "Controlnet_Pipe_strength",
            kwargs.get("ControlNet_Pipe_strength", kwargs.get("controlnet_pipe_strength", 1)),
        )
        try:
            tbg.KSAMPLER.cnet_multiply = float(cnet_multiply)
        except Exception:
            tbg.KSAMPLER.cnet_multiply = 1.0
        tbg.KSAMPLER.noise_seed = kwargs.get('seed', None)
        tbg.KSAMPLER.General_Prompt = kwargs.get('General_Prompt_Positive', "")
        tbg.KSAMPLER.General_Prompt_Negative = kwargs.get('General_Prompt_Negative', "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理")
        tbg.KSAMPLER.Flux_Guidance = kwargs.get('Flux_Guidance', None)
        tbg.KSAMPLER.Controlnet_Pipe = kwargs.get('Controlnet_Pipe', None)
        tbg.KSAMPLER.model_type = kwargs.get('model_type', None)
        tbg.KSAMPLER.model = kwargs.get('model', None)
        tbg.KSAMPLER.clip = kwargs.get('clip', None)
        tbg.KSAMPLER.vae = kwargs.get('vae', None)
        vae_encode_type = kwargs.get('vae_encode', "tiled slow")
        if isinstance(vae_encode_type, bool):
            vae_encode_type = "tiled slow" if vae_encode_type else "tbg Color-preserving fast"
        tbg.KSAMPLER.vae_encode_type = vae_encode_type
        tbg.KSAMPLER.pid_vae_decode = vae_encode_type == "Nvidia PiD 4x"
        tbg.PARAMS.vae_encode_type = vae_encode_type
        tbg.PARAMS.pid_vae_decode = tbg.KSAMPLER.pid_vae_decode
        tbg.KSAMPLER.tiled = vae_encode_type == "tiled slow"
        tbg.KSAMPLER.Enrichment_Pipe = kwargs.get('Enrichment_Pipe', None)
        RF_UntwistingRoPE = kwargs.get('RF_UntwistingRoPE', None)
        tbg.KSAMPLER.RF_UntwistingRoPE = (
            RF_UntwistingRoPE
            if isinstance(RF_UntwistingRoPE, dict) and bool(RF_UntwistingRoPE.get("enabled", False))
            else None
        )
        tbg.PARAMS.RF_UntwistingRoPE_runtime_disabled = False
        cls._rf_untwisting_missing_warned = False
        cls._rf_untwisting_error_warned = False
        if tbg.KSAMPLER.RF_UntwistingRoPE is not None and getattr(tbg.API, "status", "") == "Dev":
            print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE pipe enabled for per-tile sampling.")
        tbg.KSAMPLER.sampler = tbg.KSAMPLER.sampler_input if tbg.KSAMPLER.sampler_input is not None else KSamplerSelect_execute(tbg.KSAMPLER.sampler_name)[0]
        tbg.SIZE.tile_size_vae = kwargs.get('tile_size_vae', None)

        # TBG Enrichment_Pipe
        Enrichment_Pipe = kwargs.get('Enrichment_Pipe', None)
        if Enrichment_Pipe is not None:
            tbg.KSAMPLER.resharpen_start = Enrichment_Pipe.get("resharpen_start", 0)
            tbg.KSAMPLER.resharpen_end = Enrichment_Pipe.get("resharpen_end", 0)
            tbg.KSAMPLER.Resharpener_strength = Enrichment_Pipe.get("Resharpener_strength", False)
            tbg.KSAMPLER.Resharpener_active =  Enrichment_Pipe.get("Resharpener_active", )
            tbg.KSAMPLER.detail_daemon_active =Enrichment_Pipe.get("detail_daemon_active", False)
            tbg.KSAMPLER.detail_amount = Enrichment_Pipe.get("detail_amount", 0)
            tbg.KSAMPLER.detail_daemon_start = Enrichment_Pipe.get("detail_daemon_start", 0)
            tbg.KSAMPLER.detail_daemon_end = Enrichment_Pipe.get("detail_daemon_end", 0)
            tbg.KSAMPLER.detail_daemon_bias = Enrichment_Pipe.get("detail_daemon_bias", 0)
            tbg.KSAMPLER.detail_daemon_exponent = Enrichment_Pipe.get("detail_daemon_exponent", 0)
            tbg.KSAMPLER.detail_daemon_start_offset = Enrichment_Pipe.get("detail_daemon_start_offset", 0)
            tbg.KSAMPLER.detail_daemon_end_offset = Enrichment_Pipe.get("detail_daemon_end_offset", 0)
            tbg.KSAMPLER.daemon_fade = Enrichment_Pipe.get("detail_daemon_fade", 0)
            tbg.KSAMPLER.daemon_smooth = Enrichment_Pipe.get("detail_daemon_smooth", 0)
            tbg.KSAMPLER.detail_daemon_cfg_scale = Enrichment_Pipe.get("detail_daemon_cfg_scale", 0)
            tbg.PARAMS.inner_Upscale_type = Enrichment_Pipe.get("tile_upscale_plus", 'none')
            tbg.PARAMS.upscale_method_inpainting = Enrichment_Pipe.get("upscaler_method_inpainting", 'bilinear')
            tbg.PARAMS.upscale_model_inpainting = Enrichment_Pipe.get("upscale_model_inpainting", None)
            tbg.PARAMS.inner_Upscale_value = Enrichment_Pipe.get("upscale_tiles_by", 1)
            tbg.PARAMS.inner_Upscale_Segments =  Enrichment_Pipe.get("upscale_segments_by", 1)
            tbg.PARAMS.PRO_Fusion_Complexity_min_Denoise = Enrichment_Pipe.get("PRO_Fusion_Complexity_min_Denoise", 0.2)
            tbg.PARAMS.PRO_Fusion_Complexity_max_Denoise = Enrichment_Pipe.get("PRO_Fusion_Complexity_max_Denoise", 1)
            tbg.PARAMS.PRO_Fusion_Complexity_Mask_Blur = Enrichment_Pipe.get("PRO_Fusion_Complexity_Mask_Blur", 0)

        else:
            # Presets if no enrichment_pipe
            tbg.PARAMS.PRO_Fusion_Complexity_min_Denoise = 0.2
            tbg.PARAMS.PRO_Fusion_Complexity_max_Denoise = 1
            tbg.PARAMS.PRO_Fusion_Complexity_Mask_Blur = 0
            tbg.KSAMPLER.Resharpener_strength = 0
            tbg.KSAMPLER.Resharpener_active = False
            tbg.KSAMPLER.resharpen_end = 0
            tbg.KSAMPLER.resharpen_start = 0
            tbg.KSAMPLER.detail_daemon_active = False
            tbg.PARAMS.inner_Upscale_type = 'none'
            tbg.PARAMS.inner_Upscale_value = 1
            tbg.PARAMS.inner_Upscale_Segments = 1
            tbg.PARAMS.upscale_method_inpainting = 'bilinear'
            tbg.PARAMS.upscale_model_inpainting =None

        if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
            cls._validate_pid_vae_requirements()
            if (
                tbg.PARAMS.inner_Upscale_type in ("finer details + grain removal", "finer details")
                or tbg.PARAMS.inner_Upscale_value not in (0, 1)
                or tbg.PARAMS.inner_Upscale_Segments not in (0, 1)
            ):
                print(
                    f"TBG[Node {tbg.INFO.id}] PID VAE 4x: disabled inner tile upscale "
                    "so Flux sampling stays at the user tile size."
                )
            tbg.PARAMS.inner_Upscale_type = "none"
            tbg.PARAMS.inner_Upscale_value = 1
            tbg.PARAMS.inner_Upscale_Segments = 1


        # TBG  Custom Sigmas
        if tbg.KSAMPLER.custom_sigmas is not None:
            tbg.KSAMPLER.steps = len(tbg.KSAMPLER.custom_sigmas) #kwargs.get('steps', None)
            tbg.KSAMPLER.sigmas = tbg.KSAMPLER.custom_sigmas

        # this read from json only make sense if tile overrides node is used Input_from_PROMTER = true
        Input_from_PROMTER = (TBG_PIPE_input_ID != TBG_Tiler_input_ID)
        # tbg.PROMPTER.cache_key is the sam as TBG_PIPE_input_ID
        if tbg.PROMPTER.cache_key is not None and Input_from_PROMTER:
            try:
                from .inc.tp_cache import Tile_Prompter_Cache
                import json
                # Load existing JSON (keep non-prompt fields - use key from used tile prompter node)
                existing = Tile_Prompter_Cache.get(tbg.PROMPTER.cache_key, None)

                if isinstance(existing, str) and existing.strip():
                    try:
                        obj = json.loads(existing)
                    except Exception:
                        obj = {}
                else:
                    obj = {}
                # if the user build fist time the tile Overrides or its modified parameters could not be updated so we need to take values fro PIPE not from json , this json was though for on node load with infos...

                if len(tbg.OUTPUTS.grid_images_all) == len(obj.get("prompts")):
                    print(f"Tile Overrides set from JSON {len(tbg.OUTPUTS.grid_images_all)} Tiles and {len(obj.get("prompts"))} Prompts ")
                    tbg.PROMPTER.output_denoises = obj.get("denoises") or []
                    tbg.PROMPTER.output_seeds_js = obj.get("seeds") or []
                    tbg.PROMPTER.output_cnet_js = obj.get("cnet_strength") or []
                    tbg.PROMPTER.output_cfg_js = obj.get("cfg_overrides") or []
                    tbg.PROMPTER.output_model_js = obj.get("model_overrides") or []
                    tbg.PROMPTER.output_cnetpipe_js = obj.get("cnetpipe_overrides") or []
                    tbg.PROMPTER.output_color_match_js = obj.get("color_match_overrides") or []
                    tbg.PROMPTER.output_prompts = obj.get("prompts") or []
                else:
                    print(f"Skipped json inputs from Tile Overrides Node because {len(tbg.OUTPUTS.grid_images_all)} Tiles have not the same count than {len(obj.get("prompts"))} Prompts, using PIPE ")
            except Exception:
                print(f"Skipped json inputs from Tile Overrides Node")

        # convert string to float and if empty add general
        tbg.PROMPTER.output_denoises = [
            float(x) if x != '' else tbg.KSAMPLER.denoise
            for x in tbg.PROMPTER.output_denoises
        ]
        tbg.PROMPTER.output_seeds_js = [
            int(x) if x != '' else tbg.KSAMPLER.noise_seed
            for x in tbg.PROMPTER.output_seeds_js
        ]
        tbg.PROMPTER.output_cnet_js = [
            float(x) if x != '' else tbg.KSAMPLER.cnet_multiply
            for x in tbg.PROMPTER.output_cnet_js
        ]
        tile_count_for_models = len(getattr(tbg.OUTPUTS, "grid_images_all", []) or [])
        cfg_choices = list(getattr(tbg.PROMPTER, "output_cfg_js", []) or [])
        if len(cfg_choices) < tile_count_for_models:
            cfg_choices.extend([""] * (tile_count_for_models - len(cfg_choices)))
        elif len(cfg_choices) > tile_count_for_models:
            cfg_choices = cfg_choices[:tile_count_for_models]
        cfg_values = []
        for x in cfg_choices:
            if x in ("", None):
                cfg_values.append(tbg.KSAMPLER.cfg)
                continue
            try:
                cfg_values.append(float(x))
            except Exception:
                cfg_values.append(tbg.KSAMPLER.cfg)
        tbg.PROMPTER.output_cfg_js = cfg_values
        model_choices = list(getattr(tbg.PROMPTER, "output_model_js", []) or [])
        if len(model_choices) < tile_count_for_models:
            model_choices.extend([""] * (tile_count_for_models - len(model_choices)))
        elif len(model_choices) > tile_count_for_models:
            model_choices = model_choices[:tile_count_for_models]
        tbg.PROMPTER.output_model_js = model_choices
        cnetpipe_choices = list(getattr(tbg.PROMPTER, "output_cnetpipe_js", []) or [])
        if len(cnetpipe_choices) < tile_count_for_models:
            cnetpipe_choices.extend([""] * (tile_count_for_models - len(cnetpipe_choices)))
        elif len(cnetpipe_choices) > tile_count_for_models:
            cnetpipe_choices = cnetpipe_choices[:tile_count_for_models]
        tbg.PROMPTER.output_cnetpipe_js = cnetpipe_choices
        color_match_choices = list(getattr(tbg.PROMPTER, "output_color_match_js", []) or [])
        if len(color_match_choices) < tile_count_for_models:
            color_match_choices.extend([""] * (tile_count_for_models - len(color_match_choices)))
        elif len(color_match_choices) > tile_count_for_models:
            color_match_choices = color_match_choices[:tile_count_for_models]
        normalized_color_match_choices = []
        for value in color_match_choices:
            raw = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
            if raw in ("color match off", "color mach off", "cm off", "off"):
                normalized_color_match_choices.append("color_match_off")
            elif raw in ("protect generated", "protect new generated content", "protect generated content"):
                normalized_color_match_choices.append("protect_new_generated_content")
            elif raw in ("color match from origin", "match from origin", "from origin", "origin", "full match"):
                normalized_color_match_choices.append("color_match_from_origin")
            else:
                normalized_color_match_choices.append("")
        tbg.PROMPTER.output_color_match_js = normalized_color_match_choices


        # reset tile inner upscale
        if tbg.PARAMS.inner_Upscale_type not in ("finer details + grain removal", "finer details"):
           tbg.PARAMS.inner_Upscale_value = 1
           tbg.PARAMS.inner_Upscale_Segments = 1


        # Initialize node-specific storage (always needed for caching)
        if tbg.storage_key not in persistent_storage:
            persistent_storage[tbg.storage_key] = {
                'generated_tiles': {},
                'tile_fingerprints': {},
                'segments_cache': {},
                'settings_hash': None,
                'prompter_data': {}
            }


        tbg.OUTPUTS.denoise_mask_tiles = None
        if tbg.PARAMS.denoise_mask is not None and len(tbg.PARAMS.denoise_mask) > 0:

            # mask: [B, H, W] -> [B, H, W, 1]
            mask_img = tbg.PARAMS.denoise_mask.unsqueeze(-1)
            # upscale as image (BHWC)
            denoise_mask = nodes.ImageScale.upscale(
                0,
                mask_img,
                "area",
                tbg.OUTPUTS.upscaled_image.shape[2],  # target W
                tbg.OUTPUTS.upscaled_image.shape[1],  # target H
                "disabled"
            )[0]

            # tile the mask
            tbg.OUTPUTS.denoise_mask_tiles = TBG_Image().gridspecs_get_grid_images(
                denoise_mask,
                tbg.PARAMS.grid_specs
            )

            # --- upscale denoise tiles

            if (
                    tbg.PARAMS.inner_Upscale_type in ("finer details + grain removal","finer details")
                    and tbg.PARAMS.inner_Upscale_value not in (0, 1)
            ):

                for i, dmt in enumerate(tbg.OUTPUTS.denoise_mask_tiles):

                    dmt = dmt.expand(-1, -1, -1, 3).contiguous()  # shape [1, 1024, 1024, 3]
                    upscaled_tile = TBG_Image().helper_upscaleimage(
                        dmt,
                        tbg.PARAMS.upscale_method_inpainting,
                        tbg.PARAMS.upscale_model_inpainting,
                        tbg.PARAMS.inner_Upscale_value
                    )
                    upscaled_tile = upscaled_tile[..., :1]  # keep only first channel [B, H, W, 3] -> [B, H, W, 1]
                    # overwrite the tile at the current index

                    #upscaled_tile =upscaled_tile.squeeze(0) # [B, H, W, 1] -> [H, W, 1]
                    tbg.OUTPUTS.denoise_mask_tiles[i] = upscaled_tile

            # back to mask: [1, H, W, 1] -> [1, H, W]
            tbg.OUTPUTS.denoise_mask_tiles = [m.squeeze(-1) for m in tbg.OUTPUTS.denoise_mask_tiles]
            #tbg.OUTPUTS.denoise_mask_tiles = [m.squeeze(0) for m in tbg.OUTPUTS.denoise_mask_tiles]


        # SAFE CACHE INVALIDATION - only cleanup if cache exists from previous run
        if cls.VRAM_OPTIMIZER is not None:
            # Check if we have a populated cache (not first run)
            cache_exists = (
                    hasattr(cls.VRAM_OPTIMIZER, 'text_embeddings_cache') and
                    cls.VRAM_OPTIMIZER.text_embeddings_cache and  # Has entries
                    len(cls.VRAM_OPTIMIZER.text_embeddings_cache) > 0
            )

            if cache_exists:
                current_ts = getattr(cls.VRAM_OPTIMIZER, 'last_timestamp', None)
                if current_ts != tbg.PARAMS.timestamp:
                    log(f"TBG[Node {tbg.INFO.id}] New tiles detected, clearing {len(cls.VRAM_OPTIMIZER.text_embeddings_cache)} cached entries",
                        None, None, f"Node {tbg.INFO.id}")
                    cls.VRAM_OPTIMIZER.cleanup()
                    cls.VRAM_OPTIMIZER.last_timestamp = tbg.PARAMS.timestamp
            else:
                # First run - cache is empty, don't cleanup, just set timestamp
                log(f"TBG[Node {tbg.INFO.id}] First run detected, preserving newly built cache", None, None,
                    f"Node {tbg.INFO.id}")
                cls.VRAM_OPTIMIZER.last_timestamp = tbg.PARAMS.timestamp
    # WORKER and LOCAL !
    @classmethod
    def prepare_tiles_to_process(cls):
        storage = persistent_storage[tbg.storage_key]
        # could do this in TBG APP so i send less tiles
        tbg.OUTPUTS.grid_images_all = copy.deepcopy(tbg.OUTPUTS.orig_grid_images_all)

        # Fast_1_Tile_Preview - get first of tiles to process or first tile to Preview
        if tbg.PARAMS.Fast_1_Tile_Preview:
            if not tbg.PARAMS.tiles_to_process or len(tbg.PARAMS.tiles_to_process) == 0:
                tbg.PARAMS.tiles_to_process = [0]
            else:
                tbg.PARAMS.tiles_to_process = [tbg.PARAMS.tiles_to_process[0]]

        if not tbg.PARAMS.Fast_1_Tile_Preview:
            existing_tiles = storage.get("generated_tiles", None)
            has_existing_tiles = False
            try:
                has_existing_tiles = existing_tiles is not None and len(existing_tiles) > 0
            except Exception:
                has_existing_tiles = False
            if not has_existing_tiles:
                storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.grid_images_all)
            elif (
                bool(getattr(tbg.PARAMS, "tiles_to_process_active", False))
                and getattr(tbg.API, "status", None) == "Dev"
            ):
                try:
                    cache_count = len(existing_tiles)
                except Exception:
                    cache_count = 0
                print(
                    f"[TBG SelectedTiles] preserving generated tile cache for selected-only run "
                    f"cache_entries={cache_count}"
                )

    @classmethod
    def precompute_all_embeddings_free_VRAM(cls):
        # save guard becouse if not if tiles changes it fails ..... need better way to detect changes
        if cls.VRAM_OPTIMIZER is not None:
            cls.VRAM_OPTIMIZER.cleanup()


        cls.VRAM_OPTIMIZER = VRAMOptimizer(tbg)
        vram_before = torch.cuda.memory_allocated() / 1024 ** 3

        if tbg.lowvram and cls.VRAM_OPTIMIZER is not None:
            mm.unload_all_models()
            mm.soft_empty_cache()

            log(f"Low Vram activated: Clip, ClipVision, StyleModel, ControlNet, Preprocessors FULLY UNLOADED", None, None, f"Node {tbg.INFO.id}")

            vram_after = torch.cuda.memory_allocated() / 1024 ** 3
            vram_saved = vram_before - vram_after

            print(f"TBG[Node {tbg.INFO.id}] Low Vram UNLOADED: {vram_saved:.2f} GB. Allocated {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    @classmethod
    def _apply_fast_preview_prompt_target(cls, changed_indices=None, change_type=None):
        """Resolve Fast 1 Tile Preview after Tile Overrides/change detection is available."""
        if not getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False):
            return

        tiles = getattr(tbg.OUTPUTS, "grid_images_all", None) or []
        total_tiles = len(tiles)
        if total_tiles <= 0:
            tbg.PARAMS.tiles_to_process = []
            tbg.PARAMS.tiles_to_process_active = True
            return

        changed = []
        try:
            changed = sorted(
                int(i) for i in (changed_indices or [])
                if 0 <= int(i) < total_tiles
            )
        except Exception:
            changed = []

        prompts = list(getattr(tbg.PROMPTER, "output_prompts", []) or [])
        prompt_indices = [
            i for i, prompt in enumerate(prompts[:total_tiles])
            if str(prompt or "").strip()
        ]
        override_prompt_indices = []
        try:
            from .inc.tp_cache import Tile_Prompter_Cache
            base_key = getattr(getattr(tbg, "CACHE", None), "prompt", None) or f"input_prompts_{tbg.INFO.id}"
            base_prompts = list(Tile_Prompter_Cache.get(base_key, []) or [])
            for i, prompt in enumerate(prompts[:total_tiles]):
                base = base_prompts[i] if i < len(base_prompts) else ""
                if str(prompt or "").strip() and str(prompt or "") != str(base or ""):
                    override_prompt_indices.append(i)
        except Exception:
            override_prompt_indices = []

        prompt_source = override_prompt_indices or prompt_indices
        prompt_candidates = [i for i in prompt_source if not changed or i in changed]
        segment_prompt_candidates = [i for i in prompt_candidates if cls._is_segment_index(i)]

        previous = list(getattr(tbg.PARAMS, "tiles_to_process", []) or [])
        target = None
        reason = "default first tile"
        explicit_selection = bool(str(getattr(tbg.PARAMS, "Selected_Tiles_By_Numbers", "") or "").strip())

        if explicit_selection and previous:
            try:
                target = int(previous[0])
                reason = "explicit selected tile"
            except Exception:
                target = None
        elif segment_prompt_candidates:
            target = segment_prompt_candidates[0]
            reason = "changed segment Tile Overrides prompt" if changed else "segment Tile Overrides prompt"
        elif prompt_candidates:
            target = prompt_candidates[0]
            reason = "changed Tile Overrides prompt" if changed else "Tile Overrides prompt"
        elif changed:
            target = changed[0]
            reason = f"{change_type or 'changed'} tile"
        elif previous:
            try:
                target = int(previous[0])
                reason = "explicit selected tile"
            except Exception:
                target = None

        if target is None:
            target = 0

        target = max(0, min(total_tiles - 1, int(target)))
        tbg.PARAMS.tiles_to_process = [target]
        tbg.PARAMS.tiles_to_process_active = True

        prompt_preview = ""
        if target < len(prompts):
            prompt_preview = str(prompts[target] or "").strip()
            if len(prompt_preview) > 120:
                prompt_preview = prompt_preview[:117] + "..."

        log(
            f"TBG[Node {tbg.INFO.id}] Fast 1 Tile Preview target tile {target + 1}/{total_tiles}: "
            f"{reason}; previous={previous}; prompt='{prompt_preview}'",
            None, None, f"Node {tbg.INFO.id}"
        )

    @classmethod
    def _pid_vram_profile(cls):
        return getattr(tbg.KSAMPLER, "vram_profile", "Low VRAM Cache (Unload Models)")

    @classmethod
    def _pid_profile_is_fast(cls):
        return cls._pid_vram_profile() == "Fast Cache (Max Speed)"

    @classmethod
    def _pid_profile_is_ultra(cls):
        return cls._pid_vram_profile() == "Ultra Low Memory (Per-Tile Streaming)"

    @classmethod
    def _flux2_pid_active(cls):
        return (
            getattr(tbg.KSAMPLER, "model_type", None) == "FLUX2"
            and getattr(tbg.KSAMPLER, "pid_vae_decode", False)
            and getattr(tbg.KSAMPLER, "vae_encode_type", None) == "Nvidia PiD 4x"
        )

    @classmethod
    def _pid_vae_decode_active(cls):
        return (
            getattr(tbg.KSAMPLER, "pid_vae_decode", False)
            and getattr(tbg.KSAMPLER, "vae_encode_type", None) == "Nvidia PiD 4x"
        )

    @classmethod
    def _flux2_pid_use_flux1_baseline(cls):
        """Keep Flux2 PiD on the proven Flux1 segment path until Flux2-specific tuning is tested."""
        return bool(getattr(tbg.PARAMS, "Flux2_PiD_Use_Flux1_Baseline", True))

    @classmethod
    def _pid_color_match_active(cls):
        method = getattr(tbg.PARAMS, "color_match_method", None)
        try:
            strength = float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)
        except Exception:
            strength = 1.0
        model_type = str(getattr(tbg.KSAMPLER, "model_type", None) or "").upper()
        if model_type == "FLUX2" and cls._flux2_pid_use_flux1_baseline():
            return False
        return (
            cls._pid_vae_decode_active()
            and method is not None
            and str(method).lower() != "none"
            and strength > 0.0
        )

    @classmethod
    def _flux2_pid_normal_vae_color_match_active(cls):
        normal_active = (
            cls._flux2_pid_active()
            and bool(getattr(tbg.PARAMS, "Flux2_PiD_Normal_VAE_Color_Match", False))
        )
        return cls._cm_debug_stage_enabled(
            "03_Flux2_PID_NormalVAE_Reference",
            normal_active,
            requires_method=False,
        )

    @classmethod
    def _pid_color_method(cls):
        method = getattr(tbg.PARAMS, "color_match_method", None)
        if method is None or str(method).lower() == "none":
            return "hm-mvgd-hm"
        return method

    @staticmethod
    def _rgb_mean_shift_255(reference, target):
        if reference is None or target is None:
            return None
        if not torch.is_tensor(reference) or not torch.is_tensor(target):
            return None
        if reference.ndim != 4 or target.ndim != 4:
            return None
        height = min(int(reference.shape[1]), int(target.shape[1]))
        width = min(int(reference.shape[2]), int(target.shape[2]))
        if height <= 0 or width <= 0:
            return None
        ref = reference[:, :height, :width, :3].to(device=target.device, dtype=torch.float32)
        tgt = target[:, :height, :width, :3].to(dtype=torch.float32)
        shift = (tgt - ref).mean(dim=(0, 1, 2)) * 255.0
        return tuple(float(v) for v in shift.detach().cpu())

    @staticmethod
    def _format_rgb_shift(shift):
        if shift is None:
            return "n/a"
        return "(" + ", ".join(f"{v:+.2f}" for v in shift) + ")"

    @staticmethod
    def _pid_high_frequency_abs(image):
        if image is None or not torch.is_tensor(image):
            return None
        img = image.unsqueeze(0) if image.ndim == 3 else image
        if img.ndim != 4:
            return None
        bchw = img.to(torch.float32).permute(0, 3, 1, 2).contiguous()
        low = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(bchw, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        return float(torch.mean(torch.abs(bchw - low)).detach().cpu())

    @classmethod
    def _debug_pid_detail_retention(cls, label, before, after, mask=None):
        if before is None or after is None or not torch.is_tensor(before) or not torch.is_tensor(after):
            return
        b = before.unsqueeze(0) if before.ndim == 3 else before
        a = after.unsqueeze(0) if after.ndim == 3 else after
        if b.ndim != 4 or a.ndim != 4:
            return
        if b.shape[1:3] != a.shape[1:3]:
            b = nodes.ImageScale().upscale(b, "lanczos", int(a.shape[2]), int(a.shape[1]), False)[0]
        b = b.to(device=a.device, dtype=torch.float32).clamp(0.0, 1.0)
        a = a.to(torch.float32).clamp(0.0, 1.0)
        before_hf = cls._pid_high_frequency_abs(b)
        after_hf = cls._pid_high_frequency_abs(a)
        ratio = (after_hf / max(before_hf, 1e-8)) if before_hf is not None and after_hf is not None else 0.0
        diff = torch.abs(a - b).mean(dim=-1, keepdim=True)
        center_delta = float(diff.mean().detach().cpu())
        seam_delta = 0.0
        if mask is not None and torch.is_tensor(mask):
            m = cls._mask_to_bhw(mask)
            if m is not None:
                if m.shape[-2:] != diff.shape[1:3]:
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(1).to(device=a.device, dtype=torch.float32),
                        size=diff.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                m = m.to(device=a.device, dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
                seam_delta = float((diff * m).sum().detach().cpu() / m.sum().clamp_min(1e-6).detach().cpu())
                center_delta = float((diff * (1.0 - m)).sum().detach().cpu() / (1.0 - m).sum().clamp_min(1e-6).detach().cpu())
        rgb_delta = (a - b).mean(dim=(0, 1, 2)).detach().cpu().tolist()
        print(
            f"TBG[Node {tbg.INFO.id}] PiD detail retention {label}: "
            f"hf_before={before_hf:.8f} hf_after={after_hf:.8f} hf_ratio={ratio:.4f} "
            f"mean_abs_delta={float(diff.mean().detach().cpu()):.8f} "
            f"seam_delta={seam_delta:.8f} center_delta={center_delta:.8f} "
            f"rgb_delta=({rgb_delta[0]:+.6f},{rgb_delta[1]:+.6f},{rgb_delta[2]:+.6f})"
        )

    @classmethod
    def _flux2_pid_color_match(cls, reference, target, label, method=None, strength=1.0):
        if reference is None or target is None or not torch.is_tensor(reference) or not torch.is_tensor(target):
            return target
        ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
        tgt = target.unsqueeze(0) if target.ndim == 3 else target
        if ref.ndim != 4 or tgt.ndim != 4:
            return target
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        before_shift = cls._rgb_mean_shift_255(ref, tgt)
        corrected, metrics = cls._global_rgb_luma_match(
            ref,
            tgt,
            strength=float(strength),
            label=f"pid_color_match_{label}_rgb_luma",
        )
        cls._log_global_rgb_luma_metrics(f"pid_color_match_{label}", metrics)
        corrected = corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        after_shift = cls._rgb_mean_shift_255(ref, corrected)
        print(
            f"TBG[Node {tbg.INFO.id}] PiD color match {label}: "
            f"method=global_rgb_luma mean_shift_before={cls._format_rgb_shift(before_shift)} "
            f"mean_shift_after={cls._format_rgb_shift(after_shift)}"
        )
        return corrected[0] if target.ndim == 3 and corrected.ndim == 4 else corrected

    @classmethod
    def _global_rgb_luma_stats(cls, image, mask=None):
        if image is None or not torch.is_tensor(image):
            return None
        img = cls._ensure_bhwc_image(image)
        if img is None or img.ndim != 4:
            return None
        img = torch.nan_to_num(img.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        weight = None
        if mask is not None and torch.is_tensor(mask):
            weight = cls._mask_to_bhw(mask)
            if weight is not None:
                weight = cls._scale_pid_mask(weight.to(torch.float32).clamp(0.0, 1.0), int(img.shape[2]), int(img.shape[1]))
                weight = cls._mask_to_bhw(weight)
                if weight is not None:
                    weight = weight.to(device=img.device, dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
        if weight is None or float(weight.sum().detach().cpu()) <= 1e-6:
            weight = torch.ones_like(img[..., :1], dtype=torch.float32, device=img.device)
        denom = weight.sum(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
        rgb_mean = (img * weight).sum(dim=(0, 1, 2), keepdim=True) / denom
        luma = img[..., 0:1] * 0.2126 + img[..., 1:2] * 0.7152 + img[..., 2:3] * 0.0722
        luma_mean = (luma * weight).sum() / weight.sum().clamp_min(1e-6)
        luma_var = (((luma - luma_mean) ** 2) * weight).sum() / weight.sum().clamp_min(1e-6)
        return {
            "rgb_mean": rgb_mean,
            "luma_mean": luma_mean,
            "luma_std": torch.sqrt(luma_var.clamp_min(0.0)),
            "weight": weight,
        }

    @classmethod
    def _global_rgb_luma_match(cls, reference, target, strength=1.0, apply_mask=None, label="pid_global_rgb_luma"):
        if reference is None or target is None or not torch.is_tensor(reference) or not torch.is_tensor(target):
            return target, None
        ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
        tgt = target.unsqueeze(0) if target.ndim == 3 else target
        if ref.ndim != 4 or tgt.ndim != 4:
            return target, None
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        try:
            strength = max(0.0, min(1.0, float(strength)))
        except Exception:
            strength = 1.0
        if strength <= 0.0:
            return target, None

        tgt_work = torch.nan_to_num(tgt.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        ref_work = torch.nan_to_num(ref.to(device=tgt_work.device, dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        ref_stats = cls._global_rgb_luma_stats(ref_work, apply_mask)
        tgt_stats = cls._global_rgb_luma_stats(tgt_work, apply_mask)
        if ref_stats is None or tgt_stats is None:
            return target, None

        max_rgb_shift = 36.0 / 255.0
        rgb_shift = (ref_stats["rgb_mean"] - tgt_stats["rgb_mean"]).clamp(-max_rgb_shift, max_rgb_shift) * strength
        corrected = (tgt_work + rgb_shift).clamp(0.0, 1.0)

        corrected_stats = cls._global_rgb_luma_stats(corrected, apply_mask)
        if corrected_stats is not None:
            target_std = corrected_stats["luma_std"].clamp_min(1e-6)
            gain_raw = ref_stats["luma_std"] / target_std
            gain = (1.0 + (gain_raw.clamp(0.92, 1.08) - 1.0) * min(0.5, strength)).to(device=corrected.device, dtype=corrected.dtype)
            mean_rgb = corrected_stats["rgb_mean"].to(device=corrected.device, dtype=corrected.dtype)
            corrected = (mean_rgb + (corrected - mean_rgb) * gain).clamp(0.0, 1.0)
        try:
            delta_bchw = (ref_work - corrected).permute(0, 3, 1, 2).contiguous().clamp(-48.0 / 255.0, 48.0 / 255.0)
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

        mask = cls._mask_to_bhw(apply_mask) if apply_mask is not None and torch.is_tensor(apply_mask) else None
        if mask is not None:
            mask = cls._scale_pid_mask(mask.to(torch.float32).clamp(0.0, 1.0), int(tgt_work.shape[2]), int(tgt_work.shape[1]))
            mask = cls._mask_to_bhw(mask)
            if mask is not None:
                mask = mask.to(device=corrected.device, dtype=corrected.dtype).clamp(0.0, 1.0).unsqueeze(-1)
                corrected = (corrected * mask + tgt_work * (1.0 - mask)).clamp(0.0, 1.0)

        final_stats = cls._global_rgb_luma_stats(corrected, apply_mask)
        metrics = None
        if final_stats is not None:
            before_rgb_shift = (tgt_stats["rgb_mean"] - ref_stats["rgb_mean"]).reshape(-1)[:3] * 255.0
            after_rgb_shift = (final_stats["rgb_mean"] - ref_stats["rgb_mean"]).reshape(-1)[:3] * 255.0
            applied_rgb_shift = rgb_shift.reshape(-1)[:3] * 255.0
            metrics = {
                "label": label,
                "before_rgb_shift": tuple(float(v) for v in before_rgb_shift.detach().cpu()),
                "after_rgb_shift": tuple(float(v) for v in after_rgb_shift.detach().cpu()),
                "applied_rgb_shift": tuple(float(v) for v in applied_rgb_shift.detach().cpu()),
                "before_luma_shift": float(((tgt_stats["luma_mean"] - ref_stats["luma_mean"]) * 255.0).detach().cpu()),
                "after_luma_shift": float(((final_stats["luma_mean"] - ref_stats["luma_mean"]) * 255.0).detach().cpu()),
                "ref_luma": float((ref_stats["luma_mean"] * 255.0).detach().cpu()),
                "target_luma_before": float((tgt_stats["luma_mean"] * 255.0).detach().cpu()),
                "target_luma_after": float((final_stats["luma_mean"] * 255.0).detach().cpu()),
                "strength": float(strength),
            }
        return corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0), metrics

    @staticmethod
    def _format_rgb_tuple(values):
        if values is None:
            return "n/a"
        return "(" + ", ".join(f"{float(v):+.2f}" for v in values[:3]) + ")"

    @classmethod
    def _log_global_rgb_luma_metrics(cls, stage, metrics):
        if getattr(tbg.API, "status", None) != "Dev" or not metrics:
            return
        print(
            f"TBG[Node {tbg.INFO.id}] PiD global_rgb_luma {stage}: "
            f"skipped_per_pixel_colormatch=True strength={metrics['strength']:.3f} "
            f"rgb_shift_before={cls._format_rgb_tuple(metrics['before_rgb_shift'])} "
            f"rgb_shift_after={cls._format_rgb_tuple(metrics['after_rgb_shift'])} "
            f"applied_rgb_shift={cls._format_rgb_tuple(metrics['applied_rgb_shift'])} "
            f"luma_before={metrics['target_luma_before']:.2f} "
            f"luma_after={metrics['target_luma_after']:.2f} "
            f"ref_luma={metrics['ref_luma']:.2f} "
            f"luma_shift_before={metrics['before_luma_shift']:+.2f} "
            f"luma_shift_after={metrics['after_luma_shift']:+.2f}"
        )

    @classmethod
    def _flux2_pid_post_tone_active(cls):
        return cls._pid_post_decode_color_active()

    @classmethod
    def _pid_post_decode_color_active(cls):
        method = getattr(tbg.PARAMS, "color_match_method", None)
        try:
            strength = float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)
        except Exception:
            strength = 1.0
        normal_active = (
            cls._pid_vae_decode_active()
            and method is not None
            and str(method).lower() != "none"
            and strength > 0.0
        )
        return cls._cm_debug_stage_enabled("05_Flux2_PID_PostTone_ColorMatch", normal_active)

    @classmethod
    def _apply_pid_post_decode_color_stabilizer(cls, reference, target, index, seam_mask=None, label="pid_post_tone", apply_global=True):
        if not cls._pid_post_decode_color_active():
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] PiD post-decode color {label} skipped: inactive")
            return target
        if reference is None or target is None or not torch.is_tensor(reference) or not torch.is_tensor(target):
            return target
        ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
        tgt = target.unsqueeze(0) if target.ndim == 3 else target
        if ref.ndim != 4 or tgt.ndim != 4:
            return target
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        try:
            strength = max(0.0, min(1.0, float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)))
        except Exception:
            strength = 1.0
        before = tgt.clone()
        global_metrics = None
        if cls.is_tbg_tile_aware(getattr(tbg.PARAMS, "color_match_method", None)):
            corrected = cls._tile_aware_low_frequency_match(
                ref,
                tgt,
                seam_mask,
                strength=strength,
            )
            if getattr(tbg.API, "status", None) == "Dev":
                try:
                    delta = torch.mean(torch.abs(corrected.to(torch.float32) - before.to(torch.float32))).item()
                    cls.debug_image_to_folder(ref.to(device=corrected.device, dtype=corrected.dtype), str(index) + "_Flux2_PID_PostTone_reference_" + label)
                    cls.debug_image_to_folder(before, str(index) + "_Flux2_PID_PostTone_before_" + label)
                    cls.debug_image_to_folder(corrected, str(index) + "_Flux2_PID_PostTone_after_" + label)
                    print(
                        f"TBG[Node {tbg.INFO.id}] PiD post-decode color {label} applied "
                        f"method=tile_aware_low_frequency strength={strength:.3f} "
                        f"mean_abs_delta={delta:.8f}"
                    )
                except Exception as exc:
                    print(f"TBG[Node {tbg.INFO.id}] PiD tile-aware post-decode color {label} debug failed: {exc}")
            return corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        if apply_global:
            corrected, global_metrics = cls._global_rgb_luma_match(
                ref,
                tgt,
                strength=strength,
                label=f"post_tone_{label}_tile_{index + 1}",
            )
            if global_metrics is None:
                corrected = tgt
        else:
            corrected = tgt

        seam_mean = 0.0
        if seam_mask is not None and torch.is_tensor(seam_mask):
            seam = cls._mask_to_bhw(seam_mask)
            if seam is not None:
                seam = cls._scale_pid_mask(seam.to(torch.float32).clamp(0.0, 1.0), int(tgt.shape[2]), int(tgt.shape[1]))
                seam = cls._mask_to_bhw(seam)
            if seam is not None:
                seam = seam.to(device=corrected.device, dtype=torch.float32).clamp(0.0, 1.0)
                seam_mean = float(seam.mean().detach().cpu())
                if float(seam.max().detach().cpu()) > 1e-5:
                    seam_strength = min(0.60, strength)
                    corrected = TBG_Image.stabilize_tile_low_frequency_from_reference(
                        ref.to(device=corrected.device, dtype=torch.float32),
                        corrected.to(torch.float32),
                        seam,
                        seam,
                        seam_strength,
                    )[0].to(device=tgt.device, dtype=tgt.dtype).clamp(0.0, 1.0)

        corrected = corrected.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        if getattr(tbg.API, "status", None) == "Dev":
            try:
                delta = torch.mean(torch.abs(corrected.to(torch.float32) - before.to(torch.float32))).item()
                cls.debug_image_to_folder(ref.to(device=corrected.device, dtype=corrected.dtype), str(index) + "_Flux2_PID_PostTone_reference_" + label)
                cls.debug_image_to_folder(before, str(index) + "_Flux2_PID_PostTone_before_" + label)
                cls.debug_image_to_folder(corrected, str(index) + "_Flux2_PID_PostTone_after_" + label)
                cls._log_global_rgb_luma_metrics(f"post_tone_{label}_tile_{index + 1}", global_metrics)
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD post-decode color {label} applied "
                    f"method={'global_rgb_luma+seam_low_frequency' if apply_global else 'seam_low_frequency'} "
                    f"strength={strength:.3f} seam_mean={seam_mean:.6f} "
                    f"mean_abs_delta={delta:.8f}"
                )
            except Exception as exc:
                print(f"TBG[Node {tbg.INFO.id}] PiD post-decode color {label} debug failed: {exc}")
        return corrected

    @classmethod
    def _apply_flux2_pid_post_tone(cls, reference, target, index, seam_mask=None, label="pid_post_tone", apply_global=True):
        return cls._apply_pid_post_decode_color_stabilizer(
            reference,
            target,
            index,
            seam_mask=seam_mask,
            label=label,
            apply_global=apply_global,
        )

    @staticmethod
    def _box_blur_bchw(value, kernel):
        kernel = int(kernel)
        if kernel <= 1:
            return value
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        return torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(value, (pad, pad, pad, pad), mode="reflect"),
            kernel,
            stride=1,
        )

    @classmethod
    def _tile_aware_low_frequency_match(cls, reference, target, edge_mask, strength=1.0):
        ref = cls._ensure_bhwc_image(reference)
        tgt = cls._ensure_bhwc_image(target)
        if ref is None or tgt is None or not torch.is_tensor(ref) or not torch.is_tensor(tgt):
            return target
        if ref.ndim != 4 or tgt.ndim != 4:
            return target
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        original_device = tgt.device
        original_dtype = tgt.dtype
        work_device = original_device
        if torch.cuda.is_available():
            try:
                pixel_count = int(tgt.shape[0]) * int(tgt.shape[1]) * int(tgt.shape[2])
                estimated_bytes = pixel_count * 4 * 24
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes > int(estimated_bytes * 1.20):
                    work_device = torch.device("cuda")
            except Exception:
                work_device = torch.device("cuda")
        ref = ref.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        corrected = tgt.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        mask = cls._mask_to_bhw(edge_mask) if torch.is_tensor(edge_mask) else None
        if mask is None:
            mask = torch.ones((int(tgt.shape[0]), int(tgt.shape[1]), int(tgt.shape[2])), device=work_device, dtype=torch.float32)
        else:
            mask = cls._scale_pid_mask(mask.to(torch.float32).clamp(0.0, 1.0), int(tgt.shape[2]), int(tgt.shape[1]))
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                mask = torch.ones((int(tgt.shape[0]), int(tgt.shape[1]), int(tgt.shape[2])), device=work_device, dtype=torch.float32)
        mask = mask.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(1)
        try:
            strength = max(0.0, min(1.0, float(strength)))
        except Exception:
            strength = 1.0
        if strength <= 0.0:
            return target

        base_bchw = corrected.permute(0, 3, 1, 2).contiguous()
        ref_bchw = ref.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = base_bchw.shape
        lr_height = max(48, min(160, int(round(height / 32.0))))
        lr_width = max(48, min(160, int(round(width / 32.0))))
        ref_lr = torch.nn.functional.interpolate(ref_bchw, size=(lr_height, lr_width), mode="area")
        base_lr = torch.nn.functional.interpolate(base_bchw, size=(lr_height, lr_width), mode="area")
        mask_lr = torch.nn.functional.interpolate(mask, size=(lr_height, lr_width), mode="area").clamp(0.0, 1.0)
        diff_lr = (ref_lr - base_lr).clamp(-120.0 / 255.0, 120.0 / 255.0)

        def weighted_delta(diff_value, zone, clamp_value):
            zone = zone.to(device=diff_value.device, dtype=torch.float32).clamp(0.0, 1.0)
            denom = zone.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-5)
            value = (diff_value * zone).sum(dim=(2, 3), keepdim=True) / denom
            return value.clamp(-clamp_value, clamp_value)

        edge_kernel = max(5, min(17, int(round(min(lr_height, lr_width) * 0.12))))
        if edge_kernel % 2 == 0:
            edge_kernel += 1
        yy = torch.linspace(0.0, 1.0, lr_height, device=work_device, dtype=torch.float32).view(1, 1, lr_height, 1)
        xx = torch.linspace(0.0, 1.0, lr_width, device=work_device, dtype=torch.float32).view(1, 1, 1, lr_width)
        side_defs = (
            ((1.0 - yy).pow(2.0), xx),
            (yy.pow(2.0), xx),
            ((1.0 - xx).pow(2.0), yy),
            (xx.pow(2.0), yy),
        )
        field_sum = torch.zeros_like(diff_lr)
        weight_sum = torch.zeros_like(mask_lr)
        zone_count = 0
        for side_weight, axis in side_defs:
            for zone_index in range(4):
                lo = zone_index / 4.0
                hi = (zone_index + 1) / 4.0
                axis_zone = ((axis >= lo) & (axis <= hi)).to(torch.float32)
                zone = (mask_lr * side_weight * axis_zone).clamp(0.0, 1.0)
                if float(zone.sum().detach().cpu()) <= 1.0e-5:
                    continue
                zone_apply = cls._box_blur_bchw(zone.pow(0.45), edge_kernel).clamp(0.0, 1.0)
                zone_delta = weighted_delta(diff_lr, zone, 96.0 / 255.0)
                field_sum = field_sum + zone_delta * zone_apply
                weight_sum = weight_sum + zone_apply
                zone_count += 1
        edge_apply_lr = cls._box_blur_bchw(mask_lr.pow(0.45), edge_kernel).clamp(0.0, 1.0)
        if zone_count > 0:
            edge_field_lr = (field_sum / weight_sum.clamp_min(1.0e-5)).clamp(-96.0 / 255.0, 96.0 / 255.0)
        else:
            edge_field_lr = torch.zeros_like(diff_lr)
        strict_edge_lr = (diff_lr * mask_lr.pow(0.30)).clamp(-112.0 / 255.0, 112.0 / 255.0)
        strict_edge_lr = cls._box_blur_bchw(strict_edge_lr, 3).clamp(-112.0 / 255.0, 112.0 / 255.0)
        strict_mix_lr = mask_lr.pow(0.55).clamp(0.0, 1.0)
        edge_field_lr = (edge_field_lr * (1.0 - strict_mix_lr) + strict_edge_lr * strict_mix_lr).clamp(-112.0 / 255.0, 112.0 / 255.0)
        edge_field = torch.nn.functional.interpolate(
            edge_field_lr,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).clamp(-112.0 / 255.0, 112.0 / 255.0)
        edge_apply = torch.nn.functional.interpolate(
            edge_apply_lr,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        edge_apply = torch.maximum(edge_apply, mask * 0.98).clamp(0.0, 1.0)

        inner_lr = (1.0 - mask_lr).clamp(0.0, 1.0)
        inner_field = torch.zeros_like(base_bchw)
        inner_apply = torch.zeros_like(mask)
        if float(inner_lr.sum().detach().cpu()) > 1.0e-5:
            inner_grid = 24
            inner_clamp = 56.0 / 255.0
            global_inner_delta = weighted_delta(diff_lr, inner_lr, inner_clamp)
            control = global_inner_delta.expand(-1, -1, inner_grid, inner_grid).clone()
            for y_index in range(inner_grid):
                y_lo = y_index / float(inner_grid)
                y_hi = (y_index + 1) / float(inner_grid)
                y_zone = ((yy >= y_lo) & (yy <= y_hi)).to(torch.float32)
                for x_index in range(inner_grid):
                    x_lo = x_index / float(inner_grid)
                    x_hi = (x_index + 1) / float(inner_grid)
                    x_zone = ((xx >= x_lo) & (xx <= x_hi)).to(torch.float32)
                    zone = (inner_lr * y_zone * x_zone).clamp(0.0, 1.0)
                    if float(zone.sum().detach().cpu()) <= 1.0e-5:
                        continue
                    local_delta = weighted_delta(diff_lr, zone, inner_clamp)
                    control[:, :, y_index:y_index + 1, x_index:x_index + 1] = (
                        global_inner_delta * 0.50 + local_delta * 0.50
                    ).clamp(-inner_clamp, inner_clamp)
            for _ in range(12):
                control = cls._box_blur_bchw(control, 3)
            control = (control * 0.80 + global_inner_delta.expand_as(control) * 0.20).clamp(-inner_clamp, inner_clamp)
            inner_field = torch.nn.functional.interpolate(
                control,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).clamp(-inner_clamp, inner_clamp)
            inner_apply_lr = cls._box_blur_bchw((1.0 - edge_apply_lr * 0.55).clamp(0.0, 1.0), edge_kernel).clamp(0.0, 1.0)
            inner_apply = torch.nn.functional.interpolate(
                inner_apply_lr,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
            luma = (
                base_bchw[:, 0:1, :, :] * 0.2126
                + base_bchw[:, 1:2, :, :] * 0.7152
                + base_bchw[:, 2:3, :, :] * 0.0722
            ).clamp(0.0, 1.0)
            dark_gate = ((luma - 0.08) / 0.24).clamp(0.0, 1.0)
            inner_apply = inner_apply * (0.35 + dark_gate * 0.65)

        edge_weight = (edge_apply * min(1.0, strength)).clamp(0.0, 1.0)
        inner_weight = (inner_apply * (1.0 - edge_weight).clamp(0.0, 1.0) * min(1.0, strength)).clamp(0.0, 1.0)
        combined_field = (
            edge_field * edge_weight
            + inner_field * inner_weight
        ).clamp(-90.0 / 255.0, 90.0 / 255.0)
        corrected = (corrected + combined_field.permute(0, 2, 3, 1).contiguous()).clamp(0.0, 1.0)
        corrected = (corrected + edge_apply.permute(0, 2, 3, 1).contiguous() * (0.25 / 255.0) * min(1.0, strength)).clamp(0.0, 1.0)
        return corrected.to(device=original_device, dtype=original_dtype).clamp(0.0, 1.0)

    @classmethod
    def _segment_pixel_grid_color_match(cls, reference, target, index_seg, placement_mask=None, strength=1.0, label="segment_tile_aware", placement_source_override=None):
        ref = cls._ensure_bhwc_image(reference)
        tgt = cls._ensure_bhwc_image(target)
        if ref is None or tgt is None or not torch.is_tensor(ref) or not torch.is_tensor(tgt):
            return target, None
        if ref.ndim != 4 or tgt.ndim != 4:
            return target, None
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        try:
            strength = max(0.0, min(1.0, float(strength)))
        except Exception:
            strength = 1.0
        if strength <= 0.0:
            return target, None

        original_device = tgt.device
        original_dtype = tgt.dtype
        work_device = original_device
        if torch.cuda.is_available():
            try:
                pixel_count = int(tgt.shape[0]) * int(tgt.shape[1]) * int(tgt.shape[2])
                estimated_bytes = pixel_count * 4 * 28
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes > int(estimated_bytes * 1.20):
                    work_device = torch.device("cuda")
            except Exception:
                work_device = torch.device("cuda")

        ref = ref.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        base = tgt.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        batch, height, width, _ = base.shape

        placement = cls._mask_to_bhw(placement_mask) if torch.is_tensor(placement_mask) else None
        if placement is None:
            placement, placement_source, placement_coverage = cls._segment_native_crop_mask_from_sources(
                index_seg,
                int(width),
                int(height),
                ("compositing_mask", "segms_cropped_masks", "inpainting_mask"),
            )
        else:
            placement_source = str(placement_source_override or "provided_placement_mask")
            placement = cls._scale_pid_mask(placement.to(torch.float32).clamp(0.0, 1.0), int(width), int(height))
            placement = cls._mask_to_bhw(placement)
            placement_coverage = 0.0
            if placement is not None:
                placement_coverage = float((placement > 0.001).to(torch.float32).mean().detach().cpu())
        object_mask, object_source, object_coverage = cls._segment_native_crop_mask_from_sources(
            index_seg,
            int(width),
            int(height),
            ("segment_binary_masks", "segms_cropped_masks", "inpainting_mask", "compositing_mask"),
        )
        if placement is None and object_mask is None:
            corrected = cls._tile_aware_low_frequency_match(ref, base, None, strength=strength)
            return corrected.to(device=original_device, dtype=original_dtype).clamp(0.0, 1.0), {
                "fallback": True,
                "reason": "no_segment_masks",
                "label": label,
            }
        if placement is None:
            placement = object_mask
            placement_source = object_source
            placement_coverage = object_coverage
        if object_mask is None:
            object_mask = placement
            object_source = placement_source
            object_coverage = placement_coverage

        placement = placement.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        object_mask = object_mask.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        if int(placement.shape[-2]) != height or int(placement.shape[-1]) != width:
            placement = cls._scale_pid_mask(placement, int(width), int(height))
            placement = cls._mask_to_bhw(placement).to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        if int(object_mask.shape[-2]) != height or int(object_mask.shape[-1]) != width:
            object_mask = cls._scale_pid_mask(object_mask, int(width), int(height))
            object_mask = cls._mask_to_bhw(object_mask).to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        placement_bhw = placement.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        global_base, global_metrics = cls._global_rgb_luma_match(
            ref,
            base,
            strength=strength,
            apply_mask=placement_bhw,
            label=label + "_global_base",
        )
        global_base = cls._ensure_bhwc_image(global_base)
        if global_base is None or global_base.ndim != 4:
            global_base = base
        global_base = global_base.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)

        placement_bchw = placement_bhw.unsqueeze(1)
        object_bchw = object_mask.unsqueeze(1)
        if float(object_bchw.max().detach().cpu()) <= 0.001:
            object_bchw = placement_bchw
        use_restore_support = str(placement_source) == "pid_native_crop_restore_mask_4x"
        support_bchw = placement_bchw if use_restore_support else object_bchw

        min_side = max(1, min(int(height), int(width)))
        if min_side <= 1536:
            cell_px = 48
        elif min_side <= 4096:
            cell_px = 64
        else:
            cell_px = 96
        grid_h = max(3, min(72, int((int(height) + cell_px - 1) // cell_px)))
        grid_w = max(3, min(72, int((int(width) + cell_px - 1) // cell_px)))
        if min_side > 4096:
            border_px = max(256, min(384, int(round(cell_px * 2.75))))
        elif min_side > 1536:
            border_px = max(128, min(224, int(round(cell_px * 2.25))))
        else:
            border_px = max(64, min(128, int(round(cell_px * 1.50))))
        border_kernel = border_px * 2 + 1

        binary_threshold = 0.001 if use_restore_support else 0.5
        binary = (support_bchw > binary_threshold).to(torch.float32)
        if float(binary.sum().detach().cpu()) <= 1.0e-5:
            binary = support_bchw.clamp(0.0, 1.0)
        dilated = torch.nn.functional.max_pool2d(binary, kernel_size=border_kernel, stride=1, padding=border_px)
        erode_radius = max(2, border_px // 2)
        erode_kernel = erode_radius * 2 + 1
        eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - binary, kernel_size=erode_kernel, stride=1, padding=erode_radius)
        outer_radius = max(2, border_px // 2)
        inner_band = (binary - eroded).clamp(0.0, 1.0)
        outer_gradient = torch.zeros_like(binary)
        try:
            from scipy.ndimage import distance_transform_edt
            binary_cpu = (binary[:, 0].detach().cpu().numpy() > 0.5)
            outer_batches = []
            for batch_index in range(binary_cpu.shape[0]):
                outside_dist = distance_transform_edt(~binary_cpu[batch_index]).astype(np.float32)
                outer_np = 1.0 - np.clip(outside_dist / float(max(1, outer_radius)), 0.0, 1.0)
                outer_np[binary_cpu[batch_index]] = 0.0
                outer_batches.append(torch.from_numpy(outer_np).unsqueeze(0))
            outer_gradient = torch.stack(outer_batches, dim=0).to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        except Exception:
            outer_kernel = outer_radius * 2 + 1
            dilated_outer = torch.nn.functional.max_pool2d(binary, kernel_size=outer_kernel, stride=1, padding=outer_radius)
            outer_gradient = cls._box_blur_bchw((dilated_outer - binary).clamp(0.0, 1.0), max(3, outer_radius | 1)).clamp(0.0, 1.0)
        y = torch.arange(height, device=work_device, dtype=torch.float32).view(1, 1, height, 1)
        x = torch.arange(width, device=work_device, dtype=torch.float32).view(1, 1, 1, width)
        edge_dist = torch.minimum(
            torch.minimum(x, float(width - 1) - x),
            torch.minimum(y, float(height - 1) - y),
        )
        edge_guard_width = max(16.0, float(border_px) * 0.75)
        crop_edge_gate = (edge_dist / edge_guard_width).clamp(0.0, 1.0)
        crop_edge_gate = crop_edge_gate * crop_edge_gate * (3.0 - 2.0 * crop_edge_gate)
        if use_restore_support:
            correction_support = placement_bchw.clamp(0.0, 1.0)
            border_mask = correction_support
            inner_mask = placement_bchw.clamp(0.0, 1.0).pow(1.35)
        else:
            correction_support = torch.maximum(
                placement_bchw.clamp(0.0, 1.0),
                (binary + outer_gradient).clamp(0.0, 1.0),
            ).clamp(0.0, 1.0) * crop_edge_gate
            border_mask = torch.maximum(binary, outer_gradient).clamp(0.0, 1.0) * correction_support
            inner_mask = (object_bchw * (1.0 - inner_band * 0.35)).clamp(0.0, 1.0) * crop_edge_gate
        if float(inner_mask.sum().detach().cpu()) <= 1.0e-5:
            inner_mask = support_bchw.clamp(0.0, 1.0)

        ref_bchw = ref.permute(0, 3, 1, 2).contiguous()
        base_bchw = base.permute(0, 3, 1, 2).contiguous()
        global_bchw = global_base.permute(0, 3, 1, 2).contiguous()
        diff = (ref_bchw - global_bchw).clamp(-96.0 / 255.0, 96.0 / 255.0)

        def grid_field(mask_bchw, clamp_value, global_blend, smooth_passes):
            mask_bchw = mask_bchw.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
            masked = diff * mask_bchw
            num = torch.nn.functional.interpolate(masked, size=(grid_h, grid_w), mode="area")
            den = torch.nn.functional.interpolate(mask_bchw, size=(grid_h, grid_w), mode="area").clamp_min(1.0e-5)
            local = (num / den).clamp(-clamp_value, clamp_value)
            global_den = mask_bchw.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-5)
            global_delta = (masked.sum(dim=(2, 3), keepdim=True) / global_den).clamp(-clamp_value, clamp_value)
            field = (local * (1.0 - global_blend) + global_delta.expand_as(local) * global_blend).clamp(-clamp_value, clamp_value)
            for _ in range(int(max(0, smooth_passes))):
                field = cls._box_blur_bchw(field, 3).clamp(-clamp_value, clamp_value)
            return torch.nn.functional.interpolate(field, size=(height, width), mode="bicubic", align_corners=False).clamp(-clamp_value, clamp_value)

        placement_field_mask = correction_support.clamp(0.0, 1.0)
        if use_restore_support:
            placement_field = grid_field(placement_field_mask, 72.0 / 255.0, 0.18, 8)
            border_field = torch.zeros_like(placement_field)
            inner_field = torch.zeros_like(placement_field)
        else:
            placement_field = grid_field(placement_field_mask, 64.0 / 255.0, 0.22, 5)
            border_field = grid_field(border_mask, 56.0 / 255.0, 0.24, 5)
            inner_field = grid_field(inner_mask, 32.0 / 255.0, 0.42, 8)

        apply_kernel = max(33, min(513, int(round(border_px * 1.50))))
        if apply_kernel % 2 == 0:
            apply_kernel += 1
        if use_restore_support:
            placement_apply = correction_support.pow(0.85).clamp(0.0, 1.0)
            border_apply = torch.zeros_like(placement_apply)
            inner_apply = torch.zeros_like(placement_apply)
            soft_limiter = torch.ones_like(placement_apply)
            placement_weight = (placement_apply * strength * 0.90).clamp(0.0, 1.0)
            border_weight = torch.zeros_like(placement_weight)
            inner_weight = torch.zeros_like(placement_weight)
        else:
            placement_apply = cls._box_blur_bchw(placement_bchw.pow(0.35), apply_kernel).clamp(0.0, 1.0)
            border_apply = cls._box_blur_bchw(border_mask.pow(0.45), apply_kernel).clamp(0.0, 1.0)
            inner_apply = cls._box_blur_bchw(inner_mask.pow(0.60), apply_kernel).clamp(0.0, 1.0)
            inner_apply = (inner_apply * (1.0 - border_apply * 0.35)).clamp(0.0, 1.0)
            soft_limiter = correction_support.pow(0.35).clamp(0.0, 1.0)
            placement_weight = (placement_apply * soft_limiter * strength * 0.52).clamp(0.0, 1.0)
            border_weight = (border_apply * soft_limiter * strength * 0.62).clamp(0.0, 1.0)
            inner_weight = (inner_apply * soft_limiter * strength * 0.12).clamp(0.0, 1.0)

        combined = (
            placement_field * placement_weight
            + border_field * border_weight
            + inner_field * inner_weight
        ).clamp(-64.0 / 255.0, 64.0 / 255.0)
        outer_darken_gate = torch.zeros_like(correction_support) if use_restore_support else (outer_gradient * (1.0 - binary)).clamp(0.0, 1.0)
        if float(outer_darken_gate.max().detach().cpu()) > 0.001:
            luma_delta = (
                combined[:, 0:1, :, :] * 0.2126
                + combined[:, 1:2, :, :] * 0.7152
                + combined[:, 2:3, :, :] * 0.0722
            )
            negative_luma = luma_delta.clamp(max=0.0)
            combined = (combined - negative_luma * outer_darken_gate * 0.65).clamp(-64.0 / 255.0, 64.0 / 255.0)
        candidate_bchw_pre = (global_bchw + combined).clamp(0.0, 1.0)
        if use_restore_support:
            core_mask = placement_bchw.clamp(0.0, 1.0).pow(1.80)
            core_residual = (ref_bchw - candidate_bchw_pre).clamp(-96.0 / 255.0, 96.0 / 255.0)
            core_grid_h = grid_h
            core_grid_w = grid_w
            try:
                core_grid_h = max(3, min(96, int((int(height) + max(1, cell_px // 2) - 1) // max(1, cell_px // 2))))
                core_grid_w = max(3, min(96, int((int(width) + max(1, cell_px // 2) - 1) // max(1, cell_px // 2))))
            except Exception:
                core_grid_h = grid_h
                core_grid_w = grid_w

            def core_grid_field(mask_bchw, clamp_value, global_blend, smooth_passes):
                mask_bchw = mask_bchw.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
                masked = core_residual * mask_bchw
                num = torch.nn.functional.interpolate(masked, size=(core_grid_h, core_grid_w), mode="area")
                den = torch.nn.functional.interpolate(mask_bchw, size=(core_grid_h, core_grid_w), mode="area").clamp_min(1.0e-5)
                local = (num / den).clamp(-clamp_value, clamp_value)
                global_den = mask_bchw.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-5)
                global_delta = (masked.sum(dim=(2, 3), keepdim=True) / global_den).clamp(-clamp_value, clamp_value)
                field = (local * (1.0 - global_blend) + global_delta.expand_as(local) * global_blend).clamp(-clamp_value, clamp_value)
                for _ in range(int(max(0, smooth_passes))):
                    field = cls._box_blur_bchw(field, 3).clamp(-clamp_value, clamp_value)
                return torch.nn.functional.interpolate(field, size=(height, width), mode="bicubic", align_corners=False).clamp(-clamp_value, clamp_value)

            core_field = core_grid_field(core_mask, 48.0 / 255.0, 0.25, 5)
            candidate_bchw_pre = (candidate_bchw_pre + core_field * core_mask * strength * 0.55).clamp(0.0, 1.0)

        edge_low_kernel = max(33, min(257, int(round(border_px * 0.75))))
        if edge_low_kernel % 2 == 0:
            edge_low_kernel += 1
        region_edge_delta = (
            cls._box_blur_bchw(ref_bchw, edge_low_kernel)
            - cls._box_blur_bchw(candidate_bchw_pre, edge_low_kernel)
        ).clamp(-50.0 / 255.0, 50.0 / 255.0)
        edge_polish = torch.zeros_like(border_apply) if use_restore_support else (border_apply * soft_limiter * strength * 0.62 * (1.0 - outer_darken_gate * 0.55)).clamp(0.0, 1.0)
        candidate_bchw = (candidate_bchw_pre + region_edge_delta * edge_polish).clamp(0.0, 1.0)
        candidate = candidate_bchw.permute(0, 2, 3, 1).contiguous()
        metric_weight = (border_mask * 1.25 + inner_mask * 0.75).clamp(0.0, 1.0)
        metric_weight = torch.maximum(metric_weight, correction_support.pow(0.50) * 0.85).clamp(0.0, 1.0)
        metric_den = metric_weight.sum().clamp_min(1.0e-5)
        metric_kernel = max(17, min(129, int(round(cell_px * 0.60))))
        if metric_kernel % 2 == 0:
            metric_kernel += 1
        ref_metric = cls._box_blur_bchw(ref_bchw, metric_kernel)
        global_metric = cls._box_blur_bchw(global_bchw, metric_kernel)
        candidate_metric = cls._box_blur_bchw(candidate_bchw, metric_kernel)
        global_error = ((global_metric - ref_metric).abs() * metric_weight).sum() / metric_den
        candidate_error = ((candidate_metric - ref_metric).abs() * metric_weight).sum() / metric_den
        raw_global_error = ((global_bchw - ref_bchw).abs() * metric_weight).sum() / metric_den
        raw_candidate_error = ((candidate_bchw - ref_bchw).abs() * metric_weight).sum() / metric_den
        local_accepted = bool(
            float(candidate_error.detach().cpu()) <= float(global_error.detach().cpu()) * 1.015
            or (
                float(candidate_error.detach().cpu()) <= float(global_error.detach().cpu()) * 1.08
                and float(raw_candidate_error.detach().cpu()) <= float(raw_global_error.detach().cpu()) * 1.25
            )
        )
        corrected = candidate if local_accepted else global_base
        correction = (corrected - base).abs()
        metrics = {
            "fallback": False,
            "label": label,
            "local_accepted": local_accepted,
            "global_error": float(global_error.detach().cpu()),
            "candidate_error": float(candidate_error.detach().cpu()),
            "raw_global_error": float(raw_global_error.detach().cpu()),
            "raw_candidate_error": float(raw_candidate_error.detach().cpu()),
            "cell_px": int(cell_px),
            "grid": (int(grid_w), int(grid_h)),
            "border_px": int(border_px),
            "outer_border_px": int(outer_radius),
            "edge_guard_px": float(edge_guard_width),
            "placement_source": placement_source,
            "object_source": object_source,
            "placement_coverage": float((placement_bchw > 0.001).to(torch.float32).mean().detach().cpu()),
            "correction_support_coverage": float((correction_support > 0.001).to(torch.float32).mean().detach().cpu()),
            "object_coverage": float((object_bchw > 0.001).to(torch.float32).mean().detach().cpu()),
            "border_coverage": float((border_mask > 0.001).to(torch.float32).mean().detach().cpu()),
            "inner_coverage": float((inner_mask > 0.001).to(torch.float32).mean().detach().cpu()),
            "placement_weight_coverage": float((placement_weight > 0.001).to(torch.float32).mean().detach().cpu()),
            "edge_polish_coverage": float((edge_polish > 0.001).to(torch.float32).mean().detach().cpu()),
            "mean_correction": float(correction.mean().detach().cpu()),
            "max_correction": float(correction.max().detach().cpu()),
            "strength": float(strength),
        }
        if getattr(tbg.API, "status", None) == "Dev":
            try:
                debug_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(label))[:96]
                cls.debug_image_to_folder(MaskToImage_execute(border_mask[:, 0].detach().cpu())[0], debug_label + "_Segment_TileAwareColor_BorderMask")
                cls.debug_image_to_folder(MaskToImage_execute(inner_mask[:, 0].detach().cpu())[0], debug_label + "_Segment_TileAwareColor_InnerMask")
                cls.debug_image_to_folder((combined.permute(0, 2, 3, 1).contiguous() * 5.0 + 0.5).clamp(0.0, 1.0), debug_label + "_Segment_TileAwareColor_CorrectionField")
            except Exception as exc:
                print(f"TBG[Node {tbg.INFO.id}] segment tile-aware debug save failed: {exc}")
        return corrected.to(device=original_device, dtype=original_dtype).clamp(0.0, 1.0), metrics

    @classmethod
    def _normal_grid_specs_for_canvas(cls, width, height):
        specs = []
        source_w = int(getattr(tbg.OUTPUTS, "upscaled_image", torch.empty(1, 1, 1, 3)).shape[2]) if torch.is_tensor(getattr(tbg.OUTPUTS, "upscaled_image", None)) else width
        source_h = int(getattr(tbg.OUTPUTS, "upscaled_image", torch.empty(1, 1, 1, 3)).shape[1]) if torch.is_tensor(getattr(tbg.OUTPUTS, "upscaled_image", None)) else height
        scale_x = float(width) / max(1.0, float(source_w))
        scale_y = float(height) / max(1.0, float(source_h))
        for spec in list(getattr(tbg.PARAMS, "grid_specs", []) or []):
            try:
                if cls._is_segment_spec(spec):
                    continue
                _, _, _, x, y, tile_w, tile_h = spec[:7]
                sx = int(round(float(x) * scale_x))
                sy = int(round(float(y) * scale_y))
                sw = int(round(float(tile_w) * scale_x))
                sh = int(round(float(tile_h) * scale_y))
                sx = max(0, min(width - 1, sx))
                sy = max(0, min(height - 1, sy))
                sw = max(1, min(width - sx, sw))
                sh = max(1, min(height - sy, sh))
                specs.append((sx, sy, sw, sh))
            except Exception:
                continue
        return specs

    @classmethod
    def _normal_vae_decode_for_pid_color_reference(cls, vaedecoder, latent_output, width, height, index, label):
        try:
            if getattr(tbg.KSAMPLER, "tiled", False):
                normal_vae = (nodes.VAEDecodeTiled().decode(
                    vaedecoder,
                    latent_output,
                    tbg.SIZE.tile_size_vae,
                    tbg.SIZE.tile_size_vae // 4,
                    tbg.SIZE.tile_size_vae // 4,
                )[0].unsqueeze(0))[0]
                decode_mode = "VAEDecodeTiled"
            else:
                normal_vae = (nodes.VAEDecode().decode(vaedecoder, latent_output)[0].unsqueeze(0))[0]
                decode_mode = "VAEDecode"
            normal_vae = nodes.ImageScale().upscale(
                normal_vae,
                "lanczos",
                int(width),
                int(height),
                False,
            )[0]
            if tbg.API.status == "Dev":
                cls.debug_image_to_folder(normal_vae, str(index) + "_" + label)
                print(
                    f"TBG[Node {tbg.INFO.id}] normal VAE forensic decode tile {index + 1}: "
                    f"label={label} mode={decode_mode}"
                )
            return normal_vae
        except Exception as exc:
            print(
                f"TBG[Node {tbg.INFO.id}] Flux2 PiD normal-VAE color reference failed "
                f"for tile {index + 1}: {exc}"
            )
            return None

    @classmethod
    def _flux2_pid_segment_preprotect_image(cls, source_image, sampled_image, edit_mask, index):
        if source_image is None or sampled_image is None or edit_mask is None:
            return sampled_image, None, None
        if not torch.is_tensor(source_image) or not torch.is_tensor(sampled_image) or not torch.is_tensor(edit_mask):
            return sampled_image, None, None
        source = source_image.unsqueeze(0) if source_image.ndim == 3 else source_image
        sampled = sampled_image.unsqueeze(0) if sampled_image.ndim == 3 else sampled_image
        if source.ndim != 4 or sampled.ndim != 4:
            return sampled_image, None, None
        if int(source.shape[1]) != int(sampled.shape[1]) or int(source.shape[2]) != int(sampled.shape[2]):
            source = nodes.ImageScale().upscale(source, "lanczos", int(sampled.shape[2]), int(sampled.shape[1]), False)[0]

        mask = cls._mask_to_bhw(edit_mask)
        if mask is None:
            return sampled_image, None, None
        mask = cls._scale_pid_mask(mask.to(torch.float32).clamp(0.0, 1.0), int(sampled.shape[2]), int(sampled.shape[1]))
        mask = cls._mask_to_bhw(mask)
        if mask is None:
            return sampled_image, None, None
        mask = mask.to(device=sampled.device, dtype=torch.float32).clamp(0.0, 1.0)
        if float(mask.max().detach().cpu()) <= 1e-5:
            return source.to(device=sampled_image.device, dtype=sampled_image.dtype), mask, mask

        core_mask = (mask >= 0.50).to(torch.float32)
        feather_radius = max(2, min(12, int(round(min(int(sampled.shape[1]), int(sampled.shape[2])) / 96.0))))
        final_mask = core_mask
        if feather_radius > 0:
            final_mask = torch.nn.functional.max_pool2d(
                core_mask.unsqueeze(1),
                kernel_size=feather_radius * 2 + 1,
                stride=1,
                padding=feather_radius,
            )[:, 0]
            final_mask = torch.nn.functional.avg_pool2d(
                final_mask.unsqueeze(1),
                kernel_size=feather_radius * 2 + 1,
                stride=1,
                padding=feather_radius,
            )[:, 0]
        final_mask = torch.maximum(mask, final_mask).clamp(0.0, 1.0)

        source_f = source.to(device=sampled.device, dtype=torch.float32)
        sampled_f = sampled.to(torch.float32)
        final_bhwc = final_mask.unsqueeze(-1)
        core_bhwc = core_mask.to(device=sampled.device, dtype=torch.float32).unsqueeze(-1)

        source_low = cls._blur_bhwc(source_f, sigma=2.0)
        sampled_low = cls._blur_bhwc(sampled_f, sigma=2.0)
        low = source_low * (1.0 - final_bhwc) + sampled_low * final_bhwc
        detail = (source_f - source_low) * (1.0 - core_bhwc) + (sampled_f - sampled_low) * core_bhwc
        protected = (low + detail).clamp(0.0, 1.0).to(device=sampled_image.device, dtype=sampled_image.dtype)

        if tbg.API.status == "Dev":
            try:
                cls.debug_image_to_folder(source.to(device=sampled.device, dtype=sampled.dtype), str(index) + "_Segment_PID_PreProtect_Source")
                cls.debug_image_to_folder(sampled, str(index) + "_Segment_PID_PreProtect_Sampled")
                cls.debug_image_to_folder(MaskToImage_execute(mask)[0], str(index) + "_Segment_PID_PreProtect_EditMask")
                cls.debug_image_to_folder(MaskToImage_execute(final_mask)[0], str(index) + "_Segment_PID_PreProtect_FinalMask")
                cls.debug_image_to_folder(protected, str(index) + "_Segment_PID_PreProtect_Output")
                print(
                    f"TBG[Node {tbg.INFO.id}] Flux2 PiD segment pre-protect tile {index + 1}: "
                    f"mask_mean={float(mask.mean().detach().cpu()):.6f} "
                    f"final_mask_mean={float(final_mask.mean().detach().cpu()):.6f} "
                    f"feather_radius={feather_radius}"
                )
            except Exception as exc:
                print(f"TBG[Node {tbg.INFO.id}] Flux2 PiD segment pre-protect debug failed for tile {index + 1}: {exc}")
        return protected, mask, final_mask

    @classmethod
    def _flux2_pid_segment_context_color_lock(cls, reference, target, segment_mask, index):
        """Use the Flux2 tile tone fix only on segment background/context pixels."""
        if reference is None or target is None or not torch.is_tensor(reference) or not torch.is_tensor(target):
            return target
        if not bool(getattr(tbg.PARAMS, "Flux2_Tile_Color_Correction", True)):
            return target
        ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
        tgt = target.unsqueeze(0) if target.ndim == 3 else target
        if ref.ndim != 4 or tgt.ndim != 4:
            return target
        if int(ref.shape[1]) != int(tgt.shape[1]) or int(ref.shape[2]) != int(tgt.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "lanczos", int(tgt.shape[2]), int(tgt.shape[1]), False)[0]
        mask = cls._mask_to_bhw(segment_mask)
        if mask is None:
            return target
        mask = cls._scale_pid_mask(mask.to(torch.float32).clamp(0.0, 1.0), int(tgt.shape[2]), int(tgt.shape[1]))
        mask = cls._mask_to_bhw(mask)
        if mask is None:
            return target
        mask = mask.to(device=tgt.device, dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
        # Full correction outside the segment/context side, taper off before the
        # object core so requested inpaint color changes survive.
        authority = (1.0 - ((mask - 0.35) / 0.45).clamp(0.0, 1.0))
        authority = authority * authority * (3.0 - 2.0 * authority)
        if float(authority.sum().detach().cpu()) <= 1e-6:
            return target
        try:
            strength = float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)
        except Exception:
            strength = 1.0
        ref = ref.to(device=tgt.device, dtype=torch.float32)
        tgt_f = tgt.to(torch.float32)
        corrected = TBG_Image.stabilize_tile_low_frequency_from_reference(
            ref,
            tgt_f,
            authority[..., 0],
            authority[..., 0],
            strength,
        )[0].to(device=tgt.device, dtype=tgt.dtype).clamp(0.0, 1.0)

        def weighted_shift(reference_image, target_image, weight):
            weight = weight.to(device=target_image.device, dtype=torch.float32)
            denom = weight.sum(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
            shift = ((target_image.to(torch.float32) - reference_image.to(torch.float32)) * weight).sum(
                dim=(0, 1, 2),
                keepdim=True,
            ) / denom
            shift = shift.reshape(-1)[:3] * 255.0
            return tuple(float(v) for v in shift.detach().cpu())

        before_shift = weighted_shift(ref, tgt_f, authority)
        after_shift = weighted_shift(ref, corrected.to(torch.float32), authority)
        print(
            f"TBG[Node {tbg.INFO.id}] Flux2 PiD segment context color lock tile {index + 1}: "
            f"authority={float(authority.mean().detach().cpu()):.4f} "
            f"mean_shift_before={cls._format_rgb_shift(before_shift)} "
            f"mean_shift_after={cls._format_rgb_shift(after_shift)}"
        )
        if tbg.API.status == "Dev":
            cls.debug_image_to_folder(MaskToImage_execute(authority[..., 0])[0], str(index) + "_Segment_PID_context_color_authority_mask")
            cls.debug_image_to_folder(corrected, str(index) + "_Segment_PID_sampled_context_color_locked")
        return corrected

    @classmethod
    def _pid_release_runtime(cls, reason):
        runtime = getattr(tbg.TEMP, "pid_refiner_runtime", None)
        if runtime is None:
            return
        unload_pid_refiner_runtime(runtime, reason=reason, aggressive=cls._pid_profile_is_ultra())
        tbg.TEMP.pid_refiner_runtime = None
        tbg.TEMP.pid_refiner_runtime_key = None

    @classmethod
    def _pid_prepare_runtime(cls, latent_output, sampler_name, scheduler, steps):
        profile = cls._pid_vram_profile()
        resolved_pid_model_name = select_pid_refiner_model(
            latent_output,
            model_type=getattr(tbg.KSAMPLER, "model_type", None),
        )
        resolved_pid_spec = PID_UPSCALE_SPECS.get(resolved_pid_model_name)
        resolved_pid_file = getattr(resolved_pid_spec, "diffusion_model", None)
        print(
            f"TBG[Node {tbg.INFO.id}] PiD auto model resolved: "
            f"model_type={getattr(tbg.KSAMPLER, 'model_type', None)} "
            f"channels={int(latent_output['samples'].shape[1])} "
            f"selector='{resolved_pid_model_name}' file='{resolved_pid_file}'"
        )
        runtime_key = (
            getattr(tbg.KSAMPLER, "model_type", None),
            int(latent_output["samples"].shape[1]),
            resolved_pid_model_name,
            resolved_pid_file,
            sampler_name,
            scheduler,
            int(steps),
            id(getattr(tbg.KSAMPLER, "pid_model", None)) if getattr(tbg.KSAMPLER, "pid_model", None) is not None else None,
        )

        if cls._pid_profile_is_fast():
            runtime = getattr(tbg.TEMP, "pid_refiner_runtime", None)
            if runtime is not None and getattr(tbg.TEMP, "pid_refiner_runtime_key", None) == runtime_key:
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD VRAM profile Fast: reusing PiD model/CLIP runtime "
                    f"selector='{getattr(runtime, 'upscale_model_name', None)}' "
                    f"file='{getattr(getattr(runtime, 'spec', None), 'diffusion_model', None)}'."
                )
                return runtime
            cls._pid_release_runtime("replacing cached Fast runtime")
        else:
            print(f"TBG[Node {tbg.INFO.id}] PiD VRAM profile {profile}: unloading base models before PiD model/CLIP load.")
            mm.unload_all_models()
            mm.soft_empty_cache()
            if cls._pid_profile_is_ultra() and torch.cuda.is_available():
                torch.cuda.empty_cache()

        runtime = load_pid_refiner_runtime(
            latent_output,
            model_type=getattr(tbg.KSAMPLER, "model_type", None),
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=steps,
            denoise=1.0,
            pid_model=getattr(tbg.KSAMPLER, "pid_model", None),
        )
        tbg.TEMP.pid_refiner_runtime = runtime
        tbg.TEMP.pid_refiner_runtime_key = runtime_key
        print(
            f"TBG[Node {tbg.INFO.id}] PiD VRAM profile {profile}: loaded PiD model/CLIP runtime "
            f"selector='{getattr(runtime, 'upscale_model_name', None)}' "
            f"file='{getattr(getattr(runtime, 'spec', None), 'diffusion_model', None)}'."
        )
        return runtime

    @classmethod
    def _pid_finish_tile_runtime(cls):
        if cls._pid_profile_is_fast():
            return
        cls._pid_release_runtime(f"{cls._pid_vram_profile()} tile complete")


    @staticmethod
    def image_to_folder(image, filename):
        if image is not None:
            image = TBG_Refiner_v1._normalize_debug_image_for_save(image)
            if image is None:
                return
            filename_prefix = "TBG/compareTiles/"+ filename
            preview = nodes.PreviewImage()
            _ = preview.save_images(image, filename_prefix, None, None)['ui']['images']
            TBG_Refiner_v1.prune_compare_tiles_folder()

    @staticmethod
    def _normalize_debug_image_for_save(image):
        if not torch.is_tensor(image):
            return image
        img = image.detach()
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = img.permute(1, 2, 0).unsqueeze(0).contiguous()
            elif img.shape[-1] in (1, 3, 4):
                img = img.unsqueeze(0)
            else:
                img = img.unsqueeze(-1).expand(-1, -1, -1, 3)
        elif img.ndim == 4:
            if img.shape[-1] in (1, 3, 4):
                pass
            elif img.shape[1] in (1, 3, 4):
                img = img.permute(0, 2, 3, 1).contiguous()
            else:
                print(f"[TBG Debug] unsupported debug image shape for save: {tuple(img.shape)}")
                return None
        else:
            print(f"[TBG Debug] unsupported debug image rank for save: {tuple(img.shape)}")
            return None

        if img.shape[-1] == 1:
            img = img.expand(-1, -1, -1, 3)
        elif img.shape[-1] > 4:
            img = img[..., :3]

        if img.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            img = img.to(torch.float32)
        else:
            img = img.to(torch.float32)
        if img.numel() > 0 and float(img.max().detach().cpu()) > 1.0:
            img = img / 255.0
        return img.clamp(0.0, 1.0).contiguous()

    @staticmethod
    def prune_compare_tiles_folder(max_files=400):
        try:
            folder = os.path.join(folder_paths.get_temp_directory(), "TBG", "compareTiles")
            if not os.path.isdir(folder):
                return
            files = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, name))
            ]
            overflow = len(files) - int(max_files)
            if overflow <= 0:
                return
            files.sort(key=lambda path: os.path.getmtime(path))
            for path in files[:overflow]:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[TBG Debug] compareTiles pruning skipped: {e}")

    @classmethod
    def debug_image_to_folder(cls, image, filename):
        if image is None:
            return
        if getattr(getattr(tbg, "API", None), "status", None) != "Dev":
            return
        if not bool(getattr(getattr(tbg, "API", None), "dev_debug_enabled", True)):
            return
        try:
            filename = cls._structured_debug_filename(filename)
            cls.image_to_folder(image, filename)
        except Exception as e:
            print(f"[TBG Debug] Skipping debug save '{filename}': {e}")

    @classmethod
    def _structured_debug_filename(cls, filename):
        import re

        name = str(filename or "debug_image").strip().replace(" ", "_").replace("-", "_")
        exact = {
            "_DetailAware_Native_Base": "DetailAware_final_detail_preserving_stitch_InputNativeBase",
            "_DetailAware_Final_Stitch": "DetailAware_final_detail_preserving_stitch_OutputFinal",
            "_DetailAware_Detail_Contribution": "DetailAware_final_detail_preserving_stitch_DebugDetailContribution",
            "_DetailAware_Detail_Weight": "DetailAware_final_detail_preserving_stitch_DebugDetailWeight",
            "_DetailAware_Trusted_Mask": "DetailAware_final_detail_preserving_stitch_MaskTrusted",
            "_DetailAware_Detail_Energy": "DetailAware_final_detail_preserving_stitch_DebugDetailEnergy",
            "Final_ColorMatch_Base_Input": "FinalColorMatch_build_segment_color_base_InputUpscaledImage",
            "Final_ColorMatch_Base_With_Segments": "FinalColorMatch_build_segment_color_base_OutputSegmentAwareReference",
            "Final_ColorMatch_Base_SegmentMask": "FinalColorMatch_build_segment_color_base_MaskSegments",
        }
        if name in exact:
            return exact[name]

        patterns = [
            (r"^(\d+)Tile_before_Sampling$", r"\1_Sampler_prepare_sampling_InputTileBeforeSampling"),
            (r"^(\d+)Inpaint_Mask_before_Sampling$", r"\1_Sampler_prepare_sampling_InputInpaintMask"),
            (r"^(\d+)Complexity_Mask_before_Sampling$", r"\1_Sampler_prepare_sampling_InputComplexityMask"),
            (r"^(\d+)_Flux2Parity_Latent_Input_Decode$", r"\1_Flux2Parity_debug_sampler_input_latent_roundtrip_decode"),
            (r"^(\d+)_Flux2Parity_VAE_Roundtrip_Input_Decode$", r"\1_Flux2Parity_debug_sampler_input_latent_roundtrip_decode"),
            (r"^(\d+)_Flux2_PID_PostTone_before_(.+)$", r"\1_Flux2PIDPostTone_apply_flux2_pid_post_tone_InputBeforeCorrection_\2"),
            (r"^(\d+)_Flux2_PID_PostTone_after_(.+)$", r"\1_Flux2PIDPostTone_apply_flux2_pid_post_tone_OutputAfterCorrection_\2"),
            (r"^(\d+)_Flux2_PID_PostTone_reference_(.+)$", r"\1_Flux2PIDPostTone_apply_flux2_pid_post_tone_Reference_\2"),
            (r"^(\d+)_Flux2_PID_sampled_tile_before_correction$", r"\1_Flux2PID_apply_post_pid_flux2_correction_InputSampledTile"),
            (r"^(\d+)_Flux2_PID_sampled_tile_after_correction$", r"\1_Flux2PID_apply_post_pid_flux2_correction_OutputCorrectedTile"),
            (r"^(\d+)_tile_sampled_before_post_correction$", r"\1_PostSamplerCorrection_InputSampledTile"),
            (r"^(\d+)_VAE_decode_after_sampling$", r"\1_Sampler_decode_OutputVAEAfterSampling"),
            (r"^(\d+)_Raw_Sampler_VAE_Decode$", r"\1_Sampler_decode_DebugRawNormalVAE"),
            (r"^(\d+)_PID_after_sampler_before_post_context_4x(.*)$", r"\1_PID_run_pid_refiner_latent_decode_OutputAfterSamplerBeforePostContext4x\2"),
            (r"^(\d+)_PID_processed_region_4x(.*)$", r"\1_PID_run_pid_refiner_latent_decode_OutputProcessedRegion4x\2"),
            (r"^(\d+)_PID_processed_4x$", r"\1_PID_run_pid_refiner_latent_decode_OutputProcessedTile4x"),
            (r"^(\d+)_Segment_PID_sampler_source_reference_4x$", r"\1_SegmentPID_prepare_pid_decode_InputSourceReference4x"),
            (r"^(\d+)_Segment_PID_visible_mask_4x$", r"\1_SegmentPID_prepare_pid_decode_InputVisibleMask4x"),
            (r"^(\d+)_Segment_Flux_sampled_latent_normal_vae_debug_before_PID$", r"\1_SegmentPID_debug_flux_sampled_latent_normal_vae_decode_before_pid"),
            (r"^(\d+)_Segment_PID_PreProtect_(.+)$", r"\1_SegmentPID_apply_pid_pre_decode_hooks_\2"),
        ]
        for pattern, replacement in patterns:
            updated = re.sub(pattern, replacement, name)
            if updated != name:
                return updated
        return re.sub(r"[^A-Za-z0-9_.]+", "_", name).strip("_") or "debug_image"

    @classmethod
    def _color_match_enabled(cls):
        method = getattr(tbg.PARAMS, "color_match_method", "none")
        if method is None or str(method).lower() == "none":
            return False
        return bool(getattr(tbg.PARAMS, "Flux2_Tile_Color_Correction", True))

    @staticmethod
    def _is_rgb_luma_method(method):
        return str(method or "").strip().lower() in TBG_Refiner_v1.COLOR_STABILIZER_ALIASES

    @classmethod
    def _cm_debug_switches(cls):
        switches = getattr(tbg.PARAMS, "ColorMatch_Debug_Switches", None)
        return switches if isinstance(switches, dict) and switches.get("_connected") else None

    @classmethod
    def _cm_debug_override(cls):
        switches = cls._cm_debug_switches()
        return bool(switches and switches.get("Override_Normal_Gates", False))

    @classmethod
    def _cm_has_method(cls):
        method = getattr(tbg.PARAMS, "color_match_method", None)
        return (
            method is not None
            and str(method).lower() != "none"
        )

    @classmethod
    def _cm_debug_stage_enabled(cls, stage_key, normal_active=True, requires_method=True):
        switches = cls._cm_debug_switches()
        if not switches:
            return bool(normal_active)
        switch_enabled = bool(switches.get(stage_key, True))
        override = bool(switches.get("Override_Normal_Gates", False))
        if not bool(switches.get("Override_Normal_Gates", False)):
            result = bool(normal_active) and switch_enabled
        elif requires_method and not cls._cm_has_method():
            result = False
        else:
            result = switch_enabled
        if getattr(tbg.API, "status", None) == "Dev":
            reason = None
            if override and switch_enabled and result and not bool(normal_active):
                reason = "forced_on"
            elif bool(normal_active) and not result:
                reason = "skipped"
            elif override and switch_enabled and not result and requires_method:
                reason = "blocked_no_method"
            if reason:
                seen = getattr(cls, "_cm_debug_log_seen", set())
                key = (stage_key, reason, bool(normal_active), switch_enabled, override, result)
                if key not in seen:
                    seen.add(key)
                    cls._cm_debug_log_seen = seen
                    print(
                        f"TBG[Node {tbg.INFO.id}] ColorMatch debug gate {stage_key}: "
                        f"{reason} normal={bool(normal_active)} switch={switch_enabled} "
                        f"override={override} result={result}"
                    )
        return result

    @classmethod
    def _disable_worker_color_correction(cls, params):
        params.color_match_method = "none"
        params.color_match_str = 0
        params.Flux2_Tile_Color_Correction = False
        params.point_grid_image_stabilizer_experimental = 0
        return params

    @classmethod
    def _final_color_mode_label(cls):
        return "TBG RGB/Luma non-structural" if cls._final_mode_is_nonstructural() else "Protect New Generated Content"

    @classmethod
    def _final_mode_is_protect(cls):
        return not cls._final_mode_is_nonstructural()

    @classmethod
    def _final_mode_is_nonstructural(cls):
        return bool(getattr(tbg.PARAMS, "rgb_luma_nonstructural", False)) or cls._is_rgb_luma_method(
            getattr(tbg.PARAMS, "color_match_method", None)
        )

    @classmethod
    def _final_color_correction_enabled(cls):
        return cls._final_mode_is_nonstructural() or cls._cm_has_method()

    @classmethod
    def _apply_color_match_by_mode(cls, reference, target, method, strength, label=None):
        if cls._final_mode_is_nonstructural():
            corrected, metrics = cls._global_rgb_luma_match(
                reference,
                target,
                strength=strength,
                label=label or "final_nonstructural",
            )
            cls._log_global_rgb_luma_metrics(label or "final_nonstructural", metrics)
            return (corrected,)
        return (TBG_Image.detail_preserving_colormatch(
            reference,
            target,
            method,
            strength,
            label=label,
        )[0],)

    @classmethod
    def _apply_final_color_correction(cls, image):
        if (
            not cls._cm_debug_stage_enabled(
                "14_Final_Global_ColorMode",
                cls._final_color_correction_enabled(),
                requires_method=not cls._final_mode_is_nonstructural(),
            )
            or image is None
            or not torch.is_tensor(image)
        ):
            return image

        reference = getattr(tbg.OUTPUTS, "upscaled_image", None)
        if reference is None or not torch.is_tensor(reference):
            return image

        method = getattr(tbg.PARAMS, "color_match_method", "none")
        strength = float(getattr(tbg.PARAMS, "color_match_str", 1) or 1)
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        target = image.unsqueeze(0) if image.ndim == 3 else image

        if int(reference.shape[1]) != int(target.shape[1]) or int(reference.shape[2]) != int(target.shape[2]):
            reference = nodes.ImageScale().upscale(
                reference,
                "bilinear",
                int(target.shape[2]),
                int(target.shape[1]),
                False,
            )[0]

        try:
            corrected = cls._apply_color_match_by_mode(
                reference,
                target,
                method,
                strength,
                label=f"final_color_correction_{getattr(tbg.KSAMPLER, 'model_type', None)}",
            )[0]
            print(
                f"TBG[Node {tbg.INFO.id}] final color correction applied with '{method}' "
                f"mode='{cls._final_color_mode_label()}' to {getattr(tbg.KSAMPLER, 'model_type', None)}"
            )
            return corrected
        except Exception as exc:
            print(f"TBG[Node {tbg.INFO.id}] final color correction failed with '{method}', using uncorrected output: {exc}")
            return image

    @classmethod
    def _detail_aware_enabled(cls):
        if getattr(tbg.PARAMS, "Tile_Fusion_Mode", None) not in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion"):
            return False
        if getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False):
            return False
        if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
            return False
        # Flux2 stays on the worker NGTF composite until this mask-aware detail path
        # is verified against real Flux2 tiled-slow runs.
        if getattr(tbg.KSAMPLER, "model_type", None) == "FLUX2":
            return False
        segms_new = getattr(tbg.SEGMENTS, "segms_new", None)
        if segms_new is not None:
            try:
                _, segms = segms_new
                if segms:
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def _ensure_bhwc_image(image):
        if image is None or not torch.is_tensor(image):
            return None
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim == 4:
            return image
        return None

    @staticmethod
    def _gaussian_kernel1d(radius, sigma, device, dtype):
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
        return kernel / kernel.sum().clamp_min(1e-8)

    @classmethod
    def _blur_bhwc(cls, image, sigma=1.8):
        if sigma <= 0:
            return image
        max_radius = max(0, min(int(image.shape[1]), int(image.shape[2])) - 1)
        radius = min(max_radius, max(1, int(round(sigma * 3))))
        if radius <= 0:
            return image
        kernel = cls._gaussian_kernel1d(radius, sigma, image.device, image.dtype)
        channels = int(image.shape[-1])
        bchw = image.permute(0, 3, 1, 2)
        pad_x = (radius, radius, 0, 0)
        pad_y = (0, 0, radius, radius)
        kx = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        ky = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        bchw = torch.nn.functional.conv2d(torch.nn.functional.pad(bchw, pad_x, mode="reflect"), kx, groups=channels)
        bchw = torch.nn.functional.conv2d(torch.nn.functional.pad(bchw, pad_y, mode="reflect"), ky, groups=channels)
        return bchw.permute(0, 2, 3, 1)

    @classmethod
    def _tile_stitch_feather_mask(cls, tile_w, tile_h, x, y, width, height, feather, device, dtype):
        feather = int(max(0, min(feather, tile_w // 2, tile_h // 2)))
        mask = torch.ones((1, tile_h, tile_w, 1), device=device, dtype=dtype)
        if feather <= 0:
            return mask
        ramp = torch.linspace(0.0, 1.0, feather + 2, device=device, dtype=dtype)[1:-1]
        if x > 0:
            mask[:, :, :feather, :] *= ramp.view(1, 1, feather, 1)
        if y > 0:
            mask[:, :feather, :, :] *= ramp.view(1, feather, 1, 1)
        if x + tile_w < width:
            mask[:, :, tile_w - feather:tile_w, :] *= ramp.flip(0).view(1, 1, feather, 1)
        if y + tile_h < height:
            mask[:, tile_h - feather:tile_h, :, :] *= ramp.flip(0).view(1, feather, 1, 1)
        return mask.clamp_min(1e-4)

    @staticmethod
    def _detail_aware_specs_overlap(spec, specs):
        _, _, _, x, y, tile_w, tile_h = spec[:7]
        x, y, tile_w, tile_h = int(x), int(y), int(tile_w), int(tile_h)
        overlaps = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        for other in specs:
            if other is spec:
                continue
            try:
                _, _, _, ox, oy, ow, oh = other[:7]
                ox, oy, ow, oh = int(ox), int(oy), int(ow), int(oh)
            except Exception:
                continue

            vertical_overlap = min(y + tile_h, oy + oh) - max(y, oy)
            horizontal_overlap = min(x + tile_w, ox + ow) - max(x, ox)
            if vertical_overlap > 0:
                if ox < x:
                    overlaps["left"] = max(overlaps["left"], ox + ow - x)
                elif ox > x:
                    overlaps["right"] = max(overlaps["right"], x + tile_w - ox)
            if horizontal_overlap > 0:
                if oy < y:
                    overlaps["top"] = max(overlaps["top"], oy + oh - y)
                elif oy > y:
                    overlaps["bottom"] = max(overlaps["bottom"], y + tile_h - oy)
        return {side: max(0, int(value)) for side, value in overlaps.items()}

    @staticmethod
    def _detail_aware_ramp(length, ascending, device, dtype):
        if length <= 0:
            return None
        ramp = torch.linspace(0.0, 1.0, int(length) + 2, device=device, dtype=dtype)[1:-1]
        return ramp if ascending else ramp.flip(0)

    @classmethod
    def _detail_aware_trusted_mask(cls, crop_w, crop_h, overlaps, crop_margin, feather, device, dtype):
        mask = torch.ones((1, crop_h, crop_w, 1), device=device, dtype=dtype)
        crop_margin = max(0, int(crop_margin))
        feather = max(0, int(feather))

        def apply_start_x(overlap):
            if overlap <= 0:
                return
            cut = min(crop_w, crop_margin)
            if cut > 0:
                mask[:, :, :cut, :] = 0.0
            ramp_len = min(feather, max(0, crop_w - cut))
            ramp = cls._detail_aware_ramp(ramp_len, True, device, dtype)
            if ramp is not None:
                mask[:, :, cut:cut + ramp_len, :] *= ramp.view(1, 1, ramp_len, 1)

        def apply_end_x(overlap):
            if overlap <= 0:
                return
            seam = max(0, min(crop_w, crop_w - int(overlap) + crop_margin))
            if seam < crop_w:
                mask[:, :, seam:, :] = 0.0
            ramp_len = min(feather, seam)
            ramp = cls._detail_aware_ramp(ramp_len, False, device, dtype)
            if ramp is not None:
                mask[:, :, seam - ramp_len:seam, :] *= ramp.view(1, 1, ramp_len, 1)

        def apply_start_y(overlap):
            if overlap <= 0:
                return
            cut = min(crop_h, crop_margin)
            if cut > 0:
                mask[:, :cut, :, :] = 0.0
            ramp_len = min(feather, max(0, crop_h - cut))
            ramp = cls._detail_aware_ramp(ramp_len, True, device, dtype)
            if ramp is not None:
                mask[:, cut:cut + ramp_len, :, :] *= ramp.view(1, ramp_len, 1, 1)

        def apply_end_y(overlap):
            if overlap <= 0:
                return
            seam = max(0, min(crop_h, crop_h - int(overlap) + crop_margin))
            if seam < crop_h:
                mask[:, seam:, :, :] = 0.0
            ramp_len = min(feather, seam)
            ramp = cls._detail_aware_ramp(ramp_len, False, device, dtype)
            if ramp is not None:
                mask[:, seam - ramp_len:seam, :, :] *= ramp.view(1, ramp_len, 1, 1)

        apply_start_x(overlaps.get("left", 0))
        apply_end_x(overlaps.get("right", 0))
        apply_start_y(overlaps.get("top", 0))
        apply_end_y(overlaps.get("bottom", 0))
        return mask.clamp(0.0, 1.0)

    @classmethod
    def _detail_aware_final_stitch(cls, native_image):
        if not cls._detail_aware_enabled():
            return None
        reference = cls._ensure_bhwc_image(native_image)
        if reference is None:
            return None
        tiles = list(getattr(tbg.OUTPUTS, "grid_images_all", None) or [])
        specs = list(getattr(tbg.PARAMS, "grid_specs", None) or [])
        if not tiles or not specs or len(tiles) < len(specs):
            return None

        height = int(reference.shape[1])
        width = int(reference.shape[2])
        channels = int(reference.shape[-1])
        device = reference.device
        dtype = reference.dtype
        detail_accum = torch.zeros((1, height, width, channels), device=device, dtype=dtype)
        detail_weight = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        energy_accum = torch.zeros_like(detail_weight)
        trusted_accum = torch.zeros_like(detail_weight)

        feather = int(getattr(tbg.SIZE, "composite_blur_margin", 0) or 0)
        crop_margin = int(
            getattr(
                tbg.SIZE,
                "crop_margin",
                getattr(tbg.SIZE, "inpaint_border_margin", 0),
            )
            or 0
        )
        sigma = 1.8
        exponent = 2.0
        placed = 0
        for tile, spec in zip(tiles, specs):
            tile = cls._ensure_bhwc_image(tile)
            if tile is None:
                return None
            _, _, _, x, y, tile_w, tile_h = spec[:7]
            x, y, tile_w, tile_h = int(x), int(y), int(tile_w), int(tile_h)
            if x >= width or y >= height or x + tile_w <= 0 or y + tile_h <= 0:
                continue

            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + tile_w)
            y1 = min(height, y + tile_h)
            sx0 = x0 - x
            sy0 = y0 - y
            sx1 = min(int(tile.shape[2]), sx0 + (x1 - x0))
            sy1 = min(int(tile.shape[1]), sy0 + (y1 - y0))
            x1 = x0 + (sx1 - sx0)
            y1 = y0 + (sy1 - sy0)
            crop = tile[:, sy0:sy1, sx0:sx1, :].to(device=device, dtype=dtype)
            crop_h = int(crop.shape[1])
            crop_w = int(crop.shape[2])
            if crop_h <= 1 or crop_w <= 1:
                continue

            overlaps = cls._detail_aware_specs_overlap(spec, specs)
            trusted_mask = cls._detail_aware_trusted_mask(crop_w, crop_h, overlaps, crop_margin, feather, device, dtype)
            if float(trusted_mask.max().detach().cpu()) <= 0:
                continue
            low = cls._blur_bhwc(crop, sigma=sigma)
            detail = crop - low
            energy = detail.abs().mean(dim=-1, keepdim=True)
            energy = cls._blur_bhwc(energy, sigma=1.2).clamp_min(1e-6)
            detail_confidence = (energy ** exponent) * trusted_mask

            detail_accum[:, y0:y1, x0:x1, :] += detail * detail_confidence
            detail_weight[:, y0:y1, x0:x1, :] += detail_confidence
            energy_accum[:, y0:y1, x0:x1, :] += energy * trusted_mask
            trusted_accum[:, y0:y1, x0:x1, :] += trusted_mask
            placed += 1

        if placed == 0:
            return None

        detail_blend = detail_accum / detail_weight.clamp_min(1e-6)
        final = (reference.to(device=device, dtype=dtype) + detail_blend).clamp(0.0, 1.0)

        if tbg.API.status == "Dev":
            detail_weight_preview = (detail_weight / detail_weight.max().clamp_min(1e-6)).expand(-1, -1, -1, 3).clamp(0.0, 1.0)
            energy_preview = (energy_accum / trusted_accum.clamp_min(1e-6))
            energy_preview = (energy_preview / energy_preview.max().clamp_min(1e-6)).expand(-1, -1, -1, 3).clamp(0.0, 1.0)
            trusted_preview = (trusted_accum / trusted_accum.max().clamp_min(1e-6)).expand(-1, -1, -1, 3).clamp(0.0, 1.0)
            detail_preview = ((detail_blend * 4.0) + 0.5).clamp(0.0, 1.0)
            cls.debug_image_to_folder(reference.clamp(0.0, 1.0), "_DetailAware_Native_Base")
            cls.debug_image_to_folder(final, "_DetailAware_Final_Stitch")
            cls.debug_image_to_folder(detail_preview, "_DetailAware_Detail_Contribution")
            cls.debug_image_to_folder(detail_weight_preview, "_DetailAware_Detail_Weight")
            cls.debug_image_to_folder(trusted_preview, "_DetailAware_Trusted_Mask")
            cls.debug_image_to_folder(energy_preview, "_DetailAware_Detail_Energy")

        print(f"TBG[Node {tbg.INFO.id}] detail-aware NGTF final stitch enhanced {placed} tiles with native compositor base.")
        return final

    @classmethod
    def conditioning(cls, image, index, tile_to_process):
        neg_low = None
        pos_low = None
        negative = None
        positive = None
        tbg.TEMP.latent_index = index  # used in Cnet.py
        # Flux conditioning + Guidance + cnet if Kontext
        if tbg.KSAMPLER.model_type in {"FLUX2", "FLUX1", "FLUX1 Kontext"}:
            # FLUX KONTEXT conditioning
            if tbg.KSAMPLER.model_type in {"FLUX2", "FLUX1 Kontext"}:

                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)

                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")

                reference_image_active = cls._has_reference_image_pipe(tbg.KSAMPLER.Controlnet_Pipe)
                negative_was_zeroed = False
                if not reference_image_active:
                    negative = nodes.ConditioningZeroOut.zero_out(0, positive)[0]
                    negative_was_zeroed = True
                if tbg.KSAMPLER.Controlnet_Pipe:
                    # build from Cnet stitched and chaind referent latent combination
                    positive, negative = get_Kontext_stiched_o_chained_cond(tbg, positive, negative, tbg.KSAMPLER.Controlnet_Pipe, tile_to_process, index)
                cls._debug_conditioning_trace(
                    index,
                    "flux_reference" if reference_image_active else "flux_zeroed",
                    positive,
                    negative,
                    negative_was_zeroed,
                )
            else:
                # FLUX standard conditioning (no Kontext)

                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")
                negative = nodes.ConditioningZeroOut.zero_out(0, positive)[0]
                cls._debug_conditioning_trace(index, "flux_standard_zeroed", positive, negative, True)

            positive = FluxGuidance_execute(positive, tbg.KSAMPLER.Flux_Guidance)[0]

        # Qwen Edit conditioning
        elif tbg.KSAMPLER.model_type in {"Qwen Image", "Qwen Image Edit"}:
            if tbg.KSAMPLER.model_type == "Qwen Image Edit":
                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")
                if tbg.KSAMPLER.Controlnet_Pipe:
                    # build from Cnet stitched and chaind referent latent combination
                    positive, negative = get_qwen_stiched_o_chained_cond(tbg, positive, negative, tbg.KSAMPLER.Controlnet_Pipe, tile_to_process, index)
                cls._debug_conditioning_trace(index, "qwen_reference", positive, negative, False)
            else:

                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")

        # SDXL and Other conditioning
        else:
            positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
            if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")

        # ------------------------------------------------------------------
        # PRO Step 3.5.4 Conditioning cropped_positive cropped_negative
        # ------------------------------------------------------------------

        # Crop condition, conditioning's from input and not in tile size, add them to tile conditioning
        if tbg.KSAMPLER.cropped_positive or tbg.KSAMPLER.cropped_negative:
            _, _, _, x, y, width, height = tbg.PARAMS.grid_specs[index]
            crop_region = (x, y, x + width, y + height)
            tile_size = (width, height)
            if cls._is_segment_index(index):
                try:
                    index_seg = int(index) - int(getattr(tbg.PARAMS, "len_grid_images", 0) or 0)
                    transforms = getattr(tbg.SEGMENTS, "segment_sampling_transforms", None)
                    if isinstance(transforms, (list, tuple)) and 0 <= index_seg < len(transforms) and isinstance(transforms[index_seg], dict):
                        transform = transforms[index_seg]
                        crop_region = tuple(int(round(float(v))) for v in transform.get("sampling_crop_region", crop_region))
                        tile_size = tuple(int(round(float(v))) for v in transform.get("sampling_tile_size", tile_size))
                    elif torch.is_tensor(tile_to_process):
                        tile_size = (int(tile_to_process.shape[2]), int(tile_to_process.shape[1]))
                except Exception:
                    if torch.is_tensor(tile_to_process):
                        tile_size = (int(tile_to_process.shape[2]), int(tile_to_process.shape[1]))
            canvas_size = (image.shape[2], image.shape[1])
            init_size = (tbg.OUTPUTS.upscaled_image.shape[2], tbg.OUTPUTS.upscaled_image.shape[1])

            if tbg.KSAMPLER.cropped_positive:
                positive_crop = crop_cond(tbg.KSAMPLER.cropped_positive, crop_region, init_size, canvas_size, tile_size)
                positive = combine_conditioning([positive, positive_crop])

            if tbg.KSAMPLER.cropped_negative:
                negative_crop = crop_cond(tbg.KSAMPLER.cropped_negative, crop_region, init_size, canvas_size, tile_size)
                negative = combine_conditioning([negative, negative_crop])

        return (positive, negative, pos_low, neg_low)

    @classmethod
    def _has_reference_image_pipe(cls, cnetpipe):
        for control in cnetpipe or []:
            if normalize_controlnet_mode(control) in {"Reference_Image", "Input_Tile_CFG_Hook", "Hildegard_RefImg", "Hildegard_RefImg_CFG_Hook"}:
                return True
        return False

    @classmethod
    def _conditioning_ref_latent_count(cls, conditioning):
        count = 0
        if not isinstance(conditioning, (list, tuple)):
            return count
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            meta = entry[1]
            if isinstance(meta, dict):
                refs = meta.get("reference_latents")
                if isinstance(refs, (list, tuple)):
                    count += len(refs)
        return count

    @classmethod
    def _debug_conditioning_trace(cls, index, path, positive, negative, negative_was_zeroed):
        if getattr(getattr(tbg, "API", None), "status", None) != "Dev":
            return
        pos_entries = len(positive) if isinstance(positive, (list, tuple)) else 0
        neg_entries = len(negative) if isinstance(negative, (list, tuple)) else 0
        print(
            f"[TBG Conditioning] tile={index + 1} path={path} "
            f"negative_zeroed={negative_was_zeroed} "
            f"pos_entries={pos_entries} neg_entries={neg_entries} "
            f"pos_reference_latents={cls._conditioning_ref_latent_count(positive)} "
            f"neg_reference_latents={cls._conditioning_ref_latent_count(negative)}"
        )

    @classmethod
    def sigmas(cls, iteration, index):
            # PRO Step 3.5.1 Sigmas
            if tbg.KSAMPLER.custom_sigmas is not None:
                # use custom sigmas from input node should have denoise 1
                sigmas = tbg.KSAMPLER.custom_sigmas #* tbg.PROMPTER.output_denoises[index]
                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None,
                    f"Node {tbg.INFO.id} - Custom Sigmas Loaded {iteration}")
                cls._log_sigma_trace(
                    "pre_denoise",
                    sigmas,
                    tile=index + 1,
                    iteration=iteration,
                    denoise_method=tbg.PARAMS.denoise_method,
                    custom_sigmas=True,
                )
            else:
                # create full sigmas denoise 1
                sigmas = get_sigmas(tbg.KSAMPLER.model, tbg.KSAMPLER.scheduler, tbg.KSAMPLER.steps, 1.0 , tbg.PARAMS.denoise_method)[0]


            # ------------------------------------------------------------------
            # PRO Step 3.5.2 FLUX_Kontext Sigma corrections
            if tbg.KSAMPLER.model_type in {"FLUX1 Kontext",}:
                # Consistent Position Sigma
                if tbg.KSAMPLER.steps > 6:
                    sigma_a = torch.tensor([1.0000, 0.9910, 0.9753, 0.9547, 0.9295, 0.8994, 0.8643, 0.8236, 0.7770,
                                            0.7238, 0.6636, 0.5965, 0.5223, 0.4419, 0.3571, 0.2711, 0.1877, 0.1130,
                                            0.0527, 0.0138, 0.0000])
                    # Get first 6 steps from sigma A
                    head = sigma_a[:6]
                    # Threshold is the 6th value (index 4)
                    threshold = sigma_a[5].item()
                    # Filter sigma B to keep only values less than threshold
                    filtered_b = sigmas[sigmas < threshold]
                    # Combine the result
                    sigmas = torch.cat([head, filtered_b], dim=0)

            # ------------------------------------------------------------------
            # PRO Step 3.5.3 Denoise Correction


            if tbg.PROMPTER.output_denoises[index] != tbg.KSAMPLER.denoise and tbg.PROMPTER.output_denoises[index] != "":
                denoise = tbg.PROMPTER.output_denoises[index]

                sigmas = denoise_sigmas_tgb(sigmas, tbg.PROMPTER.output_denoises[index], tbg.PARAMS.denoise_method, tbg.KSAMPLER.model, tbg.KSAMPLER.scheduler)
                denoise_override = True
            else:
                denoise = tbg.KSAMPLER.denoise
                sigmas = denoise_sigmas_tgb(sigmas, tbg.KSAMPLER.denoise, tbg.PARAMS.denoise_method, tbg.KSAMPLER.model, tbg.KSAMPLER.scheduler)
                denoise_override = False

            cls._log_sigma_trace(
                "post_denoise",
                sigmas,
                tile=index + 1,
                iteration=iteration,
                denoise=round(float(denoise), 6) if denoise is not None else None,
                denoise_method=tbg.PARAMS.denoise_method,
                tile_override=denoise_override,
            )

            return  (denoise, sigmas)
    @classmethod
    def refine(cls, image, iteration):

        inpaintmask = None
        fusion_segment_tiles = []
        tbg.PARAMS.SegFusion_Initializer_run_once = False  # has to be false on run so that the first called seg triggers learning surroundings for all segs
        tbg.full_image_only_tiles = None
        temp_output_images = None
        output_image_new = None
        if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
            pid_tile_count = len(getattr(tbg.OUTPUTS, "grid_images_all", []))
            tbg.TEMP.pid_grid_images_4x = [None] * pid_tile_count
            tbg.TEMP.pid_grid_images_4x_source = [None] * pid_tile_count
            tbg.TEMP.pid_refiner_runtime = None
            tbg.TEMP.pid_refiner_runtime_key = None
        else:
            tbg.TEMP.pid_grid_images_4x = None
            tbg.TEMP.pid_grid_images_4x_source = None
            tbg.TEMP.pid_refiner_runtime = None
            tbg.TEMP.pid_refiner_runtime_key = None

        # ------------------------------------------------------------------
        # Step 1 prepare tile arrays
        # ------------------------------------------------------------------

        cls.prepare_tiles_to_process()  # return tbg.OUTPUTS.grid_images_all and tbg.PARAMS.tiles_to_process

        # Detect what changed
        # change_type, changed_indices, change_msg = cls._detect_changes()
        import traceback

        fast_preview = bool(getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False))
        if fast_preview:
            change_type = "ALL"
            changed_indices = set()
            change_msg = "Fast 1 Tile Preview: isolated preview run; persistent cache state unchanged"
            tbg.PARAMS.force_fresh_refiner_background = False
            log(change_msg, None, None, f"Node {tbg.INFO.id}")
        else:
            try:
                change_type, changed_indices, change_msg = cls._detect_changes()
                log(f"{change_msg}", None, None, f"Node {tbg.INFO.id}")


            except Exception as e:

                change_type = "ALL"
                changed_indices = set()
                change_msg = "ALL"
                tbg.PARAMS.force_fresh_refiner_background = True
                err = traceback.format_exc()

                #log(f"_detect_changes failed, reset to ALL\n{err}", None, None, f"Node {tbg.INFO.id}")

        tbg.TEMP.change_type = change_type
        tbg.TEMP.changed_indices = changed_indices
        tbg.TEMP.change_msg = change_msg

        cls._apply_fast_preview_prompt_target(changed_indices, change_type)

        # ------------------------------------------------------------------
        # Step 2 VRAM Optimization: precompute embeddings after preview target selection.
        # ------------------------------------------------------------------

        cls.precompute_all_embeddings_free_VRAM()
        # ------------------------------------------------------------------
        # Step 3 TBG Magic tile Loop
        # ------------------------------------------------------------------

        # Pass  storage["generated_tiles"] to OUTPUTS so its will be sent with all other images to the WORKER
        storage = persistent_storage[tbg.storage_key]
        if fast_preview:
            tbg.OUTPUTS.persistent_generated_tiles = {}
        elif getattr(tbg.PARAMS, "force_fresh_refiner_background", False):
            try:
                storage["generated_tiles"].clear()
            except Exception:
                storage["generated_tiles"] = {}
            tbg.OUTPUTS.persistent_generated_tiles = {}
            tbg.OUTPUTS.last_final_image = None
            log(
                "[TBG Cache] selected/preview run allowed to reuse cache=False reason=tiler_context_changed",
                None,
                None,
                f"Node {tbg.INFO.id}",
            )
        else:
            tbg.OUTPUTS.persistent_generated_tiles = storage["generated_tiles"]

        worker_params = SimpleNamespace(**vars(tbg.PARAMS))
        worker_params.Redux_Style_Model = None
        worker_params.Redux_Clip_Vision = None
        worker_params.output_color_match_js = list(getattr(tbg.PROMPTER, "output_color_match_js", []) or [])
        worker_params.output_ignore_general_prompt_js = list(getattr(tbg.PROMPTER, "output_ignore_general_prompt_js", []) or [])
        worker_params.output_prompts = list(getattr(tbg.PROMPTER, "output_prompts", []) or [])
        output_image_new, output_image_only_tiles, output_image_noCC = WORKER.id(tiler_id).ETUR.refiner_init(worker_params, tbg.SIZE)
        # output_image_new, output_image_only_tiles, output_image_noCC = WORKER.id(tiler_id).ETUR.refiner_init(tbg.PARAMS, tbg.SIZE)

        if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
            if getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False):
                pid_tiles = list(getattr(tbg.TEMP, "pid_grid_images_4x", None) or [])
                idx = 0
                if getattr(tbg.PARAMS, "tiles_to_process", None):
                    try:
                        idx = int(tbg.PARAMS.tiles_to_process[0])
                    except Exception:
                        idx = 0
                if idx < 0 or idx >= len(pid_tiles) or pid_tiles[idx] is None:
                    print(
                        f"TBG[Node {tbg.INFO.id}] PID Fast 1 Tile Preview: "
                        f"missing 4x preview tile at index {idx}; keeping worker preview output."
                    )
                else:
                    output_image_new = pid_tiles[idx].clone()
                    output_image_only_tiles = output_image_new.clone()
                    output_image_noCC = output_image_new.clone()
                    print(
                        f"TBG[Node {tbg.INFO.id}] PID Fast 1 Tile Preview: "
                        f"using selected 4x tile {idx + 1}; skipped full-canvas PiD rebuild and segment masks."
                    )
            else:
                output_image_new, output_image_only_tiles, output_image_noCC = cls.rebuild_pid_refiner_output_4x(
                    output_image_new,
                    output_image_only_tiles,
                    output_image_noCC,
                )
                output_image_new = cls._apply_pid_segment_final_color_match(output_image_new)
                if tbg.API.status == "Dev" and torch.is_tensor(output_image_new):
                    cls.debug_image_to_folder(output_image_new, "PID_FinalReturn_Output0_Final4x")
                if (
                    not getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False)
                    and output_image_only_tiles is not None
                    and torch.is_tensor(output_image_only_tiles)
                    and getattr(tbg.PARAMS, "color_match_method", "none") is not None
                    and str(getattr(tbg.PARAMS, "color_match_method", "none")).lower() != "none"
                    and cls._cm_debug_stage_enabled("10_Final_TileOnly_ColorCorrection", True)
                ):
                    try:
                        only_tiles_reference = nodes.ImageScale().upscale(
                            tbg.OUTPUTS.upscaled_image,
                            "bilinear",
                            int(output_image_only_tiles.shape[2]),
                            int(output_image_only_tiles.shape[1]),
                            False,
                        )[0]
                        before_only_tiles = output_image_only_tiles
                        output_image_only_tiles, global_metrics = cls._global_rgb_luma_match(
                            only_tiles_reference,
                            output_image_only_tiles,
                            strength=tbg.PARAMS.color_match_str,
                            label="pid_final_without_segments_only_tiles",
                        )
                        if global_metrics is None:
                            output_image_only_tiles = before_only_tiles
                        if getattr(tbg.API, "status", None) == "Dev":
                            cls._log_global_rgb_luma_metrics("pid_final_without_segments_only_tiles", global_metrics)
                            delta = torch.mean(torch.abs(output_image_only_tiles.to(torch.float32) - before_only_tiles.to(torch.float32))).item()
                            print(
                                f"TBG[Node {tbg.INFO.id}] PiD final without-segments color match uses input/upscaled image: "
                                f"method=global_rgb_luma strength={tbg.PARAMS.color_match_str} "
                                f"mean_abs_delta={delta:.8f}"
                            )
                    except Exception as exc:
                        print(f"TBG[Node {tbg.INFO.id}] PiD final without-segments color match failed: {exc}")
            cls._pid_release_runtime("refiner complete")
        else:
            try:
                detail_aware_output = cls._detail_aware_final_stitch(output_image_new)
                if detail_aware_output is not None:
                    output_image_new = detail_aware_output
                    output_image_only_tiles = detail_aware_output.clone()
            except Exception as exc:
                print(f"TBG[Node {tbg.INFO.id}] detail-aware NGTF final stitch failed, keeping native compositor output: {exc}")

        if output_image_noCC is None:
            output_image_noCC = output_image_new.clone() if torch.is_tensor(output_image_new) else output_image_new

        has_segments = bool(getattr(tbg.PARAMS, "len_segments", 0) or getattr(tbg.SEGMENTS, "segms_crop_regions", None))
        
        # Always update cache (needed for incremental processing)
        # tbg.OUTPUTS.generated_tiles has no infos at this point
        # storage = persistent_storage[tbg.storage_key]
        #storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.generated_tiles)

        if not fast_preview:
            if output_image_noCC is None and torch.is_tensor(output_image_new):
                output_image_noCC = output_image_new.clone()
            if has_segments and torch.is_tensor(output_image_new):
                tbg.OUTPUTS.last_final_image = output_image_new.clone()
            elif torch.is_tensor(output_image_only_tiles):
                tbg.OUTPUTS.last_final_image = output_image_only_tiles.clone()
        return output_image_new, output_image_only_tiles, output_image_noCC

    @classmethod
    def _scale_pid_mask(cls, mask, width, height):
        if mask is None:
            return None
        layout = None
        if mask.ndim == 2:
            source = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            source = mask.unsqueeze(1)
        elif mask.ndim == 4:
            if int(mask.shape[1]) == 1:
                source = mask
                layout = "bchw"
            elif int(mask.shape[-1]) == 1:
                source = mask.permute(0, 3, 1, 2)
                layout = "bhwc"
            else:
                source = mask[:, :1]
                layout = "bchw"
        else:
            return mask
        source = source.to(torch.float32)
        scaled = torch.nn.functional.interpolate(source, size=(height, width), mode="bilinear", align_corners=False)
        if mask.ndim == 2:
            return scaled[0, 0]
        if mask.ndim == 3:
            return scaled[:, 0]
        if layout == "bhwc":
            return scaled.permute(0, 2, 3, 1)
        return scaled

    @classmethod
    def _mask_height_width(cls, mask):
        if mask is None or not torch.is_tensor(mask):
            return None
        if mask.ndim == 2:
            return int(mask.shape[0]), int(mask.shape[1])
        if mask.ndim == 3:
            if int(mask.shape[-1]) == 1 and int(mask.shape[0]) != 1:
                return int(mask.shape[0]), int(mask.shape[1])
            return int(mask.shape[-2]), int(mask.shape[-1])
        if mask.ndim == 4:
            if int(mask.shape[1]) == 1:
                return int(mask.shape[2]), int(mask.shape[3])
            if int(mask.shape[-1]) == 1:
                return int(mask.shape[1]), int(mask.shape[2])
            return int(mask.shape[2]), int(mask.shape[3])
        return None

    @staticmethod
    def _is_segment_spec(spec):
        try:
            return int(spec[2]) >= 8000
        except Exception:
            return False

    @classmethod
    def _is_segment_index(cls, index):
        try:
            specs = getattr(tbg.PARAMS, "grid_specs", None) or []
            return 0 <= int(index) < len(specs) and cls._is_segment_spec(specs[int(index)])
        except Exception:
            return False

    @staticmethod
    def _mask_to_bhw(mask):
        if mask is None or not torch.is_tensor(mask):
            return None
        if mask.ndim == 2:
            return mask.unsqueeze(0)
        if mask.ndim == 3:
            return mask
        if mask.ndim == 4:
            if int(mask.shape[1]) == 1:
                return mask[:, 0]
            return mask[..., 0]
        return None

    @classmethod
    def _segment_mask(cls, index_seg, kind, width=None, height=None):
        if index_seg is None or int(index_seg) < 0:
            return None
        sources = []
        if kind == "inpaint":
            sources = ("inpainting_mask", "segms_cropped_masks", "compositing_mask")
        elif kind == "composite":
            sources = ("compositing_mask", "segms_cropped_masks", "inpainting_mask")
        elif kind == "binary":
            sources = ("segment_binary_masks", "segms_cropped_masks", "inpainting_mask")
        else:
            sources = ("segms_cropped_masks", "compositing_mask", "inpainting_mask")

        for name in sources:
            values = getattr(tbg.SEGMENTS, name, None)
            if not isinstance(values, (list, tuple)) or int(index_seg) >= len(values):
                continue
            mask = values[int(index_seg)]
            if not torch.is_tensor(mask):
                continue
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                continue
            mask = mask.to(torch.float32)
            if mask.numel() > 0 and float(mask.max().detach().cpu()) > 1.0:
                mask = mask / 255.0
            mask = mask.clamp(0.0, 1.0)
            transform = cls._segment_sampling_transform(index_seg)
            if transform is not None and width is not None and height is not None:
                tile_size = transform.get("sampling_tile_size")
                if tile_size:
                    transformed = cls._transform_segment_mask_to_sampling(mask, transform)
                    if transformed is not None:
                        tile_w = int(round(float(tile_size[0])))
                        tile_h = int(round(float(tile_size[1])))
                        if int(width) != tile_w or int(height) != tile_h:
                            transformed = cls._scale_pid_mask(transformed, int(width), int(height))
                            transformed = cls._mask_to_bhw(transformed)
                            if transformed is None:
                                continue
                        return transformed.clamp(0.0, 1.0)
            if width is not None and height is not None:
                mask = cls._scale_pid_mask(mask, int(width), int(height))
                mask = cls._mask_to_bhw(mask)
            return mask.clamp(0.0, 1.0)
        return None

    @classmethod
    def _segment_native_crop_mask(cls, index_seg, width, height):
        if index_seg is None or int(index_seg) < 0 or width is None or height is None:
            return None, None, 0.0
        for name in ("compositing_mask", "segms_cropped_masks", "inpainting_mask"):
            values = getattr(tbg.SEGMENTS, name, None)
            if not isinstance(values, (list, tuple)) or int(index_seg) >= len(values):
                continue
            mask = values[int(index_seg)]
            if not torch.is_tensor(mask):
                continue
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                continue
            mask = mask.to(torch.float32)
            if mask.numel() > 0 and float(mask.max().detach().cpu()) > 1.0:
                mask = mask / 255.0
            mask = mask.clamp(0.0, 1.0)
            mask = cls._scale_pid_mask(mask, int(width), int(height))
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                continue
            mask = mask.to(torch.float32).clamp(0.0, 1.0)
            if mask.numel() <= 0:
                continue
            max_value = float(mask.max().detach().cpu())
            coverage = float((mask > 0.001).to(torch.float32).mean().detach().cpu())
            if max_value <= 0.001 or coverage <= 1e-6:
                continue
            return mask, name, coverage
        return None, None, 0.0

    @classmethod
    def _segment_native_crop_mask_from_sources(cls, index_seg, width, height, sources):
        if index_seg is None or int(index_seg) < 0 or width is None or height is None:
            return None, None, 0.0
        for name in tuple(sources or ()):
            values = getattr(tbg.SEGMENTS, name, None)
            if not isinstance(values, (list, tuple)) or int(index_seg) >= len(values):
                continue
            mask = values[int(index_seg)]
            if not torch.is_tensor(mask):
                continue
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                continue
            mask = mask.to(torch.float32)
            if mask.numel() > 0 and float(mask.max().detach().cpu()) > 1.0:
                mask = mask / 255.0
            mask = mask.clamp(0.0, 1.0)
            mask = cls._scale_pid_mask(mask, int(width), int(height))
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                continue
            mask = mask.to(torch.float32).clamp(0.0, 1.0)
            if mask.numel() <= 0:
                continue
            max_value = float(mask.max().detach().cpu())
            coverage = float((mask > 0.001).to(torch.float32).mean().detach().cpu())
            if max_value <= 0.001 or coverage <= 1.0e-6:
                continue
            return mask, name, coverage
        return None, None, 0.0

    @classmethod
    def _segment_sampling_transform(cls, index_seg):
        transforms = getattr(tbg.SEGMENTS, "segment_sampling_transforms", None)
        if not isinstance(transforms, (list, tuple)) or index_seg is None or int(index_seg) < 0 or int(index_seg) >= len(transforms):
            return None
        transform = transforms[int(index_seg)]
        return transform if isinstance(transform, dict) else None

    @classmethod
    def _restore_segment_pid_4x_to_native_crop(cls, tile_processed_4x, index_seg):
        transform = cls._segment_sampling_transform(index_seg)
        tile_processed_4x = cls._ensure_bhwc_image(tile_processed_4x)
        if transform is None or tile_processed_4x is None:
            return tile_processed_4x

        native_crop = transform.get("native_crop_region")
        native_bbox = transform.get("native_bbox")
        native_size = transform.get("native_bbox_size")
        bbox_tile = transform.get("bbox_in_sampling_tile")
        sampling_size = transform.get("sampling_tile_size")
        sampling_crop = transform.get("sampling_crop_region")
        segment_case = transform.get("segment_case", "unknown_segment_case")
        segment_case_str = str(segment_case or "unknown_segment_case")
        small_upscaled_bbox_restore = "small_segment_upscaled_bbox_restore" in segment_case_str
        if not native_crop or not native_bbox or not native_size or not bbox_tile or not sampling_size:
            return tile_processed_4x

        cx1, cy1, cx2, cy2 = [int(round(float(v))) for v in native_crop]
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in native_bbox]
        crop_w = max(1, (cx2 - cx1) * PID_SCALE)
        crop_h = max(1, (cy2 - cy1) * PID_SCALE)
        native_w, native_h = [max(1, int(round(float(v))) * PID_SCALE) for v in native_size]
        actual_w = int(tile_processed_4x.shape[2])
        actual_h = int(tile_processed_4x.shape[1])
        sample_w, sample_h = [max(1, int(round(float(v)))) for v in sampling_size]
        sx = actual_w / float(sample_w)
        sy = actual_h / float(sample_h)

        crop_canvas = None
        base_image = cls._ensure_bhwc_image(getattr(tbg.OUTPUTS, "upscaled_image", None))
        if torch.is_tensor(base_image) and base_image.ndim == 4:
            try:
                base_crop = base_image[:, cy1:cy2, cx1:cx2, :]
                if int(base_crop.shape[1]) > 0 and int(base_crop.shape[2]) > 0:
                    crop_canvas = nodes.ImageScale().upscale(
                        base_crop.to(device=tile_processed_4x.device, dtype=tile_processed_4x.dtype),
                        "lanczos",
                        crop_w,
                        crop_h,
                        False,
                    )[0]
            except Exception as exc:
                if getattr(tbg.API, "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                        f"native 4x crop background fallback failed: {exc}"
                    )
        if crop_canvas is None:
            crop_canvas = torch.zeros(
                (tile_processed_4x.shape[0], crop_h, crop_w, tile_processed_4x.shape[-1]),
                dtype=tile_processed_4x.dtype,
                device=tile_processed_4x.device,
            )
        pid_restore_mask_full = torch.zeros(
            (tile_processed_4x.shape[0], crop_h, crop_w),
            dtype=torch.float32,
            device=crop_canvas.device,
        )

        def _resize_mask_bhw(mask, height, width, device, dtype=torch.float32):
            mask = cls._mask_to_bhw(mask)
            if mask is None:
                return None
            mask = mask.to(device=device, dtype=torch.float32)
            if mask.numel() > 0 and float(mask.max().detach().cpu()) > 1.0:
                mask = mask / 255.0
            mask = mask.clamp(0.0, 1.0)
            if int(mask.shape[-2]) != int(height) or int(mask.shape[-1]) != int(width):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1),
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            return mask.to(dtype=dtype).clamp(0.0, 1.0)

        def _native_crop_mask_bhw(name):
            values = getattr(tbg.SEGMENTS, name, None)
            if not isinstance(values, (list, tuple)) or int(index_seg) >= len(values):
                return None
            return _resize_mask_bhw(values[int(index_seg)], crop_h, crop_w, crop_canvas.device)

        def _soft_mask_composite_into(canvas, y, x, source, mask_bhwc):
            height = min(int(source.shape[1]), int(canvas.shape[1]) - int(y))
            width = min(int(source.shape[2]), int(canvas.shape[2]) - int(x))
            if height <= 0 or width <= 0:
                return canvas
            source = source[:, :height, :width, :].to(device=canvas.device, dtype=canvas.dtype)
            mask_bhwc = mask_bhwc.to(device=canvas.device, dtype=canvas.dtype).clamp(0.0, 1.0)
            if int(mask_bhwc.shape[1]) != height or int(mask_bhwc.shape[2]) != width:
                if getattr(tbg.API, "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD native crop paste: "
                        f"resized mask {int(mask_bhwc.shape[2])}x{int(mask_bhwc.shape[1])} -> {width}x{height} "
                        "to match scaled source patch."
                    )
                mask_bhwc = torch.nn.functional.interpolate(
                    mask_bhwc.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
            else:
                mask_bhwc = mask_bhwc[:, :height, :width, :]
            region = canvas[:, int(y):int(y) + height, int(x):int(x) + width, :]
            canvas = canvas.clone()
            canvas[:, int(y):int(y) + height, int(x):int(x) + width, :] = region * (1.0 - mask_bhwc) + source * mask_bhwc
            return canvas

        context_blended = False
        use_context_restore = True
        if getattr(tbg.API, "status", None) == "Dev" and sampling_crop:
            print(
                f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                f"case={segment_case}; restore mode=context_overlap; "
                "restoring sampled context into native crop."
            )
        if use_context_restore and sampling_crop:
            scx1, scy1, scx2, scy2 = [int(round(float(v))) for v in sampling_crop]
            ox1 = max(cx1, scx1)
            oy1 = max(cy1, scy1)
            ox2 = min(cx2, scx2)
            oy2 = min(cy2, scy2)
            if ox2 > ox1 and oy2 > oy1:
                crop_global_w = max(1, scx2 - scx1)
                crop_global_h = max(1, scy2 - scy1)
                tx1 = int(round((ox1 - scx1) * (actual_w / float(crop_global_w))))
                ty1 = int(round((oy1 - scy1) * (actual_h / float(crop_global_h))))
                tx2 = int(round((ox2 - scx1) * (actual_w / float(crop_global_w))))
                ty2 = int(round((oy2 - scy1) * (actual_h / float(crop_global_h))))
                tx1 = max(0, min(actual_w - 1, tx1))
                ty1 = max(0, min(actual_h - 1, ty1))
                tx2 = max(tx1 + 1, min(actual_w, tx2))
                ty2 = max(ty1 + 1, min(actual_h, ty2))
                dx1 = max(0, (ox1 - cx1) * PID_SCALE)
                dy1 = max(0, (oy1 - cy1) * PID_SCALE)
                target_w = max(1, (ox2 - ox1) * PID_SCALE)
                target_h = max(1, (oy2 - oy1) * PID_SCALE)
                context = tile_processed_4x[:, ty1:ty2, tx1:tx2, :]
                context = nodes.ImageScale().upscale(context, "lanczos", target_w, target_h, False)[0]
                restore_mask = cls._segment_mask(index_seg, "inpaint", actual_w, actual_h)
                if restore_mask is None:
                    restore_mask = cls._segment_mask(index_seg, "composite", actual_w, actual_h)
                restore_mask = _resize_mask_bhw(restore_mask, actual_h, actual_w, tile_processed_4x.device)
                if restore_mask is not None:
                    restore_mask = restore_mask[:, ty1:ty2, tx1:tx2]
                    restore_mask = _resize_mask_bhw(
                        restore_mask,
                        target_h,
                        target_w,
                        crop_canvas.device,
                        crop_canvas.dtype,
                    )
                    if restore_mask is not None and float(restore_mask.max().detach().cpu()) > 0.001:
                        restore_mask_bhwc = restore_mask.unsqueeze(-1).to(device=crop_canvas.device, dtype=crop_canvas.dtype)
                        crop_canvas = _soft_mask_composite_into(
                            crop_canvas,
                            dy1,
                            dx1,
                            context[:, :target_h, :target_w, :].to(device=crop_canvas.device, dtype=crop_canvas.dtype),
                            restore_mask_bhwc,
                        )
                        mask_h = min(int(restore_mask.shape[1]), int(pid_restore_mask_full.shape[1]) - int(dy1))
                        mask_w = min(int(restore_mask.shape[2]), int(pid_restore_mask_full.shape[2]) - int(dx1))
                        if mask_h > 0 and mask_w > 0:
                            current_mask = pid_restore_mask_full[:, int(dy1):int(dy1) + mask_h, int(dx1):int(dx1) + mask_w]
                            pid_restore_mask_full[:, int(dy1):int(dy1) + mask_h, int(dx1):int(dx1) + mask_w] = torch.maximum(
                                current_mask,
                                restore_mask[:, :mask_h, :mask_w].to(device=pid_restore_mask_full.device, dtype=pid_restore_mask_full.dtype),
                            )
                        context_blended = True
                        if getattr(tbg.API, "status", None) == "Dev":
                            try:
                                coverage = float((restore_mask > 0.001).to(torch.float32).mean().detach().cpu())
                                feather = float(((restore_mask > 0.001) & (restore_mask < 0.999)).to(torch.float32).mean().detach().cpu())
                                print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                        f"case={segment_case}; restore mode=context_overlap; "
                        f"source_rect=({tx1},{ty1},{tx2},{ty2}) "
                        f"dest_rect=({dx1},{dy1},{dx1 + target_w},{dy1 + target_h}) "
                        f"mask={target_w}x{target_h} "
                        f"coverage={coverage:.4f} feather={feather:.4f}"
                                )
                                cls.debug_image_to_folder(
                                    MaskToImage_execute(restore_mask)[0],
                                    str(int(index_seg) + 1) + "_Segment_PID_native_crop_restore_mask_4x",
                                )
                            except Exception as exc:
                                print(
                                    f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                                    f"restore mask debug failed: {exc}"
                                )
                elif getattr(tbg.API, "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                        "no transformed restore mask available; native crop keeps base outside bbox fallback."
                    )

        offset_x = max(0, (bx1 - cx1) * PID_SCALE)
        offset_y = max(0, (by1 - cy1) * PID_SCALE)
        paste_w = min(native_w, crop_w - offset_x)
        paste_h = min(native_h, crop_h - offset_y)
        if not context_blended and small_upscaled_bbox_restore and paste_w > 0 and paste_h > 0:
            x1, y1, x2, y2 = [float(v) for v in bbox_tile]
            x1 = int(round(x1 * sx))
            y1 = int(round(y1 * sy))
            x2 = int(round(x2 * sx))
            y2 = int(round(y2 * sy))
            x1 = max(0, min(actual_w - 1, x1))
            y1 = max(0, min(actual_h - 1, y1))
            x2 = max(x1 + 1, min(actual_w, x2))
            y2 = max(y1 + 1, min(actual_h, y2))
            bbox_result = tile_processed_4x[:, y1:y2, x1:x2, :]
            bbox_result = nodes.ImageScale().upscale(bbox_result, "lanczos", native_w, native_h, False)[0]
            native_comp_mask = _native_crop_mask_bhw("compositing_mask")
            if native_comp_mask is not None:
                bbox_mask = native_comp_mask[:, offset_y:offset_y + paste_h, offset_x:offset_x + paste_w]
                bbox_mask = _resize_mask_bhw(bbox_mask, paste_h, paste_w, crop_canvas.device, crop_canvas.dtype)
                if bbox_mask is not None and float(bbox_mask.max().detach().cpu()) > 0.001:
                    crop_canvas = _soft_mask_composite_into(
                        crop_canvas,
                        offset_y,
                        offset_x,
                        bbox_result[:, :paste_h, :paste_w, :].to(device=crop_canvas.device, dtype=crop_canvas.dtype),
                        bbox_mask.unsqueeze(-1).to(device=crop_canvas.device, dtype=crop_canvas.dtype),
                    )
                    mask_h = min(int(bbox_mask.shape[1]), int(pid_restore_mask_full.shape[1]) - int(offset_y))
                    mask_w = min(int(bbox_mask.shape[2]), int(pid_restore_mask_full.shape[2]) - int(offset_x))
                    if mask_h > 0 and mask_w > 0:
                        current_mask = pid_restore_mask_full[:, int(offset_y):int(offset_y) + mask_h, int(offset_x):int(offset_x) + mask_w]
                        pid_restore_mask_full[:, int(offset_y):int(offset_y) + mask_h, int(offset_x):int(offset_x) + mask_w] = torch.maximum(
                            current_mask,
                            bbox_mask[:, :mask_h, :mask_w].to(device=pid_restore_mask_full.device, dtype=pid_restore_mask_full.dtype),
                        )
                    if getattr(tbg.API, "status", None) == "Dev":
                        coverage = float((bbox_mask > 0.001).to(torch.float32).mean().detach().cpu())
                        feather = float(((bbox_mask > 0.001) & (bbox_mask < 0.999)).to(torch.float32).mean().detach().cpu())
                        print(
                            f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                            f"case={segment_case}; restore mode=small_bbox; "
                            f"source_rect=({x1},{y1},{x2},{y2}) "
                            f"dest_rect=({offset_x},{offset_y},{offset_x + paste_w},{offset_y + paste_h}) "
                            f"mask={paste_w}x{paste_h} "
                            f"coverage={coverage:.4f} feather={feather:.4f}"
                        )
            elif getattr(tbg.API, "status", None) == "Dev":
                print(
                    f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                    "small bbox restore skipped because no native compositing mask was available."
                )
        elif not context_blended and not small_upscaled_bbox_restore and getattr(tbg.API, "status", None) == "Dev":
            print(
                f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                f"case={segment_case}; bbox restore skipped reason=not_small_upscaled_segment."
            )
        print(
            f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
            f"case={segment_case}; "
            f"stored 4x native crop {crop_w}x{crop_h} from sampling tile {actual_w}x{actual_h}; "
            f"bbox={native_w}x{native_h} at ({offset_x},{offset_y}); "
            "uncovered native crop initialized from upscaled base image; PiD restore uses masks."
        )
        if float(pid_restore_mask_full.max().detach().cpu()) > 0.001:
            try:
                restore_masks = getattr(tbg.TEMP, "pid_segment_native_crop_restore_masks_4x", None)
                if not isinstance(restore_masks, list):
                    segment_count = len(getattr(tbg.SEGMENTS, "segment_tiles", []) or [])
                    restore_masks = [None] * max(segment_count, int(index_seg) + 1)
                    tbg.TEMP.pid_segment_native_crop_restore_masks_4x = restore_masks
                while len(restore_masks) <= int(index_seg):
                    restore_masks.append(None)
                restore_masks[int(index_seg)] = pid_restore_mask_full.detach().cpu()
                if getattr(tbg.API, "status", None) == "Dev":
                    cls.debug_image_to_folder(
                        MaskToImage_execute(pid_restore_mask_full.detach().cpu())[0],
                        str(int(index_seg) + 1) + "_Segment_PID_native_crop_restore_full_mask_4x",
                    )
            except Exception as exc:
                if getattr(tbg.API, "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {int(index_seg) + 1}: "
                        f"failed to store native crop restore mask for post-color: {exc}"
                    )
        return crop_canvas

    @classmethod
    def _segment_origin_reference_crop_4x(cls, index_seg, target):
        target = cls._ensure_bhwc_image(target)
        transform = cls._segment_sampling_transform(index_seg)
        source = cls._ensure_bhwc_image(getattr(tbg.OUTPUTS, "upscaled_image", None))
        if target is None or transform is None or source is None:
            return None
        native_crop = transform.get("native_crop_region")
        if not native_crop:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in native_crop]
            crop = source[:, y1:y2, x1:x2, :]
            if int(crop.shape[1]) <= 0 or int(crop.shape[2]) <= 0:
                return None
            return nodes.ImageScale().upscale(
                crop.to(device=target.device, dtype=target.dtype),
                "lanczos",
                int(target.shape[2]),
                int(target.shape[1]),
                False,
            )[0]
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] PiD segment origin reference crop failed: {exc}")
            return None

    @classmethod
    def _segment_pid_source_reference_native_crop_4x(
        cls,
        reference_tile,
        index_seg,
        target,
        pid_source_width,
        pid_source_height,
        work_crop_4x=None,
    ):
        target = cls._ensure_bhwc_image(target)
        reference_tile = cls._ensure_bhwc_image(reference_tile)
        if target is None or reference_tile is None:
            return None
        try:
            target_w = int(pid_source_width) * PID_SCALE
            target_h = int(pid_source_height) * PID_SCALE
            reference_4x = nodes.ImageScale().upscale(
                reference_tile.to(device=target.device, dtype=target.dtype),
                "lanczos",
                target_w,
                target_h,
                False,
            )[0]
            if work_crop_4x is not None:
                x, y, w, h = [int(round(float(v))) for v in work_crop_4x]
                reference_4x = reference_4x[:, y:y + h, x:x + w, :]
            restored = cls._restore_segment_pid_4x_to_native_crop(reference_4x, index_seg)
            restored = cls._ensure_bhwc_image(restored)
            if restored is None:
                return None
            if int(restored.shape[1]) != int(target.shape[1]) or int(restored.shape[2]) != int(target.shape[2]):
                restored = nodes.ImageScale().upscale(
                    restored.to(device=target.device, dtype=target.dtype),
                    "lanczos",
                    int(target.shape[2]),
                    int(target.shape[1]),
                    False,
                )[0]
            return restored.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] PiD segment source reference restore failed: {exc}")
            return None

    @classmethod
    def _apply_pid_segment_post_decode_color(
        cls,
        segment_native_crop_4x,
        source_reference_tile,
        index,
        index_seg,
        pid_source_width,
        pid_source_height,
        work_crop_4x=None,
    ):
        if segment_native_crop_4x is None or not torch.is_tensor(segment_native_crop_4x):
            return segment_native_crop_4x
        method = getattr(tbg.PARAMS, "color_match_method", "none")
        if (
            method is None
            or str(method).lower() == "none"
        ):
            return segment_native_crop_4x
        overrides = list(getattr(tbg.PROMPTER, "output_color_match_js", []) or [])
        override = cls._pid_normalize_color_area_override(overrides[index] if int(index) < len(overrides) else "")
        if override == "color_match_off":
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] PiD segment {int(index_seg) + 1} post-color skipped: Color Match Off")
            return segment_native_crop_4x

        target = cls._ensure_bhwc_image(segment_native_crop_4x)
        reference = cls._segment_pid_source_reference_native_crop_4x(
            source_reference_tile,
            index_seg,
            target,
            pid_source_width,
            pid_source_height,
            work_crop_4x=work_crop_4x,
        )
        reference_label = "source_reference"
        if override == "color_match_from_origin":
            origin_reference = cls._segment_origin_reference_crop_4x(index_seg, target)
            if origin_reference is not None:
                reference = origin_reference
                reference_label = "origin"
        if reference is None:
            reference = cls._segment_origin_reference_crop_4x(index_seg, target)
            reference_label = "origin_fallback"
        if reference is None:
            return segment_native_crop_4x

        strength = float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)
        try:
            before = target
            apply_mask, mask_source, mask_coverage = cls._segment_native_crop_mask(
                index_seg,
                int(target.shape[2]),
                int(target.shape[1]),
            )
            method_is_tile_aware = cls.is_tbg_tile_aware(method)
            if method_is_tile_aware:
                try:
                    restore_masks = getattr(tbg.TEMP, "pid_segment_native_crop_restore_masks_4x", None)
                    if isinstance(restore_masks, (list, tuple)) and int(index_seg) < len(restore_masks):
                        restore_mask = restore_masks[int(index_seg)]
                        restore_mask = cls._mask_to_bhw(restore_mask)
                        if restore_mask is not None:
                            restore_mask = cls._scale_pid_mask(
                                restore_mask.to(torch.float32).clamp(0.0, 1.0),
                                int(target.shape[2]),
                                int(target.shape[1]),
                            )
                            restore_mask = cls._mask_to_bhw(restore_mask)
                            if restore_mask is not None and float(restore_mask.max().detach().cpu()) > 0.001:
                                apply_mask = restore_mask.to(device=target.device, dtype=torch.float32).clamp(0.0, 1.0)
                                mask_source = "pid_native_crop_restore_mask_4x"
                                mask_coverage = float((apply_mask > 0.001).to(torch.float32).mean().detach().cpu())
                except Exception as exc:
                    if getattr(tbg.API, "status", None) == "Dev":
                        print(
                            f"TBG[Node {tbg.INFO.id}] PiD segment {int(index_seg) + 1} restore-mask post-color fallback: {exc}"
                        )
            segment_metrics = None
            global_metrics = None
            if method_is_tile_aware:
                corrected, segment_metrics = cls._segment_pixel_grid_color_match(
                    reference,
                    target,
                    index_seg,
                    placement_mask=apply_mask,
                    strength=strength,
                    label=f"segment_post_decode_{int(index) + 1}_tile_aware_rgb_luma",
                    placement_source_override=mask_source,
                )
                if segment_metrics is None:
                    return segment_native_crop_4x
                method_label = "segment_tile_aware_rgb_luma"
            else:
                corrected, global_metrics = cls._global_rgb_luma_match(
                    reference,
                    target,
                    strength=strength,
                    apply_mask=apply_mask,
                    label=f"segment_post_decode_{int(index) + 1}",
                )
                if global_metrics is None:
                    return segment_native_crop_4x
                method_label = "global_rgb_luma_masked" if apply_mask is not None else "global_rgb_luma_full_crop_fallback"
            if override == "protect_new_generated_content":
                decision = f"{method_label}:protect_new_generated_content"
            elif override == "color_match_from_origin":
                decision = f"{method_label}:color_match_from_origin"
            else:
                decision = f"{method_label}:preset:{cls._final_color_mode_label()}"
            corrected = cls._ensure_bhwc_image(corrected).to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
            if getattr(tbg.API, "status", None) == "Dev":
                cls.debug_image_to_folder(before, str(index) + "_Segment_PID_PostColor_InputTarget4x")
                cls.debug_image_to_folder(reference, str(index) + "_Segment_PID_PostColor_Reference4x")
                cls.debug_image_to_folder(corrected, str(index) + "_Segment_PID_PostColor_Output4x")
                delta = torch.mean(torch.abs(corrected.to(torch.float32) - before.to(torch.float32))).item()
                if global_metrics is not None:
                    cls._log_global_rgb_luma_metrics(f"segment_post_decode_{int(index) + 1}", global_metrics)
                if segment_metrics is not None:
                    print(
                        f"TBG[Node {tbg.INFO.id}] PiD segment {int(index_seg) + 1} tile-aware post-color "
                        f"local_accepted={segment_metrics.get('local_accepted')} "
                        f"lowfreq_err(global={segment_metrics.get('global_error', 0.0):.8f}, "
                        f"candidate={segment_metrics.get('candidate_error', 0.0):.8f}) "
                        f"raw_err(global={segment_metrics.get('raw_global_error', 0.0):.8f}, "
                        f"candidate={segment_metrics.get('raw_candidate_error', 0.0):.8f}) "
                        f"cell={segment_metrics.get('cell_px')}px grid={segment_metrics.get('grid')} "
                        f"border_px={segment_metrics.get('border_px')} "
                        f"outer_px={segment_metrics.get('outer_border_px')} "
                        f"edge_guard_px={segment_metrics.get('edge_guard_px', 0.0):.1f} "
                        f"placement={segment_metrics.get('placement_source')}:{segment_metrics.get('placement_coverage', 0.0):.4f} "
                        f"support={segment_metrics.get('correction_support_coverage', 0.0):.4f} "
                        f"object={segment_metrics.get('object_source')}:{segment_metrics.get('object_coverage', 0.0):.4f} "
                        f"border={segment_metrics.get('border_coverage', 0.0):.4f} "
                        f"inner={segment_metrics.get('inner_coverage', 0.0):.4f} "
                        f"placement_weight={segment_metrics.get('placement_weight_coverage', 0.0):.4f} "
                        f"edge_polish={segment_metrics.get('edge_polish_coverage', 0.0):.4f} "
                        f"mean_abs_delta={segment_metrics.get('mean_correction', 0.0):.8f} "
                        f"max_delta={segment_metrics.get('max_correction', 0.0):.8f}"
                    )
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD segment {int(index_seg) + 1} post-color applied: "
                    f"decision={decision} reference={reference_label} method={method_label} "
                    f"mask_source={mask_source or 'none'} mask_coverage={mask_coverage:.4f} "
                    f"strength={strength} mean_abs_delta={delta:.8f}"
                )
            return corrected.to(device=segment_native_crop_4x.device, dtype=segment_native_crop_4x.dtype).clamp(0.0, 1.0)
        except Exception as exc:
            print(f"TBG[Node {tbg.INFO.id}] PiD segment {int(index_seg) + 1} post-color failed: {exc}")
            return segment_native_crop_4x

    @classmethod
    def _transform_segment_mask_to_sampling(cls, mask, transform):
        source = cls._mask_to_bhw(mask)
        if source is None or transform is None:
            return None
        try:
            native_region = transform.get("native_crop_region")
            sampling_crop = transform.get("sampling_crop_region")
            tile_size = transform.get("sampling_tile_size")
            if not native_region or not sampling_crop or not tile_size:
                return None
            sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in sampling_crop]
            nx1, ny1, nx2, ny2 = [int(round(float(v))) for v in native_region]
            tile_w, tile_h = [int(round(float(v))) for v in tile_size]
            crop_w = max(1, sx2 - sx1)
            crop_h = max(1, sy2 - sy1)
            native_w = max(1, nx2 - nx1)
            native_h = max(1, ny2 - ny1)
            source = source.to(torch.float32).clamp(0.0, 1.0)
            if int(source.shape[-2]) != native_h or int(source.shape[-1]) != native_w:
                source = torch.nn.functional.interpolate(
                    source.unsqueeze(1),
                    size=(native_h, native_w),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            out = torch.zeros((source.shape[0], crop_h, crop_w), dtype=source.dtype, device=source.device)
            ox1 = max(sx1, nx1)
            oy1 = max(sy1, ny1)
            ox2 = min(sx2, nx2)
            oy2 = min(sy2, ny2)
            if ox2 > ox1 and oy2 > oy1:
                src_x1 = ox1 - nx1
                src_y1 = oy1 - ny1
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
        except Exception:
            return None

    @classmethod
    def _boost_pid_inpaint_mask(cls, mask):
        if mask is None or not torch.is_tensor(mask):
            return mask
        mask = mask.clamp(0.0, 1.0)
        return (0.90 + (mask * 0.10)).clamp(0.90, 1.0)

    @classmethod
    def _build_segment_pid_context_4x(cls, base_tile_image, tile_to_process, index_seg, color_match_context=False, debug_prefix=""):
        width = int(tile_to_process.shape[2]) * PID_SCALE
        height = int(tile_to_process.shape[1]) * PID_SCALE
        base_context = nodes.ImageScale().upscale(base_tile_image, "lanczos", width, height, False)[0]
        reference_context = nodes.ImageScale().upscale(tile_to_process, "lanczos", width, height, False)[0]
        color_match_context = cls._cm_debug_stage_enabled("08_Segment_PostVAE_ColorMatch", color_match_context)
        if color_match_context:
            if tbg.API.status == "Dev":
                cls.debug_image_to_folder(reference_context, str(debug_prefix) + "_Segment_PID_context_reference_raw_4x")
            reference_context = cls._flux2_pid_color_match(
                base_context,
                reference_context,
                f"segment_context_reference_4x_{int(index_seg) + 1}",
                method=cls._pid_color_method(),
            )
            if tbg.API.status == "Dev":
                cls.debug_image_to_folder(reference_context, str(debug_prefix) + "_Segment_PID_context_reference_matched_4x")

        if tbg.API.status == "Dev":
            cls.debug_image_to_folder(reference_context, str(debug_prefix) + "_Segment_PID_base_context_4x")
            print(
                f"TBG[Node {tbg.INFO.id}] Segment PiD context tile {int(index_seg) + 1}: "
                "no PiD inpaint/context masks are built; Flux sampler owns segment masking."
            )
        return reference_context.clamp(0.0, 1.0), None, None

    @classmethod
    def _fit_decoded_pid_base_to_source(cls, decoded, source, index, visible_mask=None):
        """Fit VAE-decoded tile to source canvas without aspect-ratio stretching."""
        if decoded is None or source is None or not torch.is_tensor(decoded) or not torch.is_tensor(source):
            return decoded
        decoded = decoded.unsqueeze(0) if decoded.ndim == 3 else decoded
        source = source.unsqueeze(0) if source.ndim == 3 else source
        if decoded.ndim != 4 or source.ndim != 4:
            return decoded

        target_h = int(source.shape[1])
        target_w = int(source.shape[2])
        src_h = int(decoded.shape[1])
        src_w = int(decoded.shape[2])
        if src_w == target_w and src_h == target_h:
            return decoded

        fitted = source.clone().to(device=decoded.device, dtype=decoded.dtype)
        copy_w = min(src_w, target_w)
        copy_h = min(src_h, target_h)
        offset_x = 0
        offset_y = 0
        mask = cls._mask_to_bhw(visible_mask)
        if mask is not None:
            mask = cls._scale_pid_mask(mask, target_w, target_h)
            mask = cls._mask_to_bhw(mask)
            if mask is not None:
                active = torch.where(mask.to(torch.float32) > 0.001)
                if len(active) >= 3 and int(active[0].numel()) > 0:
                    mask_x0 = int(active[2].min().item())
                    mask_y0 = int(active[1].min().item())
                    offset_x = max(0, min(mask_x0, target_w - copy_w))
                    offset_y = max(0, min(mask_y0, target_h - copy_h))
        fitted[:, offset_y:offset_y + copy_h, offset_x:offset_x + copy_w, :] = decoded[:, :copy_h, :copy_w, :]
        print(
            f"TBG[Node {tbg.INFO.id}] PID base decode canvas-fitted for tile {index + 1}: "
            f"{src_w}x{src_h} -> {target_w}x{target_h} at offset=({offset_x},{offset_y}) without stretching"
        )
        if tbg.API.status == "Dev":
            cls.debug_image_to_folder(decoded, str(index) + "_PID_base_decode_unfitted")
            cls.debug_image_to_folder(fitted, str(index) + "_PID_base_decode_canvas_fitted")
        return fitted

    @classmethod
    def _fit_segment_decode_to_native_geometry(cls, decoded, native_tile, index):
        """Keep sampled segment content in native local geometry without shifting or stretching."""
        if decoded is None or native_tile is None or not torch.is_tensor(decoded) or not torch.is_tensor(native_tile):
            return native_tile
        decoded = decoded.unsqueeze(0) if decoded.ndim == 3 else decoded
        native_tile = native_tile.unsqueeze(0) if native_tile.ndim == 3 else native_tile
        if decoded.ndim != 4 or native_tile.ndim != 4:
            return native_tile
        target_h = int(native_tile.shape[1])
        target_w = int(native_tile.shape[2])
        source_h = int(decoded.shape[1])
        source_w = int(decoded.shape[2])
        if source_h == target_h and source_w == target_w:
            return decoded.to(device=native_tile.device, dtype=native_tile.dtype)

        decoded_native = decoded.to(device=native_tile.device, dtype=native_tile.dtype)
        fitted = torch.empty_like(native_tile)
        copy_h = max(0, min(source_h, target_h))
        copy_w = max(0, min(source_w, target_w))
        if copy_h > 0 and copy_w > 0:
            corner = decoded_native[:, copy_h - 1:copy_h, copy_w - 1:copy_w, :]
            fitted.copy_(corner.expand_as(fitted))
            fitted[:, :copy_h, :copy_w, :] = decoded_native[:, :copy_h, :copy_w, :]
            if copy_w < target_w:
                fitted[:, :copy_h, copy_w:target_w, :] = decoded_native[:, :copy_h, copy_w - 1:copy_w, :].expand(
                    -1,
                    -1,
                    target_w - copy_w,
                    -1,
                )
            if copy_h < target_h:
                fitted[:, copy_h:target_h, :, :] = fitted[:, copy_h - 1:copy_h, :, :].expand(
                    -1,
                    target_h - copy_h,
                    -1,
                    -1,
                )
        else:
            fitted.copy_(native_tile)
        print(
            f"TBG[Node {tbg.INFO.id}] Segment PiD sampled base native-fit tile {index + 1}: "
            f"{source_w}x{source_h} -> {target_w}x{target_h} with top-left copy and sampled-edge padding only, no alignment shift."
        )
        if tbg.API.status == "Dev":
            cls.debug_image_to_folder(fitted, str(index) + "_Segment_PID_base_native_fitted_no_shift")
        return fitted

    @staticmethod
    def _segment_pid_work_offset(width, height, work_size=1024):
        return (
            max(0, (int(work_size) - int(width)) // 2),
            max(0, (int(work_size) - int(height)) // 2),
        )

    @classmethod
    def _segment_pid_work_image(cls, image, offset_x, offset_y, work_size=1024):
        if image is None or not torch.is_tensor(image):
            return image
        image = image.unsqueeze(0) if image.ndim == 3 else image
        if image.ndim != 4:
            return image
        batch, source_h, source_w, channels = image.shape
        if source_w > work_size or source_h > work_size:
            return image
        fill = image.to(torch.float32).mean(dim=(1, 2), keepdim=True).to(dtype=image.dtype)
        canvas = fill.expand(batch, int(work_size), int(work_size), channels).clone()
        canvas[:, int(offset_y):int(offset_y) + int(source_h), int(offset_x):int(offset_x) + int(source_w), :] = image
        return canvas

    @classmethod
    def _segment_pid_full_background_work_from_pid_4x(cls, x0, y0, work_size, canvas_w, canvas_h, image):
        pid_tiles = getattr(tbg.TEMP, "pid_grid_images_4x", None)
        if not isinstance(pid_tiles, (list, tuple)) or not pid_tiles:
            return None
        pid_background = pid_tiles[0]
        if not torch.is_tensor(pid_background) or pid_background.ndim != 4:
            return None
        scale = PID_SCALE
        expected_w = int(canvas_w) * scale
        expected_h = int(canvas_h) * scale
        if int(pid_background.shape[2]) != expected_w or int(pid_background.shape[1]) != expected_h:
            return None
        crop = pid_background[
            :,
            int(y0) * scale:(int(y0) + int(work_size)) * scale,
            int(x0) * scale:(int(x0) + int(work_size)) * scale,
            :,
        ]
        if int(crop.shape[1]) != int(work_size) * scale or int(crop.shape[2]) != int(work_size) * scale:
            return None
        crop = crop.to(device=image.device, dtype=image.dtype)
        work = nodes.ImageScale().upscale(crop, "lanczos", int(work_size), int(work_size), False)[0]
        print(
            f"TBG[Node {tbg.INFO.id}] Segment PiD work canvas uses 4x PiD background context "
            f"crop=({int(x0)},{int(y0)},{int(work_size)},{int(work_size)})"
        )
        return work

    @classmethod
    def _segment_pid_sampling_background_from_pid_4x(cls, transform, image):
        if not isinstance(transform, dict) or image is None or not torch.is_tensor(image):
            return None
        pid_tiles = getattr(tbg.TEMP, "pid_grid_images_4x", None)
        if not isinstance(pid_tiles, (list, tuple)) or not pid_tiles:
            return None
        pid_background = pid_tiles[0]
        if not torch.is_tensor(pid_background) or pid_background.ndim != 4:
            return None
        sampling_crop = transform.get("sampling_crop_region")
        sampling_tile_size = transform.get("sampling_tile_size")
        if not sampling_crop or not sampling_tile_size:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in sampling_crop]
        tile_w, tile_h = [max(1, int(round(float(v)))) for v in sampling_tile_size]
        scale = PID_SCALE
        px1 = max(0, min(int(pid_background.shape[2]), x1 * scale))
        py1 = max(0, min(int(pid_background.shape[1]), y1 * scale))
        px2 = max(px1 + 1, min(int(pid_background.shape[2]), x2 * scale))
        py2 = max(py1 + 1, min(int(pid_background.shape[1]), y2 * scale))
        crop = pid_background[:, py1:py2, px1:px2, :]
        if crop.numel() == 0:
            return None
        crop = crop.to(device=image.device, dtype=image.dtype)
        work = nodes.ImageScale().upscale(crop, "lanczos", tile_w, tile_h, False)[0]
        print(
            f"TBG[Node {tbg.INFO.id}] Segment PiD work canvas uses transformed 4x PiD background "
            f"sampling_crop=({x1},{y1},{x2},{y2}) -> {tile_w}x{tile_h}"
        )
        return work

    @classmethod
    def _segment_pid_context_work_image(cls, image, index, index_seg=None, visible_mask=None, work_size=1024):
        if image is None or not torch.is_tensor(image):
            return None, None
        image = image.unsqueeze(0) if image.ndim == 3 else image
        if image.ndim != 4:
            return None, None
        seg_x = seg_y = None
        seg_w = int(image.shape[2])
        seg_h = int(image.shape[1])
        if seg_w == int(work_size) and seg_h == int(work_size):
            transform = cls._segment_sampling_transform(index_seg)
            crop = transform.get("sampling_crop_region") if isinstance(transform, dict) else None
            work_x0 = int(round(float(crop[0]))) if crop else 0
            work_y0 = int(round(float(crop[1]))) if crop else 0
            pid_work = cls._segment_pid_sampling_background_from_pid_4x(transform, image)
            if torch.is_tensor(pid_work) and pid_work.ndim == 4:
                work = pid_work.to(device=image.device, dtype=image.dtype)
                mask = cls._mask_to_bhw(visible_mask)
                if mask is not None:
                    mask = cls._scale_pid_mask(mask, seg_w, seg_h)
                    mask = cls._mask_to_bhw(mask)
                if mask is None:
                    mask = torch.zeros((1, seg_h, seg_w), dtype=torch.float32, device=work.device)
                mask = mask.to(device=work.device, dtype=work.dtype).clamp(0.0, 1.0)
                paste_margin = int(max(8, min(48, seg_w // 4, seg_h // 4)))
                if paste_margin > 0:
                    mask_nchw = mask.unsqueeze(1)
                    mask_nchw = torch.nn.functional.max_pool2d(
                        mask_nchw,
                        kernel_size=paste_margin * 2 + 1,
                        stride=1,
                        padding=paste_margin,
                    )
                    smooth_margin = max(1, paste_margin // 2)
                    mask_nchw = torch.nn.functional.avg_pool2d(
                        mask_nchw,
                        kernel_size=smooth_margin * 2 + 1,
                        stride=1,
                        padding=smooth_margin,
                    )
                    mask = torch.maximum(mask, mask_nchw[:, 0]).clamp(0.0, 1.0)
                mask = mask.unsqueeze(-1)
                work = work * (1.0 - mask) + image[:, :seg_h, :seg_w, :] * mask
                print(
                    f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {int(index) + 1}: "
                    "source matches work canvas; composited segment source over transformed PiD background."
                )
                return work, (0, 0, work_x0, work_y0)
            print(
                f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {int(index) + 1}: "
                f"source already matches {int(work_size)}x{int(work_size)} work canvas; using zero offset."
            )
            return image.clone(), (0, 0, work_x0, work_y0)
        crop_regions = getattr(tbg.SEGMENTS, "segms_crop_regions", None)
        if index_seg is not None and isinstance(crop_regions, (list, tuple)) and int(index_seg) < len(crop_regions):
            try:
                crop_region = crop_regions[int(index_seg)]
                seg_x = int(round(float(crop_region[0])))
                seg_y = int(round(float(crop_region[1])))
                crop_w = int(round(float(crop_region[2]))) - seg_x
                crop_h = int(round(float(crop_region[3]))) - seg_y
                if crop_w > 0 and crop_h > 0 and (crop_w != seg_w or crop_h != seg_h):
                    print(
                        f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {int(index) + 1}: "
                        f"crop region size {crop_w}x{crop_h} differs from image {seg_w}x{seg_h}; "
                        "using crop origin and image size."
                    )
            except Exception:
                seg_x = seg_y = None
        specs = getattr(tbg.PARAMS, "grid_specs", None) or []
        if (seg_x is None or seg_y is None) and int(index) >= len(specs):
            return None, None
        try:
            if seg_x is None or seg_y is None:
                _, _, _, spec_x, spec_y, _, _ = specs[int(index)][:7]
                seg_x = int(spec_x)
                seg_y = int(spec_y)
        except Exception:
            return None, None
        if seg_w > work_size or seg_h > work_size:
            return None, None

        canvas_w = int(getattr(tbg.SIZE, "UpscaledInputImageW", 0) or 0)
        canvas_h = int(getattr(tbg.SIZE, "UpscaledInputImageH", 0) or 0)
        context = None
        candidate = getattr(tbg.OUTPUTS, "grid_images_all", [None])[0]
        if (
            torch.is_tensor(candidate)
            and candidate.ndim == 4
            and (canvas_w <= 0 or int(candidate.shape[2]) == canvas_w)
            and (canvas_h <= 0 or int(candidate.shape[1]) == canvas_h)
        ):
            context = candidate
        if not torch.is_tensor(context) or context.ndim != 4:
            context = getattr(tbg.OUTPUTS, "upscaled_image", None)
        if not torch.is_tensor(context) or context.ndim != 4:
            return None, None
        if canvas_w <= 0:
            canvas_w = int(context.shape[2])
        if canvas_h <= 0:
            canvas_h = int(context.shape[1])
        if int(context.shape[2]) != canvas_w or int(context.shape[1]) != canvas_h:
            context = nodes.ImageScale().upscale(context, "lanczos", canvas_w, canvas_h, False)[0]

        max_x0 = max(0, canvas_w - int(work_size))
        max_y0 = max(0, canvas_h - int(work_size))
        x0 = min(max(0, int(round(seg_x + seg_w * 0.5 - work_size * 0.5))), max_x0)
        y0 = min(max(0, int(round(seg_y + seg_h * 0.5 - work_size * 0.5))), max_y0)
        offset_x = seg_x - x0
        offset_y = seg_y - y0

        work = cls._segment_pid_full_background_work_from_pid_4x(
            x0,
            y0,
            work_size,
            canvas_w,
            canvas_h,
            image,
        )
        if work is None:
            work = context[:, y0:y0 + int(work_size), x0:x0 + int(work_size), :].clone()
        if int(work.shape[1]) != int(work_size) or int(work.shape[2]) != int(work_size):
            fallback = cls._segment_pid_work_image(image, offset_x, offset_y, work_size)
            return fallback, (offset_x, offset_y, x0, y0)
        work = work.to(device=image.device, dtype=image.dtype)
        target = work[:, int(offset_y):int(offset_y) + seg_h, int(offset_x):int(offset_x) + seg_w, :]
        source = image[:, :seg_h, :seg_w, :]
        mask = cls._mask_to_bhw(visible_mask)
        if mask is not None:
            mask = cls._scale_pid_mask(mask, seg_w, seg_h)
            mask = cls._mask_to_bhw(mask)
        if mask is None:
            mask = torch.ones((1, seg_h, seg_w), dtype=torch.float32, device=work.device)
        mask = mask.to(device=work.device, dtype=work.dtype).clamp(0.0, 1.0)
        paste_margin = int(max(8, min(48, seg_w // 4, seg_h // 4)))
        if paste_margin > 0:
            mask_nchw = mask.unsqueeze(1)
            mask_nchw = torch.nn.functional.max_pool2d(
                mask_nchw,
                kernel_size=paste_margin * 2 + 1,
                stride=1,
                padding=paste_margin,
            )
            smooth_margin = max(1, paste_margin // 2)
            mask_nchw = torch.nn.functional.avg_pool2d(
                mask_nchw,
                kernel_size=smooth_margin * 2 + 1,
                stride=1,
                padding=smooth_margin,
            )
            mask = torch.maximum(mask, mask_nchw[:, 0]).clamp(0.0, 1.0)
        mask = mask.unsqueeze(-1)
        work[:, int(offset_y):int(offset_y) + seg_h, int(offset_x):int(offset_x) + seg_w, :] = (
            target * (1.0 - mask) + source * mask
        )
        return work, (offset_x, offset_y, x0, y0)

    @classmethod
    def _segment_pid_work_mask(cls, mask, source_width, source_height, offset_x, offset_y, work_size=1024, device=None, dtype=torch.float32):
        mask = cls._mask_to_bhw(mask)
        if mask is None:
            mask = torch.ones((1, int(source_height), int(source_width)), dtype=torch.float32, device=device)
        else:
            mask = cls._scale_pid_mask(mask, int(source_width), int(source_height))
            mask = cls._mask_to_bhw(mask)
        if mask is None:
            mask = torch.ones((1, int(source_height), int(source_width)), dtype=torch.float32, device=device)
        mask = mask.to(device=device or mask.device, dtype=dtype).clamp(0.0, 1.0)
        canvas = torch.zeros((int(mask.shape[0]), int(work_size), int(work_size)), dtype=mask.dtype, device=mask.device)
        canvas[:, int(offset_y):int(offset_y) + int(source_height), int(offset_x):int(offset_x) + int(source_width)] = mask[:, :int(source_height), :int(source_width)]
        return canvas

    @classmethod
    def _encode_segment_pid_work_latent(cls, vae, work_image, fallback_latent):
        try:
            return nodes.VAEEncode().encode(vae, work_image)[0]
        except Exception as exc:
            print(f"TBG[Node {tbg.INFO.id}] Segment PiD work latent encode failed, using segment-native PiD path: {exc}")
        return None

    @classmethod
    def _apply_pid_pre_decode_hooks(cls, vae, latent_output, base_image, edit_mask, index, label, denoise=None):
        if not cls._flux2_pid_active():
            return latent_output, base_image, None, False
        try:
            denoise_value = float(denoise)
        except Exception:
            denoise_value = 1.0
        if abs(denoise_value - 1.0) <= 1e-6:
            if getattr(getattr(tbg, "API", None), "status", None) == "Dev":
                print(
                    f"TBG[Node {tbg.INFO.id}] Flux2 PiD pre-hook skipped tile {index + 1}: "
                    f"denoise={denoise_value:.4f} full-denoise path."
                )
            return latent_output, base_image, None, False
        if edit_mask is None or not torch.is_tensor(edit_mask):
            if getattr(getattr(tbg, "API", None), "status", None) == "Dev":
                print(
                    f"TBG[Node {tbg.INFO.id}] Flux2 PiD pre-hook skipped tile {index + 1}: "
                    "no inpaint/noise mask."
                )
            return latent_output, base_image, None, False
        try:
            if float(edit_mask.detach().float().max().cpu()) <= 1e-6:
                if getattr(getattr(tbg, "API", None), "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] Flux2 PiD pre-hook skipped tile {index + 1}: "
                        "empty inpaint/noise mask."
                    )
                return latent_output, base_image, None, False
        except Exception:
            pass
        sampled_image = cls._normal_vae_decode_for_pid_color_reference(
            vae,
            latent_output,
            int(base_image.shape[2]),
            int(base_image.shape[1]),
            index,
            label,
        )
        if sampled_image is None or edit_mask is None:
            return latent_output, base_image, None, False
        protected_image, _, protected_mask = cls._flux2_pid_segment_preprotect_image(
            base_image,
            sampled_image,
            edit_mask,
            index,
        )
        protected_latent = cls._encode_segment_pid_work_latent(vae, protected_image, latent_output)
        if protected_latent is None:
            return latent_output, base_image, None, False
        return protected_latent, protected_image, protected_mask, True

    @staticmethod
    def _crop_segment_pid_work_4x(image, crop):
        if crop is None or image is None or not torch.is_tensor(image):
            return image
        if image.ndim != 4:
            return image
        x, y, width, height = [int(v) for v in crop]
        return image[:, y:y + height, x:x + width, :]

    @classmethod
    def _build_pid_base_context_4x(cls, base_tile_image, tile_to_process, inpaintmask, context_reference_mask=None, grid_spec=None, color_match_context=False, debug_prefix=""):
        width = int(tile_to_process.shape[2]) * PID_SCALE
        height = int(tile_to_process.shape[1]) * PID_SCALE
        base_context = nodes.ImageScale().upscale(base_tile_image, "lanczos", width, height, False)[0]
        reference_context = nodes.ImageScale().upscale(tile_to_process, "lanczos", width, height, False)[0]
        color_match_context = cls._cm_debug_stage_enabled("05_Flux2_PID_PostTone_ColorMatch", color_match_context)
        if color_match_context:
            if tbg.API.status == "Dev":
                cls.debug_image_to_folder(reference_context, str(debug_prefix) + "_PID_context_reference_raw_4x")
            reference_context = cls._flux2_pid_color_match(
                base_context,
                reference_context,
                "context_reference_4x",
                method=cls._pid_color_method(),
            )
            if tbg.API.status == "Dev":
                cls.debug_image_to_folder(reference_context, str(debug_prefix) + "_PID_context_reference_matched_4x")
        inpaint_mask_4x = cls._boost_pid_inpaint_mask(cls._scale_pid_mask(inpaintmask, width, height))
        context_mask_4x = cls._scale_pid_mask(context_reference_mask, width, height)
        if context_mask_4x is None:
            context_mask_4x = inpaint_mask_4x
        if context_mask_4x is None or not torch.is_tensor(context_mask_4x):
            return base_context, inpaint_mask_4x, None

        mask = context_mask_4x.to(device=base_context.device, dtype=base_context.dtype).clamp(0.0, 1.0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4:
            mask = mask[:, 0] if mask.shape[1] == 1 else mask[..., 0]
        if mask.ndim != 3:
            return base_context, inpaint_mask_4x, None

        active = (mask > 0.02)
        band = torch.zeros_like(mask, dtype=base_context.dtype)
        batch = int(mask.shape[0])
        has_left_context = True
        has_top_context = True
        if grid_spec is not None:
            try:
                _, _, _, x, y, _, _ = grid_spec[:7]
                has_left_context = int(x) > 0
                has_top_context = int(y) > 0
            except Exception:
                pass

        def side_length_from_rows(active_b, from_left=True):
            y0 = max(0, height // 4)
            y1 = min(height, height - y0)
            lengths = []
            for y in range(y0, y1):
                row = active_b[y, :]
                scan = row if from_left else torch.flip(row, dims=(0,))
                if not bool(scan[0].item()):
                    continue
                stops = torch.where(~scan)[0]
                lengths.append(int(stops[0].item()) if len(stops) else width)
            if not lengths:
                return 0
            lengths.sort()
            return lengths[len(lengths) // 2]

        def side_length_from_cols(active_b, from_top=True):
            x0 = max(0, width // 4)
            x1 = min(width, width - x0)
            lengths = []
            for x in range(x0, x1):
                col = active_b[:, x]
                scan = col if from_top else torch.flip(col, dims=(0,))
                if not bool(scan[0].item()):
                    continue
                stops = torch.where(~scan)[0]
                lengths.append(int(stops[0].item()) if len(stops) else height)
            if not lengths:
                return 0
            lengths.sort()
            return lengths[len(lengths) // 2]

        for b in range(batch):
            active_b = active[b]
            if has_left_context:
                length = side_length_from_rows(active_b, from_left=True)
                if length > 0:
                    ramp = torch.linspace(1.0, 0.0, length, device=band.device, dtype=band.dtype)
                    band[b, :, :length] = torch.maximum(band[b, :, :length], ramp.view(1, -1))

            if has_top_context:
                length = side_length_from_cols(active_b, from_top=True)
                if length > 0:
                    ramp = torch.linspace(1.0, 0.0, length, device=band.device, dtype=band.dtype)
                    band[b, :length, :] = torch.maximum(band[b, :length, :], ramp.view(-1, 1))

        band = band.clamp(0.0, 1.0)
        blend = band.unsqueeze(-1)
        context = base_context * (1.0 - blend) + reference_context.to(device=base_context.device, dtype=base_context.dtype) * blend
        if color_match_context and tbg.API.status == "Dev":
            cls.debug_image_to_folder(context.clamp(0.0, 1.0), str(debug_prefix) + "_PID_base_context_color_locked_4x")
        return context.clamp(0.0, 1.0), inpaint_mask_4x, band

    @classmethod
    def _scaled_grid_specs(cls, grid_specs, scale=PID_SCALE):
        scaled = []
        for spec in grid_specs:
            row, col, index, x, y, width, height = spec[:7]
            scaled.append([row, col, index, int(x) * scale, int(y) * scale, int(width) * scale, int(height) * scale])
        return scaled

    @classmethod
    def _scale_refiner_size_for_pid(cls, size):
        scaled = copy.copy(size)
        skip = {"rows_qty", "cols_qty", "len_grid_images", "len_segments"}
        tokens = ("w", "h", "width", "height", "full", "tile", "crop", "margin", "blur", "shift", "overlay", "outer", "pad")
        for name, value in list(vars(scaled).items()):
            lname = name.lower()
            if name in skip or lname.endswith("_qty") or lname.endswith("count"):
                continue
            if not any(token in lname for token in tokens):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                setattr(scaled, name, int(round(value * PID_SCALE)))
            elif isinstance(value, float):
                setattr(scaled, name, value * PID_SCALE)
        return scaled

    @classmethod
    def _debug_save_raw_sampler_decode(cls, index, vaedecoder, latent_output):
        try:
            if tbg.KSAMPLER.tiled:
                raw_sampler_decode = (nodes.VAEDecodeTiled().decode(
                    vaedecoder,
                    latent_output,
                    tbg.SIZE.tile_size_vae,
                    tbg.SIZE.tile_size_vae // 4,
                    tbg.SIZE.tile_size_vae // 4,
                )[0].unsqueeze(0))[0]
            else:
                raw_sampler_decode = (nodes.VAEDecode().decode(vaedecoder, latent_output)[0].unsqueeze(0))[0]
            cls.debug_image_to_folder(raw_sampler_decode, str(index) + "_Raw_Sampler_VAE_Decode")
        except Exception as exc:
            log(
                f"tile {index + 1}: Raw sampler VAE debug decode failed: {exc}",
                None,
                None,
                f"Node {tbg.INFO.id}",
            )

    @classmethod
    def _debug_save_flux2_sampler_parity(cls, index, vaedecoder, latent_image, latent_source, positive, negative, tile_cfg, denoise, sampler_input_image=None):
        if getattr(getattr(tbg, "API", None), "status", None) != "Dev":
            return
        if getattr(tbg.KSAMPLER, "model_type", None) != "FLUX2":
            return
        try:
            samples = latent_image.get("samples") if isinstance(latent_image, dict) else None
            if torch.is_tensor(samples) and not bool(getattr(tbg.KSAMPLER, "pid_vae_decode", False)):
                latent_for_decode = {"samples": samples}
                if tbg.KSAMPLER.tiled:
                    decoded = (nodes.VAEDecodeTiled().decode(
                        vaedecoder,
                        latent_for_decode,
                        tbg.SIZE.tile_size_vae,
                        tbg.SIZE.tile_size_vae // 4,
                        tbg.SIZE.tile_size_vae // 4,
                    )[0].unsqueeze(0))[0]
                else:
                    decoded = (nodes.VAEDecode().decode(vaedecoder, latent_for_decode)[0].unsqueeze(0))[0]
                cls.debug_image_to_folder(decoded, str(index) + "_Flux2Parity_VAE_Roundtrip_Input_Decode")
                if torch.is_tensor(sampler_input_image) and sampler_input_image.ndim == 4:
                    reference = sampler_input_image.to(device=decoded.device, dtype=decoded.dtype)
                    if int(reference.shape[1]) != int(decoded.shape[1]) or int(reference.shape[2]) != int(decoded.shape[2]):
                        reference = nodes.ImageScale().upscale(reference, "bilinear", int(decoded.shape[2]), int(decoded.shape[1]), False)[0]
                    delta = torch.abs(decoded.to(torch.float32) - reference.to(torch.float32))
                    mean_delta = float(delta.mean().detach().cpu())
                    rgb_delta = tuple(float(v) for v in delta.mean(dim=(0, 1, 2)).reshape(-1)[:3].detach().cpu())
                    print(
                        "[TBG Flux2 Parity] "
                        f"tile={index + 1} latent_input_decode is VAE roundtrip preview; "
                        f"sampler_input_vs_decode mean_abs_delta={mean_delta:.8f} "
                        f"rgb_delta=({rgb_delta[0]:.6f},{rgb_delta[1]:.6f},{rgb_delta[2]:.6f})"
                    )
            elif torch.is_tensor(samples) and bool(getattr(tbg.KSAMPLER, "pid_vae_decode", False)):
                print(
                    "[TBG Flux2 Parity] "
                    f"tile={index + 1} pre-sampler VAE roundtrip preview skipped: PiD VAE is active."
                )
        except Exception as exc:
            print(f"[TBG Flux2 Parity] tile={index + 1} latent input decode failed: {exc}")

        try:
            def tensor_shape(value):
                return list(value.shape) if torch.is_tensor(value) else None

            def mask_stats(value):
                if not torch.is_tensor(value):
                    return None
                v = value.detach().float().cpu()
                return {
                    "shape": list(v.shape),
                    "min": float(v.min()) if v.numel() else 0.0,
                    "max": float(v.max()) if v.numel() else 0.0,
                    "mean": float(v.mean()) if v.numel() else 0.0,
                }

            trace = {
                "tile": int(index + 1),
                "latent_source": str(latent_source),
                "sampler_name": str(tbg.KSAMPLER.sampler_name),
                "scheduler": str(tbg.KSAMPLER.scheduler),
                "steps": int(tbg.KSAMPLER.steps),
                "denoise": float(denoise),
                "cfg": float(tile_cfg),
                "flux_guidance": float(tbg.KSAMPLER.Flux_Guidance) if tbg.KSAMPLER.Flux_Guidance is not None else None,
                "vae_encode_tiled_requested": bool(getattr(tbg.KSAMPLER, "tiled", False)),
                "tile_size_vae": int(getattr(tbg.SIZE, "tile_size_vae", 0) or 0),
                "latent_samples_shape": tensor_shape(samples),
                "has_noise_mask": isinstance(latent_image, dict) and "noise_mask" in latent_image,
                "has_private_flux2_mask": isinstance(latent_image, dict) and "_flux2_inpaint_mask" in latent_image,
                "has_private_flux2_config": isinstance(latent_image, dict) and "_flux2_differential" in latent_image,
                "noise_mask": mask_stats(latent_image.get("noise_mask")) if isinstance(latent_image, dict) else None,
                "private_flux2_mask": mask_stats(latent_image.get("_flux2_inpaint_mask")) if isinstance(latent_image, dict) else None,
                "positive_entries": len(positive) if isinstance(positive, (list, tuple)) else 0,
                "negative_entries": len(negative) if isinstance(negative, (list, tuple)) else 0,
                "positive_reference_latents": cls._conditioning_ref_latent_count(positive),
                "negative_reference_latents": cls._conditioning_ref_latent_count(negative),
            }
            folder = os.path.join(folder_paths.get_temp_directory(), "TBG", "compareTiles")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{index}_Flux2Parity_Sampler_Input.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            print(
                "[TBG Flux2 Parity] "
                f"tile={index + 1} latent_source={latent_source} cfg={float(tile_cfg):.3f} "
                f"guidance={trace['flux_guidance']} noise_mask={trace['has_noise_mask']} "
                f"private_mask={trace['has_private_flux2_mask']} refs={trace['positive_reference_latents']}/{trace['negative_reference_latents']}"
            )
        except Exception as exc:
            print(f"[TBG Flux2 Parity] tile={index + 1} trace failed: {exc}")

    @classmethod
    def _worker_cpu_value(cls, value):
        if torch.is_tensor(value):
            return value.detach().to("cpu", copy=True).contiguous()
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(**{name: cls._worker_cpu_value(item) for name, item in vars(value).items()})
        if isinstance(value, dict):
            return {key: cls._worker_cpu_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._worker_cpu_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._worker_cpu_value(item) for item in value)
        return value

    @classmethod
    def _scaled_pid_segment_rebuild_inputs(cls):
        compositing_masks = []
        crop_regions = []
        segms_new = getattr(tbg.SEGMENTS, "segms_new", None)
        if segms_new is None:
            return compositing_masks, crop_regions
        try:
            _, segms = segms_new
        except Exception:
            return compositing_masks, crop_regions

        runtime_masks = getattr(tbg.SEGMENTS, "compositing_mask", None)
        for index_seg, seg in enumerate(segms or []):
            crop_region = getattr(seg, "crop_region", None)
            if crop_region is not None:
                try:
                    crop_regions.append(tuple(int(round(float(v) * PID_SCALE)) for v in crop_region))
                except Exception:
                    crop_regions.append(crop_region)

            compositing_mask = None
            if isinstance(runtime_masks, (list, tuple)) and index_seg < len(runtime_masks):
                compositing_mask = runtime_masks[index_seg]
            if compositing_mask is None:
                compositing_mask = getattr(seg, "compositing_mask", None)
            if torch.is_tensor(compositing_mask) and compositing_mask.ndim >= 2:
                try:
                    hw = cls._mask_height_width(compositing_mask)
                    if hw is None:
                        raise ValueError(f"unsupported compositing mask shape {tuple(compositing_mask.shape)}")
                    height = int(hw[0]) * PID_SCALE
                    width = int(hw[1]) * PID_SCALE
                    compositing_mask = cls._scale_pid_mask(compositing_mask, width, height)
                    if getattr(tbg.API, "status", None) == "Dev":
                        print(
                            f"TBG[Node {tbg.INFO.id}] PID 4x final rebuild: segment {index_seg + 1} "
                            f"compositing mask scaled {hw[1]}x{hw[0]} -> {width}x{height} "
                            f"from shape={tuple(compositing_mask.shape)}"
                        )
                except Exception:
                    pass
            compositing_masks.append(compositing_mask)

        return compositing_masks, crop_regions

    @classmethod
    def _scaled_pid_segment_binary_masks(cls, crop_regions):
        masks = []
        values = getattr(tbg.SEGMENTS, "segment_binary_masks", None)
        fallback_values = getattr(tbg.SEGMENTS, "segms_cropped_masks", None)
        for index_seg, crop_region in enumerate(crop_regions or []):
            source = None
            if isinstance(values, (list, tuple)) and index_seg < len(values):
                source = values[index_seg]
            if source is None and isinstance(fallback_values, (list, tuple)) and index_seg < len(fallback_values):
                source = fallback_values[index_seg]
            if not torch.is_tensor(source):
                masks.append(None)
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in crop_region]
                masks.append(cls._scale_pid_mask(source, max(1, x2 - x1), max(1, y2 - y1)))
            except Exception:
                masks.append(None)
        return masks

    @classmethod
    def _pid_tiles_with_native_segment_crops(cls, pid_tiles, crop_regions):
        """Use true 4x PiD native segment crops for final segment compositing.

        The 1x restored segment crop is only for the sequential 1k reference
        canvas used before the next segment is sampled. The final PiD image must
        keep the 4x crop produced by PiD; otherwise the final compositor inserts
        an upscaled/downscaled 1k proxy and loses the PiD detail.
        """
        rebuilt_tiles = list(pid_tiles)
        specs = getattr(tbg.PARAMS, "grid_specs", None) or []
        grid_spec_tile = [spec for spec in specs if not cls._is_segment_spec(spec)]
        grid_spec_custom = [spec for spec in specs if cls._is_segment_spec(spec)]
        source_meta = getattr(tbg.TEMP, "pid_grid_images_4x_source", None)
        for index_seg, _spec in enumerate(grid_spec_custom):
            index_custom = index_seg + len(grid_spec_tile)
            if index_custom >= len(rebuilt_tiles):
                continue
            if index_seg >= len(crop_regions):
                continue

            try:
                x1, y1, x2, y2 = crop_regions[index_seg]
                target_w = max(1, int(round(float(x2) - float(x1))))
                target_h = max(1, int(round(float(y2) - float(y1))))
            except Exception:
                continue

            current_tile = rebuilt_tiles[index_custom]
            meta = source_meta[index_custom] if isinstance(source_meta, list) and index_custom < len(source_meta) else None
            fresh_segment_source = (
                isinstance(meta, dict)
                and meta.get("source_kind") == "pid_restored_native_crop"
                and int(meta.get("index_seg", -1)) == int(index_seg)
            )
            if torch.is_tensor(current_tile):
                if current_tile.ndim == 3:
                    current_tile = current_tile.unsqueeze(0)
                current_w = int(current_tile.shape[2])
                current_h = int(current_tile.shape[1])
                if current_w == target_w and current_h == target_h and fresh_segment_source:
                    rebuilt_tiles[index_custom] = current_tile
                    print(
                        f"TBG[Node {tbg.INFO.id}] PID 4x final rebuild: segment {index_seg + 1} "
                        f"uses true PiD native 4x crop {target_w}x{target_h} for final composite."
                    )
                    continue
            native_tile = None
            try:
                transform = cls._segment_sampling_transform(index_seg)
                native_crop = transform.get("native_crop_region") if isinstance(transform, dict) else None
                base_image = cls._ensure_bhwc_image(getattr(tbg.OUTPUTS, "upscaled_image", None))
                if native_crop and torch.is_tensor(base_image):
                    cx1, cy1, cx2, cy2 = [int(round(float(v))) for v in native_crop]
                    base_h = int(base_image.shape[1])
                    base_w = int(base_image.shape[2])
                    cx1 = max(0, min(base_w, cx1))
                    cx2 = max(0, min(base_w, cx2))
                    cy1 = max(0, min(base_h, cy1))
                    cy2 = max(0, min(base_h, cy2))
                    if cx2 > cx1 and cy2 > cy1:
                        native_tile = base_image[:, cy1:cy2, cx1:cx2, :]
            except Exception:
                native_tile = None
            if torch.is_tensor(native_tile):
                if native_tile.ndim == 3:
                    native_tile = native_tile.unsqueeze(0)
                source_w = int(native_tile.shape[2])
                source_h = int(native_tile.shape[1])
                rebuilt_tiles[index_custom] = nodes.ImageScale().upscale(
                    native_tile,
                    "bilinear",
                    target_w,
                    target_h,
                    False,
                )[0]
                if isinstance(source_meta, list) and index_custom < len(source_meta):
                    source_meta[index_custom] = {
                        "source_kind": "bilinear_original_native_crop_to_4x",
                        "index_seg": int(index_seg),
                    }
                print(
                    f"TBG[Node {tbg.INFO.id}] PID 4x final rebuild: segment {index_seg + 1} "
                    f"missing fresh PiD crop; using original native crop -> 4x "
                    f"{source_w}x{source_h} -> {target_w}x{target_h}."
                )
                continue
            previous_shape = getattr(rebuilt_tiles[index_custom], "shape", None)
            print(
                f"TBG[Node {tbg.INFO.id}] PID 4x final rebuild: segment {index_seg + 1} "
                f"has no fresh PiD crop and no original native-crop fallback; keeping existing source "
                f"{target_w}x{target_h}; current shape={previous_shape}."
            )

        return rebuilt_tiles

    @classmethod
    def _pid_tensor_to_device(cls, value, device, dtype=None):
        if torch.is_tensor(value):
            target_dtype = dtype or value.dtype
            return value.to(device=device, dtype=target_dtype, non_blocking=True).contiguous()
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(**{name: cls._pid_tensor_to_device(item, device, dtype) for name, item in vars(value).items()})
        if isinstance(value, dict):
            return {key: cls._pid_tensor_to_device(item, device, dtype) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._pid_tensor_to_device(item, device, dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._pid_tensor_to_device(item, device, dtype) for item in value)
        return value

    @classmethod
    def _pid_rebuild_cuda_device(cls, tensors):
        for tensor in tensors or []:
            if torch.is_tensor(tensor) and tensor.device.type == "cuda":
                return tensor.device
        if torch.cuda.is_available():
            return mm.get_torch_device()
        return None

    @classmethod
    def _rebuild_pid_refiner_output_4x_gpu(
        cls,
        pid_tiles,
        scaled_specs,
        scaled_params,
        scaled_size,
        reference,
        compositing_masks,
        crop_regions,
        native_nosegments,
        target_width,
        target_height,
    ):
        if not pid_gpu_final_rebuild_enabled():
            return None
        device = cls._pid_rebuild_cuda_device([reference] + [tile for tile in pid_tiles if torch.is_tensor(tile)])
        if device is None:
            return None

        start = time.perf_counter()
        reference_gpu = cls._pid_tensor_to_device(reference, device)
        pid_tiles_gpu = cls._pid_tensor_to_device(list(pid_tiles), device)
        masks_gpu = cls._pid_tensor_to_device(list(compositing_masks or []), device)
        scaled_params_gpu = cls._pid_tensor_to_device(scaled_params, device)
        scaled_size_gpu = cls._pid_tensor_to_device(scaled_size, device)
        normal_specs = [spec for spec in scaled_specs if not cls._is_segment_spec(spec)]
        normal_tiles = [pid_tiles_gpu[index] for index, spec in enumerate(scaled_specs) if not cls._is_segment_spec(spec) and index < len(pid_tiles_gpu)]
        if not normal_tiles:
            raise RuntimeError("TBG PID GPU final rebuild found no normal tiles to stitch.")

        tile_only_canvas = gpu_pid_tile_rebuild(
            normal_tiles,
            normal_specs,
            target_width,
            target_height,
            stitch_feather=int(getattr(scaled_size, "composite_blur_margin", 16) or 16),
            label=f"TBG[Node {tbg.INFO.id}] PID 4x ETUR",
        )
        tile_only_canvas = tile_only_canvas.to(device=device, dtype=reference_gpu.dtype).clamp(0.0, 1.0)
        if getattr(tbg.API, "status", None) == "Dev":
            cls.debug_image_to_folder(tile_only_canvas, "PID_FinalRebuild_TileOnlyCanvas4x")
        print(
            f"TBG[Node {tbg.INFO.id}] PID 4x GPU tile-only rebuild completed: "
            f"{target_width}x{target_height} device={tile_only_canvas.device} "
            f"({time.perf_counter() - start:.2f}s)"
        )

        if native_nosegments:
            rebuilt = tile_only_canvas
            return rebuilt, tile_only_canvas, rebuilt.clone()

        native_start = time.perf_counter()
        from ....TBG.TBG_APP.TBG_APP import TBG_Image as NativeTBGImage
        from ....TBG.TBG_APP.constants import TBG as NativeTBGProxy, get_current_tbg, get_current_tiler_id, set_current_tiler_id

        previous_tiler_id = get_current_tiler_id()
        set_current_tiler_id(getattr(tbg.INFO, "tiler_id", None))
        native_tbg = get_current_tbg()
        previous = {
            "PARAMS": getattr(native_tbg, "PARAMS", None),
            "SIZE": getattr(native_tbg, "SIZE", None),
            "API": getattr(native_tbg, "API", None),
            "OUTPUTS_grid_images_all": getattr(native_tbg.OUTPUTS, "grid_images_all", None),
            "OUTPUTS_orig_grid_images_all": getattr(native_tbg.OUTPUTS, "orig_grid_images_all", None),
            "OUTPUTS_upscaled_image": getattr(native_tbg.OUTPUTS, "upscaled_image", None),
            "OUTPUTS_last_final_image": getattr(native_tbg.OUTPUTS, "last_final_image", None),
            "SEGMENTS_crop_regions": getattr(native_tbg.SEGMENTS, "segms_crop_regions", None),
            "SEGMENTS_compositing_mask": getattr(native_tbg.SEGMENTS, "compositing_mask", None),
            "SEGMENTS_binary_masks": getattr(native_tbg.SEGMENTS, "segment_binary_masks", None),
        }
        try:
            native_tbg.PARAMS = scaled_params_gpu
            native_tbg.SIZE = scaled_size_gpu
            native_tbg.API = tbg.API
            native_tbg.OUTPUTS.grid_images_all = pid_tiles_gpu
            native_tbg.OUTPUTS.orig_grid_images_all = pid_tiles_gpu
            native_tbg.OUTPUTS.upscaled_image = reference_gpu
            native_tbg.OUTPUTS.last_final_image = None
            native_tbg.SEGMENTS.segms_crop_regions = crop_regions
            native_tbg.SEGMENTS.compositing_mask = masks_gpu
            native_tbg.SEGMENTS.segment_binary_masks = cls._pid_tensor_to_device(
                getattr(scaled_params_gpu, "pid_segment_binary_masks", None),
                device,
            )
            NativeTBGProxy.PARAMS.__dict__.update(getattr(scaled_params_gpu, "__dict__", {}))
            NativeTBGProxy.SIZE.__dict__.update(getattr(scaled_size_gpu, "__dict__", {}))
            NativeTBGProxy.PARAMS.SegFusion_Initializer_run_once = True
            rebuilt, only_tiles, _ = NativeTBGImage.rebuild_final_image(
                pid_tiles_gpu,
                reference_gpu,
                masks_gpu,
                crop_regions,
                nosegments=False,
                full_image_only_tiles=tile_only_canvas,
            )
            if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
                rebuilt = rebuilt.unsqueeze(0)
            if torch.is_tensor(only_tiles) and only_tiles.ndim == 3:
                only_tiles = only_tiles.unsqueeze(0)
            if only_tiles is None:
                only_tiles = tile_only_canvas
            if getattr(tbg.API, "status", None) == "Dev" and torch.is_tensor(rebuilt):
                cls.debug_image_to_folder(rebuilt, "PID_FinalRebuild_AfterSegments4x")
            print(
                f"TBG[Node {tbg.INFO.id}] PID 4x GPU segment composite completed: "
                f"{target_width}x{target_height} device={getattr(rebuilt, 'device', 'unknown')} "
                f"({time.perf_counter() - native_start:.2f}s)"
            )
            return rebuilt, tile_only_canvas, rebuilt.clone() if torch.is_tensor(rebuilt) else rebuilt
        finally:
            native_tbg.PARAMS = previous["PARAMS"]
            native_tbg.SIZE = previous["SIZE"]
            native_tbg.API = previous["API"]
            native_tbg.OUTPUTS.grid_images_all = previous["OUTPUTS_grid_images_all"]
            native_tbg.OUTPUTS.orig_grid_images_all = previous["OUTPUTS_orig_grid_images_all"]
            native_tbg.OUTPUTS.upscaled_image = previous["OUTPUTS_upscaled_image"]
            native_tbg.OUTPUTS.last_final_image = previous["OUTPUTS_last_final_image"]
            native_tbg.SEGMENTS.segms_crop_regions = previous["SEGMENTS_crop_regions"]
            native_tbg.SEGMENTS.compositing_mask = previous["SEGMENTS_compositing_mask"]
            native_tbg.SEGMENTS.segment_binary_masks = previous["SEGMENTS_binary_masks"]
            set_current_tiler_id(previous_tiler_id)

    @classmethod
    def _pid_final_color_base_mask_to_bhwc(cls, mask, height, width, device, dtype):
        mask = cls._mask_to_bhw(mask)
        if mask is None:
            return None
        mask = mask.unsqueeze(-1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if int(mask.shape[1]) != int(height) or int(mask.shape[2]) != int(width):
            mask = torch.nn.functional.interpolate(
                mask.permute(0, 3, 1, 2),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        return cls._widen_pid_final_color_base_mask(mask, height, width)

    @classmethod
    def _widen_pid_final_color_base_mask(cls, mask_bhwc, height, width):
        """Reference-only mask expansion for final PiD color match.

        This mask is not used for final segment placement. It only builds the
        color-match reference image, so it can use a much wider gradient than
        the compositor to avoid hard color authority changes around large
        segment crops.
        """
        if mask_bhwc is None or not torch.is_tensor(mask_bhwc):
            return mask_bhwc
        if mask_bhwc.ndim != 4:
            return mask_bhwc
        try:
            started = time.perf_counter()
            height = int(height)
            width = int(width)
            min_side = max(1, min(height, width))
            # At PiD scale, 64px 1x fusion blur is 256px. Larger segment crops
            # need more room, but cap it to avoid turning the whole crop into
            # segment authority.
            gradient_px = int(max(64 * PID_SCALE, min(512, round(min_side * 0.08))))
            if gradient_px <= 0:
                return mask_bhwc
            downscale = max(1, int((max(height, width) + 1023) // 1024))
            work_h = max(1, int(round(height / downscale)))
            work_w = max(1, int(round(width / downscale)))
            radius = max(2, int(round(gradient_px / downscale)))
            kernel = radius * 2 + 1
            source = mask_bhwc.permute(0, 3, 1, 2).to(torch.float32).clamp(0.0, 1.0)
            small = torch.nn.functional.interpolate(
                source,
                size=(work_h, work_w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            dilated = torch.nn.functional.max_pool2d(
                small,
                kernel_size=(1, kernel),
                stride=1,
                padding=(0, radius),
            )
            dilated = torch.nn.functional.max_pool2d(
                dilated,
                kernel_size=(kernel, 1),
                stride=1,
                padding=(radius, 0),
            )
            padded = torch.nn.functional.pad(dilated, (radius, radius, 0, 0), mode="reflect")
            blurred = torch.nn.functional.avg_pool2d(
                padded,
                kernel_size=(1, kernel),
                stride=1,
            ).clamp(0.0, 1.0)
            padded = torch.nn.functional.pad(blurred, (0, 0, radius, radius), mode="reflect")
            blurred = torch.nn.functional.avg_pool2d(
                padded,
                kernel_size=(kernel, 1),
                stride=1,
            ).clamp(0.0, 1.0)
            widened = torch.nn.functional.interpolate(
                blurred,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            widened = torch.maximum(source, widened).permute(0, 2, 3, 1).to(dtype=mask_bhwc.dtype)
            if getattr(tbg.API, "status", None) == "Dev":
                original_coverage = float((mask_bhwc > 0.01).to(torch.float32).mean().detach().cpu())
                widened_coverage = float((widened > 0.01).to(torch.float32).mean().detach().cpu())
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final color-match segment mask widened "
                    f"device={source.device} gradient={gradient_px}px downscale={downscale} "
                    f"work={work_w}x{work_h} kernel={kernel} "
                    f"coverage={original_coverage:.4f}->{widened_coverage:.4f} "
                    f"elapsed_ms={(time.perf_counter() - started) * 1000.0:.1f}"
                )
            return widened.clamp(0.0, 1.0)
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                print(f"TBG[Node {tbg.INFO.id}] PiD final color-match mask widen failed: {exc}")
            return mask_bhwc

    @classmethod
    def _pid_color_work_device(cls, fallback_device):
        if torch.cuda.is_available():
            try:
                return torch.device("cuda", torch.cuda.current_device())
            except Exception:
                return torch.device("cuda")
        return fallback_device

    @classmethod
    def _pid_normalize_color_area_override(cls, value):
        if value is None:
            return ""
        text = str(value).strip().lower().replace("_", " ")
        if text in ("", "preset", "default", "none"):
            return ""
        if "off" in text:
            return "color_match_off"
        if "origin" in text or "original" in text:
            return "color_match_from_origin"
        if "protect" in text:
            return "protect_new_generated_content"
        return ""

    @classmethod
    def _pid_place_color_override_mask(cls, canvas, local_mask, x1, y1, width, height):
        if canvas is None or local_mask is None:
            return
        x1 = int(x1)
        y1 = int(y1)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return
        canvas_h = int(canvas.shape[1])
        canvas_w = int(canvas.shape[2])
        x2 = min(canvas_w, max(0, x1 + width))
        y2 = min(canvas_h, max(0, y1 + height))
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x2 <= x1 or y2 <= y1:
            return
        mask = cls._mask_to_bhw(local_mask)
        if mask is None:
            return
        mask = mask.unsqueeze(-1).to(device=canvas.device, dtype=canvas.dtype).clamp(0.0, 1.0)
        target_h = y2 - y1
        target_w = x2 - x1
        if int(mask.shape[1]) != target_h or int(mask.shape[2]) != target_w:
            mask = torch.nn.functional.interpolate(
                mask.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        canvas[:, y1:y2, x1:x2, :] = torch.maximum(canvas[:, y1:y2, x1:x2, :], mask)

    @classmethod
    def _pid_build_color_area_override_masks_4x(cls, target, check_gate=True):
        if check_gate and not cls._cm_debug_stage_enabled("13_Final_PerArea_SegmentOverrides", True):
            return None
        target = cls._ensure_bhwc_image(target)
        if target is None or target.ndim != 4:
            return None
        overrides = list(getattr(tbg.PROMPTER, "output_color_match_js", []) or [])
        if not overrides:
            return None
        device = target.device
        dtype = target.dtype
        shape = (1, int(target.shape[1]), int(target.shape[2]), 1)
        masks = {
            "origin": torch.zeros(shape, device=device, dtype=dtype),
            "protect": torch.zeros(shape, device=device, dtype=dtype),
            "off": torch.zeros(shape, device=device, dtype=dtype),
        }
        placed = {"origin": 0, "protect": 0, "off": 0, "preset": 0}
        key_by_override = {
            "color_match_from_origin": "origin",
            "protect_new_generated_content": "protect",
            "color_match_off": "off",
        }

        specs = list(getattr(tbg.PARAMS, "grid_specs", None) or [])
        scaled_specs = cls._scaled_grid_specs(specs, PID_SCALE)
        normal_specs = [spec for spec in scaled_specs if not cls._is_segment_spec(spec)]
        for grid_spec in normal_specs:
            try:
                index = int(grid_spec[2])
                if index >= len(overrides):
                    continue
                override = cls._pid_normalize_color_area_override(overrides[index])
                if override == "":
                    placed["preset"] += 1
                    continue
                key = key_by_override.get(override)
                if key is None:
                    continue
                _row, _col, _order, x1, y1, width, height = grid_spec[:7]
                local_mask = torch.ones(
                    (1, max(1, int(height)), max(1, int(width)), 1),
                    device=device,
                    dtype=dtype,
                )
                cls._pid_place_color_override_mask(masks[key], local_mask, int(x1), int(y1), int(width), int(height))
                placed[key] += 1
            except Exception as exc:
                if getattr(tbg.API, "status", None) == "Dev":
                    print(f"TBG[Node {tbg.INFO.id}] PiD final tile color override mask failed: {exc}")

        segment_masks, crop_regions = cls._scaled_pid_segment_rebuild_inputs()
        tile_count = len(normal_specs)
        for index_seg, crop_region in enumerate(crop_regions or []):
            try:
                index_custom = tile_count + index_seg
                if index_custom >= len(overrides):
                    continue
                override = cls._pid_normalize_color_area_override(overrides[index_custom])
                if override == "":
                    placed["preset"] += 1
                    if getattr(tbg.API, "status", None) == "Dev":
                        print(
                            f"TBG[Node {tbg.INFO.id}] PiD segment {index_seg + 1} final color decision: "
                            f"preset -> refiner mode {cls._final_color_mode_label()}"
                        )
                    continue
                key = key_by_override.get(override)
                if key is None:
                    continue
                if index_seg >= len(segment_masks) or segment_masks[index_seg] is None:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in crop_region]
                cls._pid_place_color_override_mask(
                    masks[key],
                    segment_masks[index_seg],
                    x1,
                    y1,
                    max(1, x2 - x1),
                    max(1, y2 - y1),
                )
                placed[key] += 1
                if getattr(tbg.API, "status", None) == "Dev":
                    print(
                        f"TBG[Node {tbg.INFO.id}] PiD segment {index_seg + 1} final color decision: "
                        f"{override}"
                    )
            except Exception as exc:
                if getattr(tbg.API, "status", None) == "Dev":
                    print(f"TBG[Node {tbg.INFO.id}] PiD final segment color override mask failed: {exc}")

        if not any(placed[key] > 0 for key in ("origin", "protect", "off")):
            if getattr(tbg.API, "status", None) == "Dev":
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final color overrides: "
                    f"preset={placed['preset']} origin=0 protect=0 off=0"
                )
            return None
        if getattr(tbg.API, "status", None) == "Dev":
            try:
                for key, mask in masks.items():
                    if placed[key] > 0:
                        cls.debug_image_to_folder(
                            MaskToImage_execute(cls._mask_to_bhw(mask))[0],
                            f"PID_FinalColorMatch_PerArea_{key}_Mask4x",
                        )
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final color overrides: "
                    f"preset={placed['preset']} origin={placed['origin']} "
                    f"protect={placed['protect']} off={placed['off']}"
                )
            except Exception:
                pass
        return masks

    @classmethod
    def _pid_build_final_color_correction_allowed_mask_4x(cls, target):
        masks = cls._pid_build_color_area_override_masks_4x(target)
        if not masks:
            return None
        off_mask = masks["off"].clamp(0.0, 1.0)
        protect_mask = masks["protect"].clamp(0.0, 1.0)
        blocked = torch.maximum(off_mask, protect_mask).clamp(0.0, 1.0)
        if float(blocked.max().detach().cpu()) <= 1e-6:
            return None
        allowed = (1.0 - blocked).clamp(0.0, 1.0)
        if getattr(tbg.API, "status", None) == "Dev":
            try:
                cls.debug_image_to_folder(
                    MaskToImage_execute(cls._mask_to_bhw(allowed))[0],
                    "PID_FinalColorMatch_CorrectionAllowedMask4x",
                )
            except Exception:
                pass
        return allowed

    @classmethod
    def _pid_apply_color_area_overrides_4x(cls, reference, target, method, strength, global_result, full_reference=None):
        masks = cls._pid_build_color_area_override_masks_4x(target)
        if not masks:
            return global_result
        target = cls._ensure_bhwc_image(target)
        reference = cls._ensure_bhwc_image(reference)
        result = cls._ensure_bhwc_image(global_result)
        if target is None or reference is None or result is None:
            return global_result
        result = result.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        reference = reference.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        full_reference = cls._ensure_bhwc_image(full_reference if full_reference is not None else reference)
        if full_reference is None:
            full_reference = reference
        full_reference = full_reference.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

        origin_mask = masks["origin"].to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        protect_mask = masks["protect"].to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        off_mask = masks["off"].to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
        nonstructural = cls._final_mode_is_nonstructural()
        if float(origin_mask.max().detach().cpu()) > 1e-6:
            if nonstructural:
                full, metrics = cls._global_rgb_luma_match(
                    full_reference,
                    target,
                    strength=strength,
                    apply_mask=origin_mask,
                    label="pid_final_per_area_origin_nonstructural",
                )
                cls._log_global_rgb_luma_metrics("pid_final_per_area_origin", metrics)
            else:
                full = TBG_Image.colormatch(full_reference, target, method, strength)[0]
            full = cls._ensure_bhwc_image(full).to(device=result.device, dtype=result.dtype).clamp(0.0, 1.0)
            result = (full * origin_mask + result * (1.0 - origin_mask)).clamp(0.0, 1.0)
        if float(protect_mask.max().detach().cpu()) > 1e-6:
            if nonstructural:
                protected, metrics = cls._global_rgb_luma_match(
                    reference,
                    target,
                    strength=strength,
                    apply_mask=protect_mask,
                    label="pid_final_per_area_protect_nonstructural",
                )
                cls._log_global_rgb_luma_metrics("pid_final_per_area_protect", metrics)
            else:
                protected = TBG_Image.detail_preserving_colormatch(
                    reference,
                    target,
                    method,
                    strength,
                    label="pid_final_per_area_protect",
                )[0]
            protected = cls._ensure_bhwc_image(protected).to(device=result.device, dtype=result.dtype).clamp(0.0, 1.0)
            result = (protected * protect_mask + result * (1.0 - protect_mask)).clamp(0.0, 1.0)
        if float(off_mask.max().detach().cpu()) > 1e-6:
            result = (target * off_mask + result * (1.0 - off_mask)).clamp(0.0, 1.0)
        return result

    @classmethod
    def _pid_has_protected_color_area_override_4x(cls, target):
        masks = cls._pid_build_color_area_override_masks_4x(target, check_gate=False)
        if not masks:
            return False
        protect_mask = masks.get("protect")
        return torch.is_tensor(protect_mask) and float(protect_mask.max().detach().cpu()) > 1e-6

    @classmethod
    def _build_pid_segment_color_base_for_final_match(cls, target_image):
        """Build a 4x reference-only color base for final PiD color match."""
        try:
            if not cls._cm_debug_stage_enabled(
                "11_Final_SegmentAware_ColorBase",
                cls._final_color_correction_enabled(),
                requires_method=not cls._final_mode_is_nonstructural(),
            ):
                return None
            if target_image is None or not torch.is_tensor(target_image):
                return None
            target = cls._ensure_bhwc_image(target_image)
            if target is None or target.ndim != 4:
                return None
            masks, crop_regions = cls._scaled_pid_segment_rebuild_inputs()
            if not masks or not crop_regions:
                return None

            target_w = int(target.shape[2])
            target_h = int(target.shape[1])
            reference = nodes.ImageScale().upscale(
                tbg.OUTPUTS.upscaled_image,
                "bilinear",
                target_w,
                target_h,
                False,
            )[0]
            reference = cls._ensure_bhwc_image(reference)
            if reference is None or reference.ndim != 4:
                return None
            color_base = torch.nan_to_num(
                reference.to(device=target.device, dtype=torch.float32).clone(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            pid_tiles = list(getattr(tbg.TEMP, "pid_grid_images_4x", None) or [])
            if not pid_tiles:
                return None
            pid_tiles = cls._pid_tiles_with_native_segment_crops(pid_tiles, crop_regions)
            specs = getattr(tbg.PARAMS, "grid_specs", None) or []
            tile_count = len([spec for spec in specs if not cls._is_segment_spec(spec)])
            overrides = list(getattr(tbg.PROMPTER, "output_color_match_js", []) or [])
            full_mask = torch.zeros_like(color_base[..., :1])
            placed = 0

            for index_seg, crop_region in enumerate(crop_regions):
                index_custom = tile_count + index_seg
                if index_custom >= len(overrides):
                    continue
                override = cls._pid_normalize_color_area_override(overrides[index_custom])
                if override != "color_match_off":
                    continue
                if index_custom >= len(pid_tiles):
                    continue
                if index_seg >= len(masks) or masks[index_seg] is None:
                    continue
                segment = cls._ensure_bhwc_image(pid_tiles[index_custom])
                if segment is None or segment.ndim != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in crop_region]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                if x1 < 0 or y1 < 0 or x1 + width > target_w or y1 + height > target_h:
                    continue
                segment = segment.to(device=color_base.device, dtype=color_base.dtype)
                if int(segment.shape[1]) != height or int(segment.shape[2]) != width:
                    segment = torch.nn.functional.interpolate(
                        segment.permute(0, 3, 1, 2),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
                mask = cls._pid_final_color_base_mask_to_bhwc(
                    masks[index_seg],
                    height,
                    width,
                    color_base.device,
                    color_base.dtype,
                )
                if mask is None:
                    continue
                region = color_base[:, y1:y1 + height, x1:x1 + width, :]
                color_base[:, y1:y1 + height, x1:x1 + width, :] = (
                    region * (1.0 - mask) + segment * mask
                ).clamp(0.0, 1.0)
                full_mask[:, y1:y1 + height, x1:x1 + width, :] = torch.maximum(
                    full_mask[:, y1:y1 + height, x1:x1 + width, :],
                    mask,
                )
                placed += 1

            if placed <= 0:
                return None
            if getattr(tbg.API, "status", None) == "Dev":
                cls.debug_image_to_folder(reference, "Final_ColorMatch_Base_Input")
                cls.debug_image_to_folder(color_base, "PID_FinalColorMatch_Reference_WithColorMatchOffAreas")
                cls.debug_image_to_folder(MaskToImage_execute(cls._mask_to_bhw(full_mask))[0], "PID_FinalColorMatch_ColorMatchOffReferenceMask4x")
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final segment-aware color base built "
                    f"with {placed} explicit Color Match Off 4x segment crops."
                )
            return color_base
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final segment-aware color base failed, "
                    f"using original final reference: {exc}"
                )
            return None

    @classmethod
    def _apply_pid_segment_final_color_match(cls, output_image):
        if not cls._cm_debug_stage_enabled(
            "12_PID_Final_ColorMatch_4x",
            cls._final_color_correction_enabled(),
            requires_method=not cls._final_mode_is_nonstructural(),
        ):
            return output_image
        method = getattr(tbg.PARAMS, "color_match_method", "none")
        strength = float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)
        original_device = output_image.device
        original_dtype = output_image.dtype
        work_device = cls._pid_color_work_device(original_device)
        try:
            work_output = output_image.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
        except Exception:
            work_output = output_image.to(torch.float32).clamp(0.0, 1.0)
        color_base = cls._build_pid_segment_color_base_for_final_match(work_output)
        reference_label = "segment-aware color base"
        if color_base is None:
            try:
                color_base = nodes.ImageScale().upscale(
                    tbg.OUTPUTS.upscaled_image,
                    "bilinear",
                    int(work_output.shape[2]),
                    int(work_output.shape[1]),
                    False,
                )[0]
                reference_label = "input/upscaled image"
            except Exception as exc:
                print(f"TBG[Node {tbg.INFO.id}] PiD final input-reference color match failed to build reference: {exc}")
                return output_image
        color_base = cls._ensure_bhwc_image(color_base).to(device=work_output.device, dtype=work_output.dtype).clamp(0.0, 1.0)
        full_reference = nodes.ImageScale().upscale(
            tbg.OUTPUTS.upscaled_image,
            "bilinear",
            int(work_output.shape[2]),
            int(work_output.shape[1]),
            False,
        )[0]
        full_reference = cls._ensure_bhwc_image(full_reference).to(device=work_output.device, dtype=work_output.dtype).clamp(0.0, 1.0)
        before = work_output
        try:
            protected_override_active = cls._pid_has_protected_color_area_override_4x(work_output)
            use_protected_final_stage = cls._final_mode_is_protect() or protected_override_active
            final_stage_key = "13_Final_PerArea_SegmentOverrides" if use_protected_final_stage else "14_Final_Global_ColorMode"
            if not cls._cm_debug_stage_enabled(
                final_stage_key,
                cls._final_color_correction_enabled(),
                requires_method=not cls._final_mode_is_nonstructural(),
            ):
                return output_image
            mode_stage = "13_protect_per_area" if use_protected_final_stage else "14_full_global"
            if use_protected_final_stage:
                corrected = cls._pid_apply_color_area_overrides_4x(
                    color_base,
                    work_output,
                    method,
                    strength,
                    work_output,
                    full_reference=full_reference,
                )
            else:
                corrected = cls._apply_color_match_by_mode(
                    color_base,
                    work_output,
                    method,
                    strength,
                    label=f"pid_final_{reference_label}",
                )[0]
                corrected = cls._ensure_bhwc_image(corrected).to(device=work_output.device, dtype=work_output.dtype).clamp(0.0, 1.0)
            corrected = corrected.to(device=original_device, dtype=original_dtype).clamp(0.0, 1.0)
            if getattr(tbg.API, "status", None) == "Dev":
                delta = torch.mean(torch.abs(corrected.to(torch.float32) - output_image.to(torch.float32))).item()
                reason = "protected_override" if protected_override_active and not cls._final_mode_is_protect() else ("protect_mode" if cls._final_mode_is_protect() else "full_mode")
                print(
                    f"TBG[Node {tbg.INFO.id}] PiD final color match uses {reference_label}: "
                    f"method={method} mode={cls._final_color_mode_label()} strength={strength} "
                    f"stage={mode_stage} reason={reason} "
                    f"device={before.device} "
                    f"mean_abs_delta={delta:.8f}"
                )
            return corrected
        except Exception as exc:
            print(f"TBG[Node {tbg.INFO.id}] PiD final color match failed using {reference_label}: {exc}")
            return output_image

    @classmethod
    def rebuild_pid_refiner_output_4x(cls, output_image_new, output_image_only_tiles, output_image_noCC):
        pid_tiles = getattr(tbg.TEMP, "pid_grid_images_4x", None) or []
        if not pid_tiles:
            raise RuntimeError("TBG PID refiner did not produce 4x PID tiles for final rebuild.")

        scaled_specs = cls._scaled_grid_specs(tbg.PARAMS.grid_specs, PID_SCALE)
        segment_indices = set()
        try:
            specs = getattr(tbg.PARAMS, "grid_specs", None) or []
            grid_spec_tile = [spec for spec in specs if not cls._is_segment_spec(spec)]
            grid_spec_custom = [spec for spec in specs if cls._is_segment_spec(spec)]
            segment_indices = {len(grid_spec_tile) + index_seg for index_seg, _ in enumerate(grid_spec_custom)}
        except Exception:
            segment_indices = set()
        for i, tile in enumerate(pid_tiles):
            if tile is not None:
                continue
            if i in segment_indices:
                continue
            if i >= len(tbg.OUTPUTS.grid_images_all) or i >= len(scaled_specs):
                raise RuntimeError("TBG PID refiner missing a 4x tile and no matching 1x fallback tile exists.")
            _, _, _, _, _, tile_w, tile_h = scaled_specs[i]
            pid_tiles[i] = nodes.ImageScale().upscale(
                tbg.OUTPUTS.grid_images_all[i],
                "lanczos",
                int(tile_w),
                int(tile_h),
                False,
            )[0]
            print(f"TBG[Node {tbg.INFO.id}] PID 4x final rebuild: upscaled cached tile {i + 1} fallback.")
        target_width = int(tbg.OUTPUTS.upscaled_image.shape[2]) * PID_SCALE
        target_height = int(tbg.OUTPUTS.upscaled_image.shape[1]) * PID_SCALE
        print(
            f"TBG[Node {tbg.INFO.id}] PID 4x rebuild canvas: "
            f"{int(tbg.OUTPUTS.upscaled_image.shape[2])}x{int(tbg.OUTPUTS.upscaled_image.shape[1])} "
            f"-> {target_width}x{target_height}"
        )
        reference = nodes.ImageScale().upscale(tbg.OUTPUTS.upscaled_image, "bilinear", target_width, target_height, False)[0]
        scaled_params = SimpleNamespace(**vars(tbg.PARAMS))
        scaled_params.Redux_Style_Model = None
        scaled_params.Redux_Clip_Vision = None
        scaled_params.grid_specs = scaled_specs
        scaled_params.grid_prompts = [""] * len(scaled_specs)
        scaled_params.stitch_blending = "gpupyramid"
        scaled_params.pid_vae_decode = True
        scaled_params.pid_vae_segment_direct_composite = True
        scaled_params.model_type = str(getattr(tbg.KSAMPLER, "model_type", "") or "")
        scaled_params.denoise_mask = cls._scale_pid_mask(getattr(tbg.PARAMS, "denoise_mask", None), target_width, target_height)
        scaled_size = cls._scale_refiner_size_for_pid(tbg.SIZE)
        scaled_size.overlay_between_tiles = int(getattr(tbg.SIZE, "overlay_between_tiles", 0) or 0) * PID_SCALE
        scaled_size.composite_blur_margin = int(getattr(tbg.SIZE, "composite_blur_margin", 16) or 16) * PID_SCALE
        scaled_size.inpaint_blur_margin = int(getattr(tbg.SIZE, "inpaint_blur_margin", 64) or 64) * PID_SCALE
        scaled_size.inpaint_border_margin = int(getattr(tbg.SIZE, "inpaint_border_margin", 0) or 0) * PID_SCALE
        grid_prompts = [""] * len(scaled_specs)
        compositing_masks, crop_regions = cls._scaled_pid_segment_rebuild_inputs()
        scaled_params.pid_segment_binary_masks = cls._scaled_pid_segment_binary_masks(crop_regions)
        pid_tiles = cls._pid_tiles_with_native_segment_crops(pid_tiles, crop_regions)
        fusion_mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
        fusion_modes = {"Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion"}
        native_nosegments = False if fusion_mode in fusion_modes else not bool(crop_regions)
        try:
            gpu_result = cls._rebuild_pid_refiner_output_4x_gpu(
                pid_tiles,
                scaled_specs,
                scaled_params,
                scaled_size,
                reference,
                compositing_masks,
                crop_regions,
                native_nosegments,
                target_width,
                target_height,
            )
            if gpu_result is not None:
                return gpu_result
        except Exception as exc:
            print(
                f"TBG[Node {tbg.INFO.id}] PID 4x GPU final rebuild failed; "
                f"falling back to CPU worker compositor: {exc}"
            )

        scaled_params = cls._worker_cpu_value(scaled_params)
        scaled_size = cls._worker_cpu_value(scaled_size)
        compositing_masks = cls._worker_cpu_value(compositing_masks)
        crop_regions = cls._worker_cpu_value(crop_regions)
        print(
            f"TBG[Node {tbg.INFO.id}] PID 4x worker args sanitized to CPU: "
            f"denoise_mask={getattr(getattr(scaled_params, 'denoise_mask', None), 'shape', None)} "
            f"compositing_masks={[getattr(mask, 'shape', None) for mask in compositing_masks]}"
        )

        previous_grid_images = getattr(tbg.OUTPUTS, "grid_images_all", None)
        previous_orig_grid_images = getattr(tbg.OUTPUTS, "orig_grid_images_all", None)
        previous_upscaled_image = getattr(tbg.OUTPUTS, "upscaled_image", None)
        previous_last_final_image = getattr(tbg.OUTPUTS, "last_final_image", None)
        previous_input_image = getattr(tbg.INPUTS, "image", None)
        previous_segment_mask = getattr(tbg.SEGMENTS, "Segment_Mask", None)
        previous_output_prompts = getattr(tbg.PROMPTER, "output_prompts", None)
        try:
            tbg.OUTPUTS.grid_images_all = list(pid_tiles)
            tbg.OUTPUTS.orig_grid_images_all = list(pid_tiles)
            tbg.OUTPUTS.upscaled_image = reference
            tbg.OUTPUTS.last_final_image = None
            if torch.is_tensor(previous_input_image):
                tbg.INPUTS.image = nodes.ImageScale().upscale(previous_input_image, "bilinear", target_width, target_height, False)[0]
            if torch.is_tensor(previous_segment_mask):
                tbg.SEGMENTS.Segment_Mask = cls._scale_pid_mask(previous_segment_mask, target_width, target_height)
            tbg.PROMPTER.output_prompts = [""] * len(scaled_specs)
            tile_only_rebuilt, tile_only_canvas, tile_only_no_cc = WORKER.id(tiler_id).TBG_PIDWorkerRebuild.rebuild_final_image_with_state(
                scaled_params,
                scaled_size,
                grid_prompts,
                compositing_masks,
                crop_regions,
                nosegments=True,
                full_image_only_tiles=None,
            )
            if torch.is_tensor(tile_only_rebuilt) and tile_only_rebuilt.ndim == 3:
                tile_only_rebuilt = tile_only_rebuilt.unsqueeze(0)
            if torch.is_tensor(tile_only_canvas) and tile_only_canvas.ndim == 3:
                tile_only_canvas = tile_only_canvas.unsqueeze(0)
            if tile_only_canvas is None:
                tile_only_canvas = tile_only_rebuilt
            if tbg.API.status == "Dev" and torch.is_tensor(tile_only_canvas):
                cls.debug_image_to_folder(tile_only_canvas, "PID_FinalRebuild_TileOnlyCanvas4x")
            print(
                f"TBG[Node {tbg.INFO.id}] PID 4x worker tile-only rebuild completed before segments: "
                f"{target_width}x{target_height}"
            )
            rebuilt, only_tiles, no_cc = WORKER.id(tiler_id).TBG_PIDWorkerRebuild.rebuild_final_image_with_state(
                scaled_params,
                scaled_size,
                grid_prompts,
                compositing_masks,
                crop_regions,
                nosegments=native_nosegments,
                full_image_only_tiles=tile_only_canvas,
            )
            if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
                rebuilt = rebuilt.unsqueeze(0)
            if torch.is_tensor(only_tiles) and only_tiles.ndim == 3:
                only_tiles = only_tiles.unsqueeze(0)
            if torch.is_tensor(no_cc) and no_cc.ndim == 3:
                no_cc = no_cc.unsqueeze(0)
            if tbg.API.status == "Dev" and torch.is_tensor(rebuilt):
                cls.debug_image_to_folder(rebuilt, "PID_FinalRebuild_AfterSegments4x")
            print(
                f"TBG[Node {tbg.INFO.id}] PID 4x worker segment composite completed from tile-only canvas: "
                f"{target_width}x{target_height}"
            )
            return rebuilt, tile_only_canvas if tile_only_canvas is not None else (only_tiles if only_tiles is not None else rebuilt), no_cc if no_cc is not None else rebuilt
        finally:
            tbg.OUTPUTS.grid_images_all = previous_grid_images
            tbg.OUTPUTS.orig_grid_images_all = previous_orig_grid_images
            tbg.OUTPUTS.upscaled_image = previous_upscaled_image
            tbg.OUTPUTS.last_final_image = previous_last_final_image
            tbg.INPUTS.image = previous_input_image
            tbg.SEGMENTS.Segment_Mask = previous_segment_mask
            tbg.PROMPTER.output_prompts = previous_output_prompts

    @classmethod
    def _sanitize_tile_override_model_registry(cls):
        models = getattr(tbg.PROMPTER, "model_overrides", None)
        if not models:
            return
        try:
            from .TBG_Pipes import TILE_MODEL_OVERRIDE_REGISTRY
            key = getattr(tbg.PROMPTER, "model_override_key", None)
            if key is None:
                cache_key = str(getattr(tbg.PROMPTER, "cache_key", "") or "")
                key = cache_key.replace("tile_edits_json_", "") if cache_key.startswith("tile_edits_json_") else str(getattr(tbg.INFO, "id", ""))
            TILE_MODEL_OVERRIDE_REGISTRY[str(key)] = list(models)
            tbg.PROMPTER.model_override_key = str(key)
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] failed to sanitize model override registry: {exc}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
        finally:
            tbg.PROMPTER.model_overrides = None

    @classmethod
    def _sanitize_tile_override_cnetpipe_registry(cls):
        cnetpipes = getattr(tbg.PROMPTER, "cnetpipe_overrides", None)
        if not cnetpipes:
            return
        try:
            from .TBG_Pipes import TILE_CNETPIPE_OVERRIDE_REGISTRY
            key = getattr(tbg.PROMPTER, "cnetpipe_override_key", None)
            if key is None:
                cache_key = str(getattr(tbg.PROMPTER, "cache_key", "") or "")
                key = cache_key.replace("tile_edits_json_", "") if cache_key.startswith("tile_edits_json_") else str(getattr(tbg.INFO, "id", ""))
            TILE_CNETPIPE_OVERRIDE_REGISTRY[str(key)] = list(cnetpipes)
            tbg.PROMPTER.cnetpipe_override_key = str(key)
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] failed to sanitize cnetpipe override registry: {exc}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
        finally:
            tbg.PROMPTER.cnetpipe_overrides = None

    @classmethod
    def _resolve_tile_model_override(cls, index, default_model):
        choices = list(getattr(tbg.PROMPTER, "output_model_js", []) or [])
        raw_choice = choices[index] if index < len(choices) else ""
        choice = str(raw_choice or "").strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "model1": "model 1",
            "1": "model 1",
            "model2": "model 2",
            "2": "model 2",
            "model3": "model 3",
            "3": "model 3",
        }
        choice = aliases.get(choice, choice)
        if choice in ("", "normal", "default", "none"):
            return default_model, "normal"

        override_index = {"model 1": 0, "model 2": 1, "model 3": 2}.get(choice)
        models = cls._get_tile_override_models()
        if override_index is None or override_index >= len(models) or models[override_index] is None:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] tile {index + 1} requested {raw_choice!r} but no model input is connected; using normal model",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
            return default_model, "normal"

        return models[override_index], f"model {override_index + 1}"

    @classmethod
    def _get_tile_override_models(cls):
        models = list(getattr(tbg.PROMPTER, "model_overrides", []) or [])
        if models:
            return models
        key = getattr(tbg.PROMPTER, "model_override_key", None)
        if key is None:
            return []
        try:
            from .TBG_Pipes import TILE_MODEL_OVERRIDE_REGISTRY
            return list(TILE_MODEL_OVERRIDE_REGISTRY.get(str(key), []) or [])
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] model registry lookup failed key={key!r}: {exc}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
            return []

    @classmethod
    def _resolve_tile_cfg_override(cls, index):
        cfgs = list(getattr(tbg.PROMPTER, "output_cfg_js", []) or [])
        if index < len(cfgs) and cfgs[index] not in ("", None):
            try:
                return float(cfgs[index])
            except Exception:
                pass
        return tbg.KSAMPLER.cfg

    @classmethod
    def _resolve_tile_cnetpipe_override(cls, index, default_cnetpipe):
        choices = list(getattr(tbg.PROMPTER, "output_cnetpipe_js", []) or [])
        raw_choice = choices[index] if index < len(choices) else ""
        choice = str(raw_choice or "").strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "cnetpipe1": "cnetpipe 1",
            "cnet pipe 1": "cnetpipe 1",
            "controlnet pipe 1": "cnetpipe 1",
            "1": "cnetpipe 1",
            "cnetpipe2": "cnetpipe 2",
            "cnet pipe 2": "cnetpipe 2",
            "controlnet pipe 2": "cnetpipe 2",
            "2": "cnetpipe 2",
            "cnetpipe3": "cnetpipe 3",
            "cnet pipe 3": "cnetpipe 3",
            "controlnet pipe 3": "cnetpipe 3",
            "3": "cnetpipe 3",
        }
        choice = aliases.get(choice, choice)
        if choice in ("", "normal", "default", "none"):
            return default_cnetpipe, "normal"

        override_index = {"cnetpipe 1": 0, "cnetpipe 2": 1, "cnetpipe 3": 2}.get(choice)
        cnetpipes = cls._get_tile_override_cnetpipes()
        if override_index is None or override_index >= len(cnetpipes) or cnetpipes[override_index] is None:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] tile {index + 1} requested {raw_choice!r} but no cnetpipe input is connected; using normal cnetpipe",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
            return default_cnetpipe, "normal"

        return cnetpipes[override_index], f"cnetpipe {override_index + 1}"

    @classmethod
    def _get_tile_override_cnetpipes(cls):
        cnetpipes = list(getattr(tbg.PROMPTER, "cnetpipe_overrides", []) or [])
        if cnetpipes:
            return cnetpipes
        key = getattr(tbg.PROMPTER, "cnetpipe_override_key", None)
        if key is None:
            return []
        try:
            from .TBG_Pipes import TILE_CNETPIPE_OVERRIDE_REGISTRY
            return list(TILE_CNETPIPE_OVERRIDE_REGISTRY.get(str(key), []) or [])
        except Exception as exc:
            if getattr(tbg.API, "status", None) == "Dev":
                log(
                    f"[TBG TileOverride] cnetpipe registry lookup failed key={key!r}: {exc}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )
            return []

    @classmethod
    def sampling(cls, index, index_seg, tile_to_process, innerloop_scale_factor, inpaintmask, Complexity_Mask, tiler_id, border_correction_mask, segment_inpainting_mask=None, segment_compositing_mask=None): #pre_border_correction_mask,accumulating_background_image):
                tbg = get_tbg(tiler_id)
                if index_seg is not None and int(index_seg) >= 0:
                    for attr, value in (
                        ("inpainting_mask", segment_inpainting_mask),
                        ("compositing_mask", segment_compositing_mask),
                    ):
                        if value is None:
                            continue
                        values = getattr(tbg.SEGMENTS, attr, None)
                        if not isinstance(values, list):
                            values = list(values) if isinstance(values, tuple) else []
                        while len(values) <= int(index_seg):
                            values.append(None)
                        values[int(index_seg)] = value
                        setattr(tbg.SEGMENTS, attr, values)
                with ((torch.inference_mode(True))):
                    if tbg.API.status == "Dev":
                        cls.debug_image_to_folder(tile_to_process, str(index) + "Tile before Sampling")
                        cls.debug_image_to_folder(MaskToImage_execute(inpaintmask)[0], str(index) + "Inpaint Mask before Sampling")
                        cls.debug_image_to_folder(MaskToImage_execute(Complexity_Mask)[0], str(index) + "Complexity_Mask before Sampling"  )


                    if tbg.API.status == "Dev":
                        cls.debug_image_to_folder(tile_to_process,
                                                  str(index) + "tile_to_process-post_tile_fusion_input")



                    iteration = "TBG-ETUR"
                    # ------------------------------------------------------------------
                    # 3.4  Inner Upscale
                    # ------------------------------------------------------------------

                    tile_to_process_H = tile_to_process.shape[1]
                    tile_to_process_W = tile_to_process.shape[2]

                    if tbg.PARAMS.inner_Upscale_type == 'finer details' and innerloop_scale_factor not in (0, 1):
                        tile_to_process = TBG_Image().helper_upscaleimage(tile_to_process, tbg.PARAMS.upscale_method_inpainting, tbg.PARAMS.upscale_model_inpainting,innerloop_scale_factor)

                    if tbg.API.status == "Dev":
                        cls.debug_image_to_folder(tile_to_process, str(index) + "_Sampler_Input_Image")
                        cls.debug_image_to_folder(MaskToImage_execute(inpaintmask)[0], str(index) + "_Sampler_Inpaint_Mask")
                        cls.debug_image_to_folder(MaskToImage_execute(Complexity_Mask)[0], str(index) + "_Sampler_Complexity_Mask")
                        cls.debug_image_to_folder(MaskToImage_execute(border_correction_mask)[0], str(index) + "_Sampler_Border_Correction_Mask")
                        cls.debug_image_to_folder(
                            MaskToImage_execute((1.0 - border_correction_mask).clamp(0.0, 1.0))[0],
                            str(index) + "_Sampler_Border_Edit_Mask",
                        )

                    flux2_encode_tile = tile_to_process

                    # ------------------------------------------------------------------
                    # 3.5 Sigmas
                    # ------------------------------------------------------------------

                    denoise, sigmas = cls.sigmas(iteration, index)

                    # ------------------------------------------------------------------
                    # 3.5 Conditioning positive negative / cropped positive negative
                    # ------------------------------------------------------------------

                    base_cnetpipe = tbg.KSAMPLER.Controlnet_Pipe
                    selected_cnetpipe, selected_cnetpipe_label = cls._resolve_tile_cnetpipe_override(index, base_cnetpipe)
                    tbg.KSAMPLER.Controlnet_Pipe = selected_cnetpipe
                    if tbg.API.status == "Dev":
                        log(
                            f"[TBG TileOverride] tile {index + 1} cnetpipe={selected_cnetpipe_label}",
                            None,
                            None,
                            f"Node {tbg.INFO.id}"
                        )

                    positive, negative, pos_low, neg_low =  cls.conditioning(tbg.OUTPUTS.upscaled_image, index, tile_to_process)

                    # ------------------------------------------------------------------
                    # 3.6 Inpaint Mask
                    # ------------------------------------------------------------------

                    #inpaintmask = cls.get_inpaintmask(iteration, index, index_seg, innerloop_scale_factor, tile_to_process_H, tile_to_process_W)
                    #Complexity_Mask = cls.Pro_Fusion_Complexity_Mask(index, inpaintmask, tile_to_process)

                    #tbg.DUALMODEL.inpaintmask = inpaintmask
                    #tbg.DUALMODEL.Complexity_Mask = Complexity_Mask

                    # The inpaint mask is the Border Fusion and the Complexity_Mask the denoise for final color correction on borders do not use complexity

                    # ------------------------------------------------------------------
                    # 3.6  VAE encode Inpaint
                    # ------------------------------------------------------------------

                    flux2_hook_active = False
                    flux2_direct_sampler_selected = False

                    if (
                            # First condition block Neuro_Generative_Tile_Fusion Always
                            (
                                #tbg.SEGMENTS.segms and
                                #len(tbg.SEGMENTS.segms[0]) and
                                #len(tbg.OUTPUTS.orig_grid_images) - index > 0 and # only tiles
                                tbg.PARAMS.Tile_Fusion_Mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion") and
                                tbg.API.status in ["Free", "Pro", "Premium", "Unlimited", "Dev"]
                            )
                            # OR second condition block Soft Merge only if mask are added
                            or (
                                #tbg.SEGMENTS.segms and
                                #len(tbg.SEGMENTS.segms[0]) and
                                #len(tbg.OUTPUTS.orig_grid_images) - index > 0 and  # only tiles
                                tbg.PARAMS.Tile_Fusion_Mode in ('NONE','Soft Merge') and
                                (
                                        tbg.PARAMS.point_grid_image_stabilizer_experimental > 0.001 or
                                        tbg.PARAMS.PRO_Fusion_Complexity_Mask_Strength > 0.001 or
                                        tbg.PARAMS.PRO_Per_Pixel_Denoise_Mask_Strength > 0.001
                                ) and
                                tbg.API.status in ["Free", "Pro", "Premium", "Unlimited", "Dev"]
                            )
                    ):

                        fusion_mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)
                        if fusion_mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext"):
                            if not tbg.PARAMS.Differential_Diffusion and tbg.API.status == "Dev":
                                print(
                                    f"TBG[Node {tbg.INFO.id}] tile {index + 1}: "
                                    f"forcing Differential Diffusion ON for {fusion_mode}"
                                )
                            tbg.PARAMS.Differential_Diffusion = True
                        elif fusion_mode == "Soft Merge":
                            if tbg.PARAMS.Differential_Diffusion and tbg.API.status == "Dev":
                                print(
                                    f"TBG[Node {tbg.INFO.id}] tile {index + 1}: "
                                    "forcing Differential Diffusion OFF for Soft Merge"
                                )
                            tbg.PARAMS.Differential_Diffusion = False

                        is_flux2 = getattr(tbg.KSAMPLER, "model_type", None) == "FLUX2"
                        flux2_hook_requested = bool(getattr(tbg.PARAMS, "Flux2_Sampler_Hook", False))
                        flux2_hook_active = (
                            is_flux2
                            and flux2_hook_requested
                        )
                        flux2_direct_sampler_selected = (
                            flux2_hook_active
                            and tbg.KSAMPLER.sampler_input is None
                            and tbg.KSAMPLER.sampler_name == TBG_FLUX2_SAMPLER_NAME
                        )
                        if flux2_direct_sampler_selected:
                            if tbg.API.status == "Dev":
                                print(
                                    f"[TBG Flux2 Select] path=direct sampler={TBG_FLUX2_SAMPLER_NAME} "
                                    f"tile={index + 1} denoise={denoise}"
                                )
                        elif flux2_hook_active:
                            sampler_label = "input_override" if tbg.KSAMPLER.sampler_input is not None else tbg.KSAMPLER.sampler_name
                            if tbg.API.status == "Dev":
                                print(
                                    f"[TBG Flux2 Select] path=hook sampler={sampler_label} "
                                    f"scheduler={tbg.KSAMPLER.scheduler} tile={index + 1} "
                                    "noise_mask_forwarded=False"
                                )

                        if tbg.PARAMS.Differential_Diffusion and not flux2_hook_active:
                            before_has_dd = "denoise_mask_function" in getattr(tbg.KSAMPLER.model, "model_options", {})
                            tbg.KSAMPLER.model = DifferentialDiffusion_execute(tbg.KSAMPLER.model)[0]
                            after_has_dd = "denoise_mask_function" in getattr(tbg.KSAMPLER.model, "model_options", {})
                            if tbg.API.status == "Dev":
                                print(
                                    f"TBG[Node {tbg.INFO.id}] tile {index + 1}: Differential Diffusion active "
                                    f"before_hook={before_has_dd} after_hook={after_has_dd}"
                                )
                        elif not flux2_hook_active and tbg.API.status == "Dev":
                            print(f"TBG[Node {tbg.INFO.id}] tile {index + 1}: Differential Diffusion disabled")

                        latent_source = "unset"
                        if  tbg.PARAMS.inpaint_conditioning and not flux2_hook_active and not is_flux2:
                            InpaintModelConditioningNode = nodes.InpaintModelConditioning()
                            positive, negative, latent_image = InpaintModelConditioningNode.encode(positive, negative,
                                                                                                       flux2_encode_tile,
                                                                                                       tbg.KSAMPLER.vae,
                                                                                                       Complexity_Mask,
                                                                                                       noise_mask=True)
                            latent_source = "InpaintModelConditioning"

                            if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                                pos_low, neg_low , latent_image = InpaintModelConditioningNode.encode(positive, negative,
                                                                                                       flux2_encode_tile,
                                                                                                       tbg.KSAMPLER.vae,
                                                                                                       tbg.DUALMODEL.Complexity_Mask,
                                                                                                       noise_mask=True)
                                latent_source = "InpaintModelConditioning_dual"
                        else:
                            latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, flux2_encode_tile)[0]
                            if flux2_hook_active:
                                latent_image["_flux2_inpaint_mask"] = Complexity_Mask.reshape((-1, 1, Complexity_Mask.shape[-2], Complexity_Mask.shape[-1]))
                                latent_image["_flux2_differential"] = dict(flux2_differential.DEFAULT_CONFIG)
                                latent_source = "VAEEncode_private_flux2_mask"
                            elif is_flux2:
                                latent_image = nodes.SetLatentNoiseMask().set_mask(latent_image, Complexity_Mask)[0]
                                latent_source = "Flux2_VAEEncode_SetLatentNoiseMask"
                            else:
                                latent_image["noise_mask"] = Complexity_Mask.reshape((-1, 1, Complexity_Mask.shape[-2], Complexity_Mask.shape[-1]))
                                latent_source = "VAEEncode_noise_mask"

                        if is_flux2 and tbg.API.status == "Dev":
                            print(
                                f"[TBG Flux2 Select] tile={index + 1} "
                                f"model_type={getattr(tbg.KSAMPLER, 'model_type', None)} "
                                f"hook_requested={flux2_hook_requested} "
                                f"hook_active={flux2_hook_active} "
                                f"inpaint_conditioning={bool(tbg.PARAMS.inpaint_conditioning)} "
                                f"differential={bool(tbg.PARAMS.Differential_Diffusion)} "
                                f"latent_source={latent_source}"
                            )
                        if (
                            tbg.API.status == "Dev"
                            and bool(getattr(tbg.PARAMS, "tiles_to_process_active", False))
                            and not bool(getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False))
                        ):
                            print(
                                f"[TBG SelectedTiles] tile {index + 1} sampler stage "
                                f"selected_mask_mode=True "
                                f"latent_source={latent_source} "
                                f"flux2={is_flux2} "
                                f"flux2_hook={flux2_hook_active} "
                                f"differential={bool(tbg.PARAMS.Differential_Diffusion)} "
                                f"pid_vae={bool(getattr(tbg.KSAMPLER, 'pid_vae_decode', False))} "
                                f"sift={bool(getattr(tbg.PARAMS, 'sift_drift_correction', False))}"
                            )


                    else:
                        if tbg.KSAMPLER.tiled:
                            latent_image = nodes.VAEEncodeTiled().encode(tbg.KSAMPLER.vae, flux2_encode_tile, tbg.SIZE.tile_size_vae, tbg.SIZE.tile_size_vae // 4, tbg.SIZE.tile_size_vae // 4)[0]
                            latent_source = "VAEEncodeTiled_no_mask"
                        else:
                            latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, flux2_encode_tile)[0]
                            latent_source = "VAEEncode_no_mask"
                        if (
                            tbg.API.status == "Dev"
                            and bool(getattr(tbg.PARAMS, "tiles_to_process_active", False))
                            and not bool(getattr(tbg.PARAMS, "Fast_1_Tile_Preview", False))
                        ):
                            print(
                                f"[TBG SelectedTiles] tile {index + 1} sampler stage "
                                f"selected_mask_mode=True "
                                f"latent_source={latent_source} "
                                "flux2=False "
                                "flux2_hook=False "
                                f"differential={bool(getattr(tbg.PARAMS, 'Differential_Diffusion', False))} "
                                f"pid_vae={bool(getattr(tbg.KSAMPLER, 'pid_vae_decode', False))} "
                                f"sift={bool(getattr(tbg.PARAMS, 'sift_drift_correction', False))}"
                            )


        #-change samplers-------------------------------------------------------------------------------------------------------------------




                    if  tbg.KSAMPLER.Enrichment_Pipe is not None and tbg.KSAMPLER.detail_daemon_active:
                        log("detail daemon activated", None, None, f"Node {tbg.INFO.id}")
                        tbg.KSAMPLER.sampler = DetailDaemonSamplerNode.go(
                            tbg.KSAMPLER.sampler,
                            detail_amount=tbg.KSAMPLER.detail_amount,
                            start=tbg.KSAMPLER.detail_daemon_start,
                            end=tbg.KSAMPLER.detail_daemon_end,
                            bias=tbg.KSAMPLER.detail_daemon_bias,
                            exponent=tbg.KSAMPLER.detail_daemon_exponent,
                            start_offset=tbg.KSAMPLER.detail_daemon_start_offset,
                            end_offset=tbg.KSAMPLER.detail_daemon_end_offset,
                            fade=tbg.KSAMPLER.daemon_fade,
                            smooth=tbg.KSAMPLER.daemon_smooth,
                            cfg_scale_override=tbg.KSAMPLER.daemon_cfg_scale
                        )[0]

                    # Set or Unset Enrichment_pipe Resharpener i noise Injection (noise disabled)
                    if  tbg.KSAMPLER.Enrichment_Pipe is not None or tbg.KSAMPLER.Resharpener_active:
                       #TBGReSharpen.hook(0, latent_image, tbg.KSAMPLER.Resharpener_strength, tbg.KSAMPLER.eta)[0]
                        TBG_DetailEnhancer.hook(None, None, tbg.KSAMPLER.Resharpener_strength, 0, tbg.KSAMPLER.resharpen_start, 0, tbg.KSAMPLER.resharpen_end, 0)
                    else:
                        disable_resharpen()
        # -------------------------------------------------------------------------------------------------------------------
                    if tbg.debug:
                        filename_prefix = "TBG/compareTiles/tile_to_process" + str(index) + "_"
                        preview = nodes.PreviewImage()
                        _ = preview.save_images(tile_to_process, filename_prefix, None, None)['ui']['images']

                
        #-Start sampling-------------------------------------------------------------------------------------------------------------------
                    base_model = tbg.KSAMPLER.model
                    selected_model, selected_label = cls._resolve_tile_model_override(index, base_model)
                    tile_cfg = cls._resolve_tile_cfg_override(index)
                    tbg.KSAMPLER.model = selected_model
                    reference_model = base_model
                    sampling_model = selected_model
                    if tbg.API.status == "Dev":
                        log(
                            f"[TBG TileOverride] tile {index + 1} model={selected_label} cfg={tile_cfg}",
                            None,
                            None,
                            f"Node {tbg.INFO.id}"
                        )
                    cls._debug_save_flux2_sampler_parity(
                        index,
                        tbg.KSAMPLER.vae,
                        latent_image,
                        latent_source,
                        positive,
                        negative,
                        tile_cfg,
                        denoise,
                        flux2_encode_tile,
                    )
                    # Install the ControlNet-like reference pull only for the new reference_mode.
                    sampling_model = apply_reference_mode_hooks(
                        tbg,
                        sampling_model,
                        tbg.KSAMPLER.Controlnet_Pipe,
                        tile_to_process,
                        tbg.KSAMPLER.vae,
                        index,
                        latent_image,
                    )
                    sampling_model = cls._apply_rf_untwisting_rope_for_tile(
                        sampling_model,
                        latent_image,
                        positive,
                        index,
                        latent_source,
                    )
                    if (
                        getattr(tbg.API, "status", "") == "Dev"
                        and isinstance(getattr(tbg.KSAMPLER, "RF_UntwistingRoPE", None), dict)
                        and bool(getattr(tbg.KSAMPLER, "RF_UntwistingRoPE", {}).get("enabled", False))
                        and not bool(getattr(tbg.PARAMS, "RF_UntwistingRoPE_runtime_disabled", False))
                    ):
                        print(f"TBG[Node {tbg.INFO.id}] RF UntwistingRoPE active for normal tile sampler only.")
                    #saveguard
                    if tbg.PARAMS.LanPaint:
                        TBG_DualModelSampler = TBG_DualModelSampler_lanpaint,
                        TBG_KSamplerAdvancedSplitAware = TBG_KSamplerAdvancedSplitAware_lanpaint
                    else:
                        TBG_DualModelSampler = TBG_DualModelSampler_normal
                        TBG_KSamplerAdvancedSplitAware = TBG_KSamplerAdvancedSplitAware_normal
                    if denoise != 0:
                        if flux2_direct_sampler_selected:
                            latent_output = sample_flux2_direct(
                                model=sampling_model,
                                positive=positive,
                                negative=negative,
                                pixels=flux2_encode_tile,
                                vae=tbg.KSAMPLER.vae,
                                mask=Complexity_Mask,
                                steps=tbg.KSAMPLER.steps,
                                seed=tbg.PROMPTER.output_seeds_js[index],
                                cfg=tile_cfg,
                                denoise=denoise,
                                base_shift=float(flux2_differential.DEFAULT_CONFIG["base_shift"]),
                                max_shift=float(flux2_differential.DEFAULT_CONFIG["max_shift"]),
                                transition_width=float(flux2_differential.DEFAULT_CONFIG["transition_width"]),
                                mask_gamma=float(flux2_differential.DEFAULT_CONFIG["mask_gamma"]),
                                invert_mask=False,
                                correction_start_sigma=float(flux2_differential.DEFAULT_CONFIG["correction_start_sigma"]),
                                post_composite_preserve=bool(flux2_differential.DEFAULT_CONFIG["post_composite_preserve"]),
                                sigmas=sigmas,
                                denoise_method=tbg.PARAMS.denoise_method,
                            )

                        elif (
                                getattr(tbg.KSAMPLER, "model_type", None) == "Ideogram4"
                                and getattr(tbg.KSAMPLER, "ideogram4_guider", None) is not None
                        ):
                            if tbg.API.status == "Dev":
                                print(
                                    f"[TBG Ideogram4] path=SamplerCustomAdvanced "
                                    f"tile={index + 1} denoise={denoise}"
                                )
                            cls._log_sigma_trace(
                                "custom_sampler_advanced_input",
                                sigmas,
                                tile=index + 1,
                                model_type=getattr(tbg.KSAMPLER, "model_type", None),
                                sampler=str(getattr(tbg.KSAMPLER, "sampler_name", None) or getattr(tbg.KSAMPLER, "sampler", None)),
                            )
                            latent_output = SamplerCustomAdvanced.execute(
                                Noise_RandomNoise(tbg.PROMPTER.output_seeds_js[index]),
                                tbg.KSAMPLER.ideogram4_guider,
                                tbg.KSAMPLER.sampler,
                                sigmas,
                                latent_image
                            )[0]

                        elif tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:

                            Dualmodel_Sampler = TBG_DualModelSampler.sample
                            latent_output = Dualmodel_Sampler(0,
                                tbg.DUALMODEL.inpaint_end,
                                tbg.DUALMODEL.smoother_sharper,
                                tbg.DUALMODEL.detail_enhancer,
                                sampling_model,
                                tbg.DUALMODEL.model,
                                tbg.PROMPTER.output_seeds_js[index], #tbg.KSAMPLER.noise_seed,
                                tile_cfg,
                                tile_cfg,
                                positive,
                                negative,
                                pos_low,
                                neg_low,
                                tbg.KSAMPLER.sampler,
                                tbg.KSAMPLER.scheduler,
                                tbg.KSAMPLER.steps,
                                tbg.DUALMODEL.steps,
                                denoise,
                                tbg.DUALMODEL.model_crossover_sigma_strength,
                                1,
                                latent_image
                            )

                            latent_output = latent_output[0]  # Extract dict from tuple


                        else: # if not split sigma That's the standard Sample

                            inpaint_start = 0
                            if tbg.DUALMODEL.inpaint_end == 0:
                               inpaint_end = 10000
                            elif tbg.DUALMODEL.inpaint_end <= -50 or  tbg.KSAMPLER.steps < abs(tbg.DUALMODEL.inpaint_end):
                                inpaint_end = 0
                                inpaint_start = 0
                            else:  # inpaint_end -32 inpaint_steps 5    29 Total - 24 = 5
                               inpaint_end =  tbg.KSAMPLER.steps + tbg.DUALMODEL.inpaint_end  # 29 Total - 24 = 5

                            TBG_KSampler = TBG_KSamplerAdvancedSplitAware().sample
                            result = TBG_KSampler(
                                sampling_model,
                                True, # add noise
                                tbg.PROMPTER.output_seeds_js[index], #tbg.KSAMPLER.noise_seed,
                                tbg.KSAMPLER.steps,
                                tile_cfg,
                                tbg.KSAMPLER.sampler,
                                tbg.KSAMPLER.scheduler,
                                positive,
                                negative,
                                latent_image,
                                0, #start_at_step,
                                tbg.KSAMPLER.steps,
                                denoise, # denoise is the modified or selectig tbg.KSAMPLER.denoise is the general
                                False, #return_with_leftover_noise,
                                inpaint_end,
                                inpaint_start,
                                tbg.DUALMODEL.smoother_sharper,
                                tbg.DUALMODEL.detail_enhancer,
                                sampler_state=None
                            )
                            latent_output = result[0]



                            #latent_output = latent_output[0]  # Extract dict from tuple


            # VAE decode Debug
                    else: # if denoise 0
                        latent_output = latent_image

                    tbg.KSAMPLER.model = reference_model
                    tbg.KSAMPLER.Controlnet_Pipe = base_cnetpipe
                    sampling_model = None

                    #-End sampling-------------------------------------------------------------------------------------------------------------------
                    
        # VAE decode
                    # -VAE to RAM------------------------------------------------------------------------------------------------------------------
                    
                    if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                        vaedecoder = tbg.DUALMODEL.vae
                    else:
                        vaedecoder = tbg.KSAMPLER.vae

                    if (
                        tbg.API.status == "Dev"
                        and (len(tbg.PARAMS.tiles_to_process) == 0 or index in tbg.PARAMS.tiles_to_process)
                        and not getattr(tbg.KSAMPLER, "pid_vae_decode", False)
                    ):
                        cls._debug_save_raw_sampler_decode(index, vaedecoder, latent_output)


                    if len(tbg.PARAMS.tiles_to_process) == 0 or index in tbg.PARAMS.tiles_to_process:
                        # Downscale to Original size


                       # decode all
                        if getattr(tbg.KSAMPLER, "pid_vae_decode", False):
                            if tbg.debug:
                                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None, f"Node {tbg.INFO.id} - PIDVAE4xDecode")
                            pid_vae_active = cls._pid_vae_decode_active()
                            flux2_pid_active = cls._flux2_pid_active()
                            pid_color_match_active = cls._cm_debug_stage_enabled(
                                "04_Flux2_PID_AfterPiDVAE_ColorMatch",
                                cls._pid_color_match_active(),
                            )
                            flux2_normal_vae_color_match_active = cls._flux2_pid_normal_vae_color_match_active()
                            pid_color_method = cls._pid_color_method()
                            segment_pid_active = cls._is_segment_index(index)
                            segment_visible_mask_1x = None
                            segment_source_mask_1x = None
                            segment_edit_mask_1x = None
                            if segment_pid_active:
                                segment_visible_mask_1x = cls._segment_mask(
                                    index_seg,
                                    "composite",
                                    int(tile_to_process.shape[2]),
                                    int(tile_to_process.shape[1]),
                                )
                                segment_source_mask_1x = cls._segment_mask(
                                    index_seg,
                                    "inpaint",
                                    int(tile_to_process.shape[2]),
                                    int(tile_to_process.shape[1]),
                                )
                                segment_edit_mask_1x = cls._segment_mask(
                                    index_seg,
                                    "binary",
                                    int(tile_to_process.shape[2]),
                                    int(tile_to_process.shape[1]),
                                )
                            # PiD is the authoritative VAE in this mode. Do not run a normal
                            # VAE decode for tiles or segments; use the 1x sampler input as the
                            # PiD base/context and let PiD produce the first decoded image at 4x.
                            base_tile_image = tile_to_process.clone()
                            if tbg.API.status == "Dev":
                                debug_name = "_Segment_PID_base_sampler_authority" if segment_pid_active else "_PID_base_sampler_authority"
                                cls.debug_image_to_folder(base_tile_image, str(index) + debug_name)
                            if tbg.API.status == "Dev" and pid_color_match_active:
                                cls.debug_image_to_folder(base_tile_image, str(index) + "_Flux2_PID_sampled_tile_before_correction")
                            if pid_color_match_active and not segment_pid_active:
                                if tbg.API.status == "Dev":
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Flux2 PiD tile {index + 1}: "
                                        "pre-PiD color match skipped; PiD receives only the sampled latent tensor."
                                    )
                            elif (
                                flux2_pid_active
                                and segment_pid_active
                                and not cls._flux2_pid_use_flux1_baseline()
                                and cls._cm_debug_stage_enabled("08_Segment_PostVAE_ColorMatch", True)
                            ):
                                base_tile_image = cls._flux2_pid_segment_context_color_lock(
                                    tile_to_process,
                                    base_tile_image,
                                    segment_source_mask_1x if segment_source_mask_1x is not None else segment_visible_mask_1x,
                                    index,
                                )
                                if tbg.API.status == "Dev":
                                    cls.debug_image_to_folder(base_tile_image, str(index) + "_Flux2_PID_sampled_tile_after_correction")
                            elif flux2_pid_active and segment_pid_active:
                                print(
                                    f"TBG[Node {tbg.INFO.id}] Flux2 PiD tile {index + 1}: "
                                    "using Flux1 baseline segment path; Flux2 context color lock disabled."
                                )
                            elif tbg.API.status == "Dev" and pid_color_match_active:
                                cls.debug_image_to_folder(base_tile_image, str(index) + "_Flux2_PID_sampled_tile_after_correction")
                            source_width_1x = int(tile_to_process.shape[2])
                            source_height_1x = int(tile_to_process.shape[1])
                            if segment_pid_active:
                                print(
                                    f"TBG[Node {tbg.INFO.id}] Segment PiD path active for tile {index + 1}: "
                                    f"source={source_width_1x}x{source_height_1x}, "
                                    f"segment={index_seg + 1}"
                                )
                                if source_width_1x == 1024 and source_height_1x == 1024:
                                    pid_base_context_4x, pid_inpaint_mask_4x, pid_context_blend_mask = cls._build_segment_pid_context_4x(
                                        base_tile_image,
                                        tile_to_process,
                                        index_seg,
                                        color_match_context=False,
                                        debug_prefix=str(index),
                                    )
                                else:
                                    pid_base_context_4x = None
                                    pid_inpaint_mask_4x = None
                                    pid_context_blend_mask = None
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {index + 1}: "
                                        f"{source_width_1x}x{source_height_1x} uses tiled latent PiD decode; "
                                        "skipping segment PiD 4x context/mask rebuild."
                                    )
                                if flux2_pid_active and pid_base_context_4x is not None:
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD context color match skipped for tile {index + 1}: "
                                        "raw segment context stays spatial authority to avoid ghost imprint."
                                    )
                            else:
                                pid_base_context_4x = None
                                pid_inpaint_mask_4x = None
                                pid_context_blend_mask = None
                                if tbg.API.status == "Dev":
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] PiD tile {index + 1}: "
                                        "clean latent-only decode; no PiD inpaint mask, context image, or post-context blend."
                                    )
                            if tbg.API.status == "Dev":
                                cls.debug_image_to_folder(base_tile_image, str(index) + "_PID_base_user_tile")
                                if pid_base_context_4x is not None:
                                    cls.debug_image_to_folder(pid_base_context_4x, str(index) + "_PID_base_context_4x")
                                if pid_inpaint_mask_4x is not None and not segment_pid_active:
                                    cls.debug_image_to_folder(MaskToImage_execute(pid_inpaint_mask_4x)[0], str(index) + "_PID_inpaint_mask_4x")
                                if pid_context_blend_mask is not None and not segment_pid_active:
                                    cls.debug_image_to_folder(MaskToImage_execute(pid_context_blend_mask)[0], str(index) + "_PID_context_blend_mask_4x")
                            pid_source_width = source_width_1x
                            pid_source_height = source_height_1x
                            pid_latent_output = latent_output
                            pid_base_tile_image = None if not segment_pid_active else base_tile_image
                            pid_context_image = None if not segment_pid_active else pid_base_context_4x
                            pid_inpaint_mask = None
                            pid_post_context_mask = None
                            segment_pid_work_crop = None
                            if segment_pid_active and pid_source_width <= 1024 and pid_source_height <= 1024:
                                work_size = 1024
                                if segment_visible_mask_1x is None:
                                    segment_visible_mask_1x = cls._segment_mask(index_seg, "composite", pid_source_width, pid_source_height)
                                segment_transform = cls._segment_sampling_transform(index_seg)
                                transformed_full_work_tile = (
                                    segment_transform is not None
                                    and pid_source_width == work_size
                                    and pid_source_height == work_size
                                    and torch.is_tensor(tile_to_process)
                                    and tile_to_process.ndim == 4
                                )
                                if transformed_full_work_tile:
                                    offset_x = offset_y = work_x0 = work_y0 = 0
                                    work_base_tile = tile_to_process.clone().to(device=base_tile_image.device, dtype=base_tile_image.dtype)
                                    work_context_kind = "segment fusion sampler tile"
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                        "using fused sampler tile as work canvas; no independent PiD background rebuild."
                                    )
                                    if index_seg is not None:
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                            f"segment {int(index_seg) + 1} background may contain previous segments only; "
                                            "current segment is not independently rebuilt into the background."
                                        )
                                else:
                                    work_base_tile, work_geometry = cls._segment_pid_context_work_image(
                                        base_tile_image,
                                        index,
                                        index_seg=index_seg,
                                        visible_mask=segment_source_mask_1x if segment_source_mask_1x is not None else segment_visible_mask_1x,
                                        work_size=work_size,
                                    )
                                    if work_base_tile is not None and work_geometry is not None:
                                        offset_x, offset_y, work_x0, work_y0 = work_geometry
                                        work_context_kind = f"context crop ({work_x0},{work_y0})"
                                    else:
                                        offset_x, offset_y = cls._segment_pid_work_offset(pid_source_width, pid_source_height, work_size)
                                        work_base_tile = cls._segment_pid_work_image(base_tile_image, offset_x, offset_y, work_size)
                                        work_context_kind = "mean color fallback"
                                segment_inpaint_mask_1x = cls._segment_mask(index_seg, "inpaint", pid_source_width, pid_source_height)
                                work_inpaint_mask = cls._segment_pid_work_mask(
                                    segment_inpaint_mask_1x,
                                    pid_source_width,
                                    pid_source_height,
                                    offset_x,
                                    offset_y,
                                    work_size,
                                    device=work_base_tile.device if torch.is_tensor(work_base_tile) else None,
                                    dtype=torch.float32,
                                )
                                work_visible_mask = cls._segment_pid_work_mask(
                                    segment_visible_mask_1x,
                                    pid_source_width,
                                    pid_source_height,
                                    offset_x,
                                    offset_y,
                                    work_size,
                                    device=work_base_tile.device if torch.is_tensor(work_base_tile) else None,
                                    dtype=torch.float32,
                                )
                                if transformed_full_work_tile:
                                    pid_latent_output = latent_output
                                    if tbg.API.status == "Dev":
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                            "using Flux sampled latent as PiD source; fused tile remains PiD base/context."
                                        )
                                        try:
                                            cls._normal_vae_decode_for_pid_color_reference(
                                                vaedecoder,
                                                latent_output,
                                                int(tile_to_process.shape[2]),
                                                int(tile_to_process.shape[1]),
                                                index,
                                                "Segment_Flux_sampled_latent_normal_vae_debug_before_PID",
                                            )
                                        except Exception as exc:
                                            print(
                                                f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                                f"Flux latent normal-VAE forensic debug failed: {exc}"
                                            )
                                else:
                                    pid_latent_output = cls._encode_segment_pid_work_latent(vaedecoder, work_base_tile, latent_output)
                                if pid_latent_output is not None:
                                    pid_source_width = work_size
                                    pid_source_height = work_size
                                    pid_base_tile_image = work_base_tile
                                    pid_context_image = work_base_tile
                                    # Segment PiD is a 4x VAE/refiner decode of the already
                                    # Flux-sampled latent. Segment inpainting masks belong to
                                    # the Flux sampler stage only; passing them into PiD creates
                                    # a blurry band around the segment.
                                    pid_inpaint_mask = None
                                    pid_post_context_mask = None
                                    segment_pid_work_crop = (
                                        int(offset_x) * PID_SCALE,
                                        int(offset_y) * PID_SCALE,
                                        int(tile_to_process.shape[2]) * PID_SCALE,
                                        int(tile_to_process.shape[1]) * PID_SCALE,
                                    )
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                        f"source={int(tile_to_process.shape[2])}x{int(tile_to_process.shape[1])} "
                                        f"at ({offset_x},{offset_y}) inside {work_size}x{work_size} using {work_context_kind}"
                                    )
                                    if tbg.API.status == "Dev":
                                        cls.debug_image_to_folder(work_base_tile, str(index) + "_Segment_PID_work_canvas")
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] Segment PiD work tile {index + 1}: "
                                            "PiD 4x decode runs without inpaint/post-context masks; masks were used only by Flux sampling."
                                        )
                                        if transformed_full_work_tile:
                                            try:
                                                work_diff = (work_base_tile.to(torch.float32) - tile_to_process.to(device=work_base_tile.device, dtype=torch.float32)).abs()
                                                cls.debug_image_to_folder(work_diff.clamp(0.0, 1.0), str(index) + "_Segment_PID_work_canvas_diff")
                                                work_context_4x = nodes.ImageScale().upscale(
                                                    work_base_tile,
                                                    "lanczos",
                                                    int(work_base_tile.shape[2]) * PID_SCALE,
                                                    int(work_base_tile.shape[1]) * PID_SCALE,
                                                    False,
                                                )[0]
                                                cls.debug_image_to_folder(work_context_4x, str(index) + "_Segment_PID_work_context_4x")
                                            except Exception as exc:
                                                print(
                                                    f"TBG[Node {tbg.INFO.id}] Segment PiD work canvas diff debug failed "
                                                    f"for tile {index + 1}: {exc}"
                                                )
                                else:
                                    pid_latent_output = latent_output
                            if segment_pid_active and segment_pid_work_crop is None:
                                pid_latent_output = latent_output
                                pid_source_width = source_width_1x
                                pid_source_height = source_height_1x
                                pid_base_tile_image = base_tile_image
                                pid_context_image = None
                                pid_inpaint_mask = None
                                pid_post_context_mask = None
                                if tbg.API.status == "Dev":
                                    try:
                                        cls._normal_vae_decode_for_pid_color_reference(
                                            vaedecoder,
                                            latent_output,
                                            int(tile_to_process.shape[2]),
                                            int(tile_to_process.shape[1]),
                                            index,
                                            "Segment_Flux_sampled_latent_normal_vae_debug_before_PID",
                                        )
                                    except Exception as exc:
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] Segment PiD tile {index + 1}: "
                                            f"Flux latent normal-VAE forensic debug failed: {exc}"
                                        )
                                if source_width_1x != 1024 or source_height_1x != 1024:
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD tile {index + 1}: "
                                        f"using tiled PiD latent decode for {source_width_1x}x{source_height_1x}; "
                                        "Flux sampler latent is authority and PiD masks are disabled."
                                    )
                            tile_prompt = ""
                            try:
                                ignore_general_prompt = False
                                ignore_flags = getattr(tbg.PROMPTER, "output_ignore_general_prompt_js", []) or []
                                if index < len(ignore_flags):
                                    ignore_general_prompt = bool(ignore_flags[index])
                                tile_prompt = " ".join(
                                    part for part in (
                                        "" if ignore_general_prompt else getattr(tbg.KSAMPLER, "General_Prompt", ""),
                                        tbg.PROMPTER.output_prompts[index] if index < len(tbg.PROMPTER.output_prompts) else "",
                                    )
                                    if part
                                )
                            except Exception:
                                tile_prompt = getattr(tbg.KSAMPLER, "General_Prompt", "")
                            pid_debug_tiles = []
                            pid_debug_callback = None
                            if tbg.API.status == "Dev":
                                def pid_debug_callback(pid_region_index, raw_pid_tile, processed_pid_tile, **debug_images):
                                    pid_debug_tiles.append((
                                        pid_region_index,
                                        raw_pid_tile.detach().to("cpu", copy=True),
                                        processed_pid_tile.detach().to("cpu", copy=True),
                                        {
                                            name: value.detach().to("cpu", copy=True)
                                            for name, value in debug_images.items()
                                            if torch.is_tensor(value)
                                        },
                                    ))
                            pid_color_match_fn = None
                            if pid_color_match_active and tbg.API.status == "Dev":
                                print(
                                    f"TBG[Node {tbg.INFO.id}] PiD post-VAE color correction deferred "
                                    "until the full 4x ETUR tile/segment is rebuilt."
                                )
                            if pid_source_width != 1024 or pid_source_height != 1024:
                                tile_kind = "segment" if segment_pid_active else "tile"
                                print(
                                    f"TBG[Node {tbg.INFO.id}] PiD {tile_kind} {index + 1}: "
                                    f"{pid_source_width}x{pid_source_height} uses tiled latent decode."
                                )
                            pid_sampling_steps = 4
                            if tbg.API.status == "Dev":
                                print(
                                    f"TBG[Node {tbg.INFO.id}] PiD 4x sampler steps for tile {index + 1}: "
                                    f"{pid_sampling_steps}"
                                )
                            pid_runtime = cls._pid_prepare_runtime(
                                pid_latent_output,
                                "pid_sde",
                                "simple",
                                pid_sampling_steps,
                            )
                            try:
                                pid_decode_shift_4x = (0, 0)
                                if segment_pid_active:
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD decode correction for tile {index + 1}: "
                                        "disabled; preserving native PiD geometry before color lock."
                                    )
                                if tbg.API.status == "Dev":
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] PiD clean input tile {index + 1}: "
                                        f"steps={pid_sampling_steps} "
                                        f"base_tile={pid_base_tile_image is not None} "
                                        f"context={pid_context_image is not None} "
                                        f"inpaint_mask={pid_inpaint_mask is not None} "
                                        f"post_context_mask={pid_post_context_mask is not None} "
                                        f"prompt_empty={not bool((tile_prompt if segment_pid_active else '').strip())}"
                                    )
                                tile_sampled_4x = run_pid_refiner_latent_decode(
                                    pid_latent_output,
                                    pid_source_width,
                                    pid_source_height,
                                    base_tile_image=pid_base_tile_image,
                                    base_context_image=pid_context_image,
                                    inpaint_mask=pid_inpaint_mask,
                                    debug_callback=pid_debug_callback,
                                    prompt_text="" if not segment_pid_active else tile_prompt,
                                    seed=tbg.PROMPTER.output_seeds_js[index],
                                    model_type=getattr(tbg.KSAMPLER, "model_type", None),
                                    degrade_sigma=0.1 if not segment_pid_active else getattr(tbg.PARAMS, "PID_degrade_sigma", 0.1),
                                    sampler_name="pid_sde",
                                    scheduler="simple",
                                    steps=pid_sampling_steps,
                                    cfg=1.0,
                                    color_match_fn=pid_color_match_fn,
                                    color_match_method=pid_color_method,
                                    runtime=pid_runtime,
                                    decode_shift_4x=pid_decode_shift_4x,
                                    segment_post_context_preserve=False if segment_pid_active else segment_pid_active,
                                    post_context_mask=pid_post_context_mask,
                                )
                            finally:
                                cls._pid_finish_tile_runtime()
                            if segment_pid_work_crop is not None:
                                tile_sampled_4x = cls._crop_segment_pid_work_4x(tile_sampled_4x, segment_pid_work_crop)
                            pid_width = int(tile_sampled_4x.shape[2])
                            pid_height = int(tile_sampled_4x.shape[1])
                            tile_processed_4x = tile_sampled_4x.clone()
                            if (
                                    bool(getattr(tbg.PARAMS, "sift_drift_correction", False))
                                    and not segment_pid_active
                            ):
                                try:
                                    sift_reference_4x = nodes.ImageScale().upscale(
                                        tile_to_process,
                                        "lanczos",
                                        pid_width,
                                        pid_height,
                                        False,
                                    )[0]
                                    before_sift_4x = tile_processed_4x
                                    tile_processed_4x, sift_info = cls._apply_sift_drift_correction(
                                        sift_reference_4x,
                                        tile_processed_4x,
                                        index=index,
                                        mode=getattr(tbg.PARAMS, "sift_drift_correction_mode", "x1"),
                                    )
                                    if tbg.API.status == "Dev":
                                        reason = sift_info.get("reason", "unknown")
                                        changed = bool(sift_info.get("changed", False))
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] SIFT drift correction PiD 4x tile {index + 1}: "
                                            f"changed={changed} reason={reason} "
                                            f"matches={sift_info.get('matches', 0)} inliers={sift_info.get('inliers', 0)} "
                                            "stage=before_pid_color"
                                        )
                                        cls.debug_image_to_folder(sift_reference_4x, str(index) + "_SIFT_PID4x_reference_tile")
                                        cls.debug_image_to_folder(before_sift_4x, str(index) + "_SIFT_PID4x_before")
                                        cls.debug_image_to_folder(tile_processed_4x, str(index) + "_SIFT_PID4x_after")
                                except Exception as exc:
                                    if tbg.API.status == "Dev":
                                        print(
                                            f"TBG[Node {tbg.INFO.id}] SIFT drift correction PiD 4x tile {index + 1}: "
                                            f"skipped error={type(exc).__name__}: {exc}"
                                        )
                            flux2_pid_normal_vae_corrected = False
                            if flux2_normal_vae_color_match_active and pid_color_match_active:
                                normal_vae_reference_4x = cls._normal_vae_decode_for_pid_color_reference(
                                    vaedecoder,
                                    pid_latent_output,
                                    pid_width,
                                    pid_height,
                                    index,
                                    "Flux2_PID_normal_vae_color_reference_4x",
                                )
                                if normal_vae_reference_4x is not None:
                                    pid_match_strength = min(
                                        0.35,
                                        max(0.0, float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)),
                                    )
                                    pid_before_normal_vae_match = tile_processed_4x.clone()
                                    tile_processed_4x, global_metrics = cls._global_rgb_luma_match(
                                        normal_vae_reference_4x,
                                        tile_processed_4x,
                                        strength=pid_match_strength,
                                        label=f"normal_vae_to_pid_vae_tile_{index + 1}",
                                    )
                                    if global_metrics is None:
                                        tile_processed_4x = pid_before_normal_vae_match
                                    flux2_pid_normal_vae_corrected = True
                                    cls._debug_pid_detail_retention(
                                        f"normal_vae_color_match_tile_{index + 1}",
                                        pid_before_normal_vae_match,
                                        tile_processed_4x,
                                    )
                                    if tbg.API.status == "Dev":
                                        cls._log_global_rgb_luma_metrics(
                                            f"normal_vae_to_pid_vae_tile_{index + 1}",
                                            global_metrics,
                                        )
                                        cls.debug_image_to_folder(
                                            tile_processed_4x,
                                            str(index) + "_Flux2_PID_color_corrected_from_normal_vae_4x",
                                        )
                            if not segment_pid_active and pid_vae_active:
                                pid_tile_aware_method = cls.is_tbg_tile_aware(getattr(tbg.PARAMS, "color_match_method", None))
                                pid_post_reference = tile_to_process
                                if pid_color_match_active and not pid_tile_aware_method:
                                    try:
                                        pid_global_strength = max(
                                            0.0,
                                            min(1.0, float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)),
                                        )
                                    except Exception:
                                        pid_global_strength = 1.0
                                    before_pid_global = tile_processed_4x.clone()
                                    tile_processed_4x, global_metrics = cls._global_rgb_luma_match(
                                        pid_post_reference,
                                        tile_processed_4x,
                                        strength=pid_global_strength,
                                        label=f"after_decode_tile_{index + 1}",
                                    )
                                    if global_metrics is None:
                                        tile_processed_4x = before_pid_global
                                    else:
                                        cls._debug_pid_detail_retention(
                                            f"after_decode_global_rgb_luma_tile_{index + 1}",
                                            before_pid_global,
                                            tile_processed_4x,
                                        )
                                        if tbg.API.status == "Dev":
                                            cls._log_global_rgb_luma_metrics(
                                                f"after_decode_tile_{index + 1}",
                                                global_metrics,
                                            )
                                            cls.debug_image_to_folder(
                                                tile_processed_4x,
                                                str(index) + "_Flux2_PID_global_rgb_luma_after_decode_4x",
                                            )
                                pid_seam_mask_4x = (1.0 - border_correction_mask).clamp(0.0, 1.0)
                                tile_processed_4x = cls._apply_pid_post_decode_color_stabilizer(
                                    pid_post_reference,
                                    tile_processed_4x,
                                    index,
                                    seam_mask=pid_seam_mask_4x,
                                    label="4x",
                                    apply_global=not pid_color_match_active,
                                )
                            if segment_pid_active:
                                segment_reference_4x = nodes.ImageScale().upscale(
                                    tile_to_process,
                                    "lanczos",
                                    pid_width,
                                    pid_height,
                                    False,
                                )[0]
                                segment_visible_mask_4x = cls._segment_mask(index_seg, "composite", pid_width, pid_height)
                                if segment_visible_mask_4x is not None:
                                    if tbg.API.status == "Dev":
                                        cls.debug_image_to_folder(segment_reference_4x, str(index) + "_Segment_PID_sampler_source_reference_4x")
                                        cls.debug_image_to_folder(MaskToImage_execute(segment_visible_mask_4x)[0], str(index) + "_Segment_PID_visible_mask_4x")
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Segment PiD mask-edge color correction skipped for tile {index + 1}: "
                                        "final tiler compositing mask is the edge authority."
                                    )
                                elif tbg.API.status == "Dev":
                                    print(f"TBG[Node {tbg.INFO.id}] Segment PiD tile {index + 1}: no visible segment mask found for edge color correction.")
                            if tbg.API.status == "Dev":
                                for pid_region_index, raw_pid_tile, processed_pid_tile, debug_images in pid_debug_tiles:
                                    suffix = "" if len(pid_debug_tiles) == 1 else f"_region_{pid_region_index}"
                                    raw_pid_tile = cls._crop_segment_pid_work_4x(raw_pid_tile, segment_pid_work_crop)
                                    processed_pid_tile = cls._crop_segment_pid_work_4x(processed_pid_tile, segment_pid_work_crop)
                                    debug_images = {
                                        name: cls._crop_segment_pid_work_4x(value, segment_pid_work_crop)
                                        for name, value in debug_images.items()
                                    }
                                    cls.debug_image_to_folder(raw_pid_tile, str(index) + "_PID_after_sampler_before_post_context_4x" + suffix)
                                    if "color_reference" in debug_images:
                                        cls.debug_image_to_folder(debug_images["color_reference"], str(index) + "_PID_color_reference_4x" + suffix)
                                    if "color_matched" in debug_images:
                                        cls.debug_image_to_folder(debug_images["color_matched"], str(index) + "_PID_color_matched_4x" + suffix)
                                    if "post_context_matched" in debug_images:
                                        cls.debug_image_to_folder(debug_images["post_context_matched"], str(index) + "_PID_post_context_color_matched_4x" + suffix)
                                    if "post_context_mask" in debug_images:
                                        cls.debug_image_to_folder(MaskToImage_execute(cls._mask_to_bhw(debug_images["post_context_mask"]))[0], str(index) + "_PID_post_context_blend_mask_4x" + suffix)
                                    cls.debug_image_to_folder(processed_pid_tile, str(index) + "_PID_processed_region_4x" + suffix)
                                cls.debug_image_to_folder(tile_processed_4x, str(index) + "_PID_processed_4x")
                            segment_native_crop_4x = None
                            if segment_pid_active:
                                segment_native_crop_4x = cls._restore_segment_pid_4x_to_native_crop(
                                    tile_processed_4x,
                                    index_seg,
                                )
                                segment_native_crop_4x = cls._apply_pid_segment_post_decode_color(
                                    segment_native_crop_4x,
                                    pid_base_tile_image,
                                    index,
                                    index_seg,
                                    pid_source_width,
                                    pid_source_height,
                                    work_crop_4x=segment_pid_work_crop,
                                )
                                if tbg.API.status == "Dev":
                                    cls.debug_image_to_folder(segment_native_crop_4x, str(index) + "_Segment_PID_native_crop_4x_for_reference")
                            if getattr(tbg.TEMP, "pid_grid_images_4x", None) is not None:
                                if segment_pid_active:
                                    tbg.TEMP.pid_grid_images_4x[index] = segment_native_crop_4x
                                    source_meta = getattr(tbg.TEMP, "pid_grid_images_4x_source", None)
                                    if isinstance(source_meta, list) and index < len(source_meta):
                                        transform = cls._segment_sampling_transform(index_seg)
                                        source_meta[index] = {
                                            "source_kind": "pid_restored_native_crop",
                                            "index_seg": int(index_seg),
                                            "segment_case": (transform or {}).get("segment_case"),
                                            "native_crop_region": tuple((transform or {}).get("native_crop_region", ()) or ()),
                                            "sampling_crop_region": tuple((transform or {}).get("sampling_crop_region", ()) or ()),
                                        }
                                        if tbg.API.status == "Dev":
                                            print(
                                                f"TBG[Node {tbg.INFO.id}] PID 4x final source stored: "
                                                f"tile={index + 1} segment={int(index_seg) + 1} "
                                                f"source=pid_restored_native_crop size="
                                                f"{int(segment_native_crop_4x.shape[2])}x{int(segment_native_crop_4x.shape[1])}"
                                            )
                                else:
                                    tbg.TEMP.pid_grid_images_4x[index] = tile_processed_4x
                                    source_meta = getattr(tbg.TEMP, "pid_grid_images_4x_source", None)
                                    if isinstance(source_meta, list) and index < len(source_meta):
                                        source_meta[index] = {
                                            "source_kind": "pid_tile_4x",
                                        }
                            if segment_pid_active:
                                transform = cls._segment_sampling_transform(index_seg)
                                native_crop = transform.get("native_crop_region") if transform is not None else None
                                if native_crop:
                                    target_w = max(1, int(round(float(native_crop[2] - native_crop[0]))))
                                    target_h = max(1, int(round(float(native_crop[3] - native_crop[1]))))
                                else:
                                    target_w = int(tile_to_process.shape[2])
                                    target_h = int(tile_to_process.shape[1])
                                tile_processed = nodes.ImageScale().upscale(
                                    segment_native_crop_4x,
                                    "lanczos",
                                    target_w,
                                    target_h,
                                    False,
                                )[0]
                                print(
                                    f"TBG[Node {tbg.INFO.id}] Segment PiD tile {index + 1}: "
                                    "latent tiled PiD decode complete; returned 1x native crop from "
                                    f"{'normal-VAE-color-corrected ' if flux2_pid_normal_vae_corrected else ''}4x PiD output "
                                    "for sequential reference."
                                )
                                if tbg.API.status == "Dev":
                                    cls.debug_image_to_folder(tile_processed, str(index) + "_Segment_PID_1x_downscaled_native_crop_for_reference")
                            else:
                                tile_processed = nodes.ImageScale().upscale(
                                    tile_processed_4x,
                                    "lanczos",
                                    int(tile_to_process.shape[2]),
                                    int(tile_to_process.shape[1]),
                                    False,
                                )[0]
                                if tbg.API.status == "Dev" and flux2_pid_normal_vae_corrected:
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Flux2 PiD tile {index + 1}: "
                                        "worker tile/stitch source is normal-VAE-color-corrected PiD downscale."
                                    )
                                if cls._cm_debug_stage_enabled("05_Flux2_PID_PostTone_ColorMatch", pid_color_match_active):
                                    seam_mask = (1.0 - border_correction_mask).clamp(0.0, 1.0)
                                    seam_strength = min(
                                        0.35,
                                        max(0.0, float(getattr(tbg.PARAMS, "color_match_str", 1.0) or 1.0)),
                                    )
                                    print(
                                        f"TBG[Node {tbg.INFO.id}] Flux2 PiD tile {index + 1}: "
                                        "replacing final_downscaled_pid_tile color match with seam-local "
                                        f"low-frequency correction strength={seam_strength:.3f}; "
                                        "PiD detail is authority."
                                    )
                                    before_seam_color = tile_processed.clone()
                                    tile_processed = TBG_Image.stabilize_tile_low_frequency_from_reference(
                                        base_tile_image,
                                        tile_processed,
                                        seam_mask,
                                        seam_mask,
                                        seam_strength,
                                    )[0]
                                    cls._debug_pid_detail_retention(
                                        f"final_downscaled_pid_tile_seam_local_{index + 1}",
                                        before_seam_color,
                                        tile_processed,
                                        seam_mask,
                                    )
                                    if tbg.API.status == "Dev":
                                        cls.debug_image_to_folder(
                                            tile_processed,
                                            str(index) + "_PID_tile_processed_seam_color_matched",
                                        )
                        elif tbg.KSAMPLER.tiled:
                            if tbg.debug:
                                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None,f"Node {tbg.INFO.id} - VAEDecodingTiled ")
                            tile_processed = (nodes.VAEDecodeTiled().decode(vaedecoder, latent_output,tbg.SIZE.tile_size_vae,tbg.SIZE.tile_size_vae // 4, tbg.SIZE.tile_size_vae // 4)[0].unsqueeze(0))[0]

                        else:
                            if tbg.debug:
                                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None, f"Node {tbg.INFO.id} - VAEDecodingNormalized")

                            tile_processed = (TBG_Image.VAEDecodeFluxNormalized(vaedecoder, latent_output)[0].unsqueeze(0))[0]

                        if tbg.API.status == "Dev":
                            cls.debug_image_to_folder(tile_processed, str(index)+"_VAE_decode_after_sampling")


                        if tbg.PARAMS.inner_Upscale_type == 'finer details' and innerloop_scale_factor not in (0,1):

                            tile_to_process = TBG_Image().helper_upscaleimage(tile_to_process, tbg.PARAMS.upscale_method_inpainting,
                                                                                     tbg.PARAMS.upscale_model_inpainting,0,round(tile_to_process_W), round(tile_to_process_H))
                            tile_processed = TBG_Image().helper_upscaleimage(tile_processed, round(tile_to_process_W), round(tile_to_process_H),
                                                                                     tbg.PARAMS.upscale_method_inpainting,
                                                                                     tbg.PARAMS.upscale_model_inpainting)

                        if (
                                bool(getattr(tbg.PARAMS, "sift_drift_correction", False))
                                and not bool(getattr(tbg.KSAMPLER, "pid_vae_decode", False))
                        ):
                            before_sift = tile_processed
                            tile_processed, sift_info = cls._apply_sift_drift_correction(
                                tile_to_process,
                                tile_processed,
                                index=index,
                                mode=getattr(tbg.PARAMS, "sift_drift_correction_mode", "x1"),
                            )
                            if tbg.API.status == "Dev":
                                reason = sift_info.get("reason", "unknown")
                                changed = bool(sift_info.get("changed", False))
                                print(
                                    f"TBG[Node {tbg.INFO.id}] SIFT drift correction tile {index + 1}: "
                                    f"changed={changed} reason={reason} "
                                    f"matches={sift_info.get('matches', 0)} inliers={sift_info.get('inliers', 0)}"
                                )
                                if changed:
                                    cls.debug_image_to_folder(tile_to_process, str(index) + "_SIFT_reference_tile")
                                    cls.debug_image_to_folder(before_sift, str(index) + "_SIFT_before")
                                    cls.debug_image_to_folder(tile_processed, str(index) + "_SIFT_after")

                        # the latent en and decode VAE is producing a color even if the fusion is perfect this can produce a seam, so we need to correct this with a simple blend in the border region
                        
                    
                    
                    
                    
                        tile_sampled = tile_processed
                        tile_processed = tile_sampled.clone()
                        
                        if tbg.API.status == "Dev":
                            if tbg.PARAMS.Tile_Fusion_Mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext", "Tile_Fusion") and not tbg.PARAMS.Fast_1_Tile_Preview:
                                border_keep_mask = border_correction_mask
                                border_edit_mask = (1.0 - border_correction_mask).clamp(0.0, 1.0)
                                metric_shapes_match = (
                                    getattr(tile_to_process, "shape", None) is not None
                                    and getattr(tile_sampled, "shape", None) is not None
                                    and getattr(tile_processed, "shape", None) is not None
                                    and tile_to_process.shape[1:3] == tile_sampled.shape[1:3] == tile_processed.shape[1:3]
                                )
                                if metric_shapes_match:
                                    sampled_keep = TBG_Image.masked_mean_abs_diff(tile_to_process, tile_sampled, border_keep_mask)
                                    sampled_edit = TBG_Image.masked_mean_abs_diff(tile_to_process, tile_sampled, border_edit_mask)
                                    final_keep = TBG_Image.masked_mean_abs_diff(tile_to_process, tile_processed, border_keep_mask)
                                    final_edit = TBG_Image.masked_mean_abs_diff(tile_to_process, tile_processed, border_edit_mask)
                                    log(
                                        f"tile {index + 1}: sampled_keep={sampled_keep:.6f} "
                                        f"sampled_edit={sampled_edit:.6f} "
                                        f"final_keep={final_keep:.6f} "
                                        f"final_edit={final_edit:.6f}",
                                        None,
                                        None,
                                        f"Node {tbg.INFO.id}",
                                    )
                                    cls.debug_image_to_folder(tile_sampled, str(index) + "_tile_sampled_before_post_correction")
                                    cls.debug_image_to_folder(MaskToImage_execute(border_keep_mask)[0], str(index) + "_Border_Keep_Mask")
                                    cls.debug_image_to_folder(MaskToImage_execute(border_edit_mask)[0], str(index) + "_Border_Edit_Mask")
                                else:
                                    log(
                                        f"tile {index + 1}: skipped NGTF Dev diff metrics because sampler/input/output spaces differ "
                                        f"input={tuple(tile_to_process.shape) if getattr(tile_to_process, 'shape', None) is not None else None} "
                                        f"sampled={tuple(tile_sampled.shape) if getattr(tile_sampled, 'shape', None) is not None else None} "
                                        f"final={tuple(tile_processed.shape) if getattr(tile_processed, 'shape', None) is not None else None}",
                                        None,
                                        None,
                                        f"Node {tbg.INFO.id}",
                                    )

                                ngtf_debug_masks = getattr(tbg.OUTPUTS, "ngtf_debug_masks", None) or {}
                                if index in (6, 7, 8) and index in ngtf_debug_masks:
                                    seam_masks = ngtf_debug_masks[index]
                                    cls.debug_image_to_folder(
                                        MaskToImage_execute(seam_masks["incoming_copy"])[0],
                                        str(index) + "_Incoming_Copy_Edit_Mask",
                                    )
                                    cls.debug_image_to_folder(
                                        MaskToImage_execute(seam_masks["incoming_blend"])[0],
                                        str(index) + "_Incoming_Blend_Edit_Mask",
                                    )
                                    cls.debug_image_to_folder(
                                        MaskToImage_execute(seam_masks["outgoing_prepare"])[0],
                                        str(index) + "_Outgoing_Prepare_Edit_Mask",
                                    )
                                    cls.debug_image_to_folder(
                                        MaskToImage_execute(seam_masks["post_fix"])[0],
                                        str(index) + "_Post_Fix_Edit_Mask",
                                    )

                            cls.debug_image_to_folder(tile_processed,
                                                      str(index) + "tile_processed")

                            cls.debug_image_to_folder(MaskToImage_execute(border_correction_mask)[0], str(index) + "_Masked_Border_Correction_Mask")

                    #tile_processed = tile_to_process
                    # Save Tile to Temp
                    if tbg.PARAMS.Preview_Tiles_in_Temp_Folder:
                        if tbg.API.status == "Dev":
                            cls.debug_image_to_folder(tile_processed, str(index) + "_Masked_Border_Correction")
                        else:
                            cls.image_to_folder(tile_processed, str(index) + "_CT")


                    tbg.OUTPUTS.grid_images_all[index] = tile_processed
                    storage = persistent_storage[tbg.storage_key]
                    storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.grid_images_all)
                    return tile_processed

# Refiner end --------------------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def _sift_tensor_to_uint8_rgb(cls, image):
        return sift_tensor_to_uint8_rgb(image)

    @classmethod
    def _apply_sift_drift_correction(cls, reference_tile, sampled_tile, index=None, mode="x1"):
        return apply_sift_drift_correction(reference_tile, sampled_tile, index=index, mode=mode)

    @classmethod
    def _detect_changes(cls):
        """
        Detect what changed in the node inputs.
        Returns: (change_type, changed_tile_indices, message)
        """
        storage = persistent_storage.get(tbg.storage_key, {})
        tbg.PARAMS.force_fresh_refiner_background = False

        # Define components to check (excluding PROMPTER)
        components_to_check = {
            'OUTPUTS': {'image': tbg.OUTPUTS.upscaled_image}, # only pass the input form Tiler the upscaled image as DICT
            'KSAMPLER': tbg.KSAMPLER,
            #'DUALMODEL': tbg.DUALMODEL,
            'PARAMS': tbg.PARAMS,
            'SIZE': tbg.SIZE,
            'SEGMENTS': tbg.SEGMENTS,
        }

        # Compute current values and hashes
        current_values = {}
        current_hashes = {}

        for name, component in components_to_check.items():
            try:
                comp_dict = vars(component) if hasattr(component, '__dict__') else {}

                # Filter out tile-specific data
                filtered_dict = {}
                for k, v in comp_dict.items():
                    if k in ['sampler', 'timestamp']: #sampler has on every run a new reference  PARAMS.timestamp:
                        continue
                    if any(k.startswith(prefix) for prefix in ['generated', 'output', 'segms']):


                        continue
                    filtered_dict[k] = v

                current_values[name] = filtered_dict
                current_hashes[name] = hash(str(sorted(filtered_dict.items())))
            except:
                current_values[name] = str(component)
                current_hashes[name] = hash(str(component))

        # Get previous values and hashes
        prev_values = storage.get('component_values', {})
        prev_hashes = storage.get('component_hashes', {})

        # Check for non-PROMPTER changes
        changed_components = []
        detailed_changes = []

        for name in components_to_check.keys():
            if prev_hashes.get(name) != current_hashes[name]:
                changed_components.append(name)

                # Find specific changes within this component
                if name in prev_values and name in current_values:
                    prev_dict = prev_values[name]
                    curr_dict = current_values[name]

                    for key in set(list(prev_dict.keys()) + list(curr_dict.keys())):
                        prev_val = prev_dict.get(key)
                        curr_val = curr_dict.get(key)

                        # 1) Any tensor involved: log shapes & types, then continue
                        if isinstance(prev_val, torch.Tensor) or isinstance(curr_val, torch.Tensor):
                            prev_shape = getattr(prev_val, "shape", None)
                            curr_shape = getattr(curr_val, "shape", None)

                            # Detailed log for geometry changes (this is where 1024x1024 → segment size shows up)
                            log(
                                f"TBG[Node {tbg.INFO.id}] GEOMETRY CHANGE in {name}.{key}: "
                                f"prev_shape={prev_shape}, curr_shape={curr_shape}, "
                                f"prev_type={type(prev_val)}, curr_type={type(curr_val)}",
                                None, None, f"Node {tbg.INFO.id}"
                            )

                            if prev_shape != curr_shape:
                                # Shapes differ – record that fact
                                detailed_changes.append(
                                    f"{name}.{key}: tensor shape {prev_shape} -> {curr_shape}"
                                )
                            else:
                                # Same shape: if both tensors, compare safely
                                if isinstance(prev_val, torch.Tensor) and isinstance(curr_val, torch.Tensor):
                                    if not torch.equal(prev_val, curr_val):
                                        detailed_changes.append(
                                            f"{name}.{key}: tensor changed (shape {prev_shape})"
                                        )
                                else:
                                    # One is tensor, the other is not – treat as change
                                    detailed_changes.append(
                                        f"{name}.{key}: tensor/non-tensor mismatch "
                                        f"{type(prev_val)} -> {type(curr_val)}"
                                    )

                            # CRITICAL: never fall through to the generic `prev_val != curr_val`
                            continue

                        # 2) Lists / tuples
                        if isinstance(prev_val, (list, tuple)) and isinstance(curr_val, (list, tuple)):
                            if prev_val != curr_val:
                                detailed_changes.append(f"{name}.{key}: {prev_val} -> {curr_val}")
                                log(
                                    f"TBG[Node {tbg.INFO.id}] LIST CHANGE in {name}.{key}: "
                                    f"{prev_val} -> {curr_val}",
                                    None, None, f"Node {tbg.INFO.id}"
                                )
                            continue

                        # 3) Generic comparison for everything else
                        try:
                            if prev_val != curr_val:
                                prev_str = str(prev_val)
                                curr_str = str(curr_val)
                                if len(prev_str) > 50:
                                    prev_str = prev_str[:50] + "..."
                                if len(curr_str) > 50:
                                    curr_str = curr_str[:50] + "..."
                                detailed_changes.append(f"{name}.{key}: {prev_str} -> {curr_str}")
                                log(
                                    f"TBG[Node {tbg.INFO.id}] VALUE CHANGE in {name}.{key}: "
                                    f"{prev_str} -> {curr_str}",
                                    None, None, f"Node {tbg.INFO.id}"
                                )
                        except RuntimeError as e:
                            # Absolute safety net: if some weird type still explodes on `!=`, log and treat as changed
                            log(
                                f"TBG[Node {tbg.INFO.id}] RUNTIME ERROR comparing {name}.{key}: "
                                f"prev_type={type(prev_val)}, curr_type={type(curr_val)}, error={e}",
                                None, None, f"Node {tbg.INFO.id}"
                            )
                            detailed_changes.append(
                                f"{name}.{key}: comparison error {type(prev_val)} vs {type(curr_val)} – {e}"
                            )

        # Store current values and hashes
                # Store current values and hashes
        storage['component_values'] = current_values
        storage['component_hashes'] = current_hashes

        # If ANY non-PROMPTER component changed, must process all tiles
        if changed_components:
            #Clear tile cache since global settings changed / will be normalized in refiner
            tiler_context_keys = (
                "OUTPUTS.image",
                "PARAMS.grid_specs",
                "PARAMS.len_grid_images",
                "PARAMS.tiler_cache_key",
                "PARAMS.Tile_Fusion_Mode",
                "SIZE.",
                "SEGMENTS.",
            )
            tiler_context_changed = (
                "OUTPUTS" in changed_components
                or "SIZE" in changed_components
                or "SEGMENTS" in changed_components
                or any(change.startswith(tiler_context_keys) for change in detailed_changes)
            )


            if 'generated_tiles' in storage:
                storage['generated_tiles'].clear()
            if hasattr(tbg.OUTPUTS, "persistent_generated_tiles"):
                try:
                    tbg.OUTPUTS.persistent_generated_tiles.clear()
                except Exception:
                    # If type is not clear-able, ignore – cache will be rebuilt
                    pass
            if 'persistent_generated_tiles' in storage:
                try:
                    storage['persistent_generated_tiles'].clear()
                except Exception:
                    pass
            if tiler_context_changed:
                tbg.PARAMS.force_fresh_refiner_background = True
                tbg.OUTPUTS.last_final_image = None
                log(
                    "[TBG Cache] tiler context changed: clearing generated tile cache and last_final_image",
                    None,
                    None,
                    f"Node {tbg.INFO.id}",
                )
            # IMPORTANT: also refresh PROMPTER baseline for the *next* run
            # so that subsequent runs can correctly detect per-tile changes
            try:
                # We only need the side-effect: storage['prompter_data'] is updated
                _ = cls._get_changed_tile_indices()
            except Exception as e:
                log(
                    f"TBG[Node {tbg.INFO.id}] Failed to update PROMPTER baseline during global change detection: {e}",
                    None,
                    None,
                    f"Node {tbg.INFO.id}"
                )

            # Log detailed changes
            #log(f"TBG[Node {tbg.INFO.id}] Global components changed: {changed_components}", None, None,f"Node {tbg.INFO.id}")
            for change in detailed_changes:
                log(f"TBG[Node {tbg.INFO.id}]   {change}", None, None, f"Node {tbg.INFO.id}")

            return "ALL", [], f"Global components changed: {changed_components} ({len(detailed_changes)} specific changes)"

        # No global changes - check PROMPTER for tile-specific changes
        changed_indices = cls._get_changed_tile_indices()
        """
        if changed_indices:
            count = len(changed_indices)
            tile_word = "tile" if count == 1 else "tiles"
            return "PROMPTER", changed_indices, f"Detected TilePrompter changes in {count} {tile_word}"
        else:
            return "NONE", [], "No tile prompt changes detected"
        """
        if changed_indices:
            # Original set of tiles where prompt/denoise/seed/cnet actually changed
            original_changed = set(changed_indices)

            # Default behavior: per-tile resampling for independent tiles
            expanded_indices = set(original_changed)
            mode = getattr(tbg.PARAMS, "Tile_Fusion_Mode", None)

            # For Neuro_Generative_Tile_Fusion, each tile influences the next ones:
            # if tile k changed, we must re-sample ALL tiles from k to the end.
            if mode in ("Neuro_Generative_Tile_Fusion", "NGTF_FLUX_Kontext"):
                try:
                    # Prefer the true tile count from grid_images_all
                    total_tiles = len(getattr(tbg.OUTPUTS, "grid_images_all", []))
                    if total_tiles == 0:
                        # Fallback to PROMPTER outputs length if grid_images_all is not yet populated
                        total_tiles = len(getattr(tbg.PROMPTER, "output_prompts", []))

                    if total_tiles > 0:
                        start = min(original_changed)
                        expanded_indices = set(range(start, total_tiles))

                        log(
                            f"TBG[Node {tbg.INFO.id}] Neuro_Generative_Tile_Fusion active: "
                            f"tiles {start + 1} to {total_tiles} will be re-sampled "
                            f"(first changed tile index={start + 1})",
                            None, None, f"Node {tbg.INFO.id}"
                        )
                except Exception as e:
                    # Fail-safe: if anything goes wrong, fall back to the original sparse set
                    log(
                        f"TBG[Node {tbg.INFO.id}] Error expanding changed_indices for "
                        f"Neuro_Generative_Tile_Fusion: {e}",
                        None, None, f"Node {tbg.INFO.id}"
                    )
                    expanded_indices = set(original_changed)
            else:
                # For Soft Merge / NONE / Tile_Fusion etc., keep per-tile independence
                log(
                    f"TBG[Node {tbg.INFO.id}] Tile_Fusion_Mode={mode or 'NONE'}: "
                    f"using sparse per-tile changes ({len(original_changed)} tiles).",
                    None, None, f"Node {tbg.INFO.id}"
                )

            count_effective = len(expanded_indices)
            tile_word = "tile" if count_effective == 1 else "tiles"

            return (
                "PROMPTER",
                expanded_indices,
                f"Detected TilePrompter changes in {len(original_changed)} tiles; "
                f"re-sampling {count_effective} {tile_word} according to Tile_Fusion_Mode={mode}"
            )
        else:
            return "NONE", [], "No tile prompt changes detected"

    @classmethod
    def _get_changed_tile_indices(cls):
        """Compare current vs previous PROMPTER data to find exactly which tiles changed"""
        storage = persistent_storage.get(tbg.storage_key, {})

        # Get current tile-specific data from OUTPUTS (where PROMPTER writes to)
        current_data = {
            'grid_prompts': getattr(tbg.PROMPTER, 'output_prompts',[]),
            'grid_denoises': getattr(tbg.PROMPTER, 'output_denoises', []),
            'grid_seed': getattr(tbg.PROMPTER, 'output_seeds_js', []),
            'grid_cnet_strength': getattr(tbg.PROMPTER, 'output_cnet_js', []),
            'grid_cfg_overrides': getattr(tbg.PROMPTER, 'output_cfg_js', []),
            'grid_model_overrides': getattr(tbg.PROMPTER, 'output_model_js', []),
            'grid_model_override_fingerprints': cls._tile_model_override_fingerprints(),
            'grid_cnetpipe_overrides': getattr(tbg.PROMPTER, 'output_cnetpipe_js', []),
            'grid_cnetpipe_override_fingerprints': cls._tile_cnetpipe_override_fingerprints(),
            'grid_color_match_overrides': getattr(tbg.PROMPTER, 'output_color_match_js', []),
            'grid_ignore_general_prompts': getattr(tbg.PROMPTER, 'output_ignore_general_prompt_js', []),
        }

        # Get previous data from storage
        prev_data = storage.get('prompter_data', {})

        # Find changed tiles - only if values actually changed
        changed_indices = set()
        num_tiles = len(current_data['grid_prompts'])

        for i in range(num_tiles):
            for key, curr_list in current_data.items():
                # Ensure we have valid lists
                if not isinstance(curr_list, (list, tuple)):
                    continue

                prev_list = prev_data.get(key, [])

                # Get previous and current values (with bounds checking)
                prev_val = prev_list[i] if i < len(prev_list) else None
                curr_val = curr_list[i] if i < len(curr_list) else None

                # Compare values - treat None and empty string as equivalent
                prev_comp = prev_val if prev_val not in [None, ''] else None
                curr_comp = curr_val if curr_val not in [None, ''] else None

                if prev_comp != curr_comp:
                    changed_indices.add(i)
                    break  # No need to check other keys for this tile

        # Store current data for next comparison
        storage['prompter_data'] = current_data

        return changed_indices

    @classmethod
    def _tile_model_override_fingerprints(cls):
        choices = list(getattr(tbg.PROMPTER, "output_model_js", []) or [])
        models = cls._get_tile_override_models()
        out = []
        for choice in choices:
            norm = str(choice or "").strip().lower().replace("_", " ").replace("-", " ")
            if norm in ("1", "model1"):
                norm = "model 1"
            elif norm in ("2", "model2"):
                norm = "model 2"
            elif norm in ("3", "model3"):
                norm = "model 3"

            idx = {"model 1": 0, "model 2": 1, "model 3": 2}.get(norm)
            if idx is None:
                out.append("")
                continue
            model = models[idx] if idx < len(models) else None
            out.append("" if model is None else f"{type(model).__name__}:{id(model)}")
        return out

    @classmethod
    def _tile_cnetpipe_override_fingerprints(cls):
        choices = list(getattr(tbg.PROMPTER, "output_cnetpipe_js", []) or [])
        cnetpipes = cls._get_tile_override_cnetpipes()
        out = []
        for choice in choices:
            norm = str(choice or "").strip().lower().replace("_", " ").replace("-", " ")
            if norm in ("1", "cnetpipe1", "cnet pipe 1", "controlnet pipe 1"):
                norm = "cnetpipe 1"
            elif norm in ("2", "cnetpipe2", "cnet pipe 2", "controlnet pipe 2"):
                norm = "cnetpipe 2"
            elif norm in ("3", "cnetpipe3", "cnet pipe 3", "controlnet pipe 3"):
                norm = "cnetpipe 3"

            idx = {"cnetpipe 1": 0, "cnetpipe 2": 1, "cnetpipe 3": 2}.get(norm)
            if idx is None:
                out.append("")
                continue
            cnetpipe = cnetpipes[idx] if idx < len(cnetpipes) else None
            out.append("" if cnetpipe is None else f"{type(cnetpipe).__name__}:{id(cnetpipe)}:{len(cnetpipe) if hasattr(cnetpipe, '__len__') else ''}")
        return out

    @classmethod
    def _normalize(cls, arr, n):
        """Pad or truncate array to length n."""
        arr = list(arr or [])
        if len(arr) < n:
            arr.extend([""] * (n - len(arr)))
        elif len(arr) > n:
            arr = arr[:n]
        return arr

def combine_conditioning(conds: list):
    combined_conds = []
    for cond in conds:
        combined_conds.extend(cond)
    combined = combine_conditions(combined_conds)
    return combined
def combine_conditions(conditions):
    # Combine tensors (assuming they are identical, just use the first)
    tensor = conditions[0][0]
    # Combine dicts
    combined_dict = {}
    keys = set().union(*(d.keys() for _, d in conditions))
    for key in keys:
        values = [d[key] for _, d in conditions if key in d]
        if key == 'guidance':
            combined_dict[key] = sum(values) / len(values)  # average
        elif key == 'control':
            combined_dict[key] = values[-1]  # or some other logic
        else:
            combined_dict[key] = values[0]  # default to first, or customize
    return [[tensor, combined_dict]]

