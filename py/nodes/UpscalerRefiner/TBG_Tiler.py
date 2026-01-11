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
from .inc.prosegs import TBG_Segms
from ...vendor.ComfyUI_Janus_Pro.nodes.model_loader import JanusModelLoader
from ...vendor.ComfyUI_Janus_Pro.nodes.image_understanding import JanusImageUnderstanding
from ...vendor.ComfyUI_Unload_Models_main.py.unload_one_model import UnloadOneModelNode
from ...vendor.ComfyUI_QwenVL.AILab_QwenVL import QwenVLBase
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
                return func(cls, *args)  # no kwargs if you don't use them

            if not hasattr(cls, cache_name):
                setattr(cls, cache_name, {})
            cache = getattr(cls, cache_name)

            cache_key = None
            try:
                # 1) Take image from args (input_image), not from tbg
                image = None
                if len(args) > 0:
                    image = args[0]  # for @classmethod: input_image

                if image is not None:
                    if hasattr(image, 'cpu'):         # torch tensor
                        image_np = image.cpu().numpy()
                        image_bytes = image_np.tobytes()
                    elif hasattr(image, 'tobytes'):   # numpy array
                        image_bytes = image.tobytes()
                    else:
                        image_bytes = str(image).encode()
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                else:
                    image_hash = "none"

                # 2) Optional: keep your kwargs-based hash if you want
                relevant_keys = ['upscale_by', 'upscale_model']
                cache_kwargs = {k: v for k, v in kwargs.items()
                                if k in relevant_keys or k == 'PARAMS'}
                kwargs_str = json.dumps(cache_kwargs, sort_keys=True, default=str)
                kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()

                cache_key = f"{func.__name__}_{image_hash}_{kwargs_hash}"
            except Exception as e:
                print(f"Cache key generation failed: {e}. Running without cache.")

            # If key exists -> cache hit
            if cache_key and cache_key in cache:
                log("Using Cache for Upscaled Image", None, None, f"Node {tbg.INFO.id}")
                return cache[cache_key]

            # Otherwise run the function
            log("Upscaled Imag", None, None, f"Node {tbg.INFO.id}")
            result = func(cls, *args)

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

def make_tiler_config(tbg, labs_upscaler_dict):
    # hash image content
    img = tbg.OUTPUTS.upscaled_image
    if hasattr(img, "cpu"):
        img_bytes = img.cpu().numpy().tobytes()
    elif hasattr(img, "tobytes"):
        img_bytes = img.tobytes()
    else:
        img_bytes = str(img).encode()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    # segment: use count + mask flag; if you want, add a hash of Segment_Mask tensor too
    seg_count = len(tbg.SEGMENTS.segms or [])
    has_seg_mask = tbg.SEGMENTS.Segment_Mask is not None

    cfg = {
        "img": img_hash,
        "presets":          tbg.PARAMS.preset,
        "Fusion_strength":  tbg.SIZE.inpaint_max,
        "tile_size_w":      tbg.SIZE.fullW,
        "tile_size_h":      tbg.SIZE.fullH,
        "upscale_model":    tbg.PARAMS.upscale_model_name,
        "upscale_by":       tbg.PARAMS.upscale_by,
        "PRO_Tile_Mode":    tbg.PARAMS.Tile_Fusion_Mode,
        "composite_blur":   tbg.SIZE.composite_blur_margin,
        "fusion_margin":    tbg.SIZE.inpaint_blur_margin,
        "Segment_Mask_on":  has_seg_mask,
        "PRO_segs_count":   seg_count,
        "labs_upscaler":    labs_upscaler_dict,  # or a minimal hash of it
    }
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()



from ....TBG.SERVERS.COMFYUI_server import register_main_class
from comfy.utils import ProgressBar
@register_main_class
class TBG_Upscaler_v1():
    _pbar = None


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
        tbg.PARAMS.upscale_by = kwargs.get('upscale_by', 1)
        if tbg.PARAMS.upscale_model_name == "NONE":
            tbg.PARAMS.upscale_by = 1



        tbg.API.token = kwargs.get("PRO_api_token", None)
        WORKER.id(tiler_id).TBG_api.update(tbg.API.token,_tbg_send_images=False, )

        end_time = time.time()
        elapsed = end_time - tbg.start_time
        def get_api_infos():
            info,status,creditsleft,current_credits, info_url = WORKER.id(tiler_id).TBG_api.check_status(tbg.API.token, True,_tbg_send_images=False, )
            return  info,status,creditsleft,current_credits,info_url
        tbg.API.info, tbg.API.status, tbg.API.creditsleft, tbg.API.current_credits, tbg.API.info_url = get_api_infos()


        end_time = time.time()
        elapsed = end_time - tbg.start_time

        tbg.PARAMS.TBG_APP_ShutDown = kwargs.get(' TBG_APP_ShutDown', 'Close with Comfyui')

        # INIT SEGMENTS
        tbg.SEGMENTS.Segment_Mask = kwargs.get('Segment_Mask', None)
        tbg.SEGMENTS.segms = kwargs.get('PRO_segs', None)

        # ADD THIS BLOCK:
        if tbg.SEGMENTS.Segment_Mask is None and tbg.SEGMENTS.segms is None:
            # No segment inputs → clear everything
            tbg.SEGMENTS.upscale_factor = None,
            tbg.SEGMENTS.pad_offset = None,
            tbg.SEGMENTS.segment_tiles = None,
            tbg.SEGMENTS.orig_segment_tiles = None,
            tbg.SEGMENTS.segms_scale = None,
            tbg.SEGMENTS.segms_cropped_masks = None,
            tbg.SEGMENTS.segms_crop_regions = None,
            tbg.SEGMENTS.segms_new = None,
            tbg.SEGMENTS.inpainting_mask = None,
            tbg.SEGMENTS.compositing_mask = None,
            tbg.SEGMENTS.h = None,
            tbg.SEGMENTS.w = None,

        if tbg.SEGMENTS.Segment_Mask is not None:
            SEG = MaskToSEGS(tbg.SEGMENTS.Segment_Mask, False, 2, False, 0, False)
            if tbg.SEGMENTS.segms is not None:
                tbg.SEGMENTS.segms = combine_segs(tbg.SEGMENTS.segms, SEG)
            else:
                tbg.SEGMENTS.segms = SEG
            print(f"TBG[Node {tbg.INFO.id}] Converted {len(SEG)} segment{'s' if len(SEG) != 1 else ''} to Tiles.")

        cls.init(**kwargs)
        # Upscale - SR upscales are overwriting the tbg so we mantain the original images and pass it later again into the tbg
        # st jetzt sauber getrennt
        #orig_image = copy.deepcopy(tbg.INPUTS.image)

        tbg.OUTPUTS.upscaled_image = cls.upscale_full_input_image(tbg.INPUTS.image, **kwargs) # goes to cache wrapper with kwargs

        if Only_Upscale:
            return ((0,0,0,0,0,0,0,0,"Only Upscale",0,0,tbg.API.info_url), tbg.OUTPUTS.upscaled_image)

        #tbg.INPUTS.image = orig_image
        if tbg.OUTPUTS.upscaled_image is not None:
            tbg.SIZE.UpscaledInputImageH = tbg.OUTPUTS.upscaled_image.shape[1]
            tbg.SIZE.UpscaledInputImageW = tbg.OUTPUTS.upscaled_image.shape[2]

        # in TBG_Tiler.start_tiles or just after init()
        labs_upscaler_dict = kwargs.get('labs_upscaler', None)
        new_key = make_tiler_config(tbg, labs_upscaler_dict)
        last_key = getattr(tbg.TEMP, "last_tiler_config", None)

        if last_key == new_key:
            tbg.TEMP.skip_tiler = True
        else:
            tbg.TEMP.skip_tiler = False
            tbg.TEMP.last_tiler_config = new_key

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
        tbg.PARAMS.Prompt_seed =  kwargs.get('VLM_seed', None)
        labs_upscaler_dict = kwargs.get('labs_upscaler', None)
        if labs_upscaler_dict:
            # Optional
            Tiler_Upscale_Cache = labs_upscaler_dict.get('Tiler_Upscale_Cache', True)
            Only_Upscale = labs_upscaler_dict.get('Only_Upscale', True)
            tbg.PARAMS.SEEDVR2_VAE = labs_upscaler_dict.get('SEEDVR2_VAE', {'model': 'ema_vae_fp16.safetensors', 'device': 'cuda:0', 'offload_device': 'none',
                           'cache_model': True, 'encode_tiled': True, 'encode_tile_size': 512,
                           'encode_tile_overlap': 64,
                           'decode_tiled': True,
                           'decode_tile_size': 512, 'decode_tile_overlap': 64, 'tile_debug': 'false',
                           'torch_compile_args': 'reduce-overhead', 'node_id': '9'})
            tbg.PARAMS.SEEDVR2_DIT = labs_upscaler_dict.get('SEEDVR2_DIT',  {'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                           'device': 'cuda:0', 'offload_device': 'none', 'cache_model': True,
                           'blocks_to_swap': 0,
                           'swap_io_components': False,
                           'attention_mode': attention_mode,
                           'torch_compile_args': 'reduce-overhead', 'node_id': '8'})
            tbg.PARAMS.SEEDVR2_DIT_low = labs_upscaler_dict.get('SEEDVR2_DIT_low',  {'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                           'device': 'cuda:0', 'offload_device': 'none', 'cache_model': True,
                           'blocks_to_swap': 0,
                           'swap_io_components': False,
                           'attention_mode': attention_mode,
                           'torch_compile_args': 'reduce-overhead', 'node_id': '8'})
        else:
            Tiler_Upscale_Cache = False
            if tbg.API.status in ["Free", "Pro", "Premium", "Unlimited", "Dev"]:
                Tiler_Upscale_Cache = True
            Only_Upscale = False
        tbg.PARAMS.Inventivity=kwargs.get('Inventivity', None)
        tbg.PARAMS.Resemblance=kwargs.get('Resemblance', None)
        tbg.PARAMS.Fractality=kwargs.get('Fractality', None)
        tbg.PARAMS.Creativity=kwargs.get('Creativity', None)
        tbg.INFO.id = kwargs.get('id', None)
        tbg.INPUTS.image = kwargs.get('image', None)
        tbg.PROMPTER.Prompt_Selected_Tiles_Only = kwargs.get('VLM_Selected_Tiles_Only', False)
        tbg.PROMPTER.Prompt_Selected_Tiles_By_Numbers = kwargs.get('VLM_Selected_Tiles_By_Numbers', " ")
        tbg.LLM.prompt = kwargs.get('VLM_Prompt', " ")
        tbg.LLM.quantization =   kwargs.get('VLM_Quantization', "None (FP16)")
        tbg.LLM.model = kwargs.get('VLM_Model', 'NONE')
        tbg.PARAMS.MODEL_TYPE_SIZES = kwargs.get('MODEL_TYPE_SIZES', False)
        tbg.PARAMS.Tile_Fusion_Mode = kwargs.get('Fusion Mode', 'NONE')
        tbg.PARAMS.upscale_model_name=kwargs.get('upscale_model', "NONE")
        tbg.PARAMS.preset = kwargs.get('presets', "NONE")
        tbg.PARAMS.fragmentation = kwargs.get('fragmentation', 1)

        tbg.SIZE.fullH = kwargs.get('tile_size_h', 1024)
        tbg.SIZE.fullW = kwargs.get('tile_size_w', 1024)

        if tbg.PARAMS.fragmentation and tbg.PARAMS.fragmentation != 0 or tbg.PARAMS.fragmentation != 1:
            kwargs["tile_size_w"] = int(kwargs.get("tile_size_w", 1024) * tbg.PARAMS.fragmentation)
            kwargs["tile_size_h"] = int(kwargs.get("tile_size_h", 1024) * tbg.PARAMS.fragmentation)


        tbg.SIZE.Fusion_margin = kwargs.get('Fusion Margin', 64)
        # auto fusion border calculation
        tbg.SIZE.inpaint_blur_margin = tbg.SIZE.Fusion_margin
        tbg.SIZE.shift = 0
        tbg.SIZE.composite_blur_margin = kwargs.get('Feather Mask', 16)
        tbg.SIZE.inpaint_border_margin = int(( tbg.SIZE.Fusion_margin / 16) * 8)  # 1/2 Blur = 64
        tbg.SIZE.shifttl  = tbg.SIZE.inpaint_blur_margin + int(tbg.SIZE.Fusion_margin / 4)  # 64 same as border to eliminate border on left bottom
        tbg.SIZE.inpaint_max = kwargs.get('Fusion Strength', 0.05)

    @classmethod
    @upscale_cache_comfy_method(maxsize=1, enabled=lambda: Tiler_Upscale_Cache)
    def upscale_full_input_image(cls, input_image):

        print("[DEBUG] upscale gate:",
              "model=", tbg.PARAMS.upscale_model_name,
              "factor=", tbg.PARAMS.upscale_by)

        if input_image is None:
            raise ValueError(f"TBG Enhanced Tiled Generator id {tbg.INFO.id}: No image provided")

        log("TBG Enhanced Tiled Generator is starting", None, None, f"Node {tbg.INFO.id}")
        # log(f"Starting Upscaling  upscale_type", None, None, f"Node {tbg.INFO.id}")

        if tbg.PARAMS.upscale_model_name != "NONE" and tbg.PARAMS.upscale_by not in (0,1):
            upscaled_image = copy.copy(input_image)

            #GAN MODELS
            if tbg.PARAMS.upscale_model_name != None and tbg.PARAMS.upscale_model_name not in tbg.upscale_models:
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
    #@tiler_cache_comfy_method(maxsize=1)
    def tiler(cls, full_upscaled_image, iteration):

        tbg.OUTPUTS.upscaled_image = full_upscaled_image
        def get_tiler_init():
            PARAMS, SIZE = WORKER.id(tiler_id).ETUR.tiler_init(tbg.PARAMS, tbg.SIZE)
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
            # get updated grid_specs

            crop_regions = [seg.crop_region for seg in upscaled_segments[1]]

            h, w = upscaled_segments[0]
            def get_workers_grid_specs(h, w, crop_regions):
                grid_specs = WORKER.id(tiler_id).TBG_Segms.create_grid_specs_for_segments(h, w, crop_regions,
                                                                                        tbg.PARAMS.grid_specs,
                                                                                        tbg.SIZE.rows_qty,
                                                                                        tbg.SIZE.cols_qty,_tbg_send_images=False, )
                return grid_specs

            tbg.PARAMS.grid_specs = get_workers_grid_specs(h, w, crop_regions)

            # create array of tiles and mask

            segs_cropped_masks = []
            segment_tiles = []
            segs_scale = []
            segment_inpainting_mask = []
            segment_compositing_mask = []
            for i, seg in enumerate(upscaled_segments[1]):
                segs_cropped_masks.append(seg.cropped_mask)  # composite mask - upscaled mask
                segment_tiles.append(seg.cropped_image)
                segment_inpainting_mask.append(seg.inpainting_mask)
                segment_compositing_mask.append(seg.compositing_mask)
                segs_scale.append(1) # hardcoded to 1 - not used anymore
            tbg.PARAMS.segs_scale = segs_scale
            tbg.OUTPUTS.grid_images_all = (grid_images or []) + (segment_tiles or [])
            tbg.OUTPUTS.orig_grid_images_all = copy.copy(tbg.OUTPUTS.grid_images_all)

            tbg.SEGMENTS.segment_tiles = segment_tiles
            tbg.PARAMS.len_segments = len(segment_tiles)
            tbg.SEGMENTS.segms_scale = segs_scale
            tbg.SEGMENTS.segms_cropped_masks = segs_cropped_masks
            tbg.SEGMENTS.segms_new = upscaled_segments
            tbg.SEGMENTS.segms_crop_regions = crop_regions
            tbg.SEGMENTS.inpainting_mask = segment_inpainting_mask
            tbg.SEGMENTS.compositing_mask = segment_compositing_mask

        else:
            tbg.OUTPUTS.grid_images_all = grid_images + [] # + to capy so we can clear grid_image
            tbg.OUTPUTS.orig_grid_images_all = copy.copy(grid_images)
            tbg.SEGMENTS.segment_tiles  = None
            tbg.PARAMS.len_segments = 0
            tbg.PARAMS.segs_scale = None
            tbg.SEGMENTS.segms_cropped_masks = None
            tbg.SEGMENTS.segms_new = None

        # reset to save memory
        tbg.PARAMS.len_grid_images = len(grid_images)
        grid_images.clear()


    @classmethod
    def prompter(cls, iteration):
        print("len(tbg.OUTPUTS.grid_images_all)",len(tbg.OUTPUTS.grid_images_all))
        grid_prompts = []
        if not tbg.LLM.model == "NONE":
            if "sky" in tbg.LLM.model.lower():
                qwen = Qwen2VL_TBG()
            if "janus" in tbg.LLM.model.lower():
                janus = JanusModelLoader()
            if "qwen" in tbg.LLM.model.lower():
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

            januspromt = tbg.LLM.prompt
            llm_model = tbg.LLM.model  # modelname

            if llm_model == "Janus-Pro-1B":
                janusmodel, janusprocessor = janus.load_model("deepseek-ai/Janus-Pro-1B")
                return JanusImageUnderstanding.analyze_image(
                    tbg, janusmodel, janusprocessor, images[0], januspromt,
                    tbg.PARAMS.Prompt_seed, 0.1, 0.9, 512
                )[0]

            elif llm_model == "Janus-Pro-7B":

                janusmodel, janusprocessor = janus.load_model("deepseek-ai/Janus-Pro-7B")
                return JanusImageUnderstanding.analyze_image(
                    tbg, janusmodel, janusprocessor, images[0], januspromt,
                    tbg.PARAMS.Prompt_seed, 0.1, 0.9, 512
                )[0]

            elif tbg.LLM.model.startswith("Qwen"):


                if tbg.LLM.model.endswith("-FP8"):
                    tbg.LLM.quantization = "None (FP16)"

                seed =  tbg.PARAMS.Prompt_seed
                return QwenVL.run(
                    tbg.LLM.model,
                    tbg.LLM.quantization,
                    preset_prompt="🖼️ Detailed Analysis",
                    custom_prompt=januspromt,
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
                qwen.inference(januspromt, "SkyCaptioner-V1", "none", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "SkyCaptioner-V1_4bit":
                return \
                qwen.inference(januspromt, "SkyCaptioner-V1", "4bit", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "SkyCaptioner-V1_8bit":
                return \
                qwen.inference(januspromt, "SkyCaptioner-V1", "8bit", True, 0.7, 512, -1, images, video_path=None)[0]

            elif llm_model == "Apple FastVLM 7B Research use only":
                from nodes import NODE_CLASS_MAPPINGS
                node_name = "FastVLM7BNode"
                if node_name in NODE_CLASS_MAPPINGS:
                    FastVLM7BNode = NODE_CLASS_MAPPINGS[node_name]
                    return FastVLM7BNode.inference(0, images, januspromt, 200)[0]
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

        print(
            f"TBG[Node {tbg.INFO.id}] VLM {tbg.LLM.model} working  of {len(tbg.OUTPUTS.grid_images_all)}")

        for index, tile_to_process in enumerate(tbg.OUTPUTS.grid_images_all):
            # Skip tile only if user selected some tiles AND this index is not one of them
            if len(tbg.PROMPTER.tiles_to_process) != 0 and index not in tbg.PROMPTER.tiles_to_process:
                prompt_tile = ""
                grid_prompts.append(prompt_tile)
                continue


            prompt_tile = prompt_context
            if tbg.LLM.model != "NONE":
                print(f"TBG[Node {tbg.INFO.id}] VLM {tbg.LLM.model} working on Tile {index} of {len(tbg.OUTPUTS.grid_images_all)}")
                prompt_tile = get_prompt_tile(tile_to_process)
                sentences_with_words_to_remove = [
                    "assistant", "helpful", "vision", "Thedescription", "TheUser", "The睿", "The description",
                    "The User",
                    "It can assist", "natural language", "It can understand"
                ]
                prompt_tile = cls.remove_sentences_with_words(prompt_tile, sentences_with_words_to_remove)

                log(f"tile {index + 1}/{total} - [tile prompt] {prompt_tile}", None, None,
                    f"Node {tbg.INFO.id} - Prompting {iteration}")
            grid_prompts.append(prompt_tile)

        tbg.PROMPTER.tiler_prompts = grid_prompts

        # if the input comes direct from tiler ignore and rebuild empty values from prompter

        tbg.PROMPTER.output_prompts = tbg.PROMPTER.tiler_prompts
        tiles_len = len(tbg.PROMPTER.tiler_prompts)
        tbg.PROMPTER.output_denoises = cls._normalize([], tiles_len)
        tbg.PROMPTER.output_seeds_js = cls._normalize([], tiles_len)
        tbg.PROMPTER.output_cnet_js = cls._normalize([], tiles_len)

        if tbg.LLM.model != "NONE":
            if "Sky" in tbg.LLM.model.lower():
                UnloadOneModelNode.route(qwen)
            if "janus" in tbg.LLM.model.lower():
                UnloadOneModelNode.route(janus)
            if "Qwen" in tbg.LLM.model.lower():
                UnloadOneModelNode.route(QwenVL)
            if "florence" in tbg.LLM.model.lower():
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