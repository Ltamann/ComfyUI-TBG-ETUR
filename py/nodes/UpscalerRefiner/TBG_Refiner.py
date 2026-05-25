"""
_______________________________________________________________________________________________________________________________________________
______________________________________TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO_________________________________________________________
__________________________________________________a ToBi´s Gen production_____________________________________________________________________
"""
import time
import threading
import time
import comfy
#import numpy as np
import PIL
from PIL import Image
import copy
import nodes
from comfy_extras.nodes_mask import MaskToImage, ImageToMask
from .inc.TBG_split_aware_lanpaint_sampler  import TBG_DualModelSampler_COPY as TBG_DualModelSampler_lanpaint, TBG_KSamplerAdvancedSplitAware_Copy as TBG_KSamplerAdvancedSplitAware_lanpaint
from .inc.TBG_sampler_split_aware import TBG_DualModelSampler_COPY as TBG_DualModelSampler_normal, TBG_KSamplerAdvancedSplitAware_Copy as TBG_KSamplerAdvancedSplitAware_normal
from ...vendor.ComfyUI_UltimateSDUpscale.utils import  crop_cond
from ...vendor.ComfyUI_Detail_Daemon.detail_daemon_node import DetailDaemonSamplerNode
from ....TBG.SERVERS.WORKER_server import WORKER,TBG_Controller

from ...vendor.comfyui_resharpen_main.tbgresharpen import disable_resharpen, TBG_DetailEnhancer
from ...utils.log import log
from .inc.image import TBG_Image
from .inc.sigmas import get_sigmas
from .inc.sigmas import denoise_sigmas_tgb
from .inc.cnet import get_Kontext_stiched_o_chained_cond, get_qwen_stiched_o_chained_cond
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
    VRAM_OPTIMIZER = None
    MODEL_TYPE_SIZES = {
        'FLUX1': 1024,
        'FLUX2': 2048,
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
        output_tiles = list(copy.deepcopy(tbg.OUTPUTS.grid_images_all))
        input_tiles = list(tbg.OUTPUTS.orig_grid_images_all or [])

        # Squeeze batch dim where tile exists
        for index, torchtile in enumerate(output_tiles):  # BHWC
            if torchtile is not None:
                output_tiles[index] = torchtile.squeeze(0)  # [1,H,W,3] → [H,W,3]

        # Fill missing or non-existent output tiles from input tiles
        for index, input_tile in enumerate(input_tiles):  # BHWC
            if index >= len(output_tiles) or output_tiles[index] is None:
                output_tiles[index] = copy.deepcopy(input_tile).squeeze(0)

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
            tbg.KSAMPLER.custom_sigmas = labs_refiner_dict.get('Custom_Sigmas_!DENOISE=1', None)
            tbg.PARAMS.Alternative_Image = labs_refiner_dict.get('Resume_Tiled_Refinement_Image', None)
            # Requiered
            tbg.PARAMS.Tile_Fusion_Blend = labs_refiner_dict.get('Tile_Fusion_Blend', 0.5)
            tbg.PARAMS.inpaint_conditioning = labs_refiner_dict.get('Fusion_conditioning', True)
            tbg.PARAMS.point_grid_image_stabilizer_experimental = labs_refiner_dict.get('Color & Structure Stabilizer', 0)
            tbg.PARAMS.memorize = labs_refiner_dict.get('PRO_Tile_Cache', 'OFF')

            tbg.PARAMS.LanPaint = labs_refiner_dict.get('LanPaint', True)
            inpaint_end = labs_refiner_dict.get('Fusion_end', 0)
            tbg.PARAMS.Preview_Tiles_in_Temp_Folder = labs_refiner_dict.get('Save_Tiles_in_Temp_Folder', False)
            tbg.KSAMPLER.sampler_input = labs_refiner_dict.get('Sampler', None)
            tbg.KSAMPLER.cropped_positive = labs_refiner_dict.get('cropped_positive', None)
            tbg.KSAMPLER.cropped_negative = labs_refiner_dict.get('cropped_negative', None)
            tbg.PARAMS.stitch_blending = labs_refiner_dict.get('stitch_blending', "gpupyramid")
            tbg.PARAMS.max_upscale_size_segment_inpainting =  labs_refiner_dict.get('max_upscale_size_segment', 2048)


        else:

            tbg.KSAMPLER.custom_sigmas = None
            tbg.PARAMS.Alternative_Image = None
            tbg.PARAMS.inpaint_conditioning = True
            tbg.PARAMS.point_grid_image_stabilizer_experimental = 0
            tbg.PARAMS.memorize = 'OFF'
            tbg.SIZE.inpaint_max = 0.05

            tbg.PARAMS.LanPaint = True
            tbg.PARAMS.Preview_Tiles_in_Temp_Folder = False
            inpaint_end = 0
            tbg.KSAMPLER.sampler_input = None
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


        tbg.lowvram = kwargs.get('Low_Vram', True)



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
        tbg.PARAMS.color_match_method = kwargs.get('Color_Match', 'none')
        tbg.PARAMS.color_match_str = kwargs.get('Color_Match_Str', 1)
        tbg.PARAMS.tiles_to_process_active = kwargs.get('Selected_Tiles_Only', False)
        tbg.PARAMS.Selected_Tiles_By_Numbers =  kwargs.get('Selected_Tiles_By_Numbers', '')

        tbg.PARAMS.tiles_to_process = WORKER.id(tiler_id).TBG_Image.set_tiles_to_process(tbg.PARAMS.tiles_to_process_active,
                                              len(tbg.OUTPUTS.grid_images_all),
                                              tbg.PARAMS.Selected_Tiles_By_Numbers, False, _tbg_send_images = False)
        # tiles_to_process and Fast_1_Tile_Preview
        if tbg.PARAMS.Fast_1_Tile_Preview:
            tbg.PARAMS.tiles_to_process_active = True  # Get First from list


        tbg.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        tbg.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        tbg.KSAMPLER.steps = kwargs.get('steps', None)
        tbg.KSAMPLER.cfg = kwargs.get('cfg', None)
        tbg.KSAMPLER.denoise = kwargs.get('denoise', None)
        tbg.KSAMPLER.cnet_multiply = kwargs.get('Controlnet_Pipe_strength', 1)
        tbg.KSAMPLER.noise_seed = kwargs.get('seed', None)
        tbg.KSAMPLER.General_Prompt = kwargs.get('General_Prompt_Positive', "")
        tbg.KSAMPLER.General_Prompt_Negative = kwargs.get('General_Prompt_Negative', "低质量，模糊，噪点，失焦，曝光不良，过度曝光，欠曝光，重影，漂浮的物体，穿模，错误的结构，解剖错误，多余的肢体，多余的手指，缺少手指，手指融合，肢体融合，奇怪的骨骼，扭曲的身体，不自然的姿势，不自然的动作，不对称，身体比例不正确，脸部变形，重复的脸，五官错位，眼睛不对称，视线错误，面部畸形，表情僵硬，卡通化，非真实皮肤纹理，塑料感皮肤，过度光滑，噪点伪影，阴影错误，光照不一致，颜色溢出，奇怪的反射，重复的图案，破碎结构，AI 痕迹，水印，文字，logo，二维码，杂乱背景，物体穿插，图像缺损，像素化，低分辨率，乱色块，扭曲纹理，异常的毛发，不自然的布料褶皱，边缘锯齿，锐化过度，发光边缘，异常色彩，噪声纹理")
        tbg.KSAMPLER.Flux_Guidance = kwargs.get('Flux_Guidance', None)
        tbg.KSAMPLER.Controlnet_Pipe = kwargs.get('Controlnet_Pipe', None)
        tbg.KSAMPLER.model_type = kwargs.get('model_type', None)
        tbg.KSAMPLER.model = kwargs.get('model', None)
        tbg.KSAMPLER.clip = kwargs.get('clip', None)
        tbg.KSAMPLER.vae = kwargs.get('vae', None)
        tbg.KSAMPLER.tiled = kwargs.get('vae_encode', None)
        tbg.KSAMPLER.Enrichment_Pipe = kwargs.get('Enrichment_Pipe', None)
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

        storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.grid_images_all) # now input images are saved here to

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


    @staticmethod
    def image_to_folder(image, filename):
        if image is not None:
            filename_prefix = "TBG/compareTiles/"+ filename
            preview = nodes.PreviewImage()
            _ = preview.save_images(image, filename_prefix, None, None)['ui']['images']
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

                if tbg.KSAMPLER.Controlnet_Pipe:
                    # build from Cnet stitched and chaind referent latent combination
                    positive = get_Kontext_stiched_o_chained_cond(tbg, positive, tbg.KSAMPLER.Controlnet_Pipe, tile_to_process)
                else:
                    # only feed tile as referent latent
                    kontext_latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, tile_to_process)[0]
                    positive = append_reference_latent(positive, kontext_latent_image)
            else:
                # FLUX standard conditioning (no Kontext)

                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")

            negative = nodes.ConditioningZeroOut.zero_out(0, positive)[0]
            positive = FluxGuidance_execute(positive, tbg.KSAMPLER.Flux_Guidance)[0]

        # Qwen Edit conditioning
        elif tbg.KSAMPLER.model_type in {"Qwen Image", "Qwen Image Edit"}:
            if tbg.KSAMPLER.model_type == "Qwen Image Edit":
                positive, negative = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(tile_index=index)
                if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                    pos_low, neg_low = cls.VRAM_OPTIMIZER.unified_condition_to_gpu(index, "low")
                if tbg.KSAMPLER.Controlnet_Pipe:
                    # build from Cnet stitched and chaind referent latent combination
                    positive = get_qwen_stiched_o_chained_cond(tbg, positive, tbg.KSAMPLER.Controlnet_Pipe, tile_to_process)
                else:
                    # only feed tile as referent latent
                    kontext_latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, tile_to_process)[0]
                    positive = append_reference_latent(positive, kontext_latent_image)
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
    def sigmas(cls, iteration, index):
            # PRO Step 3.5.1 Sigmas
            if tbg.KSAMPLER.custom_sigmas is not None:
                # use custom sigmas from input node should have denoise 1
                sigmas = tbg.KSAMPLER.custom_sigmas #* tbg.PROMPTER.output_denoises[index]
                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None,
                    f"Node {tbg.INFO.id} - Custom Sigmas Loaded {iteration}")
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
            else:
                denoise = tbg.KSAMPLER.denoise
                sigmas = denoise_sigmas_tgb(sigmas, tbg.KSAMPLER.denoise, tbg.PARAMS.denoise_method, tbg.KSAMPLER.model, tbg.KSAMPLER.scheduler)

            return  (denoise, sigmas)
    @classmethod
    def refine(cls, image, iteration):

        inpaintmask = None
        fusion_segment_tiles = []
        tbg.PARAMS.SegFusion_Initializer_run_once = False  # has to be false on run so that the first called seg triggers learning surroundings for all segs
        tbg.full_image_only_tiles = None
        temp_output_images = None
        output_image_new = None

        # ------------------------------------------------------------------
        # Step 1 prepare tile arrays
        # ------------------------------------------------------------------

        cls.prepare_tiles_to_process()  # return tbg.OUTPUTS.grid_images_all and tbg.PARAMS.tiles_to_process
        # ------------------------------------------------------------------
        # Step 2 VRAM Optimization: precompute all embeddings
        # ------------------------------------------------------------------

        cls.precompute_all_embeddings_free_VRAM()
        # ------------------------------------------------------------------
        # Step 3 TBG Magic tile Loop
        # ------------------------------------------------------------------

        # Detect what changed
        # change_type, changed_indices, change_msg = cls._detect_changes()
        import traceback

        try:
            change_type, changed_indices, change_msg = cls._detect_changes()
            log(f"{change_msg}", None, None, f"Node {tbg.INFO.id}")


        except Exception as e:

            change_type = "ALL"
            changed_indices = set()
            change_msg = "ALL"
            err = traceback.format_exc()

            #log(f"_detect_changes failed, reset to ALL\n{err}", None, None, f"Node {tbg.INFO.id}")

        tbg.TEMP.change_type = change_type
        tbg.TEMP.changed_indices = changed_indices
        tbg.TEMP.change_msg = change_msg


        # Pass  storage["generated_tiles"] to OUTPUTS so its will be sent with all other images to the WORKER
        storage = persistent_storage[tbg.storage_key]
        tbg.OUTPUTS.persistent_generated_tiles = storage["generated_tiles"]

        worker_params = SimpleNamespace(**vars(tbg.PARAMS))
        worker_params.Redux_Style_Model = None
        worker_params.Redux_Clip_Vision = None
        output_image_new, output_image_only_tiles, output_image_noCC = WORKER.id(tiler_id).ETUR.refiner_init(worker_params, tbg.SIZE)
        # output_image_new, output_image_only_tiles, output_image_noCC = WORKER.id(tiler_id).ETUR.refiner_init(tbg.PARAMS, tbg.SIZE)
        
        # Always update cache (needed for incremental processing)
        # tbg.OUTPUTS.generated_tiles has no infos at this point
        # storage = persistent_storage[tbg.storage_key]
        #storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.generated_tiles)

        # TBG_APP.py, end of refine()
        # We save a copy of the final Background Images - this images will be loaded if WORKER shuts down and need to have an Inputimage for Selective Tile generation of the last final Result. It will build the Tiles out of this image
        if output_image_only_tiles is not None:
            tbg.OUTPUTS.last_final_image = output_image_only_tiles.clone()
        return output_image_new, output_image_only_tiles, output_image_noCC

    @classmethod
    def sampling(cls, index, index_seg, tile_to_process, innerloop_scale_factor, inpaintmask, Complexity_Mask, tiler_id, border_correction_mask): #pre_border_correction_mask,accumulating_background_image):
                tbg = get_tbg(tiler_id)
                with ((torch.inference_mode(True))):
                    if tbg.API.status == "Dev":
                        cls.image_to_folder(tile_to_process, str(index) + "Tile before Sampling")
                        cls.image_to_folder(MaskToImage_execute(inpaintmask)[0], str(index) + "Inpaint Mask before Sampling")
                        cls.image_to_folder(MaskToImage_execute(Complexity_Mask)[0], str(index) + "Complexity_Mask before Sampling"  )


                    if tbg.API.status == "Dev":
                        cls.image_to_folder(tile_to_process,
                                            str(index) + "tile_to_process-with-pre_Border_Correction_Mask")



                    iteration = "TBG-ETUR"
                    # ------------------------------------------------------------------
                    # 3.4  Inner Upscale
                    # ------------------------------------------------------------------

                    tile_to_process_H = tile_to_process.shape[1]
                    tile_to_process_W = tile_to_process.shape[2]

                    if tbg.PARAMS.inner_Upscale_type == 'finer details' and innerloop_scale_factor not in (0, 1):
                        tile_to_process = TBG_Image().helper_upscaleimage(tile_to_process, tbg.PARAMS.upscale_method_inpainting, tbg.PARAMS.upscale_model_inpainting,innerloop_scale_factor)



                    # ------------------------------------------------------------------
                    # 3.5 Sigmas
                    # ------------------------------------------------------------------

                    denoise, sigmas = cls.sigmas(iteration, index)

                    # ------------------------------------------------------------------
                    # 3.5 Conditioning positive negative / cropped positive negative
                    # ------------------------------------------------------------------

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

                    if (
                            # First condition block Neuro_Generative_Tile_Fusion Always
                            (
                                #tbg.SEGMENTS.segms and
                                #len(tbg.SEGMENTS.segms[0]) and
                                #len(tbg.OUTPUTS.orig_grid_images) - index > 0 and # only tiles
                                tbg.PARAMS.Tile_Fusion_Mode in ("Neuro_Generative_Tile_Fusion", "Tile_Fusion") and
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

                        if tbg.PARAMS.Differential_Diffusion:
                            tbg.KSAMPLER.model = DifferentialDiffusion_execute(tbg.KSAMPLER.model)[0]

                        if  tbg.PARAMS.inpaint_conditioning :
                            InpaintModelConditioningNode = nodes.InpaintModelConditioning()
                            positive, negative, latent_image = InpaintModelConditioningNode.encode(positive, negative,
                                                                                                       tile_to_process,
                                                                                                       tbg.KSAMPLER.vae,
                                                                                                       Complexity_Mask,
                                                                                                       noise_mask=True)

                            if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                                pos_low, neg_low , latent_image = InpaintModelConditioningNode.encode(positive, negative,
                                                                                                       tile_to_process,
                                                                                                       tbg.KSAMPLER.vae,
                                                                                                       tbg.DUALMODEL.Complexity_Mask,
                                                                                                       noise_mask=True)
                        else:
                            latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, tile_to_process)[0]
                            latent_image["noise_mask"] = Complexity_Mask.reshape((-1, 1, Complexity_Mask.shape[-2], Complexity_Mask.shape[-1]))


                    else:
                        if tbg.KSAMPLER.tiled:
                            latent_image = nodes.VAEEncodeTiled().encode(tbg.KSAMPLER.vae, tile_to_process, tbg.SIZE.tile_size_vae, tbg.SIZE.tile_size_vae // 4, tbg.SIZE.tile_size_vae // 4)[0]
                        else:
                            latent_image = nodes.VAEEncode().encode(tbg.KSAMPLER.vae, tile_to_process)[0]


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
                    #saveguard
                    if tbg.PARAMS.LanPaint:
                        TBG_DualModelSampler = TBG_DualModelSampler_lanpaint,
                        TBG_KSamplerAdvancedSplitAware = TBG_KSamplerAdvancedSplitAware_lanpaint
                    else:
                        TBG_DualModelSampler = TBG_DualModelSampler_normal
                        TBG_KSamplerAdvancedSplitAware = TBG_KSamplerAdvancedSplitAware_normal
                    if denoise != 0:
                        if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:

                            Dualmodel_Sampler = TBG_DualModelSampler.sample
                            latent_output = Dualmodel_Sampler(0,
                                tbg.DUALMODEL.inpaint_end,
                                tbg.DUALMODEL.smoother_sharper,
                                tbg.DUALMODEL.detail_enhancer,
                                tbg.KSAMPLER.model,
                                tbg.DUALMODEL.model,
                                tbg.PROMPTER.output_seeds_js[index], #tbg.KSAMPLER.noise_seed,
                                tbg.KSAMPLER.cfg,
                                tbg.DUALMODEL.cfg,
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

                            TBG_KSampler = TBG_KSamplerAdvancedSplitAware.sample
                            result = TBG_KSampler(0,
                                tbg.KSAMPLER.model,
                                True, # add noise
                                tbg.PROMPTER.output_seeds_js[index], #tbg.KSAMPLER.noise_seed,
                                tbg.KSAMPLER.steps,
                                tbg.KSAMPLER.cfg,
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

                    #-End sampling-------------------------------------------------------------------------------------------------------------------
                    
        # VAE decode
                    # -VAE to RAM------------------------------------------------------------------------------------------------------------------
                    
                    if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
                        vaedecoder = tbg.DUALMODEL.vae
                    else:
                        vaedecoder = tbg.KSAMPLER.vae

        
                    if len(tbg.PARAMS.tiles_to_process) == 0 or index in tbg.PARAMS.tiles_to_process:
                        # Downscale to Original size


                       # decode all
                        if tbg.KSAMPLER.tiled:
                            if tbg.debug:
                                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None,f"Node {tbg.INFO.id} - VAEDecodingTiled ")
                            tile_processed = (nodes.VAEDecodeTiled().decode(vaedecoder, latent_output,tbg.SIZE.tile_size_vae,tbg.SIZE.tile_size_vae // 4, tbg.SIZE.tile_size_vae // 4)[0].unsqueeze(0))[0]

                        else:
                            if tbg.debug:
                                log(f"tile {index + 1}/{len(tbg.OUTPUTS.grid_images_all)}", None, None, f"Node {tbg.INFO.id} - VAEDecodingNormalized")

                            tile_processed = (TBG_Image.VAEDecodeFluxNormalized(vaedecoder, latent_output)[0].unsqueeze(0))[0]

                        if tbg.API.status == "Dev":
                            cls.image_to_folder(tile_processed, str(index)+"_VAE_decode_after_sampling")


                        if tbg.PARAMS.inner_Upscale_type == 'finer details' and innerloop_scale_factor not in (0,1):

                            tile_to_process = TBG_Image().helper_upscaleimage(tile_to_process, tbg.PARAMS.upscale_method_inpainting,
                                                                                     tbg.PARAMS.upscale_model_inpainting,0,round(tile_to_process_W), round(tile_to_process_H))
                            tile_processed = TBG_Image().helper_upscaleimage(tile_processed, round(tile_to_process_W), round(tile_to_process_H),
                                                                                     tbg.PARAMS.upscale_method_inpainting,
                                                                                     tbg.PARAMS.upscale_model_inpainting)

                        # the latent en and decode VAE is producing a color even if the fusion is perfect this can produce a seam, so we need to correct this with a simple blend in the border region
                        
                    
                    
                    
                    
                        if tbg.PARAMS.Tile_Fusion_Mode in ("Neuro_Generative_Tile_Fusion", "Tile_Fusion") and not tbg.PARAMS.Fast_1_Tile_Preview:
                            # check if this s inpaint mask is tested but the middel is to different also it has bottom and right
                            # we try to generate a map only top left


                            # This step is very important - because the sampler is not maintaining colors so the fusion crop has a color seam - with this blend this is solved.
                            tile_processed = ImageCompositeMasked_execute(tile_to_process, tile_processed, x=0, y=0, resize_source=False, mask=border_correction_mask)[0]
                        else:
                            tile_processed = tile_processed # if soft merge or tbg.PARAMS.Fast_1_Tile_Preview
                        
                        if tbg.API.status == "Dev":

                            cls.image_to_folder(tile_processed,
                                                str(index) + "tile_processed")

                            cls.image_to_folder(MaskToImage_execute(border_correction_mask)[0], str(index) + "_Masked_Border_Correction_Mask")

                    #tile_processed = tile_to_process
                    # Save Tile to Temp
                    if tbg.PARAMS.Preview_Tiles_in_Temp_Folder:
                        if tbg.API.status == "Dev":
                            cls.image_to_folder(tile_processed, str(index) + "_Masked_Border_Correction")
                        else:
                            cls.image_to_folder(tile_processed, str(index) + "_CT")


                    tbg.OUTPUTS.grid_images_all[index] = tile_processed
                    storage = persistent_storage[tbg.storage_key]
                    storage["generated_tiles"] = copy.deepcopy(tbg.OUTPUTS.grid_images_all)
                    return tile_processed

# Refiner end --------------------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def _detect_changes(cls):
        """
        Detect what changed in the node inputs.
        Returns: (change_type, changed_tile_indices, message)
        """
        storage = persistent_storage.get(tbg.storage_key, {})

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
            if mode == "Neuro_Generative_Tile_Fusion":
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

