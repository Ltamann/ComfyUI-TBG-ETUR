# TBG_APP side
from types import SimpleNamespace
import numpy as np
from multiprocessing import shared_memory
import torch
from contextvars import ContextVar
from threading import Lock
from comfy import model_management

device = model_management.get_torch_device()



class tbg:
    def __init__(self):
        self.storage_key = None
        self.persistent_storage = {}
        self.debug = False
        self.lowvram =False
        self.temp_output_dir = ""
        try:
            import flash_attn
            attention_mode = 'flash_attn'
        except ImportError:
            attention_mode = 'sdpa'

        self.upscale_models = ["NONE",
                              "FAST/_area",
                              "FAST/_bicubic",
                              "FAST/_bilinear",
                              "FAST/_bislerp",
                              "FAST/_lanczos",
                              "FAST/_nearest-exact",
                              "SuperResolution/Tiled-SeedVR2 Fast",
                              "SuperResolution/Tiled-SeedVR2 Standard",
                              "SuperResolution/Tiled-SeedVR2 High",
                              "SuperResolution/Tiled-SeedVR2 Ultra",
                              "SuperResolution/FlashVSR-v1.1 Small 8GB",
                              "SuperResolution/FlashVSR-v1.1 Big 18GB", "Waifu/art",
                              "Waifu/art noise 1",
                              "Waifu/art noise 2",
                              "Waifu/art noise 3",
                              "Waifu/photo",
                              "Waifu/photo noise 1",
                              "Waifu/photo noise 2",
                              "Waifu/photo noise 3"]

        self.BACKGROUND = SimpleNamespace(
            accumulated_image=None,  # Full-size background with all processed tiles
            background_mask=None,  # Mask tracking which regions have content
            processed_tiles=set(),  # Track which tiles are in background (for selected tiles mode)
            last_processed_index=-1  # Last tile added to background
        )

        self.TEMP = SimpleNamespace(
            latent_index=None,
            change_type="ALL",
            changed_indices=set(),
            change_msg="ALL",
        )

        self.INFO = SimpleNamespace(
            id=None,
            tiler_id=None,
        )

        self.INPUTS = SimpleNamespace(
            image=None,
        )

        self.OUTPUTS = SimpleNamespace(
            upscaled_image=None,  # in tiler is this the Upscaled image corrected to 64 pixel  # not used in TBG APP
            denoise_mask_tiles=None,
            orig_grid_images_all=[],  # ref of original
            grid_images_all=[],  # ref of generated rest original

            persistent_generated_tiles=[],

            last_final_image=None,  # needed for selected tile generation

        )

        self.PROMPTER = SimpleNamespace(
            tiler_prompts=None,
            output_prompts=None,
            output_denoises=None,
            output_seeds_js=None,
            output_cnet_js=None,
            tiles_to_process=None,
            cache_key=None,
            Prompt_Selected_Tiles_Only=None,
            Prompt_Selected_Tiles_By_Numbers=None,
            Rebuild_only_modified_tiles=False,
        )
        self.LLM = SimpleNamespace(
            vision_model=None,
            prompt=None,
            quantization=None,
            model=None,
        )

        self.API = SimpleNamespace(
            token=None,
            status="Guest",
            info=None,
            activate_pro=False,
            creditsleft=0,
            current_credits=0,
            info_url=None,
        )

        self.DUALMODEL = SimpleNamespace(
            steps=None,
            cfg=None,
            model=None,
            clip=None,
            vae=None,
            high_low_swap=None,
            low_refiner=None,
            General_Prompt=None,
            General_Prompt_Negative=None,
            model_crossover_sigma_strength=None,
            inpaint_end=None,
            smoother_sharper=None,
            detail_enhancer=None,
        )

        self.PARAMS = SimpleNamespace(
            len_grid_images=0,
            len_segments=0,
            TBG_APP_ShutDown='Close with Comfyui',
            Prompt_seed=0,
            SegFusion_Initializer_run_once=False,
            Inventivity=None,
            Resemblance=None,
            Fractality=None,
            Creativity=None,
            timestamp=None,
            Differential_Diffusion=None,
            Alternative_ImageNone=None,
            Tile_Fusion_Blend=None,
            inpaint_conditioning=None,
            point_grid_image_stabilizer_experimental=None,
            memorize=None,
            LanPaint=None,
            Preview_Tiles_in_Temp_Folder=None,
            stitch_blending=None,
            max_upscale_size_segment_inpainting=2048,
            PRO_Fusion_Complexity_Mask_Strength=None,
            PRO_Per_Pixel_Denoise_Mask_Strength=None,
            denoise_mask=None,
            Redux_strength=None,
            contrast=None,
            highpass=None,
            denoise_method='default',
            Fast_1_Tile_Preview=None,
            Redux_Style_Model=None,
            Redux_Clip_Vision=None,
            Laplacian_Pyramid_Blending=None,
            color_match_method=None,
            color_match_str=None,
            tiles_to_process_active=None,
            tiles_to_process=None,
            Tile_Fusion_Mode=None,
            Refiner_Tile_Fusion_Mode=None,
            PRO_Fusion_Complexity_min_Denoise=0.2,
            PRO_Fusion_Complexity_max_Denoise=1,
            PRO_Fusion_Complexity_Mask_Blur=0,
            inner_Upscale_type=None,
            inner_Upscale_value=None,
            inner_Upscale_Segments=None,
            upscale_method_inpainting=None,
            upscale_model_inpainting=None,
            SEEDVR2_DIT ={'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                                                   'device': device, 'offload_device': 'none', 'cache_model': True,
                                                   'blocks_to_swap': 0,
                                                   'swap_io_components': False,
                                                   'attention_mode': attention_mode,
                                                   'torch_compile_args': 'reduce-overhead', 'node_id': '8'},
            SEEDVR2_VAE={'model': 'ema_vae_fp16.safetensors', 'device': device, 'offload_device': 'none',
                                    'cache_model': True, 'encode_tiled': True, 'encode_tile_size': 512,
                                    'encode_tile_overlap': 64,
                                    'decode_tiled': True,
                                    'decode_tile_size': 512, 'decode_tile_overlap': 64, 'tile_debug': 'false',
                                    'torch_compile_args': 'reduce-overhead', 'node_id': '9'},
            SEEDVR2_DIT_low = {'model': 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors',
                                          'device':device, 'offload_device': 'cpu', 'cache_model': True,
                                          'blocks_to_swap': 36, 'swap_io_components': False,
                                          'attention_mode': attention_mode, 'torch_compile_args': 'reduce-overhead',
                                          'node_id': '8'},
            FLUX_Kontext = False,
            Optimize_Tile_Size = 'Disabled',
            upscale_size_type = None,
            upscale_size = None,
            tile_prompting_active = False,
            grid_specs = None,
            MODEL_TYPE_SIZES = False,
            upscale_model_name = None,
            upscaler_method = None,
            upscale_by =  None,
            upscale_type = None,
            preset = "NONE",
            fragmentation = 1
            # tile_prompting_active = kwargs.get('tile_prompting_active', False),
        )



        self.KSAMPLER = SimpleNamespace(
            tiled = None,
            tile_size_vae = 256,  # LOWER LESS COLOR VARIANCE
            tiles_to_process_active = False,
            model = None,
            clip = None,
            vae = None,
            noise_seed = None,
            sampler_name = None,
            scheduler = None,
            add_noise = True,
            sigmas_type = "BasicScheduler",
            model_type = None,
            steps = None,
            cfg = None,
            denoise = None,
            control_net_name = None,
            control = None,
            custom_sigmas = None,
            sampler_input=None,
            cropped_positive=None,
            cropped_negative=None,
            cnet_multiply=None,
            General_Prompt=None,
            General_Prompt_Negative=None,
            Flux_Guidance=None,
            Controlnet_Pipe=None,
            Enrichment_Pipe=None,
            sampler=None,
            detail_daemon_active=None,
            detail_amount=None,
            detail_daemon_start=None,
            detail_daemon_end=None,
            detail_daemon_bias=None,
            detail_daemon_exponent=None,
            detail_daemon_start_offset=None,
            detail_daemon_end_offset=None,
            daemon_fade=None,
            daemon_smooth=None,
            daemon_cfg_scale=None,
            Resharpener_strength=None,
            Resharpener_active=None,
            resharpen_end=None,
            resharpen_start=None,
        )
        self.SIZE = SimpleNamespace(
            inpaint_max = 0.05,
            tile_size_vae=512,
            inpaint_border_margin =0,
            shift = 0,
            shifttl =0,
            inpaint_blur_margin = 0,
            composite_blur_margin = 0,
            crop_margin=0,
            tile_grid_W=1024,
            tile_grid_H=1024,
            rows_qty=1,
            cols_qty=1,
            outer_mask_area=0,
            overlay_between_tiles=0,
            fullW=1024,
            fullH=1024,
            UpscaledInputImageH=0,
            UpscaledInputImageW=0,
            Fusion_margin=64,
        )

        self.SEGMENTS = SimpleNamespace(
            segms=None,
            upscalefactor=None,
            padoffset=None,
            segment_tiles=[],  # must be a list, and name must match
            orig_segment_tiles=[],  # must be a list
            segms_scale=None,
            segms_cropped_masks=[],  # also lists if you share them
            segms_crop_regions=[],
            segms_new=None,
            Segment_Mask=None,
            inpainting_mask=[],
            compositing_mask=[],
            h=None,
            w=None,
        )


        self.CACHE = SimpleNamespace(
            prompt=None,
            prompt_edited=None,
            denoise=None,
            denoise_edited=None,
            seeds=None,
            seeds_edited=None,
            cnet=None,
            cnet_edited=None,
        )
# constants.py (TBG.TBGAPP.constants)


_CURRENT_TILER_ID = ContextVar("TBG_CURRENT_TILER_ID", default=None)

_TBG_INSTANCES = {}
_TBG_LOCK = Lock()


def set_current_tiler_id(tilerid):
    global _CURRENT_TILER_ID
    _CURRENT_TILER_ID.set(str(tilerid) if tilerid is not None else None)


def getcurrenttilerid():
    return _CURRENT_TILER_ID.get()


def get_tbg(tilerid):
    key = str(tilerid)
    with _TBG_LOCK:
        inst = _TBG_INSTANCES.get(key)
        if inst is None:
            inst = tbg()
            inst.INFO.id = tilerid
            _TBG_INSTANCES[key] = inst
        return _TBG_INSTANCES[key]


def _attach_path(root_obj, path: str, tensor):
    """
    Attach 'tensor' into root_obj following a path like
    'INPUTS.image' or 'OUTPUTS.grid_images_all.0'.
    """
    parts = path.split(".")
    cur = root_obj
    for i, part in enumerate(parts):
        last = (i == len(parts) - 1)
        if last:
            if part.isdigit():
                idx = int(part)
                while len(cur) <= idx:
                    cur.append(None)
                cur[idx] = tensor
            else:
                setattr(cur, part, tensor)
        else:
            if part.isdigit():
                raise RuntimeError(f"Unexpected numeric path '{part}' in '{path}'")
            cur = getattr(cur, part)

# Careful on each WORKER call oll Tensors are collected and added here to the Worker
# if Tiler has results of previous run so thees results get send first before the final new tiles are sent by the refiner - thats why we need to clear on each call the tensors.
# results are appended so old indexes could stay alive
def attach_shared_arrays_to_tbg(T, shared_meta):
    if not hasattr(T, "_shm_handles"):
        T._shm_handles = {}

    # Hard-reset critical containers every call (worker-side only)
    for ns_name, attrs in [
        ("OUTPUTS", [
            "grid_images_all",
            "orig_grid_images_all",
        ]),
        ("SEGMENTS", [
            "segms_crop_regions",
            "segms_cropped_masks",
            "segment_tiles",
            "orig_segment_tiles",
        ]),
    ]:
        ns = getattr(T, ns_name, None)
        if ns is not None:
            for attr in attrs:
                if hasattr(ns, attr):
                    setattr(ns, attr, [])  # force empty list each RPC

    cleared_targets = set()

    for path, info in shared_meta.items():
        try:
            _prepare_for_path(T, path, cleared_targets)

            name = info["name"]
            shm = T._shm_handles.get(name)
            if shm is None:
                shm = shared_memory.SharedMemory(name=name)
                T._shm_handles[name] = shm

            arr = np.ndarray(
                tuple(info["shape"]),
                dtype=np.dtype(info["dtype"]),
                buffer=shm.buf,
            )
            tensor = torch.from_numpy(arr)

            _attach_path(T, path, tensor)

        except Exception as e:
            print(f"[TBG_WORKER] attach_shared_arrays_to_tbg path={path} failed: {e}")

def _prepare_for_path(T, path: str, cleared_targets: set):
    """
    Clear the right container on T once per logical target,
    based on the path coming from shared_meta.
    """

    # Example: paths like "/INPUTS/image", "/OUTPUTS/grid_images_all/0"
    parts = [p for p in path.split("/") if p]

    if not parts:
        return

    # Top-level namespace: INPUTS, OUTPUTS, SEGMENTS, ...
    root = parts[0]

    # Handle simple tensors (no index)
    if len(parts) == 2:
        key = f"{root}.{parts[1]}"
        if key in cleared_targets:
            return

        # Clear/assign to a neutral value
        ns = getattr(T, root, None)
        if ns is not None and hasattr(ns, parts[1]):
            setattr(ns, parts[1], None)

        cleared_targets.add(key)
        return

    # Handle list-like attributes with indices,
    # e.g. "/OUTPUTS/grid_images_all/0", "/OUTPUTS/grid_images_all/1"
    if len(parts) >= 3 and parts[-1].isdigit():
        attr_name = parts[1]  # e.g. "grid_images_all"
        key = f"{root}.{attr_name}"

        if key in cleared_targets:
            return

        ns = getattr(T, root, None)
        if ns is not None and hasattr(ns, attr_name):
            # Always replace with a new empty list
            setattr(ns, attr_name, [])
        else:
            # Make sure the attribute exists as a list
            if ns is not None:
                setattr(ns, attr_name, [])

        cleared_targets.add(key)
        return

    # Fallback: nothing to do for unknown formats


def get_current_tbg():
    # Use the *value* stored in the ContextVar
    return get_tbg(getcurrenttilerid())

def get_current_tiler_id():
    return getcurrenttilerid()


class _TBGProxy:
    def __getattr__(self, name):
        # Always delegate to the current per-tiler instance
        t = get_current_tbg()
        if t is None:
            raise RuntimeError("No current TBG; set_current_tiler_id was not called")
        return getattr(t, name)

TBG = _TBGProxy()