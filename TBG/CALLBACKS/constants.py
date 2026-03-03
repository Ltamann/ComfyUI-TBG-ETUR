# comfyui SIDE


from types import SimpleNamespace

import torch
from comfy import model_management

device = model_management.get_torch_device()

from ...TBG.SERVERS.COMFYUI_server import register_main_class


# --- NEW: shared memory support ---
import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass


@dataclass
class SharedArrayRef:
    """
    Shared-memory description for one array.
    path: logical path inside tbg, e.g. "INPUTS.image" or "OUTPUTS.grid_images_all.0"
    """
    name: str
    shape: tuple
    dtype: str





class tbg:
    def __init__(self):
        self._shared_meta = {}
        self._shared_handles = {}  # keep SharedMemory objects alive
        self.WORKER_shutdown_timer = None  # threading.Timer or None
        self.WORKER_last_activity = 0.0  # timestamp
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


        self.TEMP = SimpleNamespace(
            latent_index=None,
            change_type="ALL",
            changed_indices = set(),
            change_msg = "ALL",
            last_tiler_config=None,
        )


        self.INFO = SimpleNamespace(
            id=None,
            tiler_id=None,
        )

        self.INPUTS = SimpleNamespace(
            image=None, # upscaled image
        )

        self.OUTPUTS = SimpleNamespace(
            upscaled_image=None, # in tiler is this the Upscaled image corrected to 64 pixel  # not used in TBG APP
            denoise_mask_tiles = None,
            orig_grid_images_all =  [],  # ref of original
            grid_images_all =  [],  # ref of generated rest original

            persistent_generated_tiles = [],

            last_final_image = None, # needed for selected tile generation

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
            len_grid_images = 0,
            len_segments = 0,
            TBG_APP_ShutDown = 'Close with Comfyui',
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
                                          'device': device, 'offload_device': 'cpu', 'cache_model': True,
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
            upscale_factor=None,
            pad_offset=None,
            segment_tiles=None,
            orig_segment_tiles=None,
            segms_scale=None,
            segms_cropped_masks=None,
            segms_crop_regions=None,
            segms_new=None,
            Segment_Mask=None,
            inpainting_mask=None,
            compositing_mask=None,
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
    def build_shared_meta(self):
        """
        Scan the entire tbg structure and, for every NumPy/torch array
        found, create a shared-memory copy and record it in self._shared_meta.

        Called before each worker RPC to give the worker a fresh snapshot.
        """
        from types import SimpleNamespace as _SN

        # Clean up any previous shared segments for this tiler
        for shm in getattr(self, "_shared_handles", {}).values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shared_handles = {}
        self._shared_meta = {}

        def is_array(x):
            # treat tensors and NumPy arrays as shareable arrays
            return isinstance(x, (torch.Tensor, np.ndarray))

        def store_array_old(x, path: str):
            if hasattr(x, "numpy"):
                np_arr = x.detach().cpu().numpy()
            else:
                np_arr = np.asarray(x)

            shm = shared_memory.SharedMemory(create=True, size=np_arr.nbytes)
            buf = np.ndarray(np_arr.shape, dtype=np_arr.dtype, buffer=shm.buf)
            buf[:] = np_arr[:]

            self._shared_handles[path] = shm

            self._shared_meta[path] = SharedArrayRef(
                name=shm.name,
                shape=tuple(np_arr.shape),
                dtype=str(np_arr.dtype),
            )

        # this is confertin float32 to 16 for faster trasnport and OOM issues
        def store_array_midway(x, path: str):
            """
            Convert a torch/NumPy array to a SharedMemory-backed NumPy array.
            Tiered compression for images:
            1. float32 → float16 (2x smaller)
            2. If still >4GiB: → uint8 (4x smaller total)
            Size limit checked AFTER final compression.
            """
            # Detect images by path
            is_image_path = (
                    path.startswith("INPUTS.image") or
                    path.startswith("OUTPUTS.grid_images_all")
            )

            # Step 1: torch → CPU numpy
            if isinstance(x, torch.Tensor):
                t = x.detach().cpu()
                original_bytes = t.element_size() * t.nelement()
                np_arr = t.numpy()
            else:
                np_arr = np.asarray(x)
                original_bytes = np_arr.nbytes

            original_size_gb = original_bytes / (1024 ** 3)

            # Step 2: Tier 1 - float16 compression for images
            was_compressed = False
            if is_image_path and np_arr.dtype == np.float32:
                np_arr = np_arr.astype(np.float16)
                float16_bytes = np_arr.nbytes
                """
                print(
                    f"[TBG_MAIN] Tier1 '{path}': {original_size_gb:.2f}GiB float32 → "
                    f"{float16_bytes / (1024 ** 3):.2f}GiB float16"
                )
                """
                was_compressed = True

            # Step 3: Tier 2 - uint8 if still too big (>4GiB)
            nbytes = np_arr.nbytes
            if nbytes > 4 * 1024 ** 3 and is_image_path:
                # Clamp to [0,255] and convert to uint8
                np_arr = np.clip(np_arr, 0, 255).astype(np.uint8)
                uint8_bytes = np_arr.nbytes
                """
                print(
                    f"[TBG_MAIN] Tier2 '{path}': {nbytes / (1024 ** 3):.2f}GiB → "
                    f"{uint8_bytes / (1024 ** 3):.2f}GiB uint8 (clamped [0,255])"
                )
                """
                was_compressed = True

            final_bytes = np_arr.nbytes
            final_size_gb = final_bytes / (1024 ** 3)

            # Step 4: Final size check & share
            MAX_SHARED_ARRAY_BYTES = 5 * 1024 * 1024 * 1024
            if final_bytes > MAX_SHARED_ARRAY_BYTES:
                """
                print(
                    f"[TBG_MAIN] FINAL SKIP '{path}': {final_size_gb:.2f}GiB "
                    f"> {MAX_SHARED_ARRAY_BYTES / (1024 ** 3):.2f}GiB limit "
                    f"(orig: {original_size_gb:.2f}GiB → uint8: {final_size_gb:.2f}GiB)"
                )
                """
                return

            # Share success
            shm = shared_memory.SharedMemory(create=True, size=final_bytes)
            buf = np.ndarray(np_arr.shape, dtype=np_arr.dtype, buffer=shm.buf)
            buf[:] = np_arr[:]

            self._shared_handles[path] = shm
            self._shared_meta[path] = SharedArrayRef(
                name=shm.name,
                shape=tuple(np_arr.shape),
                dtype=str(np_arr.dtype),
            )

            #if was_compressed:
                #print(f"[TBG_MAIN] ✓ Shared '{path}' ({final_size_gb:.2f}GiB {np_arr.dtype})")

        def store_array(x, path: str):
            """
            Tiered compression based on ORIGINAL size:
            ≤4GiB:     Share float32 as-is
            >4GiB:     → float16
            float16>4GiB: → uint8
            uint8>4GiB: Skip
            """
            is_image_path = (
                    path.startswith("INPUTS.image") or
                    path.startswith("OUTPUTS.grid_images_all")
            )

            # Get numpy + original size
            if isinstance(x, torch.Tensor):
                np_arr = x.detach().cpu().numpy()
            else:
                np_arr = np.asarray(x)

            original_bytes = np_arr.nbytes
            original_size_gb = original_bytes / (1024 ** 3)

            # Tier 1: ≤4GiB → share float32 as-is
            FOUR_GB = 5 * 1024 ** 3
            if original_bytes <= FOUR_GB:
                final_arr = np_arr
                #print(f"[TBG_MAIN] Direct share '{path}' ({original_size_gb:.2f}GiB float32)")
            else:
                # Tier 2: >4GiB → float16
                np_arr_f16 = np_arr.astype(np.float16)
                f16_bytes = np_arr_f16.nbytes
                f16_size_gb = f16_bytes / (1024 ** 3)

                #print(f"[TBG_MAIN] Tier1 '{path}': {original_size_gb:.2f}GiB → {f16_size_gb:.2f}GiB float16")

                if f16_bytes <= FOUR_GB:
                    final_arr = np_arr_f16
                else:
                    # Tier 3: float16 still >4GiB → uint8
                    np_arr_u8 = np.clip(np_arr_f16, 0, 1).astype(np.uint8) * 255
                    u8_bytes = np_arr_u8.nbytes
                    u8_size_gb = u8_bytes / (1024 ** 3)

                    #print(f"[TBG_MAIN] Tier2 '{path}': {f16_size_gb:.2f}GiB → {u8_size_gb:.2f}GiB uint8")

                    if u8_bytes <= FOUR_GB:
                        final_arr = np_arr_u8
                    else:
                        # Tier 4: Skip absolute giants
                        #print(f"[TBG_MAIN] SKIP '{path}': uint8 {u8_size_gb:.2f}GiB > 4GiB")
                        return

            # Share final array
            final_bytes = final_arr.nbytes
            shm = shared_memory.SharedMemory(create=True, size=final_bytes)
            buf = np.ndarray(final_arr.shape, dtype=final_arr.dtype, buffer=shm.buf)
            buf[:] = final_arr[:]

            self._shared_handles[path] = shm
            self._shared_meta[path] = SharedArrayRef(
                name=shm.name,
                shape=tuple(final_arr.shape),
                dtype=str(final_arr.dtype),
            )

            print(f"[TBG_MAIN] → [TBG APP] '{path}' ({final_arr.nbytes / (1024 ** 3):.2f}GiB {final_arr.dtype})")

        # Replace your existing store_array_midway implementation with this one.
        # Make sure this method is inside the same class that defines
        # self._shared_handles and self._shared_meta.



        def recurse(obj, base_path: str):
            # 0. If this object is an array (torch / numpy), store it as a leaf.
            if base_path and is_array(obj):
                store_array(obj, base_path)
                return

            # 1. Objects with attributes (tbg instance, nested SimpleNamespace, etc.)
            if isinstance(obj, _SN) or hasattr(obj, "__dict__"):
                for name, val in vars(obj).items():
                    # BYPASS these two elements under SEGMENTS
                    if base_path == "SEGMENTS" and name in ("segms", "segms_new"):
                        continue
                    if base_path in ("TEMP",
                                    "INFO",
                                    "LLM",
                                    "API",
                                    "DUALMODEL",
                                    "PARAMS",
                                    "KSAMPLER",
                                    "SIZE",
                                    "CACHE"):
                        continue
                    sub = f"{base_path}.{name}" if base_path else name
                    recurse(val, sub)

            # 2. Lists / tuples
            elif isinstance(obj, (list, tuple)):
                for idx, val in enumerate(obj):
                    sub = f"{base_path}.{idx}" if base_path else str(idx)
                    recurse(val, sub)

            # 3. Other leaves are ignored (not arrays)
            else:
                return

        recurse(self, "")

from threading import Lock

from threading import Lock

_TBG_INSTANCES = {}
_TBG_LOCK = Lock()

def get_tbg(tiler_id):
    """Return a dedicated tbg instance per tiler_id."""
    key = str(tiler_id)
    with _TBG_LOCK:
        if key not in _TBG_INSTANCES:
            inst = tbg() # tbg is your class
            inst.INFO.id = tiler_id # store for debugging / storagekey
            _TBG_INSTANCES[key] = inst
        return _TBG_INSTANCES[key]

def reset_tbg(tiler_id):
    """Drop the cached TBG state so a new one is created on next get_tbg()."""
    if tiler_id in _TBG_INSTANCES:
        del _TBG_INSTANCES[tiler_id]

@register_main_class
class TBGState:
    @classmethod
    def get(cls, path: str, tiler_id=None):
        """
        Read a nested attribute from the per-tiler TBG state.
        """
        TBG = get_tbg(tiler_id if tiler_id is not None else "default")
        obj = TBG
        for name in path.split("."):
            obj = getattr(obj, name)
        return obj

