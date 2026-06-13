import copy
from types import SimpleNamespace

import comfy.samplers
import nodes
import torch
import torch.nn.functional as F

from .inc.image import TBG_Image
from .inc.tbg_pid import (
    PID_DEFAULT_COLOR_MATCH,
    PID_DEFAULT_OVERLAP,
    PID_DEFAULT_SAMPLER,
    PID_VAE_COMPATIBLE_MODEL_TYPES,
    download_pid_model_bundle,
    get_pid_upscale_options,
    pid_model_name_for_model_type,
    run_pid_tiled_upscale,
)


SCALE_FACTOR = 4
INPUT_TILE_SIZE = 1024
OUTPUT_TILE_SIZE = INPUT_TILE_SIZE * SCALE_FACTOR
PID_SAMPLER_DEFAULT = PID_DEFAULT_SAMPLER
PID_SAMPLERS = (
    tuple(comfy.samplers.KSampler.SAMPLERS)
    if PID_SAMPLER_DEFAULT in comfy.samplers.KSampler.SAMPLERS
    else (PID_SAMPLER_DEFAULT, *tuple(comfy.samplers.KSampler.SAMPLERS))
)
PID_SCHEDULER_DEFAULT = "simple" if "simple" in comfy.samplers.KSampler.SCHEDULERS else comfy.samplers.KSampler.SCHEDULERS[0]


def _normalize_tiles(tiles):
    if tiles is None:
        raise ValueError("TBG PID upscale requires a Tiles input from the TBG ETUR refiner.")

    if torch.is_tensor(tiles):
        items = [tiles[i : i + 1] for i in range(tiles.shape[0])] if tiles.ndim == 4 else [tiles]
    elif isinstance(tiles, (list, tuple)):
        items = []
        for item in tiles:
            if item is None:
                items.append(None)
            elif torch.is_tensor(item) and item.ndim == 4 and item.shape[0] > 1:
                items.extend(item[i : i + 1] for i in range(item.shape[0]))
            else:
                items.append(item)
    else:
        raise ValueError(f"TBG PID upscale expected IMAGE tiles, got {type(tiles).__name__}.")

    if not items:
        raise ValueError("TBG PID upscale requires at least one tile.")

    normalized = []
    for index, tile in enumerate(items):
        if tile is None:
            normalized.append(None)
            continue
        if not torch.is_tensor(tile):
            raise ValueError(f"TBG PID upscale expected tensor tile at tile {index}, got {type(tile).__name__}.")
        if tile.ndim == 3:
            tile = tile.unsqueeze(0)
        if tile.ndim != 4:
            raise ValueError(f"TBG PID upscale expected [H,W,C] or [1,H,W,C] at tile {index}, got shape {tuple(tile.shape)}.")
        if tile.shape[0] != 1:
            raise ValueError(f"TBG PID upscale expected one image per tile at tile {index}, got batch {tile.shape[0]}.")
        height, width = int(tile.shape[1]), int(tile.shape[2])
        if width != INPUT_TILE_SIZE or height != INPUT_TILE_SIZE:
            raise ValueError(
                f"TBG PID upscale requires 1024x1024 input tiles, got {width}x{height} at tile {index}"
            )
        normalized.append(tile)

    return normalized


def _resize_image(image, width, height, mode="bilinear"):
    if image is None or not torch.is_tensor(image):
        return image
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        return image
    n, h, w, c = image.shape
    if h == height and w == width:
        return image
    x = image.movedim(-1, 1)
    if mode in {"nearest", "nearest-exact"}:
        x = F.interpolate(x, size=(height, width), mode="nearest")
    else:
        x = F.interpolate(x, size=(height, width), mode=mode, align_corners=False)
    return x.movedim(1, -1).reshape(n, height, width, c)


def _resize_mask(mask, width, height):
    if mask is None or not torch.is_tensor(mask):
        return mask
    original_ndim = mask.ndim
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        x = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        x = mask.movedim(-1, 1)
    else:
        return mask
    x = F.interpolate(x.float(), size=(height, width), mode="nearest")
    if original_ndim == 2:
        return x.squeeze(0).squeeze(0).to(mask.dtype)
    if original_ndim == 3:
        return x.squeeze(1).to(mask.dtype)
    return x.movedim(1, -1).to(mask.dtype)


def _resize_tensor_like(value, width, height):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 4 and value.shape[-1] in (1, 3, 4):
            return _resize_image(value, width, height)
        if value.ndim in (2, 3) or (value.ndim == 4 and value.shape[-1] == 1):
            return _resize_mask(value, width, height)
        return value
    if isinstance(value, list):
        return [_resize_tensor_like(v, width, height) for v in value]
    if isinstance(value, tuple):
        return tuple(_resize_tensor_like(v, width, height) for v in value)
    return value


def _copy_without_tensor_clone(value, memo=None):
    if memo is None:
        memo = {}
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]
    if torch.is_tensor(value):
        return value
    if isinstance(value, SimpleNamespace):
        copied = SimpleNamespace()
        memo[obj_id] = copied
        for key, item in vars(value).items():
            setattr(copied, key, _copy_without_tensor_clone(item, memo))
        return copied
    if isinstance(value, list):
        copied = []
        memo[obj_id] = copied
        copied.extend(_copy_without_tensor_clone(item, memo) for item in value)
        return copied
    if isinstance(value, tuple):
        copied = tuple(_copy_without_tensor_clone(item, memo) for item in value)
        memo[obj_id] = copied
        return copied
    if isinstance(value, dict):
        copied = {}
        memo[obj_id] = copied
        for key, item in value.items():
            copied[_copy_without_tensor_clone(key, memo)] = _copy_without_tensor_clone(item, memo)
        return copied
    if hasattr(value, "__dict__"):
        try:
            copied = copy.copy(value)
            memo[obj_id] = copied
            for key, item in vars(value).items():
                setattr(copied, key, _copy_without_tensor_clone(item, memo))
            return copied
        except Exception:
            return value
    return copy.deepcopy(value, memo)


def _scale_grid_specs(grid_specs, scale):
    if not grid_specs:
        return grid_specs
    scaled = []
    for spec in grid_specs:
        if not isinstance(spec, (list, tuple)) or len(spec) < 7:
            scaled.append(spec)
            continue
        values = list(spec)
        for i in (3, 4, 5, 6):
            values[i] = int(round(float(values[i]) * scale))
        scaled.append(tuple(values) if isinstance(spec, tuple) else values)
    return scaled


def _scale_namespace_numbers(ns, scale):
    if ns is None:
        return
    skip = {"rows_qty", "cols_qty", "len_grid_images", "len_segments", "Prompt_seed"}
    tokens = (
        "w",
        "h",
        "width",
        "height",
        "full",
        "tile",
        "crop",
        "margin",
        "blur",
        "shift",
        "overlay",
        "outer",
        "pad",
    )
    for name, value in list(vars(ns).items()):
        lname = name.lower()
        if name in skip or lname.endswith("_qty") or lname.endswith("count"):
            continue
        if not any(token in lname for token in tokens):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            setattr(ns, name, int(round(value * scale)))
        elif isinstance(value, float):
            setattr(ns, name, value * scale)


def _extract_segment_masks_and_regions(segms_new):
    crop_regions = []
    compositing_masks = []
    if segms_new is None:
        return crop_regions, compositing_masks
    try:
        _, segms = segms_new
    except Exception:
        return crop_regions, compositing_masks
    for seg in segms:
        if hasattr(seg, "crop_region"):
            crop_regions.append(tuple(int(round(v * SCALE_FACTOR)) for v in seg.crop_region))
        if hasattr(seg, "compositing_mask"):
            mask = seg.compositing_mask
            try:
                compositing_masks.append(_resize_mask(mask, mask.shape[-1] * SCALE_FACTOR, mask.shape[-2] * SCALE_FACTOR))
            except Exception:
                compositing_masks.append(mask)
    return crop_regions, compositing_masks


def _scale_pipe(tbg_pipe, pid_tiles):
    scaled_pipe = _copy_without_tensor_clone(tbg_pipe)
    names = ("INPUTS", "PARAMS", "KSAMPLER", "OUTPUTS", "SEGMENTS", "SIZE", "API", "PROMPTER")
    state = SimpleNamespace()
    for name, value in zip(names, scaled_pipe):
        setattr(state, name, value)

    _scale_namespace_numbers(getattr(state, "SIZE", None), SCALE_FACTOR)
    params = getattr(state, "PARAMS", None)
    outputs = getattr(state, "OUTPUTS", None)
    segments = getattr(state, "SEGMENTS", None)
    inputs = getattr(state, "INPUTS", None)

    if params is not None:
        params.grid_specs = _scale_grid_specs(getattr(params, "grid_specs", None), SCALE_FACTOR)
        params.stitch_blending = "gpupyramid"
        if getattr(params, "denoise_mask", None) is not None:
            mask = params.denoise_mask
            if torch.is_tensor(mask):
                params.denoise_mask = _resize_mask(mask, int(mask.shape[-1]) * SCALE_FACTOR, int(mask.shape[-2]) * SCALE_FACTOR)

    if outputs is not None:
        outputs.pid_input_tiles_1024 = _copy_without_tensor_clone(getattr(outputs, "grid_images_all", None))
        outputs.orig_grid_images_all = _copy_without_tensor_clone(getattr(outputs, "grid_images_all", None))
        outputs.grid_images_all = list(pid_tiles)
        outputs.denoise_mask_tiles = _resize_tensor_like(getattr(outputs, "denoise_mask_tiles", None), OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE)
        if getattr(outputs, "upscaled_image", None) is not None and torch.is_tensor(outputs.upscaled_image):
            outputs.upscaled_image = _resize_image(
                outputs.upscaled_image,
                int(outputs.upscaled_image.shape[2]) * SCALE_FACTOR,
                int(outputs.upscaled_image.shape[1]) * SCALE_FACTOR,
            )

    if inputs is not None and getattr(inputs, "image", None) is not None and torch.is_tensor(inputs.image):
        inputs.image = _resize_image(inputs.image, int(inputs.image.shape[2]) * SCALE_FACTOR, int(inputs.image.shape[1]) * SCALE_FACTOR)

    if segments is not None:
        for attr in ("segment_tiles", "orig_segment_tiles", "segms_cropped_masks", "inpainting_mask", "compositing_mask"):
            if hasattr(segments, attr):
                setattr(segments, attr, _resize_tensor_like(getattr(segments, attr), OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE))

    return tuple(
        getattr(state, name, scaled_pipe[i] if i < len(scaled_pipe) else None)
        for i, name in enumerate(names)
    ) + tuple(scaled_pipe[len(names):])


def _encode_text(clip, text):
    return nodes.CLIPTextEncode().encode(clip, text or "")[0]


def _tile_prompt_parts(tbg_pipe, index):
    prompt = ""
    try:
        prompter = tbg_pipe[7]
        prompts = getattr(prompter, "output_prompts", None) or getattr(prompter, "tiler_prompts", None) or []
        if index < len(prompts):
            prompt = prompts[index] or ""
    except Exception:
        prompt = ""

    general_prompt = ""
    try:
        general_prompt = getattr(tbg_pipe[2], "General_Prompt", "") or ""
    except Exception:
        general_prompt = ""
    try:
        ignore_flags = getattr(tbg_pipe[7], "output_ignore_general_prompt_js", []) or []
        if index < len(ignore_flags) and bool(ignore_flags[index]):
            general_prompt = ""
    except Exception:
        pass
    return general_prompt, prompt


def _tile_prompt(tbg_pipe, index):
    general_prompt, prompt = _tile_prompt_parts(tbg_pipe, index)
    return "\n".join(p for p in (general_prompt, prompt) if p)


def _local_rebuild(pid_tiles, grid_specs):
    max_w = 0
    max_h = 0
    for spec in grid_specs:
        _, _, _, x, y, w, h = spec[:7]
        max_w = max(max_w, int(x) + int(w))
        max_h = max(max_h, int(y) + int(h))
    if max_w <= 0 or max_h <= 0:
        return torch.cat(pid_tiles, dim=0)

    device = pid_tiles[0].device
    dtype = pid_tiles[0].dtype
    result = torch.zeros((1, max_h, max_w, pid_tiles[0].shape[-1]), device=device, dtype=dtype)
    weight = torch.zeros((1, max_h, max_w, 1), device=device, dtype=dtype)
    for tile, spec in zip(pid_tiles, grid_specs):
        _, _, _, x, y, w, h = spec[:7]
        x, y, w, h = int(x), int(y), int(w), int(h)
        crop = tile[:, :h, :w, :]
        result[:, y : y + h, x : x + w, :] += crop
        weight[:, y : y + h, x : x + w, :] += 1
    return result / weight.clamp_min(1)


class TBG_ETUR_PID_Tile_Upscale_Rebuild:
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://github.com/comfyanonymous/ComfyUI"
    DESCRIPTION = "Runs tiled Nvidia PiD image upscale from an input image or latent and rebuilds the final image."
    FUNCTION = "fn"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PID_Model": ("MODEL", {"label": "PID Diffusion Model"}),
                "PID_VAE_Compatible_Model": (
                    PID_VAE_COMPATIBLE_MODEL_TYPES,
                    {"label": "PiD VAE Compatible Model", "default": "FLUX1"},
                ),
                "seed": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "upscale_by": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (PID_SAMPLERS, {"default": PID_SAMPLER_DEFAULT}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": PID_SCHEDULER_DEFAULT}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.01}),
                "degrade_sigma": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE", {"label": "Image"}),
                "latent": ("LATENT", {"label": "Source Latent"}),
                "Sampler": ("SAMPLER", {"label": "Sampler Override"}),
                "PID_CLIP": ("CLIP", {"label": "PID CLIP Override"}),
                "PID_Source_VAE": ("VAE", {"label": "PID Source VAE Override"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("PID Rebuilt", "PID Tiles")
    OUTPUT_IS_LIST = (False, True)
    OUTPUT_NODE = True

    def fn(
        self,
        PID_Model,
        PID_VAE_Compatible_Model,
        seed,
        image=None,
        upscale_by=4.0,
        steps=4,
        cfg=1.0,
        sampler_name=PID_SAMPLER_DEFAULT,
        scheduler=PID_SCHEDULER_DEFAULT,
        denoise=1.0,
        degrade_sigma=0.1,
        prompt="",
        Sampler=None,
        PID_CLIP=None,
        PID_Source_VAE=None,
        latent=None,
    ):
        if image is None and latent is None:
            raise ValueError("TBG ETUR tiled Nvidia PID Image Upscale requires either an Image input or a Source Latent input.")
        pid_model_name = pid_model_name_for_model_type(PID_VAE_Compatible_Model)

        rebuilt, pid_meta = run_pid_tiled_upscale(
            image,
            pid_model_name,
            prompt_text=prompt,
            seed=seed,
            clip=PID_CLIP,
            source_vae=PID_Source_VAE,
            overlap=PID_DEFAULT_OVERLAP,
            degrade_sigma=degrade_sigma,
            color_match_fn=lambda reference_tile, pid_tile, method: (
                TBG_Image.detail_preserving_colormatch(
                    reference_tile,
                    pid_tile,
                    method,
                    1.0,
                    label="pid_upscale_tile_color",
                )[0]
                if hasattr(TBG_Image, "detail_preserving_colormatch")
                else TBG_Image.colormatch(reference_tile, pid_tile, method, 1.0)[0]
            ),
            color_match_method=PID_DEFAULT_COLOR_MATCH,
            prompt_fn=None,
            sampler=Sampler,
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            include_pid_tiles=True,
            source_latent=latent,
            pid_model=PID_Model,
            pid_model_type=PID_VAE_Compatible_Model,
            allow_hf_download=False,
        )

        if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
            rebuilt = rebuilt.unsqueeze(0)
        source_width = int(pid_meta.get("source_width") or (int(pid_meta.get("target_width", rebuilt.shape[2])) // 4))
        source_height = int(pid_meta.get("source_height") or (int(pid_meta.get("target_height", rebuilt.shape[1])) // 4))
        final_width = max(1, int(round(source_width * float(upscale_by or 1.0))))
        final_height = max(1, int(round(source_height * float(upscale_by or 1.0))))
        if int(rebuilt.shape[2]) != final_width or int(rebuilt.shape[1]) != final_height:
            rebuilt = nodes.ImageScale().upscale(rebuilt, "lanczos", final_width, final_height, False)[0]

        return rebuilt, list(pid_meta.get("pid_tiles", []))


class TBG_ETUR_Download_PID_Model:
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    HELP_LINK = "https://huggingface.co/Comfy-Org/PixelDiT/tree/main/diffusion_models"
    DESCRIPTION = "Refreshes PixelDiT/PiD model options from Hugging Face, downloads the selected PiD diffusion model and text encoder, and outputs a loaded model for PiD nodes."
    FUNCTION = "fn"

    @classmethod
    def INPUT_TYPES(cls):
        options = get_pid_upscale_options(refresh=True)
        return {
            "required": {
                "pid_model": (
                    options,
                    {
                        "label": "PID Model",
                        "default": options[0],
                        "tooltip": "Downloads/loads the selected PixelDiT/PiD diffusion model. Connect PID Model, and optionally PID CLIP, to downstream PiD nodes.",
                    },
                ),
                "load_clip": ("BOOLEAN", {"label": "Load PiD CLIP", "default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "PID_MODEL_INFO")
    RETURN_NAMES = ("PID Model", "PID CLIP", "PID Model Info")
    OUTPUT_IS_LIST = (False, False, False)
    OUTPUT_NODE = False

    def fn(self, pid_model, load_clip=True):
        model, clip, info = download_pid_model_bundle(pid_model, load_clip=load_clip, force_refresh=True)
        return model, clip, info
