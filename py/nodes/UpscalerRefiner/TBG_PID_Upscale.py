import copy
import copy
import sys
import re
from pathlib import Path
from types import SimpleNamespace

import comfy.samplers
import folder_paths
import nodes
import torch
import torch.nn.functional as F

from .inc.image import TBG_Image
from .TBG_Nodes_PRO import TBG_ETUR_ColorCorrection
from .TBG_Refiner import TBG_Refiner_v1
from .inc.sift_drift import apply_sift_drift_correction
from .inc.tbg_pid import (
    PID_DEFAULT_COLOR_MATCH,
    PID_DEFAULT_OVERLAP,
    PID_DEFAULT_SAMPLER,
    PID_DEFAULT_STITCH_BLUR,
    PID_DEFAULT_STITCH_FEATHER,
    PID_SCALE,
    PID_VAE_COMPATIBLE_MODEL_TYPES,
    download_pid_model_bundle,
    get_pid_upscale_options,
    gpu_pid_tile_rebuild,
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
    return gpu_pid_tile_rebuild(pid_tiles, grid_specs, max_w, max_h, label="TBG PID Standalone")


def _standalone_pid_internal_edge_mask(tile_w, tile_h, x, y, width, height, border, device, dtype):
    border = int(max(0, min(border, tile_w // 2, tile_h // 2)))
    mask = torch.zeros((1, tile_h, tile_w), device=device, dtype=dtype)
    if border <= 0:
        return mask

    ramp = torch.linspace(1.0, 0.0, border + 2, device=device, dtype=dtype)[1:-1]
    if int(x) > 0:
        mask[:, :, :border] = torch.maximum(mask[:, :, :border], ramp.view(1, 1, border))
    if int(y) > 0:
        mask[:, :border, :] = torch.maximum(mask[:, :border, :], ramp.view(1, border, 1))
    if int(x) + int(tile_w) < int(width):
        mask[:, :, tile_w - border:tile_w] = torch.maximum(
            mask[:, :, tile_w - border:tile_w],
            ramp.flip(0).view(1, 1, border),
        )
    if int(y) + int(tile_h) < int(height):
        mask[:, tile_h - border:tile_h, :] = torch.maximum(
            mask[:, tile_h - border:tile_h, :],
            ramp.flip(0).view(1, border, 1),
        )
    return mask.clamp(0.0, 1.0)


def _standalone_pid_low_frequency_anchor(reference_tile, pid_tile, strength=1.0):
    if reference_tile is None or pid_tile is None:
        return pid_tile, None
    if not torch.is_tensor(reference_tile) or not torch.is_tensor(pid_tile):
        return pid_tile, None
    if reference_tile.ndim != 4 or pid_tile.ndim != 4:
        return pid_tile, None

    original_device = pid_tile.device
    original_dtype = pid_tile.dtype
    try:
        strength = max(0.0, min(1.0, float(strength)))
    except Exception:
        strength = 1.0
    if strength <= 0.0:
        return pid_tile, None

    ref = reference_tile.to(device=original_device, dtype=torch.float32).clamp(0.0, 1.0)
    target = pid_tile.to(device=original_device, dtype=torch.float32).clamp(0.0, 1.0)
    if int(ref.shape[1]) != int(target.shape[1]) or int(ref.shape[2]) != int(target.shape[2]):
        ref = nodes.ImageScale().upscale(ref, "lanczos", int(target.shape[2]), int(target.shape[1]), False)[0]
        ref = ref.to(device=original_device, dtype=torch.float32).clamp(0.0, 1.0)

    ref_bchw = ref.permute(0, 3, 1, 2).contiguous()
    target_bchw = target.permute(0, 3, 1, 2).contiguous()
    height = int(target_bchw.shape[-2])
    width = int(target_bchw.shape[-1])
    low_h = max(32, min(192, int(round(height / 24.0))))
    low_w = max(32, min(192, int(round(width / 24.0))))
    low_ref = F.interpolate(
        F.interpolate(ref_bchw, size=(low_h, low_w), mode="area"),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    low_target = F.interpolate(
        F.interpolate(target_bchw, size=(low_h, low_w), mode="area"),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    field = (low_ref - low_target).clamp(-160.0 / 255.0, 160.0 / 255.0)
    corrected = (target_bchw + field * strength).clamp(0.0, 1.0)
    before_residual = torch.mean(torch.abs(low_target - low_ref))
    low_corrected = F.interpolate(
        F.interpolate(corrected, size=(low_h, low_w), mode="area"),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    after_residual = torch.mean(torch.abs(low_corrected - low_ref))
    delta = torch.mean(torch.abs(corrected - target_bchw))
    corrected = corrected.permute(0, 2, 3, 1).contiguous().to(device=original_device, dtype=original_dtype).clamp(0.0, 1.0)
    return corrected, {
        "delta": float(delta.detach().cpu()),
        "low_residual_before": float(before_residual.detach().cpu()),
        "low_residual_after": float(after_residual.detach().cpu()),
        "grid": (int(low_w), int(low_h)),
    }


def _standalone_pid_refiner_style_rebuild(pid_tiles, grid_specs, reference_image, target_width, target_height, debug_enabled=False):
    if not pid_tiles or not grid_specs or reference_image is None:
        raise ValueError("Standalone PiD refiner-style rebuild requires tiles, grid specs, and a reference image.")
    if torch.is_tensor(reference_image) and reference_image.ndim == 3:
        reference_image = reference_image.unsqueeze(0)

    normal_specs = [spec for spec in grid_specs if isinstance(spec, (list, tuple)) and len(spec) >= 7 and int(spec[2]) < 8000]
    if not normal_specs:
        normal_specs = list(grid_specs)
    rows = max((int(spec[0]) for spec in normal_specs), default=0) + 1
    cols = max((int(spec[1]) for spec in normal_specs), default=0) + 1
    first_tile = next((tile for tile in pid_tiles if torch.is_tensor(tile)), None)
    if first_tile is None:
        raise ValueError("Standalone PiD refiner-style rebuild received no tensor tiles.")
    if first_tile.ndim == 3:
        first_tile = first_tile.unsqueeze(0)
    tile_w = int(first_tile.shape[2])
    tile_h = int(first_tile.shape[1])

    params = SimpleNamespace(
        grid_specs=list(grid_specs),
        grid_prompts=[""] * len(grid_specs),
        stitch_blending="gpupyramid",
        Fast_1_Tile_Preview=False,
        SegFusion_Initializer_run_once=False,
        Tile_Fusion_Mode="NONE",
        len_grid_images=len(pid_tiles),
        len_segments=0,
    )
    size = SimpleNamespace(
        rows_qty=rows,
        cols_qty=cols,
        fullW=tile_w,
        fullH=tile_h,
        tile_grid_W=tile_w,
        tile_grid_H=tile_h,
        UpscaledInputImageW=int(target_width),
        UpscaledInputImageH=int(target_height),
        max_image_width=int(target_width),
        max_image_height=int(target_height),
        overlay_between_tiles=int(PID_DEFAULT_OVERLAP * PID_SCALE),
        composite_blur_margin=int(PID_DEFAULT_STITCH_FEATHER * PID_SCALE),
        crop_margin=0,
        inpaint_blur_margin=int(PID_DEFAULT_STITCH_BLUR * PID_SCALE),
        inpaint_border_margin=int(PID_DEFAULT_OVERLAP * PID_SCALE),
        shift=0,
        shifttl=0,
        inpaint_max=0,
    )

    custom_node_root = Path(__file__).resolve().parents[3]
    if str(custom_node_root) not in sys.path:
        sys.path.insert(0, str(custom_node_root))
    from TBG.TBG_APP.TBG_APP import TBG_PIDWorkerRebuild
    from TBG.TBG_APP.constants import get_current_tbg, get_current_tiler_id, set_current_tiler_id

    previous_tiler_id = get_current_tiler_id()
    standalone_tiler_id = "pid_standalone_refiner_style"
    set_current_tiler_id(standalone_tiler_id)
    native_tbg = get_current_tbg()
    previous = {
        "PARAMS": getattr(native_tbg, "PARAMS", None),
        "SIZE": getattr(native_tbg, "SIZE", None),
        "OUTPUTS_grid_images_all": getattr(native_tbg.OUTPUTS, "grid_images_all", None),
        "OUTPUTS_orig_grid_images_all": getattr(native_tbg.OUTPUTS, "orig_grid_images_all", None),
        "OUTPUTS_upscaled_image": getattr(native_tbg.OUTPUTS, "upscaled_image", None),
        "OUTPUTS_last_final_image": getattr(native_tbg.OUTPUTS, "last_final_image", None),
        "SEGMENTS_crop_regions": getattr(native_tbg.SEGMENTS, "segms_crop_regions", None),
        "SEGMENTS_compositing_mask": getattr(native_tbg.SEGMENTS, "compositing_mask", None),
        "SEGMENTS_binary_masks": getattr(native_tbg.SEGMENTS, "segment_binary_masks", None),
    }
    try:
        native_tbg.PARAMS = SimpleNamespace(**vars(params))
        native_tbg.SIZE = SimpleNamespace(**vars(size))
        native_tbg.OUTPUTS.grid_images_all = list(pid_tiles)
        native_tbg.OUTPUTS.orig_grid_images_all = list(pid_tiles)
        native_tbg.OUTPUTS.upscaled_image = reference_image
        native_tbg.OUTPUTS.last_final_image = None
        native_tbg.SEGMENTS.segms_crop_regions = []
        native_tbg.SEGMENTS.compositing_mask = []
        native_tbg.SEGMENTS.segment_binary_masks = []
        rebuilt, _, _ = TBG_PIDWorkerRebuild.rebuild_final_image_with_state(
            params,
            size,
            [""] * len(grid_specs),
            [],
            [],
            nosegments=True,
            full_image_only_tiles=None,
        )
        if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
            rebuilt = rebuilt.unsqueeze(0)
        if torch.is_tensor(rebuilt):
            rebuilt = rebuilt[:, :target_height, :target_width, :]
        if bool(debug_enabled):
            print(
                "[TBG PID Standalone] refiner-style gpupyramid rebuild complete "
                f"tiles={len(pid_tiles)} canvas={target_width}x{target_height}"
            )
        return rebuilt
    finally:
        native_tbg.PARAMS = previous["PARAMS"]
        native_tbg.SIZE = previous["SIZE"]
        native_tbg.OUTPUTS.grid_images_all = previous["OUTPUTS_grid_images_all"]
        native_tbg.OUTPUTS.orig_grid_images_all = previous["OUTPUTS_orig_grid_images_all"]
        native_tbg.OUTPUTS.upscaled_image = previous["OUTPUTS_upscaled_image"]
        native_tbg.OUTPUTS.last_final_image = previous["OUTPUTS_last_final_image"]
        native_tbg.SEGMENTS.segms_crop_regions = previous["SEGMENTS_crop_regions"]
        native_tbg.SEGMENTS.compositing_mask = previous["SEGMENTS_compositing_mask"]
        native_tbg.SEGMENTS.segment_binary_masks = previous["SEGMENTS_binary_masks"]
        set_current_tiler_id(previous_tiler_id)


def run_tbg_pid_upscale_pipeline(
    *,
    image=None,
    latent=None,
    upscale_by=4.0,
    seed=0,
    steps=4,
    cfg=1.0,
    sampler_name=PID_SAMPLER_DEFAULT,
    scheduler=PID_SCHEDULER_DEFAULT,
    denoise=1.0,
    degrade_sigma=0.1,
    Color_Match=TBG_Refiner_v1.COLOR_STABILIZER_METHOD,
    Geometry_Drift_Correction=True,
    Color_Match_Str=1.0,
    prompt="",
    Sampler=None,
    PID_Model=None,
    PID_VAE_Compatible_Model=None,
    PID_CLIP=None,
    PID_Source_VAE=None,
    upscale_model_name=None,
    debug_prefix="[TBG PID Standalone]",
    debug_enabled=False,
):
    if image is None and latent is None:
        raise ValueError("TBG ETUR tiled Nvidia PID Image Upscale requires either an Image input or a Source Latent input.")
    pid_model_name = upscale_model_name or pid_model_name_for_model_type(PID_VAE_Compatible_Model)
    debug_enabled = bool(debug_enabled)

    def debug_log(message):
        if debug_enabled:
            print(message)

    def dev_debug_save(image_value, label):
        if not debug_enabled or image_value is None:
            return
        try:
            normalized = TBG_Refiner_v1._normalize_debug_image_for_save(image_value)
            if normalized is None:
                raise ValueError("debug image normalization returned None")
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or "debug_image")).strip("._")
            if not safe_label:
                safe_label = "debug_image"
            folder = Path(folder_paths.get_temp_directory()) / "TBG" / "compareTiles"
            folder.mkdir(parents=True, exist_ok=True)
            prefix = f"TBG/compareTiles/{safe_label}"
            preview = nodes.PreviewImage()
            preview.save_images(normalized, prefix, None, None)
        except Exception as exc:
            debug_log(f"{debug_prefix} debug save skipped for {label}: {exc}")

    def dev_debug_tile_stage(stage_name, index, image_value, meta=None):
        if image_value is None:
            return
        suffix = ""
        if isinstance(meta, dict):
            parts = []
            for key in ("row", "col", "x", "y", "tile_w", "tile_h", "out_x", "out_y", "out_w", "out_h"):
                if key in meta:
                    parts.append(f"{key}{int(meta[key])}")
            if parts:
                suffix = "_" + "_".join(parts)
        dev_debug_save(image_value, f"{index}_{stage_name}{suffix}")

    def tbg_pid_sift_drift(reference_tile, pid_tile, index):
        if not bool(Geometry_Drift_Correction):
            return pid_tile
        corrected, info = apply_sift_drift_correction(
            reference_tile,
            pid_tile,
            index=index,
            mode=Geometry_Drift_Correction,
        )
        return corrected, info

    def tbg_pid_refiner_post_decode(reference_tile, pid_tile, index, out_x, out_y, target_width, target_height, border):
        corrected = pid_tile
        dev_debug_save(reference_tile, f"{index}_standalone_reference_tile_before_postdecode_4x")
        dev_debug_save(pid_tile, f"{index}_standalone_pid_tile_before_postdecode_4x")
        if bool(Geometry_Drift_Correction):
            corrected, sift_info = tbg_pid_sift_drift(reference_tile, corrected, index)
            dev_debug_save(corrected, f"{index}_standalone_pid_tile_after_sift_4x")
            debug_log(
                f"{debug_prefix} tile {index + 1}: SIFT drift before color "
                f"changed={bool(sift_info.get('changed', False))} "
                f"reason={sift_info.get('reason', 'unknown')} "
                f"matches={sift_info.get('matches', 0)} "
                f"inliers={sift_info.get('inliers', 0)}"
            )

        color_active = Color_Match is not None and str(Color_Match).strip().lower() != "none"
        try:
            strength = max(0.0, min(1.0, float(Color_Match_Str if Color_Match_Str is not None else 1.0)))
        except Exception:
            strength = 1.0

        if color_active and strength > 0.0:
            try:
                method_key = str(Color_Match or "").strip().lower()
                standalone_full_low_frequency = (
                    TBG_Refiner_v1.is_tbg_tile_aware(Color_Match)
                    or method_key in getattr(TBG_Refiner_v1, "COLOR_STABILIZER_ALIASES", set())
                )
                if standalone_full_low_frequency:
                    seam_mask = _standalone_pid_internal_edge_mask(
                        int(corrected.shape[2]),
                        int(corrected.shape[1]),
                        out_x,
                        out_y,
                        target_width,
                        target_height,
                        border,
                        corrected.device,
                        corrected.dtype,
                    )
                    before_full_low = corrected
                    corrected = TBG_Refiner_v1._tile_aware_low_frequency_match(
                        reference_tile,
                        corrected,
                        seam_mask,
                        strength=strength,
                    )
                    full_diff = torch.abs(corrected.to(torch.float32) - before_full_low.to(torch.float32)).mean(
                        dim=-1,
                        keepdim=True,
                    )
                    seam_weight = seam_mask.to(device=full_diff.device, dtype=full_diff.dtype).unsqueeze(-1).clamp(0.0, 1.0)
                    inner_weight = (1.0 - seam_weight).clamp(0.0, 1.0)
                    low_delta = full_diff.mean().detach().cpu()
                    border_delta = (
                        (full_diff * seam_weight).sum() / seam_weight.sum().clamp_min(1e-6)
                    ).detach().cpu()
                    inner_delta = (
                        (full_diff * inner_weight).sum() / inner_weight.sum().clamp_min(1e-6)
                    ).detach().cpu()
                    debug_log(
                        f"{debug_prefix} tile {index + 1}: refiner full-tile "
                        f"edge+inner low-frequency color applied method={Color_Match} "
                        f"strength={strength:.3f} mean_abs_delta={float(low_delta):.8f} "
                        f"border_delta={float(border_delta):.8f} inner_delta={float(inner_delta):.8f}"
                    )
                    dev_debug_save(seam_mask.unsqueeze(-1), f"{index}_standalone_pid_internal_edge_mask_4x")
                    dev_debug_save(corrected, f"{index}_standalone_pid_tile_after_full_tile_low_frequency_4x")
                    before_anchor = corrected
                    corrected, anchor_metrics = _standalone_pid_low_frequency_anchor(
                        reference_tile,
                        corrected,
                        strength=strength,
                    )
                    if anchor_metrics is not None:
                        debug_log(
                            f"{debug_prefix} tile {index + 1}: standalone whole-tile "
                            f"reference low-frequency anchor applied strength={strength:.3f} "
                            f"mean_abs_delta={anchor_metrics['delta']:.8f} "
                            f"low_residual_before={anchor_metrics['low_residual_before']:.8f} "
                            f"low_residual_after={anchor_metrics['low_residual_after']:.8f} "
                            f"grid={anchor_metrics['grid'][0]}x{anchor_metrics['grid'][1]}"
                        )
                        dev_debug_save(corrected, f"{index}_standalone_pid_tile_after_whole_tile_anchor_4x")
                    else:
                        corrected = before_anchor
                else:
                    matched, metrics = TBG_Refiner_v1._global_rgb_luma_match(
                        reference_tile,
                        corrected,
                        strength=strength,
                        label=f"standalone_pid_after_decode_tile_{index + 1}",
                    )
                    if metrics is not None:
                        corrected = matched
                        debug_log(
                            f"{debug_prefix} tile {index + 1}: refiner RGB/luma color applied "
                            f"method={Color_Match} strength={strength:.3f} "
                            f"luma_shift_before={metrics.get('before_luma_shift', 0.0):+.2f} "
                            f"luma_shift_after={metrics.get('after_luma_shift', 0.0):+.2f}"
                        )
                        dev_debug_save(corrected, f"{index}_standalone_pid_tile_after_global_rgb_luma_4x")
                    else:
                        debug_log(
                            f"{debug_prefix} tile {index + 1}: refiner RGB/luma color skipped "
                            "metrics unavailable"
                        )
            except Exception as exc:
                debug_log(f"{debug_prefix} refiner-style RGB/luma tile {index + 1} skipped: {exc}")

        if color_active and strength > 0.0:
            try:
                seam_mask = _standalone_pid_internal_edge_mask(
                    int(corrected.shape[2]),
                    int(corrected.shape[1]),
                    out_x,
                    out_y,
                    target_width,
                    target_height,
                    border,
                    corrected.device,
                    corrected.dtype,
                )
                if float(seam_mask.max().detach().cpu()) > 1e-5:
                    corrected = TBG_Image.stabilize_tile_low_frequency_from_reference(
                        reference_tile.to(device=corrected.device, dtype=corrected.dtype),
                        corrected,
                        seam_mask,
                        seam_mask,
                        min(0.60, strength),
                    )[0]
                    debug_log(
                        f"{debug_prefix} tile {index + 1}: refiner seam-local "
                        f"low-frequency color applied strength={min(0.60, strength):.3f}"
                    )
                    dev_debug_save(corrected, f"{index}_standalone_pid_tile_after_seam_local_low_frequency_4x")
            except Exception as exc:
                debug_log(f"{debug_prefix} refiner-style seam-local tile {index + 1} skipped: {exc}")

        return corrected

    rebuilt, pid_meta = run_pid_tiled_upscale(
        image,
        pid_model_name,
        prompt_text=prompt,
        seed=seed,
        rebuild_fn=lambda pid_tiles, grid_specs, reference_image, target_width, target_height: _standalone_pid_refiner_style_rebuild(
            pid_tiles,
            grid_specs,
            reference_image,
            target_width,
            target_height,
            debug_enabled=debug_enabled,
        ),
        clip=PID_CLIP,
        source_vae=PID_Source_VAE,
        overlap=PID_DEFAULT_OVERLAP,
        degrade_sigma=degrade_sigma,
        color_match_fn=None,
        pre_color_drift_fn=None,
        post_decode_tile_fn=tbg_pid_refiner_post_decode,
        color_match_method=Color_Match if Color_Match is not None else PID_DEFAULT_COLOR_MATCH,
        defer_multi_tile_color_match=False,
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
        debug_tile_stage_fn=dev_debug_tile_stage if debug_enabled else None,
        debug_enabled=debug_enabled,
    )

    for debug_index, debug_tile in enumerate(list(pid_meta.get("pid_tiles", []))):
        dev_debug_save(debug_tile, f"{debug_index}_standalone_pid_tile_final_from_pid_meta_4x")

    dev_debug_save(rebuilt, "standalone_pid_rebuilt_after_worker_rebuild_before_output_resize")

    if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
        rebuilt = rebuilt.unsqueeze(0)
    source_width = int(pid_meta.get("source_width") or (int(pid_meta.get("target_width", rebuilt.shape[2])) // 4))
    source_height = int(pid_meta.get("source_height") or (int(pid_meta.get("target_height", rebuilt.shape[1])) // 4))
    final_width = max(1, int(round(source_width * float(upscale_by or 1.0))))
    final_height = max(1, int(round(source_height * float(upscale_by or 1.0))))
    if int(rebuilt.shape[2]) != final_width or int(rebuilt.shape[1]) != final_height:
        rebuilt = nodes.ImageScale().upscale(rebuilt, "lanczos", final_width, final_height, False)[0]
    dev_debug_save(rebuilt, "standalone_pid_rebuilt_after_output_resize_before_final_color")

    tile_count = int(pid_meta.get("tile_count", 0) or 0)
    if (
        image is not None
        and torch.is_tensor(image)
        and Color_Match is not None
        and str(Color_Match).strip().lower() != "none"
        and tile_count > 1
    ):
        debug_log(
            f"{debug_prefix} applying final refiner-style full-image color correction "
            f"after corrected tile rebuild method={Color_Match} strength="
            f"{float(Color_Match_Str if Color_Match_Str is not None else 1.0):.3f} "
            f"canvas={final_width}x{final_height}"
        )
        reference_image = nodes.ImageScale().upscale(image, "bilinear", final_width, final_height, False)[0]
        dev_debug_save(reference_image, "standalone_pid_final_color_reference_image")
        rebuilt = TBG_ETUR_ColorCorrection.fn(
            rebuilt,
            reference_image,
            Color_Match=Color_Match,
            Color_Match_Str=float(Color_Match_Str if Color_Match_Str is not None else 1.0),
            Geometry_Drift_Correction=False,
        )[0]
        dev_debug_save(rebuilt, "standalone_pid_rebuilt_after_final_full_image_color")

    dev_debug_save(rebuilt, "standalone_pid_final_output")

    return rebuilt, list(pid_meta.get("pid_tiles", [])), pid_meta


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
                "Color_Match": (
                    TBG_ETUR_ColorCorrection.COLOR_MATCH_METHODS,
                    {
                        "label": "Color Match Method",
                        "default": TBG_Refiner_v1.COLOR_STABILIZER_METHOD,
                    },
                ),
                "Geometry_Drift_Correction": (
                    "BOOLEAN",
                    {
                        "label": "Geometry Drift Correction",
                        "default": True,
                        "label_on": "On",
                        "label_off": "Off",
                        "tooltip": "When On, applies SIFT drift correction to each PiD tile against its cropped reference image before color correction.",
                    },
                ),
                "Color_Match_Str": (
                    "FLOAT",
                    {
                        "label": "Color Match Strength",
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "label": "Input Image (color correction reference)",
                        "tooltip": (
                            "Optional reference image. Use image alone for normal PiD upscaling with per-tile SIFT drift correction before color correction, "
                            "or connect image together with Source Latent so PiD samples from the latent while this image is used "
                            "only as the per-tile reference for SIFT drift correction and color correction. If no image is connected, "
                            "TBG decodes the Source Latent first and uses that decode as the reference image."
                        ),
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "label": "Source Latent",
                        "tooltip": (
                            "Optional PiD source latent. Use latent alone to sample directly from the latent; TBG will VAE-decode it "
                            "first and use that decode as the per-tile reference for SIFT drift correction and color correction. Or connect "
                            "latent together with the image input to use the latent for PiD and the image as the reference for color "
                            "correction and geometry drift correction."
                        ),
                    },
                ),
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
        Color_Match=TBG_Refiner_v1.COLOR_STABILIZER_METHOD,
        Geometry_Drift_Correction=True,
        Color_Match_Str=1.0,
        prompt="",
        Sampler=None,
        PID_CLIP=None,
        PID_Source_VAE=None,
        latent=None,
        debug_enabled=False,
    ):
        rebuilt, pid_tiles, _ = run_tbg_pid_upscale_pipeline(
            image=image,
            latent=latent,
            upscale_by=upscale_by,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            degrade_sigma=degrade_sigma,
            Color_Match=Color_Match,
            Geometry_Drift_Correction=Geometry_Drift_Correction,
            Color_Match_Str=Color_Match_Str,
            prompt=prompt,
            Sampler=Sampler,
            PID_Model=PID_Model,
            PID_VAE_Compatible_Model=PID_VAE_Compatible_Model,
            PID_CLIP=PID_CLIP,
            PID_Source_VAE=PID_Source_VAE,
            debug_prefix="[TBG PID Standalone]",
            debug_enabled=debug_enabled,
        )
        return rebuilt, pid_tiles


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
