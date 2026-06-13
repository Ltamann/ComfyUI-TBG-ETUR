import inspect
import json
import os
import re
import shutil
from dataclasses import dataclass
import time
from types import SimpleNamespace
import urllib.request

import comfy.model_management as mm
import folder_paths
import node_helpers
import nodes
import torch
import comfy.latent_formats
import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, SamplerCustom


PID_REPO_ID = "Comfy-Org/PixelDiT"
PID_INPUT_SIZE = 1024
PID_OUTPUT_SIZE = 4096
PID_SCALE = 4
PID_DEFAULT_OVERLAP = 32
PID_DEFAULT_STITCH_BLUR = 128
PID_DEFAULT_STITCH_FEATHER = 32
PID_DEFAULT_COLOR_MATCH = "lab color match+detail preservation"
PID_DEFAULT_SAMPLER = "pid_sde"
PID_HF_REFRESH_TTL = 300
_PID_HF_LAST_REFRESH = 0.0


def pid_gpu_final_rebuild_enabled():
    value = os.environ.get("TBG_PID_GPU_FINAL_REBUILD", "1")
    return str(value).strip().lower() not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class PIDUpscaleSpec:
    diffusion_model: str
    text_encoder: str
    latent_format: str
    source_vae_candidates: tuple[str, ...]
    input_size: int = PID_INPUT_SIZE
    output_size: int = PID_OUTPUT_SIZE


PID_UPSCALE_SPECS = {
    "SuperResolution/PID Flux1 1K to 4K BF16": PIDUpscaleSpec(
        "pid_flux1_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "flux",
        ("ae.safetensors", "Flux/flux1DevVAE_stock.safetensors", "Flux\\flux1DevVAE_stock.safetensors", "taef1"),
    ),
    "SuperResolution/PID Flux1 1K to 4K MXFP8": PIDUpscaleSpec(
        "pid_flux1_1024_to_4096_4step_mxfp8.safetensors",
        "gemma_2_2b_it_elm_fp8_scaled.safetensors",
        "flux",
        ("ae.safetensors", "Flux/flux1DevVAE_stock.safetensors", "Flux\\flux1DevVAE_stock.safetensors", "taef1"),
    ),
    "SuperResolution/PID Flux2 1K to 4K BF16": PIDUpscaleSpec(
        "pid_flux2_1024_to_4096_4step_2606_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "flux",
        ("flux2-vae.safetensors", "taef2"),
    ),
    "SuperResolution/PID Flux2 1K to 4K BF16 (original)": PIDUpscaleSpec(
        "pid_flux2_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "flux",
        ("flux2-vae.safetensors", "taef2"),
    ),
    "SuperResolution/PID Flux2 1K to 4K MXFP8": PIDUpscaleSpec(
        "pid_flux2_1024_to_4096_4step_mxfp8.safetensors",
        "gemma_2_2b_it_elm_fp8_scaled.safetensors",
        "flux",
        ("flux2-vae.safetensors", "taef2"),
    ),
    "SuperResolution/PID SDXL 1K to 4K BF16": PIDUpscaleSpec(
        "pid_sdxl_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "sdxl",
        ("sdxl_vae.safetensors", "SDXL/sdxl_vae.safetensors", "SDXL\\sdxl_vae.safetensors", "taesdxl"),
    ),
    "SuperResolution/PID SD3 1K to 4K BF16": PIDUpscaleSpec(
        "pid_sd3_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "sd3",
        ("sd3_vae.safetensors", "SD3/sd3_vae.safetensors", "SD3\\sd3_vae.safetensors"),
    ),
    "SuperResolution/PID QwenImage 1K to 4K BF16": PIDUpscaleSpec(
        "pid_qwenimage_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "wan21",
        ("qwen_image_vae.safetensors", "Qwen/qwen_image_vae.safetensors", "Qwen\\qwen_image_vae.safetensors"),
    ),
}

PID_UPSCALE_OPTIONS = tuple(PID_UPSCALE_SPECS.keys())
PID_VAE_COMPATIBLE_MODEL_TYPES = (
    "FLUX1",
    "FLUX2",
    "Qwen Image",
    "Qwen Image Edit",
    "SDXL",
    "SD3",
    "Z-Image",
)


def _pid_family_details(raw_family):
    family = str(raw_family or "").strip().lower().replace("-", "_")
    if family == "flux1":
        return "Flux1", "flux", ("ae.safetensors", "Flux/flux1DevVAE_stock.safetensors", "Flux\\flux1DevVAE_stock.safetensors", "taef1")
    if family == "flux2":
        return "Flux2", "flux", ("flux2-vae.safetensors", "taef2")
    if family == "qwenimage":
        return "QwenImage", "wan21", ("qwen_image_vae.safetensors", "Qwen/qwen_image_vae.safetensors", "Qwen\\qwen_image_vae.safetensors")
    if family == "sdxl":
        return "SDXL", "sdxl", ("sdxl_vae.safetensors", "SDXL/sdxl_vae.safetensors", "SDXL\\sdxl_vae.safetensors", "taesdxl")
    if family == "sd3":
        return "SD3", "sd3", ("sd3_vae.safetensors", "SD3/sd3_vae.safetensors", "SD3\\sd3_vae.safetensors")
    return None, None, None


def _infer_pid_upscale_spec_from_filename(filename):
    name = os.path.basename(str(filename or ""))
    match = re.match(r"^pid_([a-z0-9]+)_(\d+)_to_(\d+)_4step(?:_([0-9]+))?_([a-z0-9]+)\.safetensors$", name, re.IGNORECASE)
    if not match:
        return None

    raw_family, input_size, output_size, revision, precision = match.groups()
    family, latent_format, vae_candidates = _pid_family_details(raw_family)
    if family is None:
        return None

    input_size = int(input_size)
    output_size = int(output_size)
    if input_size <= 0 or output_size <= 0:
        return None

    precision = str(precision or "").upper()
    text_encoder = "gemma_2_2b_it_elm_fp8_scaled.safetensors" if precision == "MXFP8" else "gemma_2_2b_it_elm_bf16.safetensors"
    suffix = f" {revision}" if revision else ""
    label = f"SuperResolution/PID {family} {input_size // 1024 if input_size >= 1024 else input_size}K to {output_size // 1024 if output_size >= 1024 else output_size}K {precision}{suffix}"
    if input_size < 1024 or output_size < 1024:
        label = f"SuperResolution/PID {family} {input_size} to {output_size} {precision}{suffix}"

    return label, PIDUpscaleSpec(
        name,
        text_encoder,
        latent_format,
        vae_candidates,
        input_size=input_size,
        output_size=output_size,
    )


def _pid_refresh_options_cache():
    global PID_UPSCALE_OPTIONS
    PID_UPSCALE_OPTIONS = tuple(PID_UPSCALE_SPECS.keys())
    return PID_UPSCALE_OPTIONS


def refresh_pid_upscale_specs_from_hf(force=False):
    global _PID_HF_LAST_REFRESH
    now = time.time()
    if not force and _PID_HF_LAST_REFRESH and (now - _PID_HF_LAST_REFRESH) < PID_HF_REFRESH_TTL:
        return _pid_refresh_options_cache()

    url = f"https://huggingface.co/api/models/{PID_REPO_ID}/tree/main/diffusion_models"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-TBG-ETUR"})
        with urllib.request.urlopen(request, timeout=8) as response:
            entries = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[TBG PID] PixelDiT model list refresh skipped: {exc}")
        return _pid_refresh_options_cache()

    added = 0
    known_diffusion_models = {spec.diffusion_model for spec in PID_UPSCALE_SPECS.values()}
    for entry in entries or []:
        path = entry.get("path") if isinstance(entry, dict) else str(entry)
        filename = os.path.basename(str(path or ""))
        if not filename.startswith("pid_") or not filename.endswith(".safetensors"):
            continue
        inferred = _infer_pid_upscale_spec_from_filename(filename)
        if inferred is None:
            continue
        label, spec = inferred
        if spec.diffusion_model == "pid_flux2_1024_to_4096_4step_bf16.safetensors":
            label = "SuperResolution/PID Flux2 1K to 4K BF16 (original)"
        if spec.diffusion_model in known_diffusion_models:
            continue
        if label in PID_UPSCALE_SPECS:
            continue
        PID_UPSCALE_SPECS[label] = spec
        known_diffusion_models.add(spec.diffusion_model)
        added += 1

    _PID_HF_LAST_REFRESH = now
    options = _pid_refresh_options_cache()
    print(f"[TBG PID] PixelDiT model list refreshed from Hugging Face: {len(options)} options ({added} added)")
    return options


def get_pid_upscale_options(refresh=True):
    if refresh:
        return refresh_pid_upscale_specs_from_hf()
    return _pid_refresh_options_cache()


def pid_model_name_for_model_type(model_type):
    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type in {"flux2", "flux 2"}:
        return "SuperResolution/PID Flux2 1K to 4K BF16"
    if normalized_model_type in {"flux1", "flux 1", "flux1 kontext", "flux 1 kontext", "z-image"}:
        return "SuperResolution/PID Flux1 1K to 4K BF16"
    if normalized_model_type in {"sdxl", "stable diffusion xl", "stable-diffusion-xl"}:
        return "SuperResolution/PID SDXL 1K to 4K BF16"
    if normalized_model_type in {"sd3", "stable diffusion 3", "stable-diffusion-3"}:
        return "SuperResolution/PID SD3 1K to 4K BF16"
    if normalized_model_type in {"qwen image", "qwen image edit", "qwenimage", "qwenimage edit"}:
        return "SuperResolution/PID QwenImage 1K to 4K BF16"
    return "SuperResolution/PID Flux1 1K to 4K BF16"


def is_pid_upscale_model(upscale_model_name):
    return upscale_model_name in PID_UPSCALE_SPECS or upscale_model_name in refresh_pid_upscale_specs_from_hf()


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if hasattr(value, "result"):
        result = value.result
        return result if isinstance(result, tuple) else (result,)
    return (value,)


def _ensure_hf_model_file(folder_name, target_subdir, filename, allow_download=True):
    for folder in folder_paths.get_folder_paths(folder_name):
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return filename

    target_dir = os.path.join(folder_paths.models_dir, target_subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return filename
    if not allow_download:
        raise RuntimeError(
            f"TBG PID model file '{filename}' was not found locally in {folder_name}. "
            "Use the TBG ETUR Download PiD Model node, or place the file in ComfyUI models manually."
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download PixelDiT/PiD models. "
            "Install it or place the required files in ComfyUI models manually."
        ) from exc

    source_path = hf_hub_download(
        repo_id=PID_REPO_ID,
        filename=f"{target_subdir}/{filename}",
    )
    shutil.copy2(source_path, target_path)
    return filename


def _local_hf_model_path(folder_name, target_subdir, filename):
    for folder in folder_paths.get_folder_paths(folder_name):
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    target_path = os.path.join(folder_paths.models_dir, target_subdir, filename)
    if os.path.exists(target_path):
        return target_path
    return filename


def _pid_spec_or_refresh(upscale_model_name, force_refresh=False):
    spec = PID_UPSCALE_SPECS.get(upscale_model_name)
    if spec is None or force_refresh:
        refresh_pid_upscale_specs_from_hf(force=force_refresh)
        spec = PID_UPSCALE_SPECS.get(upscale_model_name)
    if spec is None:
        raise ValueError(f"Unknown TBG PID upscale model: {upscale_model_name}")
    return spec


def download_pid_model_bundle(upscale_model_name, load_clip=True, force_refresh=True):
    spec = _pid_spec_or_refresh(upscale_model_name, force_refresh=force_refresh)
    diffusion_model = _ensure_hf_model_file("diffusion_models", "diffusion_models", spec.diffusion_model, allow_download=True)
    text_encoder = _ensure_hf_model_file("text_encoders", "text_encoders", spec.text_encoder, allow_download=True)
    model = nodes.UNETLoader().load_unet(diffusion_model, "default")[0]
    clip = nodes.CLIPLoader().load_clip(text_encoder, type="pixeldit")[0] if load_clip else None
    info = SimpleNamespace(
        name=upscale_model_name,
        diffusion_model=spec.diffusion_model,
        text_encoder=spec.text_encoder,
        latent_format=spec.latent_format,
        input_size=spec.input_size,
        output_size=spec.output_size,
    )
    print(f"[TBG PID] downloaded/loaded PiD bundle {upscale_model_name}: model={spec.diffusion_model}")
    return model, clip, info


def _validate_pid_input_image(image, input_size=PID_INPUT_SIZE):
    if image is None or not torch.is_tensor(image):
        raise ValueError("TBG PID upscale requires an IMAGE tensor input")
    if image.ndim != 4:
        raise ValueError(f"TBG PID upscale requires a batched IMAGE tensor, got shape {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError(f"TBG PID upscale requires exactly one input image, got batch {image.shape[0]}")

    height, width = int(image.shape[1]), int(image.shape[2])
    input_size = int(input_size or PID_INPUT_SIZE)
    if width != input_size or height != input_size:
        raise ValueError(f"TBG PID upscale requires a {input_size}x{input_size} input image, got {width}x{height}")


def _call_pid_conditioning(positive, source_latent, latent_format, degrade_sigma=0.1):
    direct_formats = {
        "sdxl": comfy.latent_formats.SDXL,
        "wan21": comfy.latent_formats.Wan21,
        "qwen_image": comfy.latent_formats.Wan21,
    }
    if latent_format in direct_formats:
        samples = source_latent["samples"]
        lq_latent = direct_formats[latent_format]().process_in(samples)
        sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)
        return node_helpers.conditioning_set_values(
            positive,
            {"lq_latent": lq_latent, "degrade_sigma": sigma_t},
        )

    pid_cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("PiDConditioning")
    if pid_cls is None:
        try:
            from comfy_extras.nodes_pid import PiDConditioning as pid_cls
        except Exception as exc:
            raise RuntimeError("PiDConditioning node is not available in this ComfyUI install") from exc

    fn = getattr(pid_cls, "execute", None)
    if fn is None:
        obj = pid_cls()
        fn = getattr(obj, "execute", None)
    if fn is None:
        raise RuntimeError("PiDConditioning node is available but has no callable execute method")

    try:
        signature = inspect.signature(fn)
        kwargs = {}
        for param in signature.parameters.values():
            if param.name in {"self", "cls"}:
                continue
            if param.name in {"positive", "conditioning"}:
                kwargs[param.name] = positive
            elif param.name in {"latent", "latent_image", "samples"}:
                kwargs[param.name] = source_latent
            elif param.name in {"latent_format", "mode", "pid_mode", "backbone", "model_type", "type"}:
                kwargs[param.name] = latent_format
            elif param.name in {"degrade_sigma", "sigma", "noise_sigma"}:
                kwargs[param.name] = float(degrade_sigma)
        if kwargs:
            return _as_tuple(fn(**kwargs))[0]
    except Exception:
        pass

    attempts = (
        lambda: fn(positive=positive, latent=source_latent, latent_format=latent_format, degrade_sigma=float(degrade_sigma)),
        lambda: fn(positive, source_latent, latent_format, float(degrade_sigma)),
    )
    last_error = None
    for attempt in attempts:
        try:
            return _as_tuple(attempt())[0]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"PiDConditioning call failed: {last_error}")


def _empty_pid_latent(width=PID_OUTPUT_SIZE, height=PID_OUTPUT_SIZE):
    samples = torch.zeros((1, 3, height, width), device=mm.intermediate_device())
    return {"samples": samples}


def _pid_torch_device(preferred=None):
    if isinstance(preferred, torch.device) and preferred.type != "cpu":
        return preferred
    if torch.cuda.is_available():
        return mm.get_torch_device()
    return preferred or mm.intermediate_device()


def _to_pid_device(tensor, device, dtype=None):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    return tensor.to(device=device, dtype=dtype or tensor.dtype)


def _pid_output_base_latent(tile, output_vae, label="source tile", output_size=PID_OUTPUT_SIZE):
    if tile is None or not torch.is_tensor(tile):
        print("[TBG PID] output latent: empty pixel-space latent (missing base tile)")
        return _empty_pid_latent(output_size, output_size)
    output_size = int(output_size or PID_OUTPUT_SIZE)
    base_image = nodes.ImageScale().upscale(tile, "bilinear", output_size, output_size, False)[0]
    print(f"[TBG PID] output latent: 4x pixel-space base from {label}")
    return nodes.VAEEncode().encode(output_vae, base_image)[0]


def _pid_output_context_latent(context_4x, output_vae, label="4x context tile"):
    if context_4x is None or not torch.is_tensor(context_4x):
        print("[TBG PID] output latent: empty pixel-space latent (missing 4x context)")
        return _empty_pid_latent()
    print(f"[TBG PID] output latent: pixel-space base from {label}")
    return nodes.VAEEncode().encode(output_vae, context_4x)[0]


def _ensure_bhwc_image(image, label):
    if image is None:
        return None
    if not torch.is_tensor(image):
        raise ValueError(f"TBG PID {label} must be a torch tensor.")
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"TBG PID {label} must be [B,H,W,C], got {getattr(image, 'shape', None)}.")


def _resize_pid_context_image(image, width, height, label, device=None, dtype=None):
    image = _ensure_bhwc_image(image, label)
    if image is None:
        return None
    if device is not None:
        image = image.to(device=device, dtype=dtype or image.dtype)
    if int(image.shape[2]) == int(width) and int(image.shape[1]) == int(height):
        return image
    return nodes.ImageScale().upscale(image, "lanczos", int(width), int(height), False)[0]


def _mask_to_bhw(mask):
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise ValueError("TBG PID inpaint mask must be a torch tensor.")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask[:, 0]
        return mask[..., 0]
    raise ValueError(f"TBG PID inpaint mask must be [H,W], [B,H,W], or [B,H,W,C], got {getattr(mask, 'shape', None)}.")


def _resize_pid_mask(mask, width, height, device=None, dtype=None):
    mask = _mask_to_bhw(mask)
    if mask is None:
        return None
    mask = mask.to(device=device or mask.device, dtype=dtype or torch.float32).clamp(0.0, 1.0)
    if int(mask.shape[-1]) != int(width) or int(mask.shape[-2]) != int(height):
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return mask.clamp(0.0, 1.0)


def _pid_mask_to_bhwc(mask):
    if mask is None:
        return None
    return mask.unsqueeze(-1)


def _pid_mask_to_noise_mask(mask):
    if mask is None:
        return None
    return mask.unsqueeze(1)


def _shift_pid_image(image, shift_x=0, shift_y=0, fill_image=None):
    if not torch.is_tensor(image) or image.ndim != 4:
        return image
    shift_x = int(shift_x or 0)
    shift_y = int(shift_y or 0)
    if shift_x == 0 and shift_y == 0:
        return image

    batch, height, width, channels = image.shape
    result = image.clone()
    fill = fill_image
    if torch.is_tensor(fill):
        fill = fill.to(device=image.device, dtype=image.dtype)
        if fill.ndim != 4 or int(fill.shape[1]) != height or int(fill.shape[2]) != width:
            fill = None
    if fill is None:
        fill = image
    result.copy_(fill)

    src_x0 = max(0, -shift_x)
    src_y0 = max(0, -shift_y)
    src_x1 = min(width, width - shift_x) if shift_x >= 0 else width
    src_y1 = min(height, height - shift_y) if shift_y >= 0 else height
    dst_x0 = max(0, shift_x)
    dst_y0 = max(0, shift_y)
    copy_w = max(0, src_x1 - src_x0)
    copy_h = max(0, src_y1 - src_y0)
    if copy_w > 0 and copy_h > 0:
        result[:, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w, :] = image[:, src_y0:src_y1, src_x0:src_x1, :]
    return result


def _rgb_mean_shift_255(reference, target):
    if reference is None or target is None:
        return None
    if not torch.is_tensor(reference) or not torch.is_tensor(target):
        return None
    if reference.ndim != 4 or target.ndim != 4:
        return None
    tile_h = min(int(reference.shape[1]), int(target.shape[1]))
    tile_w = min(int(reference.shape[2]), int(target.shape[2]))
    if tile_h <= 0 or tile_w <= 0:
        return None
    ref = reference[:, :tile_h, :tile_w, :3].to(device=target.device, dtype=torch.float32)
    tgt = target[:, :tile_h, :tile_w, :3].to(dtype=torch.float32)
    shift = (tgt - ref).mean(dim=(0, 1, 2)) * 255.0
    return tuple(float(v) for v in shift.detach().cpu())


def _format_rgb_shift(shift):
    if shift is None:
        return "n/a"
    return "(" + ", ".join(f"{v:+.2f}" for v in shift) + ")"


def _load_source_vae(spec):
    vae_names = set(folder_paths.get_filename_list("vae"))
    vae_names.update(nodes.VAELoader.vae_list(nodes.VAELoader))
    normalized_names = {name.replace("\\", "/"): name for name in vae_names}

    for candidate in spec.source_vae_candidates:
        if candidate in vae_names:
            return nodes.VAELoader().load_vae(candidate)[0]
        normalized = candidate.replace("\\", "/")
        if normalized in normalized_names:
            return nodes.VAELoader().load_vae(normalized_names[normalized])[0]

    raise RuntimeError(
        "TBG PID upscale could not find a source VAE for PiD conditioning. "
        f"Tried: {', '.join(spec.source_vae_candidates)}"
    )


def _load_pid_sampler(sampler_name="lcm"):
    if sampler_name == "pid_sde" and "pid_sde" not in comfy.samplers.KSampler.SAMPLERS:
        raise RuntimeError(
            "TBG PID mode requires the ETUR-bundled 'pid_sde' sampler, but it was not registered. "
            "Restart ComfyUI and check the TBG ETUR startup log for '[TBG PiD SDE] registered samplers'."
        )
    try:
        return KSamplerSelect.execute(sampler_name)[0]
    except Exception as exc:
        raise RuntimeError(f"TBG PID upscale could not load sampler '{sampler_name}'.") from exc


def _sampler_debug_name(sampler):
    fn = getattr(sampler, "sampler_function", None)
    if fn is not None:
        return getattr(fn, "__name__", fn.__class__.__name__)
    return sampler.__class__.__name__


def select_pid_refiner_model(latent, model_type=None):
    return _pid_model_for_latent(latent, model_type=model_type)


def _load_pid_model_and_clip(upscale_model_name, clip=None, pid_model=None, allow_hf_download=True):
    spec = _pid_spec_or_refresh(upscale_model_name, force_refresh=False)

    print(
        f"[TBG PID] loading {upscale_model_name}: "
        f"model={spec.diffusion_model} text_encoder={spec.text_encoder} latent_format={spec.latent_format}"
    )
    diffusion_model = None if pid_model is not None else _ensure_hf_model_file(
        "diffusion_models",
        "diffusion_models",
        spec.diffusion_model,
        allow_download=allow_hf_download,
    )
    text_encoder = _ensure_hf_model_file(
        "text_encoders",
        "text_encoders",
        spec.text_encoder,
        allow_download=allow_hf_download,
    )
    if pid_model is not None:
        model = pid_model
        print(f"[TBG PID] using connected PID diffusion model override for {upscale_model_name}")
    else:
        print(f"[TBG PID] local diffusion path: {_local_hf_model_path('diffusion_models', 'diffusion_models', spec.diffusion_model)}")
        model = nodes.UNETLoader().load_unet(diffusion_model, "default")[0]
    clip = clip if clip is not None else nodes.CLIPLoader().load_clip(text_encoder, type="pixeldit")[0]
    return spec, model, clip


def _load_pid_runtime(upscale_model_name, clip=None, source_vae=None, sampler=None, sampler_name="lcm", scheduler="simple", steps=4, denoise=1.0, load_source_vae=True, pid_model=None, allow_hf_download=True):
    spec, model, clip = _load_pid_model_and_clip(upscale_model_name, clip=clip, pid_model=pid_model, allow_hf_download=allow_hf_download)
    source_vae = source_vae if source_vae is not None else (_load_source_vae(spec) if load_source_vae else None)
    output_vae = nodes.VAELoader().load_vae("pixel_space")[0]
    sampler_source = "override input" if sampler is not None else f"dropdown '{sampler_name}'"
    sampler = sampler if sampler is not None else _load_pid_sampler(sampler_name)
    print(f"[TBG PID] using sampler from {sampler_source}: {_sampler_debug_name(sampler)}")
    denoise = max(0.01, min(1.0, float(denoise)))
    sigmas = BasicScheduler.execute(model, scheduler, int(steps), denoise)[0]
    try:
        sigma_head = float(sigmas[0]) if len(sigmas) else 0.0
    except Exception:
        sigma_head = 0.0
    print(f"[TBG PID] schedule steps={int(steps)} denoise={denoise:.2f} first_sigma={sigma_head:.4f}")
    return SimpleNamespace(model=model, clip=clip, source_vae=source_vae, output_vae=output_vae, sampler=sampler, sigmas=sigmas, spec=spec)


def load_pid_refiner_runtime(latent, model_type=None, sampler_name=PID_DEFAULT_SAMPLER, scheduler="simple", steps=4, denoise=1.0, clip=None, pid_model=None, allow_hf_download=True):
    upscale_model_name = select_pid_refiner_model(latent, model_type=model_type)
    spec = _pid_spec_or_refresh(upscale_model_name, force_refresh=False)
    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type in {"flux2", "flux 2"} and spec.diffusion_model != "pid_flux2_1024_to_4096_4step_2606_bf16.safetensors":
        raise RuntimeError(
            "TBG PID Flux2 auto selector must use pid_flux2_1024_to_4096_4step_2606_bf16.safetensors, "
            f"but resolved {spec.diffusion_model}."
        )
    runtime = _load_pid_runtime(
        upscale_model_name,
        clip=clip,
        sampler_name=sampler_name,
        scheduler=scheduler,
        steps=steps,
        denoise=denoise,
        source_vae=None,
        load_source_vae=False,
        pid_model=pid_model,
        allow_hf_download=allow_hf_download,
    )
    runtime.upscale_model_name = upscale_model_name
    return runtime


def unload_pid_refiner_runtime(runtime=None, reason="", aggressive=False):
    if runtime is not None:
        runtime.model = None
        runtime.clip = None
        runtime.source_vae = None
        runtime.output_vae = None
        runtime.sampler = None
        runtime.sigmas = None
    print(f"[TBG PID] released PiD model/CLIP runtime{': ' + str(reason) if reason else ''}")
    mm.unload_all_models()
    mm.soft_empty_cache()
    if aggressive and torch.cuda.is_available():
        torch.cuda.empty_cache()


def upscale_pid_tile_1024(tile, runtime, positive, negative, seed=0, degrade_sigma=0.1, source_latent=None):
    if tile is None:
        if source_latent is None:
            raise ValueError("TBG PID tile upscale requires either an IMAGE tile or a LATENT tile.")
    else:
        _validate_pid_input_image(tile, input_size=runtime.spec.input_size)

    if source_latent is None:
        source_latent = nodes.VAEEncode().encode(runtime.source_vae, tile)[0]
    positive = _call_pid_conditioning(positive, source_latent, runtime.spec.latent_format, degrade_sigma)
    output_base_latent = (
        _empty_pid_latent(runtime.spec.output_size, runtime.spec.output_size)
        if tile is None
        else _pid_output_base_latent(tile, runtime.output_vae, output_size=runtime.spec.output_size)
    )
    sampled = SamplerCustom.execute(
        runtime.model,
        True,
        int(seed or 0),
        float(getattr(runtime, "cfg", 1.0)),
        positive,
        negative,
        runtime.sampler,
        runtime.sigmas,
        output_base_latent,
    )[0]
    return nodes.VAEDecode().decode(runtime.output_vae, sampled)[0]


def _clone_latent_with_samples(latent, samples):
    # PiD receives only the sampled latent tensor as source conditioning.
    # Do not forward ETUR/Comfy metadata such as noise masks, Flux2 private
    # masks, reference latents, or model_conds into the PiD conditioning pass.
    return {"samples": samples}


def _pad_latent_samples(samples, target_h, target_w):
    _, channels, height, width = samples.shape
    if height == target_h and width == target_w:
        return samples
    padded = torch.zeros((1, channels, target_h, target_w), dtype=samples.dtype, device=samples.device)
    copy_h = min(height, target_h)
    copy_w = min(width, target_w)
    padded[:, :, :copy_h, :copy_w] = samples[:, :, :copy_h, :copy_w]
    return padded


def _latent_spatial_downscale(samples):
    channels = int(samples.shape[1])
    if channels == 128:
        return 16
    return 8


def _latent_tile_from_image_region(latent, source_width, source_height, x, y, tile_w, tile_h, scale=1.0, input_size=PID_INPUT_SIZE):
    if latent is None:
        return None
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("TBG PID latent input must be a LATENT dict with 'samples'.")
    samples = latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim != 4 or samples.shape[0] != 1:
        raise ValueError(f"TBG PID latent input requires samples [1,C,H,W], got {getattr(samples, 'shape', None)}.")

    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"TBG PID latent input got invalid source size {source_width}x{source_height}.")

    latent_h = int(samples.shape[2])
    latent_w = int(samples.shape[3])
    inv_scale = 1.0 / float(scale or 1.0)
    src_x0 = float(x) * inv_scale
    src_y0 = float(y) * inv_scale
    src_x1 = (float(x) + float(tile_w)) * inv_scale
    src_y1 = (float(y) + float(tile_h)) * inv_scale

    lx0 = int(round(src_x0 * latent_w / float(source_width)))
    ly0 = int(round(src_y0 * latent_h / float(source_height)))
    lx1 = int(round(src_x1 * latent_w / float(source_width)))
    ly1 = int(round(src_y1 * latent_h / float(source_height)))
    lx0 = max(0, min(latent_w - 1, lx0))
    ly0 = max(0, min(latent_h - 1, ly0))
    lx1 = max(lx0 + 1, min(latent_w, lx1))
    ly1 = max(ly0 + 1, min(latent_h, ly1))

    input_size = int(input_size or PID_INPUT_SIZE)
    target_w = max(1, int(round(input_size * latent_w / float(source_width))))
    target_h = max(1, int(round(input_size * latent_h / float(source_height))))
    source_samples = samples[:, :, ly0:ly1, lx0:lx1]
    padded_samples = _pad_latent_samples(source_samples, target_h, target_w)
    return _clone_latent_with_samples(latent, padded_samples)


def _source_size_from_latent(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("TBG PID latent input must be a LATENT dict with 'samples'.")
    samples = latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim != 4 or samples.shape[0] != 1:
        raise ValueError(f"TBG PID latent input requires samples [1,C,H,W], got {getattr(samples, 'shape', None)}.")
    latent_scale = _latent_spatial_downscale(samples)
    return int(samples.shape[3]) * latent_scale, int(samples.shape[2]) * latent_scale


def _pid_model_for_latent(latent, model_type=None):
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("TBG PID refiner decode requires a latent dict with 'samples'.")
    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type in {"flux2", "flux 2"}:
        return "SuperResolution/PID Flux2 1K to 4K BF16"
    if normalized_model_type in {"flux1", "flux 1", "flux1 kontext", "flux 1 kontext", "z-image"}:
        return "SuperResolution/PID Flux1 1K to 4K BF16"
    if normalized_model_type in {"sdxl", "stable diffusion xl", "stable-diffusion-xl"}:
        return "SuperResolution/PID SDXL 1K to 4K BF16"
    if normalized_model_type in {"sd3", "stable diffusion 3", "stable-diffusion-3"}:
        return "SuperResolution/PID SD3 1K to 4K BF16"
    if normalized_model_type in {"qwen image", "qwen image edit", "qwenimage", "qwenimage edit"}:
        return "SuperResolution/PID QwenImage 1K to 4K BF16"
    channels = int(samples.shape[1])
    if channels == 128:
        return "SuperResolution/PID Flux2 1K to 4K BF16"
    if channels == 16:
        return "SuperResolution/PID Flux1 1K to 4K BF16"
    raise ValueError(
        f"TBG PID refiner decode supports Flux1/Z-image/SDXL/SD3/Qwen 16-channel or Flux2 128-channel latents, got {channels} channels."
    )


def _pid_upscale_model_for_latent(upscale_model_name, latent):
    if latent is None:
        return upscale_model_name
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not torch.is_tensor(samples) or samples.ndim != 4:
        return upscale_model_name
    if any(family in str(upscale_model_name) for family in ("QwenImage", "SDXL", "SD3")):
        return upscale_model_name
    channels = int(samples.shape[1])
    wants_flux2 = channels == 128
    has_flux2 = "Flux2" in str(upscale_model_name)
    if wants_flux2 == has_flux2:
        return upscale_model_name
    precision = "MXFP8" if "MXFP8" in str(upscale_model_name) else "BF16"
    family = "Flux2" if wants_flux2 else "Flux1"
    candidate = f"SuperResolution/PID {family} 1K to 4K {precision}"
    if candidate in PID_UPSCALE_SPECS:
        print(f"[TBG PID] switched PID model to {candidate} for {channels}-channel latent input")
        return candidate
    return upscale_model_name


def run_pid_refiner_latent_decode(
    latent,
    source_width,
    source_height,
    base_tile_image=None,
    base_context_image=None,
    inpaint_mask=None,
    debug_callback=None,
    prompt_text="",
    seed=0,
    model_type=None,
    degrade_sigma=0.1,
    sampler_name=PID_DEFAULT_SAMPLER,
    scheduler="simple",
    steps=4,
    cfg=1.0,
    color_match_fn=None,
    color_match_method="lab full color match",
    runtime=None,
    decode_shift_4x=(0, 0),
    segment_post_context_preserve=False,
    post_context_mask=None,
):
    if latent is None or "samples" not in latent:
        raise ValueError("TBG PID refiner decode requires a latent dict.")
    samples = latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim != 4 or samples.shape[0] != 1:
        raise ValueError(f"TBG PID refiner decode requires latent samples [1,C,H,W], got {getattr(samples, 'shape', None)}.")

    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"TBG PID refiner decode got invalid source size {source_width}x{source_height}.")

    if runtime is None:
        runtime = load_pid_refiner_runtime(
            latent,
            model_type=model_type,
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=steps,
            denoise=1.0,
        )
    runtime.cfg = float(cfg)
    positive = nodes.CLIPTextEncode().encode(runtime.clip, prompt_text or "")[0]
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]

    latent_h = int(samples.shape[2])
    latent_w = int(samples.shape[3])
    latent_tile_w = max(1, int(round(PID_INPUT_SIZE * latent_w / float(source_width))))
    latent_tile_h = max(1, int(round(PID_INPUT_SIZE * latent_h / float(source_height))))
    output_width = source_width * PID_SCALE
    output_height = source_height * PID_SCALE
    pid_device = _pid_torch_device(samples.device)

    context_image_4x = _resize_pid_context_image(
        base_context_image,
        output_width,
        output_height,
        "base context image",
        device=pid_device,
        dtype=torch.float32,
    )
    inpaint_mask_4x = _resize_pid_mask(
        inpaint_mask,
        output_width,
        output_height,
        device=pid_device,
        dtype=samples.dtype,
    )
    post_context_mask_4x = _resize_pid_mask(
        post_context_mask if post_context_mask is not None else inpaint_mask,
        output_width,
        output_height,
        device=pid_device,
        dtype=samples.dtype,
    )

    if base_tile_image is not None:
        if not torch.is_tensor(base_tile_image) or base_tile_image.ndim != 4:
            raise ValueError(
                f"TBG PID refiner base image must be [B,H,W,C], got {getattr(base_tile_image, 'shape', None)}."
            )
        base_h = int(base_tile_image.shape[1])
        base_w = int(base_tile_image.shape[2])
        base_tile_image = _to_pid_device(base_tile_image, pid_device, torch.float32)
        if base_w != source_width or base_h != source_height:
            base_tile_image = nodes.ImageScale().upscale(
                base_tile_image,
                "lanczos",
                source_width,
                source_height,
                False,
            )[0]

    pid_tiles = []
    grid_specs = []
    tile_overlap = int(PID_DEFAULT_OVERLAP)
    stitch_feather = int(PID_DEFAULT_STITCH_FEATHER * PID_SCALE)
    tile_specs = _pid_tile_specs(source_width, source_height, overlap=tile_overlap)
    if len(tile_specs) > 1:
        print(
            "[TBG PID Refiner] tiled latent decode "
            f"source={source_width}x{source_height} "
            f"tiles={len(tile_specs)} overlap={tile_overlap}px "
            f"output_overlap={tile_overlap * PID_SCALE}px "
            f"stitch_feather={stitch_feather}px"
        )
    allow_region_color_lock = color_match_fn is not None and len(tile_specs) == 1
    if color_match_fn is not None and not allow_region_color_lock:
        print(
            "[TBG PID Refiner] color lock deferred until stitched ETUR tile; "
            f"internal_regions={len(tile_specs)}"
        )
    for index, (row, col, x, y, tile_w, tile_h) in enumerate(tile_specs):
        lx0 = int(round(x * latent_w / float(source_width)))
        ly0 = int(round(y * latent_h / float(source_height)))
        lx1 = int(round((x + tile_w) * latent_w / float(source_width)))
        ly1 = int(round((y + tile_h) * latent_h / float(source_height)))
        lx1 = max(lx0 + 1, min(latent_w, lx1))
        ly1 = max(ly0 + 1, min(latent_h, ly1))

        source_samples = samples[:, :, ly0:ly1, lx0:lx1]
        padded_samples = _pad_latent_samples(source_samples, latent_tile_h, latent_tile_w)
        padded_samples = padded_samples.to(device=pid_device)
        source_latent = _clone_latent_with_samples(latent, padded_samples)
        conditioned = _call_pid_conditioning(positive, source_latent, runtime.spec.latent_format, degrade_sigma)
        output_base_latent = _empty_pid_latent()
        out_w = int(tile_w) * PID_SCALE
        out_h = int(tile_h) * PID_SCALE
        out_x = int(x) * PID_SCALE
        out_y = int(y) * PID_SCALE
        context_region = None
        mask_region = None
        post_context_mask_region = None
        if context_image_4x is not None:
            context_region = context_image_4x[:, out_y:out_y + out_h, out_x:out_x + out_w, :]
            context_region = _pad_tile_to_pid_output_size(context_region)
            output_base_latent = _pid_output_context_latent(context_region, runtime.output_vae, "4x inpaint context tile")
        elif base_tile_image is not None:
            base_region = base_tile_image[:, y:y + tile_h, x:x + tile_w, :]
            base_region = _pad_tile_to_pid_size(base_region)
            output_base_latent = _pid_output_base_latent(base_region, runtime.output_vae, "refiner sampled tile")
        if torch.is_tensor(output_base_latent.get("samples")):
            output_base_latent["samples"] = output_base_latent["samples"].to(device=pid_device)
        if inpaint_mask_4x is not None:
            base_samples = output_base_latent.get("samples")
            mask_device = base_samples.device if torch.is_tensor(base_samples) else samples.device
            mask_dtype = base_samples.dtype if torch.is_tensor(base_samples) else samples.dtype
            mask_region = inpaint_mask_4x[:, out_y:out_y + out_h, out_x:out_x + out_w]
            mask_region = _resize_pid_mask(mask_region, PID_OUTPUT_SIZE, PID_OUTPUT_SIZE, device=mask_device, dtype=mask_dtype)
            output_base_latent["noise_mask"] = _pid_mask_to_noise_mask(mask_region)
        if post_context_mask_4x is not None:
            post_context_mask_region = post_context_mask_4x[:, out_y:out_y + out_h, out_x:out_x + out_w]
            post_context_mask_region = _resize_pid_mask(
                post_context_mask_region,
                PID_OUTPUT_SIZE,
                PID_OUTPUT_SIZE,
                device=pid_device,
                dtype=samples.dtype,
            )
        sampled = SamplerCustom.execute(
            runtime.model,
            True,
            int(seed or 0) + index,
            float(getattr(runtime, "cfg", 1.0)),
            conditioned,
            negative,
            runtime.sampler,
            runtime.sigmas,
            output_base_latent,
        )[0]
        raw_pid_tile = nodes.VAEDecode().decode(runtime.output_vae, sampled)[0]
        raw_pid_tile = raw_pid_tile[:, :out_h, :out_w, :]
        shift_x_4x, shift_y_4x = decode_shift_4x or (0, 0)
        if int(shift_x_4x or 0) != 0 or int(shift_y_4x or 0) != 0:
            raw_pid_tile = _shift_pid_image(
                raw_pid_tile,
                shift_x=int(shift_x_4x or 0),
                shift_y=int(shift_y_4x or 0),
                fill_image=context_region[:, :out_h, :out_w, :] if context_region is not None else None,
            )
            print(
                "[TBG PID Refiner] decode spatial correction "
                f"region {index + 1}/{len(tile_specs)} shift=({int(shift_x_4x or 0)},{int(shift_y_4x or 0)})px"
            )
        pid_tile = raw_pid_tile
        debug_raw = raw_pid_tile
        debug_color_reference = None
        debug_color_matched = None
        debug_post_context_matched = None
        debug_post_context_mask = None
        color_reference_4x = None
        if allow_region_color_lock and base_tile_image is not None:
            try:
                reference_region = base_tile_image[:, y:y + tile_h, x:x + tile_w, :]
                reference_region = _pad_tile_to_pid_size(reference_region)
                color_reference_4x = nodes.ImageScale().upscale(
                    reference_region.to(device=raw_pid_tile.device, dtype=raw_pid_tile.dtype),
                    "lanczos",
                    PID_OUTPUT_SIZE,
                    PID_OUTPUT_SIZE,
                    False,
                )[0][:, :out_h, :out_w, :]
                before_shift = _rgb_mean_shift_255(color_reference_4x, raw_pid_tile)
                matched = color_match_fn(color_reference_4x, raw_pid_tile, color_match_method)
                if isinstance(matched, tuple):
                    matched = matched[0]
                if torch.is_tensor(matched):
                    pid_tile = matched[:, :out_h, :out_w, :].to(device=raw_pid_tile.device, dtype=raw_pid_tile.dtype).clamp(0.0, 1.0)
                    after_shift = _rgb_mean_shift_255(color_reference_4x, pid_tile)
                    debug_color_reference = color_reference_4x
                    debug_color_matched = pid_tile
                    print(
                        "[TBG PID Refiner] color lock "
                        f"region {index + 1}/{len(tile_specs)} "
                        f"mean_shift_before={_format_rgb_shift(before_shift)} "
                        f"mean_shift_after={_format_rgb_shift(after_shift)}"
                    )
            except Exception as exc:
                print(f"[TBG PID Refiner] color lock failed on region {index + 1}, using raw PiD tile: {exc}")
        if context_region is not None and mask_region is not None:
            context_crop = context_region[:, :out_h, :out_w, :]
            blend_mask = post_context_mask_region if post_context_mask_region is not None else mask_region
            mask_crop = _pid_mask_to_bhwc(blend_mask[:, :out_h, :out_w]).to(device=pid_tile.device, dtype=pid_tile.dtype)
            if segment_post_context_preserve:
                mask_crop = mask_crop.clamp(0.0, 1.0)
                debug_post_context_mask = mask_crop
                print(
                    "[TBG PID Refiner] segment post-context blend uses feathered segment mask "
                    f"region {index + 1}/{len(tile_specs)}"
                )
            else:
                debug_post_context_mask = mask_crop
            context_crop = context_crop.to(device=pid_tile.device, dtype=pid_tile.dtype)
            pid_tile = context_crop * (1.0 - mask_crop) + pid_tile * mask_crop
            if allow_region_color_lock and color_reference_4x is not None and not segment_post_context_preserve:
                try:
                    before_shift = _rgb_mean_shift_255(color_reference_4x, pid_tile)
                    matched = color_match_fn(color_reference_4x, pid_tile, color_match_method)
                    if isinstance(matched, tuple):
                        matched = matched[0]
                    if torch.is_tensor(matched):
                        pid_tile = matched[:, :out_h, :out_w, :].to(device=pid_tile.device, dtype=pid_tile.dtype).clamp(0.0, 1.0)
                        after_shift = _rgb_mean_shift_255(color_reference_4x, pid_tile)
                        debug_post_context_matched = pid_tile
                        print(
                            "[TBG PID Refiner] post-context color lock "
                            f"region {index + 1}/{len(tile_specs)} "
                            f"mean_shift_before={_format_rgb_shift(before_shift)} "
                            f"mean_shift_after={_format_rgb_shift(after_shift)}"
                        )
                except Exception as exc:
                    print(f"[TBG PID Refiner] post-context color lock failed on region {index + 1}: {exc}")
        if debug_callback is not None:
            debug_callback(
                index,
                debug_raw,
                pid_tile,
                color_reference=debug_color_reference,
                color_matched=debug_color_matched,
                post_context_matched=debug_post_context_matched,
                post_context_mask=debug_post_context_mask,
            )
        pid_tiles.append(pid_tile)
        grid_specs.append((row, col, index, out_x, out_y, out_w, out_h))
        print(f"[TBG PID Refiner] tile region {index + 1}/{len(tile_specs)} decoded with {sampler_name}")

    return _local_rebuild(
        pid_tiles,
        grid_specs,
        output_width,
        output_height,
        stitch_feather=stitch_feather,
    )


def run_pixeldit_pid_upscale(image, upscale_model_name, prompt_text="", seed=0, clip=None, source_vae=None, degrade_sigma=0.1):
    runtime = _load_pid_runtime(upscale_model_name, clip=clip, source_vae=source_vae)
    positive = nodes.CLIPTextEncode().encode(runtime.clip, prompt_text or "")[0]
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    decoded = upscale_pid_tile_1024(image, runtime, positive, negative, seed, degrade_sigma)
    mm.soft_empty_cache()
    return decoded


def _resize_for_pid_minimum(image, input_size=PID_INPUT_SIZE):
    height, width = int(image.shape[1]), int(image.shape[2])
    shortest = min(width, height)
    input_size = int(input_size or PID_INPUT_SIZE)
    if shortest >= input_size:
        return image, 1.0

    scale = input_size / float(shortest)
    new_width = max(input_size, int(round(width * scale)))
    new_height = max(input_size, int(round(height * scale)))
    resized = nodes.ImageScale().upscale(image, "bilinear", new_width, new_height, False)[0]
    return resized, scale


def _axis_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pid_tile_specs(width, height, overlap=PID_DEFAULT_OVERLAP, input_size=PID_INPUT_SIZE):
    specs = []
    input_size = int(input_size or PID_INPUT_SIZE)
    overlap = max(0, min(int(overlap), input_size - 1))
    y_starts = _axis_starts(height, input_size, overlap)
    x_starts = _axis_starts(width, input_size, overlap)
    for row, y in enumerate(y_starts):
        tile_h = min(input_size, height - y)
        for col, x in enumerate(x_starts):
            tile_w = min(input_size, width - x)
            specs.append((row, col, x, y, tile_w, tile_h))
    return specs


def _pad_tile_to_pid_size(tile, input_size=PID_INPUT_SIZE):
    _, height, width, channels = tile.shape
    input_size = int(input_size or PID_INPUT_SIZE)
    if width == input_size and height == input_size:
        return tile
    padded = torch.zeros(
        (1, input_size, input_size, channels),
        dtype=tile.dtype,
        device=tile.device,
    )
    padded[:, :height, :width, :] = tile
    return padded


def _pad_tile_to_pid_output_size(tile, output_size=PID_OUTPUT_SIZE):
    _, height, width, channels = tile.shape
    output_size = int(output_size or PID_OUTPUT_SIZE)
    if width == output_size and height == output_size:
        return tile
    padded = torch.zeros(
        (1, output_size, output_size, channels),
        dtype=tile.dtype,
        device=tile.device,
    )
    copy_h = min(height, output_size)
    copy_w = min(width, output_size)
    padded[:, :copy_h, :copy_w, :] = tile[:, :copy_h, :copy_w, :]
    return padded


def _tile_feather_mask(tile_w, tile_h, x, y, width, height, feather, device, dtype):
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


def _tile_ownership_feather_mask(tile_w, tile_h, x, y, width, height, specs, feather, device, dtype):
    feather = int(max(0, min(feather, tile_w, tile_h)))
    mask = torch.ones((1, tile_h, tile_w, 1), device=device, dtype=dtype)
    if feather <= 0:
        return mask

    x = int(x)
    y = int(y)
    tile_w = int(tile_w)
    tile_h = int(tile_h)
    x_end = x + tile_w
    y_end = y + tile_h

    def vertical_intersects(spec):
        _, _, _, sx, sy, sw, sh = spec[:7]
        return int(sy) < y_end and int(sy) + int(sh) > y

    def horizontal_intersects(spec):
        _, _, _, sx, sy, sw, sh = spec[:7]
        return int(sx) < x_end and int(sx) + int(sw) > x

    current_row = current_col = None
    for spec in specs:
        row, col, _, sx, sy, sw, sh = spec[:7]
        if int(sx) == x and int(sy) == y:
            current_row = int(row)
            current_col = int(col)
            break

    for spec in specs:
        row, col, _, sx, sy, sw, sh = spec[:7]
        sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
        if sx == x and sy == y:
            continue

        same_row = current_row is None or int(row) == current_row
        same_col = current_col is None or int(col) == current_col

        if same_row and vertical_intersects(spec) and sx < x and sx + sw > x:
            overlap_start = x
            overlap_end = min(x_end, sx + sw)
            seam = (overlap_start + overlap_end) * 0.5
            local_start = max(0, int(round(seam - feather * 0.5)) - x)
            local_end = min(tile_w, int(round(seam + feather * 0.5)) - x)
            if local_start > 0:
                mask[:, :, :local_start, :] = 0.0
            if local_end > local_start:
                ramp = torch.linspace(0.0, 1.0, local_end - local_start, device=device, dtype=dtype).view(1, 1, -1, 1)
                mask[:, :, local_start:local_end, :] *= ramp

        if same_row and vertical_intersects(spec) and sx > x and sx < x_end:
            overlap_start = sx
            overlap_end = x_end
            seam = (overlap_start + overlap_end) * 0.5
            local_start = max(0, int(round(seam - feather * 0.5)) - x)
            local_end = min(tile_w, int(round(seam + feather * 0.5)) - x)
            if local_end < tile_w:
                mask[:, :, local_end:, :] = 0.0
            if local_end > local_start:
                ramp = torch.linspace(1.0, 0.0, local_end - local_start, device=device, dtype=dtype).view(1, 1, -1, 1)
                mask[:, :, local_start:local_end, :] *= ramp

        if same_col and horizontal_intersects(spec) and sy < y and sy + sh > y:
            overlap_start = y
            overlap_end = min(y_end, sy + sh)
            seam = (overlap_start + overlap_end) * 0.5
            local_start = max(0, int(round(seam - feather * 0.5)) - y)
            local_end = min(tile_h, int(round(seam + feather * 0.5)) - y)
            if local_start > 0:
                mask[:, :local_start, :, :] = 0.0
            if local_end > local_start:
                ramp = torch.linspace(0.0, 1.0, local_end - local_start, device=device, dtype=dtype).view(1, -1, 1, 1)
                mask[:, local_start:local_end, :, :] *= ramp

        if same_col and horizontal_intersects(spec) and sy > y and sy < y_end:
            overlap_start = sy
            overlap_end = y_end
            seam = (overlap_start + overlap_end) * 0.5
            local_start = max(0, int(round(seam - feather * 0.5)) - y)
            local_end = min(tile_h, int(round(seam + feather * 0.5)) - y)
            if local_end < tile_h:
                mask[:, local_end:, :, :] = 0.0
            if local_end > local_start:
                ramp = torch.linspace(1.0, 0.0, local_end - local_start, device=device, dtype=dtype).view(1, -1, 1, 1)
                mask[:, local_start:local_end, :, :] *= ramp

    return mask.clamp(0.0, 1.0)


def _pid_reference_stabilization_mask(tile_w, tile_h, x, y, width, height, border, device, dtype):
    border = int(max(0, min(border, tile_w // 2, tile_h // 2)))
    mask = torch.ones((1, tile_h, tile_w, 1), device=device, dtype=dtype)
    if border <= 0:
        return mask

    ramp = torch.linspace(0.0, 1.0, border + 2, device=device, dtype=dtype)[1:-1]
    if x > 0:
        mask[:, :, :border, :] *= ramp.view(1, 1, border, 1)
    if y > 0:
        mask[:, :border, :, :] *= ramp.view(1, border, 1, 1)
    if x + tile_w < width:
        mask[:, :, tile_w - border:tile_w, :] *= ramp.flip(0).view(1, 1, border, 1)
    if y + tile_h < height:
        mask[:, tile_h - border:tile_h, :, :] *= ramp.flip(0).view(1, border, 1, 1)
    return mask.clamp(0.0, 1.0)


def _stabilize_pid_tile_from_reference(pid_tile, reference_tile, x, y, width, height, border):
    if reference_tile is None:
        return pid_tile
    if not torch.is_tensor(pid_tile) or not torch.is_tensor(reference_tile):
        return pid_tile
    if pid_tile.ndim != 4 or reference_tile.ndim != 4:
        return pid_tile

    tile_h = min(int(pid_tile.shape[1]), int(reference_tile.shape[1]))
    tile_w = min(int(pid_tile.shape[2]), int(reference_tile.shape[2]))
    if tile_w <= 0 or tile_h <= 0:
        return pid_tile

    pid_crop = pid_tile[:, :tile_h, :tile_w, :]
    ref_crop = reference_tile[:, :tile_h, :tile_w, :].to(device=pid_crop.device, dtype=pid_crop.dtype)
    mask = _pid_reference_stabilization_mask(
        tile_w,
        tile_h,
        int(x),
        int(y),
        int(width),
        int(height),
        int(border),
        pid_crop.device,
        pid_crop.dtype,
    )
    blended = ref_crop * (1.0 - mask) + pid_crop * mask
    if tile_h == int(pid_tile.shape[1]) and tile_w == int(pid_tile.shape[2]):
        return blended

    stabilized = pid_tile.clone()
    stabilized[:, :tile_h, :tile_w, :] = blended
    return stabilized


def _pid_rebuild_device(pid_tiles):
    for tile in pid_tiles or []:
        if torch.is_tensor(tile) and tile.device.type == "cuda":
            return tile.device
    if pid_gpu_final_rebuild_enabled() and torch.cuda.is_available():
        return mm.get_torch_device()
    for tile in pid_tiles or []:
        if torch.is_tensor(tile):
            return tile.device
    return mm.intermediate_device()


def gpu_pid_tile_rebuild(pid_tiles, grid_specs, width, height, stitch_feather=PID_DEFAULT_STITCH_FEATHER * PID_SCALE, label="TBG PID"):
    if not pid_tiles:
        raise ValueError("TBG PID tiled upscale did not produce any tiles.")

    first_tile = next((tile for tile in pid_tiles if torch.is_tensor(tile)), None)
    if first_tile is None:
        raise ValueError("TBG PID tiled upscale did not produce any tensor tiles.")

    device = _pid_rebuild_device(pid_tiles)
    dtype = first_tile.dtype
    channels = first_tile.shape[-1]
    result = torch.zeros((1, height, width, channels), dtype=dtype, device=device)
    weight = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
    feather = int(max(0, stitch_feather))

    for tile, spec in zip(pid_tiles, grid_specs):
        if tile is None:
            continue
        if not torch.is_tensor(tile):
            raise ValueError(f"TBG PID tiled upscale expected tensor tile for rebuild, got {type(tile).__name__}.")
        if tile.ndim == 3:
            tile = tile.unsqueeze(0)
        _, _, _, x, y, tile_w, tile_h = spec[:7]
        x, y, tile_w, tile_h = int(x), int(y), int(tile_w), int(tile_h)
        crop = tile[:, :tile_h, :tile_w, :].to(device=device, dtype=dtype, non_blocking=True)
        blend = _tile_ownership_feather_mask(tile_w, tile_h, x, y, width, height, grid_specs, feather, device, dtype)
        result[:, y:y + tile_h, x:x + tile_w, :] += crop * blend
        weight[:, y:y + tile_h, x:x + tile_w, :] += blend

    rebuilt = result / weight.clamp_min(1e-4)
    print(f"[{label}] GPU/local feathered rebuild device={device} feather={feather}px canvas={width}x{height}")
    return rebuilt


def _local_rebuild(pid_tiles, grid_specs, width, height, stitch_feather=PID_DEFAULT_STITCH_FEATHER * PID_SCALE):
    return gpu_pid_tile_rebuild(pid_tiles, grid_specs, width, height, stitch_feather=stitch_feather, label="TBG PID")


def _cpu_image(image):
    if torch.is_tensor(image):
        return image.detach().to("cpu", copy=True).contiguous()
    return image


def _cpu_images(images):
    return [_cpu_image(image) for image in images]


def run_pid_tiled_upscale(
    image,
    upscale_model_name,
    prompt_text="",
    seed=0,
    rebuild_fn=None,
    clip=None,
    source_vae=None,
    overlap=PID_DEFAULT_OVERLAP,
    stitch_feather=PID_DEFAULT_STITCH_FEATHER * PID_SCALE,
    degrade_sigma=0.1,
    color_match_fn=None,
    color_match_method=PID_DEFAULT_COLOR_MATCH,
    prompt_fn=None,
    sampler=None,
    sampler_name=PID_DEFAULT_SAMPLER,
    scheduler="simple",
    steps=4,
    cfg=1.0,
    denoise=1.0,
    include_pid_tiles=False,
    source_latent=None,
    pid_model=None,
    pid_model_info=None,
    pid_model_type=None,
    allow_hf_download=True,
):
    has_image = image is not None and torch.is_tensor(image)
    if image is not None and not torch.is_tensor(image):
        raise ValueError("TBG PID tiled upscale IMAGE input must be a tensor when connected.")
    if not has_image and source_latent is None:
        raise ValueError("TBG PID tiled upscale requires either an IMAGE input or a LATENT input.")

    if pid_model_type:
        upscale_model_name = pid_model_name_for_model_type(pid_model_type)
    elif pid_model_info is not None and getattr(pid_model_info, "name", None):
        upscale_model_name = str(pid_model_info.name)
    upscale_model_name = _pid_upscale_model_for_latent(upscale_model_name, source_latent)
    spec = PID_UPSCALE_SPECS.get(upscale_model_name)
    if spec is None:
        raise ValueError(f"Unknown TBG PID upscale model: {upscale_model_name}")
    pid_input_size = int(spec.input_size or PID_INPUT_SIZE)
    pid_output_size = int(spec.output_size or PID_OUTPUT_SIZE)
    pid_scale = pid_output_size / float(pid_input_size)

    latent_source_size = _source_size_from_latent(source_latent) if source_latent is not None else None
    if has_image:
        if image.ndim != 4:
            raise ValueError(f"TBG PID tiled upscale requires a batched IMAGE tensor, got shape {tuple(image.shape)}")
        if image.shape[0] != 1:
            raise ValueError(f"TBG PID tiled upscale requires exactly one input image, got batch {image.shape[0]}")
        original_height, original_width = int(image.shape[1]), int(image.shape[2])
        if latent_source_size is not None:
            latent_width, latent_height = latent_source_size
            if original_width != latent_width or original_height != latent_height:
                print(
                    "[TBG PID] image/latent source size mismatch; "
                    f"resizing image {original_width}x{original_height} to latent source {latent_width}x{latent_height}"
                )
                image = nodes.ImageScale().upscale(image, "lanczos", latent_width, latent_height, False)[0]
                original_width, original_height = latent_width, latent_height
        normalized, pre_scale = _resize_for_pid_minimum(image, input_size=pid_input_size)
        norm_height, norm_width = int(normalized.shape[1]), int(normalized.shape[2])
    else:
        original_width, original_height = latent_source_size
        shortest = min(original_width, original_height)
        pre_scale = pid_input_size / float(shortest) if shortest < pid_input_size else 1.0
        norm_width = max(pid_input_size, int(round(original_width * pre_scale)))
        norm_height = max(pid_input_size, int(round(original_height * pre_scale)))
        normalized = None

    if source_latent is not None:
        _latent_tile_from_image_region(source_latent, original_width, original_height, 0, 0, original_width, original_height)

    target_width = int(round(norm_width * pid_scale))
    target_height = int(round(norm_height * pid_scale))
    overlap = PID_DEFAULT_OVERLAP
    stitch_feather = int(round(PID_DEFAULT_STITCH_FEATHER * pid_scale))
    tile_specs = _pid_tile_specs(norm_width, norm_height, overlap=overlap, input_size=pid_input_size)
    print(
        "[TBG PID] fusion defaults "
        f"reference_margin={PID_DEFAULT_OVERLAP}px, "
        f"fusion_blur={PID_DEFAULT_STITCH_BLUR}px, "
        f"feather_mask={PID_DEFAULT_STITCH_FEATHER}px, "
        f"color_match='{PID_DEFAULT_COLOR_MATCH}'"
    )

    runtime = _load_pid_runtime(
        upscale_model_name,
        clip=clip,
        source_vae=source_vae,
        sampler_name=sampler_name,
        scheduler=scheduler,
        steps=steps,
        denoise=denoise,
        sampler=sampler,
        load_source_vae=source_latent is None,
        pid_model=pid_model,
        allow_hf_download=allow_hf_download,
    )
    runtime.cfg = float(cfg)
    conditioning_cache = {}

    def get_conditioning(tile, index):
        tile_prompt = prompt_text or ""
        if tile is not None and prompt_fn is not None:
            try:
                generated = prompt_fn(tile, index, len(tile_specs))
                if generated:
                    tile_prompt = "\n".join(p for p in (prompt_text or "", generated) if p)
            except Exception as exc:
                print(f"[TBG PID VLM] Prompt generation failed on tile {index}, using base prompt: {exc}")
        if tile_prompt not in conditioning_cache:
            positive = nodes.CLIPTextEncode().encode(runtime.clip, tile_prompt)[0]
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
            conditioning_cache[tile_prompt] = (positive, negative)
        return conditioning_cache[tile_prompt]

    pid_tiles = []
    output_grid_specs = []
    stabilized_tiles = 0
    allow_region_color_lock = color_match_fn is not None and len(tile_specs) == 1
    if color_match_fn is not None and not allow_region_color_lock:
        print(
            "[TBG PID] color correction deferred until stitched ETUR tile; "
            f"internal_regions={len(tile_specs)}"
        )
    for index, (row, col, x, y, tile_w, tile_h) in enumerate(tile_specs):
        tile_start = time.perf_counter()
        tile = None if normalized is None else normalized[:, y:y + tile_h, x:x + tile_w, :]
        pid_input = None if tile is None else _pad_tile_to_pid_size(tile, input_size=pid_input_size)
        positive, negative = get_conditioning(pid_input, index)
        pid_source_latent = _latent_tile_from_image_region(
            source_latent,
            original_width,
            original_height,
            x,
            y,
            tile_w,
            tile_h,
            scale=pre_scale,
            input_size=pid_input_size,
        )
        pid_tile = upscale_pid_tile_1024(
            pid_input,
            runtime,
            positive,
            negative,
            int(seed or 0) + index,
            degrade_sigma,
            source_latent=pid_source_latent,
        )
        out_w = int(round(tile_w * pid_scale))
        out_h = int(round(tile_h * pid_scale))
        pid_tile = pid_tile[:, :out_h, :out_w, :]
        reference_tile = None
        if tile is not None:
            reference_tile = nodes.ImageScale().upscale(tile, "bilinear", out_w, out_h, False)[0]
        if tile is not None and allow_region_color_lock and color_match_method and color_match_method != "none":
            try:
                pid_tile = color_match_fn(reference_tile, pid_tile, color_match_method)
            except Exception as exc:
                print(f"[TBG PID] Tile color correction failed on tile {index}, using raw PID tile: {exc}")
        if reference_tile is not None:
            try:
                pid_tile = _stabilize_pid_tile_from_reference(
                    pid_tile,
                    reference_tile,
                    int(round(x * pid_scale)),
                    int(round(y * pid_scale)),
                    target_width,
                    target_height,
                    int(round(PID_DEFAULT_OVERLAP * pid_scale)),
                )
                stabilized_tiles += 1
            except Exception as exc:
                print(f"[TBG PID] Tile reference border stabilization failed on tile {index}, keeping color-matched tile: {exc}")
        pid_tiles.append(pid_tile)
        output_grid_specs.append((row, col, index, int(round(x * pid_scale)), int(round(y * pid_scale)), out_w, out_h))
        print(f"[TBG PID] tile {index + 1}/{len(tile_specs)} PID sample+decode complete ({time.perf_counter() - tile_start:.2f}s)")

    if stabilized_tiles:
        print(f"[TBG PID] reference border stabilization applied to {stabilized_tiles}/{len(tile_specs)} tiles")
    elif normalized is None:
        print("[TBG PID] reference border stabilization skipped for latent-only input")

    reference_image = None
    if normalized is not None:
        reference_image = nodes.ImageScale().upscale(normalized, "bilinear", target_width, target_height, False)[0]
    rebuilt = None
    if rebuild_fn is not None and reference_image is not None and not pid_gpu_final_rebuild_enabled():
        try:
            rebuild_start = time.perf_counter()
            rebuilt = rebuild_fn(_cpu_images(pid_tiles), output_grid_specs, _cpu_image(reference_image), target_width, target_height)
            print(f"[TBG PID] worker pyramid rebuild complete ({time.perf_counter() - rebuild_start:.2f}s)")
        except Exception as exc:
            print(f"[TBG PID] Worker gpupyramid pre-upscale stitch failed, using local rebuild: {exc}")
    elif rebuild_fn is not None and reference_image is not None:
        print("[TBG PID] GPU final rebuild enabled; skipping CPU worker pyramid rebuild callback.")

    if rebuilt is None:
        rebuild_start = time.perf_counter()
        rebuilt = _local_rebuild(pid_tiles, output_grid_specs, target_width, target_height, stitch_feather=stitch_feather)
        print(f"[TBG PID] local feathered rebuild complete feather={stitch_feather}px ({time.perf_counter() - rebuild_start:.2f}s)")

    if torch.is_tensor(rebuilt) and rebuilt.ndim == 3:
        rebuilt = rebuilt.unsqueeze(0)
    if torch.is_tensor(rebuilt):
        rebuilt = rebuilt[:, :target_height, :target_width, :]

    cache_start = time.perf_counter()
    mm.soft_empty_cache()
    print(f"[TBG PID] soft cache cleanup complete ({time.perf_counter() - cache_start:.2f}s)")
    meta = {
        "pre_scale": pre_scale,
        "source_width": original_width,
        "source_height": original_height,
        "effective_scale_x": target_width / float(original_width),
        "effective_scale_y": target_height / float(original_height),
        "normalized_width": norm_width,
        "normalized_height": norm_height,
        "target_width": target_width,
        "target_height": target_height,
        "reference_border_stabilized_tiles": stabilized_tiles,
        "tile_count": len(pid_tiles),
        "grid_specs": output_grid_specs,
        "overlap": overlap,
        "stitch_feather": stitch_feather,
        "degrade_sigma": float(degrade_sigma),
        "denoise": float(max(0.01, min(1.0, float(denoise)))),
        "color_match_method": color_match_method,
    }
    if include_pid_tiles:
        copy_start = time.perf_counter()
        meta["pid_tiles"] = _cpu_images(pid_tiles)
        print(f"[TBG PID] copied PID preview tiles to CPU ({time.perf_counter() - copy_start:.2f}s)")
    return rebuilt, meta
