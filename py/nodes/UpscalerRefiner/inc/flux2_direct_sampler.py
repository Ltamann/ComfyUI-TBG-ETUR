import math

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import latent_preview


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    for i, t in enumerate(timesteps):
        tv = t.item()
        if 0 < tv < 1:
            timesteps[i] = _time_shift(mu, 1.0, tv)
    return timesteps.tolist()


def _mask_bchw(mask, height, width):
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] != 1:
        mask = mask[:, :1]
    mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _active_mask(mask, progress, transition_width):
    threshold = torch.as_tensor(progress, device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)
    if transition_width <= 0.0:
        active = (mask >= threshold).to(mask.dtype)
    else:
        width = torch.as_tensor(transition_width, device=mask.device, dtype=mask.dtype).clamp_min(1e-6)
        active = ((mask - (threshold - width)) / width).clamp(0.0, 1.0)
        active = active * active * (3.0 - 2.0 * active)
    active = torch.where(mask >= 0.95, torch.ones_like(active), active)
    active = torch.where(mask < 0.05, torch.zeros_like(active), active)
    return active.clamp(0.0, 1.0)


def _conditioning_tensor(conditioning, batch, device, dtype):
    cond = conditioning[0][0].to(device=device, dtype=dtype)
    if cond.shape[0] != batch:
        cond = cond[:1].expand(batch, -1, -1)
    return cond


def _reference_latents(conditioning, device, dtype):
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        return None
    meta = conditioning[0][1] if len(conditioning[0]) > 1 else {}
    if not isinstance(meta, dict):
        return None
    for key in ("ref_latents", "reference_latents", "concat_latent_image"):
        if key not in meta:
            continue
        value = meta[key]
        if isinstance(value, torch.Tensor):
            return [value.to(device=device, dtype=dtype)]
        if isinstance(value, (list, tuple)):
            return [v.to(device=device, dtype=dtype) for v in value if isinstance(v, torch.Tensor)]
    return None


def _aligned_pixels_and_mask(pixels, mask, vae):
    ratio = int(vae.spacial_compression_encode())
    x_size = pixels.shape[1] // ratio * ratio
    y_size = pixels.shape[2] // ratio * ratio
    x_offset = (pixels.shape[1] % ratio) // 2
    y_offset = (pixels.shape[2] % ratio) // 2
    pixel_mask = _mask_bchw(mask, pixels.shape[1], pixels.shape[2])
    if x_size != pixels.shape[1] or y_size != pixels.shape[2]:
        pixels = pixels[:, x_offset:x_offset + x_size, y_offset:y_offset + y_size, :]
        pixel_mask = pixel_mask[:, :, x_offset:x_offset + x_size, y_offset:y_offset + y_size]
    return pixels, pixel_mask


def _gray_filled_pixels(pixels, denoise_mask):
    binary_fill = (denoise_mask >= 0.5).to(pixels.dtype)
    keep = (1.0 - binary_fill).squeeze(1)
    inpaint_pixels = pixels.clone()
    for channel in range(3):
        inpaint_pixels[..., channel] -= 0.5
        inpaint_pixels[..., channel] *= keep
        inpaint_pixels[..., channel] += 0.5
    return inpaint_pixels


def _vae_encode_samples(vae, pixels):
    encoded = vae.encode(pixels)
    if isinstance(encoded, dict):
        encoded = encoded["samples"]
    return encoded.float()


def sample_flux2_direct(
    model,
    positive,
    negative,
    pixels,
    vae,
    mask,
    steps,
    seed,
    cfg,
    denoise=1.0,
    base_shift=0.5,
    max_shift=1.15,
    transition_width=0.08,
    mask_gamma=1.0,
    invert_mask=False,
    correction_start_sigma=0.6341,
    post_composite_preserve=True,
    sigmas=None,
    denoise_method="default",
    disable_pbar=None,
):
    pixels, denoise_mask = _aligned_pixels_and_mask(pixels, mask, vae)
    if invert_mask:
        denoise_mask = 1.0 - denoise_mask

    inpaint_pixels = _gray_filled_pixels(pixels, denoise_mask)
    original_latent = _vae_encode_samples(vae, pixels)
    inpaint_latent = _vae_encode_samples(vae, inpaint_pixels)

    comfy.model_management.load_models_gpu([model])
    device = comfy.model_management.get_torch_device()
    diffusion_model = model.model.diffusion_model
    model_dtype = diffusion_model.dtype

    # The gray-filled latent is only a safe inpaint cue when Flux2 starts from
    # full noise. For lower denoise it becomes visible as a gray, low-contrast
    # sampler output, so the lower-denoise experiment must start from the
    # normal image latent and delay all mask/original correction until the gate.
    latent = inpaint_latent if denoise >= 1.0 else original_latent
    batch, _, height, width = latent.shape
    patch_size = int(getattr(diffusion_model, "patch_size", 2) or 2)
    h_tokens = (height + patch_size // 2) // patch_size
    w_tokens = (width + patch_size // 2) // patch_size
    if sigmas is not None:
        if torch.is_tensor(sigmas):
            schedule = [float(s) for s in sigmas.detach().float().cpu().tolist()]
        else:
            schedule = [float(s) for s in sigmas]
        if len(schedule) < 2:
            raise ValueError(f"TBG Flux2 direct sampler received invalid delivered sigmas: {len(schedule)} values")
        expanded_steps = None
        schedule_source = f"delivered_sigmas/{denoise_method}"
    else:
        expanded_steps = None
        schedule = _get_schedule(steps, h_tokens * w_tokens, base_shift=base_shift, max_shift=max_shift)
        schedule_source = "fallback_full_flux2_schedule_no_delivered_sigmas"

    total_steps = len(schedule) - 1
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

    if denoise < 1.0:
        t_start = schedule[0]
        x = (1.0 - t_start) * latent + t_start * noise
    else:
        x = noise

    x = x.to(device=device, dtype=model_dtype)
    original_latent = original_latent.to(device=device, dtype=model_dtype)
    noise = noise.to(device=device, dtype=model_dtype)

    latent_mask = _mask_bchw(denoise_mask, height, width).to(device=device, dtype=model_dtype)
    if mask_gamma != 1.0:
        latent_mask = latent_mask.pow(float(mask_gamma))
    latent_mask = torch.where(latent_mask >= 0.95, torch.ones_like(latent_mask), latent_mask)
    latent_mask = torch.where(latent_mask < 0.05, torch.zeros_like(latent_mask), latent_mask)

    cond = _conditioning_tensor(positive, batch, device, model_dtype)
    use_cfg = negative is not None and float(cfg) > 1.0
    neg_cond = _conditioning_tensor(negative, batch, device, model_dtype) if use_cfg else None
    ref_latents = _reference_latents(positive, device, model_dtype)

    has_guidance_embed = getattr(diffusion_model.params, "guidance_embed", False)
    guidance_vec = torch.full((batch,), 1.0, device=device, dtype=model_dtype) if has_guidance_embed else None
    transformer_options = model.model_options.get("transformer_options", {}).copy()

    gate_index = next((i for i, t in enumerate(schedule) if t <= float(correction_start_sigma)), total_steps)
    gate_index = max(1, min(gate_index, total_steps))
    print(
        "[TBG Flux2 Direct Differential] "
        f"denoise={float(denoise):.4f} steps={total_steps} expanded_steps={expanded_steps} "
        f"schedule_source={schedule_source} seed={seed} cfg={cfg} "
        f"first_sigma={float(schedule[0]):.6f} gate_step={gate_index}/{total_steps} "
        f"correction_start_sigma={float(correction_start_sigma):.4f} "
        f"start_latent={'inpaint_gray' if denoise >= 1.0 else 'original'} "
        f"refs={0 if ref_latents is None else len(ref_latents)}"
    )

    if disable_pbar is None:
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    pbar = comfy.utils.ProgressBar(total_steps)
    preview_callback = latent_preview.prepare_callback(model, total_steps)

    with torch.no_grad():
        for i in comfy.utils.model_trange(total_steps, disable=disable_pbar):
            comfy.model_management.throw_exception_if_processing_interrupted()
            t_curr = float(schedule[i])
            t_prev = float(schedule[i + 1])
            t_vec = torch.full((batch,), t_curr, device=device, dtype=model_dtype)

            if i >= gate_index:
                original_noised_curr = (1.0 - t_curr) * original_latent + t_curr * noise
                active_curr = _active_mask(latent_mask, t_curr, float(transition_width))
                x = x * active_curr + original_noised_curr * (1.0 - active_curr)

            transformer_options["sigmas"] = t_vec
            pred = diffusion_model.forward(
                x,
                t_vec,
                cond,
                y=None,
                guidance=guidance_vec,
                ref_latents=ref_latents,
                control=None,
                transformer_options=transformer_options,
            )

            if use_cfg:
                pred_uncond = diffusion_model.forward(
                    x,
                    t_vec,
                    neg_cond,
                    y=None,
                    guidance=guidance_vec,
                    ref_latents=ref_latents,
                    control=None,
                    transformer_options=transformer_options,
                )
                pred = pred_uncond + float(cfg) * (pred - pred_uncond)

            if preview_callback is not None:
                x0_est = (x - t_curr * pred) if t_curr > 1e-6 else x
                preview_callback(i, x0_est.detach().cpu().float(), x.detach().cpu().float(), total_steps)

            x = x + (t_prev - t_curr) * pred

            if i >= gate_index and post_composite_preserve:
                original_noised_prev = (1.0 - t_prev) * original_latent + t_prev * noise
                active_prev = _active_mask(latent_mask, t_prev, float(transition_width))
                x = x * active_prev + original_noised_prev * (1.0 - active_prev)

            pbar.update(1)

    return {"samples": x.detach().cpu().float()}
