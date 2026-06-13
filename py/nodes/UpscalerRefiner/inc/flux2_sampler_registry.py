import torch
import torch.nn.functional as F

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling


TBG_FLUX2_SAMPLER_NAME = "TBG Flux2 Sampler"


def _insert_after(items, value, after):
    if value in items:
        return
    try:
        index = items.index(after) + 1
    except ValueError:
        index = len(items)
    items.insert(index, value)


def _mask_bchw(mask, height, width, device, dtype):
    mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _smooth_active(mask, progress, transition_width=0.08):
    threshold = torch.as_tensor(progress, device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)
    width = torch.as_tensor(transition_width, device=mask.device, dtype=mask.dtype).clamp_min(1e-6)
    active = ((mask - (threshold - width)) / width).clamp(0.0, 1.0)
    active = active * active * (3.0 - 2.0 * active)
    active = torch.where(mask >= 0.95, torch.ones_like(active), active)
    active = torch.where(mask < 0.05, torch.zeros_like(active), active)
    return active.clamp(0.0, 1.0)


def _sigma_progress(sigma, sigmas):
    sigma_max = float(sigmas[0])
    sigma_min = float(sigmas[-1])
    denom = max(sigma_max - sigma_min, 1e-6)
    return max(0.0, min(1.0, (float(sigma) - sigma_min) / denom))


def _gate_index(sigmas, correction_start_sigma=0.6341):
    steps = max(len(sigmas) - 1, 1)
    for i, sigma in enumerate(sigmas[:-1]):
        if _sigma_progress(sigma, sigmas) <= float(correction_start_sigma):
            return max(1, min(i, steps))
    return steps


@torch.no_grad()
def sample_tbg_flux2_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    original_extra_args = dict(extra_args or {})
    denoise_mask = original_extra_args.get("denoise_mask")
    if denoise_mask is None or getattr(model, "latent_image", None) is None:
        return k_diffusion_sampling.sample_euler(
            model,
            x,
            sigmas,
            extra_args=original_extra_args,
            callback=callback,
            disable=disable,
        )
    extra_args = dict(original_extra_args)
    extra_args.pop("denoise_mask", None)

    mask = _mask_bchw(denoise_mask, x.shape[-2], x.shape[-1], x.device, x.dtype)
    latent_image = model.latent_image.to(device=x.device, dtype=x.dtype)
    noise = getattr(model, "noise", None)
    if noise is None:
        noise = torch.zeros_like(x)
    else:
        noise = noise.to(device=x.device, dtype=x.dtype)

    gate = _gate_index(sigmas)
    print(
        "[TBG Flux2 Sampler] latent sampler active "
        f"steps={len(sigmas) - 1} gate_step={gate}/{len(sigmas) - 1} "
        "noise_mask_private=True"
    )

    s_in = x.new_ones([x.shape[0]])
    for i in k_diffusion_sampling.trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_vec = sigma * s_in

        if i >= gate:
            active = _smooth_active(mask, _sigma_progress(sigma, sigmas))
            original_noised = model.inner_model.inner_model.scale_latent_inpaint(
                x=x,
                sigma=sigma_vec,
                noise=noise,
                latent_image=latent_image,
            )
            x = x * active + original_noised * (1.0 - active)

        denoised = model(x, sigma_vec, denoise_mask=None, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        d = k_diffusion_sampling.to_d(x, sigma, denoised)
        x = x + d * (sigma_next - sigma)

        if i >= gate:
            active_next = _smooth_active(mask, _sigma_progress(sigma_next, sigmas))
            original_noised_next = model.inner_model.inner_model.scale_latent_inpaint(
                x=x,
                sigma=sigma_next * s_in,
                noise=noise,
                latent_image=latent_image,
            )
            x = x * active_next + original_noised_next * (1.0 - active_next)

    return x


def register_tbg_flux2_sampler():
    setattr(k_diffusion_sampling, f"sample_{TBG_FLUX2_SAMPLER_NAME}", sample_tbg_flux2_sampler)
    _insert_after(comfy.samplers.KSAMPLER_NAMES, TBG_FLUX2_SAMPLER_NAME, "euler")
    _insert_after(comfy.samplers.SAMPLER_NAMES, TBG_FLUX2_SAMPLER_NAME, "euler")
    _insert_after(comfy.samplers.KSampler.SAMPLERS, TBG_FLUX2_SAMPLER_NAME, "euler")
    print(f"[TBG Flux2 Sampler] registered sampler: {TBG_FLUX2_SAMPLER_NAME}")
