import math

import torch


DEFAULT_CONFIG = {
    "enabled": True,
    "correction_start_sigma": 0.6341,
    "transition_width": 0.08,
    "mask_gamma": 1.0,
    "post_composite_preserve": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
}


def latent_config(latent_image):
    if not isinstance(latent_image, dict):
        return None
    config = latent_image.get("_flux2_differential")
    if not isinstance(config, dict) or not config.get("enabled"):
        return None
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return merged


def latent_mask(latent_image):
    if latent_config(latent_image) is not None:
        mask = latent_image.get("_flux2_inpaint_mask")
        if mask is not None:
            return mask
    return latent_image.get("noise_mask")


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    for i, t in enumerate(timesteps):
        tv = t.item()
        if 0 < tv < 1:
            timesteps[i] = _time_shift(mu, 1.0, tv)
    return timesteps.tolist()


def _smooth_active(mask, progress, transition_width):
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


def _model_patch_size(self_x0):
    model = getattr(getattr(self_x0, "inner_model", None), "inner_model", None)
    diffusion_model = getattr(model, "diffusion_model", None)
    return int(getattr(diffusion_model, "patch_size", 2) or 2)


def _gate_info(owner, self_x0, mask, config):
    state = getattr(owner, "_flux2_diff_state", None)
    if state is None:
        state = {}
        setattr(owner, "_flux2_diff_state", state)
    cached = state.get("gate_info")
    if cached is not None:
        return cached

    actual_sigmas = getattr(self_x0, "sigmas", None)
    if torch.is_tensor(actual_sigmas) and len(actual_sigmas) >= 2:
        schedule = actual_sigmas.detach().float().cpu().tolist()
        steps = max(len(schedule) - 1, 1)
        schedule_source = "effective_sampler_sigmas"
    else:
        steps = max(int(getattr(owner, "steps", 1)), 1)
        patch_size = _model_patch_size(self_x0)
        h_tokens = (mask.shape[-2] + patch_size // 2) // patch_size
        w_tokens = (mask.shape[-1] + patch_size // 2) // patch_size
        schedule = get_schedule(
            steps,
            h_tokens * w_tokens,
            base_shift=float(config.get("base_shift", 0.5)),
            max_shift=float(config.get("max_shift", 1.15)),
        )
        schedule_source = "flux2_time_schedule"
    start_sigma = float(config.get("correction_start_sigma", 0.6341))
    gate_index = next((i for i, t in enumerate(schedule) if t <= start_sigma), steps)
    gate_index = max(1, min(gate_index, steps))
    cached = {
        "schedule": schedule,
        "gate_index": gate_index,
        "announced": False,
        "schedule_source": schedule_source,
    }
    state["gate_info"] = cached
    return cached


def begin_step(owner, self_x0, x, sigma, denoise_mask, model_options):
    config = getattr(owner, "_flux2_diff_config", None)
    if config is None or denoise_mask is None:
        return False, x, denoise_mask, None

    mask = denoise_mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
    gamma = float(config.get("mask_gamma", 1.0))
    if gamma != 1.0:
        mask = mask.pow(gamma)
    mask = torch.where(denoise_mask.to(device=x.device, dtype=x.dtype) >= 0.95, torch.ones_like(mask), mask)
    mask = torch.where(denoise_mask.to(device=x.device, dtype=x.dtype) < 0.05, torch.zeros_like(mask), mask)

    gate = _gate_info(owner, self_x0, mask, config)
    if not gate.get("announced", False):
        print(
            "[TBG Flux2 Differential] hook gate "
            f"gate_step={gate['gate_index']}/{max(len(gate['schedule']) - 1, 1)} "
            f"effective_sigma_count={len(gate['schedule'])} "
            f"first_sigma={float(gate['schedule'][0]):.6f} "
            f"correction_start_sigma={float(config.get('correction_start_sigma', 0.6341)):.4f} "
            f"schedule_source={gate.get('schedule_source')} "
            "noise_mask_private=True"
        )
        gate["announced"] = True
    step_index = int(getattr(owner, "total_step_count", 0))
    if step_index < gate["gate_index"]:
        return True, x, mask, None

    original_noised = self_x0.inner_model.inner_model.scale_latent_inpaint(
        x=x, sigma=sigma, noise=self_x0.noise, latent_image=self_x0.latent_image
    )
    x = x * mask + original_noised * (1.0 - mask)

    schedule = gate["schedule"]
    progress = schedule[min(step_index, len(schedule) - 1)]
    active = _smooth_active(mask, progress, float(config.get("transition_width", 0.08)))
    state = getattr(owner, "_flux2_diff_state", {})
    state["composed"] = True
    return True, x, mask, active


def finish_step(owner, self_x0, out, active_mask):
    config = getattr(owner, "_flux2_diff_config", None)
    if config is None or active_mask is None or not bool(config.get("post_composite_preserve", True)):
        return out
    return out * active_mask + self_x0.latent_image.to(device=out.device, dtype=out.dtype) * (1.0 - active_mask)


def reset_state(owner):
    setattr(owner, "_flux2_diff_config", None)
    setattr(owner, "_flux2_diff_state", {})
