"""Register NVIDIA PiD SDE samplers from inside TBG ETUR."""

import torch

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling


PID_SDE_NAME = "pid_sde"
PID_CREATIVE_SDE_NAME = "pid_creative_sde"
PID_DISTILLED_TIMESTEPS = [0.999, 0.866, 0.634, 0.342, 0.0]


def _pid_timestep_strength(sigmas):
    if sigmas is None or len(sigmas) == 0:
        return 1.0
    try:
        first_sigma = float(sigmas[0])
    except Exception:
        return 1.0
    return max(0.0, min(1.0, first_sigma / PID_DISTILLED_TIMESTEPS[0]))


def _pid_timestep_list(x, sigmas, extra_semantic_step=False):
    full_t = x.new_tensor(PID_DISTILLED_TIMESTEPS)
    steps = len(sigmas) - 1
    if steps <= 0:
        return None

    if steps == 4:
        t_list = full_t
    else:
        indices = torch.linspace(0, len(full_t) - 1, steps + 1, device=x.device).round().long()
        t_list = full_t[indices]

    if extra_semantic_step and t_list.numel() >= 3:
        early_mid = (t_list[0:1] + t_list[1:2]) * 0.5
        t_list = torch.cat([t_list[0:1], early_mid, t_list[1:]])

    strength = _pid_timestep_strength(sigmas)
    if strength < 1.0:
        t_list = t_list * strength
        t_list[-1] = 0.0
    return t_list


def _noise_sampler(x, seed, noise_sampler):
    return (
        k_diffusion_sampling.default_noise_sampler(x, seed=seed)
        if noise_sampler is None
        else noise_sampler
    )


@torch.no_grad()
def sample_pid_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None):
    """NVIDIA PiD distilled SDE sampler."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _noise_sampler(x, seed, noise_sampler)
    s_in = x.new_ones([x.shape[0]])
    t_list = _pid_timestep_list(x, sigmas)
    if t_list is None:
        return x
    steps = t_list.numel() - 1

    for i in k_diffusion_sampling.trange(int(steps), disable=disable):
        t_cur = t_list[i]
        t_next = t_list[i + 1]
        denoised = model(x, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": t_cur, "sigma_hat": t_cur, "denoised": denoised})

        if t_next <= 0:
            x = denoised
        else:
            x = (1.0 - t_next) * denoised + t_next * noise_sampler(t_cur, t_next) * s_noise

    return x


@torch.no_grad()
def sample_pid_creative_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    creative_strength=0.35,
    early_noise_boost=0.15,
    early_cfg_boost=0.0,
    think_steps=1,
    extra_semantic_step=True,
):
    """Experimental creative PiD sampler."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = _noise_sampler(x, seed, noise_sampler)
    s_in = x.new_ones([x.shape[0]])
    creative_strength = float(max(0.0, min(1.0, creative_strength)))
    early_noise_boost = float(max(0.0, min(1.0, early_noise_boost)))
    think_steps = int(max(0, min(2, think_steps)))

    if early_cfg_boost:
        print("[TBG PiD SDE] early_cfg_boost is reserved for a future guider-side patch; sampler-side run continues without CFG mutation.")

    print(
        "[TBG PiD Creative SDE] active "
        f"creative_strength={creative_strength:.2f}, "
        f"early_noise_boost={early_noise_boost:.2f}, "
        f"think_steps={think_steps}, "
        f"extra_semantic_step={bool(extra_semantic_step)}"
    )

    t_list = _pid_timestep_list(x, sigmas, extra_semantic_step=extra_semantic_step and creative_strength > 0.0)
    if t_list is None:
        return x
    steps = int(t_list.numel() - 1)

    for i in k_diffusion_sampling.trange(steps, disable=disable):
        t_cur = t_list[i]
        t_next = t_list[i + 1]

        if i == 0 and creative_strength > 0.0 and think_steps > 0:
            for _ in range(think_steps):
                denoised = model(x, t_cur * s_in, **extra_args)
                eps = noise_sampler(t_cur, t_cur) * s_noise
                semantic_x = (1.0 - t_cur) * denoised + t_cur * eps
                x = x.lerp(semantic_x, creative_strength)

        denoised = model(x, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": min(i, len(sigmas) - 2), "sigma": t_cur, "sigma_hat": t_cur, "denoised": denoised})

        if t_next <= 0:
            x = denoised
        else:
            early = i < 2
            boost = early_noise_boost * creative_strength if early else 0.0
            t_renoise = (t_next + boost * (t_cur - t_next)).clamp(max=t_cur, min=t_next)
            x = (1.0 - t_renoise) * denoised + t_renoise * noise_sampler(t_cur, t_next) * s_noise

    return x


def _insert_after(items, value, after):
    if value in items:
        return
    try:
        index = items.index(after) + 1
    except ValueError:
        index = len(items)
    items.insert(index, value)


def register_tbg_pid_sde_samplers():
    setattr(k_diffusion_sampling, f"sample_{PID_SDE_NAME}", sample_pid_sde)
    setattr(k_diffusion_sampling, f"sample_{PID_CREATIVE_SDE_NAME}", sample_pid_creative_sde)
    _insert_after(comfy.samplers.KSAMPLER_NAMES, PID_SDE_NAME, "euler")
    _insert_after(comfy.samplers.KSAMPLER_NAMES, PID_CREATIVE_SDE_NAME, PID_SDE_NAME)
    _insert_after(comfy.samplers.SAMPLER_NAMES, PID_SDE_NAME, "euler")
    _insert_after(comfy.samplers.SAMPLER_NAMES, PID_CREATIVE_SDE_NAME, PID_SDE_NAME)
    _insert_after(comfy.samplers.KSampler.SAMPLERS, PID_SDE_NAME, "euler")
    _insert_after(comfy.samplers.KSampler.SAMPLERS, PID_CREATIVE_SDE_NAME, PID_SDE_NAME)
    print(f"[TBG PiD SDE] registered samplers: {PID_SDE_NAME}, {PID_CREATIVE_SDE_NAME}")


class TBG_PiD_Creative_SDE_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creative_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "early_noise_boost": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "early_cfg_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "think_steps": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "extra_semantic_step": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "TBG/Sampler"

    def build(self, creative_strength, early_noise_boost, early_cfg_boost, think_steps, extra_semantic_step):
        sampler = comfy.samplers.KSAMPLER(
            sample_pid_creative_sde,
            {
                "creative_strength": creative_strength,
                "early_noise_boost": early_noise_boost,
                "early_cfg_boost": early_cfg_boost,
                "think_steps": think_steps,
                "extra_semantic_step": extra_semantic_step,
            },
        )
        return (sampler,)
