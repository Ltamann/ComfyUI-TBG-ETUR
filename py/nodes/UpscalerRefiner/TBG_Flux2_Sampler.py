import torch
import torch.nn.functional as F
from comfy.samplers import KSamplerX0Inpaint


class TBGFlux2Sampler:
    TRANSITION_WIDTH = 0.08
    DESCRIPTION = (
        "Flux2 differential diffusion inpainting hook. Place this before any sampler. "
        "It stores the inpaint mask privately and wraps a SAMPLER so Euler, LCM, DPM++, "
        "RES/other custom samplers can delay mask correction until Flux2's sweet spot."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "inpaint_mask": ("MASK", {
                    "label": "Inpaint Mask",
                    "tooltip": "White edits, black preserves, gray values transition progressively over sampler steps.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": (
                        "Differential inpaint timing for this hook, not the downstream sampler's denoise. "
                        "The downstream sampler may use lower denoise; this value delays preservation "
                        "inside the wrapped sampler."
                    ),
                }),
                "correction_start_sigma": ("FLOAT", {
                    "default": 0.6341,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Flux2 sweet spot where mask correction may start. "
                        "The hook waits until both this sigma and the denoise timing allow correction."
                    ),
                }),
            }
            ,
            "optional": {
                "sampler": ("SAMPLER", {
                    "tooltip": (
                        "Optional but recommended. Connect any sampler here and use the wrapped sampler output "
                        "for full Flux2 sweet-spot latent correction with that sampler."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "SAMPLER")
    RETURN_NAMES = ("model", "latent", "sampler")
    FUNCTION = "hook"
    CATEGORY = "TBG/Sampler"

    def hook(self, model, latent, inpaint_mask, denoise=1.0, correction_start_sigma=0.6341, sampler=None):
        latent_out = latent.copy()
        mask = inpaint_mask.reshape((-1, 1, inpaint_mask.shape[-2], inpaint_mask.shape[-1])).float().clamp(0.0, 1.0)
        latent_out["_flux2_inpaint_mask"] = mask
        latent_out["_flux2_differential"] = {
            "enabled": True,
            "denoise": float(max(0.0, min(1.0, denoise))),
            "correction_start_sigma": float(correction_start_sigma),
            "transition_width": float(self.TRANSITION_WIDTH),
        }
        latent_out.pop("noise_mask", None)

        model_out = model.clone()
        model_out.model_options["tbg_flux2_inpaint_mask"] = mask
        model_out.model_options["tbg_flux2_differential"] = dict(latent_out["_flux2_differential"])
        sampler_out = Flux2DifferentialSamplerWrapper(
            sampler=sampler,
            denoise=denoise,
            correction_start_sigma=correction_start_sigma,
            transition_width=self.TRANSITION_WIDTH,
        ) if sampler is not None else sampler
        if sampler is None:
            print(
                "[TBG Flux2 Differential Standalone] no SAMPLER connected; "
                "mask stored privately but no delayed correction wrapper is active"
            )
        else:
            print(
                "[TBG Flux2 Differential Standalone] private mask active "
                f"denoise={float(denoise):.4f} correction_start_sigma={float(correction_start_sigma):.4f} "
                "latent_noise_mask_absent=True"
            )
        return (model_out, latent_out, sampler_out)


class Flux2DifferentialSamplerWrapper:
    def __init__(self, sampler, denoise, correction_start_sigma, transition_width):
        self.sampler = sampler
        self.denoise = float(max(0.0, min(1.0, denoise)))
        self.correction_start_sigma = float(correction_start_sigma)
        self.transition_width = float(transition_width)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        model_options = (extra_args or {}).get("model_options", {})
        private_mask = model_options.get("tbg_flux2_inpaint_mask")
        if denoise_mask is None and torch.is_tensor(private_mask):
            denoise_mask = private_mask
        if denoise_mask is None:
            return self.sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        original_call = KSamplerX0Inpaint.__call__
        state = {"composed": False}
        wrapper = self

        def patched_call(self_x0, x, sigma, denoise_mask_inner=None, model_options={}, seed=None, **kwargs):
            if denoise_mask_inner is None:
                denoise_mask_inner = kwargs.get("denoise_mask")
            if denoise_mask_inner is None:
                return self_x0.inner_model(x, sigma, model_options=model_options, seed=seed)
            mask = denoise_mask_inner.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            elif mask.ndim == 4 and mask.shape[1] != 1:
                mask = mask[:, :1]
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            step_sigmas = getattr(self_x0, "sigmas", sigmas).to(device=x.device, dtype=x.dtype)
            step_index = _current_step_index(sigma.to(x.device), step_sigmas)
            gate_index = _gate_index(step_sigmas, wrapper.denoise, wrapper.correction_start_sigma)
            if step_index == 0:
                print(
                    "[TBG Flux2 Differential Standalone] wrapped sampler gate "
                    f"denoise={wrapper.denoise:.4f} effective_sigma_count={len(step_sigmas)} "
                    f"first_sigma={float(step_sigmas[0]):.6f} gate_step={gate_index}/{max(len(step_sigmas) - 1, 1)} "
                    f"correction_start_sigma={wrapper.correction_start_sigma:.4f} "
                    "noise_mask_private=True"
                )

            if step_index < gate_index:
                return self_x0.inner_model(x, sigma, model_options=model_options, seed=seed)

            original_noised = self_x0.inner_model.inner_model.scale_latent_inpaint(
                x=x,
                sigma=sigma,
                noise=self_x0.noise,
                latent_image=self_x0.latent_image,
            )
            x = x * mask + original_noised * (1.0 - mask)
            state["composed"] = True
            out = self_x0.inner_model(x, sigma, model_options=model_options, seed=seed)

            active = _active_mask_for_step(
                mask=mask,
                step_sigmas=step_sigmas,
                step_index=step_index,
                transition_width=wrapper.transition_width,
            )
            return out * active + self_x0.latent_image.to(device=out.device, dtype=out.dtype) * (1.0 - active)

        KSamplerX0Inpaint.__call__ = patched_call
        try:
            return self.sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        finally:
            KSamplerX0Inpaint.__call__ = original_call


def _current_step_index(sigma, step_sigmas):
    current_sigma = sigma.flatten()[0].to(device=step_sigmas.device, dtype=step_sigmas.dtype)
    return int(torch.argmin(torch.abs(step_sigmas - current_sigma)).item())


def _smoothstep_mask(mask, threshold, width):
    if width <= 0.0:
        return (mask >= threshold).to(mask.dtype)
    width_t = torch.as_tensor(width, device=mask.device, dtype=mask.dtype).clamp_min(1e-6)
    t = ((mask - (threshold - width_t)) / width_t).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _flux2_active_mask(sigma, denoise_mask, extra_options, denoise, correction_start_sigma, transition_width):
    step_sigmas = extra_options.get("sigmas")
    if step_sigmas is None or len(step_sigmas) < 2:
        return denoise_mask

    step_sigmas = step_sigmas.to(device=denoise_mask.device, dtype=denoise_mask.dtype)
    step_index = _current_step_index(sigma.to(denoise_mask.device), step_sigmas)
    steps = max(len(step_sigmas) - 1, 1)

    denoise = float(max(0.0, min(1.0, denoise)))
    gate_index = _gate_index(step_sigmas, denoise, correction_start_sigma)

    if step_index < gate_index:
        return torch.ones_like(denoise_mask)

    progress = step_index / float(steps)
    threshold = torch.as_tensor(progress, device=denoise_mask.device, dtype=denoise_mask.dtype).clamp(0.0, 1.0)
    active = _smoothstep_mask(denoise_mask.clamp(0.0, 1.0), threshold, transition_width)
    active = torch.where(denoise_mask >= 0.95, torch.ones_like(active), active)
    active = torch.where(denoise_mask < 0.05, torch.zeros_like(active), active)
    return active.clamp(0.0, 1.0)


def _gate_index(step_sigmas, denoise, correction_start_sigma):
    steps = max(len(step_sigmas) - 1, 1)
    denoise = float(max(0.0, min(1.0, denoise)))
    start_by_denoise = int(round((1.0 - denoise) * steps))
    start_by_sigma = next((i for i, s in enumerate(step_sigmas) if float(s) <= float(correction_start_sigma)), steps)
    return max(1, min(steps, max(start_by_denoise, start_by_sigma)))


def _active_mask_for_step(mask, step_sigmas, step_index, transition_width):
    steps = max(len(step_sigmas) - 1, 1)
    progress = step_index / float(steps)
    threshold = torch.as_tensor(progress, device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)
    active = _smoothstep_mask(mask.clamp(0.0, 1.0), threshold, transition_width)
    active = torch.where(mask >= 0.95, torch.ones_like(active), active)
    active = torch.where(mask < 0.05, torch.zeros_like(active), active)
    return active.clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {
    "TBGFlux2Sampler": TBGFlux2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TBGFlux2Sampler": "TBG Flux2 Differential Diffusion Inpainting",
}
