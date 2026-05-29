"""MIT License

Copyright (c) [2025] [TBG Tobias Laarmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
from comfy.samplers import KSamplerX0Inpaint
import latent_preview
import math
import comfy.utils

class TBGKSampler():
    @classmethod
    def __init__(self, model, steps, scheduler, device, initial_state=None):
        self.model = model.clone()
        self.steps = steps
        self.scheduler = scheduler
        self.device = device
        self.full_sigmas = None
        self.original_denoise_mask_fn = None
        self._x0_output = {}

        self._state = initial_state
        self.step_count = 0
        self.total_step_count = 0

    @classmethod
    def calculate_full_sigmas(self, model_sampling, scheduler, steps):
        self.full_sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        )
        return self.full_sigmas

    @classmethod
    def apply_hybrid_sharpening(self, out, x_anterior, sigma, sharpener, steps):
        if sharpener == 0:
            return out
        if sigma > 0.5 and sigma < 0.8:
            strength = -(sharpener * 0.02) * sigma**0.1
            delta = out - x_anterior
            out += delta * strength

        if sigma < 0.5 and sigma > 0.3:
            low_freq = self._gaussian_blur(out, sigma)
            high_freq = out - low_freq
            strength = sharpener * (1-sigma) * 5/max(1,steps)
            sharpened_high = high_freq * strength

            out = out + sharpened_high

        return out

    @classmethod
    def _gaussian_blur(self, latent, sigma_normalized):
        import torch.nn.functional as F
        kernel_size = int(sigma_normalized * 4.0) * 2 + 1
        kernel_size = max(3, min(kernel_size, 13))
        device = latent.device
        dtype = latent.dtype
        sigma_blur = torch.tensor(sigma_normalized * 1.5 + 0.5, device=device, dtype=dtype)
        x = torch.linspace(
            -(kernel_size - 1) / 2,
            (kernel_size - 1) / 2,
            kernel_size,
            device=device,
            dtype=dtype
        )
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma_blur ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.repeat(latent.shape[1], 1, 1, 1)
        padding = kernel_size // 2
        low_freq = F.conv2d(latent, kernel_2d, padding=padding, groups=latent.shape[1])
        return low_freq

    @classmethod
    def create_denoise_mask_wrapper(self, original_denoise_mask_fn):
        def wrapper(sigma, denoise_mask, extra_options):
            original_sigmas = extra_options.get("sigmas")
            extra_options["sigmas"] = self.full_sigmas
            if original_denoise_mask_fn is not None:
                result = original_denoise_mask_fn(sigma, denoise_mask, extra_options)
            else:
                result = denoise_mask
            if original_sigmas is not None:
                extra_options["sigmas"] = original_sigmas

            return result

        return wrapper

    @classmethod
    def create_inpaint_patch(self, latent_image, inpaint_start, inpaint_end, is_split_pass, sharpener, detail_enhancer, better_inpainting):
        def get_interpolated_sigmas(sigma, detail_enhancer):
            if detail_enhancer == 0:
                return sigma, sigma
            current_idx = self.total_step_count
            if detail_enhancer > 0:
                if current_idx < len(self.full_sigmas) - 1:
                    sigma_target = self.full_sigmas[current_idx + 1]
                else:
                    sigma_target = self.full_sigmas[-1]
                interpolation_amount = abs(detail_enhancer)
                sigma_interpolated = sigma + interpolation_amount * (sigma_target - sigma)
            else:
                if current_idx > 0:
                    sigma_target = self.full_sigmas[current_idx - 1]
                else:
                    sigma_target = self.full_sigmas[0]
                interpolation_amount = abs(detail_enhancer)
                sigma_interpolated = sigma + interpolation_amount * (sigma_target - sigma)
            if detail_enhancer > 0:
                return sigma_interpolated, sigma
            else:
                return sigma_interpolated, sigma

        def model_call(self_x0, x, sigma, model_options, seed, *args, **kwargs):
            step_divisor = max(1, int(math.ceil(self.steps / 3)))
            if detail_enhancer != 0 and self.total_step_count % step_divisor == 0 and self.total_step_count > 2:
                sigma_1, sigma_2 = get_interpolated_sigmas(sigma, detail_enhancer)
                out1  = self._call_model_function(self_x0, x, sigma_1, model_options, seed, *args, **kwargs)
                out2 = self._call_model_function(self_x0, x , sigma_2 , model_options, seed, *args, **kwargs)
                if detail_enhancer > 0:
                    out = out1 * 0.7 + out2 * 0.3
                else:
                    out = out1 * 0.3 + out2 * 0.7
                return out
            else:
                out = self._call_model_function(self_x0, x, sigma, model_options, seed, *args, **kwargs)
                return out

        def min_max_for_latentmask(sigma):
            max_val, min_val = 1.0, 0.05 * sigma ** 0.1
            return max_val, min_val

        def _apply_inpaint_blending_where( denoise_mask, sigma, better_inpainting):
            if better_inpainting:
                max_val, min_val = min_max_for_latentmask(sigma)
                latent_mask = max_val - (max_val - min_val) * denoise_mask
            else:
                latent_mask = 1. - denoise_mask
            return latent_mask, denoise_mask

        def patched_inpaint_call(self_x0, x, sigma, denoise_mask,
                                 model_options={}, seed=None, *args, **kwargs):
            if self.step_count == 0 and self._state is not None:
                if "noise" in self._state:
                    self_x0.noise = self._state["noise"]
                if "original_latent" in self._state:
                    self_x0.latent_image = self._state["original_latent"]
            if "transformer_options" in model_options:
                to = model_options["transformer_options"]
                if "sample_sigmas" in to:
                    self_x0.sigmas = to["sample_sigmas"]
            if inpaint_start <= self.total_step_count <= inpaint_end and denoise_mask is not None:
                if "denoise_mask_function" in model_options:
                    denoise_mask = self._apply_denoise_mask(model_options, sigma, denoise_mask, self_x0.inner_model)
                latent_mask, denoise_mask  = _apply_inpaint_blending_where( denoise_mask, sigma, better_inpainting)
                x = x * denoise_mask +  self_x0.inner_model.inner_model.scale_latent_inpaint(
                    x=x, sigma=sigma, noise=self_x0.noise, latent_image=self_x0.latent_image
                ) * latent_mask
            if sharpener != 0 and self.total_step_count < self.steps - 3:
                x_anterior = x
            out = model_call(self_x0, x, sigma, model_options, seed, *args, **kwargs)
            if denoise_mask is not None and inpaint_start <= self.total_step_count <= inpaint_end:
                latent_mask, denoise_mask  = _apply_inpaint_blending_where(denoise_mask, sigma, better_inpainting)

                out = out * denoise_mask + self_x0.latent_image * latent_mask

            if sharpener != 0 and self.total_step_count < self.steps-3:
                out = self.apply_hybrid_sharpening(out, x_anterior, sigma, sharpener,self.steps)
            self._update_state(self_x0, out, out, denoise_mask, model_options, seed)
            self._increment_step_counters()
            return out

        def first_pass_call(self_x0, x, sigma, denoise_mask,
                            model_options={}, seed=None, *args, **kwargs):

            if inpaint_start <= self.total_step_count <= inpaint_end and denoise_mask is not None:
                denoise_mask = self._apply_denoise_mask(model_options, sigma, denoise_mask, self_x0.inner_model)
                latent_mask, denoise_mask = _apply_inpaint_blending_where(denoise_mask, sigma, better_inpainting)
                x = x * denoise_mask + self_x0.inner_model.inner_model.scale_latent_inpaint(
                    x=x, sigma=sigma, noise=self_x0.noise, latent_image=self_x0.latent_image
                ) * latent_mask
            if sharpener != 0 and self.total_step_count < self.steps - 3:
                x_anterior = x
            out = model_call(self_x0, x, sigma, model_options, seed,*args, **kwargs)
            if denoise_mask is not None and inpaint_start <= self.total_step_count <= inpaint_end:
                latent_mask, denoise_mask = _apply_inpaint_blending_where(denoise_mask, sigma, better_inpainting)
                out = out * denoise_mask + self_x0.latent_image * latent_mask
            if sharpener != 0 and self.total_step_count < self.steps - 3:
                out = self.apply_hybrid_sharpening(out, x_anterior, sigma, sharpener,self.steps)
            self._update_state(self_x0, out, out, denoise_mask, model_options, seed)
            self._increment_step_counters()
            return out

        return patched_inpaint_call if is_split_pass else first_pass_call

    @classmethod
    def _apply_denoise_mask(self, model_options, sigma, denoise_mask, model_for_options):
        if "denoise_mask_function" in model_options:
            return model_options["denoise_mask_function"](
                sigma, denoise_mask,
                extra_options={"model": model_for_options, "sigmas": self.full_sigmas}
            )
        return denoise_mask

    @classmethod
    def _call_model_function(self, self_x0, x, sigma, model_options, seed, *args, **kwargs):
        if "model_function_wrapper" in model_options:
            model_func = model_options["model_function_wrapper"]
        else:
            model_func = self_x0.inner_model
        return model_func(x, sigma, model_options=model_options, seed=seed, *args, **kwargs)

    @classmethod
    def _update_state(self, self_x0, out, x, denoise_mask, model_options, seed):
        self._state = {
            'noise': self_x0.noise,
            # 'inner_model': self_x0.inner_model,
            'original_latent': self_x0.latent_image,
            'sigmas': self_x0.sigmas,
            'output': out,
            'input_x': x,
            'denoise_mask': denoise_mask,
            'model_options': model_options,
            'seed': seed,
            'step_count': self.step_count,
            'total_step_count': self.total_step_count
        }

    @classmethod
    def _increment_step_counters(self):
        self.step_count += 1
        self.total_step_count += 1

    @classmethod
    def prepare_noise(self, latent_image, noise_seed, disable_noise, batch_inds=None):
        if disable_noise:
            return torch.zeros_like(latent_image)
        if batch_inds is None and isinstance(latent_image, dict):
            batch_inds = latent_image.get("batch_index")

        return comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)

    @classmethod
    def sample(self, noise, positive, negative, latent_image,
               cfg, sampler_name, denoise, start_at_step, end_at_step,
               force_full_denoise, noise_mask, callback, disable_pbar, seed,
               inpaint_start=0, inpaint_end=1000, disable_noise=False, sharpener=0, detail_enhancer=0, sigmas=None, better_inpainting= False):
        self.step_count = 0
        self.total_step_count = start_at_step
        model_sampling = self.model.get_model_object("model_sampling")
        if sigmas is None:
            self.full_sigmas = self.calculate_full_sigmas(model_sampling, self.scheduler,self.steps)
        else:
            self.full_sigmas = sigmas
            self.steps = len(sigmas)-1
        self.original_denoise_mask_fn = self.model.model_options.get("denoise_mask_function")
        print(f"[TBG Differential Diffusion] sampler_hook_present={self.original_denoise_mask_fn is not None}")
        mask_wrapper = self.create_denoise_mask_wrapper(self.original_denoise_mask_fn)
        self.model.set_model_denoise_mask_function(mask_wrapper)
        is_split_pass = start_at_step > 0 or self._state is not None
        original_call = KSamplerX0Inpaint.__call__
        inpaint_patch = self.create_inpaint_patch(
            latent_image, inpaint_start, inpaint_end, is_split_pass, sharpener, detail_enhancer, better_inpainting
        )
        KSamplerX0Inpaint.__call__ = inpaint_patch

        try:
            samples = comfy.sample.sample(
                self.model,
                noise=noise,
                steps=self.steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=self.scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image["samples"],
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )

            out_latent = samples.to(comfy.model_management.intermediate_device())
            result = {"samples": out_latent}
            if "noise_mask" in latent_image:
                result["noise_mask"] = latent_image["noise_mask"]
            return result, self._state

        finally:
            KSamplerX0Inpaint.__call__ = original_call
            self.model.set_model_denoise_mask_function(self.original_denoise_mask_fn)
            self._cleanup_state()

    @classmethod
    def _cleanup_state(self):
        self.step_count = 0
        self.total_step_count = 0
        self._x0output = {}
        self._state = None  # Clear the state dictionary
        # Force garbage collection if needed
        import gc
        gc.collect()


class TBG_KSamplerAdvancedSplitAware_Copy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True,"tooltip": "Enable: add random noise (first sampler). Disable: no noise added (sampler chaining/img2img). Use 'enable' for first sampler, 'disable' for subsequent ones."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "inpaint_start": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "Step number where inpainting injection begins. Inpainting blends the original latent with the denoised latent within the masked area. Set to 0 to start inpainting from the first step. Higher values delay inpainting, allowing more creative freedom before preserving masked regions."}),
                "inpaint_end": ("INT", {"default": 10000, "min": 0, "max": 10000,"tooltip": "Step number where inpainting injection ends. Set to a high value (e.g., 1000) to inpaint through all steps. Lower values allow later steps to refine without mask constraints. Works with inpaint_start to define the inpainting window."}),
                "smoother_sharper": ("FLOAT", {"default": 0, "min":-1.0, "max": 1.0,"display": "slider", "tooltip": "Dual-stage adaptive sharpening. At high sigma (early steps), adds structured noise for detail invention. At low sigma (late steps), applies high-pass edge sharpening. Positive values sharpen and add details. Negative values soften and blur. Zero disables sharpening. Higher absolute values create stronger effects."}),
                "detail_enhancer": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "display": "slider", "tooltip": "Substep evaluation for detail control. Positive values (0.1-1.0): lookahead to next sigma, adds coherent details and refinement, reduces variation. Negative values (-0.1 to -1.0): lookback to previous sigma, adds creative variation and texture complexity. Zero = disabled (single pass, fastest). Performance cost: 2x slower on affected steps."}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False,"tooltip": "Enable: keeps noise for sampler chaining. Disable: fully denoise output. Use 'enable' for split workflows, 'disable' for final step."}),
                #"better_inpainting": ("BOOLEAN",{"default": False, "tooltip": "Reduces Lattend Grid artifacts in unmasked areas, but slightly alters the masked output."}),

            },
            "optional": {
                "sampler_state": ("sampler_state", {"label": "Sampler State (for chaining)","tooltip": "Chain state from previous sampler. Leave empty for first sampler, connect for subsequent samplers in multi-pass workflows."}),
            }
        }

    RETURN_TYPES = ("LATENT", "sampler_state")
    RETURN_NAMES = ("LATENT", "sampler_state")
    FUNCTION = "sample"
    CATEGORY = "TBG/Sampler"

    @classmethod
    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step, denoise,
               return_with_leftover_noise, inpaint_end, inpaint_start,smoother_sharper,detail_enhancer,
               sampler_state=None):
        better_inpainting = False
        device = model.load_device

        if end_at_step > steps:
            end_at_step = steps

        force_full_denoise = return_with_leftover_noise == False
        disable_noise = add_noise == False
        noise_mask = latent_image.get("noise_mask")

        latent_samples = latent_image.get("samples", None).to(device).clone()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        sampler = TBGKSampler(model, steps, scheduler, device, initial_state=sampler_state)

        batch_inds = latent_image.get("batch_index")
        noise = sampler.prepare_noise(latent_samples, noise_seed, disable_noise, batch_inds)

        callback = latent_preview.prepare_callback(model, steps, sampler._x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        result, state = sampler.sample(
            noise=noise,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            cfg=cfg,
            sampler_name=sampler_name,
            denoise=denoise,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
            inpaint_start=inpaint_start,
            inpaint_end=inpaint_end,
            disable_noise=disable_noise,
            sharpener=smoother_sharper,
            detail_enhancer=detail_enhancer,
            sigmas=None,
            better_inpainting = better_inpainting,
        )

        return (result, state)



class TBG_DualModelSampler_COPY:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high": ("MODEL",{"tooltip": "First model for high sigma phase. IMPORTANT: Both model_high and model_low must use the same or compatible latent spaces and VAE models. Compatible pairs: Flux + ZImages, Qwen + WAN. Incompatible models will produce corrupted outputs due to latent space mismatch."}),
                "model_low": ("MODEL",{"tooltip": "Second model for low sigma phase. IMPORTANT: Must be latent space compatible with model_high (same VAE, same latent dimensions). Compatible pairs: Flux + ZImages, Qwen + WAN. Using incompatible models will fail or produce artifacts."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "cfg_high": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "cfg_low":  ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "positive_high": ("CONDITIONING",),
                "negative_high": ("CONDITIONING",),
                "positive_low": ("CONDITIONING",),
                "negative_low": ("CONDITIONING",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps_high": ("INT", {"default": 20, "min": 1, "max": 2048,"tooltip":"Total steps used to generate the high model's sigma schedule. This is the full step count for a typical generation, not the actual steps executed. The sigma_split_value determines which portion of this schedule is used. Example: FLUX 20 steps with split=0.5 might execute ~10 high-sigma steps."}),
                "steps_low":  ("INT", {"default": 9,  "min": 1, "max": 2048,"tooltip":"Total steps used to generate the low model's sigma schedule. This is the full step count for a typical generation, not the actual steps executed. The sigma_split_value determines which portion of this schedule is used. Example: Z-image 10 steps with split=0.5 might execute ~5 low-sigma steps."}),
                "denoise": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model_crossover_sigma_strength": ("FLOAT", {"display": "slider","default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip":"Controls the sigma strength where sampling switches from the high model to the low model.”"}),
                "low_sigma_alignment": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,"tooltip":"Shifts the lower sigma values up or down to better synchronize sampling between models. Default is 1, which works best in most cases."}),
                "inpaint_end": ("INT", { "display": "slider", "default": 0, "min": -50, "max": 0,
                                        "tooltip": "Step number from the end after which inpainting is skipped. For example, with 20 total steps, setting -10 means inpainting runs only from step 1 to 10."}),

                "smoother_sharper": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "display": "slider",
                                               "tooltip": "Dual-stage adaptive sharpening. At high sigma (early steps), adds structured noise for detail invention. At low sigma (late steps), applies high-pass edge sharpening. Positive values sharpen and add details. Negative values soften and blur. Zero disables sharpening. Higher absolute values create stronger effects."}),
                "detail_enhancer": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "display": "slider", "tooltip": "Substep evaluation for detail control. Positive values (0.1-1.0): lookahead to next sigma, adds coherent details and refinement, reduces variation. Negative values (-0.1 to -1.0): lookback to previous sigma, adds creative variation and texture complexity. Zero = disabled (single pass, fastest). Performance cost: 2x slower on affected steps."}),
                #"better_inpainting": ("BOOLEAN", {"default": False, "tooltip": "Reduces Lattend Grid artifacts in unmasked areas, but slightly alters the masked output."}),
            },
            "optional": {
                "latent_image": ("LATENT",),
            }
        }


    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "sample"
    CATEGORY = "TBG/Sampler"

    @classmethod
    def _get_sigmas_basic(self, model, scheduler, steps, denoise, label):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return torch.FloatTensor([])
            total_steps = int(steps / denoise)
        sigmas_full = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        sigmas = sigmas_full[-(steps + 1):]
        return sigmas

    @classmethod
    def _split_high_part(self, sigmas, split_val):
        s = sigmas.float()
        mask_high = s > split_val
        idxs = torch.nonzero(mask_high, as_tuple=False).flatten().tolist()
        high = s[mask_high]
        return high

    @classmethod
    def _split_low_part(self, sigmas, split_val):
        s = sigmas.float()
        mask_low = s <= split_val
        idxs = torch.nonzero(mask_low, as_tuple=False).flatten().tolist()
        low = s[mask_low]
        return low

    @classmethod
    def _normalize_segment(self, seg, start_val, end_val, label):
        if seg.numel() == 0:
            return seg
        if seg.numel() == 1:
            out = torch.tensor([start_val], device=seg.device, dtype=seg.dtype)
            return out
        t = torch.linspace(0.0, 1.0, seg.numel(), device=seg.device)
        out = start_val + (end_val - start_val) * t

        return out

    @classmethod
    def _run_custom(self,inpaint_end, better_inpainting, smoother_sharper,detail_enhancer,start_step,end_step,sigmas_full, model, add_noise, noise_seed, cfg,
                    positive, negative, sampler_name, scheduler, sigmas, latent_image, label, sampler_state=None):
        device = model.load_device
        if not add_noise:
            force_full_denoise = False
            disable_noise = True
        else:
            force_full_denoise = False
            disable_noise = False

        noise_mask = latent_image.get("noise_mask")


        latent_samples = latent_image.get("samples", None).to(device).clone()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        sampler = TBGKSampler(model, len(sigmas)-1, scheduler, device, initial_state=sampler_state)

        batch_inds = latent_image.get("batch_index")
        noise = sampler.prepare_noise(latent_samples, noise_seed, disable_noise, batch_inds)

        callback = latent_preview.prepare_callback(model, len(sigmas)-1, sampler._x0_output)



        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        inpaint_start = 0
        if inpaint_end == 0:
            inpaint_end = 10000
        elif inpaint_end <= -50 or self.steps < abs(inpaint_end):
            inpaint_end = 0
            inpaint_start = 10000
        else: #inpaint_end -32 inpaint_steps 5    29 Total - 24 = 5
            if start_step == 0:
                inpaint_steps = self.steps + inpaint_end #   29 Total - 24 = 5
                if end_step <= inpaint_steps:
                    inpaint_end = 10000
                else:
                    inpaint_end = inpaint_steps-(self.steps - start_step - end_step)

            else:

                range_steps = end_step - start_step
                if  range_steps > abs(inpaint_end):
                    inpaint_end = range_steps + inpaint_end
                else:
                    inpaint_end = 0

        result, state = sampler.sample(
            noise=noise,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            cfg=cfg,
            sampler_name=sampler_name,
            denoise=1,
            start_at_step=start_step,
            end_at_step=end_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
            inpaint_start=inpaint_start,
            inpaint_end=inpaint_end,
            disable_noise=disable_noise,
            sharpener=smoother_sharper,
            detail_enhancer=detail_enhancer,
            sigmas = sigmas_full,
            better_inpainting = better_inpainting,
        )

        return (result, state)

    @classmethod
    def sample(
            self,
            inpaint_end,
            smoother_sharper,
            detail_enhancer,
            model_high,
            model_low,
            noise_seed,
            cfg_high,
            cfg_low,
            positive_high,
            negative_high,
            positive_low,
            negative_low,
            sampler_name,
            scheduler,
            steps_high,
            steps_low,
            denoise,
            model_crossover_sigma_strength,
            low_sigma_alignment,
            latent_image=None,
    ):
        better_inpainting = False
        if latent_image is None:
            raise ValueError("TBG Sampler requires a latent_image input.")

        # 1) Get full sigma schedules
        sigmas_high_original = self._get_sigmas_basic(model_high, scheduler, steps_high, denoise, "HIGH")
        sigmas_low_original = self._get_sigmas_basic(model_low, scheduler, steps_low, denoise, "LOW")

        # Edge cases: single model only
        if model_crossover_sigma_strength <= 0.0:
            out_high, state = self._run_custom(
                inpaint_end, better_inpainting, smoother_sharper, detail_enhancer,
                0, len(sigmas_high_original) - 1, sigmas_high_original,
                model_high, True, noise_seed, cfg_high,
                positive_high, negative_high,
                sampler_name, scheduler, sigmas_high_original, latent_image, "HIGH_full",
            )
            return (out_high,)

        if model_crossover_sigma_strength >= 1.0:
            out_low, state = self._run_custom(
                inpaint_end, better_inpainting, smoother_sharper, detail_enhancer,
                0, len(sigmas_low_original) - 1, sigmas_low_original,
                model_low, True, noise_seed, cfg_low,
                positive_low, negative_low,
                sampler_name, scheduler, sigmas_low_original, latent_image, "LOW_full",
            )
            return (out_low,)

        # 2) DUAL MODEL MODE
        s_h = sigmas_high_original.float().clone()
        s_l = sigmas_low_original.float().clone()

        sigma_max_h = s_h[0].item()
        sigma_min_h = s_h[-1].item()
        sigma_max_l = s_l[0].item()
        sigma_min_l = s_l[-1].item()

        split_val = float(model_crossover_sigma_strength)



        # Step 1: Find nearest sigma in HIGH schedule to split_val
        high_distances = torch.abs(s_h - split_val)
        high_swap_idx = torch.argmin(high_distances).item()
        high_swap_sigma = s_h[high_swap_idx].item()



        # Step 2: Find where this sigma should appear in LOW schedule
        # Look for the step just before it crosses high_swap_sigma
        low_swap_idx = None
        for i in range(len(s_l) - 1):
            if s_l[i] > high_swap_sigma >= s_l[i + 1]:
                # The swap should happen between i and i+1
                # Use the step before (i) so we can interpolate to match exactly
                low_swap_idx = i
                break

        # Fallback: if not found in range, find closest
        if low_swap_idx is None:
            low_distances = torch.abs(s_l - high_swap_sigma)
            low_swap_idx = torch.argmin(low_distances).item()


        # Step 3: Interpolate FULL LOW schedule to have exact match at low_swap_idx
        sigmas_low_modified = s_l.clone()

        # We need to create a new schedule where sigmas_low_modified[low_swap_idx] = high_swap_sigma
        # Strategy: Interpolate all sigmas based on their position relative to the swap point

        # Part A: From start to swap point - interpolate from sigma_max_l to high_swap_sigma
        if low_swap_idx > 0:
            num_steps_before = low_swap_idx + 1  # Including the swap point
            before_segment = torch.linspace(
                sigma_max_l,
                high_swap_sigma,
                num_steps_before,
                dtype=s_l.dtype,
                device=s_l.device
            )
            sigmas_low_modified[:low_swap_idx + 1] = before_segment
        else:
            # Swap point is at the beginning
            sigmas_low_modified[0] = high_swap_sigma

        # Part B: From swap point to end - interpolate from high_swap_sigma to sigma_min_l
        if low_swap_idx < len(s_l) - 1:
            num_steps_after = len(s_l) - low_swap_idx  # Including the swap point
            after_segment = torch.linspace(
                high_swap_sigma,
                sigma_min_l,
                num_steps_after,
                dtype=s_l.dtype,
                device=s_l.device
            )
            sigmas_low_modified[low_swap_idx:] = after_segment
        else:
            # Swap point is at the end
            sigmas_low_modified[-1] = high_swap_sigma

        # Apply low_sigma_alignment to the portion after swap
        sigmas_low_modified[low_swap_idx:] = sigmas_low_modified[low_swap_idx:] * low_sigma_alignment

        # Re-apply the exact swap sigma after alignment
        sigmas_low_modified[low_swap_idx] = high_swap_sigma * low_sigma_alignment


        # Step 4: Determine start/end steps for sampling
        start_step_high = 0
        end_step_high = high_swap_idx
        start_step_low = low_swap_idx
        end_step_low = len(sigmas_low_modified) - 1

        steps_executed_high = end_step_high - start_step_high
        steps_executed_low = end_step_low - start_step_low
        self.steps = steps_executed_high + steps_executed_low

        # 5) HIGH segment sampling - use ORIGINAL high sigmas (no modification)

        out_high, state = self._run_custom(
            inpaint_end, better_inpainting, smoother_sharper, detail_enhancer,
            start_step_high, end_step_high, s_h,
            model_high, True, noise_seed, cfg_high,
            positive_high, negative_high,
            sampler_name, scheduler, s_h[:end_step_high + 1], latent_image, "HIGH_part"
        )

        # 6) LOW segment sampling - use INTERPOLATED low sigmas

        out_low, state = self._run_custom(
            inpaint_end, better_inpainting, smoother_sharper, detail_enhancer,
            start_step_low, end_step_low, sigmas_low_modified,
            model_low, False, noise_seed, cfg_low,
            positive_low, negative_low,
            sampler_name, scheduler, sigmas_low_modified[start_step_low:], out_high, "LOW_part", state
        )

        return (out_low,)


