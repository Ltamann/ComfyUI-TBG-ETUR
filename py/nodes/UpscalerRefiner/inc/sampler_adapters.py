"""Model-specific sampler dispatch for the ETUR refiner."""

import comfy.samplers


def _log_actual_sigmas(execution, sigmas):
    if sigmas is None:
        values = []
    elif hasattr(sigmas, "detach"):
        values = sigmas.detach().float().cpu().reshape(-1).tolist()
    else:
        values = [float(value) for value in sigmas]

    rounded = [round(value, 6) for value in values]
    print(
        f"[TBG Actual Sigmas] execution={execution} count={len(rounded)} "
        f"values={rounded}"
    )


class SamplerAdapter:
    """Model sampler boundary; RGB and decode logic do not belong here."""

    def sample(self, owner, **kwargs):
        kwargs.pop("flux2_direct_sampler_selected", None)
        kwargs["ideogram_sampler"] = False
        return sample_generic(owner._sampler_runtime(), **kwargs)


class Flux2SamplerAdapter(SamplerAdapter):
    def sample(self, owner, **kwargs):
        if kwargs.get("denoise") == 0:
            return kwargs["latent_image"]
        direct = kwargs.pop("flux2_direct_sampler_selected", False)
        if direct:
            return sample_flux2_direct(owner._sampler_runtime(), **kwargs)
        kwargs["ideogram_sampler"] = False
        return sample_generic(owner._sampler_runtime(), **kwargs)


class FluxSamplerAdapter(SamplerAdapter):
    pass


class QwenSamplerAdapter(SamplerAdapter):
    pass


class Krea2SamplerAdapter(SamplerAdapter):
    pass


class IdeogramSamplerAdapter(SamplerAdapter):
    def sample(self, owner, **kwargs):
        kwargs.pop("flux2_direct_sampler_selected", None)
        kwargs["ideogram_sampler"] = True
        return sample_generic(owner._sampler_runtime(), **kwargs)


class GenericSamplerAdapter(SamplerAdapter):
    pass


def sample_flux2_direct(runtime, *, index, denoise, sigmas, sampling_model,
                        positive, negative, flux2_encode_tile,
                        complexity_mask, tile_cfg):
    if denoise == 0:
        return runtime.latent_image
    _log_actual_sigmas("Flux2Direct", sigmas)
    config = runtime.flux2_differential.DEFAULT_CONFIG
    return runtime.sample_flux2_direct_fn(
        model=sampling_model,
        positive=positive,
        negative=negative,
        pixels=flux2_encode_tile,
        vae=runtime.tbg.KSAMPLER.vae,
        mask=complexity_mask,
        steps=runtime.tbg.KSAMPLER.steps,
        seed=runtime.tbg.PROMPTER.output_seeds_js[index],
        cfg=tile_cfg,
        denoise=denoise,
        base_shift=float(config["base_shift"]),
        max_shift=float(config["max_shift"]),
        transition_width=float(config["transition_width"]),
        mask_gamma=float(config["mask_gamma"]),
        invert_mask=False,
        correction_start_sigma=float(config["correction_start_sigma"]),
        post_composite_preserve=bool(config["post_composite_preserve"]),
        sigmas=sigmas,
        denoise_method=runtime.tbg.PARAMS.denoise_method,
    )


def sample_generic(runtime, *, index, denoise, sigmas, sampling_model,
                    positive, negative, pos_low, neg_low,
                    flux2_encode_tile, complexity_mask, latent_image,
                    tile_cfg, ideogram_sampler=False):
    tbg = runtime.tbg
    if denoise == 0:
        return latent_image

    if getattr(tbg.PARAMS, "Sampler_Execution_Mode", "ETUR (current)") == "ComfyUI SamplerCustomAdvanced":
        _log_actual_sigmas("ComfyUI SamplerCustomAdvanced", sigmas)
        guider = comfy.samplers.CFGGuider(sampling_model)
        guider.set_conds(positive, negative)
        guider.set_cfg(tile_cfg)
        print(f"[TBG Sampler] execution=ComfyUI SamplerCustomAdvanced tile={index + 1} cfg={tile_cfg}")
        return runtime.sampler_custom_advanced.execute(
            runtime.noise_random_noise(tbg.PROMPTER.output_seeds_js[index]),
            guider,
            tbg.KSAMPLER.sampler,
            sigmas,
            latent_image,
        )[0]

    if ideogram_sampler and getattr(tbg.KSAMPLER, "ideogram4_guider", None) is not None:
        _log_actual_sigmas("Ideogram SamplerCustomAdvanced", sigmas)
        guider = tbg.KSAMPLER.ideogram4_guider
        return runtime.sampler_custom_advanced.execute(
            runtime.noise_random_noise(tbg.PROMPTER.output_seeds_js[index]),
            guider,
            tbg.KSAMPLER.sampler,
            sigmas,
            latent_image,
        )[0]

    if tbg.DUALMODEL.model is not None and tbg.DUALMODEL.clip is not None and tbg.DUALMODEL.vae is not None:
        _log_actual_sigmas("TBG DualModel", sigmas)
        sampler_cls = (
            runtime.dual_sampler_lanpaint
            if tbg.PARAMS.LanPaint
            else runtime.dual_sampler_normal
        )
        return sampler_cls.sample(
            0,
            tbg.DUALMODEL.inpaint_end,
            tbg.DUALMODEL.smoother_sharper,
            tbg.DUALMODEL.detail_enhancer,
            sampling_model,
            tbg.DUALMODEL.model,
            tbg.PROMPTER.output_seeds_js[index],
            tile_cfg,
            tile_cfg,
            positive,
            negative,
            pos_low,
            neg_low,
            tbg.KSAMPLER.sampler,
            tbg.KSAMPLER.scheduler,
            tbg.KSAMPLER.steps,
            tbg.DUALMODEL.steps,
            denoise,
            tbg.DUALMODEL.model_crossover_sigma_strength,
            1,
            latent_image,
        )[0]

    if tbg.DUALMODEL.inpaint_end == 0:
        inpaint_end = 10000
        inpaint_start = 0
    elif tbg.DUALMODEL.inpaint_end <= -50 or tbg.KSAMPLER.steps < abs(tbg.DUALMODEL.inpaint_end):
        inpaint_end = 0
        inpaint_start = 0
    else:
        inpaint_end = tbg.KSAMPLER.steps + tbg.DUALMODEL.inpaint_end
        inpaint_start = 0

    sampler_cls = (
        runtime.split_sampler_lanpaint
        if tbg.PARAMS.LanPaint
        else runtime.split_sampler_normal
    )
    sampling_model.model_options = dict(sampling_model.model_options)
    sampling_model.model_options["tbg_sigma_jump"] = {
        "enabled": bool(getattr(tbg.PARAMS, "Sigma_Jump_Enabled", False)),
        "strength": float(getattr(tbg.PARAMS, "Sigma_Jump_Strength", 0.0)),
        "start": float(getattr(tbg.PARAMS, "Sigma_Jump_Start", 0.0)),
        "end": float(getattr(tbg.PARAMS, "Sigma_Jump_End", 1.0)),
    }
    _log_actual_sigmas(
        "TBG LanPaint split-aware" if tbg.PARAMS.LanPaint else "TBG ETUR split-aware",
        sigmas,
    )
    sampler_kwargs = dict(
        model=sampling_model,
        add_noise=True,
        noise_seed=tbg.PROMPTER.output_seeds_js[index],
        steps=tbg.KSAMPLER.steps,
        cfg=tile_cfg,
        sampler_name=tbg.KSAMPLER.sampler,
        scheduler=tbg.KSAMPLER.scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        start_at_step=0,
        end_at_step=tbg.KSAMPLER.steps,
        denoise=denoise,
        return_with_leftover_noise=False,
        inpaint_end=inpaint_end,
        inpaint_start=inpaint_start,
        smoother_sharper=tbg.DUALMODEL.smoother_sharper,
        detail_enhancer=tbg.DUALMODEL.detail_enhancer,
        sampler_state=None,
        sigmas=sigmas,
    )
    if tbg.PARAMS.LanPaint:
        sampler_kwargs["lanpaint_steps"] = int(
            getattr(tbg.PARAMS, "LanPaint_Internal_Steps", 5)
        )

    return sampler_cls().sample(**sampler_kwargs)[0]
