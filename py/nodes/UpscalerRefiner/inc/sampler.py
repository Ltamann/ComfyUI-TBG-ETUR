# getClownSampler setting fro ComfyUI/custom_nodes/TBG_upscaler/py/vendor/RES4LYF check licence here
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy_extras
import nodes
import torch
import torch.nn.functional as F
from ..inc.image import TBG_Image
from ..inc.sigmas import inject_noise, _get_sigmas


class TBG_sampler():
    @staticmethod
    def getSampler(self,positive,negative,sigmas,latent_image,index):

        latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.cfg,
            positive,
            negative,
            self.KSAMPLER.sampler,
            sigmas,
            latent_image,
        )[0]
        return latent_output

    @staticmethod
    def latentupscale(self, positive, negative, sigmas, latent_image, index, latent_image_W, latent_image_H, latent_output):
        # upscale in pixel space - for flux its gives more consistency than latent space, can add blur
        pixels = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]
        scale_factor = 1.2
        w = pixels.shape[2] * scale_factor
        h = pixels.shape[1] * scale_factor

        pixels = TBG_Image().better_downscale(pixels, round(latent_image_W), round(latent_image_H),
                                              self.PARAMS.upscale_method_inpainting,
                                              self.PARAMS.upscale_model_inpainting)
        latent_output = nodes.VAEEncode().encode(self.KSAMPLER.vae, pixels)[0]
        latent_output = inject_noise(latent_output, self.KSAMPLER.latentupscale_noise)[0]
        LatentUpscaleByScalesigmas = _get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model,
                                                 self.KSAMPLER.latentupscale_steps,
                                                 self.KSAMPLER.latentupscale_denoise, self.KSAMPLER.scheduler,
                                                 self.KSAMPLER.model_type)
        latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.cfg,
            positive,
            negative,
            self.KSAMPLER.EPsampler,
            LatentUpscaleByScalesigmas,
            latent_output
        )[0]
        # inject noise  self.KSAMPLER.SplitSteps_noise
        latent_output = inject_noise(latent_output, self.KSAMPLER.latentupscale_noise)[0]
        # downscale latent to old size
        pixels = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]
        pixels = TBG_Image().better_downscale(pixels, round(latent_image_W), round(latent_image_H),
                                              self.PARAMS.upscale_method_inpainting,
                                              self.PARAMS.upscale_model_inpainting)

        latent_output = nodes.VAEEncode().encode(self.KSAMPLER.vae, pixels)[0]

        latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.cfg,
            positive,
            negative,
            self.KSAMPLER.EPsampler,
            LatentUpscaleByScalesigmas,
            latent_output
        )[0]
        return latent_output





    @staticmethod
    def getClownSampler(self):



        noise_type_sde = "gaussian"  # "brownian"
        noise_type_sde_substep = "gaussian"  # "gaussian"
        noise_mode_sde = "soft"  # "hard"

        eta = self.KSAMPLER.eta
        eta_var = 0.0
        d_noise = 1.0
        s_noise = 1.0
        alpha_sde = -1.0
        k_sde = 1.0
        cfgpp = 0.0
        c1 = 0.0
        c2 = 0.5
        c3 = 1.0
        noise_seed_sde = -1
        sampler_name = self.KSAMPLER.sampler_name
        implicit_sampler_name = "gauss-legendre_2s"
        t_fn_formula = None
        sigma_fn_formula = None
        implicit_steps = 0
        latent_guide = None
        latent_guide_inv = None
        guide_mode = ""
        latent_guide_weights = None
        latent_guide_weights_inv = None
        latent_guide_mask = None
        latent_guide_mask_inv = None
        rescale_floor = True
        sigmas_override = None
        guides = None
        options = None
        sde_noise = None
        sde_noise_steps = 1
        extra_options = ""
        automation = None
        etas = None
        s_noises = None
        unsample_resample_scales = None
        regional_conditioning_weights = None
        frame_weights_grp = None
        eta_substep = 0.5
        noise_mode_sde_substep = "hard"
        try:
            sampler = comfy.samplers.ksampler("rk",
                                                            {"eta": eta, "eta_var": eta_var, "s_noise": s_noise,
                                                             "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde,
                                                             "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp,
                                                             "noise_sampler_type": noise_type_sde,
                                                             "noise_mode": noise_mode_sde,
                                                             "noise_seed": noise_seed_sde,
                                                             "rk_type": sampler_name,
                                                             "implicit_sampler_name": implicit_sampler_name,
                                                             "t_fn_formula": t_fn_formula,
                                                             "sigma_fn_formula": sigma_fn_formula,
                                                             "implicit_steps": implicit_steps,
                                                             "latent_guide": latent_guide,
                                                             "latent_guide_inv": latent_guide_inv,
                                                             "mask": latent_guide_mask,
                                                             "mask_inv": latent_guide_mask_inv,
                                                             "latent_guide_weights": latent_guide_weights,
                                                             "latent_guide_weights_inv": latent_guide_weights_inv,
                                                             "guide_mode": guide_mode,
                                                             "LGW_MASK_RESCALE_MIN": rescale_floor,
                                                             "sigmas_override": sigmas_override,
                                                             "sde_noise": sde_noise,
                                                             "extra_options": extra_options,
                                                             "etas": etas, "s_noises": s_noises,
                                                             "unsample_resample_scales": unsample_resample_scales,
                                                             "regional_conditioning_weights": regional_conditioning_weights,
                                                             "guides": guides,
                                                             "frame_weights_grp": frame_weights_grp,
                                                             "eta_substep": eta_substep,
                                                             "noise_mode_sde_substep": noise_mode_sde_substep,
                                                             })

            return sampler
        except Exception as e:
            raise Exception(
                f"Enrichment Pipes [Sampler side noise injection] failed: {e}\n"
                f"RES4LYF custom node might be missing: https://github.com/ClownsharkBatwing/RES4LYF"
                f"install https://github.com/ClownsharkBatwing/RES4LYF or set [Sampler side noise injection] = 0"
            )

    @staticmethod
    def inject_noise(samples, noise_std, mask, pixel_scale=1):
        s = samples.copy()
        img = s["samples"]  # shape: [B, C, H, W]

        # Generate noise
        noise = torch.randn_like(img) * noise_std

        if pixel_scale > 1:
            b, c, h, w = noise.shape
            noise = F.interpolate(noise, size=(h // pixel_scale, w // pixel_scale), mode='bilinear',
                                  align_corners=False)
            noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)

        # Prepare mask: [H, W] → [1, 1, H, W] to broadcast
        mask = mask.to(img.device).float().unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

        # Apply noise only inside the masked area
        s["samples"] = img + noise * mask

        return (s,)

    @staticmethod
    def splitstep(self,positive,negative,latent_image,mask,index):

        latent_image_with_leftover_noise = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.EPsampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent_image,
            0, # start steps
            self.KSAMPLER.SplitSteps_steps,  # end steps
            True, #left ofer noise True False
            self.KSAMPLER.denoise)[0]

        # inject noise  self.KSAMPLER.SplitSteps_noise
        noise_std = self.KSAMPLER.SplitSteps_noise/1000
        #latent_image_with_leftover_noise = inject_noise(latent_image_with_leftover_noise, noise_std)[0]
        latent_image_with_leftover_noise = inject_noise(latent_image_with_leftover_noise, noise_std, mask, pixel_scale=1)[0]

        latent_output = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.EPsampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent_image_with_leftover_noise,
            self.KSAMPLER.SplitSteps_steps, # start steps
            self.KSAMPLER.steps,  # end steps
            False, #left ofer noise True False
            self.KSAMPLER.denoise)[0]

        return latent_output


    @staticmethod
    def splitstep_with_sigmacurve(self, positive, negative, latent_image, mask , sigma_curve, multiplier):

        latent = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.EPsampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent_image,
            0,  # start steps
            self.KSAMPLER.SplitStepsStart,  # end steps
            True,  # left ofer noise True False
            self.KSAMPLER.denoise)[0]

        for i in range(self.KSAMPLER.SplitStepsStart,  self.KSAMPLER.SplitStepsEnd):
        #for i in range(self.KSAMPLER.steps):
            if  i != 0 and i != self.KSAMPLER.steps:
                # Compute noise to inject
                noise = sigma_curve[i] * multiplier/100

                # Inject noise into the latent image
                latent_image_with_leftover_noise = \
                latent =inject_noise(latent, noise)[0]

                # Perform one sampling step (you might need to modify your sampler to support single-step or use a custom scheduler loop)
                latent = nodes.KSamplerAdvanced().sample(
                    self.KSAMPLER.model,
                    self.KSAMPLER.add_noise,
                    self.KSAMPLER.noise_seed,
                    self.KSAMPLER.steps,
                    self.KSAMPLER.cfg,
                    self.KSAMPLER.EPsampler,
                    self.KSAMPLER.scheduler,
                    positive,
                    negative,
                    latent,
                    i,  # step range 0–1 to do a single step
                    i+1,
                    True,  # left over noise
                    self.KSAMPLER.denoise)[0]

        latent = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            self.KSAMPLER.noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.EPsampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent,
            self.KSAMPLER.SplitStepsEnd,  # start steps
            self.KSAMPLER.steps,  # end steps
            False,  # left ofer noise True False
            self.KSAMPLER.denoise)[0]


        return latent
