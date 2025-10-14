import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy_extras
import nodes
import copy
import torch
import torch.nn.functional as F
import math

from comfy_extras.nodes_custom_sampler import SamplerEulerAncestral, SamplerCustomAdvanced
from ..inc.image import TBG_Image
from ..inc.sigmas import inject_noise, _get_sigmas

# V3 scheme name changes
Sampler_custom = comfy_extras.nodes_custom_sampler.SamplerCustom()
if hasattr(Sampler_custom, "sample"):
    Sampler_custom_execute = Sampler_custom.sample
elif hasattr(Sampler_custom, "execute"):
    Sampler_custom_execute = Sampler_custom.execute

DEBUG = False

try:
    from custom_nodes.RES4LYF.legacy.samplers import ClownSamplerAdvanced
    HAS_CLOWNSAMPLER = True
except ImportError:
    ClownSamplerAdvanced = None
    HAS_CLOWNSAMPLER = False

class TBG_sampler():
    @staticmethod
    def getSampler(self,positive,negative,sigmas,latent_image,index):

        if DEBUG:
            print("SamplerCustom used")

        grid_seed = self.OUTPUTS.grid_seed[index]
        if grid_seed is not None and isinstance(grid_seed, (int)):
            noise_seed = grid_seed

        else:
            noise_seed = self.KSAMPLER.noise_seed

        latent_output = Sampler_custom_execute(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            noise_seed,
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

        pixels = TBG_Image().helper_upscaleimage(pixels, round(latent_image_W), round(latent_image_H),
                                                 self.PARAMS.upscale_method_inpainting,
                                                 self.PARAMS.upscale_model_inpainting)
        latent_output = nodes.VAEEncode().encode(self.KSAMPLER.vae, pixels)[0]
        latent_output = inject_noise(latent_output, self.KSAMPLER.latentupscale_noise)[0]
        LatentUpscaleByScalesigmas = _get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model,
                                                 self.KSAMPLER.latentupscale_steps,
                                                 self.KSAMPLER.latentupscale_denoise, self.KSAMPLER.scheduler,
                                                 self.KSAMPLER.model_type)

        latent_output = Sampler_custom.execute(
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
        pixels = TBG_Image().helper_upscaleimage(pixels, round(latent_image_W), round(latent_image_H),
                                                 self.PARAMS.upscale_method_inpainting,
                                                 self.PARAMS.upscale_model_inpainting)

        latent_output = nodes.VAEEncode().encode(self.KSAMPLER.vae, pixels)[0]

        # V3 scheme name changes

        latent_output = Sampler_custom_execute(
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
    def getClownSampler(self, inputsampler):

        cs_eta = self.KSAMPLER.eta * 2
        cs_s_noise = self.KSAMPLER.eta * 0.5
        if HAS_CLOWNSAMPLER == True:
            if DEBUG:
                print("HAS_CLOWNSAMPLER used")
         #   print("Switching to ClownSampler res_2m for eta noise injection")

            noise_type_sde = "gaussian"  # "brownian"
            noise_type_sde_substep = "gaussian"  # "gaussian"
            noise_mode_sde = "soft"  # "hard"

            eta = cs_eta
            eta_var = 0.0
            d_noise = 1.0
            s_noise = cs_s_noise
            alpha_sde = -1.0
            k_sde = 1.0
            cfgpp = 0.0
            c1 = 0.0
            c2 = 0.5
            c3 = 1.0
            noise_seed_sde = -1
            sampler_name = "res_2m" # self.KSAMPLER.sampler_name
            implicit_sampler_name = "gauss-legendre_2s"
            t_fn_formula = None#
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

                sampler = comfy.samplers.ksampler("rk", {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde, "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp,
                                                         "noise_sampler_type": noise_type_sde, "noise_mode": noise_mode_sde, "noise_seed": noise_seed_sde, "rk_type": sampler_name,
                                                         "implicit_sampler_name": implicit_sampler_name,
                                                         "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                         "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "mask_inv": latent_guide_mask_inv,
                                                         "latent_guide_weights": latent_guide_weights, "latent_guide_weights_inv": latent_guide_weights_inv, "guide_mode": guide_mode,
                                                         "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override, "sde_noise": sde_noise,
                                                         "extra_options": extra_options,
                                                         "etas": etas, "s_noises": s_noises, "unsample_resample_scales": unsample_resample_scales, "regional_conditioning_weights": regional_conditioning_weights,
                                                         "guides": guides, "frame_weights_grp": frame_weights_grp, "eta_substep": eta_substep, "noise_mode_sde_substep": noise_mode_sde_substep,
                                                         })


                return sampler
            except Exception as e:
                print("e")
        else:
            if DEBUG:
                print("SamplerEulerAncestral used")
           # print("Switching to SamplerEulerAncestral for eta noise injection")
            eta = min(self.KSAMPLER.eta * 4, 3)
            s_noise = min(self.KSAMPLER.eta * 1.2, 1)

            # V3 scheme name changes
            if hasattr(SamplerEulerAncestral, "sample"):
                SamplerEulerAncestral_execute = SamplerEulerAncestral.get_sampler
            elif hasattr(SamplerEulerAncestral, "execute"):
                SamplerEulerAncestral_execute = SamplerEulerAncestral.execute


            sampler = SamplerEulerAncestral_execute(0, eta, s_noise)[0]
        return sampler








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

    import torch

    def modify_sigma_curve(sigma_curve: torch.Tensor, stepnumber: int,
                                              p=2.0, wiggle_amplitude=0.1, alpha=0.0):
        sigma_low = sigma_curve[:stepnumber]
        sigma_high = sigma_curve[stepnumber:]
        n = len(sigma_high)
        device = sigma_curve.device

        start_val = sigma_high[0]

        t = torch.linspace(0, 1, n, device=device)
        decay_curve = (1 - t ** p)
        wiggle = wiggle_amplitude * torch.sin(t * math.pi)
        combined_curve = (decay_curve + wiggle).clamp(min=0)

        shaped_curve = start_val * combined_curve

        # Interpolate between original and max(original, shaped) curve using alpha
        interp_curve = (1 - alpha) * sigma_high + alpha * torch.max(sigma_high, shaped_curve)

        sigma_modified = torch.cat([sigma_low, interp_curve], dim=0)

        return sigma_modified



    @staticmethod
    def Sigmas_crative_tail(self, sigmas ):
        simgasmod = copy.copy(sigmas)
        #print(sigmas)
        simgasmod = TBG_sampler.modify_sigma_curve(simgasmod, int(len(sigmas)/2), 2, 0.1, min(self.KSAMPLER.Sigmas_creative_tail_strength,1))
        #print(simgasmod)
        return simgasmod

    @staticmethod
    def splitstep(self, positive, negative, latent_image, sigmas, index):

        grid_seed = self.OUTPUTS.grid_seed[index]
        if grid_seed is not None and isinstance(grid_seed, (int)):
            noise_seed = grid_seed

        else:
            noise_seed = self.KSAMPLER.noise_seed


        latent_image_with_leftover_noise = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            "enable",
            noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.sampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent_image,
            0, # start steps
            self.KSAMPLER.SplitSteps_steps,  # end steps
            "enable", #left ofer noise True False
            self.KSAMPLER.denoise)[0]

        # inject noise  self.KSAMPLER.SplitSteps_noise
        noise_std = self.KSAMPLER.SplitSteps_noise #/1000

        latent_image_with_leftover_noise = inject_noise(latent_image_with_leftover_noise, noise_std)[0]

        latent_output = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            "disable",
            noise_seed,
            self.KSAMPLER.steps,
            self.KSAMPLER.cfg,
            self.KSAMPLER.sampler,
            self.KSAMPLER.scheduler,
            positive,
            negative,
            latent_image_with_leftover_noise,
            self.KSAMPLER.SplitSteps_steps, # start steps
            self.KSAMPLER.steps,  # end steps
            "disable", #left ofer noise True False
            self.KSAMPLER.denoise)[0]

        return latent_output


    @staticmethod
    def splitstep_with_sigmacurve(self, positive, negative, latent_image, mask , sigma_curve, multiplier, index):

        grid_seed = self.OUTPUTS.grid_seed[index]
        if grid_seed is not None and isinstance(grid_seed, (int)):
            noise_seed = grid_seed

        else:
            noise_seed = self.KSAMPLER.noise_seed

        # wrong denoise because KSamplerAdvanced its building its own
        latent = nodes.KSamplerAdvanced().sample(
            self.KSAMPLER.model,
            self.KSAMPLER.add_noise,
            noise_seed,
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
                    noise_seed,
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
            noise_seed,
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
