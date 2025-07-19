# from ComfyUI/custom_nodes/TBG_upscaler/py/vendor/ComfyUI_Fluxtapoz check licence here, modifications: just incorporation in TBG nodes for tiled Upscaling
import node_helpers
import numpy as np
import torch
import torch.nn.functional as F
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, DisableNoise, BasicGuider
from comfy_extras.nodes_mask import LatentCompositeMasked
from ....vendor.ComfyUI_Fluxtapoz.nodes.flux_deguidance_node import FluxDeGuidance
from ....vendor.ComfyUI_Fluxtapoz.nodes.influx_model_pred_node import InFluxModelSamplingPredNode, OutFluxModelSamplingPredNode
from ....vendor.ComfyUI_Fluxtapoz.nodes.rectified_sampler_nodes import FluxForwardODESamplerNode, FluxReverseODESamplerNode


@staticmethod
def modedsigmas(RFsigma):
    # Assume RFsigma is a torch tensor
    RFsigma = torch.linspace(1, 0, steps=100)  # Example tensor

    # Reference shape as a NumPy array
    ref_shape = np.array([999, 100, 10, 200, 20, 550, 200, 100, 50, 30, 0])

    # Normalize to 0–1
    ref_norm = (ref_shape - ref_shape[-1]) / (ref_shape[0] - ref_shape[-1])

    # Interpolate to match the number of steps in RFsigma
    x_ref = np.linspace(0, 1, len(ref_norm))
    x_target = np.linspace(0, 1, RFsigma.shape[0])
    RFsigma_interp = np.interp(x_target, x_ref, ref_norm)

    # Convert interpolated array to torch tensor
    RFsigma_interp = torch.tensor(RFsigma_interp, dtype=RFsigma.dtype, device=RFsigma.device)

    # Scale to match original start and end
    start, end = RFsigma[0], RFsigma[-1]
    RFsigma_scaled = RFsigma_interp * (start - end) + end

    # Final result:
    return (RFsigma_scaled)



class RF_inversion():
    @classmethod
    def get_RF_latent_flux(self,upscaled_image_grid, latent_image, model, sigmas, conditioning, mask, max_shift=1.15, base_shift=0.5,multip=0):
        width = upscaled_image_grid.shape[2]
        height = upscaled_image_grid.shape[1]
        rf_sigmas = sigmas.flip(0)#*1.05 #higermultiplier more rest noise
        if rf_sigmas[0] == 0:
            rf_sigmas[0] = 1e-3
        rf_noise = DisableNoise.get_noise(self)[0]
        rf_positive = FluxDeGuidance.append(self,conditioning, guidance=0)[0]
        rf_model = InFluxModelSamplingPredNode.patch(self,model, max_shift, base_shift, width, height)[0]
        rf_sampler = FluxForwardODESamplerNode.build(self,gamma=0.5, seed=0)[0]
        rf_guider = BasicGuider.get_guider(self,rf_model, rf_positive)[0]
        rf_latent = SamplerCustomAdvanced.sample(self, rf_noise, rf_guider, rf_sampler, rf_sigmas, latent_image)[0]
        # RF_inversion is a multiplier for freedom 0 no 1 many
        eta = multip   # smaller more freedom 0 to 1
        start_step = 0
        print("rf_sigmas",rf_sigmas)
        print("rf_sigmas", sigmas)
        #reducerby = -int(len(sigmas) / 2) - 1  # by full its get blurry by 50 is sharp sweet spot
        #reduced_sigmas = sigmas[:reducerby]

        aumented_steps = int(len(sigmas) / 0.5)
        sigmasx = sigmas.view(1, 1, -1)
        interpolated_sigmas = F.interpolate(sigmasx, size=aumented_steps, mode='linear', align_corners=True).view(-1)
        reduced_sigmas = interpolated_sigmas[-(len(sigmas)):]
        #reduced_sigmas = modedsigmas(reduced_sigmas)

        print("reduced_sigmas",reduced_sigmas)

        end_step = int(len(reduced_sigmas) - len(reduced_sigmas) * multip)
        end_step = max(3, min(end_step, len(reduced_sigmas)))
        eta = max(0, min(eta, 1))

        rf_newSampler = FluxReverseODESamplerNode.build(self, rf_model, latent_image, eta, start_step, end_step, eta_trend='linear_increase')[0]
        rf_newModel = OutFluxModelSamplingPredNode.patch(self, model, max_shift, base_shift, width, height, reverse_ode=True)[0]
        rf_positive = FluxDeGuidance.append(self, conditioning, guidance=3.5)[0]
        rf_guider = BasicGuider.get_guider(self, rf_newModel, rf_positive)[0]
        # bei dobble unsampilg sigmas were rf-sigmas
        rf_latent = SamplerCustomAdvanced.sample(self, rf_noise, rf_guider, rf_newSampler, reduced_sigmas, latent_image)[0]
        rf_latent= LatentCompositeMasked.composite(self, rf_latent, latent_image, 0, 0, False, mask = mask)[0]
        return (rf_latent, rf_newSampler, rf_newModel)


    @classmethod
    def InpaintModelConditioninglatent(self, positive, negative, latent, mask, noise_mask=True):

        concat_latent =latent["samples"]
        orig_latent = latent["samples"]

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)