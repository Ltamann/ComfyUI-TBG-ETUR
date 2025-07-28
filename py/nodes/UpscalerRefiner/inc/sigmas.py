
# Standard library imports

import PIL
import torch.nn.functional as F
import torch.nn.functional as Fnoise

PIL.Image.MAX_IMAGE_PIXELS = 592515344
from PIL import Image


# Application-specific imports
import comfy
import nodes

# Relative imports - local module
from ..inc.prompt import Node as NodePrompt
from ....utils.log import log, COLORS

import math
import comfy.samplers
import comfy.sample

import latent_preview
import torch
import comfy.utils
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

@staticmethod
def tbgsample(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
    latent = latent_image
    latent_image = latent["samples"]
    latent = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    latent["samples"] = latent_image

    if not add_noise:
        noise = Noise_EmptyNoise().generate_noise(latent)
    else:
        noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

    # ✅ PATCH HERE: Force clean Gaussian + Jitter to avoid grid artifacts
    noise = torch.randn_like(noise)  # Replace any latent grid-patterned noise
    noise += torch.randn_like(noise) * 0.01  # Optional: small jitter
    # Debug test: force complete randomness
    # latent_image = torch.randn_like(latent_image)
    # noise = torch.randn_like(latent_image) + torch.randn_like(latent_image) * 0.01

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                         noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
                                         seed=noise_seed)


    out = latent.copy()
    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return (out, out_denoised)



def _get_sigmas(sigmas_type, model, steps, denoise, scheduler, model_type):
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sigmas 

def inject_noise( samples, noise_std):
    s = samples.copy()
    pixel_scale=1
    noise = torch.randn_like(s["samples"]) * noise_std

    if pixel_scale > 1:
        # Reduce resolution and upsample to control grain size
        b, c, h, w = noise.shape
        noise = Fnoise.interpolate(noise, size=(h // pixel_scale, w // pixel_scale), mode='bilinear', align_corners=False)
        noise = Fnoise.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)
    
    s["samples"] = s["samples"] + noise
    return (s,)



def set_tiles_to_process(tiles_to_process_active, tiles, tiles_to_process=''):

    max_tiles = len(tiles)
    max = max_tiles if max_tiles > 0 else NodePrompt.INPUT_QTY
    
    def is_valid_index(index, max = NodePrompt.INPUT_QTY):
        return 1 <= index <= max
    def to_computer_index(human_index):
        return human_index - 1

    _tiles_to_process = []
   
    if not tiles_to_process_active:
        return _tiles_to_process
    if tiles_to_process == '':
        return _tiles_to_process

    indexes = tiles_to_process.split(',')
    
    for index in indexes:
        index = index.strip()
        if '-' in index:
            # Range of indexes
            start, end = map(int, index.split('-'))
            if is_valid_index(start, max) and is_valid_index(end, max):
                _tiles_to_process.extend(range(to_computer_index(start), to_computer_index(end) + 1))
            else:
                _tiles_to_process.append(-1)
                log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node ")
        else:
            # Single index
            try:
                index = int(index)
                if is_valid_index(index, max):
                    _tiles_to_process.append(to_computer_index(index))
                else:
                    _tiles_to_process.append(-1)
                    log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node ")
            except ValueError:
                _tiles_to_process.append(-1)
                # Ignore non-integer values
                pass

    # Remove duplicates and sort
    _tiles_to_process = sorted(set(_tiles_to_process))
    if -1 in _tiles_to_process:
        _tiles_to_process = [-1]

    return _tiles_to_process



def process_image_to_tiles(self, input_image): # for controllnet preprocessing with exterior file only
        # upscale cnet preprocessor iamge to image size
        resized_image = nodes.ImageScale().upscale(input_image, self.PARAMS.upscale_method, round(self.OUTPUTS.image.shape[2]), round(self.OUTPUTS.image.shape[1]), False)[0]
                               
        # Get tiled grid specifications
        grid_specs = MS_Image().get_tiled_grid_specs(resized_image, self.SIZE.actual_inner_tile_sizeH, self.SIZE.actual_inner_tile_sizeW, self.PARAMS.rows_qty, self.PARAMS.cols_qty , self.SIZE.feather_mask_margin,self.SIZE.shift)[0]
        # Generate grid images (tiles)
        grid_images = MS_Image().get_grid_images(resized_image, grid_specs)
        return grid_images




def denoise_sigmas(sigmas, denoise, denoise_method):
    total_steps = sigmas.shape[0]
    if denoise_method == "default":
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            aumented_steps = int(total_steps/denoise)
            sigmas = sigmas.view(1, 1, -1)
            interpolated_sigmas = F.interpolate( sigmas, size=aumented_steps, mode='linear', align_corners=True    ).view(-1)
            sigmas = interpolated_sigmas[-(total_steps):]

    if denoise_method == "multiplyed":
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            sigmas = sigmas * denoise   

    if denoise_method == 'multiplyed normalized':
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)            
            max_sigma = sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = sigmas * scale_factor

    if denoise_method == "normalized":
        # first get default sigmas 
        # noramlized scales the default sigmas to the denoise value
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            aumented_steps = int(total_steps/denoise)
           
            sigmas = sigmas.view(1, 1, -1)    
            interpolated_sigmas = F.interpolate(  sigmas, size=aumented_steps, mode='linear', align_corners=True).view(-1)
            sigmas = interpolated_sigmas[-(total_steps):]

            # second scale the sigmas to max value = denoise value            
            max_sigma = sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = sigmas * scale_factor

    if denoise_method == "default short ":
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            reduced_steps = math.ceil(total_steps*denoise)
            sigmas = sigmas[-(reduced_steps):]


    if denoise_method == "normalized advanced":
        # noramlized advanced cuts the sigmas where denoise fits restnoise 
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)

         # Fund the step where denoise = restnoise
            differences = torch.abs(sigmas - denoise)
            closest_step = torch.argmin(differences).item()
         # Cut the tail of the sigmas starting from closest_step
            sliced_sigmas = sigmas[closest_step:]

            if sliced_sigmas.shape[0] < 2:
                # Interpolation needs at least 2 points
                return (torch.FloatTensor([]),)

            # Reshape for 1D linear interpolation
            sliced_sigmas = sliced_sigmas.view(1, 1, -1)

            # Interpolate to total_steps
            
            interpolated_sigmas = F.interpolate(sliced_sigmas, size=total_steps, mode='linear', align_corners=True).view(-1)
            # Scale to restnoise = denoise
            max_sigma = interpolated_sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = interpolated_sigmas * scale_factor
            # Normalize so that max becomes `denoise`
            max_sigma = interpolated_sigmas.max()
            if max_sigma > 0:
                scale_factor = denoise / max_sigma
                sigmas = interpolated_sigmas * scale_factor
            else:
                return (torch.FloatTensor([]),)

    return sigmas


def get_sigmas(model, scheduler, total_steps, denoise, denoise_method):
    if denoise == 1:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        return (sigmas, )
    if denoise_method == "default":
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            aumented_steps = int(total_steps/denoise)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, aumented_steps).cpu()
        # Cuts out the lowSteps from the totalSteps
        sigmas = sigmas[-(total_steps):]

    if denoise_method == "multiplyed":
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            sigmas = sigmas * denoise      

    if denoise_method == 'multiplyed normalized':
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)            
            max_sigma = sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = sigmas * scale_factor

    if denoise_method == "normalized":
        # first get default sigmas 
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            aumented_steps = int(total_steps/denoise)
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, aumented_steps).cpu()
            # Cuts out the lowSteps from the totalSteps
            sigmas = sigmas[-(total_steps):]

        # second scale the sigmas to max value = denoise value            
        max_sigma = sigmas.max()
        scale_factor = denoise / max_sigma
        sigmas = sigmas * scale_factor

    if denoise_method == "default short ":
        # first get default sigmas 
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            reduced_steps = int(total_steps*denoise)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        # Cuts out the lowSteps from the totalSteps
        sigmas = sigmas[-(reduced_steps + 1):] 

    if denoise_method == "normalized advanced":
        if denoise is not None and denoise < 1.0:
            if denoise <= 0.0:
                returneturn (torch.FloatTensor([]),)
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
         # Fund the step where denoise = restnoise
            differences = torch.abs(sigmas - denoise)
            closest_step = torch.argmin(differences).item()
         # Cut the tail of the sigmas starting from closest_step
            sliced_sigmas = sigmas[closest_step:]

            if sliced_sigmas.shape[0] < 2:
                # Interpolation needs at least 2 points
                return (torch.FloatTensor([]),)

            # Reshape for 1D linear interpolation
            sliced_sigmas = sliced_sigmas.view(1, 1, -1)

            # Interpolate to total_steps
            interpolated_sigmas = F.interpolate(
                sliced_sigmas, size=total_steps, mode='linear', align_corners=True
            ).view(-1)
            # Scale to restnoise = denoise
            max_sigma = interpolated_sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = interpolated_sigmas * scale_factor
            # Normalize so that max becomes `denoise`
            max_sigma = interpolated_sigmas.max()
            if max_sigma > 0:
                scale_factor = denoise / max_sigma
                sigmas = interpolated_sigmas * scale_factor
            else:
                return (torch.FloatTensor([]),)

    return (sigmas, )

def stretch_sigma_head(sigma: torch.Tensor, multiplier: float):
    total_len = len(sigma)
    head_len = int(round(total_len * 0.1))  # First 10%
    stretched_len = int(round(head_len * multiplier))

    if head_len < 2:
        raise ValueError("Need at least 2 values to interpolate.")

    # Get the head and tail
    head = sigma[:head_len]
    tail = sigma[head_len:]

    # Interpolate the head
    head = head.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, head_len]
    stretched_head = F.interpolate(head, size=stretched_len, mode='linear', align_corners=True)
    stretched_head = stretched_head.squeeze()

    # Combine stretched head with the tail
    new_sigma = torch.cat([stretched_head, tail], dim=0)
    return new_sigma