import numpy as np

import comfy.model_management as model_management
import nodes
import comfy_extras
from comfy_extras.nodes_edit_model import ReferenceLatent
from ..inc.sigmas import process_image_to_tiles
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.canny.canny import CannyDetector
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything_v2.da2tgb import DepthAnythingV2Detector
from ....vendor.comfyui_controlnet_aux.utils import common_annotator_call

import torch
import torch.nn.functional as F

def apply_controlnets_from_pipe(self, cnetpipe, positive, negative, full_image, tile_image, vae):

    controlnet_node = nodes.ControlNetApplyAdvanced()
    for control in cnetpipe:
        controlnet_model = control["controlnet"]
        strength = control["strength"]
        start = control["start"]
        end = control["end"]
        preprocessor = control["preprocessor"]
        canny_high_threshold = control["canny_high_threshold"]
        canny_low_threshold = control["canny_low_threshold"]
        noise_image = control["noise_image"]
        # set image for CNET
        cnet_image = tile_image
        if noise_image is not None:
            grid_images = process_image_to_tiles(self,noise_image)
            cnet_image = grid_images[self.KSAMPLER.latent_index]
            if isinstance(cnet_image, tuple):
                cnet_image = np.array(cnet_image)

        strength = strength*self.KSAMPLER.cnet_multiply 
        # Preprocessero

        if preprocessor=="DepthAnythingV2":  
            model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
            cnet_image = common_annotator_call(model, cnet_image, resolution=1024, max_depth=1)
            del model    
        if preprocessor=="Canny Edge":
            cnet_image = common_annotator_call(CannyDetector(), cnet_image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=1024)
        positive, negative = controlnet_node.apply_controlnet(
            positive, negative, controlnet_model, cnet_image, strength, start, end, vae
        )
    return positive, negative

def get_Kontext_stiched_o_chained_cond(self, positive, cnetpipe, tile_image):
    kontext_img_combined = tile_image
    cnet_image = tile_image
    originaladded = False
    for control in cnetpipe:

        patch_for_Flux_Kontext = control["patch_for_Flux_Kontext"]
        controlnet_model = control["controlnet"]
        strength = control["strength"]
        start = control["start"]
        end = control["end"]
        preprocessor = control["preprocessor"]
        canny_high_threshold = control["canny_high_threshold"]
        canny_low_threshold = control["canny_low_threshold"]
        noise_image = control["noise_image"]


        if noise_image is not None:
            grid_images = process_image_to_tiles(self,noise_image)
            cnet_image = grid_images[self.KSAMPLER.latent_index]
            if isinstance(cnet_image, tuple):
                cnet_image = np.array(cnet_image)

        strength = strength*self.KSAMPLER.cnet_multiply
        # Preprocessor

        if preprocessor=="DepthAnythingV2":
            model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
            cnet_image = common_annotator_call(model, cnet_image, resolution=1024, max_depth=1)
            del model
        if preprocessor=="Canny Edge":
            cnet_image = common_annotator_call(CannyDetector(), cnet_image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=1024)
        if patch_for_Flux_Kontext != "NONE" and preprocessor in ("DepthAnythingV2","Canny Edge"):
            if patch_for_Flux_Kontext == "Stitched":
                # first stitch images is tile
                kontext_img_combined = comfy_extras.nodes_images.ImageStitch.stitch(
                    0,
                    kontext_img_combined,
                    'right',
                    True,
                    0,
                    'white',
                    cnet_image,
                )[0]
            if patch_for_Flux_Kontext == "Chained":
                if not originaladded:
                    originaladded == True

                    kontext_latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, tile_image)[0]
                    positive = ReferenceLatent.append(0, positive, kontext_latent_image)[0]

                cnet_image_latent = nodes.VAEEncode().encode(self.KSAMPLER.vae, cnet_image)[0]
                positive = ReferenceLatent.append(0, positive, cnet_image_latent)[0]

    # add stitched to chained
    kontext_latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, kontext_img_combined)[0]
    positive = ReferenceLatent.append(0, positive, kontext_latent_image)[0]
    return positive

import torch
import torch.nn.functional as F
import copy

def downscale_to_cnet_scale(cond1, cond2, interp_mode='bilinear'):
    """
    Given two ComfyUI CONDITIONING objects (each a list-of-[tensor, dict]),
    downscale their spatial tensors to the smaller H×W among them (ControlNet scale),
    leaving pooled_output and other 1D/2D tensors untouched,
    and return two new CONDITIONING objects with identical nesting.
    """
    # Helper to collect each main spatial tensor size
    def get_spatial_size(cond):
        sizes = []
        for tensor, meta in cond:
            if tensor.dim() >= 3:
                sizes.append((tensor.shape[-2], tensor.shape[-1]))
            # also check any reference_latents if present
            if 'reference_latents' in meta:
                for r in meta['reference_latents']:
                    if r.dim() >= 3:
                        sizes.append((r.shape[-2], r.shape[-1]))
        return sizes

    # 1) Compute the target (min_h, min_w) = smallest spatial dims across both
    all_sizes = get_spatial_size(cond1) + get_spatial_size(cond2)
    if not all_sizes:
        return cond1, cond2
    min_h = min(h for h, w in all_sizes)
    min_w = min(w for h, w in all_sizes)

    # 2) Function to downscale a single conditioning
    def downscale_cond(cond):
        new = copy.deepcopy(cond)
        for i, (tensor, meta) in enumerate(cond):
            # downscale main tensor
            if tensor.dim() >= 3:
                t = tensor
                added = False
                if t.dim() == 3:
                    t = t.unsqueeze(0); added = True
                # for 'area' mode omit align_corners
                if interp_mode in ('linear','bilinear','bicubic','trilinear'):
                    t2 = F.interpolate(t, size=(min_h, min_w), mode=interp_mode, align_corners=False)
                else:
                    t2 = F.interpolate(t, size=(min_h, min_w), mode=interp_mode)
                if added:
                    t2 = t2.squeeze(0)
                new[i][0] = t2.to(tensor.dtype).to(tensor.device)
            # downscale any reference_latents
            if 'reference_latents' in meta:
                out_refs = []
                for r in meta['reference_latents']:
                    if r.dim() >= 3:
                        rr = r
                        added = False
                        if rr.dim() == 3:
                            rr = rr.unsqueeze(0); added = True
                        if interp_mode in ('linear','bilinear','bicubic','trilinear'):
                            rr2 = F.interpolate(rr, size=(min_h, min_w), mode=interp_mode, align_corners=False)
                        else:
                            rr2 = F.interpolate(rr, size=(min_h, min_w), mode=interp_mode)
                        if added:
                            rr2 = rr2.squeeze(0)
                        out_refs.append(rr2.to(r.dtype).to(r.device))
                    else:
                        out_refs.append(r)
                new[i][1]['reference_latents'] = out_refs
        return new

    # 3) Apply to both conditionings
    eq1 = downscale_cond(cond1)
    eq2 = downscale_cond(cond2)
    return eq1, eq2

#

import torch
import torch.nn.functional as F
import copy


import torch
import torch.nn.functional as F
import copy

import torch
import torch.nn.functional as F
import copy

def adapt_cnet_to_biggest_reference(cond, interp_mode='bilinear'):
    """
    For one ComfyUI CONDITIONING (list of [tensor, meta_dict]):
      - Find the maximum H×W among all meta['reference_latents'] tensors.
      - Upscale only meta['control'].cond_hint to (max_h, max_w).
      - Leave all other tensors unchanged.
      - Preserve structure exactly.
    """
    # 1. Find max H×W among all reference_latents
    max_h = max_w = 0
    for tensor, meta in cond:
        for ref in meta.get('reference_latents', []):
            if isinstance(ref, torch.Tensor) and ref.dim() >= 3:
                h, w = ref.shape[-2], ref.shape[-1]
                if h > max_h: max_h = h
                if w > max_w: max_w = w

    # If no reference_latents or already uniform, return original
    if max_h == 0 or max_w == 0:
        return cond

    # 2. Deep copy and upscale only ControlNet cond_hint
    new_cond = copy.deepcopy(cond)
    for i, (tensor, meta) in enumerate(cond):
        cnet = meta.get('control', None)
        if cnet is not None:
            hint = getattr(cnet, 'cond_hint', None)
            if isinstance(hint, torch.Tensor) and hint.dim() >= 3:
                # Prepare for interpolation
                # If hint is CHW, add batch dim; if BCHW, leave as is
                batched = False
                if hint.dim() == 3:
                    hint = hint.unsqueeze(0)
                    batched = True

                # Upscale with bilinear (supports align_corners)
                up = F.interpolate(hint, size=(max_h, max_w),
                                   mode=interp_mode, align_corners=False)

                # Restore shape and device/dtype
                if batched:
                    up = up.squeeze(0)
                up = up.to(hint.dtype).to(hint.device)

                # Assign back into copied structure
                new_cond[i][1]['control'].cond_hint = up

    return new_cond


def debug_conditioning(cond):
    """
    Print every tensor in the conditioning structure along with its shape,
    and count how many 'reference_latents' entries (latent images) are present.
    """
    latent_count = 0

    def recurse(obj, path="cond"):
        nonlocal latent_count
        if torch.is_tensor(obj):
            print(f"{path}: tensor shape = {tuple(obj.shape)}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                recurse(v, f"{path}[{i}]")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k == "reference_latents" and isinstance(v, list):
                    for j, ref in enumerate(v):
                        if torch.is_tensor(ref):
                            latent_count += 1
                            recurse(ref, f"{path}['{k}'][{j}]")
                        else:
                            recurse(ref, f"{path}['{k}'][{j}]")
                else:
                    recurse(v, f"{path}['{k}']")

    recurse(cond)
    print(f"Total reference_latents tensors (latent images): {latent_count}")

def equalize_spatial_tensors(cond, interp_mode='area'):
    """
    Given a CONDITIONING (list of [tensor, meta_dict]):
    - Compute min H×W among all spatial tensors (dim ≥ 3, including reference_latents).
    - Resize only those spatial tensors to (min_h, min_w) using interp_mode.
    - Leave flat (dim ≤ 2) tensors unchanged.
    - Return a deep-copied conditioning with spatial sizes equalized.
    """
    # 1. Gather spatial sizes
    sizes = []
    for tensor, meta in cond:
        if tensor.dim() >= 3:
            sizes.append((tensor.shape[-2], tensor.shape[-1]))
        for ref in meta.get('reference_latents', []):
            if isinstance(ref, torch.Tensor) and ref.dim() >= 3:
                sizes.append((ref.shape[-2], ref.shape[-1]))
    if not sizes:
        return cond
    min_h, min_w = min(h for h, w in sizes), min(w for h, w in sizes)

    # 2. Deep copy and resize only spatial tensors
    new_cond = copy.deepcopy(cond)
    for i, (tensor, meta) in enumerate(cond):
        # Resize main spatial tensor
        if tensor.dim() >= 3:
            t = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
            if interp_mode in ('linear','bilinear','bicubic','trilinear'):
                t2 = F.interpolate(t, size=(min_h, min_w), mode=interp_mode, align_corners=False)
            else:
                t2 = F.interpolate(t, size=(min_h, min_w), mode=interp_mode)
            new_cond[i][0] = t2.squeeze(0) if tensor.dim() == 3 else t2.to(tensor.dtype).to(tensor.device)

        # Resize reference_latents if present
        if 'reference_latents' in meta:
            out_refs = []
            for ref in meta['reference_latents']:
                if isinstance(ref, torch.Tensor) and ref.dim() >= 3:
                    r = ref.unsqueeze(0) if ref.dim() == 3 else ref
                    if interp_mode in ('linear','bilinear','bicubic','trilinear'):
                        r2 = F.interpolate(r, size=(min_h, min_w), mode=interp_mode, align_corners=False)
                    else:
                        r2 = F.interpolate(r, size=(min_h, min_w), mode=interp_mode)
                    out_refs.append(r2.squeeze(0) if ref.dim() == 3 else r2.to(ref.dtype).to(ref.device))
                else:
                    out_refs.append(ref)
            new_cond[i][1]['reference_latents'] = out_refs

    return new_cond


def equalize_to_smallest(cond, interp_mode='bilinear'):
    """
    Downscale all spatial tensors in a ComfyUI conditioning structure
    to the smallest H×W found among them, preserving structure.

    Args:
        cond: A conditioning list-of-[tensor, dict] (and nested dict/list) structure.
        interp_mode: Interpolation mode for downscaling ('area' recommended).
    Returns:
        New conditioning structure with every spatial tensor resized
        to (min_h, min_w).
    """
    # 1. Collect all tensor infos
    tensor_infos = []

    def collect(obj, path):
        if torch.is_tensor(obj):
            tensor_infos.append({'tensor': obj, 'path': path, 'shape': obj.shape})
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                collect(v, path + [i])
        elif isinstance(obj, dict):
            for k, v in obj.items():
                collect(v, path + [k])

    collect(cond, [])

    # 2. Determine minimum spatial dims among tensors of rank ≥3
    min_h = float('inf')
    min_w = float('inf')
    for info in tensor_infos:
        sh = info['shape']
        if len(sh) >= 3:
            h, w = sh[-2], sh[-1]
            min_h, min_w = min(min_h, h), min(min_w, w)
    # If no spatial tensors or only one size => nothing to do
    if min_h == float('inf') or min_w == float('inf'):
        return cond

    # 3. Schedule downscaling for any tensor larger than (min_h, min_w)
    updates = []
    for info in tensor_infos:
        t = info['tensor']
        sh = info['shape']
        if len(sh) >= 3:
            h, w = sh[-2], sh[-1]
            if h != min_h or w != min_w:
                # Move to CPU for safer memory usage
                cpu_t = t.detach().cpu()
                added_batch = False
                if cpu_t.dim() == 3:
                    cpu_t = cpu_t.unsqueeze(0)
                    added_batch = True

                # Downscale on CPU
                down = F.interpolate(cpu_t, size=(min_h, min_w),
                                     mode=interp_mode, align_corners=False)

                if added_batch:
                    down = down.squeeze(0)
                # Back to original device and dtype
                down = down.to(t.dtype).to(t.device)

                updates.append((info['path'], down))
                del cpu_t, down
                torch.cuda.empty_cache()

    # 4. Apply updates to a deep copy of the conditioning
    new_cond = copy.deepcopy(cond)
    for path, new_t in updates:
        ptr = new_cond
        for key in path[:-1]:
            ptr = ptr[key]
        ptr[path[-1]] = new_t

    return new_cond


def equalize_single_conditioning(cond, mode='bilinear'):
    """
    Upsample all spatial tensors in one conditioning structure to the
    maximum H×W found, using CPU for interpolation to avoid GPU OOM.
    mode: interpolation mode ('area', 'nearest', 'bilinear', etc.)
    """
    # 1. Collect tensor infos
    tensor_infos = []
    def collect(obj, path):
        if torch.is_tensor(obj):
            tensor_infos.append({'tensor': obj, 'path': path, 'shape': obj.shape})
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                collect(v, path + [i])
        elif isinstance(obj, dict):
            for k, v in obj.items():
                collect(v, path + [k])
    collect(cond, [])

    # 2. Compute max spatial dims
    max_h = max_w = 0
    for info in tensor_infos:
        sh = info['shape']
        if len(sh) >= 3:
            h, w = sh[-2], sh[-1]
            max_h, max_w = max(max_h, h), max(max_w, w)
    if max_h==0 or max_w==0:
        return cond

    # 3. Prepare updates
    updates = []
    for info in tensor_infos:
        t = info['tensor']
        sh = info['shape']
        if len(sh) >= 3 and (sh[-2]!=max_h or sh[-1]!=max_w):
            # move to CPU, add batch dim if needed
            cpu_t = t.detach().cpu()
            added_batch = False
            if cpu_t.dim()==3:
                cpu_t = cpu_t.unsqueeze(0)
                added_batch = True

            # interpolate on CPU with cheap mode
            up = F.interpolate(cpu_t, size=(max_h, max_w), mode=mode, align_corners=False)

            # restore batch dim, move back to original device & dtype
            if added_batch:
                up = up.squeeze(0)
            up = up.to(t.dtype).to(t.device)

            updates.append((info['path'], up))

            # free memory
            del cpu_t, up
            torch.cuda.empty_cache()

    # 4. Apply updates to deep copy
    new_cond = copy.deepcopy(cond)
    for path, new_t in updates:
        ptr = new_cond
        for key in path[:-1]:
            ptr = ptr[key]
        ptr[path[-1]] = new_t

    return new_cond
