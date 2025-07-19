import math
from collections import namedtuple

import PIL
import numpy as np
import torchvision
from PIL import Image
from numpy import sqrt
from requests import HTTPError
from .image import TBG_Image

PIL.Image.MAX_IMAGE_PIXELS = 592515344
# Application-specific imports
import comfy
import comfy.model_management as model_management
import comfy_extras
import nodes
import comfy_extras.nodes_images
import comfy_extras.nodes_mask
import torch
import requests
import json
import torch
from ....utils.constants import get_apiurl
from scipy.ndimage import gaussian_filter, grey_dilation, distance_transform_edt
import torchvision.transforms.functional as TF

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper', 'inpainting_mask', 'compositing_mask'],
                 defaults=[None])


class TBG_Segms():
    @classmethod
    def resize_with_lanczos(self, tensor_img, new_h, new_w):
        # Accepts shape (H, W), (1, H, W), or (C, H, W)
        if tensor_img.ndim == 4:
            tensor_img = tensor_img.squeeze(0)  # from [1, C, H, W] → [C, H, W]
        elif tensor_img.ndim == 2:
            tensor_img = tensor_img.unsqueeze(0)  # [H, W] → [1, H, W]
        elif tensor_img.ndim == 1:
            raise ValueError("1D image not supported")

        if tensor_img.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor_img.shape}")

        # Convert to PIL image
        img = TF.to_pil_image(tensor_img.cpu())
        img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
        # Convert back to tensor
        return TF.to_tensor(img_resized).to(tensor_img.device)

    @classmethod
    def upscale_segm_to_match_div8_and_upscalebysettings(self, segs, target_shape):
        """
        This function rescales segmentation metadata and cropped masks to match a target_shape resolution.
        Modified to ensure final cropped_image and mask pixels are divisible by 8.

        """
        h = segs[0][0]
        w = segs[0][1]

        th = target_shape.shape[1]
        tw = target_shape.shape[2]

        rh = th / h
        rw = tw / w

        if (h == th and w == tw) or h == 0 or w == 0:
            rh = 1
            rw = 1

        new_segs = []

        for seg in segs[1]:
            cropped_image = seg.cropped_image # initial None
            cropped_mask = seg.cropped_mask
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            # Calculate initial scaled dimensions
            crop_region = int(x1 * rw), int(y1 * rw), int(x2 * rh), int(y2 * rh)
            bbox = int(bx1 * rw), int(by1 * rw), int(bx2 * rh), int(by2 * rh)

            # Calculate initial dimensions
            initial_new_w = crop_region[2] - crop_region[0]
            initial_new_h = crop_region[3] - crop_region[1]

            # Make dimensions divisible by 8
            new_w =  ((initial_new_w + 7) // 8) * 8
            new_h =  ((initial_new_h + 7) // 8) * 8

            # Update crop_region to reflect the new dimensions
            crop_region = (crop_region[0], crop_region[1],
                           crop_region[0] + new_w, crop_region[1] + new_h)

            x1, y1, x2, y2 = crop_region
            cropped_image = comfy_extras.nodes_images.ImageCrop().crop(target_shape, x2 - x1, y2 - y1, x1, y1)[0]

            if isinstance(cropped_mask, np.ndarray):
                cropped_mask = torch.from_numpy(cropped_mask)


            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0).squeeze(0) # HW
            #cropped_mask = cropped_mask.permute(0, 2, 3, 1) # BCHW to BHWC
            #convert mask to  1 and 0 without gradient
            binary_mask = (cropped_mask > 0.5).float()

            compositing_mask = TBG_Segms.grow_and_blur_mask(binary_mask, 48, 16)
            inpainting_mask = TBG_Segms.grow_and_blur_mask(binary_mask,32, 64)
            cropped_mask = TBG_Segms.grow_and_blur_mask(binary_mask, 16, 32)

            new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper, inpainting_mask, compositing_mask)
            new_segs.append(new_seg)

        return (th, tw), new_segs

    @classmethod
    def create_grid_specs_for_segments(self, segs, grid_specs, maxrow, maxcol):
        h = segs[0][0]
        w = segs[0][1]
        _, seg_list = segs
        i = 9000

        for seg in segs[1]:

            x1, y1, x2, y2 = seg.crop_region
            tile_height = y2-y1
            tile_width = x2-x1

            # set col_index so that we know if the segment is touching the full image borders
            if x1 == 0:
                col_index = 0
            elif x2 > (w - 1):
                col_index = maxcol
            else:
                col_index = 1

            # set row_index so that we know if the segment is touching the full image borders
            if y1 == 0:
                row_index = 0
            elif y2 > (h - 1):
                row_index = maxrow
            else:
                row_index = 1
            i = i + 1
            order = i + 1  # unic number for each tile from 9000 up

            grid_specs.append([
                row_index,
                col_index,
                order,
                x1,  # x
                y1,  # y
                tile_width,  # width
                tile_height,  # height
            ])

        return grid_specs

    @classmethod
    def grow_and_blur_mask(self, mask, grow_margin, blur_radius_in_px):
        blur_sigma = blur_radius_in_px / 3.0
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []

        for m in growmask:
            mask_np = m.numpy()

            # Use fixed grow margin

            kernel = np.ones((grow_margin, grow_margin), dtype=np.uint8)
            dilated_mask = grey_dilation(mask_np, footprint=kernel)

            output = dilated_mask.astype(np.float32) * 255
            output = torch.from_numpy(output)
            out.append(output)

        mask = torch.stack(out, dim=0)
        mask = torch.clamp(mask, 0.0, 1.0)

        # Apply fixed Gaussian blur
        mask_np = mask.numpy()
        filtered_mask = gaussian_filter(mask_np, sigma=blur_sigma)
        mask = torch.from_numpy(filtered_mask)
        mask = torch.clamp(mask, 0.0, 1.0)

        return mask

