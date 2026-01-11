import math
from collections import namedtuple

import PIL
import numpy as np
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 592515344
# Application-specific imports
import torch
from scipy.ndimage import gaussian_filter, grey_dilation, distance_transform_edt
import torchvision.transforms.functional as TF
from collections import namedtuple
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
    def upscale_segm_to_match_div8_and_upscalebysettings(cls, segs, target_shape):
        """
        Rescale segmentation metadata and cropped masks so that they can be
        composited onto the tiled‑upscaled image.

        The important change is that **coordinates are never altered after the
        padding/scale step** (the transformation performed in
        `TBG_Image.transform_segment_coordinates`).  We only snap the four
        corners to the nearest multiple of 8 once, then keep those values for
        both cropping *and* stitching.  The mask tensor is forced to 8‑divisible
        size, but it is cropped back to the exact region so the spatial
        coordinates stay unchanged.
        """
        # -----------------------------------------------------------------
        # 0 – original image size (height, width) that was supplied to the
        #     function as the first element of ``segs``.
        # -----------------------------------------------------------------
        h, w = segs[0]  # (height, width) of the ORIGINAL image
        th, tw = target_shape.shape[1], target_shape.shape[2]  # size of the up‑scaled full image

        # -----------------------------------------------------------------
        # 1 – scaling factors from original → up‑scaled image.
        # -----------------------------------------------------------------
        rh = th / h if h else 1.0  # vertical scale
        rw = tw / w if w else 1.0  # horizontal scale

        # -----------------------------------------------------------------
        # Helper: round a value **down** to the nearest multiple of 8.
        # -----------------------------------------------------------------
        def _floor8(v: int) -> int:
            return (v // 8) * 8

        # Helper: round a value **up** to the nearest multiple of 8.
        # -----------------------------------------------------------------
        def _ceil8(v: int) -> int:
            return ((v + 7) // 8) * 8

        new_segs = []

        # -----------------------------------------------------------------
        # 2 – Walk through every segment and produce a new SEG where
        #     *crop_region* is 8‑aligned but otherwise identical to the
        #     already‑scaled region.
        # -----------------------------------------------------------------
        for seg in segs[1]:
            # -----------------------------------------------------------------
            # a) Get the *already‑scaled* coordinates that came from
            #    `transform_segment_coordinates`.  They are still in pixel units
            #    of the up‑scaled, padded image.
            # -----------------------------------------------------------------
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            # -----------------------------------------------------------------
            # b) Snap every corner to the 8‑grid (left/top down, right/bottom up).
            #    This guarantees that the **rectangle is fully 8‑divisible**
            #    while staying inside the original region.
            # -----------------------------------------------------------------
            x1_al = _floor8(int(x1))
            y1_al = _floor8(int(y1))
            x2_al = _ceil8(int(x2))
            y2_al = _ceil8(int(y2))

            # Same rule for the optional bbox (only used for visualisation).
            bx1_al = _floor8(int(bx1))
            by1_al = _floor8(int(by1))
            bx2_al = _ceil8(int(bx2))
            by2_al = _ceil8(int(by2))

            # The **final** crop region that will be used for both the crop
            # operation and later stitching.
            crop_region = (x1_al, y1_al, x2_al, y2_al)

            # -----------------------------------------------------------------
            # c) Crop the image from the up‑scaled whole picture.
            # -----------------------------------------------------------------

            def crop(image, width, height, x, y):
                x = min(x, image.shape[2] - 1)
                y = min(y, image.shape[1] - 1)
                to_x = width + x
                to_y = height + y
                img = image[:, y:to_y, x:to_x, :]
                return (img,)


            cropped_image =crop(
                target_shape,
                x2_al - x1_al,  # width  (already 8‑aligned)
                y2_al - y1_al,  # height (already 8‑aligned)
                x1_al, y1_al
            )[0]
            # print(target_shape.shape)torch.Size([1, 768, 1344, 3])
            # print(cropped_image.shape)torch.Size([1, 208, 200, 3])

            # -----------------------------------------------------------------
            # d) Prepare the mask.
            #    • Convert numpy → torch if needed.
            #    • Resize the mask **to a size that is a multiple of 8** (VAE
            #      requirement).
            #    • Then centre‑crop it back to the *exact* width/height of the
            #      aligned rectangle so that the spatial coordinates stay correct.
            # -----------------------------------------------------------------
            cropped_mask = seg.cropped_mask
            if isinstance(cropped_mask, np.ndarray):
                cropped_mask = torch.from_numpy(cropped_mask)

            # Exact width/height we need for compositing
            exact_w = x2_al - x1_al
            exact_h = y2_al - y1_al

            # Rounded size for the VAE‑friendly interpolation
            rounded_w = _ceil8(exact_w)
            rounded_h = _ceil8(exact_h)


            # Interpolate mask to the rounded size (adds the required padding)
            cropped_mask = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
                size=(rounded_h, rounded_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # back to [H,W]



            # If we padded the mask, remove the extra pixels so that mask size
            # exactly matches the image size we will paste.
            if rounded_w != exact_w or rounded_h != exact_h:
                off_x = (rounded_w - exact_w) // 2
                off_y = (rounded_h - exact_h) // 2
                cropped_mask = cropped_mask[
                               off_y:off_y + exact_h,
                               off_x:off_x + exact_w
                               ]

            # Convert the binary mask (no gradients) – same as original code.
            binary_mask = (cropped_mask > 0.01).float()

            # -----------------------------------------------------------------
            # e) Build the three auxiliary masks that are used later for the
            #    different blending steps.
            # -----------------------------------------------------------------
            compositing_mask = cls.grow_and_blur_mask(binary_mask, 64, 4) # 64-4
            inpainting_mask = cls.grow_and_blur_mask(binary_mask, 32, 32) # 32-64
            cropped_mask = cls.grow_and_blur_mask(binary_mask, 24, 48) # 24-48

            # -----------------------------------------------------------------
            # f) Assemble the new SEG.  All coordinates have already been
            #    aligned; nothing will be changed later in the pipeline.
            # -----------------------------------------------------------------
            new_seg = SEG(
                cropped_image=cropped_image,
                cropped_mask=cropped_mask,
                confidence=seg.confidence,
                crop_region=crop_region,  # <-- aligned, never touched again
                bbox=(bx1_al, by1_al, bx2_al, by2_al),
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper,
                inpainting_mask=inpainting_mask,
                compositing_mask=compositing_mask
            )
            new_segs.append(new_seg)

        # -----------------------------------------------------------------
        # Return the size of the up‑scaled image (height, width) *and* the list
        # of new SEG objects.
        # -----------------------------------------------------------------
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

