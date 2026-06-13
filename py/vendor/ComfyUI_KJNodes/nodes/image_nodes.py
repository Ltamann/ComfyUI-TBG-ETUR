
import os
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

class ColorMatch:

    def colormatch(self, image_ref, image_target, method, strength=1.0):

        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            # Apply the strength multiplier
            image_result = image_target_np + strength * (image_result - image_target_np)
            out.append(torch.from_numpy(image_result))
            
        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


class ETURColorMatchV2:
    """ETUR-owned internal ColorMatch V2 clone; not registered as a ComfyUI node."""

    METHODS = ('mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm', 'reinhard_lab_gpu')

    def colormatch(self, image_ref, image_target, method, strength=1.0, multithread=True):
        if strength == 0:
            return (image_target,)
        if method not in self.METHODS:
            raise ValueError(f"ETURColorMatchV2 unsupported method: {method}")

        if method == "reinhard_lab_gpu":
            try:
                import kornia
                from comfy import model_management
            except ImportError as e:
                raise ImportError("ETURColorMatchV2 reinhard_lab_gpu requires kornia") from e

            device = model_management.get_torch_device()
            target_device = image_target.device
            image_target_device = image_target.to(device=device, dtype=torch.float32)
            image_ref_device = image_ref.to(device=device, dtype=torch.float32)
            batch_size, height, width, channels = image_target_device.shape
            if channels != 3:
                image_target_device = image_target_device[..., :3]
                image_ref_device = image_ref_device[..., :3]

            src_bchw = image_target_device.permute(0, 3, 1, 2).contiguous()
            ref_bchw = image_ref_device.permute(0, 3, 1, 2).contiguous()
            src_lab = kornia.color.rgb_to_lab(src_bchw)
            ref_lab = kornia.color.rgb_to_lab(ref_bchw)

            src_flat = src_lab.view(batch_size, 3, -1)
            ref_flat = ref_lab.view(ref_lab.shape[0], 3, -1)
            src_std, src_mean = torch.std_mean(src_flat, dim=-1, keepdim=True, unbiased=False)
            ref_std, ref_mean = torch.std_mean(ref_flat, dim=-1, keepdim=True, unbiased=False)
            src_std = src_std.clamp_min_(1e-6)
            if ref_lab.shape[0] == 1 and batch_size > 1:
                ref_mean = ref_mean.expand(batch_size, -1, -1)
                ref_std = ref_std.expand(batch_size, -1, -1)

            corrected_lab = ((src_flat - src_mean) * (ref_std / src_std) + ref_mean).view(batch_size, 3, height, width)
            corrected_rgb = kornia.color.lab_to_rgb(corrected_lab)
            out = (1.0 - float(strength)) * src_bchw + float(strength) * corrected_rgb
            out = out.permute(0, 2, 3, 1).contiguous().to(target_device).float().clamp_(0, 1)
            return (out,)

        try:
            from color_matcher import ColorMatcher
        except ImportError as e:
            raise ImportError(
                "Can't import color-matcher, did you install requirements.txt? "
                "Manual install: pip install color-matcher"
            ) from e

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        ref_batch_size = image_ref.size(0)

        def process(i):
            cm = ColorMatcher()
            image_target_np = image_target[i].numpy()
            image_ref_np = image_ref[min(i, ref_batch_size - 1)].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np, method=method)
                if strength != 1:
                    image_result = image_target_np + strength * (image_result - image_target_np)
                return torch.from_numpy(image_result)
            except Exception as e:
                logging.error(f"ETURColorMatchV2 thread {i} error: {e}")
                return torch.from_numpy(image_target_np)

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)

    def colormatch_with_mask(self, image_ref, image_target, method, mask, strength=1.0):
        print("TBG colormatch")

        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()

        batch_size = image_target.size(0)
        out = []

        # Check batch/ref sizes
        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            # Extract individual images in CHW, convert to HWC numpy
            image_target_i = image_target[i] if batch_size > 1 else image_target[0]
            image_ref_i = image_ref[i] if image_ref.size(0) > 1 else image_ref[0]

            image_target_np = image_target_i.permute(1, 2, 0).numpy()  # HWC numpy float32
            image_ref_np = image_ref_i.permute(1, 2, 0).numpy()

            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break

            image_result = image_target_np + strength * (image_result - image_target_np)

            image_result_tensor = torch.from_numpy(image_result).float()  # HWC tensor
            image_target_tensor = torch.from_numpy(image_target_np).float()

            # Process mask_i similarly: (H, W) or (1, H, W) etc.
            mask_i = mask if batch_size == 1 else mask[i]
            if mask_i.ndim == 4:
                mask_i = mask_i.squeeze(0).squeeze(-1)
            elif mask_i.ndim == 3 and mask_i.shape[0] == 1:
                mask_i = mask_i.squeeze(0)

            if mask_i.max() > 1.0:
                mask_i = mask_i / 255.0

            mask_hwc = mask_i.unsqueeze(-1).expand(-1, -1, 3)

            blended = image_target_tensor * (1 - mask_hwc) + image_result_tensor * mask_hwc

            # Convert to CHW for output
            blended_chw = blended.permute(2, 0, 1)

            out.append(blended_chw)

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)

        return (out,)
