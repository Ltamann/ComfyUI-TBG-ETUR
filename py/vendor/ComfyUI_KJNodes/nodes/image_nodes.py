
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
