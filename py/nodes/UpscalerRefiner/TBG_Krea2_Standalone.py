import comfy
import comfy.samplers
import node_helpers
import nodes
import torch
import torch.nn.functional as F

from .inc.TBG_split_aware_lanpaint_sampler import TBG_KSamplerAdvancedSplitAware_Copy
from .inc.vl_encode import (
    apply_krea2_vl_mask,
    build_vl_conditioning,
    encode_tile,
    parse_krea2_layer_weights,
)


def _batch_item(value, index, batch_size):
    if value is None:
        return None
    if value.shape[0] == 1:
        return value[:1]
    if value.shape[0] != batch_size:
        raise ValueError(f"Batch input has {value.shape[0]} items; expected 1 or {batch_size}")
    return value[index:index + 1]


def _mask_bhw(mask, image):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask[:, 0] if mask.shape[1] == 1 else mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"Expected a mask with shape [B,H,W], got {tuple(mask.shape)}")
    if mask.shape[-2:] != image.shape[1:3]:
        mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=image.shape[1:3],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return mask.clamp(0.0, 1.0)


def _encode_inpaint_latent(vae, image, mask):
    compression = int(vae.spacial_compression_encode())
    height = (int(image.shape[1]) // compression) * compression
    width = (int(image.shape[2]) // compression) * compression
    image = image[:, :height, :width, :3]
    mask = _mask_bhw(mask, image)
    if mask is None:
        mask = torch.ones(
            (image.shape[0], image.shape[1], image.shape[2]),
            device=image.device,
            dtype=image.dtype,
        )
    else:
        mask = mask[:, :height, :width]

    masked = image.clone()
    masked = (masked - 0.5) * (1.0 - mask.unsqueeze(-1)) + 0.5
    concat_latent = vae.encode(masked)
    samples = vae.encode(image)
    return samples, concat_latent, mask.unsqueeze(1)


def _crop_vl_reference(image, mask):
    if image is None or mask is None:
        return image
    mask = _mask_bhw(mask, image)
    active = mask[0] > 1.0e-5
    if not bool(active.any()):
        return image
    ys, xs = torch.where(active)
    return image[:, int(ys.min()):int(ys.max()) + 1,
                 int(xs.min()):int(xs.max()) + 1, :]


def _mask_vl_input(image, mask):
    if mask is None:
        return image
    mask = _mask_bhw(mask, image).unsqueeze(-1)
    return image * mask + 0.5 * (1.0 - mask)


def _encode_global_crop(clip, image, mask, strength):
    mask = _mask_bhw(mask, image)
    if mask is None:
        crop_x0, crop_y0 = 0, 0
        crop_w, crop_h = int(image.shape[2]), int(image.shape[1])
    else:
        active = mask[0] > 1.0e-5
        if not bool(active.any()):
            crop_x0, crop_y0 = 0, 0
            crop_w, crop_h = int(image.shape[2]), int(image.shape[1])
        else:
            ys, xs = torch.where(active)
            crop_x0, crop_y0 = int(xs.min()), int(ys.min())
            crop_w = int(xs.max()) - crop_x0 + 1
            crop_h = int(ys.max()) - crop_y0 + 1
    grid_spec = (0, 0, 0, crop_x0, crop_y0, crop_w, crop_h)
    return build_vl_conditioning(
        clip=clip,
        full_image=image,
        grid_specs=[grid_spec],
        canvas_h=int(image.shape[1]),
        canvas_w=int(image.shape[2]),
        model_type="Krea2",
        vl_strength=strength,
    )[0]


def _laplacian_composite(base, source, mask):
    base_bchw = base.permute(0, 3, 1, 2).contiguous()
    source_bchw = source.permute(0, 3, 1, 2).contiguous()
    mask_bchw = mask.unsqueeze(1).to(device=base.device, dtype=base.dtype).clamp(0.0, 1.0)
    base_pyramid = [base_bchw]
    source_pyramid = [source_bchw]
    mask_pyramid = [mask_bchw]
    levels = 4
    for _ in range(levels):
        if min(base_pyramid[-1].shape[-2:]) < 8:
            break
        base_pyramid.append(F.avg_pool2d(base_pyramid[-1], 2, stride=2))
        source_pyramid.append(F.avg_pool2d(source_pyramid[-1], 2, stride=2))
        mask_pyramid.append(F.avg_pool2d(mask_pyramid[-1], 2, stride=2))

    base_laplacian = []
    source_laplacian = []
    for pyramid, laplacian in (
        (base_pyramid, base_laplacian),
        (source_pyramid, source_laplacian),
    ):
        for index in range(len(pyramid) - 1):
            up = F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laplacian.append(pyramid[index] - up)
        laplacian.append(pyramid[-1])

    result = source_laplacian[-1] * mask_pyramid[-1] + base_laplacian[-1] * (1.0 - mask_pyramid[-1])
    for index in range(len(base_laplacian) - 2, -1, -1):
        result = F.interpolate(
            result,
            size=base_laplacian[index].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask_level = mask_pyramid[index]
        result = result + source_laplacian[index] * mask_level + base_laplacian[index] * (1.0 - mask_level)
    return result.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _erode_mask(mask, pixels):
    radius = int(pixels)
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    inverted = 1.0 - mask.unsqueeze(1)
    inverted = F.pad(inverted, (radius, radius, radius, radius), value=1.0)
    return (1.0 - F.max_pool2d(inverted, kernel_size=kernel, stride=1))[:, 0]


def _stitch_reference_image(base, inpaint_mask, reference, reference_mask, mask_shrink_pixels=16):
    if reference is None:
        return base
    target_mask = _mask_bhw(inpaint_mask, base)
    if target_mask is None:
        target_mask = torch.ones(
            (base.shape[0], base.shape[1], base.shape[2]),
            device=base.device,
            dtype=base.dtype,
        )
    # Leave a 16-pixel border between the stitched reference and the outer
    # inpainting boundary.  This changes only the stitch blend mask; the
    # original mask is still used unchanged for VAE inpainting.
    stitch_mask = _erode_mask(target_mask, mask_shrink_pixels)
    active = stitch_mask[0] > 1.0e-5
    if not bool(active.any()):
        return base
    ys, xs = torch.where(active)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    reference = _crop_vl_reference(reference, reference_mask)
    patch = F.interpolate(
        reference.permute(0, 3, 1, 2),
        size=(y1 - y0, x1 - x0),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
    # Keep the original mask, including its gradient, as the only spatial
    # blend mask.  Blurring a cropped bounding box would turn it into a
    # rectangular mask and create a visible square around the stitch.
    source = base.clone()
    source[:, y0:y1, x0:x1, :] = patch
    blended = _laplacian_composite(base, source, stitch_mask)
    return blended * stitch_mask.unsqueeze(-1) + base * (1.0 - stitch_mask.unsqueeze(-1))


def _merge_vl_sources(sources):
    tensors = []
    ranges = []
    offset = 0
    for source in sources:
        for tensor, metadata in source:
            tensors.append(tensor)
            for row_start, row_end in metadata.get("tbg_vl_vision_ranges", ()):
                ranges.append((int(row_start) + offset, int(row_end) + offset))
            offset += int(tensor.shape[1])

    if not tensors:
        return None

    tensor = torch.cat(tensors, dim=1)
    token_mask = torch.zeros((1, tensor.shape[1]), device=tensor.device, dtype=torch.float32)
    for row_start, row_end in ranges:
        token_mask[:, row_start:row_end] = 1.0
    metadata = {
        "tbg_vl_vision_ranges": ranges,
        "tbg_standalone_vl_token_mask": token_mask,
    }
    return [[tensor, metadata]]


def _pad_vl_batch(entries):
    max_length = max(int(entry[0][0].shape[1]) for entry in entries)
    feature_dim = int(entries[0][0][0].shape[-1])
    device = entries[0][0][0].device
    dtype = entries[0][0][0].dtype
    batch = torch.zeros((len(entries), max_length, feature_dim), device=device, dtype=dtype)
    attention = torch.zeros((len(entries), max_length), device=device, dtype=torch.long)
    vision = torch.zeros((len(entries), max_length), device=device, dtype=torch.float32)

    for batch_index, source in enumerate(entries):
        tensor, metadata = source[0]
        length = int(tensor.shape[1])
        batch[batch_index, :length] = tensor[0]
        attention[batch_index, :length] = 1
        token_mask = metadata.get("tbg_standalone_vl_token_mask")
        if token_mask is not None:
            vision[batch_index, :length] = token_mask[0].to(device=device, dtype=torch.float32)

    return [[batch, {
        "attention_mask": attention,
        "tbg_standalone_vl_token_mask": vision,
        "tbg_vl_step_schedule": True,
    }]]


def _expand_conditioning(conditioning, batch_size):
    result = []
    for tensor, metadata in conditioning:
        if tensor.shape[0] == batch_size:
            expanded = tensor
        elif tensor.shape[0] == 1:
            expanded = tensor.expand(batch_size, *tensor.shape[1:])
        else:
            raise ValueError("Text conditioning batch does not match the image batch")
        result.append([expanded, dict(metadata)])
    return result


def _apply_text_rebalance(conditioning, layer_weights, multiplier, mode, crossover, overlap):
    if layer_weights is None:
        return conditioning
    effective_multiplier = 1.0 if mode == "OLD" else float(multiplier)
    rebalanced = apply_krea2_vl_mask(
        conditioning, None, 1.0, layer_weights, effective_multiplier, True
    )
    if mode != "GATED REFERENCE":
        return rebalanced

    crossover = max(0.0, min(1.0, float(crossover)))
    overlap = max(0.0, min(0.5, float(overlap)))
    return (
        node_helpers.conditioning_set_values(
            rebalanced,
            {"start_percent": 0.0, "end_percent": min(1.0, crossover + overlap)},
        )
        + node_helpers.conditioning_set_values(
            conditioning,
            {"start_percent": max(0.0, crossover - overlap), "end_percent": 1.0},
        )
    )


class TBG_Krea2InpaintingConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vl_mode": ([
                    "Off",
                    "Full Input Image",
                    "Input Mask Area",
                    "Input Mask Area + Input Image Global",
                    "Input Mask Area + Cropped Input Image Global",
                    "Reference Mask Area",
                    "Reference Mask Area + Input Image Global",
                    "Reference Mask Area + Cropped Input Image Global",
                    "Input Mask Area + Reference Mask Area",
                ], {
                    "default": "Reference Mask Area + Cropped Input Image Global",
                    "label": "VL Source Mode",
                    "tooltip": "Global encodes the full input image for complete composition and scene context. Cropped Global also starts from the full input image, preserving its semantic features, colors, and composition, but keeps explicit visual tokens only from the active mask area for stronger local detail; this is useful for img2img refinement. Mask Area uses full-size gradient pixel masking with neutral gray outside the mask.",
                }),
                "input_image_vl_strength": ("FLOAT", {"label": "Input Image VL Strength", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "inpaint_mask_vl_strength": ("FLOAT", {"label": "Inpaint Mask VL Strength", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image_vl_strength": ("FLOAT", {"label": "Reference Image VL Strength", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_mask_vl_strength": ("FLOAT", {"label": "Reference Mask VL Strength", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image_stitch": ("BOOLEAN", {
                    "label": "Reference Image Stitch",
                    "default": False,
                    "label_on": "ON",
                    "label_off": "OFF",
                    "tooltip": "When enabled, crops the reference mask area, scales it to fill the input inpaint area, and inserts it with a feathered multi-scale Laplacian composite before VAE encoding and VL conditioning.",
                }),
                "stitch_mask_shrink": ("INT", {
                    "label": "Stitch Mask Shrink (px)",
                    "default": 16,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Erodes the inpainting mask before reference stitching. 16 means the stitch stops about 16 pixels inside the original mask boundary. The original inpainting mask remains unchanged.",
                }),
                "vl_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vl_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "layer_weights": ("STRING", {"default": "1,1,1,1,1,1,1,1,1,1,1,1"}),
                "layer_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "inpaint_mask": ("MASK", {
                    "label": "Inpaint Mask (optional)",
                    "tooltip": "Optional gradient inpaint mask. If disconnected, the full input image is used as the active area.",
                }),
                "reference_image_1": ("IMAGE",),
                "reference_image_1_mask": ("MASK", {
                    "label": "Reference Mask (optional)",
                    "tooltip": "Optional reference crop mask. If disconnected, the complete reference image is encoded.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "stitched_image")
    FUNCTION = "encode"
    CATEGORY = "TBG/Krea2 Inpainting"

    def encode(self, clip, vae, image, positive_prompt, negative_prompt, inpaint_mask,
               vl_mode, input_image_vl_strength, inpaint_mask_vl_strength,
               reference_image_vl_strength, reference_mask_vl_strength,
               reference_image_stitch, stitch_mask_shrink,
               vl_start_percent, vl_end_percent,
               layer_weights, layer_multiplier, reference_image_1=None,
               reference_image_1_mask=None):
        batch_size = int(image.shape[0])
        layer_values = parse_krea2_layer_weights(layer_weights)
        start_percent = max(0.0, min(1.0, float(vl_start_percent)))
        end_percent = max(0.0, min(1.0, float(vl_end_percent)))
        if end_percent < start_percent:
            start_percent, end_percent = end_percent, start_percent

        text_positive = _expand_conditioning(
            nodes.CLIPTextEncode().encode(clip, positive_prompt)[0], batch_size
        )
        text_negative = _expand_conditioning(
            nodes.CLIPTextEncode().encode(clip, negative_prompt)[0], batch_size
        )
        text_positive = _apply_text_rebalance(
            text_positive, layer_values, layer_multiplier, "GATED REFERENCE", 0.4, 0.1,
        )

        latent_items = []
        vl_items = []
        stitched_images = []
        for batch_index in range(batch_size):
            base = image[batch_index:batch_index + 1]
            base_mask = _batch_item(inpaint_mask, batch_index, batch_size)
            reference = _batch_item(reference_image_1, batch_index, batch_size)
            reference_mask = _batch_item(reference_image_1_mask, batch_index, batch_size)
            if reference_image_stitch and reference is not None:
                base = _stitch_reference_image(
                    base, base_mask, reference, reference_mask, stitch_mask_shrink
                )
            stitched_images.append(base)
            samples, concat_latent, noise_mask = _encode_inpaint_latent(vae, base, base_mask)
            latent_items.append((samples, concat_latent, noise_mask))

            if vl_mode == "Off":
                vl_items.append(None)
                continue

            input_masked = _mask_vl_input(base, base_mask)
            if vl_mode == "Full Input Image":
                sources = [encode_tile(
                    clip, base, "Krea2", vl_strength=input_image_vl_strength
                )]
            elif vl_mode == "Input Mask Area":
                sources = [encode_tile(
                    clip, input_masked, "Krea2", vl_strength=inpaint_mask_vl_strength
                )]
            elif vl_mode == "Input Mask Area + Input Image Global":
                sources = [
                    encode_tile(clip, input_masked, "Krea2", vl_strength=inpaint_mask_vl_strength),
                    encode_tile(clip, base, "Krea2", vl_strength=input_image_vl_strength),
                ]
            elif vl_mode == "Input Mask Area + Cropped Input Image Global":
                sources = [
                    encode_tile(clip, input_masked, "Krea2", vl_strength=inpaint_mask_vl_strength),
                    _encode_global_crop(clip, base, base_mask, input_image_vl_strength),
                ]
            elif vl_mode in (
                "Reference Mask Area",
                "Reference Mask Area + Input Image Global",
                "Reference Mask Area + Cropped Input Image Global",
                "Input Mask Area + Reference Mask Area",
            ) and reference_image_1 is not None:
                reference = _batch_item(reference_image_1, batch_index, batch_size)
                reference_mask = _batch_item(reference_image_1_mask, batch_index, batch_size)
                if reference_mask is None:
                    reference_source = reference
                    reference_strength = reference_image_vl_strength
                else:
                    reference_source = _crop_vl_reference(reference, reference_mask)
                    reference_strength = reference_mask_vl_strength
                reference_conditioning = encode_tile(
                    clip, reference_source, "Krea2", vl_strength=reference_strength
                )
                if vl_mode in (
                    "Reference Mask Area + Input Image Global",
                    "Reference Mask Area + Cropped Input Image Global",
                ):
                    global_source = base
                    if vl_mode == "Reference Mask Area + Cropped Input Image Global":
                        global_conditioning = _encode_global_crop(
                            clip, base, base_mask, input_image_vl_strength
                        )
                    else:
                        global_conditioning = encode_tile(
                            clip, global_source, "Krea2", vl_strength=input_image_vl_strength
                        )
                    sources = [
                        reference_conditioning,
                        global_conditioning,
                    ]
                elif vl_mode == "Input Mask Area + Reference Mask Area":
                    sources = [
                        encode_tile(clip, input_masked, "Krea2", vl_strength=inpaint_mask_vl_strength),
                        reference_conditioning,
                    ]
                else:
                    sources = [reference_conditioning]
            elif vl_mode == "Input Mask Area + Reference Mask Area":
                sources = [encode_tile(
                    clip, input_masked, "Krea2", vl_strength=inpaint_mask_vl_strength
                )]
            elif vl_mode == "Reference Mask Area + Cropped Input Image Global":
                sources = [_encode_global_crop(
                    clip, base, base_mask, input_image_vl_strength
                )]
            else:
                sources = [encode_tile(
                    clip, base, "Krea2", vl_strength=input_image_vl_strength
                )]
            vl_items.append(_merge_vl_sources(sources))

        vl_positive = [] if vl_mode == "Off" else _pad_vl_batch(vl_items)
        if vl_positive:
            vl_positive[0][1]["tbg_vl_start_percent"] = start_percent
            vl_positive[0][1]["tbg_vl_end_percent"] = end_percent

        samples = torch.cat([item[0] for item in latent_items], dim=0)
        concat_latent = torch.cat([item[1] for item in latent_items], dim=0)
        noise_mask = torch.cat([item[2] for item in latent_items], dim=0)
        stitched_image = torch.cat(stitched_images, dim=0)
        inpaint_values = {
            "concat_latent_image": concat_latent,
            "concat_mask": noise_mask,
        }
        positive = node_helpers.conditioning_set_values(vl_positive + text_positive, inpaint_values)
        negative = node_helpers.conditioning_set_values(text_negative, inpaint_values)
        latent = {"samples": samples, "noise_mask": noise_mask}
        return (positive, negative, latent, stitched_image)


def _install_standalone_vl_schedule(model, start_percent, end_percent):
    patched = model.clone()
    patched.model_options = patched.model_options.copy()
    previous = patched.model_options.get("sampler_calc_cond_batch_function")
    state = {"first_sigma": None}

    def scale_conditions(conditions, factor):
        result = []
        for branch in conditions:
            if branch is None:
                result.append(None)
                continue
            if not isinstance(branch, (list, tuple)):
                result.append(branch)
                continue
            scaled_branch = []
            for entry in branch:
                if isinstance(entry, dict):
                    metadata = entry
                    token_mask = metadata.get("tbg_standalone_vl_token_mask")
                    model_conds = metadata.get("model_conds", {})
                    if token_mask is None:
                        scaled_branch.append(entry)
                        continue
                    updated = dict(metadata)
                    updated_conds = dict(model_conds)
                    for key, cond_obj in model_conds.items():
                        tensor = getattr(cond_obj, "cond", None)
                        if not torch.is_tensor(tensor):
                            continue
                        mask = token_mask.to(device=tensor.device, dtype=tensor.dtype)
                        if mask.shape[0] == 1 and tensor.shape[0] > 1:
                            mask = mask.expand(tensor.shape[0], -1)
                        gain = 1.0 + mask[:, :tensor.shape[1]].unsqueeze(-1) * (factor - 1.0)
                        updated_conds[key] = cond_obj._copy_with(tensor * gain)
                    updated["model_conds"] = updated_conds
                    scaled_branch.append(updated)
                    continue
                tensor, metadata = entry[0], entry[1]
                token_mask = metadata.get("tbg_standalone_vl_token_mask")
                if token_mask is None:
                    scaled_branch.append(entry)
                    continue
                mask = token_mask.to(device=tensor.device, dtype=tensor.dtype)
                if mask.shape[0] == 1 and tensor.shape[0] > 1:
                    mask = mask.expand(tensor.shape[0], -1)
                gain = 1.0 + mask[:, :tensor.shape[1]].unsqueeze(-1) * (factor - 1.0)
                scaled_branch.append([tensor * gain, metadata])
            result.append(scaled_branch)
        return result

    def calc_cond_batch(args):
        sigma = args["sigma"]
        sigma_value = float(sigma.detach().flatten()[0].cpu())
        if state["first_sigma"] is None:
            state["first_sigma"] = max(abs(sigma_value), 1.0e-6)
        progress = 1.0 - max(0.0, min(1.0, abs(sigma_value) / state["first_sigma"]))
        start = float(args["model_options"].get("tbg_standalone_vl_start_percent", 0.0))
        end = float(args["model_options"].get("tbg_standalone_vl_end_percent", 1.0))
        factor = 1.0 if start <= progress <= end else 0.0
        nested = dict(args)
        nested["conds"] = scale_conditions(args["conds"], factor)
        nested_options = args["model_options"].copy()
        nested_options.pop("sampler_calc_cond_batch_function", None)
        nested["model_options"] = nested_options
        if previous is not None:
            return previous(nested)
        return comfy.samplers.calc_cond_batch(
            args["model"], nested["conds"], args["input"], args["sigma"], nested_options
        )

    patched.model_options["tbg_standalone_vl_start_percent"] = float(start_percent)
    patched.model_options["tbg_standalone_vl_end_percent"] = float(end_percent)
    patched.model_options["sampler_calc_cond_batch_function"] = calc_cond_batch
    return patched


class TBG_SplitAwareInpaintSampler(TBG_KSamplerAdvancedSplitAware_Copy):
    CATEGORY = "TBG/Krea2 Inpainting"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["optional"] = dict(inputs.get("optional", {}))
        inputs["optional"]["Sampler"] = (
            "SAMPLER",
            {
                "label": "Sampler Override",
                "tooltip": "Optional sampler object. When connected, it overrides the Sampler Name selection.",
            },
        )
        return inputs

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step, denoise,
               return_with_leftover_noise, inpaint_end, inpaint_start,
               smoother_sharper, detail_enhancer, sampler_state=None, Sampler=None):
        if Sampler is not None:
            sampler_name = Sampler
        start_percent = 0.0
        end_percent = 1.0
        for _tensor, metadata in positive:
            if not isinstance(metadata, dict):
                continue
            start_percent = metadata.get("tbg_vl_start_percent", start_percent)
            end_percent = metadata.get("tbg_vl_end_percent", end_percent)
            break
        model = _install_standalone_vl_schedule(
            model, start_percent, end_percent
        )
        return super().sample(
            model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, start_at_step, end_at_step, denoise,
            return_with_leftover_noise, inpaint_end, inpaint_start,
            smoother_sharper, detail_enhancer, sampler_state,
        )
