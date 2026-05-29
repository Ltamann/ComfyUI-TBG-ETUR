import comfy.utils
import copy
import math
import nodes
import numpy as np
import torch
import torch.nn.functional as F


from ..inc.sigmas import process_image_to_tiles
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.canny.canny import CannyDetector
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything_v2.da2tgb import DepthAnythingV2Detector
from ....vendor.comfyui_controlnet_aux.utils import common_annotator_call
from ....utils.log import log



import comfy.model_management as model_management


def normalize_controlnet_mode(control):
    """
    Backward/forward-compatible mode resolver used by vram_optimizing.
    Accepts dict-like and object-like control pipe entries.
    """
    if control is None:
        return "ControlNet"

    mode = None
    if isinstance(control, dict):
        mode = control.get("controlnet_mode")
        if mode is None:
            mode = control.get("mode")
        if mode is None:
            mode = control.get("type")
        if mode is None:
            mode = control.get("patch_for_Flux_Kontext")
    else:
        mode = getattr(control, "controlnet_mode", None)
        if mode is None:
            mode = getattr(control, "mode", None)
        if mode is None:
            mode = getattr(control, "type", None)
        if mode is None:
            mode = getattr(control, "patch_for_Flux_Kontext", None)

    if isinstance(mode, str):
        m = mode.strip().lower()
        if m in ("model_patch", "model patch", "patch"):
            return "Model_Patch"
        if m in ("reference_image", "reference image", "reference"):
            return "Reference_Image"
        if m in ("controlnet", "control net", "", "none"):
            return "ControlNet"

    return "ControlNet"


def _control_value(control, key, default=None):
    if isinstance(control, dict):
        return control.get(key, default)
    return getattr(control, key, default)


def _resolve_kontext_patch_mode(control):
    """
    Resolve the legacy Flux Kontext patch mode without crashing on V3 pipe data.
    Newer V3 workflows store `controlnet_mode` instead of
    `patch_for_Flux_Kontext`. The Kontext path only understands the old
    Stitched/Chained modes, so `Reference_Image` falls back to `Chained`.
    """
    legacy_mode = _control_value(control, "patch_for_Flux_Kontext")
    if isinstance(legacy_mode, str) and legacy_mode:
        return legacy_mode
    if normalize_controlnet_mode(control) == "Reference_Image":
        return "Chained"
    return "NONE"


def _match_reference_latent_shape(reference_latent, denoised):
    """
    Match the reference latent to the current denoised latent shape.

    This mirrors the upstream Identity Guidance safeguards:
    - align batch size
    - resize spatial dimensions
    - align channels by trimming/padding
    """
    ref = reference_latent.to(device=denoised.device, dtype=denoised.dtype)

    if ref.shape[0] != denoised.shape[0]:
        ref = ref[:1].expand(denoised.shape[0], -1, -1, -1)

    if ref.shape[2:] != denoised.shape[2:]:
        ref = F.interpolate(
            ref,
            size=denoised.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    if ref.shape[1] != denoised.shape[1]:
        if ref.shape[1] > denoised.shape[1]:
            ref = ref[:, :denoised.shape[1]]
        else:
            ref = F.pad(ref, (0, 0, 0, 0, 0, denoised.shape[1] - ref.shape[1]))

    return ref


def _sanitize_reference_window(start_percent, end_percent):
    start = float(max(0.0, min(1.0, start_percent)))
    end = float(max(0.0, min(1.0, end_percent)))
    if start >= end:
        return None
    return start, end


def _is_full_reference_conditioning(control):
    """
    Only build the normal ReferenceLatent conditioning path for the
    "locked" case: full strength and full denoise window.

    We keep weaker settings on the sampler-side correction path only so the
    Strength slider behaves more like a true creativity slider.
    """
    if normalize_controlnet_mode(control) != "Reference_Image":
        return False

    strength = float(_control_value(control, "strength", 0.0))
    start = float(_control_value(control, "start", 0.0))
    end = float(_control_value(control, "end", 1.0))
    return strength >= 0.999 and start <= 0.001 and end >= 0.999


def _reference_dev_log(self, message):
    api = getattr(self, "API", None)
    if getattr(api, "status", None) == "Dev":
        log(message, None, None, f"Node {getattr(getattr(self, 'INFO', None), 'id', '?')}")


def _reference_window_weight(progress, start_percent, end_percent):
    if progress < start_percent or progress > end_percent:
        return 0.0

    span = max(1e-6, end_percent - start_percent)
    t = (progress - start_percent) / span

    # Raised-sine envelope: avoids hard repeated pulls at the window edges,
    # which can cause reference ghosting in high-denoise refinement.
    return math.sin(math.pi * t)


def _reference_schedule_progress(sigma, args):
    model_options = args.get("model_options", {}) or {}
    transformer_options = model_options.get("transformer_options", {}) or {}
    sample_sigmas = transformer_options.get("sample_sigmas")

    sigma_value = float(sigma.flatten()[0])

    if isinstance(sample_sigmas, torch.Tensor) and sample_sigmas.numel() > 1:
        schedule = sample_sigmas.flatten().to(device=sigma.device, dtype=sigma.dtype)
        idx = int(torch.argmin(torch.abs(schedule - sigma.flatten()[0])).item())
        progress = idx / max(1, schedule.numel() - 1)
        return max(0.0, min(1.0, progress)), idx, int(schedule.numel())

    progress = max(0.0, min(1.0, 1.0 - sigma_value))
    return progress, None, None


def _reference_post_cfg_fn(reference_latent, strength, start_percent, end_percent, debug_owner=None):
    """
    ControlNet-style scheduling for reference influence.

    This is *not* a real ControlNet model. We add a small scheduled nudge
    toward a stored reference latent after each sampling step.
    """

    window = _sanitize_reference_window(start_percent, end_percent)
    if window is None or reference_latent is None or strength <= 0.0:
        return None

    start_percent, end_percent = window

    def post_cfg_fn(args):
        denoised = args["denoised"]
        sigma = args["sigma"]

        # Use the sampler's actual sigma schedule when available. This keeps
        # Start/End closer to ControlNet-style step percentages.
        progress, sigma_index, sigma_count = _reference_schedule_progress(sigma, args)
        if progress < start_percent or progress > end_percent:
            return denoised

        ref_resized = _match_reference_latent_shape(reference_latent, denoised)
        correction = ref_resized - denoised
        out = denoised + correction * float(strength)

        # Dev-only trace so we can see the hook is live and how strong the
        # correction is on the current step.
        _reference_dev_log(
            self=debug_owner,
            message=(
                f"Reference Image active: progress={progress:.4f} "
                f"window={start_percent:.2f}-{end_percent:.2f} "
                f"strength={strength:.3f} "
                f"sigma_index={sigma_index}/{sigma_count} "
                f"mean_abs_delta={torch.mean(torch.abs(out - denoised)).item():.6f}"
            ),
        )

        return out

    return post_cfg_fn


def stitch(
        self,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
):
    if image2 is None:
        return (image1,)

    # Handle batch size differences
    if image1.shape[0] != image2.shape[0]:
        max_batch = max(image1.shape[0], image2.shape[0])
        if image1.shape[0] < max_batch:
            image1 = torch.cat(
                [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
            )
        if image2.shape[0] < max_batch:
            image2 = torch.cat(
                [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
            )

    # Match image sizes if requested
    if match_image_size:
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        aspect_ratio = w2 / h2

        if direction in ["left", "right"]:
            target_h, target_w = h1, int(h1 * aspect_ratio)
        else:  # up, down
            target_w, target_h = w1, int(w1 / aspect_ratio)

        image2 = comfy.utils.common_upscale(
            image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
        ).movedim(1, -1)

    color_map = {
        "white": 1.0,
        "black": 0.0,
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }

    color_val = color_map[spacing_color]

    # When not matching sizes, pad to align non-concat dimensions
    if not match_image_size:
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        pad_value = 0.0
        if not isinstance(color_val, tuple):
            pad_value = color_val

        if direction in ["left", "right"]:
            # For horizontal concat, pad heights to match
            if h1 != h2:
                target_h = max(h1, h2)
                if h1 < target_h:
                    pad_h = target_h - h1
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                if h2 < target_h:
                    pad_h = target_h - h2
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
        else:  # up, down
            # For vertical concat, pad widths to match
            if w1 != w2:
                target_w = max(w1, w2)
                if w1 < target_w:
                    pad_w = target_w - w1
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                if w2 < target_w:
                    pad_w = target_w - w2
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

    # Ensure same number of channels
    if image1.shape[-1] != image2.shape[-1]:
        max_channels = max(image1.shape[-1], image2.shape[-1])
        if image1.shape[-1] < max_channels:
            image1 = torch.cat(
                [
                    image1,
                    torch.ones(
                        *image1.shape[:-1],
                        max_channels - image1.shape[-1],
                        device=image1.device,
                    ),
                ],
                dim=-1,
            )
        if image2.shape[-1] < max_channels:
            image2 = torch.cat(
                [
                    image2,
                    torch.ones(
                        *image2.shape[:-1],
                        max_channels - image2.shape[-1],
                        device=image2.device,
                    ),
                ],
                dim=-1,
            )

    # Add spacing if specified
    if spacing_width > 0:
        spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

        if direction in ["left", "right"]:
            spacing_shape = (
                image1.shape[0],
                max(image1.shape[1], image2.shape[1]),
                spacing_width,
                image1.shape[-1],
            )
        else:
            spacing_shape = (
                image1.shape[0],
                spacing_width,
                max(image1.shape[2], image2.shape[2]),
                image1.shape[-1],
            )

        spacing = torch.full(spacing_shape, 0.0, device=image1.device)
        if isinstance(color_val, tuple):
            for i, c in enumerate(color_val):
                if i < spacing.shape[-1]:
                    spacing[..., i] = c
            if spacing.shape[-1] == 4:  # Add alpha
                spacing[..., 3] = 1.0
        else:
            spacing[..., : min(3, spacing.shape[-1])] = color_val
            if spacing.shape[-1] == 4:
                spacing[..., 3] = 1.0

    # Concatenate images
    images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
    if spacing_width > 0:
        images.insert(1, spacing)

    concat_dim = 2 if direction in ["left", "right"] else 1
    return (torch.cat(images, dim=concat_dim),)


# V3 scheme name changes

ImageStitch_execute = stitch


from comfy_extras.nodes_edit_model import ReferenceLatent
if hasattr(ReferenceLatent, "append"):
    ReferenceLatent_execute = ReferenceLatent.append
elif hasattr(ReferenceLatent, "execute"):
    ReferenceLatent_execute = ReferenceLatent.execute


def append_reference_latent(conditioning, latent):
    try:
        return ReferenceLatent_execute(conditioning, latent)[0]
    except TypeError:
        return ReferenceLatent_execute(0, conditioning, latent)[0]



def apply_controlnets_from_pipe(self,SELF, cnetpipe, positive, negative, full_image, tile_image, vae, index):
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
            grid_images = process_image_to_tiles(self, noise_image)
            cnet_image = grid_images[self.TEMP.latent_index]
            if isinstance(cnet_image, tuple):
                cnet_image = np.array(cnet_image)

        grid_cnetstrength = self.PROMPTER.output_cnet_js[index]
        if grid_cnetstrength is not None and isinstance(grid_cnetstrength, (float, int)):
            strength = grid_cnetstrength  * self.KSAMPLER.cnet_multiply
        else:

            strength = strength * self.KSAMPLER.cnet_multiply

        # Preprocessero

        if preprocessor == "DepthAnythingV2":
            model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
            cnet_image = common_annotator_call(model, cnet_image, resolution=1024, max_depth=1)
            del model
        if preprocessor == "Canny Edge":
            cnet_image = common_annotator_call(CannyDetector(), cnet_image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=1024)
        positive, negative = controlnet_node.apply_controlnet(
            positive, negative, controlnet_model, cnet_image, strength, start, end, vae
        )
        if preprocessor == "ControlNetInpaintingAliMama":
            from .image import TBG_Image
            inpaintmask = \
                TBG_Image().mask_get_fusion_mask(SELF,
                                                 cnet_image.shape[1],
                                                 cnet_image.shape[2],
                                                 SELF.SIZE.cols_qty,
                                                 SELF.SIZE.rows_qty,
                                                 SELF.PARAMS.grid_specs[index],
                                                 SELF.SIZE.inpaint_blur_margin,
                                                 SELF.SIZE.shift,
                                                 SELF.SIZE.shifttl,
                                                 SELF.SIZE.inpaint_border_margin,
                                                 1,
                                                 SELF.SIZE.inpaint_max)[0]

            extra_concat = []
            if controlnet_model.concat_mask:
                mask = 1.0 - inpaintmask.reshape((-1, 1, inpaintmask.shape[-2], inpaintmask.shape[-1]))
                mask_apply = comfy.utils.common_upscale(mask, cnet_image.shape[2], cnet_image.shape[1], "bilinear", "center").round()
                image = cnet_image * mask_apply.movedim(1, -1).repeat(1, 1, 1, cnet_image.shape[3])
                extra_concat = [mask]

            positive, negative  = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, controlnet_model, image, strength, start, end, vae=vae, extra_concat=extra_concat)



        #if self.lowvram:
        #    UnloadOneModelNode.route(controlnet_model)

    return positive, negative

def get_canny_mask_inverted(image,canny_low_threshold=100,canny_high_threshold=150):
    resolution = min(image.shape[1],image.shape[2])
    mask = common_annotator_call(CannyDetector(), image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=resolution)
    out = 1.0 - mask
    return out

def _get_reference_source_image(self, control, tile_image, require_explicit=False):
    reference_image = None if require_explicit else tile_image
    source_image = _control_value(control, "noise_image")
    if source_image is None:
        source_image = _control_value(control, "custom_controlnet_image")
    if source_image is not None:
        grid_images = process_image_to_tiles(self, source_image)
        reference_image = grid_images[self.TEMP.latent_index]
        if isinstance(reference_image, tuple):
            reference_image = np.array(reference_image)
    return reference_image


def _build_reference_image_for_control(self, control, tile_image):
    patch_for_flux_kontext = _resolve_kontext_patch_mode(control)
    if patch_for_flux_kontext == "NONE":
        return None, None

    reference_image = _get_reference_source_image(self, control, tile_image)
    preprocessor = _control_value(control, "preprocessor", "None")
    canny_high_threshold = _control_value(control, "canny_high_threshold", 150)
    canny_low_threshold = _control_value(control, "canny_low_threshold", 100)
    strength = _control_value(control, "strength", 1.0) * self.KSAMPLER.cnet_multiply

    if preprocessor == "DepthAnythingV2":
        model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
        reference_image = common_annotator_call(model, reference_image, resolution=1024, max_depth=1)
        reference_image = reference_image * strength + 0.5 * (1 - strength)
        del model
    elif preprocessor in ("Canny Edge", "Canny"):
        reference_image = common_annotator_call(CannyDetector(), reference_image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=1024)
        reference_image = reference_image * strength + 0.5 * (1 - strength)
    elif preprocessor != "None":
        # Unsupported preprocessors do not emit a reference image.
        return patch_for_flux_kontext, None

    return patch_for_flux_kontext, reference_image


def _append_reference_images(self, positive, reference_images):
    for reference_image in reference_images:
        reference_latent = nodes.VAEEncode().encode(self.KSAMPLER.vae, reference_image)[0]
        positive = append_reference_latent(positive, reference_latent)
    return positive


def _stitch_reference_images(reference_images):
    if not reference_images:
        return None

    combined_image = reference_images[0]
    for reference_image in reference_images[1:]:
        combined_image = ImageStitch_execute(
            0,
            combined_image,
            'right',
            True,
            0,
            'white',
            reference_image,
        )[0]
    return combined_image


def _build_reference_conditioning(self, positive, cnetpipe, tile_image):
    chained_reference_images = []
    stitched_reference_images = []

    for control in cnetpipe:
        if not _is_full_reference_conditioning(control):
            _reference_dev_log(
                self,
                "Reference Image conditioning path skipped: using sampler-only correction because strength/window is not full."
            )
            # We intentionally skip the normal conditioning path for weaker
            # settings. Those cases rely on the sampler-side correction only.
            continue
        _reference_dev_log(
            self,
            "Reference Image conditioning path enabled: full strength and full window."
        )
        patch_for_flux_kontext, reference_image = _build_reference_image_for_control(self, control, tile_image)
        if reference_image is None:
            continue

        if patch_for_flux_kontext == "Chained":
            chained_reference_images.append(reference_image)
        elif patch_for_flux_kontext == "Stitched":
            stitched_reference_images.append(reference_image)

    if chained_reference_images:
        positive = _append_reference_images(self, positive, chained_reference_images)

    stitched_reference_image = _stitch_reference_images(stitched_reference_images)
    if stitched_reference_image is not None:
        positive = _append_reference_images(self, positive, [stitched_reference_image])

    return positive


def apply_reference_mode_hooks(self, model, cnetpipe, tile_image, vae):
    """
    Attach FLUX2Klein-style reference correction hooks to a cloned model.

    The hook operates after CFG prediction in the sampling loop, which gives
    the familiar ControlNet-like "Strength / Start % / End %" UX without
    loading a real ControlNet conditioning model.
    """
    reference_controls = []

    for control in cnetpipe or []:
        if normalize_controlnet_mode(control) != "Reference_Image":
            continue

        # If no external image is provided, fall back to the current tile.
        # That keeps the UX consistent with the existing Reference_Image path.
        reference_image = _get_reference_source_image(self, control, tile_image, require_explicit=False)
        if reference_image is None or vae is None:
            continue

        # Reuse the same ControlNet sliders the user already understands.
        reference_strength = float(_control_value(control, "strength", 0.5))
        reference_start_percent = float(_control_value(control, "start", 0.0))
        reference_end_percent = float(_control_value(control, "end", 0.5))

        window = _sanitize_reference_window(reference_start_percent, reference_end_percent)
        if window is None or reference_strength <= 0.0:
            continue

        reference_latent = nodes.VAEEncode().encode(vae, reference_image)[0]["samples"]
        reference_controls.append((reference_latent, reference_strength, window[0], window[1]))

    if not reference_controls:
        _reference_dev_log(self, "Reference Image inactive: no Reference_Image entries or strength/window disabled.")
        return model

    _reference_dev_log(
        self,
        f"Reference Image hook enabled: {len(reference_controls)} control(s) attached."
    )

    hooked_model = model.clone()
    post_cfg_functions = hooked_model.model_options.setdefault("sampler_post_cfg_function", [])

    for reference_latent, reference_strength, reference_start_percent, reference_end_percent in reference_controls:
        post_cfg_fn = _reference_post_cfg_fn(
            reference_latent=reference_latent,
            strength=reference_strength,
            start_percent=reference_start_percent,
            end_percent=reference_end_percent,
            debug_owner=self,
        )
        if post_cfg_fn is not None:
            post_cfg_functions.append(post_cfg_fn)

    return hooked_model


def get_Kontext_stiched_o_chained_cond(self, positive, cnetpipe, tile_image):
    positive = _build_reference_conditioning(self, positive, cnetpipe, tile_image)
    return positive


def get_qwen_stiched_o_chained_cond(self, positive, cnetpipe, tile_image):
    positive = _build_reference_conditioning(self, positive, cnetpipe, tile_image)
    return positive


def Qwen_Scale_image(self, image=None):
    ref_latent = None
    if image is None:
        images = []
    else:
        samples = image.movedim(-1, 1)
        total = int(1024 * 1024)

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        image = s.movedim(1, -1)
    return image


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
                    t = t.unsqueeze(0);
                    added = True
                # for 'area' mode omit align_corners
                if interp_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
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
                            rr = rr.unsqueeze(0);
                            added = True
                        if interp_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
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
            if interp_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
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
                    if interp_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
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
    if max_h == 0 or max_w == 0:
        return cond

    # 3. Prepare updates
    updates = []
    for info in tensor_infos:
        t = info['tensor']
        sh = info['shape']
        if len(sh) >= 3 and (sh[-2] != max_h or sh[-1] != max_w):
            # move to CPU, add batch dim if needed
            cpu_t = t.detach().cpu()
            added_batch = False
            if cpu_t.dim() == 3:
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
