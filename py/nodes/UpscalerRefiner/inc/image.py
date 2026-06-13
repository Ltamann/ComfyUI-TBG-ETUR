import math
import comfy_extras
import cv2
import copy
import nodes
import numpy as np
import PIL.Image
import PIL.ImageFilter
from comfy import model_management
import torch
import torch.nn.functional as F
from .....TBG.SERVERS.WORKER_server import WORKER
from comfy_extras.nodes_mask import ImageCompositeMasked
if hasattr(ImageCompositeMasked, "execute"):
    # execute is a @classmethod; no instance needed
    def ImageCompositeMasked_execute(destination, source, x, y, resize_source, mask=None):
        return ImageCompositeMasked.execute(destination, source, x, y, resize_source, mask)
elif hasattr(ImageCompositeMasked, "composite"):
    # composite is an instance method; needs an instance
    def ImageCompositeMasked_execute(destination, source, x, y, resize_source, mask=None):
        node = ImageCompositeMasked()  # or cache a global instance if you prefer
        return node.composite(destination, source, x, y, resize_source, mask)
from comfy_extras.nodes_mask import MaskToImage
if hasattr(MaskToImage, "execute"):
    # execute is a @classmethod; no instance needed
    def MaskToImage_execute(mask):
        return MaskToImage.execute(mask)
elif hasattr(MaskToImage, "composite"):
    # composite is an instance method; needs an instance
    def MaskToImage_execute(mask):
        node = MaskToImage()  # or cache a global instance if you prefer
        return node.composite(mask)

from ....vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch, ETURColorMatchV2
from ....vendor.seedvr2_videoupscaler.src.utils.color_fix  import wavelet_adaptive_color_correction,adaptive_instance_normalization,lab_color_transfer,hsv_saturation_histogram_match,wavelet_reconstruction
from .....TBG.SERVERS.COMFYUI_server import register_main_class

@register_main_class
class TBG_Image():
    @staticmethod
    def _normalize_colormatch_image(image, name):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"ColorMatch {name} must be a torch.Tensor, got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"ColorMatch {name} expected [B,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] not in (1, 3, 4) and image.shape[1] in (1, 3, 4):
            image = image.permute(0, 2, 3, 1).contiguous()
        if image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"ColorMatch {name} expected channel-last image, got {tuple(image.shape)}")
        return image

    @classmethod
    def _normalize_colormatch_batches(cls, image_ref, image_target):
        ref = cls._normalize_colormatch_image(image_ref, "reference")
        targ = cls._normalize_colormatch_image(image_target, "target")
        ref_batch = int(ref.shape[0])
        targ_batch = int(targ.shape[0])
        if ref_batch > 1 and ref_batch != targ_batch:
            print(
                f"[TBG ColorMatch] reference batch {ref_batch} does not match target batch {targ_batch}; "
                "using the first reference image."
            )
            ref = ref[:1]
        return ref, targ

    @classmethod
    def ai_source_pattern_cleanup(
        cls,
        image,
        reduce=0.35,
        radius=128.0,
        return_debug=False,
    ):
        """
        Attenuate narrow periodic source/reference patterns before VAE/reference
        encoding. This targets FFT peak energy so real high-frequency image
        detail is preserved better than a broad blur/low-pass pass.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"AI Source Pattern Cleanup expects torch.Tensor, got {type(image)}")

        source = cls._normalize_colormatch_image(image, "ai_source").clamp(0.0, 1.0)
        amount = max(0.0, min(1.0, float(reduce)))
        user_radius = max(1.0, min(256.0, float(radius)))
        if amount <= 0.0:
            if return_debug:
                zero = torch.zeros_like(source)
                return source, zero, {"high_freq_before": 0.0, "high_freq_after": 0.0}
            return source

        target_device = source.device
        target_dtype = source.dtype
        work = source.to(dtype=torch.float32)

        rgb = work[..., :3] if work.shape[-1] >= 3 else work.expand(-1, -1, -1, 3)
        cleaned_rgb_batches = []
        preview_batches = []
        metrics = []
        notch_radius = int(max(1, min(12, round(math.sqrt(user_radius)))))

        for batch in rgb:
            h, w, _ = batch.shape
            bchw = batch.permute(2, 0, 1).unsqueeze(0)
            luma = (
                batch[..., 0] * 0.299
                + batch[..., 1] * 0.587
                + batch[..., 2] * 0.114
            )

            # Detect narrow periodic peaks in the high-frequency residual, then
            # subtract only the corresponding RGB component with edge protection.
            luma_bchw = luma.view(1, 1, h, w)
            local_low = F.avg_pool2d(luma_bchw, kernel_size=7, stride=1, padding=3)
            residual = (luma_bchw - local_low)[0, 0]
            residual = residual - residual.mean()

            hann_y = torch.hann_window(h, periodic=False, device=target_device, dtype=torch.float32)
            hann_x = torch.hann_window(w, periodic=False, device=target_device, dtype=torch.float32)
            window = hann_y[:, None] * hann_x[None, :]
            detector_fft = torch.fft.fftshift(torch.fft.fft2(residual * window))
            magnitude = torch.log1p(detector_fft.abs())

            yy = torch.arange(h, device=target_device, dtype=torch.float32)[:, None] - (h / 2.0)
            xx = torch.arange(w, device=target_device, dtype=torch.float32)[None, :] - (w / 2.0)
            freq_y = yy / max(float(h), 1.0)
            freq_x = xx / max(float(w), 1.0)
            radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)
            valid_band = radial > 0.035
            valid_values = magnitude[valid_band]

            if valid_values.numel() > 0:
                source_threshold = valid_values.mean() + valid_values.std(unbiased=False) * 1.35
            else:
                source_threshold = torch.tensor(float("inf"), device=target_device, dtype=torch.float32)

            signature_band = max(0.0025, min(0.014, (notch_radius + 1) / max(float(h), float(w))))
            source_pattern_points = torch.zeros_like(magnitude, dtype=torch.bool)
            source_pattern_targets = (
                (-0.5, 0.0),
                (0.5, 0.0),
                (0.0, -0.5),
                (0.0, 0.5),
                (0.25, -0.25),
                (-0.25, 0.25),
            )
            for target_fx, target_fy in source_pattern_targets:
                dx = torch.minimum((freq_x - target_fx).abs(), (freq_x + target_fx).abs() if abs(target_fx) == 0.5 else (freq_x - target_fx).abs())
                dy = torch.minimum((freq_y - target_fy).abs(), (freq_y + target_fy).abs() if abs(target_fy) == 0.5 else (freq_y - target_fy).abs())
                source_pattern_points |= torch.sqrt(dx * dx + dy * dy) <= signature_band

            peak_mask = source_pattern_points & (magnitude >= source_threshold)

            if peak_mask.any():
                peak_field = peak_mask.to(torch.float32).view(1, 1, h, w)
                kernel = notch_radius * 2 + 1
                notch = F.max_pool2d(peak_field, kernel_size=kernel, stride=1, padding=notch_radius)[0, 0]
                notch = torch.fft.ifftshift(notch.clamp(0.0, 1.0))

                rgb_fft = torch.fft.fft2(bchw[0])
                periodic = torch.fft.ifft2(rgb_fft * notch).real.permute(1, 2, 0)

                dx = F.pad(luma[:, 1:] - luma[:, :-1], (0, 1, 0, 0))
                dy = F.pad(luma[1:, :] - luma[:-1, :], (0, 0, 0, 1))
                grad = torch.sqrt(dx * dx + dy * dy)
                grad_scale = torch.quantile(grad.flatten(), 0.92).clamp_min(1e-4)
                detail_protect = (1.0 - (grad / grad_scale).clamp(0.0, 1.0)).pow(2.0)
                cleanup_gate = (0.04 + 0.96 * detail_protect).unsqueeze(-1)

                padded_rgb = F.pad(bchw, (1, 1, 1, 1), mode="reflect")
                micro_low = F.avg_pool2d(padded_rgb, kernel_size=3, stride=1)[0].permute(1, 2, 0)
                micro_component = (batch - micro_low).clamp(-0.018, 0.018)

                # The measured FFT signature gates the cleanup, but direct FFT
                # subtraction can ring into dark speckles. Keep it as a tiny
                # bounded correction and let the edge-protected anti-checker
                # pass do the visible work in smooth skin/background regions.
                bounded_periodic = periodic.clamp(-0.006, 0.006)
                micro_amount = amount * 0.10
                periodic_amount = amount * 0.18

                removed = (bounded_periodic * periodic_amount + micro_component * micro_amount) * cleanup_gate
                cleaned = (batch - removed).clamp(0.0, 1.0)

                r = cleaned[..., 0]
                g = cleaned[..., 1]
                b = cleaned[..., 2]
                maxc = torch.maximum(torch.maximum(r, g), b)
                minc = torch.minimum(torch.minimum(r, g), b)
                saturation = (maxc - minc) / maxc.clamp_min(1e-4)
                skin_mask = (
                    (r > 0.22)
                    & (r > g * 0.92)
                    & (g > b * 0.82)
                    & (saturation > 0.10)
                    & (saturation < 0.62)
                ).to(torch.float32)

                cleaned_bchw = cleaned.permute(2, 0, 1).unsqueeze(0)
                local_rgb = F.avg_pool2d(
                    F.pad(cleaned_bchw, (2, 2, 2, 2), mode="reflect"),
                    kernel_size=5,
                    stride=1,
                )[0].permute(1, 2, 0)
                skin_low_rgb = F.avg_pool2d(
                    F.pad(cleaned_bchw, (3, 3, 3, 3), mode="reflect"),
                    kernel_size=7,
                    stride=1,
                )[0].permute(1, 2, 0)
                local_luma = (
                    local_rgb[..., 0] * 0.299
                    + local_rgb[..., 1] * 0.587
                    + local_rgb[..., 2] * 0.114
                )
                cleaned_luma = (
                    cleaned[..., 0] * 0.299
                    + cleaned[..., 1] * 0.587
                    + cleaned[..., 2] * 0.114
                )
                dark_speckle = (local_luma - cleaned_luma - 0.006).clamp(0.0, 0.05)
                speckle_gate = skin_mask * detail_protect
                skin_smooth = (speckle_gate * amount * 0.42).unsqueeze(-1)
                cleaned = (cleaned * (1.0 - skin_smooth) + skin_low_rgb * skin_smooth).clamp(0.0, 1.0)
                speckle_lift = dark_speckle.unsqueeze(-1) * speckle_gate.unsqueeze(-1) * amount * 0.85
                cleaned = (cleaned + speckle_lift).clamp(0.0, 1.0)
                removed_preview = (removed.abs() * 8.0).clamp(0.0, 1.0)
                metric_before = float(periodic.abs().mean().detach().cpu())
                metric_after = float((periodic * (1.0 - cleanup_gate * amount)).abs().mean().detach().cpu())
            else:
                cleaned = batch
                removed_preview = torch.zeros_like(batch)
                metric_before = 0.0
                metric_after = 0.0

            cleaned_rgb_batches.append(cleaned)
            preview_batches.append(removed_preview)
            metrics.append((metric_before, metric_after))

        cleaned_rgb = torch.stack(cleaned_rgb_batches, dim=0)
        preview = torch.stack(preview_batches, dim=0)
        if work.shape[-1] == 1:
            cleaned = cleaned_rgb.mean(dim=-1, keepdim=True)
            preview = preview.mean(dim=-1, keepdim=True)
        elif work.shape[-1] > 3:
            cleaned = torch.cat((cleaned_rgb, work[..., 3:]), dim=-1)
            preview = torch.cat((preview, torch.zeros_like(work[..., 3:])), dim=-1)
        else:
            cleaned = cleaned_rgb

        result = cleaned.to(dtype=target_dtype)

        if return_debug:
            before = sum(m[0] for m in metrics) / max(1, len(metrics))
            after = sum(m[1] for m in metrics) / max(1, len(metrics))
            metrics = {
                "high_freq_before": before,
                "high_freq_after": after,
                "notch_radius": float(notch_radius),
            }
            return result, preview.to(dtype=target_dtype), metrics
        return result

    @classmethod
    def ImageCompositeMasked(cls, full_image, _image, x_start,y_start, resize_source, used_mask):
        return  ImageCompositeMasked.execute(full_image, _image, x=x_start, y=y_start, resize_source=False, mask=used_mask)

    @classmethod
    def upscale(self, image, upscale_method, width, height, crop):
        return nodes.ImageScale().upscale(image, upscale_method, width, height, crop)[0]

    # LOCAL POSTPROCESSING
    @classmethod
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        """
        image_ref, image_target: BHWC, float in [0,1] (Comfy-style)
        Returns: BHWC, float in [0,1]
        """
        # Make sure we are in torch and float32, and break reference chains
        assert isinstance(image_ref, torch.Tensor)
        assert isinstance(image_target, torch.Tensor)
        image_ref, image_target = self._normalize_colormatch_batches(image_ref, image_target)

        # Branch: ColorMatch V2 family works on BHWC in [0,1]
        if method in ('mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm', 'reinhard_lab_gpu'):
            ref = image_ref.to(torch.float32).clone()
            targ = image_target.to(torch.float32).clone()
            try:
                print(f"[TBG ColorMatch] using ETUR ColorMatchV2 method={method} strength={strength}")
                out = ETURColorMatchV2().colormatch(ref, targ, method, strength, True)[0]
            except Exception as exc:
                print(f"[TBG ColorMatch] ETUR ColorMatchV2 failed method={method}: {exc}; falling back to legacy ColorMatch")
                out = ColorMatch().colormatch(ref, targ, method, strength)[0]
            return (out,)

        # --- New methods expect BCHW in [-1,1] ---
        target_device = image_target.device
        processing_device = target_device
        if processing_device.type == "cpu" and torch.cuda.is_available():
            processing_device = model_management.get_torch_device()

        ref = image_ref.to(device=processing_device, dtype=torch.float32).clone()
        targ = image_target.to(device=processing_device, dtype=torch.float32).clone()

        # BHWC -> BCHW
        ti = targ.permute(0, 3, 1, 2).contiguous()
        ri = ref.permute(0, 3, 1, 2).contiguous()

        # [0,1] -> [-1,1]
        ti = ti.mul(2.0).sub(1.0)
        ri = ri.mul(2.0).sub(1.0)

        # Apply selected method
        if method == "lab color match+detail preservation":
            out = lab_color_transfer(ti, ri, None, luminance_weight=0.8)
        elif method == "lab full color match":
            out = lab_color_transfer(ti, ri, None, luminance_weight=0.0)
        elif method == "wavelet_adaptive":
            out = wavelet_adaptive_color_correction(ti, ri, None)
        elif method == "wavelet":
            out = wavelet_reconstruction(ti, ri, None)
        elif method == "hsv":
            out = hsv_saturation_histogram_match(ti, ri, None)
        elif method == "adain":
            out = adaptive_instance_normalization(ti, ri)
        else:
            # Fallback: no correction
            out = ti

        # [-1,1] -> [0,1]
        out = out.add(1.0).mul(0.5).clamp_(0.0, 1.0)

        # BCHW -> BHWC
        out = out.permute(0, 2, 3, 1).contiguous()
        if out.device != target_device:
            out = out.to(target_device)

        return (out,)

    @staticmethod
    def _low_frequency_bhwc(image_bhwc, max_pool=1):
        bchw = image_bhwc.to(torch.float32).permute(0, 3, 1, 2).contiguous()
        height, width = bchw.shape[-2:]
        pooled_h = max(1, min(int(max_pool), max(1, height // 8)))
        pooled_w = max(1, min(int(max_pool), max(1, width // 8)))
        low = F.interpolate(
            F.adaptive_avg_pool2d(bchw, (pooled_h, pooled_w)),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return low.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _small_low_frequency_bhwc(image_bhwc, kernel_size=5):
        bchw = image_bhwc.to(torch.float32).permute(0, 3, 1, 2).contiguous()
        kernel_size = max(3, int(kernel_size) | 1)
        radius = kernel_size // 2
        padded = F.pad(bchw, (radius, radius, radius, radius), mode="reflect")
        low = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
        return low.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _band_abs_mean(band):
        return float(torch.mean(torch.abs(band)).detach().cpu())

    @staticmethod
    def _masked_bhwc_mean(image, mask=None):
        if mask is None:
            return image.mean(dim=(1, 2), keepdim=True)
        weight = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        if int(weight.shape[0]) == 1 and int(image.shape[0]) > 1:
            weight = weight.expand(int(image.shape[0]), -1, -1, -1)
        denom = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (image * weight).sum(dim=(1, 2), keepdim=True) / denom

    @classmethod
    def _detail_color_metrics(cls, before, after, low_before, low_after, mid_before, mid_after, high_before, high_after, mask=None):
        eps = 1e-8
        low_delta = cls._band_abs_mean(low_after - low_before)
        mid_delta = cls._band_abs_mean(mid_after - mid_before)
        high_delta = cls._band_abs_mean(high_after - high_before)
        mid_ref = cls._band_abs_mean(mid_before)
        high_ref = cls._band_abs_mean(high_before)
        mid_retention = max(0.0, 1.0 - (mid_delta / max(mid_ref, eps)))
        high_retention = max(0.0, 1.0 - (high_delta / max(high_ref, eps)))
        rgb_shift = (after.to(torch.float32) - before.to(torch.float32)).mean(dim=(0, 1, 2))[:3]
        diff = torch.abs(after.to(torch.float32) - before.to(torch.float32)).mean(dim=-1, keepdim=True)
        seam_delta = None
        center_delta = None
        if mask is not None:
            m = cls._mask_to_bchw(mask, before.permute(0, 3, 1, 2).contiguous())
            if m is not None:
                m = m.permute(0, 2, 3, 1).to(device=diff.device, dtype=diff.dtype)
                seam_delta = float((diff * m).sum().detach().cpu() / m.sum().clamp_min(1e-6).detach().cpu())
                inv = 1.0 - m
                center_delta = float((diff * inv).sum().detach().cpu() / inv.sum().clamp_min(1e-6).detach().cpu())
        return {
            "low_delta": low_delta,
            "mid_retention": mid_retention,
            "high_retention": high_retention,
            "rgb_shift": tuple(float(v) for v in rgb_shift.detach().cpu()),
            "seam_delta": seam_delta,
            "center_delta": center_delta,
        }

    @classmethod
    def detail_preserving_colormatch(cls, image_ref, image_target, method, strength=1.0, apply_mask=None, label=None):
        """
        Color-match only as a global color delta. This intentionally avoids
        spatial low-frequency fields, because pooled fields can imprint grids.
        """
        if method is None or str(method).lower() == "none":
            return (image_target,)

        ref, target = cls._normalize_colormatch_batches(image_ref, image_target)
        original_device = target.device
        original_dtype = target.dtype
        ref = ref.to(torch.float32)
        target = target.to(torch.float32)

        if int(ref.shape[1]) != int(target.shape[1]) or int(ref.shape[2]) != int(target.shape[2]):
            ref = nodes.ImageScale().upscale(ref, "bilinear", int(target.shape[2]), int(target.shape[1]), False)[0]
            ref = ref.to(device=target.device, dtype=torch.float32)

        mask_bhwc = None
        mask_coverage = 0.0
        if apply_mask is not None:
            mask_bchw = cls._mask_to_bchw(apply_mask, target.permute(0, 3, 1, 2).contiguous())
            if mask_bchw is not None:
                mask_bhwc = mask_bchw.permute(0, 2, 3, 1).to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)
                if mask_bhwc.numel() > 0 and float(mask_bhwc.max().detach().cpu()) > 0.001:
                    mask_coverage = float((mask_bhwc > 0.001).to(torch.float32).mean().detach().cpu())
                else:
                    mask_bhwc = None

        low_ref = cls._masked_bhwc_mean(ref, mask_bhwc)
        low_target = cls._masked_bhwc_mean(target, mask_bhwc)
        small_low_target = cls._small_low_frequency_bhwc(target)
        mid_target = small_low_target - low_target
        high_target = target - small_low_target

        low_delta = (low_ref - low_target).clamp(-0.15, 0.15) * float(strength)

        if mask_bhwc is not None:
            low_delta = low_delta * mask_bhwc

        corrected_low = low_target + low_delta
        corrected = (corrected_low + mid_target + high_target).clamp(0.0, 1.0)
        corrected = corrected.to(device=original_device, dtype=original_dtype)

        if label:
            low_after = corrected.to(torch.float32).mean(dim=(1, 2), keepdim=True)
            small_low_after = cls._small_low_frequency_bhwc(corrected.to(torch.float32))
            mid_after = small_low_after - low_after
            high_after = corrected.to(torch.float32) - small_low_after
            metrics = cls._detail_color_metrics(
                target,
                corrected.to(torch.float32),
                low_target,
                low_after,
                mid_target,
                mid_after,
                high_target,
                high_after,
                mask=apply_mask,
            )
            rgb = metrics["rgb_shift"]
            extra = ""
            if metrics["seam_delta"] is not None:
                extra = f" seam_delta={metrics['seam_delta']:.8f} center_delta={metrics['center_delta']:.8f}"
            print(
                f"[TBG DetailColor] {label}: method={method} strength={float(strength):.3f} "
                f"stats={'masked' if mask_bhwc is not None else 'full'} "
                f"mask_coverage={mask_coverage:.4f} "
                f"low_delta={metrics['low_delta']:.8f} "
                f"mid_retention={metrics['mid_retention']:.4f} "
                f"high_retention={metrics['high_retention']:.4f} "
                f"rgb_shift=({rgb[0]:+.6f},{rgb[1]:+.6f},{rgb[2]:+.6f})"
                f"{extra}"
            )

        return (corrected,)

    @staticmethod
    def _mask_to_bchw(mask, ref_tensor):
        if mask is None:
            return None

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(device=ref_tensor.device, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            if mask.shape[0] == ref_tensor.shape[0]:
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)

        if mask.ndim != 4:
            return None

        if mask.shape[-2:] != ref_tensor.shape[-2:]:
            mask = F.interpolate(mask, size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False)

        if mask.shape[0] != ref_tensor.shape[0]:
            mask = mask.expand(ref_tensor.shape[0], -1, -1, -1)

        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _masked_channel_stats(image_bchw, mask_bchw):
        if mask_bchw.device != image_bchw.device or mask_bchw.dtype != image_bchw.dtype:
            mask_bchw = mask_bchw.to(device=image_bchw.device, dtype=image_bchw.dtype)

        weight = mask_bchw.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mean = (image_bchw * mask_bchw).sum(dim=(-2, -1), keepdim=True) / weight
        var = (((image_bchw - mean) ** 2) * mask_bchw).sum(dim=(-2, -1), keepdim=True) / weight
        std = var.clamp_min(1e-6).sqrt()
        return mean, std

    @classmethod
    def _bchw_mask_to_bhw(cls, mask_bchw):
        if mask_bchw is None:
            return None
        if mask_bchw.ndim == 4 and mask_bchw.shape[1] == 1:
            return mask_bchw.squeeze(1)
        return mask_bchw

    @classmethod
    def build_post_sampling_masks(cls, image_ref, border_correction_mask, denoise_mask=None, protect_threshold=(1.0 / 255.0)):
        if not isinstance(image_ref, torch.Tensor):
            return {}

        ref_bchw = image_ref.to(torch.float32).permute(0, 3, 1, 2).contiguous()
        border_mask = cls._mask_to_bchw(border_correction_mask, ref_bchw)
        if border_mask is None:
            border_mask = torch.ones(
                (ref_bchw.shape[0], 1, ref_bchw.shape[-2], ref_bchw.shape[-1]),
                device=ref_bchw.device,
                dtype=ref_bchw.dtype,
            )

        # border_mask is the post-sampling protection map: white=center preserve, black=editable seam.
        preserved_strip = border_mask.clamp(0.0, 1.0)

        explicit_protected = torch.zeros_like(preserved_strip)
        has_explicit_protection = False
        if denoise_mask is not None:
            denoise_mask_bchw = cls._mask_to_bchw(denoise_mask, ref_bchw)
            if denoise_mask_bchw is not None:
                explicit_protected = (denoise_mask_bchw <= float(protect_threshold)).to(ref_bchw.dtype)
                has_explicit_protection = bool(explicit_protected.max().item() > 0.0)

        hard_lock = torch.maximum(preserved_strip, explicit_protected)
        trusted = hard_lock
        editable = (1.0 - hard_lock).clamp_(0.0, 1.0)
        # Final blend should keep the editable seam from the sampled tile, not re-impose the source strip.
        final_blend = editable

        return {
            "border_blend": cls._bchw_mask_to_bhw(border_mask),
            "preserved_strip": cls._bchw_mask_to_bhw(preserved_strip),
            "explicit_protected": cls._bchw_mask_to_bhw(explicit_protected),
            "hard_lock": cls._bchw_mask_to_bhw(hard_lock),
            "trusted": cls._bchw_mask_to_bhw(trusted),
            "editable": cls._bchw_mask_to_bhw(editable),
            "final_blend": cls._bchw_mask_to_bhw(final_blend),
            "has_explicit_protection": has_explicit_protection,
        }

    @classmethod
    def stabilize_tile_low_frequency_from_reference(cls, image_ref, image_target, trusted_mask, apply_mask=None, strength=1.0):
        if trusted_mask is None:
            return (image_target,)

        if not isinstance(image_ref, torch.Tensor) or not isinstance(image_target, torch.Tensor):
            return (image_target,)

        work_device = image_target.device
        ref = image_ref.to(device=work_device, dtype=torch.float32).clone()
        targ = image_target.to(device=work_device, dtype=torch.float32).clone()

        ref_bchw = ref.permute(0, 3, 1, 2).contiguous()
        targ_bchw = targ.permute(0, 3, 1, 2).contiguous()
        trusted_mask_bchw = cls._mask_to_bchw(trusted_mask, ref_bchw)
        if trusted_mask_bchw is None or trusted_mask_bchw.max().item() <= 1e-5:
            return (image_target,)
        apply_mask_bchw = cls._mask_to_bchw(apply_mask, ref_bchw) if apply_mask is not None else None

        height, width = ref_bchw.shape[-2:]
        pooled_h = max(8, min(64, height // 8))
        pooled_w = max(8, min(64, width // 8))
        low_ref = F.interpolate(
            F.adaptive_avg_pool2d(ref_bchw, (pooled_h, pooled_w)),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        low_targ = F.interpolate(
            F.adaptive_avg_pool2d(targ_bchw, (pooled_h, pooled_w)),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        ref_mean, ref_std = cls._masked_channel_stats(low_ref, trusted_mask_bchw)
        targ_mean, targ_std = cls._masked_channel_stats(low_targ, trusted_mask_bchw)

        gain = (ref_std / targ_std).clamp(0.85, 1.15)
        bias = (ref_mean - (targ_mean * gain)).clamp(-0.15, 0.15)

        low_targ_corrected = (low_targ * gain) + bias
        high_targ = targ_bchw - low_targ
        corrected_full = (high_targ + low_targ_corrected).clamp_(0.0, 1.0)

        if apply_mask_bchw is not None:
            corrected = targ_bchw + (corrected_full - targ_bchw) * apply_mask_bchw * float(strength)
        else:
            corrected = targ_bchw + (corrected_full - targ_bchw) * float(strength)

        corrected = corrected.clamp_(0.0, 1.0)
        corrected = corrected.permute(0, 2, 3, 1).contiguous()

        return (corrected,)

    @classmethod
    def restore_from_reference_mask(cls, image_ref, image_target, protect_mask):
        if protect_mask is None:
            return (image_target,)

        if not isinstance(image_ref, torch.Tensor) or not isinstance(image_target, torch.Tensor):
            return (image_target,)

        work_device = image_target.device
        ref_bchw = image_ref.to(device=work_device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        targ_bchw = image_target.to(device=work_device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        protect_mask_bchw = cls._mask_to_bchw(protect_mask, ref_bchw)
        if protect_mask_bchw is None or protect_mask_bchw.max().item() <= 1e-5:
            return (image_target,)

        restored = targ_bchw + (ref_bchw - targ_bchw) * protect_mask_bchw
        restored = restored.clamp_(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
        return (restored,)

    @classmethod
    def masked_mean_abs_diff(cls, image_a, image_b, mask=None):
        if not isinstance(image_a, torch.Tensor) or not isinstance(image_b, torch.Tensor):
            return 0.0

        a_bchw = image_a.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        b_bchw = image_b.to(device=image_a.device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        diff = (a_bchw - b_bchw).abs().mean(dim=1, keepdim=True)

        if mask is None:
            return float(diff.mean().item())

        mask_bchw = cls._mask_to_bchw(mask, a_bchw)
        if mask_bchw is None:
            return float(diff.mean().item())

        weight = mask_bchw.sum().clamp_min(1e-6)
        return float(((diff * mask_bchw).sum() / weight).item())

    # LOCAL HELPER
    @classmethod
    def is_divisible_by_8(self, image):
        width, height = image.shape[2], image.shape[1]
        return (width % 8 == 0) and (height % 8 == 0)

    # LOCAL HELPER
    @classmethod
    def calculate_new_dimensions(self, image_width, image_height):
        def round_up_to_nearest_8(x):
            return math.ceil(x / 8) * 8

        new_width = round_up_to_nearest_8(image_width)
        new_height = round_up_to_nearest_8(image_height)
        return new_width, new_height

    # LOCAL HELPER
    @classmethod
    def format_2_divby8(self, image):
        original_width, original_height = image.shape[2], image.shape[1]

        if not self.is_divisible_by_8(image):
            new_width, new_height = self.calculate_new_dimensions(original_width, original_height)

            # Calculate padding offset (assuming center crop)
            pad_offset_x = (new_width - original_width) // 2
            pad_offset_y = (new_height - original_height) // 2

            image = nodes.ImageScale.upscale(nodes.ImageScale, image, "bilinear", new_width, new_height, "center")[0]

            return image, new_width, new_height, False, (pad_offset_x, pad_offset_y)

        return image, original_width, original_height, True, (0, 0)

    # LOCAL HELPER
    @staticmethod
    def to_numpy_uint8(mask):
        """Converts a PyTorch tensor or NumPy array to a NumPy uint8 array."""
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()  # Convert tensor to NumPy
        mask = np.squeeze(mask)  # Remove singleton dimensions if needed
        if mask.max() > 1:
            mask = mask / mask.max()
        return (mask * 255).astype(np.uint8)

    # LOCAL SEGMENTS
    def transform_segment_coordinates(self, segments, pad_offset, upscale_factor):
        """Transform segment coordinates through the same pipeline as the image"""
        if not segments:
            return segments

        pad_x, pad_y = pad_offset
        transformed_segments = []

        # segments should be (height, width), [SEG objects]
        if isinstance(segments, tuple) and len(segments) == 2:
            img_dims, seg_list = segments

            for seg in seg_list:
                if hasattr(seg, 'crop_region'):
                    x1, y1, x2, y2 = seg.crop_region
                    bx1, by1, bx2, by2 = getattr(seg, "bbox", seg.crop_region)

                    # Apply transformations: padding first, then scaling
                    new_x1 = int((x1 + pad_x) * upscale_factor)
                    new_y1 = int((y1 + pad_y) * upscale_factor)
                    new_x2 = int((x2 + pad_x) * upscale_factor)
                    new_y2 = int((y2 + pad_y) * upscale_factor)
                    new_bx1 = int((bx1 + pad_x) * upscale_factor)
                    new_by1 = int((by1 + pad_y) * upscale_factor)
                    new_bx2 = int((bx2 + pad_x) * upscale_factor)
                    new_by2 = int((by2 + pad_y) * upscale_factor)

                    if hasattr(seg, "_fields"):
                        values = {name: getattr(seg, name) for name in seg._fields}
                        values["crop_region"] = (new_x1, new_y1, new_x2, new_y2)
                        values["bbox"] = (new_bx1, new_by1, new_bx2, new_by2)
                        new_seg = type(seg)(**values)
                    else:
                        from collections import namedtuple
                        SEG_impakt = namedtuple("SEG",
                                                ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                                                defaults=[None])
                        new_seg = SEG_impakt(
                            cropped_image=seg.cropped_image,
                            cropped_mask=seg.cropped_mask,
                            confidence=seg.confidence,
                            crop_region=(new_x1, new_y1, new_x2, new_y2),
                            bbox=(new_bx1, new_by1, new_bx2, new_by2),
                            label=seg.label,
                            control_net_wrapper=getattr(seg, "control_net_wrapper", None)
                        )
                    transformed_segments.append(new_seg)
                else:
                    print(f"Warning: Expected SEG object, got {type(seg)}")
                    transformed_segments.append(seg)

            # Update image dimensions
            new_img_dims = (
                int((img_dims[0] + 2 * pad_y) * upscale_factor),
                int((img_dims[1] + 2 * pad_x) * upscale_factor)
            )

            return (new_img_dims, transformed_segments)
        else:
            print(f"Unexpected segments structure: {type(segments)}")
            return segments

    # LOCAL VAE
    @classmethod
    def VAEDecodeFluxNormalized(self, vae, samples):
        tile_size = 512
        fast_mode = True
        compression = vae.spacial_compression_decode()
        tile_latent = tile_size // compression
        overlap = tile_latent // 4  # 25% overlap for smooth blending

        if fast_mode:
            # Single-pass tiled decode (3x faster)
            images = self.TBG_VAE_decode_single_pass(vae, samples["samples"], tile_latent, overlap)
        else:
            # Use the original 3-pass tiled decode for maximum quality
            images = vae.decode_tiled(samples["samples"], tile_x=tile_latent, tile_y=tile_latent, overlap=overlap)

        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        return (images,)

    # LOCAL VAE
    @classmethod
    def TBG_VAE_decode_single_pass(cls, vae, samples, tile_x, overlap):
        """Single-pass tiled decode - 3x faster than the original 3-pass method"""
        import comfy.utils

        vae.throw_exception_if_invalid()

        # Calculate steps for progress bar
        steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_x, overlap)
        # pbar = ProgressBar(steps)

        # Load VAE to GPU
        memory_used = vae.memory_used_decode(samples.shape, vae.vae_dtype)
        model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

        # Decode function
        decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).float()

        # Single tiled_scale pass (instead of 3 passes)
        output = comfy.utils.tiled_scale(
            samples,
            decode_fn,
            tile_x,
            tile_x,  # Use square tiles
            overlap,
            upscale_amount=vae.upscale_ratio,
            output_device=vae.output_device,
            # pbar=pbar
        )

        # Apply process_output and move channels
        output = vae.process_output(output)
        return output.movedim(1, -1)
    # WORKER GRID SPECS

    @classmethod # used by cnet sigmas.py  only
    def gridspecs_get_tiled_grid_specs(cls, SELF, fullimageH, fullimageW):
        # Passed to TBG_WORKER
        grid_specs, _, _, _  = WORKER.id(SELF.INFO.tiler_id).TBG_Image.gridspecs_get_tiled_grid_specs(SELF.SIZE, fullimageH, fullimageW, SELF.PARAMS.Tile_Fusion_Mode,_tbg_send_images=False,)
        return grid_specs
    # LOCAL GRID SPECS - image crop
    @classmethod
    def gridspecs_get_grid_images(cls, image, grid_specs):
        grids = [
            image[
                :,
                int(y_start):int(y_start + height_inc),
                int(x_start):int(x_start + width_inc),
                :
            ] for _, _, _, x_start, y_start, width_inc, height_inc in grid_specs
        ]
        return grids


    # WORKER MASK
    @classmethod #used in msk preview Only
    def mask_get_composite_and_fusion_mask(self, SELF,tiler_id, tile_width, tile_height, composite_blur_margin=64, inpaint_border_margin=0, inpaint_blur_margin=0, inpaint_shift=0, shift_left_top=0, inpaint_max=1,
                                               feather_left=False, feather_top=False, feather_right=False, feather_bottom=False):
        c_mask, i_mask , SELF.SIZE=  WORKER.id(tiler_id).TBG_Image.mask_get_composite_and_fusion_mask(SELF.SIZE, tile_width, tile_height, composite_blur_margin=64, inpaint_border_margin=0, inpaint_blur_margin=0, inpaint_shift=0, shift_left_top=0, inpaint_max=1,
                                               feather_left=False, feather_top=False, feather_right=False, feather_bottom=False,_tbg_send_images=False,)
        return c_mask, i_mask
    # WORKER MASK #used in msk preview Only
     #used in cnet inpaint allimama and in mask preview
    def mask_get_fusion_mask(self, SELF, latent_image_H, latent_image_W, cols_qty, rows_qty, grid_specs_index, inpaint_blur_margin, inpaint_shift, shift_left_top, inpaint_border_margin, Upscale_Detailer, inpaint_max=1):
        i_mask , SELF.SIZE = WORKER.id(SELF.INFO.tiler_id).TBG_Image.mask_get_fusion_mask(SELF.SIZE, SELF.PARAMS, latent_image_H, latent_image_W, cols_qty, rows_qty, grid_specs_index, inpaint_blur_margin, inpaint_shift, shift_left_top, inpaint_border_margin, Upscale_Detailer, inpaint_max=1,_tbg_send_images=False,)
        return i_mask
    # WORKER MASK
    @classmethod
    def mask_set_constants(self, SELF):
        SELF.SIZE = WORKER.id(SELF.INFO.tiler_id).TBG_Image.mask_set_constants(SELF.SIZE, SELF.PARAMS.Tile_Fusion_Mode,_tbg_send_images=False,)

        return (
            SELF.SIZE.crop_margin,
            SELF.SIZE.tile_grid_W,
            SELF.SIZE.tile_grid_H,
            SELF.SIZE.rows_qty,
            SELF.SIZE.cols_qty,
            SELF.SIZE.outer_mask_area,
            SELF.SIZE.overlay_between_tiles)
    #WORKER REBUILD
    """
    @classmethod # pyramidal
    def rebuild_final_image(self, SELF, iteration, output_grid_images_all, original_image_upscaled, grid_specs, upscale_scale, rows_qty, cols_qty, grid_prompts, segs_scales, segms_cropped_masks, segms_new, upscale_method, nosegments=False):
        SELF.full_image_only_tiles = getattr(SELF, "full_image_only_tiles", None)
        # Extract from each (size, [seg1, seg2, ...]) tuple
        all_crop_regions = []
        all_compositing_masks = []
        if segms_new is not None:
            _ , segms= segms_new
            for seg in segms:
                all_crop_regions.append(seg.crop_region)
                all_compositing_masks.append(seg.compositing_mask)

        full_image, full_image_only_tiles, tiles_order =   WORKER.id(SELF.INFO.tiler_id).TBG_Image.rebuild_final_image(
            output_grid_images_all,
            original_image_upscaled,
            grid_prompts, all_compositing_masks,
            all_crop_regions,
            nosegments=False,
            full_image_only_tiles=SELF.full_image_only_tiles)

        return full_image, full_image_only_tiles, tiles_order
    """


    # LOCAL used in Refiner
    def helper_upscaleimage(self, image, upscale_method="bilinear", upscale_model_name=None, scale_factor=1, width=0, height=0):
        if scale_factor != 0:
            width = int(image.shape[2] * scale_factor)
            height = int(image.shape[1] * scale_factor)
        if upscale_method == "with model":
            upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader.load_model(self, upscale_model_name)[0]
            image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(self, upscale_model, image)[0]
        image = nodes.ImageScale().upscale(image, upscale_method, width, height, False)[0]

        return image


    def Preview_Mask(cls, self,tiler_id):
        _, H, W, _ = self.INPUTS.image.shape

        if self.PARAMS.preset == 'Full size Image no Tiles' or W < self.SIZE.fullW or H < self.SIZE.fullH:
            return self.INPUTS.image

        # , mask1, mask2, background_img, SIZE, feather_inpaint, feather, fusion_mode

        feather_mask = self.SIZE.composite_blur_margin
        inpaint_mask = self.SIZE.inpaint_border_margin
        inpaint_blur_margin = self.SIZE.inpaint_blur_margin
        inpaint_shift = self.SIZE.shift
        crop_margin, tile_grid_W, tile_grid_H, rows_qty, cols_qty ,_,_= cls.mask_set_constants(self)


        # safeguard if  self.OUTPUTS.grid_images not exist of Full images selected
        if self.PARAMS.len_grid_images == 0 or self.PARAMS.preset == 'Full size Image no Tiles':
            # work around  the immage can be missing if scale factor smaler than 1
            overlay_masks_image = self.INPUTS.image
        else:
            overlay_masks_image = copy.copy(self.OUTPUTS.grid_images_all[0])
            result_batch = []
            for i in [0, 1]:
                # fake a grid position row 2 col 2 / inpainting fakes 10 row 10 col so we get all boarders
                grid = []
                grid.append([
                    2,  #
                    2,  #
                    0,  #
                    0,  # x
                    0,  # y
                    overlay_masks_image.shape[2],  # width
                    overlay_masks_image.shape[1],  # height
                ])

                # using the same function used in the refiner to get the fusion mask
                inpaintmask = TBG_Image().mask_get_fusion_mask(self, overlay_masks_image.shape[1], overlay_masks_image.shape[2], 10, 10, grid[0], self.SIZE.inpaint_blur_margin, self.SIZE.shift, self.SIZE.shifttl,
                                                               self.SIZE.inpaint_border_margin, 1)[0]

                # using the same function used in rebuild images script
                compositing_mask, _ = TBG_Image().mask_get_composite_and_fusion_mask(self, tiler_id,overlay_masks_image.shape[2], overlay_masks_image.shape[1],
                                                                                     composite_blur_margin=self.SIZE.composite_blur_margin,
                                                                                     inpaint_border_margin=crop_margin,
                                                                                     inpaint_blur_margin=0,
                                                                                     inpaint_shift=0,
                                                                                     shift_left_top=0,
                                                                                     inpaint_max=0,
                                                                                     feather_left=True,
                                                                                     feather_top=True,
                                                                                     feather_right=True,
                                                                                     feather_bottom=True)

                mask1 = compositing_mask
                mask2 = inpaintmask
                if i < len(self.PARAMS.len_grid_images):

                    background_img = overlay_masks_image
                else:
                    # create a 1024x1024 black image (RGB) as a torch tensor
                    background_img = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)

                #background_img = self.OUTPUTS.grid_images[i]
                feather_inpaint = inpaint_mask
                feather = feather_mask
                fusion_mode = None



                square_height = self.SIZE.actual_inner_tile_sizeH
                square_width = self.SIZE.actual_inner_tile_sizeW
                W = background_img.shape[2]
                H = background_img.shape[1]
                # saveguard images to small
                if W < square_width or H < square_width:
                    return background_img

                def keep_middle_ring(mask, W, H, border=50):
                    # Create an all-zero mask
                    output = np.zeros_like(mask)
                    height, width = mask.shape[:2]

                    # Coordinates of the center rectangle
                    center_y, center_x = height // 2, width // 2
                    top = center_y - H // 2
                    bottom = center_y + H // 2
                    left = center_x - W // 2
                    right = center_x + W // 2

                    # Define the bounding box excluding the center and borders
                    y1, y2 = border, height - border
                    x1, x2 = border, width - border

                    # Copy the region between border and center rectangle
                    output[y1:top, x1:x2] = mask[y1:top, x1:x2]  # Top middle band
                    output[bottom:y2, x1:x2] = mask[bottom:y2, x1:x2]  # Bottom middle band
                    output[top:bottom, x1:left] = mask[top:bottom, x1:left]  # Left middle band
                    output[top:bottom, right:x2] = mask[top:bottom, right:x2]  # Right middle band

                    return output

                # Convert masks to numpy arrays
                mask1 = cls.to_numpy_uint8(mask1)  # red
                mask1 = 255 - mask1  # invert mask
                mask2 = cls.to_numpy_uint8(mask2)  # green
                mask2 = 255 - mask2  # invert mask
                # mask1 = keep_middle_ring( mask1, square_height, square_width, border=feather_inpaint)

                # mask2 =  255 - mask2 # invert mask
                background_img = cls.to_numpy_uint8(background_img[0])

                # Check if the background image is a tensor and convert to NumPy if necessary
                if isinstance(background_img, torch.Tensor):
                    # If it's a tensor, convert it to NumPy
                    background_img = background_img.detach().cpu().numpy()

                    # Remove the batch dimension (if it exists) and ensure the correct order
                    if background_img.ndim == 4:  # (1, C, H, W)
                        background_img = background_img[0]  # Remove batch dimension

                    # Convert (C, H, W) to (H, W, C)
                    if background_img.ndim == 3 and background_img.shape[0] in [1, 3]:
                        background_img = np.transpose(background_img, (1, 2, 0))

                # If background_img is already a numpy array, ensure it's in the correct format

                if isinstance(background_img, np.ndarray):
                    # Convert (C, H, W) to (H, W, C) if necessary
                    if background_img.ndim == 3 and background_img.shape[0] in [1, 3]:
                        background_img = np.transpose(background_img, (1, 2, 0))

                    # Ensure dtype is uint8 for OpenCV
                    if background_img.max() <= 1.0:
                        background_img = (background_img * 255).astype(np.uint8)
                    else:
                        background_img = background_img.astype(np.uint8)

                # Ensure the background image is not empty
                assert background_img is not None and background_img.size > 0, "Background image is empty or not loaded"
                background_resized = background_img

                # Center the masks on the background
                def center_mask_on_canvas(mask):
                    canvas = np.zeros((H, W), dtype=np.uint8)
                    h, w = mask.shape
                    # Ensure mask fits in canvas
                    if h > H or w > W:
                        mask = mask[:min(h, H), :min(w, W)]
                        h, w = mask.shape
                    y_off = (H - h) // 2
                    x_off = (W - w) // 2
                    canvas[y_off:y_off + h, x_off:x_off + w] = mask
                    return canvas

                # Centered and colored masks
                red = center_mask_on_canvas(mask1)
                green = center_mask_on_canvas(mask2)

                red_layer = cv2.merge([red, np.zeros_like(red), np.zeros_like(red)])
                green_layer = cv2.merge([np.zeros_like(green), green, np.zeros_like(green)])

                # Overlay masks on the resized background

                result = cv2.addWeighted(background_resized, 1.0, red_layer, 0.5, 0)
                result = cv2.addWeighted(result, 1.0, green_layer, 0.5, 0)
                # Draw a centered black square for final image rebuild crop margin
                if not crop_margin == 0:
                    top_left = (
                        int(crop_margin + 3),
                        int(crop_margin + 3)
                    )
                    bottom_right = (
                        int(min(W, W - crop_margin - 3 + self.SIZE.shifttl)),
                        int(min(H, H - crop_margin - 3 + self.SIZE.shifttl))
                    )
                    cv2.rectangle(result, top_left, bottom_right, color=(245, 66, 215), thickness=3)
                # Draw a centered black square

                left_margin = int((W - tile_grid_W) / 2)
                top_margin = int((H - tile_grid_H) / 2)
                top_left = (
                    int(left_margin),
                    int(top_margin)
                )
                bottom_right = (
                    int(W - left_margin),
                    int(H - top_margin)
                )
                cv2.rectangle(result, top_left, bottom_right, color=(0, 0, 0), thickness=2)

                # Convert result to ComfyUI tensor format
                result_tensor = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                # Ensure the tensor is the correct type and shape
                result_tensor = result_tensor.type(torch.FloatTensor)  # Ensure it's a float tensor

                # Convert to shape [B, H, W, C]
                result_tensor = result_tensor.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
                result_batch.append(result_tensor)

            result_tensor = torch.cat([result_batch[0], result_batch[1]], dim=0)
            return result_tensor
    """
    #WORKER
    @classmethod
    def neighbor_add_last_generated_to_base_image(cls, SELF, iteration, current_tile, grid_specs, current_index,
                                                       output_images, upscalemultiplier=1):
        combined = WORKER.id(SELF.INFO.tiler_id).TBG_Fusion.neighbor_add_last_generated_to_base_image(SELF.SIZE, SELF.PARAMS,
                                                        current_tile, current_index, output_images, upscalemultiplier=1)
        return combined
    """
