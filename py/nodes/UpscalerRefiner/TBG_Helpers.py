
from math import gcd
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List


from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from pygments.lexer import default

from comfy_extras.nodes_mask import ImageToMask


if hasattr(ImageToMask, "execute"):
    # execute is a @classmethod; no instance needed
    def ImageToMask_execute(image,chanel):
        return ImageToMask.execute(image,chanel)
elif hasattr(ImageToMask, "image_to_mask"):
    # composite is an instance method; needs an instance
    def ImageToMask_execute(image,chanel):
        node = ImageToMask()  # or cache a global instance if you prefer
        return node.image_to_mask(image,chanel)



def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    #if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
    #    image = image.unsqueeze(-1)
    return image

# Preferred resolutions for Flux Kontext (from your list)
KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


QWENEDIT_RESOLUTIONS: List[Tuple[int, int]] = [
    (1328, 1328), (1088, 1632), (1632, 1088),
    (1152, 1536), (1536, 1152), (1008, 1792),
    (1792, 1008), (768, 2304), (2304, 768),(1024, 1024), (832, 1248), (1248, 832), (864, 1152), (1152, 864), (720, 1280), (1280, 720), (592, 1776), (1776, 592)
]

def ratio_label(w: int, h: int) -> str:
    divisor = gcd(w, h)
    rw = w // divisor
    rh = h // divisor
    approx = round(w * h / 1_000_000, 2)

    name = ""
    if rw == rh:
        name = " (square)"
    elif (rw, rh) not in [(16, 9), (9, 16), (3, 2), (2, 3), (4, 3), (3, 4), (21, 9), (9, 21)]:
        name = " (custom)"

    return f"{w}x{h} • {rw}:{rh} (≈{approx}MP){name}"

def to_labels(pairs: List[Tuple[int, int]]) -> List[str]:
    return [ratio_label(w, h) for (w, h) in pairs]

def parse_wh(label: str) -> Tuple[int, int]:
    raw = label.split("•", 1)[0].strip()
    raw = raw.lower().replace("×", "x")
    w_str, h_str = [s.strip() for s in raw.split("x", 1)]
    return int(w_str), int(h_str)

class QwenEditResolution:
    @classmethod
    def INPUT_TYPES(cls):
        labels = to_labels(QWENEDIT_RESOLUTIONS)
        return {
            "required": {
                "resolution": (labels, {
                    "default": labels[0],  # must be a single string
                    "display": "dropdown",
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = "TBG/Helpers"
    OUTPUT_NODE = False

    def run(cls, resolution: str):
        w, h = parse_wh(resolution)
        return (w, h)

class FluxKontextResolution:
    @classmethod
    def INPUT_TYPES(cls):
        labels = to_labels(KONTEXT_RESOLUTIONS)
        return {
            "required": {
                "resolution": (labels, {
                    "default": labels[0],  # must be a single string
                    "display": "dropdown",
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = "TBG/Helpers"
    OUTPUT_NODE = False

    def run(cls, resolution: str):
        w, h = parse_wh(resolution)
        return (w, h)

class IncrementBatch:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "seed": ("INT", {"label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),

            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "run"
    CATEGORY = "TBG/Helpers"
    OUTPUT_NODE = False

    def run(cls,seed):
        return (seed,)









class ImageComplexityMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["variance", "edges", "hybrid"], {"default":"variance"}),
                "window_size": ("INT", {"default": 15, "min": 3, "max": 99, "step": 2}),
                "edge_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),

                # Area/block analysis (fixed scaling)
                "area_size": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 2}),
                "area_stride": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 2}),
                "area_metric": (["variance", "std", "range"], {"default":"range"}),
                "area_scale_mode": (["fixed", "normalize", "0", "1"], {"default": "fixed"}),
                "area_max_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "area_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # Size-aware boosting
                "size_boost_mode": (["off", "gain", "difference"], {"default": "off"}),
                "ref_area": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "gain_power": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1}),
                "gain_min": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.05}),
                "gain_max": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.05}),

                # Multi-scale "difference" mode params
                "diff_large_area": ("INT", {"default": 64, "min": 2, "max": 2048, "step": 2}),
                "diff_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "diff_relu": ("BOOLEAN", {"default": False}),

                # Final softening
                "final_blur": ("BOOLEAN", {"default": True}),
                "blur_radius": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "blur_max_radius": ("INT", {"default": 128, "min": 1, "max": 255, "step": 1}),

                # Final white-threshold and ramp
                "white_mode": (["off", "hard", "soft"], {"default": "soft"}),
                "white_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "white_soft_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "white_gamma": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "make_map"
    CATEGORY = "TBG/Helpers/Mask"

    def _coerce_area_scale_mode(self, v):
        if v in (True, 1, "1", "normalize"):
            return "normalize"
        return "fixed"

    def _block_metric(self, patch, metric):
        # Var = E[x^2] - (E[x])^2; Std = sqrt(Var); Range = max - min
        if metric == "variance":
            m = patch.mean()
            s2 = (patch * patch).mean()
            return s2 - m * m
        elif metric == "std":
            return patch.std()
        else:
            return float(patch.max() - patch.min())

    def _compute_block_map(self, gray, area_size, area_stride, metric):
        h, w = gray.shape
        if area_size <= 0:
            return None
        stride = area_stride if area_stride > 0 else area_size
        out = np.zeros_like(gray, dtype=np.float32)
        for y in range(0, h, stride):
            y2 = min(y + area_size, h)
            for x in range(0, w, stride):
                x2 = min(x + area_size, w)
                patch = gray[y:y2, x:x2]
                if patch.size == 0:
                    continue
                score = self._block_metric(patch, metric)
                out[y:y2, x:x2] = score
        return out

    def _scale_fixed(self, block_raw, metric, max_override):
        if block_raw is None:
            return None
        if max_override > 0.0:
            fixed_max = float(max_override)
        else:
            fixed_max = 0.25 if metric == "variance" else (0.5 if metric == "std" else 1.0)
        return np.clip(block_raw / fixed_max, 0.0, 1.0).astype(np.float32)

    def _scale_mode(self, block_raw, metric, mode, max_override):
        if block_raw is None:
            return None
        if mode == "normalize":
            mn, mx = float(block_raw.min()), float(block_raw.max())
            if mx > mn:
                return (block_raw - mn) / (mx - mn)
            return np.zeros_like(block_raw, dtype=np.float32)
        return self._scale_fixed(block_raw, metric, max_override)

    def _apply_size_gain(self, area_map, area_size, ref_area, power, gmin, gmax):
        # Gain increases as area_size decreases
        a = max(1, int(area_size))
        gain = (float(ref_area) / float(a)) ** float(power)
        gain = float(np.clip(gain, gmin, gmax))
        return np.clip(area_map * gain, 0.0, 1.0)

    def _final_gaussian_blur(self, m, blur_radius, blur_max_radius, area_size):
        proposed = (area_size // 2) if (blur_radius <= 0 and area_size > 0) else blur_radius
        r = max(0, min(int(proposed), int(blur_max_radius)))
        if r <= 0:
            return m
        k = 2 * r + 1
        return cv2.GaussianBlur(m, (k, k), 0)

    def _ramp_to_white(self, m, thr, mode, strength, gamma):
        # m is float32 in [0,1]
        if mode == "hard":
            # Set values >= thr directly to white
            return np.where(m >= thr, 1.0, m).astype(np.float32)
        elif mode == "soft":
            # Push values above thr toward white with a non-linear gamma ramp
            out = m.copy()
            above = m >= thr
            denom = max(1e-6, (1.0 - thr))
            t = np.zeros_like(m, dtype=np.float32)
            t[above] = (m[above] - thr) / denom
            t = np.clip(t, 0.0, 1.0)
            t = np.power(t, float(gamma))  # gamma < 1 ramps faster to white
            out[above] = m[above] + float(strength) * (1.0 - m[above]) * t[above]
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        return m

    def make_map(self, image, method, window_size, edge_strength,
                 area_size, area_stride, area_metric, area_scale_mode, area_max_override, area_strength,
                 size_boost_mode, ref_area, gain_power, gain_min, gain_max,
                 diff_large_area, diff_weight, diff_relu,
                 final_blur, blur_radius, blur_max_radius,
                 white_mode, white_threshold, white_soft_strength, white_gamma):

        image = tensor_to_pil(image, batch_index=0)

        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Base map (variance/edges/hybrid)
        complexity_map = np.zeros_like(gray, dtype=np.float32)
        if method in ["variance", "hybrid"]:
            mean = cv2.GaussianBlur(gray, (window_size, window_size), 0)
            sqmean = cv2.GaussianBlur(gray * gray, (window_size, window_size), 0)
            var_map = sqmean - mean * mean
            var_map = cv2.normalize(var_map, None, 0, 1, cv2.NORM_MINMAX)
            complexity_map = var_map

        if method in ["edges", "hybrid"]:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200).astype(np.float32)
            edges = cv2.GaussianBlur(edges, (5, 5), 0)
            edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
            if method == "edges":
                complexity_map = edges
            else:
                complexity_map = (1 - edge_strength) * complexity_map + edge_strength * edges

        # Area map(s)
        mode = self._coerce_area_scale_mode(area_scale_mode)

        # Small-scale (requested) area map
        small_raw = self._compute_block_map(gray, area_size, area_stride, area_metric)
        small_map = self._scale_mode(small_raw, area_metric, mode, area_max_override)

        # Optional large-scale area map for difference mode
        large_map = None
        if size_boost_mode == "difference":
            la = max(int(diff_large_area), 2)
            large_raw = self._compute_block_map(gray, la, la, area_metric)  # typically non-overlapping
            large_map = self._scale_mode(large_raw, area_metric, mode, area_max_override)

        # Size-aware adjustments
        if small_map is not None:
            if size_boost_mode == "gain":
                small_map = self._apply_size_gain(small_map, area_size, ref_area, gain_power, gain_min, gain_max)
            elif size_boost_mode == "difference" and large_map is not None:
                # Promote details present at small scale but not at large scale
                diff = small_map - float(diff_weight) * large_map
                if diff_relu:
                    diff = np.maximum(diff, 0.0)
                small_map = np.clip(diff, 0.0, 1.0)

            # Blend with base
            complexity_map = (1.0 - area_strength) * complexity_map + area_strength * small_map

        # Final softening with bounded Gaussian blur
        if final_blur:
            complexity_map = self._final_gaussian_blur(complexity_map, blur_radius, blur_max_radius, area_size)

        # Final white-threshold/ramp in float space
        if white_mode != "off":
            complexity_map = self._ramp_to_white(
                complexity_map,
                float(white_threshold),
                white_mode,
                float(white_soft_strength),
                float(white_gamma),
            )

        out = Image.fromarray((np.clip(complexity_map, 0, 1) * 255).astype(np.uint8))
        out = pil_to_tensor(out)
        image = out.unsqueeze(-1)  # shape becomes (1, 1024, 1024, 1)
        image = image.repeat(1, 1, 1, 3)  # shape becomes (1, 1024, 1024, 3)
        mask = ImageToMask_execute( image, "green")[0]
        return (image, mask,)



class MaskMultiplyZeroWins:
    """
    Custom ComfyUI node for multiplying masks where 0 values always win.
    Includes option for invert-multiply-invert logic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "operation": (["multiply", "invert_multiply_invert"],),
                "clamp_result": ("BOOLEAN", {"default": True}),
                "round_result": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "multiply_masks"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Multiplies two masks where 0 values always win"

    def multiply_masks(self, mask1, mask2, operation, clamp_result, round_result):
        # Ensure masks are the same size
        if mask1.shape != mask2.shape:
            # Resize mask2 to match mask1
            mask2 = torch.nn.functional.interpolate(
                mask2.unsqueeze(0).unsqueeze(0),
                size=(mask1.shape[-2], mask1.shape[-1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        if operation == "multiply":
            # Standard multiplication - 0 values naturally win
            result = mask1 * mask2

        elif operation == "invert_multiply_invert":
            # Invert both masks, multiply, then invert result
            inverted_mask1 = 1.0 - mask1
            inverted_mask2 = 1.0 - mask2
            multiplied = inverted_mask1 * inverted_mask2
            result = 1.0 - multiplied

        # Apply post-processing options
        if clamp_result:
            result = torch.clamp(result, 0.0, 1.0)

        if round_result:
            result = torch.round(result)

        return (result,)


class MaskMultiplyAdvanced:
    """
    Advanced version with more options for mask multiplication
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "operation": ([
                                  "multiply",
                                  "invert_multiply_invert",
                                  "multiply_with_threshold",
                                  "weighted_multiply"
                              ],),
            },
            "optional": {
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "weight1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "weight2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "clamp_result": ("BOOLEAN", {"default": True}),
                "round_result": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process_masks"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Advanced mask multiplication with multiple operation modes"

    def process_masks(self, mask1, mask2, operation, threshold=0.5,
                      weight1=1.0, weight2=1.0, clamp_result=True, round_result=False):

        # Ensure masks are the same size
        if mask1.shape != mask2.shape:
            mask2 = torch.nn.functional.interpolate(
                mask2.unsqueeze(0).unsqueeze(0),
                size=(mask1.shape[-2], mask1.shape[-1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        if operation == "multiply":
            result = mask1 * mask2

        elif operation == "invert_multiply_invert":
            inverted_mask1 = 1.0 - mask1
            inverted_mask2 = 1.0 - mask2
            multiplied = inverted_mask1 * inverted_mask2
            result = 1.0 - multiplied

        elif operation == "multiply_with_threshold":
            # Zero out values below threshold before multiplication
            mask1_thresh = torch.where(mask1 < threshold, 0.0, mask1)
            mask2_thresh = torch.where(mask2 < threshold, 0.0, mask2)
            result = mask1_thresh * mask2_thresh

        elif operation == "weighted_multiply":
            # Apply weights before multiplication
            weighted_mask1 = mask1 * weight1
            weighted_mask2 = mask2 * weight2
            result = weighted_mask1 * weighted_mask2

        # Apply post-processing
        if clamp_result:
            result = torch.clamp(result, 0.0, 1.0)

        if round_result:
            result = torch.round(result)

        return (result,)


import numpy as np


import torch.nn.functional as F



def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """Create a 2D Gaussian kernel for blurring"""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def apply_gaussian_blur(tensor, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur using 2D convolution"""
    if kernel_size == 0:
        return tensor

    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape [1,1,k,k]

    # Ensure tensor has correct dimensions
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)  # [B,1,H,W]

    padding = kernel_size // 2
    blurred = F.conv2d(tensor, kernel, padding=padding)

    # Return to original shape
    return blurred.squeeze()


class MaskUpdateInpaintFromConditioning:
    """
    Updates inpainting mask by painting white (1) in areas where conditioning mask
    has values between 0 and a specified threshold.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainting_mask": ("MASK",),
                "conditioning_mask": ("MASK",),
                "threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "mode": ([
                             "paint_white",
                             "add_to_existing",
                             "union",
                             "replace_only_range"
                         ], {"default": "paint_white"}),
                "smooth_transition": ("BOOLEAN", {"default": False}),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("updated_inpainting_mask",)
    FUNCTION = "update_inpainting_mask"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Paints white in inpainting mask where conditioning mask values are 0 to threshold"

    def update_inpainting_mask(self, inpainting_mask, conditioning_mask, threshold,
                               mode="paint_white", smooth_transition=False, feather_amount=0):

        # Ensure masks are 2D (remove batch dimension if present)
        if len(inpainting_mask.shape) == 3:
            inpainting_mask = inpainting_mask.squeeze(0)
        if len(conditioning_mask.shape) == 3:
            conditioning_mask = conditioning_mask.squeeze(0)

        # Resize conditioning mask to match inpainting mask dimensions
        if inpainting_mask.shape != conditioning_mask.shape:
            conditioning_mask = F.interpolate(
                conditioning_mask.unsqueeze(0).unsqueeze(0),
                size=(inpainting_mask.shape[-2], inpainting_mask.shape[-1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Create condition: areas where conditioning mask has values 0 to threshold
        condition_mask = (conditioning_mask >= 0.0) & (conditioning_mask <= threshold)
        condition_mask = condition_mask.float()

        # Apply feathering if requested
        if feather_amount > 0:
            kernel_size = feather_amount * 2 + 1
            sigma = feather_amount / 3.0
            condition_mask = apply_gaussian_blur(condition_mask, kernel_size, sigma)

        # Update inpainting mask based on mode
        updated_mask = inpainting_mask.clone()

        if mode == "paint_white":
            # Paint white (1) where condition is true
            updated_mask[condition_mask > 0.5] = 1.0

        elif mode == "add_to_existing":
            # Add condition areas to existing mask
            updated_mask = updated_mask + condition_mask
            updated_mask = torch.clamp(updated_mask, 0.0, 1.0)

        elif mode == "union":
            # Union operation: max of both masks where condition is true
            updated_mask = torch.max(updated_mask, condition_mask)

        elif mode == "replace_only_range":
            # Only replace areas that fall within the conditioning range
            if smooth_transition:
                # Gradual transition based on conditioning values
                range_factor = torch.clamp(conditioning_mask / threshold, 0.0, 1.0)
                updated_mask = torch.where(condition_mask > 0.5, range_factor, updated_mask)
            else:
                updated_mask = torch.where(condition_mask > 0.5, 1.0, updated_mask)

        # Ensure output is in correct format (add batch dimension back)
        if len(updated_mask.shape) == 2:
            updated_mask = updated_mask.unsqueeze(0)

        return (updated_mask,)


class MaskConditionalPaint:
    """
    Simplified version focusing on the core functionality
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainting_mask": ("MASK",),
                "conditioning_mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "paint_mask"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Simple: Paint white where conditioning mask is 0 to threshold"

    def paint_mask(self, inpainting_mask, conditioning_mask, threshold):
        # Handle dimensions
        if len(inpainting_mask.shape) == 3:
            inpainting_mask = inpainting_mask.squeeze(0)
        if len(conditioning_mask.shape) == 3:
            conditioning_mask = conditioning_mask.squeeze(0)

        # Resize if needed
        if inpainting_mask.shape != conditioning_mask.shape:
            conditioning_mask = F.interpolate(
                conditioning_mask.unsqueeze(0).unsqueeze(0),
                size=inpainting_mask.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # Find areas in conditioning mask between 0 and threshold
        paint_areas = (conditioning_mask >= 0.0) & (conditioning_mask <= threshold)

        # Copy inpainting mask and paint white where condition is true
        result = inpainting_mask.clone()
        result[paint_areas] = 1.0

        # Restore batch dimension if needed
        if len(result.shape) == 2:
            result = result.unsqueeze(0)

        return (result,)


import torch



class MaskToDenoiseInterpolator:
    """
    Multiplies mask gray values by denoise value.
    Denoise 1.0 = no change, Denoise 0.0 = black mask
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "multiply_mask"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Multiplies mask gray values by denoise value"

    def multiply_mask(self, mask, denoise):
        # Simply multiply mask by denoise value
        result_mask = mask * denoise
        return (result_mask,)


import torch


class MaskGrayValueScaler:
    """
    Scales mask gray values between min_gray and max_gray based on multiplier.
    Prevents going to pure black and allows controlled gray value ranges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_gray": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_gray": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "scale_gray_values"
    CATEGORY = "TBG/Helpers/Mask"
    DESCRIPTION = "Scales mask gray values between min and max based on multiplier"

    def scale_gray_values(self, mask, multiplier, min_gray, max_gray):
        # Ensure min_gray <= max_gray
        out = 1.0 - mask
        if min_gray > max_gray:
            mask = 1.0 - mask
            min_gray, max_gray = max_gray, min_gray

        # When multiplier = 0: everything becomes min_gray
        # When multiplier = 1: values scale from min_gray to max_gray based on original mask

        # Scale original mask values to the min_gray to max_gray range
        scaled_range = max_gray - min_gray
        target_mask = min_gray + mask * scaled_range

        # Interpolate between min_gray (at multiplier=0) and target_mask (at multiplier=1)
        result_mask = min_gray + multiplier * (target_mask - min_gray)

        # Clamp to valid range
        result_mask = torch.clamp(result_mask, 0.0, 1.0)

        return (result_mask,)



# Node mapping



NODE_CLASS_MAPPINGS["MaskGrayValueScaler"] = MaskGrayValueScaler
NODE_CLASS_MAPPINGS["FluxKontextResolution"] = FluxKontextResolution
NODE_CLASS_MAPPINGS["QwenEditResolution"] = QwenEditResolution
NODE_CLASS_MAPPINGS["IncrementBatch"] = IncrementBatch
NODE_CLASS_MAPPINGS["ImageComplexityMap"] = ImageComplexityMap
NODE_CLASS_MAPPINGS["MaskMultiplyZeroWins"] = MaskMultiplyZeroWins
NODE_CLASS_MAPPINGS["MaskMultiplyAdvanced"] = MaskMultiplyAdvanced
NODE_CLASS_MAPPINGS["MaskUpdateInpaintFromConditioning"] = MaskUpdateInpaintFromConditioning
NODE_CLASS_MAPPINGS["MaskConditionalPaint"] = MaskConditionalPaint
NODE_CLASS_MAPPINGS["MaskToDenoiseInterpolator"] = MaskToDenoiseInterpolator

NODE_DISPLAY_NAME_MAPPINGS["FluxKontextResolution"] = "Flux Kontext Resolution"
NODE_DISPLAY_NAME_MAPPINGS["QwenEditResolution"] = "Qwen Edit Resolution"
NODE_DISPLAY_NAME_MAPPINGS["IncrementBatch"] = "IncrementBatch"
NODE_DISPLAY_NAME_MAPPINGS["ImageComplexityMap"] = "Image Complexity Map"
NODE_DISPLAY_NAME_MAPPINGS["MaskMultiplyZeroWins"] = "Mask Multiply Zero Wins"
NODE_DISPLAY_NAME_MAPPINGS["MaskMultiplyAdvanced"] = "MaskMultiplyAdvanced"
NODE_DISPLAY_NAME_MAPPINGS["MaskUpdateInpaintFromConditioning"] = "Mask Hard Blend"
NODE_DISPLAY_NAME_MAPPINGS["MaskConditionalPaint"] = "MaskConditionalPaint"
NODE_DISPLAY_NAME_MAPPINGS["MaskToDenoiseInterpolator"] = "Complexity Mask to Denoise"
NODE_DISPLAY_NAME_MAPPINGS["MaskGrayValueScaler"] = "Complexity Mask to Denoise Min-Max"

