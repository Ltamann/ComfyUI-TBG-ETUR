import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, distance_transform_edt


def calculate_blend_params(tile_shape,
                           transition_width=None,
                           max_levels: int = 6,
                           min_deep_dim: int = 16,
                           sigma_min_default: float = 0.5):
    """
    Compute optimal pyramid levels and mask-blur sigmas based on tile size
    and desired mask ramp width.
    Returns: (levels, sigma_min, sigma_max)
    """
    H, W = tile_shape
    min_dim = min(H, W)
    if transition_width is None:
        transition_width = min_dim / 8.0
    safe_levels = int(np.floor(np.log2(min_dim / min_deep_dim)))
    levels = max(1, min(safe_levels, max_levels))
    sigma_max = max(1.0, transition_width / 4.0)
    sigma_min = sigma_min_default
    return levels, sigma_min, sigma_max

def calculate_safe_levels(image_shape):
    """Ensure pyramid levels keep smallest band ≥16 px."""
    min_dim = min(image_shape[:2])
    max_safe = int(np.log2(min_dim)) - 4
    return max(1, min(max_safe, 6))

def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gp = [img]
    current = img.copy()
    for _ in range(levels):
        if min(current.shape[:2]) < 4:
            break
        current = gaussian_filter(current, sigma=1)[::2, ::2]
        gp.append(current)
    return gp

def laplacian_pyramid(gp: list[np.ndarray]) -> list[np.ndarray]:
    lp = []
    for i in range(len(gp) - 1):
        target_shape = gp[i].shape
        expanded = np.zeros(target_shape, dtype=gp[i].dtype)
        src = gp[i + 1]
        h, w = src.shape[:2]
        expanded[:h*2:2, :w*2:2] = src
        expanded = gaussian_filter(expanded, sigma=1) * 4
        lp.append(gp[i] - expanded)
    lp.append(gp[-1])
    return lp

def reconstruct_laplacian(lp: list[np.ndarray]) -> np.ndarray:
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        target_shape = lp[i].shape
        expanded = np.zeros(target_shape, dtype=lp[i].dtype)
        h, w = img.shape[:2]
        expanded[:h*2:2, :w*2:2] = img
        expanded = gaussian_filter(expanded, sigma=1) * 4
        img = expanded + lp[i]
    return img

def gaussian_pyramid_mask(mask: np.ndarray,
                          levels: int,
                          sigma_min: float = 0.5,
                          sigma_max: float = 2.0) -> list[np.ndarray]:
    """
    Build mask pyramid with level-dependent blur:
    coarse levels use sigma_max, fine levels use sigma_min.
    """
    gp = []
    current = mask.astype(np.float32)
    for i in range(levels + 1):
        sigma = sigma_max - (sigma_max - sigma_min) * (i / levels)
        blurred = gaussian_filter(current, sigma=sigma)
        gp.append(np.clip(blurred, 0.0, 1.0))
        if i < levels:
            current = blurred[::2, ::2]
    return gp

# --- Mask preprocessing methods ---
def preprocess_mask_with_blur(mask: np.ndarray, blur_radius: float = 3.0) -> np.ndarray:
    return np.clip(gaussian_filter(mask.astype(np.float32), sigma=blur_radius), 0, 1)

def distance_feather_mask(mask: np.ndarray, feather_distance: float = 10.0) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    d_in = distance_transform_edt(binary)
    d_out = distance_transform_edt(1 - binary)
    feathered = np.zeros_like(mask, dtype=np.float32)
    feathered[d_in > feather_distance] = 1.0
    t_mask = (d_in <= feather_distance) & (binary > 0)
    feathered[t_mask] = d_in[t_mask] / feather_distance
    o_mask = (d_out <= feather_distance) & (binary == 0)
    feathered[o_mask] = 1.0 - (d_out[o_mask] / feather_distance)
    return np.clip(feathered, 0, 1)

def sigmoid_mask_smoothing(mask: np.ndarray, steepness: float = 10, midpoint: float = 0.5) -> np.ndarray:
    return 1 / (1 + np.exp(-steepness * (mask - midpoint)))

def bilateral_mask_smoothing(mask: np.ndarray, sigma_spatial: float = 5, sigma_color: float = 0.1) -> np.ndarray:
    m8 = (mask * 255).astype(np.uint8)
    sm = cv2.bilateralFilter(m8, d=9, sigmaColor=sigma_color*255, sigmaSpace=sigma_spatial)
    return (sm / 255.0).astype(np.float32)

def multiscale_mask_processing(mask: np.ndarray, scales: list[int] = [1, 2, 4]) -> np.ndarray:
    H, W = mask.shape[:2]
    if H == 0 or W == 0:
        return mask.copy()
    enhanced = np.zeros_like(mask, dtype=np.float32)
    for s in scales:
        if s > 1:
            new_h = max(1, H // s)
            new_w = max(1, W // s)
            small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = mask.copy()
        sm = gaussian_filter(small, sigma=s * 0.5)
        if s > 1:
            sm = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
        enhanced += sm * (1.0 / s)
    weight_sum = sum(1.0 / s for s in scales)
    return np.clip(enhanced / weight_sum, 0, 1)

def adaptive_sigmoid_mask(mask: np.ndarray, edge_threshold: float = 0.1) -> np.ndarray:
    from scipy.ndimage import sobel
    edges = np.hypot(sobel(mask, 0), sobel(mask, 1))
    steep = np.where(edges > edge_threshold, 5, 15)
    out = np.zeros_like(mask, dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            out[i, j] = 1 / (1 + np.exp(-steep[i, j] * (mask[i, j] - 0.5)))
    return out

def edge_aware_mask_refinement(mask: np.ndarray, image: np.ndarray, edge_threshold: float = 0.1) -> np.ndarray:
    try:
        from skimage.feature import canny
        img_gray = image.mean(axis=2) if image.ndim == 3 else image
        edges = canny(img_gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        dist = distance_transform_edt(~edges)
        sigma = np.clip(dist / 5.0, 0.5, 3.0)
        return gaussian_filter(mask, sigma=sigma)
    except ImportError:
        return gaussian_filter(mask, sigma=1.0)

def enhanced_laplacian_blend(overlay: np.ndarray,
                             background: np.ndarray,
                             mask: np.ndarray,
                             mask_preprocessing: str,
                             transition_width: float = None) -> np.ndarray:
    """
    Blend two single-channel images with Laplacian pyramid,
    using an adaptive mask pyramid built from transition_width.
    """
    levels, sigma_min, sigma_max = calculate_blend_params(overlay.shape[:2], transition_width)
    # Preprocess mask
    if mask_preprocessing == 'gaussian':
        m = preprocess_mask_with_blur(mask, blur_radius=sigma_max)
    elif mask_preprocessing == 'distance':
        m = distance_feather_mask(mask, feather_distance=sigma_max * 4)
    elif mask_preprocessing == 'sigmoid':
        m = sigmoid_mask_smoothing(mask)
    elif mask_preprocessing == 'adaptive_sigmoid':
        m = adaptive_sigmoid_mask(mask)
    elif mask_preprocessing == 'bilateral':
        m = bilateral_mask_smoothing(mask)
    elif mask_preprocessing == 'multiscale':
        m = multiscale_mask_processing(mask)
    elif mask_preprocessing == 'edge_aware':
        m = edge_aware_mask_refinement(mask, overlay)
    else:
        m = mask.copy()
    m = gaussian_filter(m, sigma=0.8)

    gp_overlay = gaussian_pyramid(overlay, levels)
    gp_background = gaussian_pyramid(background, levels)
    lp_overlay = laplacian_pyramid(gp_overlay)
    lp_background = laplacian_pyramid(gp_background)
    gp_mask = gaussian_pyramid_mask(m, levels, sigma_min, sigma_max)

    blended_pyr = []
    for ov_lap, bg_lap, M in zip(lp_overlay, lp_background, gp_mask):
        if M.size == 0:
            M = np.zeros(ov_lap.shape, dtype=M.dtype)
        elif M.shape != ov_lap.shape:
            M = cv2.resize(M, (ov_lap.shape[1], ov_lap.shape[0]), interpolation=cv2.INTER_LINEAR)
        smooth_m = gaussian_filter(M, sigma=0.3)
        blended_pyr.append(ov_lap * smooth_m + bg_lap * (1 - smooth_m))

    return np.clip(reconstruct_laplacian(blended_pyr), 0, 1)

class LaplacianCompositeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "mask": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "y": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "transition_width": ("FLOAT", {"default": None, "min": 0.0}),
                "mask_method": (
                    ["gaussian", "distance", "sigmoid", "adaptive_sigmoid",
                     "bilateral", "multiscale", "edge_aware"],
                    {"default": "multiscale"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/blending"

    def process(self,
                base_image,
                overlay_image,
                mask,
                x: int,
                y: int,
                transition_width=None,
                mask_method='multiscale'):
        # 1) Use the mask as-is (do NOT invert)
        mask_orig = mask.clone()

        def to_np(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu()
                if t.ndim == 4:
                    t = t[0]
                return t.numpy()
            return t

        # 2) Convert to numpy and normalize
        base = to_np(base_image).astype(np.float32)
        over = to_np(overlay_image).astype(np.float32)
        m = to_np(mask_orig).astype(np.float32)
        for arr in (base, over, m):
            if arr.max() > 1.0:
                arr /= 255.0

        H, W = base.shape[:2]
        h, w = over.shape[:2]

        # 3) Compute blend parameters
        levels, sigma_min, sigma_max = calculate_blend_params((h, w), transition_width)

        # 4) Crop to overlay region
        x0 = max(0, x)
        x1 = min(W, x + w)
        y0 = max(0, y)
        y1 = min(H, y + h)
        ox0 = x0 - x
        oy0 = y0 - y

        base_crop = base[y0:y1, x0:x1].copy()
        over_crop = over[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
        mask_crop = m[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]

        # 5) Squeeze and broadcast mask dims
        while mask_crop.ndim > 2 and mask_crop.shape[0] == 1:
            mask_crop = np.squeeze(mask_crop, axis=0)
        if mask_crop.ndim == 2 and base_crop.ndim == 3:
            mask_crop = np.repeat(mask_crop[..., None], base_crop.shape[2], axis=2)

        # 6) Blend inside crop: output = overlay * mask + background * (1 - mask)
        if base_crop.ndim == 3:
            out_bb = np.zeros_like(base_crop)
            for c in range(base_crop.shape[2]):
                ch_mask = mask_crop[..., c] if mask_crop.ndim == 3 else mask_crop
                out_bb[..., c] = enhanced_laplacian_blend(
                    over_crop[..., c],  # overlay first!
                    base_crop[..., c],  # background second!
                    ch_mask,
                    mask_preprocessing=mask_method,
                    transition_width=transition_width
                )
        else:
            out_bb = enhanced_laplacian_blend(
                over_crop,
                base_crop,
                mask_crop,
                mask_preprocessing=mask_method,
                transition_width=transition_width
            )

        # 7) Paste blended region back
        base[y0:y1, x0:x1] = out_bb
        base = np.clip(base, 0, 1)

        # 8) Convert back to torch tensor
        out = torch.from_numpy(base)
        if out.ndim == 3:
            out = out.unsqueeze(0)
        return (out,)