
from ..TBG_Helpers import ImageComplexityMap, MaskMultiplyZeroWins, MaskGrayValueScaler


def create_complexity_map(image, fusionmask, multiplier=1, min_gray=0.3, max_gray=1, blurmulti=0):
    # create complexity mask
    fnComplexityMap = ImageComplexityMap()
    fnMaskMultiplyZeroWins = MaskMultiplyZeroWins()
    fnMaskGrayValueScaler = MaskGrayValueScaler()

    blur = 128 + int(128 * blurmulti)
    area = 8 + int(8 * blurmulti)

    mask_image, complexity_mask =  fnComplexityMap.make_map( image, "method", 15, 1,
                 area, 8, "range", "fixed", 0, 1,
                 "off", 128, 0.5, 0.25, 4,
                 64, 1, False,
                 True, blur, blur,
                 "soft", 0.01, 6, 1)
    # set  complexity mask denoise clamping
    fusion_denoise_mask = fnMaskGrayValueScaler.scale_gray_values( complexity_mask, multiplier, min_gray, max_gray)[0]
    # combine fusion mask with complexity_mask
    complexity_fusion_mask_combined = fnMaskMultiplyZeroWins.multiply_masks( fusion_denoise_mask, fusionmask, "multiply", True, False)[0]
    return (complexity_fusion_mask_combined,)

def create_denoise_map(complexity_mask, fusionmask, multiplier=1, min_gray=0.3, max_gray=1, blurmulti=0):
    # create complexity mask
    fnMaskMultiplyZeroWins = MaskMultiplyZeroWins()
    complexity_fusion_mask_combined = fnMaskMultiplyZeroWins.multiply_masks( complexity_mask, fusionmask, "multiply", True, False)[0]
    return (complexity_fusion_mask_combined,)