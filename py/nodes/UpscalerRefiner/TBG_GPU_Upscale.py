import folder_paths

from .inc.image import TBG_Image
from .TBG_Nodes_PRO import TBG_ETUR_ColorCorrection
from .TBG_Refiner import TBG_Refiner_v1


class TBG_ETUR_Upscale_Image_GPU_Using_Model:
    CATEGORY = "TBG/ETUR Tiled Upscaler and Refiner"
    DESCRIPTION = "Standalone GPU-first upscale node for ESRGAN/RealESRGAN-style upscale models without modifying ComfyUI native code."
    FUNCTION = "fn"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": (
                    folder_paths.get_filename_list("upscale_models"),
                    {"label": "Upscale Model"},
                ),
                "image": ("IMAGE", {"label": "Image"}),
                "Color_Match": (
                    TBG_ETUR_ColorCorrection.COLOR_MATCH_METHODS,
                    {
                        "label": "Tile Color Match",
                        "default": TBG_Refiner_v1.COLOR_STABILIZER_METHOD,
                        "tooltip": "Applied per upscaled tile before stitch/composite when the GPU node falls back to tiled processing.",
                    },
                ),
                "Color_Match_Str": (
                    "FLOAT",
                    {
                        "label": "Tile Color Match Strength",
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
        }

    @classmethod
    def fn(cls, image, upscale_model, Color_Match, Color_Match_Str):
        upscaled = TBG_Image.upscale_with_model_gpu_first(
            image,
            upscale_model,
            color_match_method=Color_Match,
            color_match_strength=Color_Match_Str,
        )
        return (upscaled,)
