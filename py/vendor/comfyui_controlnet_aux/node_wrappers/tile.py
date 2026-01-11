from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class Tile_Preprocessor:

    def execute(self, image, pyrUp_iters, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.tile import TileDetector

        return (common_annotator_call(TileDetector(), image, pyrUp_iters=pyrUp_iters, resolution=resolution),)

class TTPlanet_TileGF_Preprocessor:

    def execute(self, image, scale_factor, blur_strength, radius, eps, **kwargs):
        from ..src.custom_controlnet_aux.tile import TTPlanet_Tile_Detector_GF

        return (common_annotator_call(TTPlanet_Tile_Detector_GF(), image, scale_factor=scale_factor, blur_strength=blur_strength, radius=radius, eps=eps),)

class TTPlanet_TileSimple_Preprocessor:

    def execute(self, image, scale_factor, blur_strength):
        from ..src.custom_controlnet_aux.tile import TTPLanet_Tile_Detector_Simple

        return (common_annotator_call(TTPLanet_Tile_Detector_Simple(), image, scale_factor=scale_factor, blur_strength=blur_strength),)

