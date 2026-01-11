from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT

class ImageLuminanceDetector:

    def execute(self, image, gamma_correction=1.0, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.recolor import Recolorizer
        return (common_annotator_call(Recolorizer(), image, mode="luminance", gamma_correction=gamma_correction , resolution=resolution), )

class ImageIntensityDetector:

    def execute(self, image, gamma_correction=1.0, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.recolor import Recolorizer
        return (common_annotator_call(Recolorizer(), image, mode="intensity", gamma_correction=gamma_correction , resolution=resolution), )

