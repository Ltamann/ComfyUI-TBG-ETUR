from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class Lineart_Standard_Preprocessor:

    def execute(self, image, guassian_sigma=6, intensity_threshold=8, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.lineart_standard import LineartStandardDetector
        return (common_annotator_call(LineartStandardDetector(), image, guassian_sigma=guassian_sigma, intensity_threshold=intensity_threshold, resolution=resolution), )

