from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs


class Canny_Edge_Preprocessor:

    def execute(self, image, low_threshold=100, high_threshold=200, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.canny import CannyDetector

        return (common_annotator_call(CannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )


