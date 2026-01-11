from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs


class PyraCanny_Preprocessor:

    def execute(self, image, low_threshold=64, high_threshold=128, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.pyracanny import PyraCannyDetector

        return (common_annotator_call(PyraCannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )

