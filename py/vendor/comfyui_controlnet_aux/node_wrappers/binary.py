from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs


class Binary_Preprocessor:

    def execute(self, image, bin_threshold=100, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.binary import BinaryDetector

        return (common_annotator_call(BinaryDetector(), image, bin_threshold=bin_threshold, resolution=resolution), )


