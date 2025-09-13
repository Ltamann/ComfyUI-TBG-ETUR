from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs


class Color_Preprocessor:

    def execute(self, image, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.color import ColorDetector

        return (common_annotator_call(ColorDetector(), image, resolution=resolution), )


