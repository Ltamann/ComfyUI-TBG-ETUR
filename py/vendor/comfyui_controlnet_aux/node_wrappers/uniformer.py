import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class Uniformer_SemSegPreprocessor:

    def semantic_segmentate(self, image, resolution=512):
        from ..src.custom_controlnet_aux.uniformer import UniformerSegmentor

        model = UniformerSegmentor.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )

