import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class DSINE_Normal_Map_Preprocessor:

    def execute(self, image, fov=60.0, iterations=5, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.dsine import DsineDetector

        model = DsineDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, fov=fov, iterations=iterations, resolution=resolution)
        del model
        return (out,)

