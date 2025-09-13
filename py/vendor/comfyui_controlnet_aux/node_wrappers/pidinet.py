import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class PIDINET_Preprocessor:

    def execute(self, image, safe, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.pidi import PidiNetDetector

        model = PidiNetDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, safe = safe == "enable")
        del model
        return (out, )
