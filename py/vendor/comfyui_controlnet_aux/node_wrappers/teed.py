import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class TEED_Preprocessor:

    def execute(self, image, safe_steps=2, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.teed import TEDDetector

        model = TEDDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, safe_steps=safe_steps)
        del model
        return (out, )

