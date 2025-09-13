import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class HED_Preprocessor:

    def execute(self, image, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.hed import HEDdetector

        model = HEDdetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, safe = kwargs["safe"] == "enable")
        del model
        return (out, )

class Fake_Scribble_Preprocessor:

    def execute(self, image, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.hed import HEDdetector
        
        model = HEDdetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, scribble=True, safe=kwargs["safe"]=="enable")
        del model
        return (out, )

