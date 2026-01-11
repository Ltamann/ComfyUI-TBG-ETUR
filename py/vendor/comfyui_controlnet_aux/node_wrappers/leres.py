import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class LERES_Depth_Map_Preprocessor:

    def execute(self, image, rm_nearest=0, rm_background=0, resolution=512, boost="disable", **kwargs):
        from ..src.custom_controlnet_aux.leres import LeresDetector

        model = LeresDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, thr_a=rm_nearest, thr_b=rm_background, boost=boost == "enable")
        del model
        return (out, )
    
