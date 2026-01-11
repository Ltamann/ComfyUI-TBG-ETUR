import comfy.model_management as model_management
from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs


class Depth_Anything_V2_Preprocessor:

    def execute(self, image, ckpt_name="depth_anything_v2_vitl.pth", resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector

        model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, max_depth=1)
        del model
        return (out, )

