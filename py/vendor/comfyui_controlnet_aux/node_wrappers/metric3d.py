import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, MAX_RESOLUTION


class Metric3D_Depth_Map_Preprocessor:

    def execute(self, image, backbone="vit-small", fx=1000, fy=1000, resolution=512):
        from ..src.custom_controlnet_aux.metric3d import Metric3DDetector
        model = Metric3DDetector.from_pretrained(filename=f"metric_depth_{backbone.replace('-', '_')}_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[0]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )

class Metric3D_Normal_Map_Preprocessor:

    def execute(self, image, backbone="vit-small", fx=1000, fy=1000, resolution=512):
        from ..src.custom_controlnet_aux.metric3d import Metric3DDetector
        model = Metric3DDetector.from_pretrained(filename=f"metric_depth_{backbone.replace('-', '_')}_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[1]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )
