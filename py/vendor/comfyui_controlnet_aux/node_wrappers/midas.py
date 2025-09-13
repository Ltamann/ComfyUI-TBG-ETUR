import comfy.model_management as model_management
import numpy as np
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class MIDAS_Normal_Map_Preprocessor:

    def execute(self, image, a=np.pi * 2.0, bg_threshold=0.1, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.midas import MidasDetector

        model = MidasDetector.from_pretrained().to(model_management.get_torch_device())
        #Dirty hack :))
        cb = lambda image, **kargs: model(image, **kargs)[1]
        out = common_annotator_call(cb, image, resolution=resolution, a=a, bg_th=bg_threshold, depth_and_normal=True)
        del model
        return (out, )

class MIDAS_Depth_Map_Preprocessor:


    def execute(self, image, a=np.pi * 2.0, bg_threshold=0.1, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.midas import MidasDetector

        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        model = MidasDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, a=a, bg_th=bg_threshold)
        del model
        return (out, )
