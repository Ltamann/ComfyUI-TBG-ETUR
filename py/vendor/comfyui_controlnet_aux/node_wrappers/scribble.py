import comfy.model_management as model_management
import cv2
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, nms


class Scribble_Preprocessor:

    def execute(self, image, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.scribble import ScribbleDetector

        model = ScribbleDetector()
        return (common_annotator_call(model, image, resolution=resolution), )

class Scribble_XDoG_Preprocessor:

    def execute(self, image, threshold=32, resolution=512, **kwargs):
        from ..src.custom_controlnet_aux.scribble import ScribbleXDog_Detector

        model = ScribbleXDog_Detector()
        return (common_annotator_call(model, image, resolution=resolution, thr_a=threshold), )

class Scribble_PiDiNet_Preprocessor:

    def execute(self, image, safe="enable", resolution=512):
        def model(img, **kwargs):
            from ..src.custom_controlnet_aux.pidi import PidiNetDetector
            pidinet = PidiNetDetector.from_pretrained().to(model_management.get_torch_device())
            result = pidinet(img, scribble=True, **kwargs)
            result = nms(result, 127, 3.0)
            result = cv2.GaussianBlur(result, (0, 0), 3.0)
            result[result > 4] = 255
            result[result < 255] = 0
            return result
        return (common_annotator_call(model, image, resolution=resolution, safe=safe=="enable"),)
