import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class OneFormer_COCO_SemSegPreprocessor:

    def semantic_segmentate(self, image, resolution=512):
        from ..src.custom_controlnet_aux.oneformer import OneformerSegmentor

        model = OneformerSegmentor.from_pretrained(filename="150_16_swin_l_oneformer_coco_100ep.pth")
        model = model.to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)

class OneFormer_ADE20K_SemSegPreprocessor:

    def semantic_segmentate(self, image, resolution=512):
        from ..src.custom_controlnet_aux.oneformer import OneformerSegmentor

        model = OneformerSegmentor.from_pretrained(filename="250_16_swin_l_oneformer_ade20k_160k.pth")
        model = model.to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)

