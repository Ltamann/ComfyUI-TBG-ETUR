import sys

import comfy.model_management as model_management
from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, run_script


def install_deps():
    try:
        import sklearn
    except:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'scikit-learn'])

class DiffusionEdge_Preprocessor:

    def execute(self, image, environment="indoor", patch_batch_size=4, resolution=512, **kwargs):
        install_deps()
        from ..src.custom_controlnet_aux.diffusion_edge import DiffusionEdgeDetector

        model = DiffusionEdgeDetector \
            .from_pretrained(filename = f"diffusion_edge_{environment}.pt") \
            .to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, patch_batch_size=patch_batch_size)
        del model
        return (out, )

