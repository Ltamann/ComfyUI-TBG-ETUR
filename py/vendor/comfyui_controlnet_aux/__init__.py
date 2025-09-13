import importlib
import os
import sys
import traceback
from pathlib import Path

#from .hint_image_enchance import NODE_CLASS_MAPPINGS as HIE_NODE_CLASS_MAPPINGS
#from .hint_image_enchance import NODE_DISPLAY_NAME_MAPPINGS as HIE_NODE_DISPLAY_NAME_MAPPINGS
#from .log import log, blue_text, cyan_text, get_summary, get_label
from .utils import here, define_preprocessor_inputs, INPUT


#Enable CPU fallback for ops not being supported by MPS like upsample_bicubic2d.out
#https://github.com/pytorch/pytorch/issues/77764
#https://github.com/Fannovel16/comfyui_controlnet_aux/issues/2#issuecomment-1763579485
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", '1')


def load_nodes():
    shorted_errors = []
    full_error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in (here / "node_wrappers").iterdir():
        module_name = filename.stem
        if module_name.startswith('.'): continue #Skip hidden files created by the OS (e.g. [.DS_Store](https://en.wikipedia.org/wiki/.DS_Store))
        try:
            module = importlib.import_module(
                f".node_wrappers.{module_name}", package=__package__
            )
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            #log.debug(f"Imported {module_name} nodes")

        except AttributeError:
            pass  # wip nodes
        except Exception:
            error_message = traceback.format_exc()
            full_error_messages.append(error_message)
            error_message = error_message.splitlines()[-1]
            shorted_errors.append(
                f"Failed to import module {module_name} because {error_message}"
            )
    
    if len(shorted_errors) > 0:
        full_err_log = '\n\n'.join(full_error_messages)
        print(f"\n\nFull error log from comfyui_controlnet_aux: \n{full_err_log}\n\n")

    return node_class_mappings, node_display_name_mappings

AUX_NODE_MAPPINGS, AUX_DISPLAY_NAME_MAPPINGS = load_nodes()

#For nodes not mapping image to image or has special requirements
AIO_NOT_SUPPORTED = []
AIO_NOT_SUPPORTED += []
AIO_NOT_SUPPORTED += []

def preprocessor_options():
    auxs = list(AUX_NODE_MAPPINGS.keys())
    auxs.insert(0, "none")
    for name in AIO_NOT_SUPPORTED:
        if name in auxs:
            auxs.remove(name)
    return auxs


PREPROCESSOR_OPTIONS = preprocessor_options()

class AIO_Preprocessor:

    def execute(self, preprocessor, image, resolution=512):
        if preprocessor == "none":
            return (image, )
        else:
            aux_class = AUX_NODE_MAPPINGS[preprocessor]
            input_types = aux_class.INPUT_TYPES()
            input_types = {
                **input_types["required"],
                **(input_types["optional"] if "optional" in input_types else {})
            }
            params = {}
            for name, input_type in input_types.items():
                if name == "image":
                    params[name] = image
                    continue

                if name == "resolution":
                    params[name] = resolution
                    continue

                if len(input_type) == 2 and ("default" in input_type[1]):
                    params[name] = input_type[1]["default"]
                    continue

                default_values = { "INT": 0, "FLOAT": 0.0 }
                if type(input_type[0]) is list:
                    for input_type_value in input_type[0]:
                        if input_type_value in default_values:
                            params[name] = default_values[input_type[0]]
                else:
                    if input_type[0] in default_values:
                        params[name] = default_values[input_type[0]]

            return getattr(aux_class(), aux_class.FUNCTION)(**params)

class ControlNetAuxSimpleAddText:

    def execute(self, image, text):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import torch

        font = ImageFont.truetype(str((here / "NotoSans-Regular.ttf").resolve()), 40)
        img = Image.fromarray(image[0].cpu().numpy().__mul__(255.).astype(np.uint8))
        ImageDraw.Draw(img).text((0,0), text, fill=(0,255,0), font=font)
        return (torch.from_numpy(np.array(img)).unsqueeze(0) / 255.,)

class ExecuteAllControlNetPreprocessors:

    def execute(self, image, resolution=512):
        try:
            from comfy_execution.graph_utils import GraphBuilder
        except:
            raise RuntimeError("ExecuteAllControlNetPreprocessor requries [Execution Model Inversion](https://github.com/comfyanonymous/ComfyUI/commit/5cfe38). Update ComfyUI/SwarmUI to get this feature")
        
        graph = GraphBuilder()
        curr_outputs = []
        for preprocc in PREPROCESSOR_OPTIONS:
            preprocc_node = graph.node("AIO_Preprocessor", preprocessor=preprocc, image=image, resolution=resolution)
            hint_img = preprocc_node.out(0)
            add_text_node = graph.node("ControlNetAuxSimpleAddText", image=hint_img, text=preprocc)
            curr_outputs.append(add_text_node.out(0))
        
        while len(curr_outputs) > 1:
            _outputs = []
            for i in range(0, len(curr_outputs), 2):
                if i+1 < len(curr_outputs):
                    image_batch = graph.node("ImageBatch", image1=curr_outputs[i], image2=curr_outputs[i+1])
                    _outputs.append(image_batch.out(0))
                else:
                    _outputs.append(curr_outputs[i])
            curr_outputs = _outputs

        return {
            "result": (curr_outputs[0],),
            "expand": graph.finalize(),
        }

class ControlNetPreprocessorSelector:

    def get_preprocessor(self, preprocessor: str):
        return (preprocessor,)
