import traceback

from .py.nodes.UpscalerRefiner.TBG_Nodes import TBG_Tiled_Upscaler_CE, TBG_Refiner_CE
from .py.nodes.UpscalerRefiner.TBG_Pipes import TBG_TilePrompter_v1, TBG_ControlNetPipeline, TBG_enrichment_pipe
from .py.utils.constants import NAMESPACE, get_name

try:
    from .py.nodes.UpscalerRefiner.TBG_Nodes_PRO import TBG_Upscaler_v1_pro, TBG_Refiner_v1_pro, EdgePadNode, TBG_masked_attention
    # NODE MAPPING
    NODE_CLASS_MAPPINGS = {

        f"{NAMESPACE}_Refiner_v1_pro": TBG_Refiner_v1_pro,
        f"{NAMESPACE}_ControlNetPipeline": TBG_ControlNetPipeline,
        f"{NAMESPACE}_TilePrompter_v1": TBG_TilePrompter_v1,
        f"{NAMESPACE}_enrichment_pipe": TBG_enrichment_pipe,
        f"{NAMESPACE}_Upscaler_v1_pro": TBG_Upscaler_v1_pro,
        f"{NAMESPACE}_masked_attention": TBG_masked_attention,
        f"{NAMESPACE}_Refiner_CE": TBG_Refiner_CE,
        f"{NAMESPACE}_Tiled_Upscaler_CE": TBG_Tiled_Upscaler_CE,

    }
    print('\033[34m[TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO] \033[92mLoaded\033[0m')
except Exception as e:
    print(f"Error message: {e}")
    traceback.print_exc()

    print('\033[34m[TBG_Tiled Upscaler and Refiner FLUX, support my work and get the PRO version TBG_Enhanced Tiled Upscaler and Refiner FLUX PRO at https://www.patreon.com/TB_LAAR ] \033[92mLoaded\033[0m')


WEB_DIRECTORY = "./web/assets/js"
# A dictionary that contains the friendly/humanly readable titles for the nodes 
# Define getter and setter functions
def _get_CATEGORY(cls):
    return cls._CATEGORY

def _set_CATEGORY(cls, value):
    cls._CATEGORY = value
    

NODE_DISPLAY_NAME_MAPPINGS = {
    key: get_name(value, getattr(value, "NAME", value.__name__), getattr(value, "SHORTCUT", "")) for key, value in NODE_CLASS_MAPPINGS.items()
}



