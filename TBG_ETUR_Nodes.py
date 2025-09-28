import traceback
"""
import sys, os

# add TBG_upscaler2 folder to path
node_root = os.path.dirname(__file__)
if node_root not in sys.path:
    sys.path.insert(0, node_root)


import sys, py
print("Loaded py from:", py.__file__)
print("sys.path:", sys.path[:5])  # first few entries
"""

from .py.nodes.UpscalerRefiner.TBG_Nodes import TBG_Tiled_Upscaler_CE, TBG_Refiner_CE
from .py.nodes.UpscalerRefiner.TBG_Pipes import TBG_TilePrompter_v1, TBG_ControlNetPipeline, TBG_enrichment_pipe
from .py.utils.constants import NAMESPACE, get_name
from .py.vendor.comfyui_resharpen_main.tbgresharpen import TBG_DetailEnhancer

try:
    from .py.nodes.UpscalerRefiner.TBG_Nodes_PRO import TBG_Upscaler_v1_pro, TBG_Refiner_v1_pro, EdgePadNode, TBG_masked_attention
    from .py.nodes.UpscalerRefiner.TBG_MAGNIFIC_MAGNIFIER import TBG_magnific_ETUR
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
        f"{NAMESPACE}_magnific_ETUR": TBG_magnific_ETUR,
        f"{NAMESPACE}_DetailEnhancer": TBG_DetailEnhancer,

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

