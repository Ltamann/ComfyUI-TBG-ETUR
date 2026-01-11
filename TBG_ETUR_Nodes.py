import traceback

from .py.nodes.UpscalerRefiner.TBG_Nodes_CE import TBG_ETUR_Upscaler_and_Tile_Generator_CE, TBG_ETUR_Refiner_CE
from .py.nodes.UpscalerRefiner.TBG_Pipes import TBG_TilePrompter_v1, TBG_ControlNetPipeline, TBG_enrichment_pipe
from .py.utils.constants import NAMESPACE, get_name
from .py.vendor.comfyui_resharpen_main.tbgresharpen import TBG_DetailEnhancer

try:
    from .py.nodes.UpscalerRefiner.TBG_Nodes_PRO import TBG_ETUR_Upscaler_and_Tile_Generator_PRO, TBG_ETUR_Refiner_PRO, TBG_ETUR_Labs_Refiner,TBG_ETUR_Labs_Upscaler
    #from .py.nodes.UpscalerRefiner.TBG_MAGNIFIC_MAGNIFIER import TBG_magnific_ETUR

    # NODE MAPPING
    NODE_CLASS_MAPPINGS = {

        f"{NAMESPACE} ETUR Labs Upscaler": TBG_ETUR_Labs_Upscaler,
        f"{NAMESPACE} ETUR Refiner PRO": TBG_ETUR_Refiner_PRO,
        f"{NAMESPACE} ETUR Control Net Pipeline": TBG_ControlNetPipeline,
        f"{NAMESPACE} ETUR Tile Overrides": TBG_TilePrompter_v1,
        f"{NAMESPACE} ETUR enrichment pipe": TBG_enrichment_pipe,
        f"{NAMESPACE} ETUR Upscaler and Tile Generator PRO": TBG_ETUR_Upscaler_and_Tile_Generator_PRO,
        f"{NAMESPACE} ETUR Refiner CE": TBG_ETUR_Refiner_CE,
        f"{NAMESPACE} ETUR Upscaler and Tile Generator CE": TBG_ETUR_Upscaler_and_Tile_Generator_CE,
        #f"{NAMESPACE} ETUR Magnific Magnifier": TBG_magnific_ETUR,
        f"{NAMESPACE} Detail Enhancer": TBG_DetailEnhancer,
        f"{NAMESPACE} ETUR Labs for Refiner": TBG_ETUR_Labs_Refiner,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        f"{NAMESPACE} ETUR Labs Upscaler": "TBG ETUR Labs Upscaler",
        f"{NAMESPACE} ETUR Refiner PRO": "TBG ETUR Refiner PRO",
        f"{NAMESPACE} ETUR Control Net Pipeline": "TBG ETUR ControlNet Pipeline",
        f"{NAMESPACE} ETUR Tile Overrides": "TBG ETUR Tile Overrides",
        f"{NAMESPACE} ETUR enrichment pipe": "TBG ETUR Enrichment Pipe",
        f"{NAMESPACE} ETUR Upscaler and Tile Generator PRO": "TBG ETUR Upscaler and Tile Generator PRO",
        f"{NAMESPACE} ETUR Refiner CE": "TBG ETUR Refiner CE",
        f"{NAMESPACE} ETUR Upscaler and Tile Generator CE": "TBG ETUR Upscaler and Tile Generator CE",
        #f"{NAMESPACE} ETUR Magnific Magnifier": "TBG ETUR Magnific Magnifier",
        f"{NAMESPACE} Detail Enhancer": "Detail Enhancer",
        f"{NAMESPACE} ETUR Labs for Refiner": "TBG ETUR Labs for Refiner",
    }
    print('\033[34m[TBG ETUR Enhanced Tiled Upscaler and Refiner PRO] \033[92mLoaded\033[0m')
except Exception as e:
    print(f"Error message: {e}")
    traceback.print_exc()




WEB_DIRECTORY = "./web/assets/js"


"""
NODE_DISPLAY_NAME_MAPPINGS = {
    key: get_name(value, getattr(value, "NAME", value.__name__), getattr(value, "SHORTCUT", "")) for key, value in NODE_CLASS_MAPPINGS.items()
}
"""

