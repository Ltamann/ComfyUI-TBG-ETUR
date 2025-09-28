# Project: TBG ETUR

import os
import sys

# Ensure the parent folder of 'py' is in sys.path so relative imports work
node_root = os.path.dirname(os.path.dirname(__file__))  # parent of 'py'
if node_root not in sys.path:
    sys.path.insert(0, node_root)

import folder_paths

from .patreon_node import register_routes
from .py.utils.version import VERSION
#from .py.vendor.ComfyUI_MaraScott_Nodes.py.inc.lib.llm import MS_Llm
#from .tbg_min_sampler import register_res_sde_samplers
#from .tbg_sampler_eta import add_tbg_rk_samplers
#add_tbg_rk_samplers()

register_routes()
#MS_Llm.prestartup_script()

python = sys.executable
p310_plus = (sys.version_info >= (3, 12))

__ROOT__file__ = __file__


# Directory where you want to save the file
base_dir = os.path.abspath(os.path.dirname(__ROOT__file__))
root_dir = os.path.join(base_dir, "..", "..")
web_dir = os.path.join(root_dir, "web", "extensions", "TBG")
web_dir = os.path.realpath(web_dir)
if not os.path.exists(web_dir):
    os.makedirs(web_dir)
__WEB_DIR__ = web_dir

sessions_dir = os.path.join(web_dir, "sessions")
if not os.path.exists(sessions_dir):
    os.makedirs(sessions_dir)
__SESSIONS_DIR__ = sessions_dir

profiles_dir = os.path.join(web_dir, "profiles")
if not os.path.exists(profiles_dir):
    os.makedirs(profiles_dir)
__PROFILES_DIR__ = profiles_dir

tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
if not os.path.exists(tbg_temp_dir):
    os.makedirs(tbg_temp_dir)
__TGB_TEMP__ = tbg_temp_dir

cache_dir = os.path.join(tbg_temp_dir, "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
__CACHE_DIR__ = cache_dir

from .py.nodes.UpscalerRefiner.TBG_Helpers import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from .TBG_ETUR_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from .py.nodes.UpscalerRefiner.inc.api  import NODE_CLASS_MAPPINGS as PATREON_CLASSES , NODE_DISPLAY_NAME_MAPPINGS as PATREON_NAMES
NODE_CLASS_MAPPINGS.update(PATREON_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(PATREON_NAMES)




__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",

]

MANIFEST = {
    "name": "TGB ETUR Enhanced Tiled Upscaler and Refiner PRO",
    "version": VERSION,
    "author": "TBG Think build Generate - Tobias Laarmann",
    "project": "https://github.com/Ltamann/ComfyUI-TBG-ETUR",
    "description": "TGB ETUR Enhanced Tiled Upscaler and Refiner PRO",

}

