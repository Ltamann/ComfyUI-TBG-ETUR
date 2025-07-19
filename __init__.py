# Project: TBG Refiner 
# By TBG  ¡
# All modifications Copyright 2025 TBG ToLAA
# Based on ComfyUI_MaraScott_Nodes By MaraScott (Discord: davask#4370) By MaraScott (Discord: davask#4370)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import sys

import folder_paths
from .patreon_node import register_routes
from .py.utils.version import VERSION
from .py.vendor.ComfyUI_MaraScott_Nodes.py.inc.lib.llm import MS_Llm

register_routes()

MS_Llm.prestartup_script()

python = sys.executable
p310_plus = (sys.version_info >= (3, 10))

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

from .TBG_ETUR_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from .TBG_ETUR_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from .py.nodes.UpscalerRefiner.inc.api  import NODE_CLASS_MAPPINGS as PATREON_CLASSES , NODE_DISPLAY_NAME_MAPPINGS as PATREON_NAMES
NODE_CLASS_MAPPINGS.update(PATREON_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(PATREON_NAMES)



__all__ = [ 'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY' ]

MANIFEST = {
    "name": "TBGUpscaler_Nodes",
    "version": VERSION,
    "author": "TBG ToLAA",
    "project": "https://github.com/",
    "description": "UpScaler Refiner FLUX",
}
