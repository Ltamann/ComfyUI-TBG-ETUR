# TBG ETUR Custom Node - Global UTF-8 Fix for Windows
import atexit
import logging
import os
import sys

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

TBG_OK = "✓"
TBG_ERROR = "✗"
TBG_WARN = "⚠"

logging.getLogger("TBG_APP").addFilter(
    lambda record: setattr(record, "msg", str(record.msg).encode("ascii", "replace").decode("ascii")) or True
)

# Ensure the parent folder of 'py' is in sys.path so relative imports work
node_root = os.path.dirname(os.path.dirname(__file__))
if node_root not in sys.path:
    sys.path.insert(0, node_root)

import folder_paths

from .py.api import tbg_gguf_install_routes  # noqa: F401
from .py.nodes.UpscalerRefiner.inc.flux2_sampler_registry import register_tbg_flux2_sampler
from .py.nodes.UpscalerRefiner.inc.pid_sde_sampler_registry import register_tbg_pid_sde_samplers
from .py.utils.version import VERSION

register_tbg_pid_sde_samplers()
register_tbg_flux2_sampler()

# Pinned v3 API guard
try:
    from comfy_api.v0_0_2 import ComfyExtension, io, ui  # noqa: F401
except Exception as e:
    raise ImportError(
        "ComfyUI-TBG-ETUR requires ComfyUI API v0_0_2. "
        "Expected import: from comfy_api.v0_0_2 import ComfyExtension, io, ui"
    ) from e


python = sys.executable
p310_plus = sys.version_info >= (3, 12)

__ROOT__file__ = __file__

base_dir = os.path.abspath(os.path.dirname(__ROOT__file__))
root_dir = os.path.join(base_dir, "..", "..")

web_dir = os.path.join(root_dir, "web", "extensions", "TBG")
web_dir = os.path.realpath(web_dir)
os.makedirs(web_dir, exist_ok=True)
__WEB_DIR__ = web_dir

sessions_dir = os.path.join(web_dir, "sessions")
os.makedirs(sessions_dir, exist_ok=True)
__SESSIONS_DIR__ = sessions_dir

profiles_dir = os.path.join(web_dir, "profiles")
os.makedirs(profiles_dir, exist_ok=True)
__PROFILES_DIR__ = profiles_dir

tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
os.makedirs(tbg_temp_dir, exist_ok=True)
__TGB_TEMP__ = tbg_temp_dir

cache_dir = os.path.join(tbg_temp_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)
__CACHE_DIR__ = cache_dir

from .py.v3.extension import comfy_entrypoint

WEB_DIRECTORY = "./web/assets/js"

__all__ = [
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]

MANIFEST = {
    "name": "TGB ETUR Enhanced Tiled Upscaler and Refiner PRO",
    "version": VERSION,
    "author": "TBG Think Build Generate - Tobias Laarmann",
    "project": "https://github.com/Ltamann/ComfyUI-TBG-ETUR",
    "description": "TGB ETUR Enhanced Tiled Upscaler and Refiner PRO",
}

from .TBG.SERVERS.WORKER_server import TBG_Controller

try:
    TBG_Controller.start_worker_on_init()
    print("TBG Worker started during initialization", file=sys.stderr)
except Exception as e:
    print(f"Failed to start TBG worker: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)


def cleanup_tbg_worker():
    if TBG_Controller._worker_process is not None:
        print("Terminating TBG worker process", file=sys.stderr)
        TBG_Controller._worker_process.terminate()
        TBG_Controller._worker_process.wait(timeout=5)


atexit.register(cleanup_tbg_worker)

