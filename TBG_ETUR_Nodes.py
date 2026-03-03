"""Deprecated legacy node-mapping shim.

ComfyUI v3 registration is handled by `comfy_entrypoint()` in `py/v3/extension.py`.
This module remains only to avoid hard import failures in older internal references.
"""

import warnings

warnings.warn(
    "TBG_ETUR_Nodes.py is deprecated. Use v3 comfy_entrypoint registration.",
    DeprecationWarning,
    stacklevel=2,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web/assets/js"
