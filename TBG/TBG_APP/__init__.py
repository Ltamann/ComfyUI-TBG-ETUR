import os, sys
import importlib.util


# 2. THEN: load the compiled .pyd
_here = os.path.dirname(__file__)
_impl_path = os.path.join(_here, "TBG_APP-windows-3.12.pyd")

if not os.path.exists(_impl_path):
    raise ImportError(f"Compiled module not found: {_impl_path}")

# Load the .pyd directly as the main module content
spec = importlib.util.spec_from_file_location(__name__, _impl_path)
if spec and spec.loader:
    mod = importlib.util.module_from_spec(spec)


    spec.loader.exec_module(mod)

    # Copy all public attributes from .pyd to this __init__ namespace
    for name in dir(mod):
        if not name.startswith("_"):
            globals()[name] = getattr(mod, name)

    print(f"[TBG_APP] Loaded compiled module from {_impl_path}", file=sys.stderr)
else:
    raise ImportError(f"Failed to load {_impl_path}")

__all__ = [k for k in globals().keys() if not k.startswith("_")]
