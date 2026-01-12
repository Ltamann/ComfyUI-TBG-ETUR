import os
import sys
import importlib.util
import platform

# --------------------------------------------------
# Environment detection
# --------------------------------------------------

_here = os.path.dirname(__file__)

py_major = sys.version_info.major
py_minor = sys.version_info.minor
py_ver = f"{py_major}.{py_minor}"

plat = sys.platform

if plat.startswith("win"):
    plat_name = "windows"
    ext = ".pyd"
elif plat.startswith("linux"):
    plat_name = "linux"
    ext = ".so"
elif plat == "darwin":
    plat_name = "macos"
    ext = ".so"
else:
    raise ImportError(f"[TBG_APP] Unsupported platform: {plat}")

# --------------------------------------------------
# Candidate selection
# Priority:
#   0 → exact   (macos-3.13)
#   1 → latest  (macos-latest3.13)
# --------------------------------------------------

exact_tag = f"{plat_name}-{py_ver}"
latest_tag = f"{plat_name}-latest{py_ver}"

candidates = []

for fname in os.listdir(_here):
    if not fname.endswith(ext):
        continue

    if exact_tag in fname:
        candidates.append((0, fname))
    elif latest_tag in fname:
        candidates.append((1, fname))

if not candidates:
    available = "\n".join(
        f"  - {f}" for f in os.listdir(_here)
        if f.endswith((".pyd", ".so"))
    )
    raise ImportError(
        f"[TBG_APP] No compatible compiled module found\n"
        f"Platform : {plat_name}\n"
        f"Python   : {py_ver}\n\n"
        f"Available compiled files:\n{available}"
    )

# Sort by priority, then filename for determinism
candidates.sort()
chosen = candidates[0][1]
_impl_path = os.path.join(_here, chosen)

# --------------------------------------------------
# Load compiled module
# --------------------------------------------------

spec = importlib.util.spec_from_file_location(__name__, _impl_path)
if spec is None or spec.loader is None:
    raise ImportError(f"[TBG_APP] Failed to load spec for {_impl_path}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# --------------------------------------------------
# Re-export public symbols
# --------------------------------------------------

for name in dir(mod):
    if not name.startswith("_"):
        globals()[name] = getattr(mod, name)

__all__ = [k for k in globals().keys() if not k.startswith("_")]

print(
    f"[TBG_APP] Loaded {chosen} "
    f"(Python {py_ver}, {platform.system()})",
    file=sys.stderr
)


"""
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
"""