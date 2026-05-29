import os
import sys
import platform
import importlib.util


_here = os.path.dirname(__file__)
_source_name = "TBG_APP.py"
_source_path = os.path.join(_here, _source_name)


def _load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[TBG_APP] Failed to load spec for {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reexport_public_symbols(mod):
    for name in dir(mod):
        if not name.startswith("_"):
            globals()[name] = getattr(mod, name)


if os.path.exists(_source_path):
    mod = _load_module(f"{__name__}._source", _source_path)
    _reexport_public_symbols(mod)
    __all__ = [k for k in globals().keys() if not k.startswith("_")]
    print(
        f"[TBG_APP] Loaded {_source_name} "
        f"(Python {sys.version_info.major}.{sys.version_info.minor}, {platform.system()})",
        file=sys.stderr,
    )
else:
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
            f"[TBG_APP] No compatible source or compiled module found\n"
            f"Platform : {plat_name}\n"
            f"Python   : {py_ver}\n\n"
            f"Available compiled files:\n{available}"
        )

    candidates.sort()
    chosen = candidates[0][1]
    _impl_path = os.path.join(_here, chosen)

    mod = _load_module(__name__, _impl_path)
    _reexport_public_symbols(mod)
    __all__ = [k for k in globals().keys() if not k.startswith("_")]
    print(
        f"[TBG_APP] Loaded {chosen} "
        f"(Python {py_ver}, {platform.system()})",
        file=sys.stderr,
    )
