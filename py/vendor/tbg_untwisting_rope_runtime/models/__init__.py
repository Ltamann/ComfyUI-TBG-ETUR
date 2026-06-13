"""Auto-discovered model architecture adapters for UntwistingRoPE.

The top-level node code imports this package only. Every model-specific name,
module lookup, and optional preprocessing routine should live in an adapter module in this folder.

Architecture recognition is intentionally metadata-only: adapters are selected
from ComfyUI's loaded MODEL metadata (``model.model_config`` and
``model.model_config.unet_config``), never from diffusion-module shape probes.

Drop-in adapter rule:
    Create ``models/new_model.py`` and expose at least:
        ARCHITECTURE = "new_model"
        DISPLAY_NAME = "New Model"
        def matches_model(model_info: dict) -> bool: ...
        def find_diffusion_model(model_patcher): ...

Optional hooks are discovered automatically when present:
        default_runtime_cfg(dm=None) -> dict
        prepare_reference_conditioning(...)
        patch_attention_modules(dm, stats, helpers=None)
        is_joint_attention(module) -> bool
        is_attention_name(name, min_layer=0, max_layer=999) -> bool
        uses_reference_branch_kv() -> bool

No edit to this registry is needed for a new adapter module. If more than one
adapter matches the same ComfyUI model metadata, that is a plugin bug and the
registry raises an explicit error instead of guessing an order.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

CONFIG_KEY = "untwisting_rope"

_IMPORT_ERRORS: dict[str, str] = {}


_MODEL_ROOT_PATHS = (
    "",
    "model",
    "model.model",
    "inner_model",
    "model.inner_model",
    "wrapped",
    "model.wrapped",
)


def _plain_value(value: Any) -> Any:
    """Return a debug-safe scalar while preserving metadata adapters match on."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items()}
    try:
        return str(value)
    except Exception:
        return repr(type(value))


def _plain_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): _plain_value(v) for k, v in value.items()}


def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def _iter_model_roots(model_patcher: Any) -> tuple[tuple[str, Any], ...]:
    """Yield common ComfyUI wrapper layers until a BaseModel metadata object is found."""
    roots: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path in _MODEL_ROOT_PATHS:
        if path:
            obj, ok = _get_attr_path(model_patcher, path)
            if not ok:
                continue
        else:
            obj = model_patcher
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        roots.append((path or "self", obj))
    return tuple(roots)


def _put_type(info: dict[str, Any], prefix: str, obj: Any) -> None:
    if obj is None:
        return
    cls = type(obj)
    info.setdefault(f"{prefix}_class", cls.__name__)
    info.setdefault(f"{prefix}_module", getattr(cls, "__module__", ""))


def _merge_unet_config(info: dict[str, Any], unet_config: Any) -> None:
    cfg = _plain_dict(unet_config)
    if not cfg:
        return
    info.setdefault("unet_config", cfg)
    for key, value in cfg.items():
        info.setdefault(key, value)
        info.setdefault(f"unet_config.{key}", value)


def build_model_info(model_patcher: Any, base_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build an identity dictionary from ComfyUI's loaded MODEL metadata only.

    ComfyUI ``BaseModel`` stores the selected ``supported_models`` instance on
    ``model.model_config`` and its official config on
    ``model.model_config.unet_config``. Adapter selection uses only that
    metadata; there is no architecture inference from diffusion-module shape.
    """
    destination = base_info if isinstance(base_info, dict) else None
    info: dict[str, Any] = dict(destination or {})

    for path, root in _iter_model_roots(model_patcher):
        model_config = getattr(root, "model_config", None)
        if model_config is None:
            continue

        info["has_comfy_model_config"] = True
        info.setdefault("model_config_path", path)
        _put_type(info, "model_config", model_config)
        _merge_unet_config(info, getattr(model_config, "unet_config", None))

        latent_format = getattr(model_config, "latent_format", None)
        if latent_format is not None:
            _put_type(info, "latent_format", latent_format)

        for attr in ("memory_usage_factor", "manual_cast_dtype"):
            if hasattr(model_config, attr):
                try:
                    info.setdefault(attr, _plain_value(getattr(model_config, attr)))
                except Exception:
                    pass
        break

    if destination is not None:
        destination.clear()
        destination.update(info)
        return destination
    return info


def _is_adapter_module(module: ModuleType) -> bool:
    """Return True when a discovered module exposes the metadata adapter API."""
    return (
        callable(getattr(module, "matches_model", None))
        and callable(getattr(module, "find_diffusion_model", None))
    )


def _load_adapters() -> tuple[ModuleType, ...]:
    """Import every non-private Python module in this package as an adapter."""
    adapters: list[ModuleType] = []
    _IMPORT_ERRORS.clear()

    package_path = __path__  # type: ignore[name-defined]
    package_name = __name__

    for module_info in pkgutil.iter_modules(package_path):
        name = module_info.name
        if name.startswith("_"):
            continue
        if module_info.ispkg:
            continue

        qualified_name = f"{package_name}.{name}"
        try:
            module = importlib.import_module(qualified_name)
        except Exception as exc:
            _IMPORT_ERRORS[name] = repr(exc)
            continue

        if _is_adapter_module(module):
            adapters.append(module)

    adapters.sort(key=lambda module: str(getattr(module, "ARCHITECTURE", module.__name__)))
    return tuple(adapters)


# Loaded once at package import. A freshly added adapter file is picked up on the
# next Python/ComfyUI reload, which is the normal plugin-development workflow.
REGISTERED_ADAPTERS = _load_adapters()


def refresh() -> tuple[ModuleType, ...]:
    """Reload the adapter list after files are added during a live session."""
    global REGISTERED_ADAPTERS
    REGISTERED_ADAPTERS = _load_adapters()
    return REGISTERED_ADAPTERS


def identify(model_patcher: Any, model_info: dict[str, Any] | None = None) -> ModuleType:
    """Return the unique adapter selected by ComfyUI MODEL metadata."""
    model_info = build_model_info(model_patcher, model_info)

    matched_adapters: list[ModuleType] = []
    match_errors: dict[str, str] = {}

    for adapter in REGISTERED_ADAPTERS:
        matches = getattr(adapter, "matches_model", None)
        if not callable(matches):
            continue
        try:
            if matches(model_info):
                matched_adapters.append(adapter)
        except Exception as exc:
            match_errors[adapter_key(adapter)] = repr(exc)

    if len(matched_adapters) == 1:
        return matched_adapters[0]

    summary = ", ".join(
        f"{key}={model_info.get(key)!r}"
        for key in ("model_config_class", "model_config_module", "image_model", "dim")
        if key in model_info
    ) or "no ComfyUI model_config metadata found"

    details = ""
    if _IMPORT_ERRORS:
        details += " Adapter import errors: " + "; ".join(
            f"{name}: {error}" for name, error in sorted(_IMPORT_ERRORS.items())
        )
    if match_errors:
        details += " Adapter match errors: " + "; ".join(
            f"{name}: {error}" for name, error in sorted(match_errors.items())
        )

    if matched_adapters:
        matched = ", ".join(adapter_key(adapter) for adapter in matched_adapters)
        raise RuntimeError(
            "Multiple adapters matched the same ComfyUI MODEL metadata "
            f"({summary}): {matched}. Adapter metadata matches must be unique."
            f"{details}"
        )

    raise RuntimeError(
        "Could not resolve a supported diffusion architecture from ComfyUI MODEL metadata "
        f"({summary}).{details}"
    )


def adapter_label(adapter: Any) -> str:
    return str(getattr(adapter, "DISPLAY_NAME", getattr(adapter, "ARCHITECTURE", type(adapter).__name__)))


def adapter_key(adapter: Any) -> str:
    return str(getattr(adapter, "ARCHITECTURE", type(adapter).__name__))


__all__ = [
    "CONFIG_KEY",
    "REGISTERED_ADAPTERS",
    "refresh",
    "build_model_info",
    "identify",
    "adapter_key",
    "adapter_label",
]
