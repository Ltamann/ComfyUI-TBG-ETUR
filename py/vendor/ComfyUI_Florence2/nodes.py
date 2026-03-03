from collections.abc import Callable
import torch
import torchvision.transforms.functional as F
import io
import os
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
import importlib.util
import json
from pathlib import Path
from packaging import version as pkg_version
import glob
import re
import gc
import time

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

import transformers

from safetensors.torch import load_file, save_file

FLORENCE_LOADER_REV = "2026-03-03-ms-seq2seq-fallback-v2"
_FLORENCE_TIE_PATCH_ATTEMPTED = set()
_FLORENCE_FORCE_BIN_CACHE = {}
_FLORENCE_MODEL_CACHE = {}
_FLORENCE_UNLOAD_PATCHED = False
_FLORENCE_LOADER_LAST_MODEL = {}
_FLORENCE_VERBOSE = os.environ.get("TBG_FLORENCE_VERBOSE", "").strip() == "1"
_FLORENCE_LOG_ONCE = set()


def _flog(msg, always=False):
    if always or _FLORENCE_VERBOSE:
        print(msg)


def _flog_once(key, msg, always=False):
    if key in _FLORENCE_LOG_ONCE:
        return
    _FLORENCE_LOG_ONCE.add(key)
    _flog(msg, always=always)


def _resolve_florence_repo_id(repo_id):
    """
    Keep user-selected repo by default.
    Optional opt-in remap can be enabled with TBG_FLORENCE_REMAP_TO_COMMUNITY=1.
    """
    if os.environ.get("TBG_FLORENCE_REMAP_TO_COMMUNITY", "").strip() != "1":
        return repo_id
    mapping = {
        "microsoft/Florence-2-base": "florence-community/Florence-2-base",
        "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
        "microsoft/Florence-2-large": "florence-community/Florence-2-large",
        "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
    }
    return mapping.get(repo_id, repo_id)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        _flog_once("no_flash_attn", "No flash_attn import to remove")
        pass
    return imports


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]): 
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths if Path(path).exists() for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(model_directory, exist_ok=True)

# Ensure ComfyUI knows about the LLM model path
folder_paths.add_model_folder_path("LLM", model_directory)

from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, set_seed
try:
    from transformers import Florence2ForConditionalGeneration as HFFlorence2ForConditionalGeneration
except Exception:
    HFFlorence2ForConditionalGeneration = None


def _load_florence_processor(model_path, source_id=None, use_fast=None):
    """
    Load Florence processor with a compatibility fallback for newer transformers
    where tokenizer objects may miss attributes expected by processing_florence2.py.
    """
    processor_source = source_id or model_path
    processor_kwargs = {"trust_remote_code": True}
    if use_fast is not None:
        processor_kwargs["use_fast"] = bool(use_fast)
    try:
        return AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    except AttributeError as exc:
        msg = str(exc)
        known_tokenizer_mismatch = (
            "additional_special_tokens" in msg or "image_token" in msg
        )
        if not known_tokenizer_mismatch:
            raise

        image_kwargs = {"trust_remote_code": True}
        if use_fast is not None:
            image_kwargs["use_fast"] = bool(use_fast)
        image_processor = AutoImageProcessor.from_pretrained(model_path, **image_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

        if not hasattr(tokenizer, "additional_special_tokens"):
            tokenizer.additional_special_tokens = []
        if not hasattr(tokenizer, "image_token"):
            tokenizer.image_token = "<image>"

        processor_py = Path(model_path) / "processing_florence2.py"
        if not processor_py.exists():
            raise

        spec = importlib.util.spec_from_file_location("local_processing_florence2", str(processor_py))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)


def _is_transformers_lt_451():
    """
    Robust semver check that works for transformers 4.x/5.x and local build suffixes.
    """
    try:
        current = pkg_version.parse(str(transformers.__version__).split("+", 1)[0])
        return current < pkg_version.parse("4.51.0")
    except Exception:
        # Conservative fallback if version parsing fails.
        return str(transformers.__version__) < "4.51.0"


def _is_transformers_lt_500():
    """
    Gate new Florence compatibility layer to transformers >= 5 only.
    transformers 4.x should keep the legacy optimized loading path.
    """
    try:
        current = pkg_version.parse(str(transformers.__version__).split("+", 1)[0])
        return current < pkg_version.parse("5.0.0")
    except Exception:
        return str(transformers.__version__) < "5.0.0"


def _get_model_dtype_kwarg(dtype):
    """
    transformers 5.x prefers `dtype`; 4.x uses `torch_dtype`.
    """
    try:
        current = pkg_version.parse(str(transformers.__version__).split("+", 1)[0])
        if current >= pkg_version.parse("5.0.0"):
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def _normalize_hf_attention(attention):
    """
    Normalize external attention mode names to HF-supported values.
    """
    if attention in {"flash_attention_2", "sdpa", "eager"}:
        return attention
    attn = str(attention).strip().lower() if attention is not None else ""
    if attn in {"sage", "sageattn", "sage_attention", "sage-attention"}:
        _flog_once("attn_sage_map", "[Florence2] attention='sage' is not a HF attn_implementation; using 'sdpa'.")
        return "sdpa"
    if attn:
        _flog_once(f"attn_bad_{attn}", f"[Florence2] unsupported attention '{attention}', falling back to 'eager'.")
    return "eager"


def _load_florence_legacy_model(model_path, attention, dtype, offload_device, repo_id=None):
    """
    Legacy auto-loading path for older transformers.
    Prefer Seq2Seq (correct Florence architecture) and fallback to CausalLM for
    older model configs that still map only AutoModelForCausalLM.
    """
    attention = _normalize_hf_attention(attention)
    is_microsoft = _is_microsoft_florence_repo(repo_id)
    prefer_causal = is_microsoft or _path_prefers_causallm(model_path)
    force_bin = _should_force_bin_checkpoint(model_path)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        # For Microsoft Florence on transformers 4.x, prefer vendored model first so
        # optimized attention paths (sdpa/flash_attention_2) can still be used.
        if is_microsoft:
            try:
                _flog("[Florence2][legacy] trying vendored Florence2ForConditionalGeneration first.")
                from .modeling_florence2 import Florence2ForConditionalGeneration
                loaded = Florence2ForConditionalGeneration.from_pretrained(
                    model_path,
                    attn_implementation=attention,
                    **_get_model_dtype_kwarg(dtype),
                    use_safetensors=(not force_bin),
                    output_loading_info=True,
                )
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    model, loading_info = loaded
                else:
                    model, loading_info = loaded, {}
                _retie_florence_language_weights(model)
                if _has_critical_lm_missing_keys(loading_info):
                    _flog(
                        "[Florence2][legacy] vendored load missing critical LM keys; "
                        "falling back to CausalLM remote code."
                    )
                else:
                    return model.to(offload_device)
            except Exception as vendored_err:
                _flog(f"[Florence2][legacy] vendored first-load failed: {vendored_err}")

        if not prefer_causal:
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    attn_implementation=attention,
                    **_get_model_dtype_kwarg(dtype),
                    trust_remote_code=True,
                ).to(offload_device)
            except AttributeError as seq_attr_err:
                if "_supports_sdpa" in str(seq_attr_err) and attention != "eager":
                    _flog("Florence legacy Seq2Seq missing _supports_sdpa; retrying with attn_implementation='eager'.")
                    return AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        attn_implementation="eager",
                        **_get_model_dtype_kwarg(dtype),
                        trust_remote_code=True,
                    ).to(offload_device)
                _flog(f"Florence legacy Seq2Seq autoload failed, falling back to CausalLM: {seq_attr_err}")
            except Exception as seq_err:
                _flog(f"Florence legacy Seq2Seq autoload failed, falling back to CausalLM: {seq_err}")
        else:
            _flog("[Florence2][legacy] skipping Seq2Seq autoload (config/repo prefers CausalLM).")

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attention,
                **_get_model_dtype_kwarg(dtype),
                use_safetensors=(not force_bin),
                trust_remote_code=True,
            ).to(offload_device)
        except AttributeError as causal_attr_err:
            if "_supports_sdpa" in str(causal_attr_err) and attention != "eager":
                _flog("Florence legacy CausalLM missing _supports_sdpa; retrying with attn_implementation='eager'.")
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="eager",
                    **_get_model_dtype_kwarg(dtype),
                    trust_remote_code=True,
                ).to(offload_device)
            raise


def _read_config_json(model_path):
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_microsoft_florence_repo(repo_id):
    return isinstance(repo_id, str) and repo_id.lower().startswith("microsoft/florence-2")


def _path_prefers_causallm(model_path):
    cfg = _read_config_json(model_path)
    auto_map = cfg.get("auto_map", {})
    return "AutoModelForCausalLM" in auto_map and "AutoModelForSeq2SeqLM" not in auto_map


def _has_critical_lm_missing_keys(loading_info):
    if not isinstance(loading_info, dict):
        return False
    missing = set(loading_info.get("missing_keys", []))
    critical = {
        "language_model.model.encoder.embed_tokens.weight",
        "language_model.model.decoder.embed_tokens.weight",
        "language_model.lm_head.weight",
    }
    return len(missing.intersection(critical)) > 0


def _retie_florence_language_weights(model):
    """
    Ensure encoder/decoder/lm_head share the main token embedding weights.
    Some Florence checkpoints ship shared embeddings only; without tying, missing
    keys can leave randomly initialized heads and produce garbage/OOB generation.
    """
    try:
        lm = getattr(model, "language_model", None)
        if lm is None:
            return False
        core = getattr(lm, "model", None)
        if core is None or not hasattr(core, "shared"):
            return False
        shared = core.shared
        if hasattr(core, "encoder") and hasattr(core.encoder, "embed_tokens"):
            core.encoder.embed_tokens.weight = shared.weight
        if hasattr(core, "decoder") and hasattr(core.decoder, "embed_tokens"):
            core.decoder.embed_tokens.weight = shared.weight
        if hasattr(lm, "lm_head"):
            lm.lm_head.weight = shared.weight
        # Run model-level tie as well where available.
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        return True
    except Exception as e:
        _flog(f"Florence re-tie skipped ({e})")
        return False


def _ensure_florence_checkpoint_tied_weights(model_path):
    """
    Patch local Florence checkpoint files so tied language weights physically exist.
    This avoids repeated `MISSING` init for encoder/decoder embeds and lm_head on
    transformer stacks that do not auto-materialize tied tensors from shared weights.
    """
    # Run once per model path per process to avoid repeated file-write attempts while
    # weights are memory-mapped/locked during active runs.
    if model_path in _FLORENCE_TIE_PATCH_ATTEMPTED:
        return
    _FLORENCE_TIE_PATCH_ATTEMPTED.add(model_path)

    required = [
        "language_model.model.encoder.embed_tokens.weight",
        "language_model.model.decoder.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]
    shared_key = "language_model.model.shared.weight"

    safepath = Path(model_path) / "model.safetensors"
    if safepath.exists():
        try:
            sd = load_file(str(safepath), device="cpu")
            if shared_key in sd:
                missing = [k for k in required if k not in sd]
                if missing:
                    shared = sd[shared_key]
                    for k in missing:
                        sd[k] = shared.clone()
                    save_file(sd, str(safepath))
                    _flog(f"[Florence2] patched {safepath.name}: added tied keys {missing}")
                    return
        except Exception as e:
            _flog(f"[Florence2] safetensors tie patch skipped ({e})")

    binpath = Path(model_path) / "pytorch_model.bin"
    if binpath.exists():
        try:
            sd = torch.load(str(binpath), map_location="cpu")
            if isinstance(sd, dict) and shared_key in sd:
                missing = [k for k in required if k not in sd]
                if missing:
                    shared = sd[shared_key]
                    for k in missing:
                        sd[k] = shared.clone()
                    torch.save(sd, str(binpath))
                    _flog(f"[Florence2] patched {binpath.name}: added tied keys {missing}")
        except Exception as e:
            _flog(f"[Florence2] bin tie patch skipped ({e})")


def _should_force_bin_checkpoint(model_path):
    """
    If safetensors exists but lacks critical tied language weights, prefer loading
    from pytorch_model.bin (which may contain full tensors).
    """
    cached = _FLORENCE_FORCE_BIN_CACHE.get(model_path)
    if cached is not None:
        return cached

    required = {
        "language_model.model.encoder.embed_tokens.weight",
        "language_model.model.decoder.embed_tokens.weight",
        "language_model.lm_head.weight",
    }
    safepath = Path(model_path) / "model.safetensors"
    binpath = Path(model_path) / "pytorch_model.bin"
    if not safepath.exists() or not binpath.exists():
        _FLORENCE_FORCE_BIN_CACHE[model_path] = False
        return False
    try:
        sd = load_file(str(safepath), device="cpu")
        missing = [k for k in required if k not in sd]
        if missing:
            _flog(f"[Florence2] safetensors missing tied keys {missing}; forcing .bin checkpoint load.")
            _FLORENCE_FORCE_BIN_CACHE[model_path] = True
            return True
    except Exception as e:
        _flog(f"[Florence2] safetensors inspection failed ({e}); forcing .bin checkpoint load.")
        _FLORENCE_FORCE_BIN_CACHE[model_path] = True
        return True
    _FLORENCE_FORCE_BIN_CACHE[model_path] = False
    return False


def _local_florence_config_from_json(model_path):
    """
    Build Florence2Config from local config.json but drop auto_map so transformers
    won't require trust_remote_code for model loading.
    """
    from .configuration_florence2 import Florence2Config

    cfg_dict = _read_config_json(model_path)
    cfg_dict.pop("auto_map", None)
    return Florence2Config(**cfg_dict)


def _patch_cached_hf_florence_configs():
    """
    Patch cached HF microsoft Florence config files to avoid transformers 5.x
    forced_bos_token_id AttributeError in remote code path.
    """
    patched = 0
    home = Path.home()
    cache_root = home / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not cache_root.exists():
        return 0

    pattern = str(cache_root / "**" / "configuration_florence2.py")
    for cfg_file in glob.glob(pattern, recursive=True):
        p = Path(cfg_file)
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        # Match both quote styles and minor whitespace variations.
        rx = r"if\s+self\.forced_bos_token_id\s+is\s+None\s+and\s+kwargs\.get\((['\"])force_bos_token_to_be_generated\1,\s*False\):"
        new = "if getattr(self, \"forced_bos_token_id\", None) is None and kwargs.get(\"force_bos_token_to_be_generated\", False):"
        patched_text, replacements = re.subn(rx, new, text)
        if replacements > 0:
            try:
                p.write_text(patched_text, encoding="utf-8")
                patched += 1
            except Exception:
                pass
    return patched


def _patch_local_model_florence_config(model_path):
    """
    Patch configuration_florence2.py in the local model directory so regenerated
    dynamic-module cache files also carry the fix.
    """
    p = Path(model_path) / "configuration_florence2.py"
    if not p.exists():
        return 0
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return 0
    rx = r"if\s+self\.forced_bos_token_id\s+is\s+None\s+and\s+kwargs\.get\((['\"])force_bos_token_to_be_generated\1,\s*False\):"
    new = "if getattr(self, \"forced_bos_token_id\", None) is None and kwargs.get(\"force_bos_token_to_be_generated\", False):"
    patched_text, replacements = re.subn(rx, new, text)
    if replacements > 0:
        try:
            p.write_text(patched_text, encoding="utf-8")
            return 1
        except Exception:
            return 0
    return 0


def _patch_florence_modeling_meta_issue(modeling_file):
    """
    Patch known meta-tensor-unsafe drop-path list build in Florence modeling files.
    """
    p = Path(modeling_file)
    if not p.exists():
        return 0
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return 0

    old = "dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]"
    new = "dpr = torch.linspace(0, drop_path_rate, sum(depths)*2, device=\"cpu\").tolist()"
    if old in text:
        try:
            p.write_text(text.replace(old, new), encoding="utf-8")
            return 1
        except Exception:
            return 0
    return 0


def _patch_local_model_florence_modeling(model_path):
    return _patch_florence_modeling_meta_issue(Path(model_path) / "modeling_florence2.py")


def _patch_cached_hf_florence_modeling():
    """
    Patch cached HF Florence modeling files for meta-tensor safety on newer torch/transformers.
    """
    patched = 0
    home = Path.home()
    cache_root = home / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not cache_root.exists():
        return 0
    pattern = str(cache_root / "**" / "modeling_florence2.py")
    for mf in glob.glob(pattern, recursive=True):
        patched += _patch_florence_modeling_meta_issue(mf)
    return patched


def _purge_florence_cache_for_model(target_model):
    """
    Remove only cache entries that reference the specific model object.
    """
    if target_model is None:
        return 0
    removed = 0
    for cache in (_FLORENCE_MODEL_CACHE, DownloadAndLoadFlorence2Model._CACHE, Florence2ModelLoader._CACHE):
        to_delete = []
        for key, value in cache.items():
            if isinstance(value, dict) and value.get("model") is target_model:
                to_delete.append(key)
        for key in to_delete:
            del cache[key]
            removed += 1
    for loader_id, mdl in list(_FLORENCE_LOADER_LAST_MODEL.items()):
        if mdl is target_model:
            del _FLORENCE_LOADER_LAST_MODEL[loader_id]
    return removed


def _unload_specific_model(target_model):
    """
    Unload a concrete model object via comfy model-management APIs.
    """
    if target_model is None:
        return False
    loaded_models = mm.loaded_models()
    removed = False
    if target_model in loaded_models:
        loaded_models.remove(target_model)
        removed = True
    mm.free_memory(1e30, mm.get_torch_device(), loaded_models)
    mm.soft_empty_cache(True)
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(0.1)
    _purge_florence_cache_for_model(target_model)
    return removed


def _find_loader_model_instance(loader_obj):
    """
    Resolve currently loaded model instance for a Florence loader object.
    """
    model = _FLORENCE_LOADER_LAST_MODEL.get(id(loader_obj))
    if model is not None:
        return model
    for cache in (DownloadAndLoadFlorence2Model._CACHE, Florence2ModelLoader._CACHE, _FLORENCE_MODEL_CACHE):
        for value in cache.values():
            if isinstance(value, dict) and "model" in value:
                return value["model"]
    return None


def _install_florence_unload_route_patch():
    """
    Patch UnloadOneModelNode.route to handle the existing Tiler class-style call
    UnloadOneModelNode.route(florence_loader) without editing Tiler.
    """
    global _FLORENCE_UNLOAD_PATCHED
    if _FLORENCE_UNLOAD_PATCHED:
        return
    try:
        from ...vendor.ComfyUI_Unload_Models_main.py.unload_one_model import UnloadOneModelNode
    except Exception:
        return

    original_route = UnloadOneModelNode.route

    def patched_route(self, **kwargs):
        # Compatibility path: route invoked as UnloadOneModelNode.route(florence_loader)
        if not kwargs and isinstance(self, (DownloadAndLoadFlorence2Model, Florence2ModelLoader)):
            target_model = _find_loader_model_instance(self)
            if target_model is not None:
                _flog("Unload Model:", always=True)
                _flog(" - Florence compatibility route: unloading loader-resolved model...", always=True)
                _unload_specific_model(target_model)
                return ([self],)
        return original_route(self, **kwargs)

    UnloadOneModelNode.route = patched_route
    _FLORENCE_UNLOAD_PATCHED = True


def _load_microsoft_remote_fallback(model_path, attention, dtype, offload_device, force_bin):
    attention = _normalize_hf_attention(attention)
    _flog("Trying microsoft remote-code fallback from local model path.")
    patched_local = _patch_local_model_florence_config(model_path)
    if patched_local > 0:
        _flog("Patched local model configuration_florence2.py for forced_bos compatibility.")
    patched_local_modeling = _patch_local_model_florence_modeling(model_path)
    if patched_local_modeling > 0:
        _flog("Patched local model modeling_florence2.py for meta-tensor compatibility.")
    patched = _patch_cached_hf_florence_configs()
    if patched > 0:
        _flog(f"Patched {patched} cached microsoft Florence config file(s) for forced_bos compatibility.")
    patched_modeling = _patch_cached_hf_florence_modeling()
    if patched_modeling > 0:
        _flog(f"Patched {patched_modeling} cached microsoft Florence modeling file(s) for meta-tensor compatibility.")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                attn_implementation=attention,
                **_get_model_dtype_kwarg(dtype),
                use_safetensors=(not force_bin),
                trust_remote_code=True,
                local_files_only=True,
            )
        except AttributeError as e:
            if "_supports_sdpa" not in str(e):
                raise
            _flog("Microsoft remote Florence model lacks _supports_sdpa; retrying with attn_implementation='eager'.")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                **_get_model_dtype_kwarg(dtype),
                use_safetensors=(not force_bin),
                trust_remote_code=True,
                local_files_only=True,
            )
    _retie_florence_language_weights(model)
    return model.to(offload_device), model_path


def _load_florence_modern_model(model_path, attention, dtype, offload_device, repo_id=None):
    """
    Modern path (transformers >= 4.51):
    - Microsoft Florence repos: prefer transformers built-in Florence2 class, no remote code.
    - Others: use vendored Florence2 class.
    - Safety fallback: if vendored load reports critical LM weights missing, retry with CausalLM (non-microsoft only).
    """
    if _is_transformers_lt_500():
        _flog(
            f"[Florence2][modern] bypassed for transformers={transformers.__version__}; "
            "using legacy loader."
        )
        return _load_florence_legacy_model(model_path, attention, dtype, offload_device), model_path

    attention = _normalize_hf_attention(attention)
    is_microsoft = _is_microsoft_florence_repo(repo_id)
    force_bin = _should_force_bin_checkpoint(model_path)
    use_causal = is_microsoft or _path_prefers_causallm(model_path)
    causal_source = repo_id if is_microsoft else model_path
    _flog(
        f"[Florence2][modern] rev={FLORENCE_LOADER_REV} transformers={transformers.__version__} "
        f"module={__file__} is_microsoft={is_microsoft} use_causal={use_causal} source={causal_source}"
    )
    if use_causal and is_microsoft:
        _flog("[Florence2][modern] loader=Microsoft remote-code stable path")
        return _load_microsoft_remote_fallback(model_path, attention, dtype, offload_device, force_bin)

    if use_causal and (not is_microsoft):
        try:
            _flog("[Florence2][modern] loader=AutoModelForSeq2SeqLM (non-microsoft, no remote code)")
            cfg = AutoConfig.from_pretrained(causal_source, trust_remote_code=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                causal_source,
                config=cfg,
                attn_implementation=attention,
                **_get_model_dtype_kwarg(dtype),
                trust_remote_code=False,
            )
            _retie_florence_language_weights(model)
            return model.to(offload_device), causal_source
        except Exception as causal_err:
            _flog(
                f"Florence non-microsoft seq2seq loader failed ({causal_err}); "
                "falling back to vendored Florence2ForConditionalGeneration."
            )

    _flog("[Florence2][modern] loader=Vendored Florence2ForConditionalGeneration")
    from .modeling_florence2 import Florence2ForConditionalGeneration
    loaded = Florence2ForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=attention,
        **_get_model_dtype_kwarg(dtype),
        use_safetensors=(not force_bin),
        output_loading_info=True,
    )
    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, loading_info = loaded
    else:
        model, loading_info = loaded, {}
    _retie_florence_language_weights(model)

    if _has_critical_lm_missing_keys(loading_info):
        _flog("Florence vendored load reported missing LM embed/lm_head weights.")
        if is_microsoft:
            return _load_microsoft_remote_fallback(model_path, attention, dtype, offload_device, force_bin)
        _flog("Retrying with AutoModelForSeq2SeqLM (non-microsoft repo).")
        cfg = AutoConfig.from_pretrained(causal_source, trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            causal_source,
            config=cfg,
            attn_implementation=attention,
            **_get_model_dtype_kwarg(dtype),
            trust_remote_code=False,
        )
        _retie_florence_language_weights(model)
        return model.to(offload_device), causal_source

    return model.to(offload_device), model_path

class DownloadAndLoadFlorence2Model:
    _CACHE = {}

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA',
                    'thwri/CogFlorence-2.1-Large',
                    'thwri/CogFlorence-2.2-Large',
                    'gokaygokay/Florence-2-SD3-Captioner',
                    'gokaygokay/Florence-2-Flux-Large',
                    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
                    'PJMixers-Images/Florence-2-base-Castollux-v0.5'
                    ],
                    {
                    "default": 'microsoft/Florence-2-base'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    @classmethod
    def clear_cache(cls):
        cls._CACHE.clear()

    def loadmodel(self, model, precision, attention, lora=None, convert_to_safetensors=False):
        _install_florence_unload_route_patch()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_id = _resolve_florence_repo_id(model)
        if model_id != model:
            _flog(f"[Florence2] remap repo: {model} -> {model_id}")

        model_name = model_id.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            _flog(f"Downloading Florence2 model to: {model_path}", always=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        _flog(f"Florence2 using {attention} for attention", always=True)
        attn_key = _normalize_hf_attention(attention)
        model_cache_key = (
            "global",
            model_id,
            model_name,
            str(transformers.__version__),
            precision,
            str(attn_key),
            str(lora) if lora is not None else "",
        )
        global_cached = _FLORENCE_MODEL_CACHE.get(model_cache_key)
        if global_cached is not None:
            _FLORENCE_LOADER_LAST_MODEL[id(self)] = global_cached.get("model")
            return (global_cached,)
        
        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                _flog(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        _flog(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        _flog(f"Original {model_weight_path} file deleted.")
        _ensure_florence_checkpoint_tied_weights(model_path)
        
        cache_key = (
            "download_loader",
            model_id,
            model_path,
            str(transformers.__version__),
            precision,
            str(attn_key),
            str(lora) if lora is not None else "",
        )
        cached = self._CACHE.get(cache_key)
        if cached is not None:
            _FLORENCE_MODEL_CACHE[model_cache_key] = cached
            _FLORENCE_LOADER_LAST_MODEL[id(self)] = cached.get("model")
            return (cached,)

        if _is_transformers_lt_500():
            _flog(f"[Florence2] path=legacy transformers={transformers.__version__}")
            model = _load_florence_legacy_model(model_path, attn_key, dtype, offload_device, repo_id=model_id)
            processor_source = model_path
        else:
            _flog(f"[Florence2] path=modern transformers={transformers.__version__}")
            model, processor_source = _load_florence_modern_model(
                model_path, attn_key, dtype, offload_device, repo_id=model_id
            )
    
        processor_use_fast = None
        if (not _is_transformers_lt_500()) and _is_microsoft_florence_repo(model_id):
            processor_use_fast = False
        processor = _load_florence_processor(model_path, source_id=processor_source, use_fast=processor_use_fast)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
            }
        self._CACHE[cache_key] = florence2_model
        _FLORENCE_MODEL_CACHE[model_cache_key] = florence2_model
        _FLORENCE_LOADER_LAST_MODEL[id(self)] = florence2_model.get("model")

        return (florence2_model,)
    
class DownloadAndLoadFlorence2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'NikshepShetty/Florence-2-pixelprose',
                    ],
                  ),            
            },
          
        }

    RETURN_TYPES = ("PEFTLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model):
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            _flog(f"Downloading Florence2 lora model to: {model_path}", always=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        return (model_path,)
    
class Florence2ModelLoader:
    _CACHE = {}

    @classmethod
    def INPUT_TYPES(s):
        all_llm_paths = folder_paths.get_folder_paths("LLM")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())

        return {"required": {
            "model": ([*s.model_paths], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
            "precision": (['fp16','bf16','fp32'],),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    @classmethod
    def clear_cache(cls):
        cls._CACHE.clear()

    def loadmodel(self, model, precision, attention, lora=None, convert_to_safetensors=False):
        _install_florence_unload_route_patch()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = Florence2ModelLoader.model_paths.get(model)
        _flog(f"Loading model from {model_path}")
        _flog(f"Florence2 using {attention} for attention", always=True)
        attn_key = _normalize_hf_attention(attention)
        model_cache_key = (
            "global_path_loader",
            model,
            model_path,
            str(transformers.__version__),
            precision,
            str(attn_key),
            str(lora) if lora is not None else "",
        )
        global_cached = _FLORENCE_MODEL_CACHE.get(model_cache_key)
        if global_cached is not None:
            _FLORENCE_LOADER_LAST_MODEL[id(self)] = global_cached.get("model")
            return (global_cached,)
        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                _flog(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        _flog(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        _flog(f"Original {model_weight_path} file deleted.")
        _ensure_florence_checkpoint_tied_weights(model_path)

        cache_key = (
            "path_loader",
            model,
            model_path,
            str(transformers.__version__),
            precision,
            str(attn_key),
            str(lora) if lora is not None else "",
        )
        cached = self._CACHE.get(cache_key)
        if cached is not None:
            _FLORENCE_MODEL_CACHE[model_cache_key] = cached
            _FLORENCE_LOADER_LAST_MODEL[id(self)] = cached.get("model")
            return (cached,)

        if _is_transformers_lt_500():
            _flog(f"[Florence2] path=legacy transformers={transformers.__version__}")
            model = _load_florence_legacy_model(model_path, attn_key, dtype, offload_device, repo_id=None)
            processor_source = model_path
        else:
            _flog(f"[Florence2] path=modern transformers={transformers.__version__}")
            model, processor_source = _load_florence_modern_model(
                model_path, attn_key, dtype, offload_device, repo_id=None
            )
        processor_use_fast = None
        if (not _is_transformers_lt_500()) and _path_prefers_causallm(model_path):
            processor_use_fast = False
        processor = _load_florence_processor(model_path, source_id=processor_source, use_fast=processor_use_fast)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
            }
        self._CACHE[cache_key] = florence2_model
        _FLORENCE_MODEL_CACHE[model_cache_key] = florence2_model
        _FLORENCE_LOADER_LAST_MODEL[id(self)] = florence2_model.get("model")
   
        return (florence2_model,)
    
class Florence2Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "florence2_model": ("FL2MODEL", ),
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "task": (
                    [ 
                    'region_caption',
                    'dense_region_caption',
                    'region_proposal',
                    'caption',
                    'detailed_caption',
                    'more_detailed_caption',
                    'caption_to_phrase_grounding',
                    'referring_expression_segmentation',
                    'ocr',
                    'ocr_with_region',
                    'docvqa',
                    'prompt_gen_tags',
                    'prompt_gen_mixed_caption',
                    'prompt_gen_analyze',
                    'prompt_gen_mixed_caption_plus',
                    ],
                   ),
                "fill_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "output_mask_select": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "JSON")
    RETURN_NAMES =("image", "mask", "caption", "data") 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)

    def _prepare_inputs(self, processor, prompt, image_pil, device, dtype, model):
        """
        Keep token indices as int64 and cast only image tensors to model dtype.
        Casting input_ids to fp16 can corrupt ids and trigger embedding index OOB.
        """
        inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False)
        inputs["input_ids"] = inputs["input_ids"].to(device=device, dtype=torch.long)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(device=device, dtype=torch.long)
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)

        vocab = int(model.get_input_embeddings().weight.shape[0])
        input_min = int(inputs["input_ids"].min().item())
        input_max = int(inputs["input_ids"].max().item())
        if input_min < 0 or input_max >= vocab:
            raise ValueError(
                f"Florence input_ids out of range: min={input_min}, max={input_max}, vocab={vocab}."
            )
        return inputs

    def _sanitize_generation_ids(self, model):
        """
        Ensure generation-critical ids are in-range for current embedding vocab.
        Deterministic policy:
        - invalid forced_bos_token_id -> disable (None)
        - invalid bos/eos/decoder_start -> set to a safe id
        """
        vocab = int(model.get_input_embeddings().weight.shape[0])

        def in_range(v):
            return isinstance(v, int) and 0 <= v < vocab

        # Build safe fallback id from valid config ids, else 0.
        candidate_ids = []
        for obj in (getattr(model, "generation_config", None), getattr(model, "config", None)):
            if obj is None:
                continue
            for name in ("bos_token_id", "eos_token_id", "decoder_start_token_id"):
                v = getattr(obj, name, None)
                if in_range(v):
                    candidate_ids.append(v)
        safe_id = candidate_ids[0] if candidate_ids else 0

        for obj in (getattr(model, "generation_config", None), getattr(model, "config", None)):
            if obj is None:
                continue
            for name in ("bos_token_id", "eos_token_id", "decoder_start_token_id"):
                v = getattr(obj, name, None)
                if v is not None and not in_range(v):
                    setattr(obj, name, int(safe_id))
            forced_bos = getattr(obj, "forced_bos_token_id", None)
            if forced_bos is not None and not in_range(forced_bos):
                setattr(obj, "forced_bos_token_id", None)

    def encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, 
            num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="", seed=None):
        device = mm.get_torch_device()
        _, height, width, _ = image.shape
        offload_device = mm.unet_offload_device()
        annotated_image_tensor = None
        mask_tensor = None
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(device)
        
        if seed:
            set_seed(self.hash_seed(seed))

        colormap = ['blue','orange','green','purple','brown','pink','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','gold','tan','skyblue']

        prompts = {
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        task_prompt = prompts.get(task, '<OD>')

        if (task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']) and text_input:
            raise ValueError("Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'")

        if text_input != "":
            prompt = task_prompt + " " + text_input
        else:
            prompt = task_prompt

        image = image.permute(0, 3, 1, 2)
        
        out = []
        out_masks = []
        out_results = []
        out_data = []
        pbar = ProgressBar(len(image))
        for img in image:
            image_pil = F.to_pil_image(img)
            inputs = self._prepare_inputs(processor, prompt, image_pil, device, dtype, model)
            self._sanitize_generation_ids(model)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=False,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            _flog(results)
            # cleanup the special tokens from the final list
            if task == 'ocr_with_region':
                clean_results = str(results)       
                cleaned_string = re.sub(r'</?s>|<[^>]*>', '\n',  clean_results)
                clean_results = re.sub(r'\n+', '\n', cleaned_string)
            else:
                clean_results = str(results)       
                clean_results = clean_results.replace('</s>', '')
                clean_results = clean_results.replace('<s>', '')

             #return single string if only one image for compatibility with nodes that can't handle string lists
            if len(image) == 1:
                out_results = clean_results
            else:
                out_results.append(clean_results)

            W, H = image_pil.size
            
            parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))

            if task == 'region_caption' or task == 'dense_region_caption' or task == 'caption_to_phrase_grounding' or task == 'region_proposal':           
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.imshow(image_pil)
                bboxes = parsed_answer[task_prompt]['bboxes']
                labels = parsed_answer[task_prompt]['labels']

                mask_indexes = []
                # Determine mask indexes outside the loop
                if output_mask_select != "":
                    mask_indexes = [n for n in output_mask_select.split(",")]
                    _flog(str(mask_indexes))
                else:
                    mask_indexes = [str(i) for i in range(len(bboxes))]

                # Initialize mask_layer only if needed
                if fill_mask:
                    mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                    # Modify the label to include the index
                    indexed_label = f"{index}.{label}"
                    
                    if fill_mask:
                        # Ensure y1 is greater than or equal to y0 for mask drawing
                        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                        if y1 < y0:
                            y0, y1 = y1, y0
                        if x1 < x0:
                            x0, x1 = x1, x0
                            
                        if str(index) in mask_indexes:
                            _flog(f"match index: {str(index)} in mask_indexes: {mask_indexes}")
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                        if label in mask_indexes:
                            _flog("match label")
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

                    # Create a Rectangle patch
                    # Ensure y1 is greater than or equal to y0
                    y0, y1 = bbox[1], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    
                    rect = patches.Rectangle(
                        (bbox[0], y0),  # (x,y) - lower left corner
                        bbox[2] - bbox[0],   # Width
                        y1 - y0,   # Height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none',
                        label=indexed_label
                    )
                     # Calculate text width with a rough estimation
                    text_width = len(label) * 6  # Adjust multiplier based on your font size
                    text_height = 12  # Adjust based on your font size

                    # Get corrected coordinates
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    if x1 < x0:
                        x0, x1 = x1, x0

                    # Initial text position
                    text_x = x0
                    text_y = y0 - text_height  # Position text above the top-left of the bbox

                    # Adjust text_x if text is going off the left or right edge
                    if text_x < 0:
                        text_x = 0
                    elif text_x + text_width > W:
                        text_x = W - text_width

                    # Adjust text_y if text is going off the top edge
                    if text_y < 0:
                        text_y = y1  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox

                    # Add the rectangle to the plot
                    ax.add_patch(rect)
                    facecolor = random.choice(colormap) if len(image) == 1 else 'red'
                    # Add the label
                    plt.text(
                        text_x,
                        text_y,
                        indexed_label,
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor=facecolor, alpha=0.5)
                    )
                if fill_mask:             
                    mask_tensor = F.to_tensor(mask_layer)
                    mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                    mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                    mask_tensor = mask_tensor[:, :, :, 0]
                    out_masks.append(mask_tensor)           

                # Remove axis and padding around the image
                ax.axis('off')
                ax.margins(0,0)
                ax.get_xaxis().set_major_locator(plt.NullLocator())
                ax.get_yaxis().set_major_locator(plt.NullLocator())
                fig.canvas.draw() 
                buf = io.BytesIO()
                plt.savefig(buf, format='png', pad_inches=0)
                buf.seek(0)
                annotated_image_pil = Image.open(buf)

                annotated_image_tensor = F.to_tensor(annotated_image_pil)
                out_tensor = annotated_image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(out_tensor)
               
                if task == 'caption_to_phrase_grounding':
                    out_data.append(parsed_answer[task_prompt])
                else:
                    out_data.append(bboxes)

                
                pbar.update(1)
    
                plt.close(fig)

            elif task == 'referring_expression_segmentation':
                # Create a new black image
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
  
                predictions = parsed_answer[task_prompt]
    
                # Iterate over polygons and labels  
                for polygons, label in zip(predictions['polygons'], predictions['labels']):
                    color = random.choice(colormap)
                    for _polygon in polygons:  
                        _polygon = np.array(_polygon).reshape(-1, 2)
                        # Clamp polygon points to image boundaries
                        _polygon = np.clip(_polygon, [0, 0], [W - 1, H - 1])
                        if len(_polygon) < 3:  
                            _flog(f"Invalid polygon: {_polygon}")
                            continue  
                        
                        _polygon = _polygon.reshape(-1).tolist()
                        
                        # Draw the polygon
                        if fill_mask:
                            overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                            image_pil = image_pil.convert('RGBA')
                            draw = ImageDraw.Draw(overlay)
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(_polygon, outline=color, fill=color_with_opacity, width=3)
                            image_pil = Image.alpha_composite(image_pil, overlay)
                        else:
                            draw = ImageDraw.Draw(image_pil)
                            draw.polygon(_polygon, outline=color, width=3)

                        #draw mask
                        mask_draw.polygon(_polygon, outline="white", fill="white")
                        
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float() 
                out.append(image_tensor)

                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)
                pbar.update(1)

            elif task == 'ocr_with_region':
                try:
                    font = ImageFont.load_default().font_variant(size=24)
                except:
                    font = ImageFont.load_default()
                predictions = parsed_answer[task_prompt]
                scale = 1
                image_pil = image_pil.convert('RGBA')
                overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                bboxes, labels = predictions['quad_boxes'], predictions['labels']
                
                # Create a new black image for the mask
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
                
                for box, label in zip(bboxes, labels):
                    scaled_box = [v / (width if idx % 2 == 0 else height) for idx, v in enumerate(box)]
                    out_data.append({"label": label, "box": scaled_box})
                    
                    color = random.choice(colormap)
                    new_box = (np.array(box) * scale).tolist()
                    
                    # Ensure polygon coordinates are valid
                    # For polygons, we need to make sure the points form a valid shape
                    # This is a simple check to ensure the polygon has at least 3 points
                    if len(new_box) >= 6:  # At least 3 points (x,y pairs)
                        if fill_mask:
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(new_box, outline=color, fill=color_with_opacity, width=3)
                        else:
                            draw.polygon(new_box, outline=color, width=3)
                        
                        # Get the first point for text positioning
                        text_x, text_y = new_box[0]+8, new_box[1]+2
                        
                        draw.text((text_x, text_y),
                                  "{}".format(label),
                                  align="right",
                                  font=font,
                                  fill=color)
                        
                        # Draw the mask
                        mask_draw.polygon(new_box, outline="white", fill="white")
                
                image_pil = Image.alpha_composite(image_pil, overlay)
                image_pil = image_pil.convert('RGB')
                
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(image_tensor)

                # Process the mask
                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)

                pbar.update(1)
            
            elif task == 'docvqa':
                if text_input == "":
                    raise ValueError("Text input (prompt) is required for 'docvqa'")
                prompt = "<DocVQA> " + text_input

                inputs = self._prepare_inputs(processor, prompt, image_pil, device, dtype, model)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    use_cache=False,
                )

                results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                clean_results = results.replace('</s>', '').replace('<s>', '')
                
                if len(image) == 1:
                    out_results = clean_results
                else:
                    out_results.append(clean_results)
                    
                out.append(F.to_tensor(image_pil).unsqueeze(0).permute(0, 2, 3, 1).cpu().float())

                pbar.update(1)
            
        if len(out) > 0:
            out_tensor = torch.cat(out, dim=0)
        else:
            out_tensor = torch.zeros((1, 64,64, 3), dtype=torch.float32, device="cpu")
        if len(out_masks) > 0:
            out_mask_tensor = torch.cat(out_masks, dim=0)
        else:
            out_mask_tensor = torch.zeros((1,64,64), dtype=torch.float32, device="cpu")

        if not keep_model_loaded:
            _flog("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()
        
        return (out_tensor, out_mask_tensor, out_results, out_data)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model,
    "DownloadAndLoadFlorence2Lora": DownloadAndLoadFlorence2Lora,
    "Florence2ModelLoader": Florence2ModelLoader,
    "Florence2Run": Florence2Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "DownloadAndLoadFlorence2Model",
    "DownloadAndLoadFlorence2Lora": "DownloadAndLoadFlorence2Lora",
    "Florence2ModelLoader": "Florence2ModelLoader",
    "Florence2Run": "Florence2Run",
}
