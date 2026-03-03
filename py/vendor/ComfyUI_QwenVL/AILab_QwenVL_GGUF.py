import base64
import gc
import inspect
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import folder_paths


NODE_DIR = Path(__file__).parent
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
GGUF_MODEL_CONFIGS = {}


def _load_gguf_model_configs():
    global GGUF_MODEL_CONFIGS
    try:
        with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
            GGUF_MODEL_CONFIGS = json.load(fh)
    except Exception as exc:
        GGUF_MODEL_CONFIGS = {}
        print(f"[QwenVL-GGUF] Config load failed: {exc}")


_load_gguf_model_configs()


def _normalized_gguf_catalog():
    """
    Normalize both supported catalog shapes:
    1) Flat alias map:
       {"Alias": {...}, "Alias2": {...}}
    2) Upstream nested map:
       {"qwenVL_model": {"RepoName": {"repo_id":..., "mmproj_file":..., "model_files":[...]}}}
    """
    raw = GGUF_MODEL_CONFIGS or {}
    normalized = {}

    # Flat alias map (current ETUR style)
    for key, value in raw.items():
        if key in {"base_dir", "Qwen_model", "qwenVL_model"}:
            continue
        if isinstance(value, dict) and "model_file" in value and "mmproj_file" in value:
            normalized[key] = value

    # Upstream nested structure (ComfyUI-QwenVL style)
    qwen_vl_section = raw.get("qwenVL_model")
    if isinstance(qwen_vl_section, dict):
        for repo_name, repo_cfg in qwen_vl_section.items():
            if not isinstance(repo_cfg, dict):
                continue
            model_files = repo_cfg.get("model_files") or []
            mmproj_file = repo_cfg.get("mmproj_file")
            repo_id = repo_cfg.get("repo_id")
            if not repo_id or not mmproj_file:
                continue

            for model_file in model_files:
                alias = Path(model_file).stem
                normalized.setdefault(
                    alias,
                    {
                        "repo_id": repo_id,
                        "alt_repo_ids": repo_cfg.get("alt_repo_ids", []),
                        "model_file": model_file,
                        "mmproj_file": mmproj_file,
                        "chat_format": "qwen2vl",
                        "n_ctx": int((repo_cfg.get("defaults") or {}).get("context_length", 4096)),
                        "n_gpu_layers": int((repo_cfg.get("defaults") or {}).get("gpu_layers", -1)),
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "_source_repo_name": repo_name,
                    },
                )
    return normalized


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.current_signature = None
        print("[QwenVL-GGUF] Node initialized")

    def clear(self):
        self.llm = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _parse_alias(model_name):
        prefix = "GGUF\\"
        if not model_name.startswith(prefix):
            raise ValueError(f"[QwenVL-GGUF] Invalid GGUF model name '{model_name}'")
        alias = model_name[len(prefix) :].strip()
        if not alias:
            raise ValueError("[QwenVL-GGUF] Empty GGUF alias")
        return alias

    @staticmethod
    def _tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _pil_to_data_uri(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _build_chat_handler(chat_format, mmproj_path):
        try:
            from llama_cpp import llama_chat_format as lcf
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL-GGUF] llama-cpp-python is installed but chat_format helpers are unavailable."
            ) from exc

        if chat_format != "qwen2vl":
            raise ValueError(f"[QwenVL-GGUF] Unsupported chat_format '{chat_format}'")

        handler_cls = None
        for handler_name in ("Qwen3VLChatHandler", "Qwen25VLChatHandler", "Qwen2VLChatHandler"):
            if hasattr(lcf, handler_name):
                handler_cls = getattr(lcf, handler_name)
                print(f"[QwenVL-GGUF] Using chat handler '{handler_name}'")
                break
        if handler_cls is None:
            available = [n for n in dir(lcf) if "Qwen" in n and "ChatHandler" in n]
            raise RuntimeError(
                "[QwenVL-GGUF] This llama-cpp-python build does not include a supported Qwen VL chat handler. "
                f"Available Qwen handlers: {available}"
            )

        sig = inspect.signature(handler_cls)
        params = sig.parameters
        candidate_keys = ("clip_model_path", "mmproj_path", "mmproj", "projector_path", "model_path", "proj_path")

        # 1) Explicit parameter support (legacy and explicit handlers)
        for key in candidate_keys:
            if key in params:
                print(f"[QwenVL-GGUF] Chat handler mmproj key: '{key}' (explicit)")
                return handler_cls(**{key: mmproj_path})

        # 2) kwargs-only handlers (e.g., Qwen3VLChatHandler with **kwargs)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        attempted_keys = []
        if has_var_kw:
            for key in candidate_keys:
                attempted_keys.append(key)
                try:
                    print(f"[QwenVL-GGUF] Chat handler mmproj key: '{key}' (via **kwargs)")
                    return handler_cls(**{key: mmproj_path})
                except TypeError:
                    # Try next alias for handlers that validate kwargs internally
                    continue
                except Exception:
                    # Non-TypeError means constructor accepted path key but failed elsewhere;
                    # surface upstream error directly.
                    raise

        available_qwen_handlers = [n for n in dir(lcf) if "Qwen" in n and "ChatHandler" in n]
        raise RuntimeError(
            "[QwenVL-GGUF] Could not resolve mmproj argument for detected Qwen VL chat handler. "
            f"handler={handler_cls.__name__}, signature={sig}, attempted_keys={attempted_keys or list(candidate_keys)}, "
            f"supports_kwargs={has_var_kw}, available_qwen_handlers={available_qwen_handlers}. "
            "This build may use a newer kwargs-only constructor; update llama-cpp-python if needed."
        )

    @staticmethod
    def _llama_supports_qwen3vl():
        """
        Best-effort feature probe for qwen3vl support in installed llama-cpp-python.
        Older builds can parse GGUF headers but fail to instantiate qwen3vl models.
        """
        try:
            import llama_cpp  # local import to keep GGUF optional

            pkg_dir = Path(llama_cpp.__file__).parent
            for py_file in pkg_dir.rglob("*.py"):
                try:
                    text = py_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if "qwen3vl" in text.lower():
                    return True
        except Exception:
            return False
        return False

    def _resolve_model_files(self, alias):
        catalog = _normalized_gguf_catalog()
        cfg = catalog.get(alias)
        if cfg is None:
            supported = ", ".join(sorted(catalog.keys()))
            raise ValueError(f"[QwenVL-GGUF] Unknown GGUF alias '{alias}'. Supported: {supported}")

        repo_id = cfg["repo_id"]
        alt_repo_ids = [x for x in (cfg.get("alt_repo_ids") or []) if x]
        repo_ids = [repo_id] + [x for x in alt_repo_ids if x != repo_id]
        model_file = cfg["model_file"]
        mmproj_file = cfg["mmproj_file"]
        model_root = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL-GGUF" / alias
        model_root.mkdir(parents=True, exist_ok=True)

        model_path = model_root / model_file
        mmproj_path = model_root / mmproj_file
        model_hit = model_path.exists()
        mmproj_hit = mmproj_path.exists()
        print(
            f"[QwenVL-GGUF] Alias='{alias}' repo='{repo_id}' model='{model_file}' mmproj='{mmproj_file}'"
        )
        print(f"[QwenVL-GGUF] Download cache model={'hit' if model_hit else 'miss'} mmproj={'hit' if mmproj_hit else 'miss'}")

        def _download_with_fallback(filename):
            last_exc = None
            attempted = []
            for rid in repo_ids:
                attempted.append(rid)
                try:
                    hf_hub_download(
                        repo_id=rid,
                        filename=filename,
                        local_dir=str(model_root),
                        local_dir_use_symlinks=False,
                    )
                    return attempted
                except Exception as exc:
                    last_exc = exc
                    print(f"[QwenVL-GGUF] Download failed repo='{rid}' file='{filename}': {exc}")
            raise FileNotFoundError(
                f"[QwenVL-GGUF] Could not download file '{filename}' for alias '{alias}'. "
                f"Attempted repos: {attempted}. Hint: verify names in gguf_models.json "
                f"or use local predownloaded files."
            ) from last_exc

        tried_model_repos = []
        tried_mmproj_repos = []
        if not model_hit:
            tried_model_repos = _download_with_fallback(model_file)
        if not mmproj_hit:
            tried_mmproj_repos = _download_with_fallback(mmproj_file)

        if not model_path.exists():
            raise FileNotFoundError(
                f"[QwenVL-GGUF] Missing GGUF model file for alias '{alias}': '{model_file}'. "
                f"Path checked: {model_path}. "
                f"Attempted repos: {tried_model_repos or repo_ids}. "
                f"Hint: verify names in gguf_models.json or use local predownloaded files."
            )
        if not mmproj_path.exists():
            raise FileNotFoundError(
                f"[QwenVL-GGUF] Missing mmproj file for alias '{alias}': '{mmproj_file}'. "
                f"Path checked: {mmproj_path}. "
                f"Attempted repos: {tried_mmproj_repos or repo_ids}. "
                f"Hint: verify names in gguf_models.json or use local predownloaded files."
            )
        return cfg, str(model_path), str(mmproj_path)

    def _load_model(self, model_name, keep_model_loaded):
        alias = self._parse_alias(model_name)
        cfg, model_path, mmproj_path = self._resolve_model_files(alias)
        chat_format = cfg.get("chat_format", "qwen2vl")

        signature = (alias, model_path, mmproj_path, cfg.get("n_ctx", 4096), cfg.get("n_gpu_layers", -1))
        if keep_model_loaded and self.llm is not None and self.current_signature == signature:
            return cfg

        self.clear()

        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL-GGUF] Missing dependency 'llama-cpp-python'. "
                "This usually means the node was updated without reinstalling requirements. "
                "Install a compatible wheel with Comfy embedded Python "
                "(GGUF runtime install is optional and manual)."
            ) from exc

        if alias.startswith("Qwen3-VL-") and not self._llama_supports_qwen3vl():
            raise RuntimeError(
                "[QwenVL-GGUF] Installed llama-cpp-python build does not support Qwen3-VL GGUF yet.\n"
                "Please upgrade/reinstall in Comfy embedded env:\n"
                "  1) python_embeded\\python.exe -m pip install -U pip setuptools wheel scikit-build-core cmake ninja\n"
                "  2) python_embeded\\python.exe -m pip install -U "
                "https://github.com/JamePeng/llama-cpp-python/releases/download/"
                "v0.3.17-cu128-cudnn/llama_cpp_python-0.3.17-cp312-cp312-win_amd64.whl\n"
                "If needed, use a newer prebuilt wheel or build llama-cpp-python from source with a recent llama.cpp."
            )

        chat_handler = self._build_chat_handler(chat_format, mmproj_path)
        print(
            f"[QwenVL-GGUF] Loading backend=gguf alias='{alias}' n_ctx={cfg.get('n_ctx', 4096)} "
            f"n_gpu_layers={cfg.get('n_gpu_layers', -1)}"
        )
        try:
            self.llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=int(cfg.get("n_ctx", 4096)),
                n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
                verbose=False,
            )
        except Exception as exc:
            # Common failure for older llama.cpp builds: model metadata is readable,
            # but new architectures like qwen3vl are not compiled/supported yet.
            msg = str(exc)
            lower_msg = msg.lower()
            if "failed to load model from file" in lower_msg:
                import traceback

                tb = traceback.format_exc().lower()
                if "unknown model architecture: 'qwen3vl'" in tb or "unknown model architecture: 'qwen3vl'" in lower_msg:
                    raise RuntimeError(
                        "[QwenVL-GGUF] Your installed llama-cpp-python build does not support the "
                        "Qwen3-VL GGUF architecture ('qwen3vl').\n"
                        "Please upgrade/reinstall llama-cpp-python in the Comfy embedded env.\n"
                        "Recommended order:\n"
                        "  1) python_embeded\\python.exe -m pip install -U pip setuptools wheel scikit-build-core cmake ninja\n"
                        "  2) python_embeded\\python.exe -m pip install -U "
                        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
                        "v0.3.17-cu128-cudnn/llama_cpp_python-0.3.17-cp312-cp312-win_amd64.whl\n"
                        "If this still fails on Windows, install a newer prebuilt wheel or build llama-cpp-python "
                        "from source with a recent llama.cpp."
                    ) from exc
            raise
        self.current_signature = signature
        return cfg

    def run(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        keep_model_loaded,
        attention_mode,
        use_torch_compile,
        device,
    ):
        if quantization:
            print(f"[QwenVL-GGUF] VLM_Quantization is ignored for GGUF models: '{quantization}'")

        if video is not None:
            raise ValueError("[QwenVL-GGUF] Video input is not supported in this GGUF path")

        cfg = self._load_model(model_name, keep_model_loaded=keep_model_loaded)
        prompt = (custom_prompt or "").strip() or (preset_prompt or "").strip()
        pil_image = self._tensor_to_pil(image)
        if pil_image is None:
            raise ValueError("[QwenVL-GGUF] Missing image input")
        image_data_uri = self._pil_to_data_uri(pil_image)

        effective_temperature = float(cfg.get("temperature", temperature))
        effective_top_p = float(cfg.get("top_p", top_p))
        effective_seed = int(seed) if seed is not None else -1

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=effective_temperature,
            top_p=effective_top_p,
            seed=effective_seed,
        )
        text = response["choices"][0]["message"]["content"]

        if not keep_model_loaded:
            self.clear()
        return (text,)
