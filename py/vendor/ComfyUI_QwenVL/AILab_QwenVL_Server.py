import base64
import io
import json
import os
import time
from pathlib import Path
from urllib import error, parse, request

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

import folder_paths
from .response_cleanup import strip_thinking


NODE_DIR = Path(__file__).parent
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"


def _safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[QwenVL-Server] Failed to load GGUF catalog: {exc}")
        return {}


def _normalized_catalog():
    raw = _safe_json_load(GGUF_CONFIG_PATH)
    normalized = {}

    for alias, cfg in raw.items():
        if isinstance(cfg, dict) and "model_file" in cfg and "mmproj_file" in cfg:
            normalized[alias] = cfg

    qwen_vl_section = raw.get("qwenVL_model")
    if isinstance(qwen_vl_section, dict):
        for _repo_name, repo_cfg in qwen_vl_section.items():
            if not isinstance(repo_cfg, dict):
                continue
            repo_id = repo_cfg.get("repo_id")
            mmproj_file = repo_cfg.get("mmproj_file")
            model_files = repo_cfg.get("model_files") or []
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
                        "server_model": alias,
                    },
                )

    return normalized


def _normalize_v1_base(base_url):
    base = (base_url or "http://127.0.0.1:11434/v1").strip().rstrip("/")
    if not base.endswith("/v1"):
        if base.endswith("/v1/"):
            base = base[:-1]
        else:
            base = f"{base}/v1"
    return base


def _parse_openai_content(payload):
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return None


class QwenVLServerClient:
    def __init__(self, server_cfg):
        self.server_cfg = server_cfg or {}
        self.base_v1 = ""
        self.chat_url = ""
        self.health_url = ""
        self.reconfigure_base_url(self.server_cfg.get("base_url"), self.server_cfg.get("health_endpoint", "/v1/models"))
        self.timeout_s = float(self.server_cfg.get("request_timeout_s", 90))
        self._catalog = _normalized_catalog()
        self._resolved_alias = {}

    def reconfigure_base_url(self, base_url, health_endpoint="/v1/models"):
        self.base_v1 = _normalize_v1_base(base_url)
        self.chat_url = f"{self.base_v1}/chat/completions"
        self.health_url = self._resolve_health_url(health_endpoint)

    def _resolve_health_url(self, endpoint):
        endpoint = (endpoint or "/v1/models").strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        parsed = parse.urlparse(self.base_v1)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if endpoint.startswith("/v1"):
            return f"{root}{endpoint}"
        return f"{self.base_v1}{endpoint}"

    @staticmethod
    def _tensor_to_data_uri(tensor):
        if tensor is None:
            raise ValueError("[QwenVL-Server] Missing image input.")
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        env_name = (self.server_cfg.get("api_key_env") or "TBG_ETUR_OPENAI_API_KEY").strip()
        api_key = os.getenv(env_name, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def resolve_alias(self, alias):
        if alias in self._resolved_alias:
            return self._resolved_alias[alias]

        cfg = self._catalog.get(alias)
        if cfg is None:
            supported = ", ".join(sorted(self._catalog.keys()))
            raise ValueError(f"[QwenVL-Server] Unknown GGUF alias '{alias}'. Supported: {supported}")

        repo_id = cfg.get("repo_id")
        model_file = cfg.get("model_file")
        mmproj_file = cfg.get("mmproj_file")
        if not repo_id or not model_file or not mmproj_file:
            raise ValueError(f"[QwenVL-Server] Incomplete catalog entry for alias '{alias}'.")

        model_root = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL-GGUF" / alias
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / model_file
        mmproj_path = model_root / mmproj_file

        repo_ids = [repo_id] + [r for r in (cfg.get("alt_repo_ids") or []) if r and r != repo_id]

        def _download_if_missing(file_path, filename):
            if file_path.exists():
                return "hit", repo_id
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
                    return "miss", rid
                except Exception as exc:
                    last_exc = exc
            raise FileNotFoundError(
                f"[QwenVL-Server] Could not download '{filename}' for alias '{alias}'. Attempted repos: {attempted}."
            ) from last_exc

        model_cache, model_repo = _download_if_missing(model_path, model_file)
        mmproj_cache, mmproj_repo = _download_if_missing(mmproj_path, mmproj_file)
        print(
            f"[QwenVL-Server] Alias='{alias}' model='{model_file}' mmproj='{mmproj_file}' "
            f"model_repo='{model_repo}' mmproj_repo='{mmproj_repo}'"
        )
        print(f"[QwenVL-Server] Download cache model={model_cache} mmproj={mmproj_cache}")

        resolved = {
            "alias": alias,
            "server_model": cfg.get("server_model") or Path(model_file).stem,
            "repo_id": repo_id,
            "model_file": model_file,
            "mmproj_file": mmproj_file,
            "model_path": str(model_path),
            "mmproj_path": str(mmproj_path),
        }
        self._resolved_alias[alias] = resolved
        return resolved

    def health_check(self):
        req = request.Request(self.health_url, headers=self._headers(), method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return int(resp.status), resp.read(256).decode("utf-8", errors="ignore")

    def run(self, alias, custom_prompt, image, seed=0):
        resolved = self.resolve_alias(alias)
        model_name = resolved["server_model"]
        image_data_uri = self._tensor_to_data_uri(image)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt or "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 1024,
            "seed": int(seed or 0),
            "presence_penalty": 1.5,
            "frequency_penalty": 1.0,
            "extra_body": {
                "top_k": 20,
                "repetition_penalty": 1.0,
            },
        }

        req = request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.time()
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                status = int(resp.status)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"[QwenVL-Server] Request failed stage=request alias='{alias}' model='{model_name}' "
                f"url='{self.base_v1}' status={getattr(exc, 'code', 'unknown')} body='{body[:512]}'"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"[QwenVL-Server] Request failed stage=request alias='{alias}' model='{model_name}' "
                f"url='{self.base_v1}' error={exc}"
            ) from exc

        if status < 200 or status >= 300:
            raise RuntimeError(
                f"[QwenVL-Server] Request failed stage=request alias='{alias}' model='{model_name}' "
                f"url='{self.base_v1}' status={status} body='{body[:512]}'"
            )

        try:
            parsed = json.loads(body)
        except Exception as exc:
            raise RuntimeError(
                f"[QwenVL-Server] Request failed stage=parse alias='{alias}' model='{model_name}' "
                f"url='{self.base_v1}' error=invalid_json body='{body[:512]}'"
            ) from exc

        content = strip_thinking(_parse_openai_content(parsed), "[QwenVL-Server]")
        if not content:
            raise RuntimeError(
                f"[QwenVL-Server] Request failed stage=parse alias='{alias}' model='{model_name}' "
                f"url='{self.base_v1}' error=missing_message_content"
            )
        elapsed = time.time() - started
        print(f"[QwenVL-Server] Tile request done alias='{alias}' model='{model_name}' latency={elapsed:.2f}s")
        return (content,)
