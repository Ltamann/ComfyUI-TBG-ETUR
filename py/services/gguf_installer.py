import re
import subprocess
import sys
from typing import Dict, Optional


SERVER_REQUIREMENTS = [
    "uvicorn>=0.22.0",
    "fastapi>=0.100.0",
    "pydantic-settings>=2.0.1",
    "sse-starlette>=1.6.1",
    "starlette-context>=0.3.6,<0.4",
]

WHEEL_MAP = {
    ("win32", "cp312", "cu128"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu128-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
    ("win32", "cp312", "cu130"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu130-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
    ("win32", "cp312", "cu131"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu130-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
}


def _run(*cmd: str) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _python_abi_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _detect_cuda_tag() -> Optional[str]:
    try:
        out = _run("nvidia-smi")
        m = re.search(r"CUDA Version:\\s*([0-9]+)\\.([0-9]+)", out)
        if m:
            return f"cu{m.group(1)}{m.group(2)}"
    except Exception:
        return None
    return None


def _is_llama_cpp_installed() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


def _pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *args])


def _manual_install_command() -> str:
    abi = _python_abi_tag()
    cuda_tag = _detect_cuda_tag() or "<cuXYZ>"
    return (
        f"{sys.executable} -m pip install -U pip setuptools wheel scikit-build-core cmake ninja && "
        f"{sys.executable} -m pip install -U <llama_cpp_python wheel for platform={sys.platform}, abi={abi}, cuda={cuda_tag}>"
    )


def _build_advisory() -> str:
    return (
        "Optional GGUF install also sets up local llama-cpp server runtime dependencies. "
        "If you already have a llama-cpp/OpenAI-compatible server running, it is recommended to use that existing server "
        "through TBG ETUR Labs (OpenAI-Compatible setup) instead of running a second local server. "
        "This avoids port conflicts and duplicate model/VRAM usage."
    )


def install_optional_gguf_runtime() -> Dict[str, object]:
    advisory = _build_advisory()
    try:
        already = _is_llama_cpp_installed()
        if already:
            return {
                "ok": True,
                "status": "already_installed",
                "message": f"llama-cpp-python already installed. Restart ComfyUI before using GGUF. {advisory}",
                "requires_restart": True,
                "manual_command": None,
            }

        abi = _python_abi_tag()
        cuda_tag = _detect_cuda_tag()
        wheel_url = WHEEL_MAP.get((sys.platform, abi, cuda_tag))
        if not wheel_url:
            return {
                "ok": False,
                "status": "no_compatible_wheel",
                "message": (
                    "No compatible hardcoded llama-cpp-python wheel for this setup. "
                    f"Detected platform={sys.platform}, abi={abi}, cuda={cuda_tag}. "
                    "Install manually if you want GGUF support."
                ),
                "requires_restart": False,
                "manual_command": _manual_install_command(),
            }

        _pip_install("pip", "setuptools", "wheel", "scikit-build-core", "cmake", "ninja")
        _pip_install(wheel_url)
        _pip_install(*SERVER_REQUIREMENTS)

        return {
            "ok": True,
            "status": "installed",
            "message": f"GGUF runtime installed successfully. Restart ComfyUI before using GGUF. {advisory}",
            "requires_restart": True,
            "manual_command": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "message": f"GGUF runtime install failed: {exc}",
            "requires_restart": False,
            "manual_command": _manual_install_command(),
        }
