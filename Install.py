import os
import subprocess
import sys
import re

import requests


def install_requirements():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def _run(*cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _python_abi_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _detect_cuda_tag():
    """
    Convert `nvidia-smi` CUDA version to release token, e.g.:
    12.8 -> cu128
    """
    try:
        out = _run("nvidia-smi")
        m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
        if m:
            return f"cu{m.group(1)}{m.group(2)}"
    except Exception:
        pass
    return None


def _install_llama_cpp_python():
    print("Installing GGUF runtime (llama-cpp-python)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "scikit-build-core", "cmake", "ninja"]
        )
    except Exception as exc:
        print(f"[TBG Installer] GGUF bootstrap deps update failed (non-fatal): {exc}")
        print("[TBG Installer] Continuing without GGUF auto-install. Install manually if GGUF is needed.")
        return False

    abi = _python_abi_tag()
    cuda_tag = _detect_cuda_tag()
    platform_key = sys.platform
    print(f"Detected platform={platform_key}, ABI={abi}, CUDA={cuda_tag or 'unknown'}")

    # Hardcoded wheel matrix. Add new entries here when validating new setups.
    # Key: (platform, abi, cuda_tag)
    wheel_map = {
        ("win32", "cp312", "cu128"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu128-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
        ("win32", "cp312", "cu130"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu130-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
        # Driver/runtime reports CUDA 13.1 on some systems; use cu130 wheel as closest supported build.
        ("win32", "cp312", "cu131"): "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.27-cu130-Basic-win-20260223/llama_cpp_python-0.3.27-cp312-cp312-win_amd64.whl",
    }

    wheel_url = wheel_map.get((platform_key, abi, cuda_tag))
    if not wheel_url:
        print(
            "[TBG Installer] No hardcoded llama-cpp-python wheel for this setup. "
            f"(platform={platform_key}, abi={abi}, cuda={cuda_tag})\n"
            "GGUF will not be auto-installed. Install a compatible wheel manually "
            "or do not use GGUF models."
        )
        return False

    print(f"[TBG Installer] Installing llama-cpp-python wheel: {wheel_url}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", wheel_url])
    except Exception as exc:
        print(
            "[TBG Installer] Failed to install hardcoded llama-cpp-python wheel.\n"
            f"Reason: {exc}\n"
            "Install a compatible wheel manually or do not use GGUF models."
        )
        return False
    return True


def _install_llama_cpp_server_requirements():
    """
    Install llama_cpp.server runtime dependencies only when GGUF runtime
    was successfully installed via a compatible wheel.
    """
    server_reqs = [
        "uvicorn>=0.22.0",
        "fastapi>=0.100.0",
        "pydantic-settings>=2.0.1",
        "sse-starlette>=1.6.1",
        "starlette-context>=0.3.6,<0.4",
    ]
    print("[TBG Installer] Installing llama_cpp.server runtime dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *server_reqs])
    except Exception as exc:
        quoted = " ".join([f'"{r}"' if any(ch in r for ch in "<>") else r for r in server_reqs])
        raise RuntimeError(
            "[TBG Installer] Failed to install llama_cpp.server dependencies.\n"
            f"Reason: {exc}\n"
            f"Please install manually with: {sys.executable} -m pip install -U {quoted}"
        ) from exc

def download_depth_anything_v2_vitl():
    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth"
    target_dir = os.path.join(
        "py", "vendor", "comfyui_controlnet_aux", "ckpts",
        "depth-anything", "Depth-Anything-V2-Large"
    )
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, "depth_anything_v2_vitl.pth")

    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}")
        return

    print("Downloading model...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded model to {dest_path}")

if __name__ == "__main__":
    download_depth_anything_v2_vitl()
    install_requirements()


