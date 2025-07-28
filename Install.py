import os
import subprocess
import sys

import requests


def install_requirements():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

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


