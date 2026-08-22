from pathlib import Path

import yaml


_PRESET_FILE = Path(__file__).with_name("TBG_Refiner_Presets.yaml")

def get_refiner_preset(model_type):
    try:
        with _PRESET_FILE.open("r", encoding="utf-8") as stream:
            presets = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[TBG] Refiner preset file unavailable: {exc}")
        return {}

    return {
        **presets.get("defaults", {}),
        **presets.get("models", {}).get(model_type, {}),
    }
