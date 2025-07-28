import gc
import time

import comfy.model_management as model_management
import torch


# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
any = AnyType("*")

class UnloadOneModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any, )},
            "optional": {"model": (any, )},
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any, )
    FUNCTION = "route"
    CATEGORY = "Unload Modules"

    def route(self, **kwargs):
        
        # Remove model if we can
        print("Unload Model:")
        loaded_models = model_management.loaded_models()
        if kwargs.get("model") in loaded_models:
            print(" - Model found in memory, unloading...")
            loaded_models.remove(kwargs.get("model"))
        model_management.free_memory(1e30, model_management.get_torch_device(), loaded_models)
        model_management.soft_empty_cache(True)
        try:
            print(" - Clearing Cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            print("      - Unable to clear cache")
        time.sleep(2)

        return (list(kwargs.values()))
   

NODE_CLASS_MAPPINGS = {
    "UnloadOneModel": UnloadOneModelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnloadOneModel": "Unload One Model",
}
