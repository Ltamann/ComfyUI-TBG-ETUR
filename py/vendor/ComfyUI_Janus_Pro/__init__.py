from .nodes.image_generation import JanusImageGeneration
from .nodes.image_understanding import JanusImageUnderstanding
from .nodes.model_loader import JanusModelLoader

NODE_CLASS_MAPPINGS = {
    "JanusModelLoader": JanusModelLoader,
    "JanusImageUnderstanding": JanusImageUnderstanding,
    "JanusImageGeneration": JanusImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusModelLoader": "Janus Model Loader",
    "JanusImageUnderstanding": "Janus Image Understanding",
    "JanusImageGeneration": "Janus Image Generation",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 