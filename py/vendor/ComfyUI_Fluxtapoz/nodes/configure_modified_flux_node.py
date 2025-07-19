from ..flux.layers import inject_blocks
from ..flux.model import inject_flux


class ConfigureModifiedFluxNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "fluxtapoz"
    FUNCTION = "apply"

    def apply(self, model):
        inject_flux(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)

