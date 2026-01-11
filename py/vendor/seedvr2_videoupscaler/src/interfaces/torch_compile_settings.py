"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from comfy_api.latest import io
from typing import Dict, Any, Tuple


class SeedVR2TorchCompileSettings():
    """Configure torch.compile optimization for DiT and VAE models"""
    

    @classmethod
    def execute(cls, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int) -> io.NodeOutput:
        """
        Create torch.compile configuration for model optimization
        
        Args:
            backend: Compilation backend ("inductor" or "cudagraphs")
            mode: Optimization mode ("default", "reduce-overhead", "max-autotune", etc.)
            fullgraph: Whether to compile entire model as single graph
            dynamic: Whether to handle varying input shapes without recompilation
            dynamo_cache_size_limit: Maximum cached compiled versions per function
            dynamo_recompile_limit: Maximum recompilation attempts before fallback
            
        Returns:
            NodeOutput containing torch.compile configuration dictionary
        """
        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
        }
        return io.NodeOutput(compile_args)