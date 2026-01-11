"""
SeedVR2 DiT Model Loader Node
Configure DiT (Diffusion Transformer) model with memory optimization
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_dit_models, DEFAULT_DIT
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadDiTModel():
    """
    Configure DiT (Diffusion Transformer) model loader with memory optimization
    
    Provides configuration for:
    - Model selection and device placement
    - BlockSwap memory optimization for limited VRAM
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_DIT configuration dictionary for main upscaler node
    """
    

    @classmethod
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, blocks_to_swap: int = 0, 
                     swap_io_components: bool = False, attention_mode: str = "sdpa",
                     torch_compile_args: Dict[str, Any] = None) -> io.NodeOutput:
        """
        Create DiT model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            blocks_to_swap: Number of transformer blocks to swap (requires offload_device != device)
            swap_io_components: Whether to offload I/O components (requires offload_device != device)
            attention_mode: Attention computation backend ('sdpa' or 'flash_attn')
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If BlockSwap is enabled but offload_device is invalid
        """
        # Validate BlockSwap configuration
        if (blocks_to_swap > 0 or swap_io_components) and (offload_device == "none" or offload_device == device):
            raise ValueError(
                "BlockSwap requires offload_device to be set and different from device. "
                f"Current: device='{device}', offload_device='{offload_device}'. "
                "Please set offload_device to a different device (e.g., 'cpu' or another GPU)."
            )
        
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached DiT model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "attention_mode": attention_mode,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
        }
        
        return io.NodeOutput(config)