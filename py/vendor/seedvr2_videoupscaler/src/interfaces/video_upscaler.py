"""
SeedVR2 Video Upscaler Node
Main ComfyUI node for high-quality video upscaling using diffusion models
"""

import torch
from comfy_api.latest import io
from typing import Tuple, Dict, Any, Optional
from ..utils.constants import get_base_cache_dir, __version__
from ..utils.downloads import download_weight
from ..utils.debug import Debug
from ..core.generation_phases import (
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches,
    postprocess_all_batches
)
from ..core.generation_utils import (
    setup_generation_context, 
    prepare_runner,
    compute_generation_info,
    log_generation_start
)
from ..optimization.memory_manager import (
    cleanup_text_embeddings,
    complete_cleanup,
    get_device_list
)
from ......TBG.SERVERS.COMFYUI_server import register_main_class
# Import ComfyUI progress reporting
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

@register_main_class
class TBG_SeedVR2VideoUpscaler():
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """

    @classmethod
    def execute(cls, image: torch.Tensor, dit: Dict[str, Any], vae: Dict[str, Any], 
                seed: int, resolution: int = 1080, max_resolution: int = 0, batch_size: int = 5,
                uniform_batch_size: bool = False, temporal_overlap: int = 0, prepend_frames: int = 0,
                color_correction: str = "wavelet", input_noise_scale: float = 0.0,
                latent_noise_scale: float = 0.0, offload_device: str = "none", 
                enable_debug: bool = False, tbg_steps: int = 3) -> io.NodeOutput:
        """
        Execute SeedVR2 video upscaling with progress reporting
        
        Main entry point for ComfyUI node execution.
        Automatically detects and preserves input format (RGB or RGBA). Handles model downloads,
        configuration unpacking, and delegates to upscaling pipeline.
        
        Args:
            image: Input video frames as tensor (N, H, W, C) in [0, 1] range
            dit: DiT model configuration from SeedVR2LoadDiTModel node
            vae: VAE model configuration from SeedVR2LoadVAEModel node
            seed: Random seed for reproducible generation
            resolution: Target resolution for shortest edge (maintains aspect ratio)
            max_resolution: Maximum resolution for any edge (0 = no limit)
            batch_size: Frames per batch (minimum 5 for temporal consistency)
            uniform_batch_size: Whether to pad final batch to match batch_size
            temporal_overlap: Overlapping frames between batches (0-16)
            prepend_frames: Frames to prepend (0-32) to reduce initial artifacts.
            color_correction: Color correction method
            input_noise_scale: Input noise injection scale [0.0-1.0]
            latent_noise_scale: Latent noise injection scale [0.0-1.0]
            offload_device: Device to offload intermediate tensors
            enable_debug: Enable detailed logging and memory tracking
            
        Returns:
            NodeOutput containing upscaled video tensor (N, H', W', C) in [0, 1] range
            
        Raises:
            ValueError: If model files cannot be downloaded or configuration is invalid
            RuntimeError: If generation fails
        """
        # Initialize debug (stateless - stored in local variable)
        debug = Debug(enabled=enable_debug)
        
        # Track execution state in local variables (not instance)
        runner = None
        ctx = None
        pbar = None
        
        # Define progress callback as local closure
        def progress_callback(current_step: int, total_steps: int, 
                            current_frames: int, phase_name: str) -> None:
            """
            Update progress bar based on pipeline phase
            
            Args:
                current_step: Current step within phase
                total_steps: Total steps in phase
                current_frames: Number of frames being processed
                phase_name: Name of current phase
            """
            if pbar is None:
                return
            
            phase_weights = {
                "Phase 1: Encoding": 0.2,
                "Phase 2: Upscaling": 0.25,
                "Phase 3: Decoding": 0.5,
                "Phase 4: Post-processing": 0.05
            }
            phase_offset = {
                "Phase 1: Encoding": 0.0,
                "Phase 2: Upscaling": 0.2,
                "Phase 3: Decoding": 0.45,
                "Phase 4: Post-processing": 0.95
            }
            
            # Extract phase key from phase_name
            phase_key = phase_name.split(" (")[0] if " (" in phase_name else phase_name
            
            # Get weight and offset
            weight = phase_weights.get(phase_key, 1.0)
            offset = phase_offset.get(phase_key, 0.0)
            
            # Calculate weighted progress
            phase_progress = (current_step / total_steps) if total_steps > 0 else 0
            overall_progress = offset + (phase_progress * weight)
            
            # Update the progress bar with the overall progress
            progress_value = int(overall_progress * 100)
            pbar.update_absolute(progress_value, 100)
        
        # Define cleanup as local function
        def cleanup(dit_cache: bool = False, vae_cache: bool = False) -> None:
            """Cleanup resources after upscaling"""
            nonlocal runner, ctx
            
            # Use complete_cleanup for all cleanup operations
            if runner:
                complete_cleanup(runner=runner, debug=debug, 
                               dit_cache=dit_cache, vae_cache=vae_cache)
                
                # Delete runner only if neither model is cached
                if not (dit_cache or vae_cache):
                    runner = None
            
            # Clean up context text embeddings if they exist
            if ctx:
                cleanup_text_embeddings(ctx, debug)
                ctx = None
        
        # Extract configuration from dict inputs
        dit_model = dit["model"]
        vae_model = vae["model"]
        dit_device = torch.device(dit["device"])
        vae_device = torch.device(vae["device"])
        dit_id = dit["node_id"]
        vae_id = vae["node_id"]

        # OPTIONAL inputs - use .get() with defaults
        dit_cache = dit.get("cache_model", False)
        attention_mode = dit.get("attention_mode", "sdpa")
        vae_cache = vae.get("cache_model", False)

        # BlockSwap configuration - construct from individual values
        blocks_to_swap = dit.get("blocks_to_swap", 0)
        swap_io_components = dit.get("swap_io_components", False)
        dit_offload_str = dit.get("offload_device", "none")

        block_swap_config = None
        if blocks_to_swap > 0 or swap_io_components:
            # Convert offload device string to torch.device for BlockSwap
            if dit_offload_str != "none":
                block_swap_config = {
                    "blocks_to_swap": blocks_to_swap,
                    "swap_io_components": swap_io_components,
                    "offload_device": torch.device(dit_offload_str)
                }

        # Device configuration for offloading - convert "none" to None, else torch.device
        vae_offload_str = vae.get("offload_device", "none")
        dit_offload_device = torch.device(dit_offload_str) if dit_offload_str != "none" else None
        vae_offload_device = torch.device(vae_offload_str) if vae_offload_str != "none" else None
        tensor_offload_device = torch.device(offload_device) if offload_device != "none" else None

        # VAE tiling configuration
        encode_tiled = vae.get("encode_tiled", False)
        encode_tile_size = vae.get("encode_tile_size", 512)
        encode_tile_overlap = vae.get("encode_tile_overlap", 64)
        decode_tiled = vae.get("decode_tiled", False)
        decode_tile_size = vae.get("decode_tile_size", 512)
        decode_tile_overlap = vae.get("decode_tile_overlap", 64)
        tile_debug = vae.get("tile_debug", False)

        # TorchCompile args (optional connection, can be None)
        dit_torch_compile_args = dit.get("torch_compile_args")
        vae_torch_compile_args = vae.get("torch_compile_args")
        
        # Print header
        debug.print_header()

        debug.start_timer("total_execution", force=True)

        debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        debug.log_memory_state("Before model preparation", show_tensors=False, detailed_tensors=False)
        debug.start_timer("model_preparation")

        # Check if download succeeded
        debug.log("Checking and downloading models if needed...", category="download")
        if not download_weight(dit_model=dit_model, vae_model=vae_model, debug=debug):
            raise RuntimeError(
                f"Failed to download required model files. "
                f"DiT model: {dit_model}, VAE model: {vae_model}. "
                "Please check the console output above for specific file failures and manual download instructions."
            )
        
        try:
            # Initialize ComfyUI progress bar if available
            if ProgressBar is not None:
                pbar = ProgressBar(100)
            
            # Setup generation context with device configuration
            ctx = setup_generation_context(
                dit_device=dit_device,
                vae_device=vae_device,
                dit_offload_device=dit_offload_device,
                vae_offload_device=vae_offload_device,
                tensor_offload_device=tensor_offload_device,
                debug=debug
            )

            # Prepare runner with model state management and global cache
            runner, cache_context = prepare_runner(
                dit_model=dit_model, 
                vae_model=vae_model, 
                model_dir=get_base_cache_dir(),
                debug=debug,
                ctx=ctx,
                dit_cache=dit_cache,
                vae_cache=vae_cache,
                dit_id=dit_id,
                vae_id=vae_id,
                block_swap_config=block_swap_config,
                encode_tiled=encode_tiled,
                encode_tile_size=(encode_tile_size, encode_tile_size),
                encode_tile_overlap=(encode_tile_overlap, encode_tile_overlap),
                decode_tiled=decode_tiled,
                decode_tile_size=(decode_tile_size, decode_tile_size),
                decode_tile_overlap=(decode_tile_overlap, decode_tile_overlap),
                tile_debug=tile_debug,
                attention_mode=attention_mode,
                torch_compile_args_dit=dit_torch_compile_args,
                torch_compile_args_vae=vae_torch_compile_args
            )

            # Store cache context in ctx for use in generation phases
            ctx['cache_context'] = cache_context

            debug.log_memory_state("After model preparation", show_tensors=False, detailed_tensors=False)
            debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)
            
            # Compute generation info and log start (handles prepending internally)
            image, gen_info = compute_generation_info(
                ctx=ctx,
                images=image,
                resolution=resolution,
                max_resolution=max_resolution,
                batch_size=batch_size,
                uniform_batch_size=uniform_batch_size,
                seed=seed,
                prepend_frames=prepend_frames,
                temporal_overlap=temporal_overlap,
                debug=debug
            )
            
            # Log generation start in consistent format
            log_generation_start(gen_info, debug)
            
            debug.start_timer("generation")
            
            # Phase 1: Encode
            ctx = encode_all_batches(
                runner,
                ctx=ctx,
                images=image,
                debug=debug,
                batch_size=batch_size,
                uniform_batch_size=uniform_batch_size,
                seed=seed,
                progress_callback=progress_callback,
                temporal_overlap=temporal_overlap,
                resolution=resolution,
                max_resolution=max_resolution,
                input_noise_scale=input_noise_scale,
                color_correction=color_correction
            )

            # Phase 2: Upscale
            ctx = upscale_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                seed=seed,
                latent_noise_scale=latent_noise_scale,
                cache_model=dit_cache,
                tbg_steps=tbg_steps
            )

            # Phase 3: Decode
            ctx = decode_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                cache_model=vae_cache
            )

            # Phase 4: Post-processing
            ctx = postprocess_all_batches(
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                color_correction=color_correction,
                prepend_frames=prepend_frames,
                temporal_overlap=temporal_overlap,
                batch_size=batch_size
            )

            sample = ctx['final_video']
            debug.log("", category="none", force=True)

            # Ensure CPU tensor in float32 for maximum ComfyUI compatibility
            if torch.is_tensor(sample):
                if sample.is_cuda or sample.is_mps:
                    sample = sample.cpu()
                if sample.dtype != torch.float32:
                    src_dtype = sample.dtype
                    try:
                        sample = sample.to(torch.float32)
                        debug.log(f"Converted output from {src_dtype} to float32", category="precision")
                    except Exception as e:
                        debug.log(f"Could not convert to float32: {e}. Output is {src_dtype}, compatibility with other nodes not guaranteed", 
                                  level="WARNING", category="precision", force=True)

            debug.log("Upscaling completed successfully!", category="success", force=True)
            debug.end_timer("generation", "Video generation")

            # Final cleanup
            debug.start_timer("final_cleanup")
            cleanup(dit_cache=dit_cache, vae_cache=vae_cache)
            debug.end_timer("final_cleanup", "Final cleanup")

            debug.log_memory_state("After all phases complete", show_tensors=False, detailed_tensors=False)
            
            # Final peak vram summary
            debug.log_peak_memory_summary()

            # Final timing summary
            debug.log("", category="none")
            debug.log("────────────────────────", category="none")
            child_times = {
                "Model preparation": debug.timer_durations.get("model_preparation", 0),
                "Video generation": debug.timer_durations.get("generation", 0),
                "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
            }
            if "phase1_encoding" in debug.timer_durations:
                child_times["  Phase 1: VAE encoding"] = debug.timer_durations.get("phase1_encoding", 0)
            if "phase2_upscaling" in debug.timer_durations:
                child_times["  Phase 2: DiT upscaling"] = debug.timer_durations.get("phase2_upscaling", 0)
            if "phase3_decoding" in debug.timer_durations:
                child_times["  Phase 3: VAE decoding"] = debug.timer_durations.get("phase3_decoding", 0)
            if "phase4_postprocessing" in debug.timer_durations:
                child_times["  Phase 4: Post-processing"] = debug.timer_durations.get("phase4_postprocessing", 0)

            total_execution_time = debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
            
            if total_execution_time > 0:
                fps = gen_info['total_frames'] / total_execution_time
                debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)

            # Print footer
            debug.print_footer()

            debug.clear_history()
            pbar = None
            ctx = None

            # V3-compatible return with optional UI preview
            return io.NodeOutput(sample)
            
        except Exception as e:
            cleanup(dit_cache=dit_cache, vae_cache=vae_cache)
            raise e