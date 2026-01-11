from functools import wraps
from typing import Callable
import torch
import latent_preview
import threading
import weakref
import atexit
import comfy.model_management


# Thread-safe storage for global state
class ThreadSafeState:
    def __init__(self):
        self._lock = threading.RLock()
        self._reset()

    def _reset(self):
        with self._lock:
            self.RESHARPEN_STRENGTH = 0.0
            self.ETA_STRENGTH = 0.0
            self.RESHARPEN_START = 0.0
            self.RESHARPEN_END = 1.0
            self.ETA_START = 0.0
            self.ETA_END = 0.0
            self.LATENT_CACHE = None
            self.ACTIVE_SAMPLINGS = set()

    def set_values(self, resharpen_strength=0.0, eta_strength=0.0,
                   resharpen_start=0.0, resharpen_end=1.0,
                   eta_start=0.0, eta_end=0.0):
        with self._lock:
            self.RESHARPEN_STRENGTH = resharpen_strength
            self.ETA_STRENGTH = eta_strength
            self.RESHARPEN_START = resharpen_start
            self.RESHARPEN_END = resharpen_end
            self.ETA_START = eta_start
            self.ETA_END = eta_end

    def get_values(self):
        with self._lock:
            return (self.RESHARPEN_STRENGTH, self.ETA_STRENGTH,
                    self.RESHARPEN_START, self.RESHARPEN_END,
                    self.ETA_START, self.ETA_END)

    def set_cache(self, cache):
        with self._lock:
            self.LATENT_CACHE = cache

    def get_cache(self):
        with self._lock:
            return self.LATENT_CACHE

    def add_sampling_id(self, sampling_id):
        with self._lock:
            self.ACTIVE_SAMPLINGS.add(sampling_id)

    def remove_sampling_id(self, sampling_id):
        with self._lock:
            self.ACTIVE_SAMPLINGS.discard(sampling_id)

    def has_active_samplings(self):
        with self._lock:
            return len(self.ACTIVE_SAMPLINGS) > 0

    def disable(self):
        with self._lock:
            self._reset()


# Global thread-safe state
_STATE = ThreadSafeState()

# Store the original callback
ORIGINAL_PREP: Callable = latent_preview.prepare_callback


class TBG_DetailEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "Sharpener_Strength": ("FLOAT", {"default": 0.5, "min": -5, "max": 5, "round": 0.01}),
                "S_Start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "round": 0.01}),
                "S_End": ("FLOAT", {"default": 1, "min": 0, "max": 1, "round": 0.01}),
                "ETA_Strength": ("FLOAT", {"default": 0, "min": 0, "max": 1, "round": 0.01}),
                "E_Start": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "round": 0.01}),
                "E_End": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "hook"
    CATEGORY = "TBG/Helpers"

    def hook(self, model, Sharpener_Strength, ETA_Strength, S_Start, E_Start, S_End, E_End):
        if Sharpener_Strength == 0.0 and ETA_Strength == 0.0:
            disable_resharpen()
        else:
            _STATE.set_values(
                resharpen_strength=Sharpener_Strength / -10.0,  # Negative scaling
                eta_strength=ETA_Strength,
                resharpen_start=S_Start,
                resharpen_end=S_End,
                eta_start=E_Start,
                eta_end=E_End
            )
        return (model,)


def disable_resharpen():
    """Disable resharpening and clear state."""
    _STATE.disable()
    #print("TBG Detail_Enhancer disabled and state reset")


def is_step_in_range(step, totalstep, start_percentage, end_percentage):
    start_step = totalstep * start_percentage
    end_step = totalstep * end_percentage
    return start_step <= step <= end_step


def hijack(PREP: Callable) -> Callable:
    """Hijack the latent preview preparation callback conditionally."""

    @wraps(PREP)
    def prep_callback(*args, **kwargs):
        # Generate unique sampling ID
        import uuid
        sampling_id = str(uuid.uuid4())

        # Register this sampling session
        _STATE.add_sampling_id(sampling_id)

        # Get original callback
        original_callback: Callable = PREP(*args, **kwargs)

        # Get current state values
        (resharpen_strength, eta_strength, resharpen_start,
         resharpen_end, eta_start, eta_end) = _STATE.get_values()

        # If both strengths are 0, return unmodified callback
        if resharpen_strength == 0.0 and eta_strength == 0.0:
            _STATE.remove_sampling_id(sampling_id)
            return original_callback

        @torch.inference_mode()
        @wraps(original_callback)
        def hijack_callback(step, x0, x, total_steps):
            try:
                # Check for interruption
                if comfy.model_management.processing_interrupted():
                    #print("Sampling interrupted, cleaning up resharpen state")
                    _STATE.remove_sampling_id(sampling_id)
                    disable_resharpen()
                    return original_callback(step, x0, x, total_steps)

                # Get current cache
                latent_cache = _STATE.get_cache()

                # Always update cache for next iteration, even on first run
                _STATE.set_cache(x.detach().clone())

                # Apply resharpen if we have a previous cache (skip first iteration)
                if latent_cache is not None:
                    if resharpen_strength != 0 and is_step_in_range(step, total_steps, resharpen_start, resharpen_end):
                        delta = x.detach().clone() - latent_cache
                        x += delta * resharpen_strength
                        #print(f"Resharpen applied at step {step}/{total_steps} with strength {resharpen_strength}")

                    if eta_strength > 0 and is_step_in_range(step, total_steps, eta_start, eta_end):
                        noise = torch.randn_like(x) * (eta_strength / 10)
                        x += noise
                        #print(f"ETA noise applied at step {step}/{total_steps} with strength {eta_strength}")

                # Check if this is the last step
                if step >= total_steps - 1:
                    #print(f"Sampling completed, cleaning up (step {step}/{total_steps})")
                    _STATE.remove_sampling_id(sampling_id)
                    # Only disable if no other samplings are active
                    if not _STATE.has_active_samplings():
                        disable_resharpen()

                return original_callback(step, x0, x, total_steps)

            except Exception as e:
                #print(f"Error in resharpen callback: {e}")
                _STATE.remove_sampling_id(sampling_id)
                # Clean up on error
                if not _STATE.has_active_samplings():
                    disable_resharpen()
                return original_callback(step, x0, x, total_steps)

        return hijack_callback

    return prep_callback


# Replace the callback in latent_preview with our hijacked version
latent_preview.prepare_callback = hijack(ORIGINAL_PREP)


# Cleanup function for application exit
def cleanup_on_exit():
    disable_resharpen()


# Register cleanup
atexit.register(cleanup_on_exit)

# Initialize with clean state
disable_resharpen()


class TBGReSharpen:
    def hook(self, latent: torch.Tensor, Sharpener_Strength: float, eta_strength: float):

        if Sharpener_Strength == 0.0:
            disable_resharpen()
        else:
            _STATE.set_values(
                resharpen_strength=Sharpener_Strength*5,  # Negative scaling
                eta_strength=eta_strength,
                resharpen_start=0,
                resharpen_end=0.2,
                eta_start=0,
                eta_end=0.5
            )
            (resharpen_strength, eta_strength, resharpen_start,
             resharpen_end, eta_start, eta_end) = _STATE.get_values()
            #print("resharpen_strength, eta_strength, resharpen_start, resharpen_end, eta_start, eta_end",resharpen_strength, eta_strength, resharpen_start, resharpen_end, eta_start, eta_end)
            TBG_DetailEnhancer.hook(None, None, resharpen_strength, eta_strength, resharpen_start, eta_start, resharpen_end, eta_end)


        return (latent,)
