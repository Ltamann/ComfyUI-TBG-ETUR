import server
from typing import Callable
import execution
import inspect
from .tbgresharpen import TBG_DetailEnhancer, disable_resharpen
from functools import wraps
from typing import Optional
import comfy.model_management

def version_tuple(v):
    return tuple(map(int, v.split(".")))

def comfyui_version_is_at_least(min_version):
    current_version = getattr(server, "__version__", None)
    if current_version is None:
        return False
    return version_tuple(current_version) >= version_tuple(min_version)

def find_node(prompt: dict) -> bool:
    found = any(v.get("class_type") == "TBG_DetailEnhancer" for v in prompt.values())
    #print(f"TBG_DetailEnhancer node found: {found}")
    return found

# Hook into interrupt processing to clean up state
original_interrupt_processing = comfy.model_management.interrupt_processing

def enhanced_interrupt_processing(value=True):
    if value:  # If we're setting interrupt to True
        #print("Processing interrupted, disabling resharpen")
        disable_resharpen()
    return original_interrupt_processing(value)

comfy.model_management.interrupt_processing = enhanced_interrupt_processing

current_version = getattr(server, "__version__", None)

if comfyui_version_is_at_least("0.3.48"):
    #print("ComfyUI version >= 0.3.48 detected")
    original_validate: Callable = execution.validate_prompt

    @wraps(original_validate)
    async def hijack_validate(prompt_id, prompt: dict, partial_execution_list: Optional[list[str]] = None):
        #print("Inside hijack_validate", flush=True)
        try:
            if not find_node(prompt):
                #print("TBG_DetailEnhancer node not found, disabling resharpen", flush=True)
                disable_resharpen()
            return await original_validate(prompt_id, prompt, partial_execution_list)
        except Exception as e:
            #print(f"Error in validate hijack: {e}")
            disable_resharpen()
            raise

    execution.validate_prompt = hijack_validate

elif current_version == "0.3.47":
    original_validate: Callable = execution.validate_prompt

    @wraps(original_validate)
    async def hijack_validate(prompt_id, prompt: dict):
        try:
            if not find_node(prompt):
                disable_resharpen()
            return await original_validate(prompt_id, prompt)
        except Exception as e:
            #print(f"Error in validate hijack: {e}")
            disable_resharpen()
            raise

    execution.validate_prompt = hijack_validate

else:
    original_validate: Callable = execution.validate_prompt

    @wraps(original_validate)
    def hijack_validate(prompt: dict) -> bool:
        try:
            if not find_node(prompt):
                disable_resharpen()
            return original_validate(prompt)
        except Exception as e:
            #print(f"Error in validate hijack: {e}")
            disable_resharpen()
            raise

    execution.validate_prompt = hijack_validate
