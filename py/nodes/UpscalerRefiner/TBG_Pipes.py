# Standard library imports
import glob
import os
from types import SimpleNamespace

import PIL
# Third-party imports
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 592515344
from PIL import Image
from aiohttp import web

# Application-specific imports
import folder_paths

from server import PromptServer

#from ...vendor.rgthree_comfy.py.image_comparer import RgthreeImageComparer as ImageComparer
from ...vendor.ComfyUI_MaraScott_Nodes.py.inc.lib.cache import MS_Cache
from ...vendor.ComfyUI_MaraScott_Nodes.py.utils.constants import get_category
from ...utils.log import log

# Relative imports - local module
from .inc.prompt import Node as NodePrompt

# Relative imports - root constants
from .... import root_dir


class TBG_ControlNetPipeline:
    NAME = "TBG Tile ControlNet Pipe"
    PREPROCESSOR_OPTIONS = {
        'None': "None",
        'DepthAnythingV2': "DepthAnythingV2",
        'Canny Edge': "Canny Edge",
        'Canny': "Canny",
    }

    KONTEXT_OPTIONS = {
        'NONE': "NONE",
        'Stitched': "Stitched",
        'Chained': "Chained",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("CONTROL_NET",),  # ControlNet model
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),  # Strength of ControlNet
                "start": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),  # Start influence
                "end": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),  # End influence
                "canny_low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),  # Low threshold
                "canny_high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),  # High threshold
                "preprocessor": (list(cls.PREPROCESSOR_OPTIONS.keys()),),  # Now required
                "patch_for_Flux_Kontext": (list(cls.KONTEXT_OPTIONS.keys()), {"label": "patch_for_Flux_Kontext", "default": 'NONE', }),  # Now required

            },
            "optional": {
                "Controlnet_Pipe": ("Controlnet_Pipe",),  # Incoming pipeline (empty or existing list)
                "custom_controlnet_image": ("IMAGE", {"label": "custom_controlnet_image"}),
            }
        }


    RETURN_TYPES = ("Controlnet_Pipe", "STRING")  # Added STRING for debugging
    RETURN_NAMES = ("Controlnet_Pipe", "INFO")  # Added STRING for debugging
    FUNCTION = "update_pipe"
    CATEGORY = "ControlNet"

    def update_pipe(self, controlnet, strength, start, end, canny_low_threshold, canny_high_threshold, preprocessor, patch_for_Flux_Kontext, custom_controlnet_image=None, Controlnet_Pipe=None):
        # Ensure pipe is a list
        if Controlnet_Pipe is None or not isinstance(Controlnet_Pipe, list):
            Controlnet_Pipe = []

        # Append new ControlNet settings
        Controlnet_Pipe.append({
            "controlnet": controlnet,
            "preprocessor": preprocessor,  # Now always required
            "strength": strength,
            "start": start,
            "end": end,
            "canny_low_threshold": canny_low_threshold,
            "canny_high_threshold": canny_high_threshold,
            "noise_image": custom_controlnet_image,
            "patch_for_Flux_Kontext" : patch_for_Flux_Kontext,
        })

        # Convert pipe to string for debugging
        pipe_str = str(Controlnet_Pipe)

        return Controlnet_Pipe, pipe_str


class TBG_enrichment_pipe:
    NAME = "TBG Tile Enrichment Pipe"

    INNERUPSCALE_METHODS = [
        'none',
        'finer details',
        'finer details + grain removal',
    ]

    UPSCALE_METHODS = [
        "area",
        "bicubic",
        "bilinear",
        "bislerp",
        "lanczos",
        "nearest-exact",
        "with model",
    ]

    COLOR_MATCH_METHODS = [
        'none',
        'mkl',
        'hm',
        'reinhard',
        'mvgd',
        'hm-mvgd-hm',
        'hm-mkl-hm',
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detail_daemon_active": ("BOOLEAN", {"label": "Use Detail Daemon", "default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "detail_amount": ("FLOAT", {"default": 1.17, "min": -5.0, "max": 5.0, "step": 0.01, "label": "Detail Amount"}),
                "detail_daemon_start": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Start Detail"}),
                "detail_daemon_end": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "label": "End Detail"}),
                "detail_daemon_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_daemon_exponent": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.05}),
                "detail_daemon_start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "detail_daemon_end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "detail_daemon_fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_daemon_smooth": ("BOOLEAN", {"default": True}),
                "detail_daemon_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "latentupscale": ("BOOLEAN", {"label": "latentupscale", "default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "latentupscale_noise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "latentupscale_steps": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "latentupscale_denoise": ("FLOAT", {"default": 0.56, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),      
                "SplitSteps": ("BOOLEAN", {"label": "latentupscale", "default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "SplitSteps_noise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "SplitSteps_steps": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "SplitStepsSigmas": ("BOOLEAN", {"label": "SplitStepsSigmas", "default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "SplitStepsMultiplyer": ("FLOAT", {"SplitStepsMultiplyer": 0.7, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),

                "SplitStepsStart": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "SplitStepsEnd": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),

                "Sampler_side_noise_injection": ("FLOAT", {"label": "ETA", "default": 0.00, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "RF_inversion": ("FLOAT", {"label": "RF_inversion", "default": 0, "min": 0, "max": 1, "step": 0.1, "round": 0.1}),
                "tile_upscale_plus": (cls.INNERUPSCALE_METHODS, {"label": "tile_upscale_plus", "default": 'none'}),
                "upscaler_method_inpainting": (cls.UPSCALE_METHODS, {"label": "Upscale Method", "default": 'lanczos'}),
                "upscale_model_inpainting": (folder_paths.get_filename_list("upscale_models"),
                                             {"label": "Upscale Model"}),
                "upscale_tiles_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 2, "step": 0.1,
                                               "round": 0.1}),
                "upscale_segments_by": ("FLOAT", {"label": "upscale_by", "default": 1.0, "min": 0.5, "max": 2, "step": 0.1,
                                                  "round": 0.1}),

            },
            "optional": {
                "SplitStepsSigmasCurve": ("SIGMAS", {"label": "SplitStepsSigmasCurve"}),
            }
        }


    RETURN_TYPES = ("Enrichment_Pipe", "STRING")  # Added STRING for debugging
    RETURN_NAMES = ("Enrichment_Pipe", "INFO")
    FUNCTION = "update_pipe"
    CATEGORY = "enrichment"

    def update_pipe(self, detail_daemon_active, detail_amount, detail_daemon_start, detail_daemon_end, detail_daemon_bias,
                    detail_daemon_exponent, detail_daemon_start_offset, detail_daemon_end_offset,
                    detail_daemon_fade, detail_daemon_smooth, detail_daemon_cfg_scale,
                    latentupscale, latentupscale_noise, SplitSteps, SplitSteps_noise, SplitSteps_steps, latentupscale_steps, latentupscale_denoise,
                    SplitStepsSigmas, SplitStepsMultiplyer, SplitStepsSigmasCurve, SplitStepsStart, SplitStepsEnd,Sampler_side_noise_injection, RF_inversion,tile_upscale_plus,upscaler_method_inpainting,upscale_model_inpainting, upscale_tiles_by,upscale_segments_by
                    ):



        Enrichment_Pipe = []

        # Append new ControlNet settings
        Enrichment_Pipe.append({
            "detail_daemon_active": detail_daemon_active,
            "detail_amount": detail_amount,
            "detail_daemon_start": detail_daemon_start,
            "detail_daemon_end": detail_daemon_end,
            "detail_daemon_bias": detail_daemon_bias,
            "detail_daemon_exponent": detail_daemon_exponent,
            "detail_daemon_start_offset": detail_daemon_start_offset,
            "detail_daemon_end_offset": detail_daemon_end_offset,
            "detail_daemon_fade": detail_daemon_fade,
            "detail_daemon_smooth": detail_daemon_smooth,
            "detail_daemon_cfg_scale": detail_daemon_cfg_scale,
            "latentupscale": latentupscale,
            "latentupscale_noise": latentupscale_noise,
            "latentupscale_steps": latentupscale_steps,
            "latentupscale_denoise": latentupscale_denoise,
            "SplitSteps": SplitSteps,
            "SplitSteps_noise": SplitSteps_noise,
            "SplitSteps_steps": SplitSteps_steps,
            "SplitStepsSigmas": SplitStepsSigmas,
            "SplitStepsMultiplyer": SplitStepsMultiplyer,
            "SplitStepsSigmasCurve": SplitStepsSigmasCurve,
            "SplitStepsStart": SplitStepsStart,
            "SplitStepsEnd": SplitStepsEnd,
            "eta": Sampler_side_noise_injection,
            "RF_inversion": RF_inversion,
            "tile_upscale_plus": tile_upscale_plus,
            "upscaler_method_inpainting": upscaler_method_inpainting,
            "upscale_model_inpainting": upscale_model_inpainting,
            "upscale_tiles_by": upscale_tiles_by,
            "upscale_segments_by": upscale_segments_by,


        })
        pipe_str = str(Enrichment_Pipe)


        return Enrichment_Pipe, pipe_str  # Return both the structure and the string version






class TBG_TilePrompter_v1():
    NAME = "TBG Tile Prompt Pipe"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{    
                "Tile_Prompt_Pipe": ("Tile_Prompt_Pipe", {"label": "Tile Prompt Pipe" }),
            },
            "optional": {
                "requeue": ("INT", { "label": "requeue (automatic or manual)", "default": 0, "min": 0, "max": 99999999999, "step": 1}),                
                **NodePrompt.ENTRIES,
            }
        }

    RETURN_TYPES = ("TILE_Prompt_PIPE_OUT",)
    RETURN_NAMES = ("Tile_Prompt_Pipe",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True
    CATEGORY = get_category("Upscaling")
    DESCRIPTION = "A \"Tile Prompt Editor\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
                        
        input_prompts, input_tiles, segment_tiles = kwargs.get('Tile_Prompt_Pipe', (None, None))
        input_denoises = ('', ) * len(input_prompts)

        self.init(**kwargs)
        
        log("TBG (PromptEditor) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        _input_prompts = MS_Cache.get(self.CACHE.prompt, input_prompts)
        _input_prompts_edited = MS_Cache.get(self.CACHE.prompt_edited, input_prompts)
        _input_denoises = MS_Cache.get(self.CACHE.denoise, input_denoises)
        _input_denoises_edited = MS_Cache.get(self.CACHE.denoise_edited, input_denoises)


        refresh = False
        
        if not MS_Cache.isset(self.CACHE.denoise):
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, _input_denoises)
        if not MS_Cache.isset(self.CACHE.prompt) or _input_prompts != input_prompts:
            _input_prompts = input_prompts
            MS_Cache.set(self.CACHE.prompt, _input_prompts)
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, input_denoises)
            refresh = True

        if not MS_Cache.isset(self.CACHE.denoise_edited) or refresh:
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        if not MS_Cache.isset(self.CACHE.prompt_edited) or refresh:
            _input_prompts_edited = input_prompts
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        elif len(_input_prompts_edited) != len(_input_prompts):
            _input_prompts_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_prompts_edited, input_prompts)]
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_denoises_edited, input_denoises)]
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)

        if _input_denoises_edited != _input_denoises:
            input_denoises = _input_denoises_edited
        if _input_prompts_edited != _input_prompts:
            input_prompts = _input_prompts_edited

        output_prompts_js = input_prompts
        input_prompts_js = _input_prompts
        output_prompts = output_prompts_js
        output_denoises_js = input_denoises
        input_denoises_js = _input_denoises
        output_denoises = output_denoises_js

        results = list()
        filename_prefix = "TBG" + "_temp_" + "tilePrompter" + "_id_" + self.INFO.id # #TBG_temp_tilePrompter_id_1913_00000.png
        tbg_temp_dir = os.path.join(folder_paths.get_temp_directory(), "TBG")
        __TBG_TEMP__ = tbg_temp_dir
        search_pattern = os.path.join(__TBG_TEMP__, filename_prefix + '*')

        
        files_to_delete = glob.glob(search_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # log(f"Deleted: {file_path}", None, None, f"Node {self.INFO.id} - SUCCESS")
            except Exception as e:
                log(f"Error deleting {file_path}: {e}", None, None, "Node {self.INFO.id} - ERROR")    

        flat_tiles = [tile[0] for tile in input_tiles]
        for index, torchtile in enumerate(flat_tiles):
           
            torchtile = torchtile.squeeze(0) #@ from Torch[1, 3, H, W] to [3, H, W] squeeze the Batch part
            tile = torchtile.squeeze(0) #@ from Torch[3, H, W] to [H, W] squeeze the Batch part


            full_output_folder, filename, counter, subfolder, subfolder_filename_prefix = folder_paths.get_save_image_path(f"TBG/{filename_prefix}", self.output_dir, tile.shape[1], tile.shape[0])
            file = f"{filename}_{index:05}.png" #TBG_temp_tilePrompter_id_1913_00000.png
            file_path = os.path.join(full_output_folder, file)
            
            if not os.path.exists(file_path):
                i = 255. * tile.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)                

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })
            counter += 1
       
        
        # Add segments to Tiles
        if segment_tiles: 
            flat_tiles = [tile[0] for tile in segment_tiles]


            for index, torchtile in enumerate(flat_tiles):
                index = index+len(input_tiles)
                torchtile = torchtile.squeeze(0) #@ from Torch[1, 3, H, W] to [3, H, W] squeeze the Batch part
                tile = torchtile.squeeze(0) #@ from Torch[3, H, W] to [H, W] squeeze the Batch part


                full_output_folder, filename, counter, subfolder, subfolder_filename_prefix = folder_paths.get_save_image_path(f"TBG/{filename_prefix}", self.output_dir, tile.shape[1], tile.shape[0])
                file = f"{filename}_{index:05}.png" #TBG_temp_tilePrompter_id_1913_00000.png
                file_path = os.path.join(full_output_folder, file)
                
                if not os.path.exists(file_path):
                    i = 255. * tile.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)                

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "temp"
                })
                counter += 1
            

        log("TBG (PromptEditor) is done", None, None, f"Node {self.INFO.id}")
                    
        return {"ui": {
            "prompts_out": output_prompts_js, 
            "prompts_in": input_prompts_js , 
            "denoises_out": output_denoises_js, 
            "denoises_in": input_denoises_js , 
            "tiles": results,
        }, "result": ((output_prompts, output_denoises),)}

    @classmethod
    def init(self, **kwargs):
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', 0),
        )
        self.CACHE = SimpleNamespace(
            prompt = f'input_prompts_{self.INFO.id}',
            prompt_edited = None,
            denoise = f'input_denoises_{self.INFO.id}',
            denoise_edited = None,
        )
        self.CACHE.prompt_edited = f'{self.CACHE.prompt}_edited'
        self.CACHE.denoise_edited = f'{self.CACHE.denoise}_edited'
        
        self.output_dir = folder_paths.get_temp_directory()
   
        log( self.output_dir, None, None, f"Node {self.INFO.id}")

        #A:\SD\ComfyUI\ComfyUI_windows_portable\ComfyUI\temp

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = MS_Cache.get(cache_name, [])
    return web.json_response({ "prompts_in": input_prompts })
    
@PromptServer.instance.routes.get("/TBG/McBoaty/v5/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = MS_Cache.get(cache_name, [])
    return web.json_response({ "denoises_in": input_denoises })
    
@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_prompt")
async def set_prompt(request):
    prompt = request.query.get("prompt", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    # clientId = request.query.get("clientId", None)
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_prompts = MS_Cache.get(cache_name, [])
    _input_prompts_edited = MS_Cache.get(cache_name_edited, _input_prompts)
    if _input_prompts_edited and index < len(_input_prompts_edited):
        _input_prompts_edited_list = list(_input_prompts_edited)
        _input_prompts_edited_list[index] = prompt
        _input_prompts_edited = tuple(_input_prompts_edited_list)
        MS_Cache.set(cache_name_edited, _input_prompts_edited)
    return web.json_response(f"Tile {index} prompt has been updated :{prompt}")
#http://localhost:8188/TBG/McBoaty/v5/set_prompt?prompt=Hello&index=0&node=123

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/set_denoise")
async def set_denoise(request):
    denoise = request.query.get("denoise", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    # clientId = request.query.get("clientId", None)
    cache_name = f'input_denoises_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_denoises = MS_Cache.get(cache_name, [])
    _input_denoises_edited = MS_Cache.get(cache_name_edited, _input_denoises)
    if _input_denoises_edited and index < len(_input_denoises_edited):
        _input_denoises_edited_list = list(_input_denoises_edited)
        _input_denoises_edited_list[index] = denoise
        _input_denoises_edited = tuple(_input_denoises_edited_list)
        MS_Cache.set(cache_name_edited, _input_denoises_edited)
    return web.json_response(f"Tile {index} denoise has been updated: {denoise}")
    #http://localhost:8188/TBG/McBoaty/v5/set_denoise?denoise=0.5&index=0&node=123

@PromptServer.instance.routes.get("/TBG/McBoaty/v5/tile_prompt")
async def tile_prompt(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = os.path.join(root_dir, type)
    image_path = os.path.abspath(os.path.join(
        target_dir, 
        request.query.get("subfolder", ""), 
        request.query["filename"]
    ))

    log(target_dir, None, None, f"Node {self.INFO.id}")
    log(image_path, None, None, f"Node {self.INFO.id}")

    c = os.path.commonpath((image_path, target_dir))
    if c != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    return web.json_response(f"here is the prompt \n{image_path}")
    #http://localhost:8188/TBG/McBoaty/v5/tile_prompt?filename=example.png&type=output
    #IMAGE_DIR = r"A:\\SD\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI\temp\\TBG"
    #CACHE_DIR = r"A:\\SD\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI\temp\\TBG\\cache"

