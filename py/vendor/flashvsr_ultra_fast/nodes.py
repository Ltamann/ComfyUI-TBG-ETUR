#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,gc
import math
import torch
import folder_paths
import comfy.utils

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import snapshot_download
from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .src.models.TCDecoder import build_tcdecoder
from .src.models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
from .src.models import wan_video_dit
from ....TBG.SERVERS.COMFYUI_server import register_main_class

device_choices = get_device_list()

def log(message:str, message_type:str='normal'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(folder_paths.models_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    # TBG for single images

    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)

    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(model, mode, device, dtype, alt_vae="none"):
    model_downlod(model_name="JunhaoZhuang/"+model)
    model_path = os.path.join(folder_paths.models_dir, model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"')
    if alt_vae != "none":
        vae_path = folder_paths.get_full_path_or_raise("vae", alt_vae)
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"{alt_vae}" does not exist!')
    else:
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!\nPlease save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
    
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    pipe.offload_model()

    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            val = next(self.iterable)
            if self.pbar:
                self.pbar.update(1)
            return val
        except StopIteration:
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

def flashvsr(pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload,num_inference_steps):
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype

    add = next_8n5(frames.shape[0]) - frames.shape[0]
    padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    _frames = torch.cat([frames, padding_frames], dim=0)
        
    if tiled_dit:
        N, H, W, C = _frames.shape
        
        final_output_canvas = torch.zeros(
            (N, H * scale, W * scale, C), 
            dtype=torch.float16, 
            device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []
        
        for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles")):
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})", message_type='info')
            input_tile = _frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ_tile = LQ_tile.to(_device)
                
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=num_inference_steps, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, force_offload=force_offload
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("[FlashVSR] Preparing frames...")
        LQ, th, tw, F = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        if not isinstance(pipe, FlashVSRTinyLongPipeline):
            LQ = LQ.to(_device)
        log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type='info')
        
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            progress_bar_cmd=cqdm, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit, force_offload=force_offload
        )
        
        final_output = tensor2video(video).to('cpu')
        
        del video, LQ
        clean_vram()
        
    log("[FlashVSR] Done.", message_type='info')
    if frames.shape[0] == 1:
        final_output = final_output.to(_device)
        stacked_image_tensor = torch.median(final_output, dim=0).values.unsqueeze(0).float().to('cpu')
        del final_output
        clean_vram()
        return stacked_image_tensor
    
    return final_output[:frames.shape[0], :, :, :]


 # TBG for single images
def pipe_line(model, mode, alt_vae, device):
    _device = device
    if device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
    if _device == "auto" or _device not in device_choices:
        raise RuntimeError("No devices found to run FlashVSR!")

    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
    """
    if attention_mode == "sparse_sage_attention":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
    """

    try:
        import sageattention
        print( "sparse_sage_attention")
        wan_video_dit.USE_BLOCK_ATTN = False
    except Exception:
        print("USE_BLOCK_ATTN")
        wan_video_dit.USE_BLOCK_ATTN = True

    dtype = select_dtype() # tbg overwrite auto

    pipe = init_pipeline(model, mode, _device, dtype, alt_vae=alt_vae)
    return pipe
 # TBG for single images
def select_dtype():
    if not torch.cuda.is_available():
        return torch.float32

    major, minor = torch.cuda.get_device_capability()

    # Ampere (8.0) and newer support bf16 well
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    # Volta / Turing / Ampere fallback
    if major >= 7:
        return torch.float16

    # Old GPUs
    return torch.float32
 # TBG for single images


def refine_tile(model, mode, tile_frames, scale, device="auto",
                tiled_vae=True, tiled_dit=False, tile_size=1024, tile_overlap=24,
                sparse_ratio=2.0, kv_ratio=3.0, local_range=11, seed=42,
                unload_dit=False, force_offload=True, num_inference_steps = 1, **kwargs):
    """
    Refine a single IMAGE tile with FlashVSR - mirrors original flashvsr() exactly.
    """
    # --- dtype handling -------------------------------------------------


    scale = int(scale)

    # --- basic input validation -----------------------------------------
    if tile_frames is None or tile_frames.numel() == 0:
        raise ValueError("refine_tile: tile_frames is empty or None")

    if tile_frames.dim() == 3:
        tile_frames = tile_frames.unsqueeze(0)

    N0, H, W, C = tile_frames.shape
    if C != 3:
        raise ValueError(f"refine_tile: expected 3 channels, got {C}")

    pipe = pipe_line(model, mode, "none", device)

      # --- Call original flashvsr logic ----------------------------------
    # Instead of reimplementing, just call the original function!
    return flashvsr(
        pipe=pipe,
        frames=tile_frames,
        scale=scale,
        color_fix=True,
        tiled_vae=tiled_vae,
        tiled_dit=tiled_dit,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        unload_dit=unload_dit,
        sparse_ratio=sparse_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        seed=seed,
        force_offload=force_offload,
        num_inference_steps=num_inference_steps
    )[:N0, :, :, :]  # Remove padding frames

@register_main_class
class flashvr():
    @classmethod
    def refine_tiles(self, model, mode , tile_frames, scale, device="auto",
                tiled_vae=True, tiled_dit=False, tile_size=1024, tile_overlap=24,
                sparse_ratio=2.0, kv_ratio=3.0, local_range=11, seed=42,
                unload_dit=False, force_offload=True, num_inference_steps=1, **kwargs):
        return refine_tile(model, mode, tile_frames, scale, device,
                    tiled_vae, tiled_dit, tile_size, tile_overlap,
                    sparse_ratio, kv_ratio, local_range, seed,
                    unload_dit, force_offload, num_inference_steps, **kwargs)