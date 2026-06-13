from __future__ import annotations

import torch
import types
from typing import Any
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked

ARCHITECTURE = "zimage"
DISPLAY_NAME = "Z-Image"
CONFIG_KEY = "untwisting_rope"

# Support all Lumina-2 NextDiT variants defined in supported_models.py
SUPPORTED_MODEL_CONFIG_CLASSES = {"ZImage", "ZImagePixelSpace", "Lumina2"}
DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)

def matches_model(model_info: dict[str, Any]) -> bool:
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES

def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True

def find_diffusion_model(model_patcher: Any) -> Any:
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError("Could not find ComfyUI BaseModel.diffusion_model for Z-Image.")

def is_joint_attention(module: Any) -> bool:
    return (
        hasattr(module, "qkv") and hasattr(module, "out")
        and hasattr(module, "q_norm") and hasattr(module, "k_norm")
        and hasattr(module, "n_local_heads") and hasattr(module, "n_local_kv_heads")
        and hasattr(module, "head_dim")
        and callable(getattr(module, "forward", None))
    )

def is_main_layers_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Z-Image uses both layers.N and noise_refiner.N"""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[2] != "attention":
        return False
    if parts[0] not in ("layers", "noise_refiner"):
        return False
    try:
        idx = int(parts[1])
    except Exception:
        return False
    return int(min_layer) <= idx <= int(max_layer)

def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    return {"architecture": ARCHITECTURE}

def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    return is_main_layers_attention_name(name, min_layer, max_layer)

def prepare_reference_conditioning(ref_conditioning: Any, dm: Any, device: Any, dtype: Any, stats: Any, label: str = "", helpers: dict[str, Any] | None = None):
    return ref_conditioning, "not-applicable"

def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean

def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    helpers = helpers or {}

    required_helpers = (
        "patch_context_refiner_mask_modules",
        "patch_patchify_and_embed",
        "build_frequency_scale_vector",
        "apply_qkv_shared_effects",
        "apply_attention_output_shared_effects",
    )
    missing = [name for name in required_helpers if not callable(helpers.get(name))]
    if missing:
        raise RuntimeError(f"Z-Image adapter missing required helper(s): {missing}")

    # Apply context refiner and patchify patches so cfg['ref_real_ranges'] populates correctly.
    helpers["patch_context_refiner_mask_modules"](dm, stats)
    helpers["patch_patchify_and_embed"](dm, stats)

    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    apply_qkv_shared_effects = helpers["apply_qkv_shared_effects"]
    apply_attention_output_shared_effects = helpers["apply_attention_output_shared_effects"]
    
    matched = installed = restored = 0
    patched_names = []

    for name, module in dm.named_modules():
        if not is_main_layers_attention_name(name, 0, 999):
            continue
        if not is_joint_attention(module):
            continue

        matched += 1
        patched_names.append(name)

        if hasattr(module, "_untwist_orig_forward"):
            module.forward = module._untwist_orig_forward
            restored += 1
        else:
            module._untwist_orig_forward = module.forward
            
        original_forward = module._untwist_orig_forward

        def make_forward(orig, module_name):
            def patched_forward(self, x, x_mask, freqs_cis, transformer_options={}):
                cfg = transformer_options.get(CONFIG_KEY) if isinstance(transformer_options, dict) else None
                if not cfg or not cfg.get("enabled"):
                    return orig(x, x_mask, freqs_cis, transformer_options=transformer_options)

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                if target_bsz <= 0:
                    raise RuntimeError(f"Z-Image Untwisting enabled in {module_name}, but cross_batch_target_batch={target_bsz}.")

                if not torch.is_tensor(x) or x.ndim != 3:
                    raise RuntimeError(f"Z-Image Untwisting expected x as [B,S,C] tensor in {module_name}; got {type(x).__name__} with ndim={getattr(x, 'ndim', None)}.")

                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    raise RuntimeError(f"Z-Image Untwisting expected at least target+reference batches in {module_name}; bsz={bsz}, target_bsz={target_bsz}.")

                block_idx = int(transformer_options.get("block_index", -1))
                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and block_idx not in active_blocks:
                    return orig(x, x_mask, freqs_cis, transformer_options=transformer_options)

                # Fix for Z-Image: noise_refiner runs only on image. Main layers run on text+image.
                is_noise_refiner = "noise_refiner" in module_name
                if is_noise_refiner:
                    img_s, img_e = 0, seqlen
                else:
                    ref_ranges = cfg.get("ref_real_ranges", [])
                    if not ref_ranges:
                        raise RuntimeError(f"Z-Image Untwisting enabled in {module_name}, but cfg['ref_real_ranges'] is missing. patchify_and_embed did not populate the image token range.")
                    img_s, img_e = ref_ranges[0]
                        
                img_s = max(0, min(img_s, seqlen))
                img_e = max(img_s, min(img_e, seqlen))

                if img_e <= img_s:
                    raise RuntimeError(f"Z-Image Untwisting has an empty image token range in {module_name}: {(img_s, img_e)} for seqlen={seqlen}.")

                if hasattr(stats, "attn_calls"):
                    stats.attn_calls += 1

                xq, xk, xv = torch.split(
                    self.qkv(x),
                    [
                        self.n_local_heads * self.head_dim,
                        self.n_local_kv_heads * self.head_dim,
                        self.n_local_kv_heads * self.head_dim,
                    ],
                    dim=-1,
                )
                xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
                xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
                xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

                xq = self.q_norm(xq)
                xk = self.k_norm(xk)

                xq, xk, xv = apply_qkv_shared_effects(
                    xq, xk, xv,
                    cfg,
                    target_bsz,
                    module_name,
                    layout="BSHD",
                    token_ranges=[(img_s, img_e)],
                )

                xq, xk = apply_rope(xq, xk, freqs_cis)

                # Untwisting RoPE Frequencies
                progress = float(cfg.get("progress", 0.0))
                high_scale = _lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
                low_scale  = _lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
                beta       = float(cfg.get("beta", 2.0))

                scale_vec = build_frequency_scale_vector(
                    self.head_dim, 
                    cfg.get("axes_dims") or getattr(dm, "axes_dims", []),
                    high_scale, low_scale, beta, 
                    xk.device, xk.dtype,
                    runtime_cfg=cfg
                ).view(1, 1, 1, self.head_dim)

                ref_k = xk[target_bsz:target_bsz*2, img_s:img_e] * scale_vec
                ref_v = xv[target_bsz:target_bsz*2, img_s:img_e]

                n_rep = self.n_local_heads // self.n_local_kv_heads
                def expand_kv(k_tensor, v_tensor):
                    if self.n_local_kv_heads <= 0:
                        raise RuntimeError(f"Z-Image Untwisting invalid KV head count in {module_name}: n_local_kv_heads={self.n_local_kv_heads}.")
                    if self.n_local_heads % self.n_local_kv_heads != 0:
                        raise RuntimeError(f"Z-Image Untwisting cannot expand KV heads in {module_name}: q_heads={self.n_local_heads}, kv_heads={self.n_local_kv_heads}.")
                    n_rep_local = self.n_local_heads // self.n_local_kv_heads
                    if n_rep_local < 1:
                        raise RuntimeError(f"Z-Image Untwisting invalid KV repeat factor in {module_name}: n_rep={n_rep_local}.")
                    k_tensor = k_tensor.unsqueeze(3).repeat(1, 1, 1, n_rep_local, 1).flatten(2, 3)
                    v_tensor = v_tensor.unsqueeze(3).repeat(1, 1, 1, n_rep_local, 1).flatten(2, 3)
                    return k_tensor, v_tensor

                # TARGET STREAM
                k_t_full = torch.cat([xk[:target_bsz], ref_k], dim=1)
                v_t_full = torch.cat([xv[:target_bsz], ref_v], dim=1)
                k_t_full, v_t_full = expand_kv(k_t_full, v_t_full)
                
                xq_t = xq[:target_bsz]
                
                mask_t = x_mask[:target_bsz] if x_mask is not None else None
                if mask_t is not None:
                    ref_len = img_e - img_s
                    if mask_t.ndim < 2:
                        raise RuntimeError(f"Z-Image Untwisting cannot append reference mask in {module_name}: mask ndim={mask_t.ndim}, expected >=2.")
                    pad = torch.zeros((*mask_t.shape[:-1], ref_len), device=mask_t.device, dtype=mask_t.dtype)
                    mask_t = torch.cat([mask_t, pad], dim=-1)

                out_t = optimized_attention_masked(
                    xq_t.movedim(1, 2), k_t_full.movedim(1, 2), v_t_full.movedim(1, 2),
                    self.n_local_heads, mask_t, skip_reshape=True, transformer_options=transformer_options
                )

                # REFERENCE STREAM
                xq_r = xq[target_bsz:target_bsz*2]
                xk_r, xv_r = expand_kv(xk[target_bsz:target_bsz*2], xv[target_bsz:target_bsz*2])
                mask_r = x_mask[target_bsz:target_bsz*2] if x_mask is not None else None
                
                out_r = optimized_attention_masked(
                    xq_r.movedim(1, 2), xk_r.movedim(1, 2), xv_r.movedim(1, 2),
                    self.n_local_heads, mask_r, skip_reshape=True, transformer_options=transformer_options
                )

                out_t, out_r = apply_attention_output_shared_effects(
                    out_t, out_r,
                    cfg,
                    target_bsz,
                    module_name,
                    layout="BSD",
                    token_ranges=[(img_s, img_e)],
                )

                outs = [out_t, out_r]

                # EXTRA BATCHES (uncond)
                if bsz > target_bsz * 2:
                    xq_e = xq[target_bsz*2:]
                    xk_e, xv_e = expand_kv(xk[target_bsz*2:], xv[target_bsz*2:])
                    mask_e = x_mask[target_bsz*2:] if x_mask is not None else None
                    out_e = optimized_attention_masked(
                        xq_e.movedim(1, 2), xk_e.movedim(1, 2), xv_e.movedim(1, 2),
                        self.n_local_heads, mask_e, skip_reshape=True, transformer_options=transformer_options
                    )
                    outs.append(out_e)

                final_out = torch.cat(outs, dim=0)
                return self.out(final_out)

            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        installed += 1

    if installed <= 0:
        raise RuntimeError("Z-Image adapter patch failed: no compatible attention modules were installed.")
    return matched, installed, restored, patched_names

def uses_reference_branch_kv() -> bool:
    return False
