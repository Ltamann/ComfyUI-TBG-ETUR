from __future__ import annotations

"""
Custom-node-local SDPA guard for Untwisting RoPE.

This module uses ComfyUI's documented/available
``transformer_options["optimized_attention_override"]`` hook. It does not patch
ComfyUI core files and it is a no-op when the selected ComfyUI attention backend
is not ``attention_pytorch`` (for example when SageAttention is enabled).

For Z-Image/Lumina-style additive key-padding masks, it compacts K/V tokens so
SDPA can be called without an attention mask. That avoids the high-VRAM masked
math fallback while also bypassing ComfyUI's Windows CuDNN-prioritized SDPA
wrapper for this custom-node path only.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import comfy.model_management
from comfy.ldm.modules.attention import attention_sub_quad


_TORCH_HAS_GQA_FOR_UNTWIST = getattr(comfy.model_management, "torch_version_numeric", 0) >= (2, 5)

try:
    from torch.nn.attention import SDPBackend as _UntwistSDPBackend, sdpa_kernel as _untwist_sdpa_kernel
    import inspect as _untwist_inspect

    _UNTWIST_SDPA_HAS_SET_PRIORITY = "set_priority" in _untwist_inspect.signature(_untwist_sdpa_kernel).parameters
    _UNTWIST_SDPA_NO_CUDNN_BACKENDS = [
        _UntwistSDPBackend.FLASH_ATTENTION,
        _UntwistSDPBackend.EFFICIENT_ATTENTION,
        _UntwistSDPBackend.MATH,
    ]
except Exception:
    _untwist_sdpa_kernel = None
    _UNTWIST_SDPA_HAS_SET_PRIORITY = False
    _UNTWIST_SDPA_NO_CUDNN_BACKENDS = None


def _scaled_dot_product_attention_no_cudnn(q, k, v, *args, **kwargs):
    """Run PyTorch SDPA without ComfyUI's Windows CuDNN-priority wrapper."""
    if _untwist_sdpa_kernel is None or getattr(q, "device", None).type != "cuda":
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
    if _UNTWIST_SDPA_HAS_SET_PRIORITY:
        with _untwist_sdpa_kernel(_UNTWIST_SDPA_NO_CUDNN_BACKENDS, set_priority=True):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
    with _untwist_sdpa_kernel(_UNTWIST_SDPA_NO_CUDNN_BACKENDS):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


def _prepare_key_padding_mask_for_compaction(mask, b: int, k_tokens: int, device) -> Optional[torch.Tensor]:
    """
    Return a [B, K] boolean valid-key mask when ``mask`` is a pure key-padding
    mask: 0/True for valid tokens and large negative/False for invalid tokens.

    Query-dependent masks or soft logit-bias masks return None and are handled by
    the fallback path.
    """
    if mask is None or not torch.is_tensor(mask):
        return None
    try:
        m = mask.detach()
        if m.ndim == 4:
            # Accept [B,1,1,K] / [1,1,1,K], not query-dependent [B,H,Q,K].
            if int(m.shape[-1]) != int(k_tokens) or int(m.shape[-2]) != 1 or int(m.shape[-3]) != 1:
                return None
            m = m.reshape(int(m.shape[0]), int(k_tokens))
        elif m.ndim == 3:
            # Accept [B,1,K] / [1,1,K].
            if int(m.shape[-1]) != int(k_tokens) or int(m.shape[-2]) != 1:
                return None
            m = m.reshape(int(m.shape[0]), int(k_tokens))
        elif m.ndim == 2:
            if int(m.shape[-1]) != int(k_tokens):
                return None
        elif m.ndim == 1:
            if int(m.shape[0]) != int(k_tokens):
                return None
            m = m.reshape(1, int(k_tokens))
        else:
            return None

        if int(m.shape[0]) == 1 and b > 1:
            m = m.expand(b, -1)
        elif int(m.shape[0]) != b:
            return None

        if m.dtype == torch.bool:
            valid = m.to(device=device, dtype=torch.bool)
        else:
            mf = m.to(device=device, dtype=torch.float32)
            valid = mf >= -0.5
            invalid = ~valid
            # Only handle hard padding masks. Do not compact soft/logit-bias masks.
            if bool(invalid.any().item()) and bool((mf[invalid] > -1000.0).any().item()):
                return None
            if bool((mf[valid].abs() > 1.0e-4).any().item()):
                return None

        # One dense compacted K/V cannot represent different key lengths per batch.
        # Z-Image's patched target/reference streams are B=1, but identical masks
        # are also safe.
        if b > 1 and not bool((valid == valid[:1]).all().item()):
            return None
        return valid
    except Exception:
        return None


def _compact_kv_by_key_padding_mask(k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    try:
        keep = valid[0].to(device=k.device, dtype=torch.bool)
        if int(keep.numel()) != int(k.shape[-2]):
            return k, v, False
        keep_count = int(keep.long().sum().item())
        if keep_count <= 0:
            return k, v, False
        if keep_count == int(k.shape[-2]):
            return k, v, True
        idx = keep.nonzero(as_tuple=False).flatten()
        return k.index_select(-2, idx), v.index_select(-2, idx), True
    except Exception:
        return k, v, False


def _attention_pytorch_compact_mask_no_cudnn(
    q, k, v, heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    valid = _prepare_key_padding_mask_for_compaction(mask, int(b), int(k.shape[-2]), k.device)
    if valid is not None:
        k, v, compact_ok = _compact_kv_by_key_padding_mask(k, v, valid)
        if compact_ok:
            mask = None

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    sdpa_keys = ("scale", "enable_gqa") if _TORCH_HAS_GQA_FOR_UNTWIST else ("scale",)
    sdpa_extra = {k_: v_ for k_, v_ in kwargs.items() if k_ in sdpa_keys}

    out = _scaled_dot_product_attention_no_cudnn(
        q, k, v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        **sdpa_extra,
    )

    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def _attention_backend_override(orig_func: Callable, *args, **kwargs):
    # SageAttention and every non-pytorch backend are left untouched.
    if str(getattr(orig_func, "__name__", "")) != "attention_pytorch":
        return orig_func(*args, **kwargs)
    try:
        return _attention_pytorch_compact_mask_no_cudnn(*args, **kwargs)
    except Exception:
        # Correctness fallback for unusual masks. This is intentionally local to
        # the custom node and should only be hit if mask compaction/SDPA fails.
        return attention_sub_quad(*args, **kwargs)


def install_optimized_attention_override(transformer_options: Dict[str, Any]) -> None:
    """Install the local override into a ComfyUI transformer_options dict."""
    if isinstance(transformer_options, dict):
        transformer_options["optimized_attention_override"] = _attention_backend_override
