from __future__ import annotations

import math
import types
from typing import Any, Iterable, Optional, Tuple

import torch
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention_masked


# ---------------------------------------------------------------------------
# Adapter identity
# ---------------------------------------------------------------------------

ARCHITECTURE = "qwen_image"
DISPLAY_NAME = "Qwen-Image"
CONFIG_KEY = "untwisting_rope"

# ComfyUI supported_models class names. Keep this metadata-driven so the
# adapter does not accidentally claim unrelated DiT/MMDiT architectures.
SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {
    "QwenImage",
    "QwenImageEdit",
}

DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


# ---------------------------------------------------------------------------
# Generic lookup / coercion helpers
# ---------------------------------------------------------------------------

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


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default

def _sequence_of_ints(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        i = _safe_int(item)
        if i is None:
            return []
        out.append(i)
    return out

def _repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if int(x.shape[0]) == int(batch):
        return x
    try:
        import comfy.utils
        if hasattr(comfy.utils, "repeat_to_batch_size"):
            return comfy.utils.repeat_to_batch_size(x, int(batch))
    except Exception:
        pass
    reps = math.ceil(int(batch) / max(1, int(x.shape[0])))
    return x.repeat((reps,) + (1,) * (x.ndim - 1))[: int(batch)]


def _pad_or_truncate_tokens(x: torch.Tensor, tokens: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    tokens = int(tokens)
    cur = int(x.shape[-1] if x.ndim == 2 else x.shape[1])
    if cur == tokens:
        return x
    if x.ndim == 2:
        if cur > tokens:
            return x[:, :tokens]
        pad = torch.zeros((x.shape[0], tokens - cur), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    if cur > tokens:
        return x[:, :tokens, ...]
    pad_shape = (x.shape[0], tokens - cur, *x.shape[2:])
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

# ---------------------------------------------------------------------------
# Required adapter hooks
# ---------------------------------------------------------------------------

def matches_model(model_info: dict[str, Any]) -> bool:
    """Select Qwen-Image only from ComfyUI's explicit MODEL metadata."""
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def is_model_identity(model_info: dict[str, Any]) -> bool:
    """Backward-compatible alias for older callers."""
    return matches_model(model_info)


def find_diffusion_model(model_patcher: Any) -> Any:
    """Return ComfyUI BaseModel.diffusion_model after metadata selected this adapter."""
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError(f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}.")


# ---------------------------------------------------------------------------
# Qwen-Image metadata helpers
# ---------------------------------------------------------------------------

def _first_transformer_block(dm: Any) -> Any:
    blocks = getattr(dm, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError(f"{DISPLAY_NAME} metadata lookup failed: dm.transformer_blocks is missing.")
    try:
        return blocks[0]
    except Exception as exc:
        raise RuntimeError(f"{DISPLAY_NAME} metadata lookup failed: dm.transformer_blocks[0] is unavailable.") from exc


def _first_attention(dm: Any) -> Any:
    block = _first_transformer_block(dm)
    attn = getattr(block, "attn", None)
    if attn is None:
        raise RuntimeError(f"{DISPLAY_NAME} metadata lookup failed: transformer_blocks[0].attn is missing.")
    return attn


def head_dim_from_dm(dm: Any | None) -> int:
    if dm is None:
        raise RuntimeError(f"{DISPLAY_NAME} head_dim lookup failed: diffusion module is None.")
    attn = _first_attention(dm)
    head_dim = _safe_int(getattr(attn, "dim_head", None))
    if head_dim is None:
        head_dim = _safe_int(getattr(_first_transformer_block(dm), "attention_head_dim", None))
    if head_dim is None or head_dim <= 0:
        raise RuntimeError(f"{DISPLAY_NAME} head_dim lookup failed: invalid dim_head={head_dim!r}.")
    return int(head_dim)


def num_heads_from_dm(dm: Any | None) -> int:
    if dm is None:
        raise RuntimeError(f"{DISPLAY_NAME} num_heads lookup failed: diffusion module is None.")
    attn = _first_attention(dm)
    heads = _safe_int(getattr(attn, "heads", None))
    if heads is None:
        heads = _safe_int(getattr(_first_transformer_block(dm), "num_attention_heads", None))
    if heads is None or heads <= 0:
        raise RuntimeError(f"{DISPLAY_NAME} num_heads lookup failed: invalid heads={heads!r}.")
    return int(heads)


def axes_dims_from_dm(dm: Any | None) -> list[int]:
    """Read Qwen-Image RoPE axes from the selected diffusion module when possible."""
    if dm is None:
        raise RuntimeError(f"{DISPLAY_NAME} axes_dims lookup failed: diffusion module is None.")

    # ComfyUI's QwenImageTransformer2DModel builds EmbedND(..., axes_dim=list(axes_dims_rope)).
    # Different ComfyUI revisions may expose this on the embedder with slightly different names.
    for root in (dm, getattr(dm, "pe_embedder", None), getattr(dm, "params", None)):
        if root is None:
            continue
        for attr in ("axes_dims_rope", "axes_dim", "axes_dims", "axes_dim_rope"):
            axes = _sequence_of_ints(getattr(root, attr, None))
            if axes and sum(axes) == head_dim_from_dm(dm):
                return axes

    # Qwen-Image default in the provided ComfyUI model is (16, 56, 56) for head_dim=128.
    # If a future checkpoint changes head_dim and the embedder does not expose axes, fall back
    # to a single axis so the shared scale builder remains shape-correct instead of guessing.
    hd = head_dim_from_dm(dm)
    if hd == 128:
        return [16, 56, 56]
    return [hd]


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}
    cfg["head_dim"] = head_dim_from_dm(dm)
    cfg["axes_dims"] = axes_dims_from_dm(dm)
    return cfg


# ---------------------------------------------------------------------------
# Patch target predicates
# ---------------------------------------------------------------------------

def _index_in_range(parts: list[str], min_layer: int, max_layer: int) -> bool:
    if len(parts) < 2:
        return False
    idx = _safe_int(parts[1])
    if idx is None:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def is_qwen_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Qwen-Image attention modules are named transformer_blocks.N.attn."""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[0] != "transformer_blocks" or parts[2] != "attn":
        return False
    return _index_in_range(parts, min_layer, max_layer)


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    return is_qwen_attention_name(name, min_layer, max_layer)


def block_index_from_name(name: str) -> int:
    parts = str(name).split(".")
    if len(parts) >= 2 and parts[0] == "transformer_blocks":
        idx = _safe_int(parts[1], -1)
        return -1 if idx is None else int(idx)
    return -1


def is_qwen_attention_module(module: Any) -> bool:
    required_attrs = (
        "to_q", "to_k", "to_v",
        "add_q_proj", "add_k_proj", "add_v_proj",
        "norm_q", "norm_k", "norm_added_q", "norm_added_k",
        "to_out", "to_add_out",
        "heads", "dim_head", "forward",
    )
    return all(hasattr(module, attr) for attr in required_attrs) and callable(getattr(module, "forward", None))


def is_joint_attention(module: Any) -> bool:
    return is_qwen_attention_module(module)


# ---------------------------------------------------------------------------
# Tensor helpers used by the patched attention
# ---------------------------------------------------------------------------

def _qwen_kv_heads_if_needed(k: torch.Tensor, v: torch.Tensor, q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV heads for [B,H,S,D] tensors if a future Qwen variant uses GQA."""
    kv_heads = int(k.shape[1])
    q_heads = int(q_heads)
    if kv_heads == q_heads:
        return k, v
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise RuntimeError(f"Cannot expand Qwen-Image KV heads: q={q_heads}, kv={kv_heads}.")
    n = q_heads // kv_heads
    k = k.unsqueeze(2).repeat(1, 1, n, 1, 1).flatten(1, 2)
    v = v.unsqueeze(2).repeat(1, 1, n, 1, 1).flatten(1, 2)
    return k, v


def _native_reference_token_count(transformer_options: dict[str, Any] | None) -> int:
    if not isinstance(transformer_options, dict):
        return 0
    value = transformer_options.get("reference_image_num_tokens", None)
    if value is None:
        return 0
    if torch.is_tensor(value):
        try:
            return max(0, int(value.detach().long().sum().item()))
        except Exception:
            return 0
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            i = _safe_int(item, 0)
            if i is not None:
                total += max(0, int(i))
        return total
    i = _safe_int(value, 0)
    return max(0, int(i or 0))


def _target_image_range(seq_txt: int, seq_img: int, transformer_options: dict[str, Any] | None) -> tuple[int, int]:
    """Return the real target-image token range in joint [txt, img] sequence coordinates.

    Qwen's native ref_latents are appended after the actual target image tokens in the
    image stream. They should not be used as the target image range for cross-batch
    AdaIN/V-injection/KV append, because the model later keeps only the original
    target-image tokens in its output.
    """
    seq_txt = int(seq_txt)
    seq_img = int(seq_img)
    native_ref = min(max(0, _native_reference_token_count(transformer_options)), seq_img)
    return seq_txt, seq_txt + max(0, seq_img - native_ref)


def _coerce_text_mask(mask: Any, batch_size: int, seq_txt: int, device: Any, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if torch.is_tensor(mask):
        out = mask.detach().to(device=device)
    else:
        try:
            out = torch.as_tensor(mask, device=device)
        except Exception as exc:
            raise RuntimeError("Qwen-Image attention mask conversion failed.") from exc

    if out.ndim == 0:
        return None
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.ndim > 2:
        out = out.reshape(out.shape[0], -1)

    if int(out.shape[0]) != int(batch_size):
        out = _repeat_to_batch(out, int(batch_size))
    if int(out.shape[1]) != int(seq_txt):
        out = _pad_or_truncate_tokens(out, int(seq_txt))

    if out.is_floating_point():
        return out.to(dtype=dtype)

    # Bool/int masks use ComfyUI Qwen's convention: valid=1 -> 0, invalid=0 -> -max.
    out = out.to(dtype=dtype)
    return (out - 1.0) * torch.finfo(dtype).max


def _build_joint_text_mask(
    encoder_hidden_states_mask: Any,
    cfg: dict[str, Any] | None,
    batch_size: int,
    seq_txt: int,
    seq_img: int,
    device: Any,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    forced = cfg.get("forced_cap_mask", None) if isinstance(cfg, dict) else None
    text_mask = None
    if torch.is_tensor(forced):
        text_mask = _coerce_text_mask(forced, batch_size, seq_txt, device, dtype)
    if text_mask is None:
        text_mask = _coerce_text_mask(encoder_hidden_states_mask, batch_size, seq_txt, device, dtype)
    if text_mask is None:
        return None
    attn_mask = torch.zeros((int(batch_size), 1, int(seq_txt) + int(seq_img)), dtype=dtype, device=device)
    attn_mask[:, 0, : int(seq_txt)] = text_mask
    return attn_mask


def _append_ref_padding_to_mask(mask: Any, target_bsz: int, ref_len: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if int(ref_len) <= 0:
        raise RuntimeError(f"Cannot append Qwen-Image reference padding: ref_len={ref_len}.")
    mask_t = mask[: int(target_bsz)]
    if mask_t.ndim < 2:
        raise RuntimeError(f"Cannot append Qwen-Image reference padding: mask ndim={mask_t.ndim}, expected >=2.")
    padding = torch.zeros((*mask_t.shape[:-1], int(ref_len)), device=mask_t.device, dtype=mask_t.dtype)
    return torch.cat([mask_t, padding], dim=-1)


def _slice_mask(mask: Any, start: int, end: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask[int(start): int(end)]

def _attention_with_reference_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: torch.Tensor,
    attn_mask: Any,
    transformer_options: dict[str, Any],
    image_range: tuple[int, int],
    module_name: str,
    dm: Any,
    stats: Any,
    helpers: dict[str, Any],
) -> torch.Tensor:
    """Qwen-Image joint attention replacement for tensors in [B,H,S,D]."""
    config_key = str(helpers.get("config_key", CONFIG_KEY))
    cfg = transformer_options.get(config_key) if isinstance(transformer_options, dict) else None

    heads = int(q.shape[1]) if torch.is_tensor(q) and q.ndim >= 2 else 0
    if not cfg or not cfg.get("enabled"):
        q_rope = apply_rope1(q, pe)
        k_rope = apply_rope1(k, pe)
        return optimized_attention_masked(
            q_rope, k_rope, v, heads, attn_mask,
            skip_reshape=True, transformer_options=transformer_options,
        )

    target_bsz = int(cfg.get("cross_batch_target_batch", 0))
    if target_bsz <= 0:
        raise RuntimeError(f"{DISPLAY_NAME} Untwisting enabled in {module_name}, but cross_batch_target_batch={target_bsz}.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            f"{DISPLAY_NAME} Untwisting expected q/k/v as [B,H,S,D] tensors in {module_name}; "
            f"got q.ndim={q.ndim}, k.ndim={k.ndim}, v.ndim={v.ndim}."
        )

    bsz, q_heads, seqlen, head_dim = q.shape
    if int(bsz) < target_bsz * 2:
        raise RuntimeError(
            f"{DISPLAY_NAME} Untwisting expected at least target+reference batches in {module_name}; "
            f"bsz={bsz}, target_bsz={target_bsz}."
        )

    block_idx = int(transformer_options.get("block_index", block_index_from_name(module_name)))
    active_blocks = cfg.get("active_blocks", set())
    if active_blocks and block_idx not in active_blocks:
        q_rope = apply_rope1(q, pe)
        k_rope = apply_rope1(k, pe)
        return optimized_attention_masked(
            q_rope, k_rope, v, int(q_heads), attn_mask,
            skip_reshape=True, transformer_options=transformer_options,
        )

    img_s, img_e = int(image_range[0]), int(image_range[1])
    img_s = max(0, min(img_s, int(seqlen)))
    img_e = max(img_s, min(img_e, int(seqlen)))
    if img_e <= img_s:
        raise RuntimeError(
            f"{DISPLAY_NAME} Untwisting has an empty target image token range in {module_name}: "
            f"{(img_s, img_e)} for seqlen={seqlen}."
        )

    if hasattr(stats, "attn_calls"):
        stats.attn_calls += 1
    if hasattr(stats, "adapter_attn_calls"):
        stats.adapter_attn_calls += 1

    # Publish Qwen-specific runtime metadata for debugging and shared helper fallbacks.
    axes_dims = cfg.get("axes_dims") or axes_dims_from_dm(dm)
    cfg["axes_dims"] = list(axes_dims)
    cfg["head_dim"] = int(head_dim)
    cfg["seq_len"] = int(seqlen)
    cfg["target_real_range"] = (int(img_s), int(img_e))
    cfg["ref_real_ranges"] = [(int(img_s), int(img_e))]
    cfg["ref_k_ranges"] = [(int(img_s), int(img_e))]
    cfg["target_qk_adain_ranges"] = [(int(img_s), int(img_e))]

    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    apply_qkv_shared_effects = helpers["apply_qkv_shared_effects"]
    apply_attention_output_shared_effects = helpers["apply_attention_output_shared_effects"]
    lerp = helpers["lerp"]

    q, k, v = apply_qkv_shared_effects(
        q, k, v,
        cfg,
        target_bsz,
        module_name,
        layout="BHSD",
        token_ranges=[(img_s, img_e)],
    )

    # Match ComfyUI Qwen's native order: apply RoPE before optimized attention.
    q = apply_rope1(q, pe)
    k = apply_rope1(k, pe)

    progress = float(cfg.get("progress", 0.0))
    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
    low_scale = lerp(cfg["low_scale_start"], cfg["low_scale_end"], progress)
    beta = float(cfg.get("beta", 2.0))
    cfg["_debug_high_scale"] = float(high_scale)
    cfg["_debug_low_scale"] = float(low_scale)

    scale_vec = build_frequency_scale_vector(
        int(head_dim),
        axes_dims,
        high_scale,
        low_scale,
        beta,
        k.device,
        k.dtype,
        runtime_cfg=cfg,
    ).view(1, 1, 1, int(head_dim))

    print_debug = helpers.get("print_rope_scale_debug", None)
    if callable(print_debug):
        print_debug(stats, cfg, module_name, scale_vec)

    ref_k = k[target_bsz: target_bsz * 2, :, img_s:img_e, :] * scale_vec
    ref_v = v[target_bsz: target_bsz * 2, :, img_s:img_e, :]
    ref_len = int(ref_k.shape[2])
    if ref_len <= 0:
        raise RuntimeError(
            f"{DISPLAY_NAME} Untwisting produced empty reference K/V in {module_name}; "
            f"image_range={(img_s, img_e)}."
        )

    # Target stream attends to its normal full joint sequence plus paired reference image K/V.
    q_t = q[:target_bsz]
    k_t = torch.cat([k[:target_bsz], ref_k], dim=2)
    v_t = torch.cat([v[:target_bsz], ref_v], dim=2)
    k_t, v_t = _qwen_kv_heads_if_needed(k_t, v_t, int(q_heads))
    mask_t = _append_ref_padding_to_mask(attn_mask, target_bsz, ref_len)

    out_t = optimized_attention_masked(
        q_t, k_t, v_t, int(q_heads), mask_t,
        skip_reshape=True, transformer_options=transformer_options,
    )

    # Reference stream is evaluated normally so later blocks receive valid reference activations.
    q_r = q[target_bsz: target_bsz * 2]
    k_r = k[target_bsz: target_bsz * 2]
    v_r = v[target_bsz: target_bsz * 2]
    k_r, v_r = _qwen_kv_heads_if_needed(k_r, v_r, int(q_heads))
    mask_r = _slice_mask(attn_mask, target_bsz, target_bsz * 2)
    out_r = optimized_attention_masked(
        q_r, k_r, v_r, int(q_heads), mask_r,
        skip_reshape=True, transformer_options=transformer_options,
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

    # Preserve any additional batches without cross-batch reference injection.
    if int(bsz) > target_bsz * 2:
        q_e = q[target_bsz * 2:]
        k_e = k[target_bsz * 2:]
        v_e = v[target_bsz * 2:]
        k_e, v_e = _qwen_kv_heads_if_needed(k_e, v_e, int(q_heads))
        mask_e = _slice_mask(attn_mask, target_bsz * 2, int(bsz))
        outs.append(optimized_attention_masked(
            q_e, k_e, v_e, int(q_heads), mask_e,
            skip_reshape=True, transformer_options=transformer_options,
        ))

    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Optional conditioning / patch hooks
# ---------------------------------------------------------------------------

def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Qwen-Image reference conditioning does not need adapter preprocessing."""
    return ref_conditioning, "not-applicable"

def patch_attention_modules(
    dm: Any,
    stats: Any,
    helpers: dict[str, Any] | None = None,
) -> tuple[int, int, int, list[str]]:
    """Patch Qwen-Image Attention.forward modules.

    Qwen-Image builds joint text+image Q/K/V inside ``Attention.forward`` as
    [batch, heads, sequence, dim], applies ``apply_rope1`` to Q/K, and then calls
    ``optimized_attention_masked``. This patch intercepts that exact point so
    target batches can append paired reference image K/V after RoPE.
    """
    helpers = helpers or {}
    prefix = str(helpers.get("prefix", "[UntwistingRoPE]"))
    config_key = str(helpers.get("config_key", CONFIG_KEY))

    required_helpers = ("lerp", "build_frequency_scale_vector", "apply_qkv_shared_effects", "apply_attention_output_shared_effects")
    missing = [name for name in required_helpers if not callable(helpers.get(name))]
    if missing:
        raise RuntimeError(f"{DISPLAY_NAME} adapter missing required helper(s): {missing}")

    # Store the resolved config key in the helper map passed down to the local attention function.
    local_helpers = dict(helpers)
    local_helpers["config_key"] = config_key

    matched = installed = restored = 0
    patched_names: list[str] = []

    for name, module in dm.named_modules():
        if not is_attention_name(name, 0, 999):
            continue
        if not is_qwen_attention_module(module):
            continue

        matched += 1
        patched_names.append(name)

        if hasattr(module, "_untwist_orig_qwen_image_forward"):
            module.forward = module._untwist_orig_qwen_image_forward
            restored += 1
        else:
            module._untwist_orig_qwen_image_forward = module.forward
        original_forward = module._untwist_orig_qwen_image_forward

        def make_forward(orig, module_name: str):
            def patched_forward(
                self,
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: torch.FloatTensor = None,
                encoder_hidden_states_mask: torch.FloatTensor = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                image_rotary_emb: Optional[torch.Tensor] = None,
                transformer_options={},
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                try:
                    cfg = (
                        transformer_options.get(config_key)
                        if isinstance(transformer_options, dict) else None
                    )
                    if not cfg or not cfg.get("enabled"):
                        return orig(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            attention_mask=attention_mask,
                            image_rotary_emb=image_rotary_emb,
                            transformer_options=transformer_options,
                        )

                    if encoder_hidden_states is None:
                        raise RuntimeError(f"{DISPLAY_NAME} attention patch expected encoder_hidden_states in {module_name}.")
                    if image_rotary_emb is None:
                        raise RuntimeError(f"{DISPLAY_NAME} attention patch expected image_rotary_emb in {module_name}.")
                    if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} attention patch expected hidden_states [B,S,C] in {module_name}; "
                            f"got {type(hidden_states).__name__} ndim={getattr(hidden_states, 'ndim', None)}."
                        )
                    if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} attention patch expected encoder_hidden_states [B,T,C] in {module_name}; "
                            f"got {type(encoder_hidden_states).__name__} ndim={getattr(encoder_hidden_states, 'ndim', None)}."
                        )

                    batch_size = int(hidden_states.shape[0])
                    seq_img = int(hidden_states.shape[1])
                    seq_txt = int(encoder_hidden_states.shape[1])
                    target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                    if target_bsz <= 0:
                        raise RuntimeError(f"{DISPLAY_NAME} Untwisting enabled in {module_name}, but cross_batch_target_batch={target_bsz}.")
                    if batch_size < target_bsz * 2:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} Untwisting expected at least target+reference batches in {module_name}; "
                            f"batch_size={batch_size}, target_bsz={target_bsz}."
                        )

                    block_idx = int(transformer_options.get("block_index", block_index_from_name(module_name)))
                    active_blocks = cfg.get("active_blocks", set())
                    if active_blocks and block_idx not in active_blocks:
                        return orig(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            attention_mask=attention_mask,
                            image_rotary_emb=image_rotary_emb,
                            transformer_options=transformer_options,
                        )

                    transformer_patches = transformer_options.get("patches", {}) if isinstance(transformer_options, dict) else {}
                    extra_options = transformer_options.copy() if isinstance(transformer_options, dict) else {}

                    img_query = self.to_q(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
                    img_key = self.to_k(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
                    img_value = self.to_v(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2)

                    txt_query = self.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
                    txt_key = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
                    txt_value = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)

                    img_query = self.norm_q(img_query)
                    img_key = self.norm_k(img_key)
                    txt_query = self.norm_added_q(txt_query)
                    txt_key = self.norm_added_k(txt_key)

                    joint_query = torch.cat([txt_query, img_query], dim=2)
                    joint_key = torch.cat([txt_key, img_key], dim=2)
                    joint_value = torch.cat([txt_value, img_value], dim=2)

                    # Preserve Qwen's public patch metadata: img_slice covers the complete image stream.
                    extra_options["img_slice"] = [int(seq_txt), int(joint_query.shape[2])]
                    if "attn1_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_patch"]:
                            out = patch(
                                joint_query,
                                joint_key,
                                joint_value,
                                pe=image_rotary_emb,
                                attn_mask=encoder_hidden_states_mask,
                                extra_options=extra_options,
                            )
                            joint_query = out.get("q", joint_query)
                            joint_key = out.get("k", joint_key)
                            joint_value = out.get("v", joint_value)
                            image_rotary_emb = out.get("pe", image_rotary_emb)
                            encoder_hidden_states_mask = out.get("attn_mask", encoder_hidden_states_mask)

                    # Build the additive joint mask after patches/forced_cap_mask so RF reference
                    # conditioning masks can apply even though Qwen has no patchify hook.
                    attn_mask = _build_joint_text_mask(
                        encoder_hidden_states_mask,
                        cfg,
                        int(joint_query.shape[0]),
                        seq_txt,
                        seq_img,
                        hidden_states.device,
                        hidden_states.dtype,
                    )

                    image_range = _target_image_range(seq_txt, seq_img, transformer_options)
                    joint_hidden_states = _attention_with_reference_kv(
                        joint_query,
                        joint_key,
                        joint_value,
                        image_rotary_emb,
                        attn_mask,
                        transformer_options,
                        image_range,
                        module_name,
                        dm,
                        stats,
                        local_helpers,
                    )

                    if "attn1_output_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_output_patch"]:
                            joint_hidden_states = patch(joint_hidden_states, extra_options)

                    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
                    img_attn_output = joint_hidden_states[:, seq_txt:, :]

                    img_attn_output = self.to_out[0](img_attn_output)
                    img_attn_output = self.to_out[1](img_attn_output)
                    txt_attn_output = self.to_add_out(txt_attn_output)

                    return img_attn_output, txt_attn_output
                except Exception as exc:
                    if hasattr(stats, "adapter_attn_failures"):
                        stats.adapter_attn_failures += 1
                    raise RuntimeError(
                        f"{DISPLAY_NAME} attention patch failed in {module_name}; "
                        f"strict mode refuses to call original forward after patch failure: {exc}"
                    ) from exc
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        setattr(module, "_untwist_qwen_image_active", True)
        installed += 1

    if installed <= 0:
        raise RuntimeError(f"{DISPLAY_NAME} adapter patch failed: no compatible transformer_blocks.N.attn modules were installed.")
    return matched, installed, restored, patched_names


def uses_reference_branch_kv() -> bool:
    return False

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_match(model_info: dict[str, Any]) -> str:
    model_config_class = str(model_info.get("model_config_class", ""))
    unet_config = model_info.get("unet_config", {})
    if isinstance(unet_config, dict):
        image_model = str(unet_config.get("image_model", ""))
    else:
        image_model = str(model_info.get("image_model", ""))
    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (
        f"{DISPLAY_NAME}: model_config_class={model_config_class!r}, "
        f"image_model={image_model!r}, supported_classes={{{supported}}}"
    )


__all__ = [
    "ARCHITECTURE",
    "DISPLAY_NAME",
    "CONFIG_KEY",
    "SUPPORTED_MODEL_CONFIG_CLASSES",
    "matches_model",
    "is_model_identity",
    "find_diffusion_model",
    "default_runtime_cfg",
    "axes_dims_from_dm",
    "head_dim_from_dm",
    "num_heads_from_dm",
    "is_attention_name",
    "is_qwen_attention_name",
    "block_index_from_name",
    "is_qwen_attention_module",
    "is_joint_attention",
    "prepare_reference_conditioning",
    "patch_attention_modules",
    "uses_reference_branch_kv",
    "describe_match",
]
