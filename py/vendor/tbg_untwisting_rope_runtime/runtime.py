from __future__ import annotations
import math
import types
from typing import Any, Callable, Dict, List, Optional, Tuple
from . import models as model_adapters
import torch
import comfy.utils
from . import verbose_prints as vp
from .sdpa_fix import install_optimized_attention_override as _maybe_install_untwist_attention_override

_TRANSFORMER_CONFIG_KEY = model_adapters.CONFIG_KEY

def _coerce_strength01(value: Any, default: float = 0.0) -> float:
    try:
        strength = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid strength value: {value!r}.') from exc
    if not math.isfinite(strength):
        raise ValueError(f'Invalid strength value: {value!r} is not finite.')
    return max(0.0, min(1.0, strength))

_AXIS0_ROPE_MODES = {'default', 'match_axes', 'constant'}

def _coerce_axis0_rope_mode(value: Any = None, legacy_scale: Any = None) -> str:
    """Normalize the axis-0 RoPE behavior selector."""
    if value is None:
        if legacy_scale is not None:
            try:
                return 'default' if float(legacy_scale) < 0.0 else 'constant'
            except Exception as exc:
                raise ValueError(f'Invalid legacy axis0_rope_scale value: {legacy_scale!r}.') from exc
        return 'default'

    mode = str(value or 'default').strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'legacy': 'default',
        'low': 'default',
        'low_scale': 'default',
        'match_axis1': 'match_axes',
        'match_axis_1': 'match_axes',
        'match_axis_1_plus': 'match_axes',
        'match_axes_1plus': 'match_axes',
        'same_as_axes': 'match_axes',
        'same_as_axis1': 'match_axes',
        'override': 'constant',
        'fixed': 'constant',
    }
    mode = aliases.get(mode, mode)
    if mode not in _AXIS0_ROPE_MODES:
        raise ValueError(f'Invalid axis0_rope_mode={value!r}. Expected one of {sorted(_AXIS0_ROPE_MODES)}.')
    return mode

def _coerce_axis0_rope_scale(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid axis0_rope_scale value: {value!r}.') from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid axis0_rope_scale value: {value!r} is not finite.')
    return max(0.0, v)

# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_active_blocks(blocks_str):
    active = set()
    if not isinstance(blocks_str, str) or not blocks_str.strip():
        return active
    for part in blocks_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_i, end_i = int(start), int(end)
            except ValueError as exc:
                raise ValueError(f'Invalid active block range {part!r}.') from exc
            if end_i < start_i:
                raise ValueError(f'Invalid active block range {part!r}: end is smaller than start.')
            active.update(range(start_i, end_i + 1))
        else:
            try:
                active.add(int(part))
            except ValueError as exc:
                raise ValueError(f'Invalid active block index {part!r}.') from exc
    return active

def _select_model_adapter(model_patcher: Any, model_info: Optional[Dict[str, Any]] = None) -> Any:
    adapter = model_adapters.identify(model_patcher, model_info or {})
    if isinstance(model_info, dict):
        model_info['architecture'] = model_adapters.adapter_key(adapter)
        model_info['architecture_name'] = model_adapters.adapter_label(adapter)
    return adapter

def _safe_get_diffusion_model(model_patcher: Any, adapter: Any) -> Any:
    return adapter.find_diffusion_model(model_patcher)

def _repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if comfy is not None and hasattr(comfy.utils, 'repeat_to_batch_size'):
        return comfy.utils.repeat_to_batch_size(x, batch)
    reps = math.ceil(batch / x.shape[0])
    return x.repeat((reps,) + (1,) * (x.ndim - 1))[:batch]

def _clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out = options.copy()
    out['transformer_options'] = options.get('transformer_options', {}).copy()
    return out

def _clone_conditioning_for_rf(c: Dict[str, Any]) -> Dict[str, Any]:
    out = c.copy()
    to  = out.get('transformer_options', {})
    if isinstance(to, dict):
        to = to.copy()
        to.pop(_TRANSFORMER_CONFIG_KEY, None)
        out['transformer_options'] = to
    else:
        out['transformer_options'] = {}
    return out

def _slice_conditioning_batch(obj: Any, start: int, end: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) >= end:
                return obj[start:end]
        except Exception as exc:
            raise RuntimeError('Conditioning batch slice failed.') from exc
        return obj
    if isinstance(obj, dict):
        return {
            k: (v if k == 'transformer_options' else _slice_conditioning_batch(v, start, end))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_slice_conditioning_batch(v, start, end) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_conditioning_batch(v, start, end) for v in obj)
    return obj

def _build_rf_conditioning_kwargs(
    c: Dict[str, Any],
    ref_conditioning: Any,
    target_b: int,
) -> Tuple[Dict[str, Any], str]:
    if ref_conditioning is None:
        raise RuntimeError(
            'RF conditioning failed: ref_conditioning is required when RFInversion uses a reference latent.'
        )
    if target_b <= 0:
        raise RuntimeError(f'RF conditioning failed: invalid target batch size {target_b}.')

    try:
        merged, _forced = _merge_reference_conditioning_into_c(c, ref_conditioning, target_b)
        ref_only = _slice_conditioning_batch(merged, target_b, target_b * 2)
    except Exception as exc:
        raise RuntimeError('RF conditioning failed while merging reference conditioning.') from exc

    return _clone_conditioning_for_rf(ref_only), 'reference'

def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def _triangle_ramp01(progress: Any) -> float:
    """0→1 from progress 0→0.5, then 1→0 from progress 0.5→1."""
    p = _coerce_strength01(progress)
    return max(0.0, min(1.0, 1.0 - abs((2.0 * p) - 1.0)))

def _repeat_conditioning_tree(obj: Any, src: int, tgt: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) == src:
                return _repeat_to_batch(obj, tgt)
        except Exception as exc:
            raise RuntimeError('Conditioning tree repeat failed.') from exc
        return obj
    if isinstance(obj, dict):
        return {
            k: v if k in ('transformer_options', 'ref_latents', 'ref_contexts')
            else _repeat_conditioning_tree(v, src, tgt)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_repeat_conditioning_tree(v, src, tgt) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_repeat_conditioning_tree(v, src, tgt) for v in obj)
    return obj

_TEXT_CONDITIONING_KEYS = {
    'c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond',
    'encoder_hidden_states', 'txt', 'text', 'text_embeddings',
}
_POOLED_CONDITIONING_KEYS = {
    'pooled_output', 'clip_pooled', 'pooled', 'y', 'vector',
}
_MASK_CONDITIONING_KEYS = {
    'attention_mask', 'crossattn_mask', 'c_crossattn_mask',
    'cap_mask', 'cond_mask', 'mask',
}
_NUM_TOKEN_KEYS = {
    'num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens',
}
_CONDITIONING_META_ALIASES = {
    'pooled_output': ('pooled_output', 'clip_pooled', 'pooled', 'y', 'vector'),
    'clip_pooled':   ('clip_pooled',   'pooled_output', 'pooled', 'y', 'vector'),
    'pooled':        ('pooled',        'pooled_output', 'clip_pooled', 'y', 'vector'),
    'y':             ('y',             'pooled_output', 'clip_pooled', 'pooled', 'vector'),
    'vector':        ('vector',        'pooled_output', 'clip_pooled', 'pooled', 'y'),
    'attention_mask':   ('attention_mask',   'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'),
    'crossattn_mask':   ('crossattn_mask',   'attention_mask', 'c_crossattn_mask', 'cap_mask', 'mask'),
    'c_crossattn_mask': ('c_crossattn_mask', 'attention_mask', 'crossattn_mask',   'cap_mask', 'mask'),
    'cap_mask':         ('cap_mask',         'attention_mask', 'crossattn_mask',   'c_crossattn_mask', 'mask'),
    'mask':             ('mask',             'attention_mask', 'crossattn_mask',   'c_crossattn_mask', 'cap_mask'),
    'num_tokens':     ('num_tokens',     'tokens_num', 'n_tokens', 'cap_num_tokens'),
    'tokens_num':     ('tokens_num',     'num_tokens', 'n_tokens', 'cap_num_tokens'),
    'n_tokens':       ('n_tokens',       'num_tokens', 'tokens_num', 'cap_num_tokens'),
    'cap_num_tokens': ('cap_num_tokens', 'num_tokens', 'tokens_num', 'n_tokens'),
}

def _first_tensor_in_conditioning_entry(entry: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if torch.is_tensor(entry):
        return entry, meta
    if isinstance(entry, dict):
        meta.update(entry)
        for key in ('c_crossattn', 'crossattn', 'conditioning', 'cond', 'context', 'cap_feats'):
            value = entry.get(key)
            if torch.is_tensor(value):
                return value, meta
        for value in entry.values():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value, meta
        return None, meta
    if isinstance(entry, (list, tuple)):
        cond: Optional[torch.Tensor] = None
        for item in entry:
            if torch.is_tensor(item) and cond is None:
                cond = item
            elif isinstance(item, dict):
                meta.update(item)
        if cond is not None:
            return cond, meta
        for item in entry:
            cond, nested_meta = _first_tensor_in_conditioning_entry(item)
            if nested_meta:
                meta.update(nested_meta)
            if cond is not None:
                return cond, nested_meta
    return None, meta

def _extract_reference_conditioning(ref_conditioning: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if ref_conditioning is None:
        return None, {}
    if torch.is_tensor(ref_conditioning) or isinstance(ref_conditioning, dict):
        return _first_tensor_in_conditioning_entry(ref_conditioning)
    if isinstance(ref_conditioning, (list, tuple)):
        merged_meta: Dict[str, Any] = {}
        for entry in ref_conditioning:
            cond, meta = _first_tensor_in_conditioning_entry(entry)
            if meta:
                merged_meta.update(meta)
            if cond is not None:
                return cond, merged_meta
        return None, merged_meta
    return None, {}

def _meta_get(meta: Dict[str, Any], key: str) -> Any:
    for alias in _CONDITIONING_META_ALIASES.get(key, (key,)):
        if alias in meta:
            return meta[alias]
    return None

def _as_tensor_like(value: Any, like: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(
            device=like.device,
            dtype=like.dtype if value.is_floating_point() else value.dtype,
        )
    try:
        return torch.as_tensor(value, device=like.device)
    except Exception as exc:
        raise RuntimeError(f'Could not convert reference metadata value to tensor: {value!r}.') from exc

def _coerce_ref_tensor_like_target(ref_value, target_value, target_b):
    ref = ref_value.to(
        device=target_value.device,
        dtype=target_value.dtype if ref_value.is_floating_point() else ref_value.dtype,
    )
    if target_value.ndim >= 2 and ref.ndim == target_value.ndim - 1:
        ref = ref.unsqueeze(0)
    if ref.ndim > 0 and int(ref.shape[0]) != target_b:
        ref = _repeat_to_batch(ref, target_b)
    if target_value.ndim >= 3 and ref.ndim >= 3:
        if int(ref.shape[1]) != int(target_value.shape[1]):
            ref = _pad_or_truncate_tokens(ref, int(target_value.shape[1]))
    elif target_value.ndim >= 2 and ref.ndim >= 2:
        if int(ref.shape[1]) != int(target_value.shape[1]):
            ref = _pad_or_truncate_tokens(ref, int(target_value.shape[1]))
    return ref

def _conditioning_mask_from_source(source, batch, padded_tokens, device):
    if source is None:
        return None
    if torch.is_tensor(source):
        x = source.detach().to(device=device)
        if x.ndim == 0:
            return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
        if x.ndim == 1:
            if x.numel() == batch and not x.is_floating_point():
                return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
            if x.numel() == 1:
                return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
            x = x.view(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if int(x.shape[0]) != batch:
            x = _repeat_to_batch(x, batch)
        if int(x.shape[1]) != padded_tokens:
            x = _pad_or_truncate_tokens(x, padded_tokens)
        if x.is_floating_point() and torch.any(x < 0):
            return (x >= 0).to(torch.bool)
        return x.to(torch.bool)
    if isinstance(source, (list, tuple)):
        try:
            return _conditioning_mask_from_source(
                torch.as_tensor(source, device=device), batch, padded_tokens, device
            )
        except Exception as exc:
            raise RuntimeError('Conditioning mask conversion failed for list/tuple source.') from exc
    try:
        return _num_tokens_to_valid_mask(int(source), batch, padded_tokens, device)
    except Exception as exc:
        raise RuntimeError(f'Conditioning mask conversion failed for source={source!r}.') from exc

def _target_valid_mask_from_c(c, target_b, padded_tokens, device):
    for key in ('attention_mask', 'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'):
        mask = _conditioning_mask_from_source(c.get(key), target_b, padded_tokens, device)
        if mask is not None:
            return mask
    for key in ('num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens'):
        mask = _conditioning_mask_from_source(c.get(key), target_b, padded_tokens, device)
        if mask is not None:
            return mask
    return torch.ones((target_b, padded_tokens), device=device, dtype=torch.bool)

def _reference_valid_mask_from_conditioning(ref_cond, ref_meta, target_b, padded_tokens, device):
    for key in ('attention_mask', 'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'):
        mask = _conditioning_mask_from_source(
            _meta_get(ref_meta, key), target_b, padded_tokens, device
        )
        if mask is not None:
            return mask
    for key in ('num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens'):
        mask = _conditioning_mask_from_source(
            _meta_get(ref_meta, key), target_b, padded_tokens, device
        )
        if mask is not None:
            return mask
    real_tokens = int(ref_cond.shape[1]) if ref_cond.ndim >= 2 else padded_tokens
    return _num_tokens_to_valid_mask(real_tokens, target_b, padded_tokens, device)

def _conditioning_counts_from_mask(mask):
    m = mask.to(torch.bool)
    if m.ndim == 1:
        m = m.view(1, -1)
    return m.long().sum(dim=1)

def _concat_batch_conditioning_value(key, value, ref_cond, ref_meta, target_b, forced_cap_mask):
    if not torch.is_tensor(value):
        if key in _NUM_TOKEN_KEYS:
            try:
                target_counts = torch.as_tensor(
                    value, device=forced_cap_mask.device, dtype=torch.long,
                ).flatten()
                if target_counts.numel() == 1:
                    target_counts = target_counts.repeat(target_b)
                elif target_counts.numel() != target_b:
                    target_counts = _repeat_to_batch(
                        target_counts.view(-1, 1), target_b
                    ).flatten()
                ref_counts = _conditioning_counts_from_mask(
                    forced_cap_mask[target_b:target_b * 2]
                )
                return torch.cat([target_counts, ref_counts], dim=0)
            except Exception as exc:
                raise RuntimeError(f'Conditioning num-token merge failed for key={key!r}.') from exc
        if key in _MASK_CONDITIONING_KEYS:
            return forced_cap_mask
        return _repeat_conditioning_tree(value, target_b, target_b * 2)

    try:
        if value.ndim == 0 or int(value.shape[0]) != target_b:
            return value
    except Exception as exc:
        raise RuntimeError(f'Conditioning batch compatibility check failed for key={key!r}.') from exc

    ref_value: Optional[torch.Tensor] = None

    if key in _TEXT_CONDITIONING_KEYS or (
        value.ndim >= 3 and ref_cond.ndim >= 3
        and int(value.shape[-1]) == int(ref_cond.shape[-1])
    ):
        ref_value = ref_cond
    elif key in _POOLED_CONDITIONING_KEYS:
        meta_value = _meta_get(ref_meta, key)
        if meta_value is not None:
            ref_value = _as_tensor_like(meta_value, value)
    elif key in _MASK_CONDITIONING_KEYS:
        ref_value = forced_cap_mask[target_b:target_b * 2].to(
            device=value.device,
            dtype=value.dtype if value.is_floating_point() else torch.bool,
        )
        if value.is_floating_point() and torch.any(value < 0):
            ref_value = _mask_to_additive(ref_value.to(torch.bool), dtype=value.dtype)
    elif key in _NUM_TOKEN_KEYS:
        ref_value = _conditioning_counts_from_mask(
            forced_cap_mask[target_b:target_b * 2]
        ).to(device=value.device, dtype=value.dtype)

    if ref_value is None:
        ref_value = value

    ref_value = _coerce_ref_tensor_like_target(ref_value, value, target_b)
    return torch.cat([value, ref_value], dim=0)

def _merge_reference_conditioning_into_c(c, ref_conditioning, target_b):
    ref_cond, ref_meta = _extract_reference_conditioning(ref_conditioning)
    if ref_cond is None:
        raise RuntimeError(
            'ref_conditioning must be connected and must contain a valid '
            'CONDITIONING tensor when reference_latent is connected.'
        )

    target_text = None
    for key in ('c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond',
                'encoder_hidden_states'):
        value = c.get(key)
        if (torch.is_tensor(value) and value.ndim >= 3
                and int(value.shape[0]) == target_b):
            target_text = value
            break

    if target_text is None:
        for key, value in c.items():
            if (
                key != 'transformer_options'
                and torch.is_tensor(value)
                and value.ndim >= 3
                and int(value.shape[0]) == target_b
                and ref_cond.ndim >= 3
                and int(value.shape[-1]) == int(ref_cond.shape[-1])
            ):
                target_text = value
                break

    if target_text is None:
        raise RuntimeError(
            'Could not find the target text-conditioning tensor in model kwargs.'
        )

    if ref_cond.ndim == target_text.ndim - 1:
        ref_cond = ref_cond.unsqueeze(0)

    if ref_cond.ndim < 3 or int(ref_cond.shape[-1]) != int(target_text.shape[-1]):
        raise RuntimeError(
            f'ref_conditioning incompatible shape {tuple(ref_cond.shape)} '
            f'vs {tuple(target_text.shape)}.'
        )

    padded_tokens   = int(target_text.shape[1])
    device          = target_text.device
    target_mask     = _target_valid_mask_from_c(c, target_b, padded_tokens, device)
    ref_mask        = _reference_valid_mask_from_conditioning(
        ref_cond, ref_meta, target_b, padded_tokens, device
    )
    forced_cap_mask = torch.cat([target_mask, ref_mask], dim=0).to(torch.bool)

    out: Dict[str, Any] = {}
    for key, value in c.items():
        if key == 'transformer_options':
            out[key] = value
            continue
        out[key] = _concat_batch_conditioning_value(
            key, value, ref_cond, ref_meta, target_b, forced_cap_mask
        )
    return out, forced_cap_mask

# ═══════════════════════════════════════════════════════════════════════════════
# Token / mask helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_or_truncate_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    cur = int(x.shape[1])
    if cur == target_tokens:
        return x
    if cur > target_tokens:
        return x[:, :target_tokens, ...]
    pad_shape    = list(x.shape)
    pad_shape[1] = target_tokens - cur
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def _num_tokens_to_valid_mask(num_tokens, batch, padded_tokens, device):
    if torch.is_tensor(num_tokens):
        counts = num_tokens.detach().to(device=device).flatten().long()
        if counts.numel() == 1:
            counts = counts.repeat(batch)
        elif counts.numel() != batch:
            counts = _repeat_to_batch(counts.view(-1, 1), batch).flatten().long()
    elif isinstance(num_tokens, (list, tuple)):
        counts = torch.tensor(num_tokens, device=device, dtype=torch.long).flatten()
        if counts.numel() == 1:
            counts = counts.repeat(batch)
        elif counts.numel() != batch:
            counts = _repeat_to_batch(counts.view(-1, 1), batch).flatten().long()
    else:
        counts = torch.full(
            (batch,),
            int(num_tokens) if num_tokens is not None else padded_tokens,
            device=device, dtype=torch.long,
        )
    counts = counts.clamp(min=0, max=padded_tokens)
    ar = torch.arange(padded_tokens, device=device).view(1, padded_tokens)
    return ar < counts.view(batch, 1)

def _coerce_forced_cap_mask_for_feats(forced_cap_mask, cap_feats):
    mask = forced_cap_mask.to(device=cap_feats.device)
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    if mask.ndim > 0 and int(mask.shape[0]) != int(cap_feats.shape[0]):
        mask = _repeat_to_batch(mask, int(cap_feats.shape[0]))
    if mask.ndim == 2 and int(mask.shape[1]) != int(cap_feats.shape[1]):
        mask = _pad_or_truncate_tokens(mask, int(cap_feats.shape[1]))
    return mask.to(torch.bool)

def _mask_to_additive(valid_mask, dtype=torch.float32):
    valid = valid_mask.to(torch.bool)
    out   = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
    return out.masked_fill(~valid, -10000.0)

def _build_joint_additive_mask_from_cap_mask(
    cap_valid_mask, seq_len, text_range, device, dtype=torch.float32
):
    if not torch.is_tensor(cap_valid_mask) or cap_valid_mask.ndim < 2:
        return None
    if text_range is None:
        return None
    ts, te = int(text_range[0]), int(text_range[1])
    ts = max(0, min(ts, int(seq_len)))
    te = max(ts, min(te, int(seq_len)))
    if te <= ts:
        return None
    cap_valid_mask = cap_valid_mask.to(device=device).to(torch.bool)
    batch      = int(cap_valid_mask.shape[0])
    text_slots = te - ts
    text_valid = torch.zeros((batch, text_slots), device=device, dtype=torch.bool)
    copy_len   = min(text_slots, int(cap_valid_mask.shape[1]))
    if copy_len > 0:
        text_valid[:, :copy_len] = cap_valid_mask[:, :copy_len]
    full_valid = torch.ones((batch, int(seq_len)), device=device, dtype=torch.bool)
    full_valid[:, ts:te] = text_valid
    return _mask_to_additive(full_valid, dtype=dtype)

# ═══════════════════════════════════════════════════════════════════════════════
# RoPE frequency scale vector
# ═══════════════════════════════════════════════════════════════════════════════

def _build_frequency_scale_vector(
    head_dim, axes_dims, high_scale, low_scale, beta, device, dtype,
    axis0_rope_scale: Any = None,
    axis0_rope_mode: Any = None,
    runtime_cfg: Optional[Dict[str, Any]] = None,
):
    if not axes_dims or sum(int(x) for x in axes_dims) != head_dim:
        axes_dims = [head_dim]
    axes_dims = [int(x) for x in axes_dims]

    has_separate_axis0 = len(axes_dims) >= 2

    legacy_axis0_scale = None
    if isinstance(runtime_cfg, dict):
        legacy_axis0_scale = runtime_cfg.get('axis0_rope_scale', None)
        if axis0_rope_mode is None:
            axis0_rope_mode = runtime_cfg.get('axis0_rope_mode', None)
        if axis0_rope_scale is None:
            axis0_rope_scale = legacy_axis0_scale

    axis0_rope_mode = _coerce_axis0_rope_mode(
        axis0_rope_mode, legacy_scale=legacy_axis0_scale
    )
    axis0_rope_scale = _coerce_axis0_rope_scale(axis0_rope_scale, default=0.0)

    def _curve_scales(n_pairs: int) -> torch.Tensor:
        d_tilde = (
            torch.zeros(1, device=device, dtype=torch.float32)
            if n_pairs == 1
            else torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
        )
        return high_scale + (low_scale - high_scale) * d_tilde.pow(float(beta))

    pieces: List[torch.Tensor] = []
    for axis_idx, axis_dim in enumerate(axes_dims):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        if has_separate_axis0 and axis_idx == 0:
            if axis0_rope_mode == 'match_axes':
                # Axis 0 uses the same per-pair curve as axes 1+.
                pair_scales = _curve_scales(n_pairs)
            elif axis0_rope_mode == 'constant':
                pair_scales = torch.full(
                    (n_pairs,), float(axis0_rope_scale), device=device, dtype=torch.float32
                )
            else:
                # Default/legacy behavior: axis 0 is flat at low_scale.
                pair_scales = torch.full(
                    (n_pairs,), float(low_scale), device=device, dtype=torch.float32
                )
        else:
            pair_scales = _curve_scales(n_pairs)
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))
    out = torch.cat(pieces, dim=0)
    if out.numel() >= head_dim:
        return out[:head_dim]
    return torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# AdaIN helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _adain(target, style, eps=1e-6):
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std  = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std  = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean

def _reference_variance_channel_mask(style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Build a [B, 1, H, D] mask from reference/style V variance across sequence.

    High mask values indicate channels whose reference activations vary strongly
    over tokens, which tends to correspond to texture/color/style-bearing
    features.
    """
    if not torch.is_tensor(style) or style.ndim != 4:
        raise RuntimeError(
            f'{vp._PREFIX} variance-gated V effects expected reference V as rank-4 BSHD, '
            f'got {tuple(style.shape) if torch.is_tensor(style) else type(style).__name__}.'
        )
    style_var = style.float().var(dim=1, keepdim=True, unbiased=False)
    style_var_max = style_var.amax(dim=-1, keepdim=True).clamp_min(eps)
    return (style_var / style_var_max).clamp(0.0, 1.0).detach()

def _cosine_gated_v_injection(
    v_t: torch.Tensor,
    v_r: torch.Tensor,
    strength: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cosine-similarity-gated V injection.
    Only injects V_ref into V_target where they are semantically aligned
    (positive cosine similarity). Mismatched regions receive no injection.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return v_t

    v_tf = v_t.float()
    v_rf = v_r.float()

    t_u = v_tf / v_tf.norm(dim=-1, keepdim=True).clamp_min(eps)
    r_u = v_rf / v_rf.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Per-token per-head cosine similarity -> gate in [0, 1].
    gate = (t_u * r_u).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)  # [B,S,H,1]

    # Gated delta: only aligned positions receive injection.
    delta = (v_rf - v_tf) * gate * strength
    return (v_tf + delta).to(dtype=v_t.dtype)

def _cross_batch_adain_qk(xq, xk, cfg, target_bsz, strength, eps=1e-6, xv=None):
    return_v = xv is not None
    if target_bsz <= 0 or xq.shape[0] < target_bsz * 2:
        return (xq, xk, xv) if return_v else (xq, xk)
    a = max(0.0, min(1.0, strength))
    if a <= 0.0:
        return (xq, xk, xv) if return_v else (xq, xk)
    seqlen = xq.shape[1]
    for s, e in (cfg.get('target_qk_adain_ranges') or []):
        s, e = max(0, int(s)), min(int(e), seqlen)
        if e <= s:
            continue
        q_t, k_t = xq[:target_bsz, s:e], xk[:target_bsz, s:e]
        q_r, k_r = xq[target_bsz:target_bsz*2, s:e], xk[target_bsz:target_bsz*2, s:e]
        xq[:target_bsz, s:e] = q_t * (1 - a) + _adain(q_t, q_r, eps) * a
        xk[:target_bsz, s:e] = k_t * (1 - a) + _adain(k_t, k_r, eps) * a
    return (xq, xk, xv) if return_v else (xq, xk)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared Q/K/V effects
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_token_ranges(ranges: Any, seqlen: int) -> List[Tuple[int, int]]:
    """Clamp token ranges to the current sequence length and drop empty ranges."""
    out: List[Tuple[int, int]] = []
    for item in ranges or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        s = max(0, min(int(item[0]), int(seqlen)))
        e = max(s, min(int(item[1]), int(seqlen)))
        if e > s:
            out.append((s, e))
    return out

def _shared_effect_ranges(
    cfg: Dict[str, Any],
    seqlen: int,
    token_ranges: Any = None,
) -> List[Tuple[int, int]]:
    """Resolve explicit token ranges used by shared Q/K/V effects.
    """
    ranges = _normalize_token_ranges(token_ranges, seqlen)
    if ranges:
        return ranges
    raise RuntimeError(
        f'{vp._PREFIX} shared Q/K/V effects require explicit token_ranges; '
    )

def _apply_qkv_shared_effects(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: Dict[str, Any],
    target_bsz: int,
    module_name: str,
    *,
    layout: str = 'BSHD',
    token_ranges: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared core Q/K/V effect stack for adapter-owned attention patches.

    Adapters expose where Q/K/V exist, their layout, and the applicable token
    range. This function owns shared user-facing Q/K/V features.

    Supported layouts:
      - BSHD: [batch, sequence, heads, head_dim]
      - BHSD: [batch, heads, sequence, head_dim]
    """
    if not isinstance(cfg, dict) or not cfg.get('enabled', False):
        return q, k, v
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return q, k, v
    if int(target_bsz) <= 0 or int(v.shape[0]) < int(target_bsz) * 2:
        return q, k, v

    layout_u = str(layout or 'BSHD').upper()
    if layout_u == 'BSHD':
        q_bshd, k_bshd, v_bshd = q, k, v
        restore = lambda qq, kk, vv: (qq, kk, vv)
    elif layout_u == 'BHSD':
        q_bshd, k_bshd, v_bshd = q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)
        restore = lambda qq, kk, vv: (qq.movedim(1, 2), kk.movedim(1, 2), vv.movedim(1, 2))
    else:
        raise RuntimeError(
            f'{vp._PREFIX} shared QKV effects failed in {module_name}: '
            f'unsupported layout={layout!r}.'
        )

    if q_bshd.ndim != 4 or k_bshd.ndim != 4 or v_bshd.ndim != 4:
        raise RuntimeError(
            f'{vp._PREFIX} shared QKV effects failed in {module_name}: '
            f'expected Q/K/V as rank-4 after layout normalization, got '
            f'q={tuple(q_bshd.shape)}, k={tuple(k_bshd.shape)}, v={tuple(v_bshd.shape)}.'
        )

    seqlen = int(v_bshd.shape[1])
    ranges = _shared_effect_ranges(cfg, seqlen, token_ranges)

    # Shared pre-RoPE Q/K AdaIN.
    adain_strength = _coerce_strength01(cfg.get('adain_strength', 0.0)) if cfg.get('apply_adain') else 0.0
    if adain_strength > 0.0:
        cfg_for_adain = dict(cfg)
        cfg_for_adain['target_qk_adain_ranges'] = list(ranges)
        q_bshd = q_bshd.clone()
        k_bshd = k_bshd.clone()
        q_bshd, k_bshd = _cross_batch_adain_qk(
            q_bshd, k_bshd, cfg_for_adain, int(target_bsz), float(adain_strength)
        )
        cfg['_debug_qk_adain_strength'] = float(adain_strength)
        cfg['_debug_qk_adain_module'] = str(module_name)
        cfg['_debug_qk_adain_ranges'] = list(ranges)

    # Shared key-subspace Gram-Schmidt alignment.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    key_subspace_target = _coerce_strength01(cfg.get('key_subspace_alignment', 0.0))
    key_subspace_progress = _coerce_strength01(cfg.get('progress', 0.0))
    key_subspace_ramp = _triangle_ramp01(key_subspace_progress)
    key_subspace = _lerp(0.0, key_subspace_target, key_subspace_ramp)
    if key_subspace > 0.0:
        k_bshd = k_bshd.clone()
        for s, e in ranges:
            k_t = k_bshd[:target_bsz, s:e]
            k_r = k_bshd[target_bsz:target_bsz * 2, s:e]
            if k_t.shape != k_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared key-subspace alignment failed in {module_name}: '
                    f'target/ref K range shape mismatch: target={tuple(k_t.shape)} ref={tuple(k_r.shape)}.'
                )

            k_tf = k_t.float()
            k_rf = k_r.float()
            dot_num = (k_tf * k_rf).sum(dim=-1, keepdim=True)
            dot_den = (k_rf * k_rf).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            proj = k_rf * (dot_num / dot_den)
            aligned = k_tf * (1.0 - key_subspace) + proj * key_subspace
            k_bshd[:target_bsz, s:e] = aligned.to(dtype=k_bshd.dtype)

        cfg['_debug_key_subspace_alignment_strength'] = float(key_subspace)
        cfg['_debug_key_subspace_alignment_target'] = float(key_subspace_target)
        cfg['_debug_key_subspace_alignment_progress'] = float(key_subspace_progress)
        cfg['_debug_key_subspace_alignment_ramp'] = float(key_subspace_ramp)
        cfg['_debug_key_subspace_alignment_module'] = str(module_name)
        cfg['_debug_key_subspace_alignment_ranges'] = list(ranges)

    # Shared cosine-gated V injection.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    cosine_v_inj_target = _coerce_strength01(cfg.get('cosine_gated_v_injection', 0.0))
    cosine_v_inj_progress = _coerce_strength01(cfg.get('progress', 0.0))
    cosine_v_inj_ramp = _triangle_ramp01(cosine_v_inj_progress)
    cosine_v_inj = _lerp(0.0, cosine_v_inj_target, cosine_v_inj_ramp)
    if cosine_v_inj > 0.0:
        v_bshd = v_bshd.clone()
        for s, e in ranges:
            v_t = v_bshd[:target_bsz, s:e]
            v_r = v_bshd[target_bsz:target_bsz * 2, s:e]
            if v_t.shape != v_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared cosine-gated V injection failed in {module_name}: '
                    f'target/ref V range shape mismatch: target={tuple(v_t.shape)} ref={tuple(v_r.shape)}.'
                )

            v_bshd[:target_bsz, s:e] = _cosine_gated_v_injection(
                v_t, v_r, strength=cosine_v_inj
            )

        cfg['_debug_cosine_gated_v_injection_strength'] = float(cosine_v_inj)
        cfg['_debug_cosine_gated_v_injection_target'] = float(cosine_v_inj_target)
        cfg['_debug_cosine_gated_v_injection_progress'] = float(cosine_v_inj_progress)
        cfg['_debug_cosine_gated_v_injection_ramp'] = float(cosine_v_inj_ramp)
        cfg['_debug_cosine_gated_v_injection_module'] = str(module_name)
        cfg['_debug_cosine_gated_v_injection_ranges'] = list(ranges)

    # Shared variance-gated V-AdaIN.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    var_v_adain_target = _coerce_strength01(cfg.get('variance_gated_v_adain', 0.0))
    var_v_adain_progress = _coerce_strength01(cfg.get('progress', 0.0))
    var_v_adain_ramp = _triangle_ramp01(var_v_adain_progress)
    var_v_adain = _lerp(0.0, var_v_adain_target, var_v_adain_ramp)
    if var_v_adain > 0.0:
        v_bshd = v_bshd.clone()
        eps = 1e-6

        for s, e in ranges:
            v_t = v_bshd[:target_bsz, s:e]
            v_r = v_bshd[target_bsz:target_bsz * 2, s:e]
            if v_t.shape != v_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared variance-gated V-AdaIN failed in {module_name}: '
                    f'target/ref V range shape mismatch: target={tuple(v_t.shape)} ref={tuple(v_r.shape)}.'
                )

            # [B, 1, H, D], normalized per head over the head-dim/channel axis.
            v_r_mask = _reference_variance_channel_mask(v_r, eps=eps)
            v_t_adain = _adain(v_t, v_r, eps=eps)
            alpha = (v_r_mask * var_v_adain).to(v_t.dtype)

            v_bshd[:target_bsz, s:e] = v_t * (1.0 - alpha) + v_t_adain * alpha

        cfg['_debug_variance_gated_v_adain'] = float(var_v_adain)
        cfg['_debug_variance_gated_v_adain_target'] = float(var_v_adain_target)
        cfg['_debug_variance_gated_v_adain_progress'] = float(var_v_adain_progress)
        cfg['_debug_variance_gated_v_adain_ramp'] = float(var_v_adain_ramp)
        cfg['_debug_variance_gated_v_adain_module'] = str(module_name)
        cfg['_debug_variance_gated_v_adain_ranges'] = list(ranges)


    return restore(q_bshd, k_bshd, v_bshd)

def _apply_attention_output_shared_effects(
    out_t: torch.Tensor,
    out_r: torch.Tensor,
    cfg: Dict[str, Any],
    target_bsz: int,
    module_name: str,
    *,
    layout: str = 'BSD',
    token_ranges: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared post-attention output effect stack for adapter-owned attention patches.

    Adapters expose the target/reference attention outputs, layout, and optional
    architecture-specific token ranges.

    Supported layouts:
      - BSD/BSC: [batch, sequence, channels]
    """
    if not isinstance(cfg, dict) or not cfg.get('enabled', False):
        return out_t, out_r
    if not (torch.is_tensor(out_t) and torch.is_tensor(out_r)):
        return out_t, out_r
    if int(target_bsz) <= 0:
        return out_t, out_r

    layout_u = str(layout or 'BSD').upper()
    if layout_u not in ('BSD', 'BSC'):
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'unsupported layout={layout!r}.'
        )
    if out_t.ndim != 3 or out_r.ndim != 3:
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'expected target/ref outputs as rank-3, got target={tuple(out_t.shape)} ref={tuple(out_r.shape)}.'
        )
    if out_t.shape[0] != out_r.shape[0] or out_t.shape[2:] != out_r.shape[2:]:
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'target/ref output shape mismatch: target={tuple(out_t.shape)} ref={tuple(out_r.shape)}.'
        )

    seqlen = int(out_t.shape[1])
    ranges = _shared_effect_ranges(cfg, seqlen, token_ranges)

    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    post_a_target = _coerce_strength01(cfg.get('post_attention_adain_strength', 0.0))
    post_a_progress = _coerce_strength01(cfg.get('progress', 0.0))
    post_a_ramp = _triangle_ramp01(post_a_progress)
    post_a = _lerp(0.0, post_a_target, post_a_ramp)
    if post_a > 0.0:
        out_t = out_t.clone()
        for s, e in ranges:
            t_slice = out_t[:, s:e]
            r_slice = out_r[:, s:e]
            if t_slice.shape != r_slice.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared post-attention AdaIN failed in {module_name}: '
                    f'target/ref range shape mismatch: target={tuple(t_slice.shape)} ref={tuple(r_slice.shape)}.'
                )
            out_t[:, s:e] = t_slice * (1.0 - post_a) + _adain(t_slice, r_slice, eps=1e-6) * post_a

        cfg['_debug_post_attention_adain_strength'] = float(post_a)
        cfg['_debug_post_attention_adain_target'] = float(post_a_target)
        cfg['_debug_post_attention_adain_progress'] = float(post_a_progress)
        cfg['_debug_post_attention_adain_ramp'] = float(post_a_ramp)
        cfg['_debug_post_attention_adain_module'] = str(module_name)
        cfg['_debug_post_attention_adain_ranges'] = list(ranges)

    return out_t, out_r

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture detection
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVE_MODEL_ADAPTER: Any = None

# ═══════════════════════════════════════════════════════════════════════════════
# Context-refiner cap_mask patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_context_refiner_mask_modules(dm, stats):
    refiner = getattr(dm, 'context_refiner', None)
    if refiner is None:
        return 0, 0, 0
    if isinstance(refiner, (list, tuple, torch.nn.ModuleList)):
        modules = list(refiner)
    else:
        modules = [refiner]
    matched = installed = restored = 0
    for idx, module in enumerate(modules):
        if not hasattr(module, 'forward') or not callable(getattr(module, 'forward', None)):
            continue
        matched += 1
        if hasattr(module, '_untwist_orig_context_refiner_forward'):
            module.forward = module._untwist_orig_context_refiner_forward
            restored += 1
        else:
            module._untwist_orig_context_refiner_forward = module.forward
        original_forward = module._untwist_orig_context_refiner_forward

        def make_forward(orig, layer_index):
            def patched_forward(self, *args, **kwargs):
                transformer_options = kwargs.get('transformer_options', None)
                if transformer_options is None and len(args) >= 4 and isinstance(args[3], dict):
                    transformer_options = args[3]
                cfg = (
                    transformer_options.get(_TRANSFORMER_CONFIG_KEY)
                    if isinstance(transformer_options, dict) else None
                )
                forced_cap_mask = (
                    cfg.get('forced_cap_mask', None) if isinstance(cfg, dict) else None
                )
                if torch.is_tensor(forced_cap_mask):
                    args_list = list(args)
                    cap_feats = (
                        args_list[0]
                        if len(args_list) >= 1 and torch.is_tensor(args_list[0])
                        else None
                    )
                    if cap_feats is not None:
                        replacement_mask = _coerce_forced_cap_mask_for_feats(
                            forced_cap_mask, cap_feats
                        )
                        substituted = False

                        # Helper to ensure the replacement mask matches the expected attention shape
                        def _align_mask(orig_m):
                            if torch.is_tensor(orig_m):
                                try:
                                    return replacement_mask.view(orig_m.shape)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f'Context-refiner mask alignment failed: replacement={tuple(replacement_mask.shape)} '
                                        f'expected={tuple(orig_m.shape)}.'
                                    ) from exc
                            if replacement_mask.ndim == 2:
                                return replacement_mask.unsqueeze(1).unsqueeze(1)
                            return replacement_mask

                        if len(args_list) >= 2:
                            if args_list[1] is None or torch.is_tensor(args_list[1]):
                                args_list[1] = _align_mask(args_list[1])
                                substituted  = True
                        else:
                            for key in ('cap_mask', 'mask', 'x_mask'):
                                if key in kwargs and (
                                    kwargs[key] is None or torch.is_tensor(kwargs[key])
                                ):
                                    kwargs[key] = _align_mask(kwargs[key])
                                    substituted = True
                                    break
                        if substituted:
                            stats.context_refiner_calls += 1
                            return orig(*args_list, **kwargs)
                return orig(*args, **kwargs)
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, idx), module)
        installed += 1

    vp._vprint(stats,
        f'{vp._PREFIX} Context-refiner mask patch: '
        f'matched={matched} installed={installed} restored={restored}')
    return matched, installed, restored

# ═══════════════════════════════════════════════════════════════════════════════
# patchify_and_embed patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_patchify_and_embed(dm, stats):
    if hasattr(dm, '_untwist_orig_patchify'):
        dm.patchify_and_embed = dm._untwist_orig_patchify
    else:
        dm._untwist_orig_patchify = dm.patchify_and_embed
    original = dm._untwist_orig_patchify

    def patched(self, x, cap_feats, cap_mask, t, num_tokens,
                ref_latents=[], ref_contexts=[], siglip_feats=[],
                transformer_options={}, *args, **kwargs):

        cfg_pre = (
            transformer_options.get(_TRANSFORMER_CONFIG_KEY)
            if isinstance(transformer_options, dict) else None
        )
        forced_cap_mask = (
            cfg_pre.get('forced_cap_mask', None) if isinstance(cfg_pre, dict) else None
        )

        if torch.is_tensor(forced_cap_mask):
            cap_mask = forced_cap_mask.to(device=cap_feats.device)
            if cap_mask.ndim == 1:
                cap_mask = cap_mask.view(1, -1)
            if cap_mask.ndim > 0 and int(cap_mask.shape[0]) != int(cap_feats.shape[0]):
                cap_mask = _repeat_to_batch(cap_mask, int(cap_feats.shape[0]))
            if cap_mask.ndim == 2 and int(cap_mask.shape[1]) != int(cap_feats.shape[1]):
                cap_mask = _pad_or_truncate_tokens(cap_mask, int(cap_feats.shape[1]))

        result = original(x, cap_feats, cap_mask, t, num_tokens, *args,
                          ref_latents=ref_latents, ref_contexts=ref_contexts,
                          siglip_feats=siglip_feats,
                          transformer_options=transformer_options, **kwargs)
        stats.patchify_calls += 1

        try:
            img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = result
            cfg = transformer_options.get(_TRANSFORMER_CONFIG_KEY)
            if not cfg or not cfg.get('enabled'):
                return result

            cfg['axes_dims'] = list(getattr(self, 'axes_dims', []))
            cfg['head_dim']  = (int(getattr(self, 'dim', 0))
                                // max(1, int(getattr(self, 'n_heads', 1))))
            cfg['seq_len']   = int(img.shape[1])
            cfg['patch_size']= int(getattr(self, 'patch_size', 2))
            try:
                cfg['rope_theta'] = float(
                    getattr(getattr(self, 'rope_embedder', None), 'theta', 10000.0)
                )
            except Exception as exc:
                raise RuntimeError('patchify_and_embed failed: rope_theta could not be read.') from exc

            p = cfg['patch_size']
            target_range = target_text_range = None
            ref_ranges:      List[Tuple[int,int]] = []
            ref_real_ranges: List[Tuple[int,int]] = []

            if timestep_zero_index:
                target_range = tuple(int(v) for v in timestep_zero_index[0])
                if len(timestep_zero_index) > 1:
                    target_text_range = tuple(int(v) for v in timestep_zero_index[1])
            else:
                try:
                    cap0 = int(cap_size[0]) if isinstance(cap_size, (list, tuple)) else int(cap_size)
                except Exception as exc:
                    raise RuntimeError('patchify_and_embed failed: cap_size could not be converted to int.') from exc
                target_text_range = (0, cap0) if cap0 > 0 else None
                target_range = (max(0, cap0), int(img.shape[1]))

            real_range = target_range
            if target_range is not None:
                ts, te = int(target_range[0]), int(target_range[1])
                try:
                    real_tok   = (x.shape[-2] // p) * (x.shape[-1] // p)
                    real_range = (ts, min(ts + real_tok, te))
                except Exception as exc:
                    raise RuntimeError('patchify_and_embed failed: target real token range could not be computed.') from exc
                cfg['target_real_range'] = real_range
                ref_ranges.append((ts, te))
                ref_real_ranges.append(real_range)

            cfg.update({
                'ref_k_ranges':           ref_ranges,
                'ref_real_ranges':        ref_real_ranges,
                'target_range':           target_range,
                'target_text_range':      target_text_range,
                'target_qk_adain_ranges':
                    [cfg.get('target_real_range', target_range)] if target_range else [],
            })

            forced_mask_for_joint = cfg.get('forced_cap_mask', None)
            if torch.is_tensor(forced_mask_for_joint):
                joint_mask = _build_joint_additive_mask_from_cap_mask(
                    forced_mask_for_joint, int(img.shape[1]),
                    target_text_range, img.device, dtype=torch.float32,
                )
                if torch.is_tensor(joint_mask):
                    cfg['forced_joint_x_mask'] = joint_mask
                    if mask is None:
                        mask   = joint_mask
                        result = (img, mask, img_size, cap_size, freqs_cis, timestep_zero_index)

            transformer_options[_TRANSFORMER_CONFIG_KEY] = cfg
        except Exception as exc:
            raise RuntimeError('patchify_and_embed strict metadata patch failed.') from exc

        return result

    dm.patchify_and_embed = types.MethodType(patched, dm)
    vp._vprint(stats, f'{vp._PREFIX} patchify_and_embed patched.')

# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI Nodes — split RF inversion from Untwisting RoPE
# ═══════════════════════════════════════════════════════════════════════════════

def _adapter_helpers() -> Dict[str, Any]:
    return {
        'prefix': vp._PREFIX,
        'config_key': _TRANSFORMER_CONFIG_KEY,
        'lerp': _lerp,
        'cross_batch_adain_qk': _cross_batch_adain_qk,
        'build_frequency_scale_vector': _build_frequency_scale_vector,
        'apply_qkv_shared_effects': _apply_qkv_shared_effects,
        'apply_attention_output_shared_effects': _apply_attention_output_shared_effects,
        'print_rope_scale_debug': vp._untwist_print_rope_scale_debug,
        'print_rope_scale_debug_from_cfg': (
            lambda stats, cfg, module_name, device, dtype:
                vp._untwist_print_rope_scale_debug_from_cfg(
                    stats, cfg, module_name, device, dtype,
                    _build_frequency_scale_vector,
                )
        ),
        'patch_context_refiner_mask_modules': _patch_context_refiner_mask_modules,
        'patch_patchify_and_embed': _patch_patchify_and_embed,
    }

def _prepare_reference_conditioning_for_adapter(
    adapter: Any,
    ref_conditioning: Any,
    dm: Any,
    device,
    dtype,
    stats: Optional[vp._RuntimeStats] = None,
    label: str = '',
) -> Tuple[Any, str]:
    fn = getattr(adapter, 'prepare_reference_conditioning', None)
    if not callable(fn):
        raise RuntimeError(
            f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
            'prepare_reference_conditioning.'
        )
    return fn(ref_conditioning, dm, device, dtype, stats, label=label, helpers=_adapter_helpers())

class UnofficialExtensions:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('UNTWISTING_ROPE_EXTENSIONS',)
    RETURN_NAMES = ('unofficial_extensions',)
    FUNCTION = 'build'
    DESCRIPTION = (
        'Optional unofficial toggles for Untwisting RoPE. '
        'These are intentionally separated from the main paper settings.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'post_attention_adain_strength': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Blend strength for matching the target attention output to the reference attention output.',
                }),
                'axis0_rope_mode': (['default', 'match_axes', 'constant'], {
                    'default': 'match_axes',
                    'tooltip': (
                        'Axis-0 RoPE behavior.\n'
                        'default -> Its values are equal to low_scale;\n'
                        'match_axes -> makes axis 0 use the same curve as axes 1+;\n'
                        'constant -> uses axis0_rope_scale.'
                    ),
                }),
                'axis0_rope_scale': ('FLOAT', {
                    'default': 1.0,
                    'min': 0.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'RoPE scale used only when axis0_rope_mode = constant.',
                }),
                'cosine_gated_v_injection': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Only aligned target/reference V tokens receive reference injection.',
                }),
                'variance_gated_v_adain': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': (
                        'Applies V AdaIN only on high-reference-variance channels. '
                    ),
                }),
                'key_subspace_alignment': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': (
                        'Projects target keys onto reference keys to restrict routing '
                        'towards the reference key subspace.'
                    ),
                }),
            },
        }

    def build(
        self,
        post_attention_adain_strength: float = 0.0,
        axis0_rope_mode: str = 'default',
        axis0_rope_scale: float = 0.0,
        cosine_gated_v_injection: float = 0.0,
        variance_gated_v_adain: float = 0.0,
        key_subspace_alignment: float = 0.0,
    ):
        return ({
            'post_attention_adain_strength': _coerce_strength01(post_attention_adain_strength),
            'axis0_rope_mode': _coerce_axis0_rope_mode(axis0_rope_mode),
            'axis0_rope_scale': _coerce_axis0_rope_scale(axis0_rope_scale, default=0.0),
            'cosine_gated_v_injection': _coerce_strength01(cosine_gated_v_injection),
            'variance_gated_v_adain': _coerce_strength01(variance_gated_v_adain),
            'key_subspace_alignment': _coerce_strength01(key_subspace_alignment),
        },)

class UntwistingRoPE:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'patch'
    DESCRIPTION = (
        'Patches supported attention/RoPE modules and uses the RFInversion LATENT trajectory. '
        'RF inversion settings live on the LATENT; the sampler sigma schedule is captured internally.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'rf_inversion': ('LATENT',),
                'beta': ('FLOAT', {
                    'default': 50.0,
                    'min': 0.01,
                    'max': 100.0,
                    'step': 0.01,
                    'tooltip': 'Controls the steepness of the frequency scale curve. Higher values prevent the model from copying the reference image too closely.'
                }),
                'high_scale_start': ('FLOAT', {
                    'default': 1.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to high-frequency components. The higher the value, the more the final image will resemble the structure of the reference image.'
                }),
                'high_scale_end': ('FLOAT', {
                    'default': 0.00,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to high-frequency components. The higher the value, the more the final image will resemble the structure of the reference image.'
                }),
                'low_scale_start': ('FLOAT', {
                    'default': 1.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to low-frequency components. Controls the strength of the style image.'
                }),
                'low_scale_end': ('FLOAT', {
                    'default': 3.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to low-frequency components. Controls the strength of the style image.'
                }),
                'adain_strength': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'AdaIN aligns the target style statistics toward the reference.'
                }),
                'blocks': ('STRING', {
                    'default': '0-999',
                    'tooltip': 'Specify block ranges to patch, e.g -> 0-8, 28-37'
                }),
                'verbose': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Enable verbose logging.'
                }),
            },
            'optional': {
                'unofficial_extensions': ('UNTWISTING_ROPE_EXTENSIONS',),
            },
        }

    def patch(
        self,
        model,
        beta: float,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        blocks: str,
        adain_strength: float,
        verbose: bool = False,
        rf_inversion: Optional[Dict[str, Any]] = None,
        unofficial_extensions: Optional[Dict[str, Any]] = None,
    ):
        rf_active, rf_cfg, rf_state, ref_clean_cpu, ref_conditioning, rf_source = _rf_latent_get_config(rf_inversion)
        node_verbose = vp._coerce_bool(verbose)
        rf_verbose = vp._coerce_bool(rf_cfg.get('verbose', False))
        stats = vp._RuntimeStats(verbose=node_verbose, rf_verbose=rf_verbose)
        stats.rf_prefix = vp._RF_PREFIX
        debug_store = _rf_new_debug_store()

        ext_cfg = unofficial_extensions if isinstance(unofficial_extensions, dict) else {}
        cosine_gated_v_injection = _coerce_strength01(ext_cfg.get('cosine_gated_v_injection', 0.0))
        variance_gated_v_adain = _coerce_strength01(ext_cfg.get('variance_gated_v_adain', 0.0))
        post_attention_adain_strength = _coerce_strength01(ext_cfg.get('post_attention_adain_strength', 0.0))
        axis0_rope_mode = _coerce_axis0_rope_mode(
            ext_cfg.get('axis0_rope_mode', None),
            legacy_scale=ext_cfg.get('axis0_rope_scale', None),
        )
        axis0_rope_scale = _coerce_axis0_rope_scale(ext_cfg.get('axis0_rope_scale', 0.0), default=0.0)
        key_subspace_alignment = _coerce_strength01(ext_cfg.get('key_subspace_alignment', 0.0))

        if rf_active:
            stats.rf_sigma_cache = rf_state.get('cache', {}) if isinstance(rf_state.get('cache', {}), dict) else {}
            stats.rf_schedule_built = bool(rf_state.get('schedule_built', False))
            stats.parameterization = str(rf_inversion.get('untwist_rf_parameterization', 'unknown')) if isinstance(rf_inversion, dict) else 'unknown'
            debug_store['cache'] = stats.rf_sigma_cache
            debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or []) if isinstance(rf_state, dict) else []
            debug_store['built_sigmas'] = list(rf_state.get('schedule_sorted') or []) if isinstance(rf_state, dict) else []
            debug_store['parameterization'] = stats.parameterization

        vp._vprint(stats, f'\n{vp._PREFIX} ═══════════════════════════════════════')
        vp._vprint(stats, f'{vp._PREFIX} PATCH START  (split nodes: RFInversion + UntwistingRoPE)')
        vp._vprint(stats, f'{vp._PREFIX} ═══════════════════════════════════════')
        vp._vprint(stats, f'{vp._PREFIX} beta={beta}')
        vp._vprint(stats, f'{vp._PREFIX} high_scale: {high_scale_start:.3f} → {high_scale_end:.3f}')
        vp._vprint(stats, f'{vp._PREFIX} low_scale:  {low_scale_start:.3f} → {low_scale_end:.3f}')
        vp._vprint(stats,
            f'{vp._PREFIX} blocks: {blocks if blocks.strip() else "all"}  '
            f'adain={adain_strength:.2f}  '
            f'unofficial: '
            f'cosine_gated_v_injection={cosine_gated_v_injection:.2f}  '
            f'variance_gated_v_adain={variance_gated_v_adain:.2f}  '
            f'post_attention_adain_strength={post_attention_adain_strength:.2f}  '
            f'axis0_rope_mode={axis0_rope_mode}  '
            f'axis0_rope_scale={axis0_rope_scale:.3f}  '
            f'key_subspace_alignment={key_subspace_alignment:.2f}'
        )
        vp._vprint(stats, f'{vp._PREFIX} RF latent connected: {rf_active}  source={rf_source}')
        if rf_active:
            vp._vprint(stats, f'{vp._PREFIX} RF trajectory: {_rf_format_trajectory_config(rf_cfg)}')
            vp._vprint(stats, f'{vp._PREFIX} RF schedule: captured from sampler at runtime; no SIGMAS input')

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        if rf_active:
            setattr(model_clone, '_untwisting_rope_rf_state', rf_state)
            setattr(model_clone, '_untwisting_rope_rf_config', rf_cfg)

        model_info = vp._rf_model_identity(model_clone)
        adapter = _select_model_adapter(model_clone, model_info)
        dm = _safe_get_diffusion_model(model_clone, adapter)
        vp._vprint(stats, f'{vp._PREFIX} Diffusion model type: {type(dm).__name__}')

        global _ACTIVE_MODEL_ADAPTER
        previous_adapter = _ACTIVE_MODEL_ADAPTER
        _ACTIVE_MODEL_ADAPTER = adapter
        try:
            patch_fn = getattr(adapter, 'patch_attention_modules', None)
            if not callable(patch_fn):
                raise RuntimeError(
                    f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
                    'patch_attention_modules.'
                )

            result = patch_fn(dm, stats, _adapter_helpers())
            if not isinstance(result, tuple):
                raise RuntimeError(
                    f'Unexpected adapter patch return from {type(adapter).__name__}: {result!r}'
                )
            if len(result) == 4:
                matched, installed, restored, patched_names = result
            elif len(result) == 3:
                matched, installed, restored = result
                patched_names = []
            else:
                raise RuntimeError(
                    f'Unexpected adapter patch return from {type(adapter).__name__}: {result!r}'
                )

            disp_name = getattr(adapter, 'DISPLAY_NAME', type(adapter).__name__)

            vp._vprint(stats, f'{vp._PREFIX} {disp_name} attention patch: matched={matched} installed={installed} restored={restored}')
            for n in patched_names:
                vp._vprint(stats, f'{vp._PREFIX}   - {n}')

            if installed == 0:
                raise RuntimeError(f'{vp._PREFIX} No {disp_name} attention blocks were patched.')
        finally:
            _ACTIVE_MODEL_ADAPTER = previous_adapter

        old_wrapper = model_clone.model_options.get('model_function_wrapper', None)

        parsed_blocks = _parse_active_blocks(blocks)

        def model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            stats.wrapper_calls += 1
            call_n = stats.wrapper_calls

            input_x = args['input']
            timestep = args['timestep']
            c = args['c'].copy()
            cond_or_uncond = args.get('cond_or_uncond', None)
            to = c.get('transformer_options', {}).copy()
            _maybe_install_untwist_attention_override(to)

            sigma = _sigma_from_timestep(timestep)

            if rf_active and isinstance(rf_state, dict) and rf_state.get('sigma_probe_active', False):
                rf_state['probe_model_calls'] = int(rf_state.get('probe_model_calls', 0)) + 1
                debug_store['probe_model_calls'] = int(rf_state['probe_model_calls'])
                _rf_record_probe_sigma(rf_state, debug_store, sigma)
                return torch.zeros_like(input_x)

            progress = _sigma_to_progress(timestep, list(rf_state['sampler_sigmas']))
            target_b = int(input_x.shape[0])

            cfg: Dict[str, Any] = {
                'enabled': True,
                'beta': float(beta),
                'high_scale_start': float(high_scale_start),
                'high_scale_end': float(high_scale_end),
                'low_scale_start': float(low_scale_start),
                'low_scale_end': float(low_scale_end),
                'active_blocks': parsed_blocks,
                'apply_adain': True,
                'adain_strength': float(adain_strength),
                'cosine_gated_v_injection': cosine_gated_v_injection,
                'variance_gated_v_adain': variance_gated_v_adain,
                'post_attention_adain_strength': post_attention_adain_strength,
                'axis0_rope_mode': axis0_rope_mode,
                'axis0_rope_scale': axis0_rope_scale,
                'key_subspace_alignment': key_subspace_alignment,
                'cross_batch_target_batch': target_b if rf_active else 0,
                'progress': progress,
                'sigma': sigma,
                'wrapper_call': call_n,
                '_rope_scale_debug_printed': False,
            }
            default_cfg = getattr(adapter, 'default_runtime_cfg', None)
            if not callable(default_cfg):
                raise RuntimeError(
                    f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
                    'default_runtime_cfg.'
                )
            cfg.update(default_cfg(dm))
            to[_TRANSFORMER_CONFIG_KEY] = cfg

            input_for_model = input_x
            timestep_for_model = timestep
            ref_noisy = None
            sigma_key = round(float(sigma), 6)
            rf_cache_hit = False
            rf_cond_mode = 'not-connected'
            ref_mode = 'target-only'
            should_print = vp._coerce_bool(getattr(stats, 'verbose', False))

            if rf_active and torch.is_tensor(ref_clean_cpu):
                try:
                    ref_clean = ref_clean_cpu.to(device=input_x.device, dtype=input_x.dtype)
                    ref = _repeat_to_batch(ref_clean, target_b)

                    if not rf_state.get('schedule_built', False) and rf_state.get('sampler_sigmas', None) is not None:
                        effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                            adapter, ref_conditioning, dm, input_x.device,
                            c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                            stats, label='UntwistingRoPE',
                        )
                        rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                        rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                        rf_ref_clean = _repeat_to_batch(ref_clean, target_b)
                        sampler_sigmas = list(rf_state['sampler_sigmas'])
                        built_cache, eps, sorted_sigmas, cache_key, persistent_hit = _rf_ensure_trajectory_cache(
                            rf_inversion=rf_inversion,
                            rf_state=rf_state,
                            rf_cfg=rf_cfg,
                            ref_clean_cpu=ref_clean_cpu,
                            ref_clean_for_build=rf_ref_clean,
                            ref_conditioning=ref_conditioning,
                            sampler_sigmas=sampler_sigmas,
                            target_b=target_b,
                            rf_cond_mode=rf_cond_mode,
                            apply_model_fn=_make_raw_velocity_apply_model_fn(apply_model),
                            base_model_kwargs=rf_kwargs,
                            device=input_x.device,
                            dtype=input_x.dtype,
                            stats=stats,
                            preview_callback_factory=lambda: _rf_make_preview_callback(model_clone, max(1, len(sampler_sigmas) - 1)),
                            debug_store=debug_store,
                            parameterization=stats.parameterization,
                        )
                        ref_mode = 'RF sampler-sigma trajectory (persistent-cache hit)' if persistent_hit else 'RF sampler-sigma trajectory (built)'
                        stats.rf_sigma_cache = rf_state['cache']
                        stats.rf_eps = rf_state['eps']
                        stats.rf_schedule_built = True
                        stats.rf_step_count = max(0, len(sorted_sigmas) - 1)
                    elif rf_state.get('schedule_built', False):
                        ref_mode = 'RF sampler-sigma trajectory (cached)'
                    else:
                        raise RuntimeError(
                            'UntwistingRoPE failed: sampler sigma schedule was not captured and no RF trajectory was built. '
                            'SAMPLER_SAMPLE must run before UntwistingRoPE model calls.'
                        )

                    cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                    cached, used_sigma_key, cache_lookup = _rf_cache_lookup(cache, sigma_key, allow_nearest=True)
                    if cached is None:
                        raise RuntimeError(
                            f'UntwistingRoPE failed: no RF cache entry for sigma={sigma_key:.6f}.'
                        )
                    rf_cache_hit = True
                    rf_state['last_cache_lookup'] = cache_lookup
                    rf_state['last_cache_key'] = float(used_sigma_key)
                    debug_store['last_cache_lookup'] = cache_lookup
                    debug_store['last_cache_key'] = float(used_sigma_key)

                    ref_noisy = _repeat_to_batch(cached.to(device=input_x.device, dtype=input_x.dtype), target_b)

                    if ref_noisy.shape[-2:] == input_x.shape[-2:]:
                        input_for_model = torch.cat([input_x, ref_noisy], dim=0)
                        try:
                            if (torch.is_tensor(timestep)
                                    and timestep.ndim > 0
                                    and int(timestep.shape[0]) == target_b):
                                timestep_for_model = torch.cat([timestep, timestep], dim=0)
                            else:
                                timestep_for_model = _repeat_to_batch(timestep, target_b * 2)
                        except Exception as exc:
                            raise RuntimeError('UntwistingRoPE failed while duplicating timestep for reference batch.') from exc

                        effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                            adapter, ref_conditioning, dm, input_x.device,
                            c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                            stats, label='UntwistingRoPEMerge',
                        )
                        c, forced_cap_mask = _merge_reference_conditioning_into_c(c, effective_ref_conditioning, target_b)
                        cfg['adapter_ref_conditioning_status'] = adapter_ref_status
                        cfg['forced_cap_mask'] = forced_cap_mask.to(device=input_x.device)
                        cfg['cross_batch_target_batch'] = target_b

                        try:
                            if isinstance(cond_or_uncond, list):
                                cond_or_uncond = cond_or_uncond + cond_or_uncond
                        except Exception as exc:
                            raise RuntimeError('UntwistingRoPE failed while duplicating cond_or_uncond metadata.') from exc
                    else:
                        raise RuntimeError(
                            f'UntwistingRoPE failed: spatial mismatch input_x={tuple(input_x.shape[-2:])} '
                            f'ref_noisy={tuple(ref_noisy.shape[-2:])}. '
                            f'Make sure the resolution of the reference image fed into the RF inversion node matches the final image resolution (same width and height).'
                        )
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError('UntwistingRoPE RF latent preparation failed.') from exc

            c['transformer_options'] = to

            if old_wrapper is not None:
                raw_result = old_wrapper(apply_model, {
                    'input': input_for_model,
                    'timestep': timestep_for_model,
                    'c': c,
                    'cond_or_uncond': cond_or_uncond,
                })
            else:
                raw_result = apply_model(input_for_model, timestep_for_model, **c)

            if should_print:
                vp._untwist_print_rope_scale_debug_from_cfg(
                    stats, cfg, 'model_wrapper', input_x.device, input_x.dtype,
                    _build_frequency_scale_vector,
                )

            if (rf_active
                    and ref_noisy is not None
                    and torch.is_tensor(raw_result)
                    and raw_result.shape[0] >= target_b * 2):
                target_pred = raw_result[:target_b]
                ref_pred = raw_result[target_b:target_b * 2]

                try:
                    ref_xsigma = ref_noisy[:target_b]
                    debug_store['pred_cache'][sigma_key] = ref_pred[:1].detach().clone()
                    debug_store['xhat_cache'][sigma_key] = (ref_xsigma - float(sigma) * ref_pred)[:1].detach().clone()
                    debug_store['xhat_plus_cache'][sigma_key] = (ref_xsigma + float(sigma) * ref_pred)[:1].detach().clone()
                except Exception as exc:
                    raise RuntimeError(
                        f'UntwistingRoPE failed while caching RF debug latents at σ={float(sigma):.6f}.'
                    ) from exc

                return target_pred

            return raw_result

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        vp._untwist_print_patch_complete(stats, rf_active, adapter)

        return (model_clone,)

from .rf_inversion import (
    RFInversion,
    _RF_LAST_DEBUG_STORE,
    _append_conditioning_status,
    _rf_cache_lookup,
    _rf_ensure_trajectory_cache,
    _rf_record_probe_sigma,
    _rf_format_trajectory_config,
    _rf_latent_get_config,
    _rf_make_preview_callback,
    _rf_new_debug_store,
    _make_raw_velocity_apply_model_fn,
    _sigma_from_timestep,
    _sigma_to_progress,
)
