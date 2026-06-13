from __future__ import annotations
import inspect
import math
import hashlib
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import latent_preview

from . import verbose_prints as vp
from .sdpa_fix import install_optimized_attention_override as _maybe_install_untwist_attention_override
from .runtime import (
    _select_model_adapter,
    _safe_get_diffusion_model,
    _repeat_to_batch,
    _clone_model_options,
    _build_rf_conditioning_kwargs,
    _prepare_reference_conditioning_for_adapter,
)

# Module-level debug store. Comfy/KSampler may clone or pass model objects
# through different instances, so the export node reads this if the model-local
_RF_LAST_DEBUG_STORE: Dict[str, Any] = {
    'cache': {},
    'sampler_sigmas': None,
    'built_sigmas': None,
    'run_count': 0,
}

# Persistent RF trajectory cache shared across prompt executions.
# Keyed by reference latent, reference conditioning, sigma schedule, RF mode,
# and RF parameters. Values are stored on CPU to avoid pinning VRAM.
_RF_PERSISTENT_TRAJECTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_RF_PERSISTENT_CACHE_MAX_ITEMS = 4
# ═══════════════════════════════════════════════════════════════════════════════
# Persistent RF cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_update_tensor(h: "hashlib._Hash", t: torch.Tensor, full: bool = True) -> None:
    td = t.detach().to(device='cpu').contiguous()
    h.update(str(tuple(td.shape)).encode('utf-8'))
    h.update(str(td.dtype).encode('utf-8'))
    if full:
        h.update(td.numpy().tobytes())
    else:
        flat = td.flatten()
        if flat.numel() > 4096:
            flat = flat[torch.linspace(0, flat.numel() - 1, 4096).long()]
        h.update(flat.numpy().tobytes())

def _hash_any(obj: Any, h: Optional["hashlib._Hash"] = None, depth: int = 0) -> str:
    if h is None:
        h = hashlib.sha1()
    if depth > 12:
        h.update(b'<maxdepth>')
        return h.hexdigest()
    if torch.is_tensor(obj):
        h.update(b'TENSOR')
        _hash_update_tensor(h, obj, full=True)
    elif isinstance(obj, dict):
        h.update(b'DICT')
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if str(k) == 'transformer_options':
                continue
            h.update(str(k).encode('utf-8'))
            _hash_any(obj[k], h, depth + 1)
    elif isinstance(obj, (list, tuple)):
        h.update(b'LIST' if isinstance(obj, list) else b'TUPLE')
        h.update(str(len(obj)).encode('utf-8'))
        for v in obj:
            _hash_any(v, h, depth + 1)
    elif obj is None:
        h.update(b'NONE')
    else:
        h.update(repr(obj).encode('utf-8', errors='ignore'))
    return h.hexdigest()

_RF_CONFIG_NON_TRAJECTORY_KEYS = {
    'verbose',
    'apply_model_output',
    'model_info',
}

_RF_BUILD_FIXED_KWARGS = {
    'ref_clean',
    'sampler_sigmas',
    'apply_model_fn',
    'base_model_kwargs',
    'stats',
    'eps',
    'preview_callback',
}

def _coerce_unit_interval(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    return max(0.0, min(1.0, v))

def _rf_trajectory_config_for_cache(rf_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the RF parameters that affect the trajectory/cache."""
    src = dict(rf_cfg or {})
    mode = str(src.get('rf_mode', 'rf_gamma') or 'rf_gamma')
    return {
        'rf_mode': mode,
        'gamma': float(src.get('gamma', 0.5)),
        'pmi_alpha': _coerce_unit_interval(src.get('pmi_alpha', 0.4), 0.4),
        'otip_strength': _coerce_otip_strength(src.get('otip_strength', 0.0)),
        'otip_phase': _coerce_otip_phase(src.get('otip_phase', 1.0)),
        'otip_clip_norm': _coerce_otip_clip_norm(src.get('otip_clip_norm', 10.0)),
        'otip_respect_model_norm': False,
        'seed': int(src.get('seed', 42)),
    }

def _rf_build_kwargs_from_config(rf_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract only RF config values accepted by the trajectory builder.

    This is signature-driven so callers outside rf_inversion.py do not need to
    change when a new RF parameter is added to the builder.
    """
    cfg = _rf_trajectory_config_for_cache(rf_cfg)
    params = inspect.signature(_rf_build_cache_from_sampler_sigmas).parameters
    return {
        name: cfg[name]
        for name in params.keys()
        if name not in _RF_BUILD_FIXED_KWARGS and name in cfg
    }

def _rf_format_trajectory_config(rf_cfg: Optional[Dict[str, Any]]) -> str:
    cfg = _rf_trajectory_config_for_cache(rf_cfg)
    return '  '.join(f'{k}={cfg[k]}' for k in sorted(cfg.keys()))

def _make_rf_persistent_key(
    ref_clean: torch.Tensor,
    ref_conditioning: Any,
    sampler_sigmas: List[float],
    target_b: int,
    cond_mode: str,
    rf_config: Optional[Dict[str, Any]] = None,
    **legacy_rf_params: Any,
) -> str:
    h = hashlib.sha1()

    h.update(str(tuple(ref_clean.shape)).encode('utf-8'))
    _hash_update_tensor(h, ref_clean, full=True)

    h.update(_hash_any(ref_conditioning).encode('utf-8'))

    h.update(str([round(float(s), 8) for s in sampler_sigmas]).encode('utf-8'))
    h.update(str(int(target_b)).encode('utf-8'))
    h.update(str(cond_mode).encode('utf-8'))

    cfg = dict(rf_config or {})
    cfg.update({k: v for k, v in legacy_rf_params.items() if v is not None})
    h.update(_hash_any(_rf_trajectory_config_for_cache(cfg)).encode('utf-8'))

    return h.hexdigest()

def _cache_to_cpu(cache: Dict[float, torch.Tensor]) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.detach().to(device='cpu').clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _cache_to_device(cache: Dict[float, torch.Tensor], device, dtype) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.to(device=device, dtype=dtype).detach().clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _put_persistent_rf_cache(key: str, entry: Dict[str, Any]) -> None:
    _RF_PERSISTENT_TRAJECTORY_CACHE[key] = entry
    while len(_RF_PERSISTENT_TRAJECTORY_CACHE) > _RF_PERSISTENT_CACHE_MAX_ITEMS:
        oldest = next(iter(_RF_PERSISTENT_TRAJECTORY_CACHE.keys()))
        _RF_PERSISTENT_TRAJECTORY_CACHE.pop(oldest, None)

def _rf_ensure_trajectory_cache(
    *,
    rf_inversion: Optional[Dict[str, Any]],
    rf_state: Dict[str, Any],
    rf_cfg: Dict[str, Any],
    ref_clean_cpu: torch.Tensor,
    ref_clean_for_build: torch.Tensor,
    ref_conditioning: Any,
    sampler_sigmas: List[float],
    target_b: int,
    rf_cond_mode: str,
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    stats: Optional[vp._RuntimeStats],
    preview_callback_factory: Optional[Callable[[], Optional[Callable]]] = None,
    verbose_flag: Optional[bool] = None,
    debug_store: Optional[Dict[str, Any]] = None,
    parameterization: Optional[str] = None,
) -> Tuple[Dict[float, torch.Tensor], torch.Tensor, List[float], str, bool]:
    """
    Own persistent RF trajectory cache lookup/build/update in rf_inversion.py.

    UntwistingRoPE should call this instead of manually constructing keys,
    checking _RF_PERSISTENT_TRAJECTORY_CACHE, or listing RF trajectory params.
    """
    sampler_sigmas = list(sampler_sigmas)
    cache_key = _make_rf_persistent_key(
        ref_clean=ref_clean_cpu.detach().to(device='cpu'),
        ref_conditioning=ref_conditioning,
        sampler_sigmas=sampler_sigmas,
        target_b=target_b,
        cond_mode=rf_cond_mode,
        rf_config=rf_cfg,
    )

    if verbose_flag is None:
        verbose_flag = vp._coerce_bool(getattr(stats, 'rf_verbose', False)) if stats else False

    cached_entry = _RF_PERSISTENT_TRAJECTORY_CACHE.get(cache_key)
    if cached_entry is not None:
        built_cache = _cache_to_device(cached_entry['cache'], device, dtype)
        eps = cached_entry['eps'].to(device=device, dtype=dtype)
        sorted_sigmas = list(cached_entry['built_sigmas'])
        persistent_hit = True
        vp._rf_print_persistent_cache_hit(verbose_flag, cache_key, built_cache)
    else:
        persistent_hit = False
        preview_callback = rf_state.get('preview_callback', None)
        if preview_callback is None and preview_callback_factory is not None:
            preview_callback = preview_callback_factory()
            rf_state['preview_callback'] = preview_callback
        vp._rf_print_persistent_cache_miss(verbose_flag, cache_key)
        built_cache, eps, sorted_sigmas = _rf_build_cache_from_sampler_sigmas(
            ref_clean=ref_clean_for_build,
            sampler_sigmas=sampler_sigmas,
            apply_model_fn=apply_model_fn,
            base_model_kwargs=base_model_kwargs,
            stats=stats,
            eps=rf_state['eps'].to(device=device, dtype=dtype)
                if torch.is_tensor(rf_state.get('eps', None)) else None,
            preview_callback=preview_callback,
            **_rf_build_kwargs_from_config(rf_cfg),
        )
        _put_persistent_rf_cache(cache_key, {
            'cache': _cache_to_cpu(built_cache),
            'eps': eps.detach().to(device='cpu').clone(),
            'built_sigmas': list(sorted_sigmas),
            'rf_config': _rf_trajectory_config_for_cache(rf_cfg),
        })
        try:
            import time
            import tqdm
            if hasattr(tqdm.tqdm, '_instances'):
                now = time.time()
                for instance in list(getattr(tqdm.tqdm, '_instances', [])):
                    instance.start_t = now
                    instance.last_print_t = now
        except Exception:
            pass

    rf_state['cache'] = built_cache
    rf_state['eps'] = eps.detach().clone()
    rf_state['schedule_sorted'] = list(sorted_sigmas)
    rf_state['schedule_built'] = True
    rf_state['persistent_cache_key'] = cache_key
    rf_state['persistent_cache_hit'] = bool(persistent_hit)

    if isinstance(rf_inversion, dict):
        rf_inversion['untwist_rf_cache'] = _cache_to_cpu(built_cache)
        rf_inversion['untwist_rf_eps'] = eps.detach().to(device='cpu').clone()
        rf_inversion['untwist_rf_sigmas'] = list(sorted_sigmas)
        rf_inversion['untwist_rf_state'] = rf_state

    if debug_store is not None:
        debug_store['cache'] = rf_state['cache']
        debug_store['sampler_sigmas'] = sampler_sigmas
        debug_store['built_sigmas'] = list(sorted_sigmas)
        debug_store['run_count'] = int(rf_state.get('run_count', 0))
        debug_store['persistent_cache_key'] = cache_key
        debug_store['persistent_cache_hit'] = bool(persistent_hit)
        if parameterization is not None:
            debug_store['parameterization'] = parameterization

    return built_cache, eps, list(sorted_sigmas), cache_key, bool(persistent_hit)

# ═══════════════════════════════════════════════════════════════════════════════
# Raw transformer velocity path
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_comfy_convert_tensor(extra: Any, dtype: torch.dtype, device: torch.device) -> Any:
    """Mirror ComfyUI model_base.convert_tensor for raw model calls."""
    if hasattr(extra, "dtype"):
        if extra.dtype != torch.int and extra.dtype != torch.long:
            extra = comfy.model_management.cast_to_device(extra, device, dtype)
        else:
            extra = comfy.model_management.cast_to_device(extra, device, None)
    return extra

def _rf_validate_raw_velocity_model(base_model: Any) -> None:
    model_type = getattr(base_model, 'model_type', None)
    model_type_name = str(getattr(model_type, 'name', model_type))
    if model_type_name not in {'FLOW', 'FLUX'}:
        raise RuntimeError(
            'RFInversion raw-velocity mode only supports Comfy FLOW/FLUX models. '
            f'Got model_type={model_type_name!r}. No x0/denoised fallback is enabled.'
        )
    if not hasattr(base_model, 'diffusion_model'):
        raise RuntimeError('RFInversion raw-velocity mode failed: BaseModel.diffusion_model is missing.')
    if not hasattr(base_model, 'model_sampling'):
        raise RuntimeError('RFInversion raw-velocity mode failed: BaseModel.model_sampling is missing.')

def _rf_raw_transformer_velocity(
    apply_model: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    c_concat: Optional[torch.Tensor] = None,
    c_crossattn: Optional[torch.Tensor] = None,
    control: Any = None,
    transformer_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Strict raw-velocity equivalent of ComfyUI BaseModel._apply_model.
    """
    base_model = getattr(apply_model, '__self__', None)
    if base_model is None:
        raise RuntimeError(
            'RFInversion raw-velocity mode requires a bound Comfy BaseModel.apply_model. '
            'No x0/denoised fallback is enabled.'
        )
    _rf_validate_raw_velocity_model(base_model)

    if not torch.is_tensor(x):
        raise RuntimeError('RFInversion raw-velocity mode expected tensor input x.')
    if not torch.is_tensor(t):
        raise RuntimeError('RFInversion raw-velocity mode expected tensor timestep/sigma.')

    sigma = t
    xc = base_model.model_sampling.calculate_input(sigma, x)

    if c_concat is not None:
        xc = torch.cat(
            [xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)],
            dim=1,
        )

    context = c_crossattn
    dtype = base_model.get_dtype_inference()

    xc = xc.to(dtype)
    device = xc.device
    t_model = base_model.model_sampling.timestep(t).float()
    if context is not None:
        context = comfy.model_management.cast_to_device(context, device, dtype)

    extra_conds: Dict[str, Any] = {}
    for name, extra in kwargs.items():
        if hasattr(extra, 'dtype'):
            extra = _rf_comfy_convert_tensor(extra, dtype, device)
        elif isinstance(extra, list):
            extra = [_rf_comfy_convert_tensor(item, dtype, device) for item in extra]
        extra_conds[name] = extra

    t_model = base_model.process_timestep(t_model, x=x, **extra_conds)
    if 'latent_shapes' in extra_conds:
        xc = comfy.utils.unpack_latents(xc, extra_conds.pop('latent_shapes'))

    transformer_options = (transformer_options or {}).copy()
    transformer_options['prefetch_dynamic_vbars'] = (
        base_model.current_patcher is not None and base_model.current_patcher.is_dynamic()
    )

    model_output = base_model.diffusion_model(
        xc,
        t_model,
        context=context,
        control=control,
        transformer_options=transformer_options,
        **extra_conds,
    )
    if len(model_output) > 1 and not torch.is_tensor(model_output):
        model_output, _ = comfy.utils.pack_latents(model_output)
    if not torch.is_tensor(model_output):
        raise RuntimeError(
            'RFInversion raw-velocity mode expected diffusion_model to return a tensor after packing.'
        )
    return model_output.float()

def _make_raw_velocity_apply_model_fn(apply_model: Callable) -> Callable:
    """Return an apply_model-shaped callable that yields raw FLOW/FLUX velocity only."""
    def raw_velocity_apply_model(
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[torch.Tensor] = None,
        c_crossattn: Optional[torch.Tensor] = None,
        control: Any = None,
        transformer_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return _rf_raw_transformer_velocity(
            apply_model,
            x,
            t,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=transformer_options,
            **kwargs,
        )
    return raw_velocity_apply_model

def _velocity_from_pred(
    x_sigma: torch.Tensor,
    pred: torch.Tensor,
    sigma: float,
    parameterization: str,
) -> torch.Tensor:
    """Accept raw transformer velocity only."""
    mode = str(parameterization or '').lower()
    if mode not in ('raw_velocity', 'velocity_raw', 'model_velocity'):
        raise RuntimeError(
            'RFInversion expected raw transformer velocity but received '
            f'parameterization={parameterization!r}. No x0/denoised fallback is enabled.'
        )
    if not torch.is_tensor(pred):
        raise RuntimeError('RFInversion raw-velocity mode received a non-tensor model output.')
    return pred.float()


def _flow_denoised_preview_from_raw_velocity(
    x_sigma: torch.Tensor,
    raw_velocity: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Recreate Comfy's FLOW/FLUX denoised/x0 value for preview only.
    """
    if not torch.is_tensor(x_sigma) or not torch.is_tensor(raw_velocity):
        raise RuntimeError('RF preview conversion requires tensor x_sigma and raw_velocity.')
    sigma_t = torch.as_tensor(float(sigma), device=x_sigma.device, dtype=raw_velocity.dtype)
    sigma_t = sigma_t.reshape((1,) * raw_velocity.ndim)
    return x_sigma.to(dtype=raw_velocity.dtype) - raw_velocity * sigma_t

# ═══════════════════════════════════════════════════════════════════════════════
# RF utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_base_mode(mode: str) -> str:
    """Return the actual RF solver mode.

    OTIP is no longer encoded in rf_mode names. Select one of the normal RF
    solvers and set otip_strength > 0 to add OTIP transport guidance.
    """
    return str(mode or 'rf_gamma')


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    value_l = str(value).strip().lower()
    if value_l in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value_l in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return bool(default)


def _coerce_otip_strength(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid otip_strength value: {value!r}.') from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid otip_strength value: {value!r} is not finite.')
    # The official repo documents a practical range around 0.1–1.2.
    return max(0.0, min(2.0, v))

def _coerce_otip_phase(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid otip_phase value: {value!r}.') from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid otip_phase value: {value!r} is not finite.')
    return max(1e-6, min(1.0, v))

def _coerce_otip_clip_norm(value: Any, default: float = 10.0) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid otip_clip_norm value: {value!r}.') from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid otip_clip_norm value: {value!r} is not finite.')
    return max(1e-6, min(1e4, v))


def _otip_feature_norm(x: torch.Tensor) -> torch.Tensor:
    """Feature-wise norm matching the official OT-RF implementation.

    The official Flux pipeline operates on packed latents shaped [B, tokens, C]
    and clips with torch.norm(..., dim=-1, keepdim=True). Comfy latent tensors
    here are normally [B, C, H, W], so the feature dimension is channel dim=1.
    """
    if x.ndim == 3:
        return torch.linalg.vector_norm(x.float(), ord=2, dim=-1, keepdim=True).to(dtype=x.dtype)
    if x.ndim >= 4:
        return torch.linalg.vector_norm(x.float(), ord=2, dim=1, keepdim=True).to(dtype=x.dtype)
    dims = tuple(range(1, x.ndim))
    if not dims:
        return x.detach().abs().clamp_min(1e-8)
    return torch.linalg.vector_norm(x.float(), ord=2, dim=dims, keepdim=True).to(dtype=x.dtype)


def _otip_compute_transport_direction(
    current_state: torch.Tensor,
    target_state: torch.Tensor,
    timestep: float,
    clip_norm: float = 10.0,
    denom_eps: float = 0.01,
) -> torch.Tensor:
    """Closed-form OT-RF/W2 displacement direction.

    This is the official OT-RF direction adapted to Comfy latents:
        direction = (target_state - current_state) / max(1 - timestep, 0.01)
    followed by feature-wise norm clipping.
    """
    denom = max(1.0 - float(timestep), float(denom_eps))
    direction = (target_state.to(device=current_state.device, dtype=current_state.dtype) - current_state) / denom
    direction_norm = _otip_feature_norm(direction).clamp_min(1e-8)
    max_norm = torch.as_tensor(float(clip_norm), device=direction.device, dtype=direction.dtype)
    scale_factor = torch.minimum(torch.ones_like(direction_norm), max_norm / direction_norm)
    return direction * scale_factor


def _otip_adaptive_transport_strength(
    step_index: int,
    total_steps: int,
    base_strength: float,
    timestep_value: float,
    phase_shift: float = 1.0,
) -> float:
    """Return the OTIP transport strength for one inversion step.

    ``base_strength`` remains the main OTIP on/off and magnitude control.
    The ComfyUI node hardcodes phase to 1.0, so OTIP is applied on every RF
    step whenever ``otip_strength > 0``. The ``phase_shift`` argument remains
    for backward-compatible metadata/config loading only.
    """
    strength = max(0.0, float(base_strength))
    if strength <= 0.0:
        return 0.0

    phase = max(0.0, min(1.0, float(phase_shift)))
    if phase >= 0.999:
        return strength

    sigma = max(0.0, min(1.0, float(timestep_value)))
    start_sigma = max(0.0, 1.0 - phase)
    if sigma <= start_sigma:
        return 0.0

    u = (sigma - start_sigma) / max(phase, 1e-8)
    u = max(0.0, min(1.0, u))
    smooth = 0.5 - 0.5 * math.cos(math.pi * u)
    return strength * smooth


def _otip_apply_velocity_guidance(
    v_base: torch.Tensor,
    current_state: torch.Tensor,
    target_state: torch.Tensor,
    timestep: float,
    step_index: int,
    total_steps: int,
    base_strength: float,
    phase_shift: float,
    clip_norm: float,
    respect_model_norm: bool = True,
) -> Tuple[torch.Tensor, float, float]:
    """Apply OTIP's additive velocity-field correction.

    The official code computes ot_contribution = strength * (ot_direction - v_t)
    and adds it to v_t. With respect_model_norm=True it rescales the correction
    by ||v_t|| / ||ot_direction|| so OT does not swamp the RF velocity field.
    """
    strength = _otip_adaptive_transport_strength(
        step_index=step_index,
        total_steps=total_steps,
        base_strength=base_strength,
        timestep_value=timestep,
        phase_shift=phase_shift,
    )
    if strength <= 0.0:
        return v_base, 0.0, 0.0

    ot_direction = _otip_compute_transport_direction(
        current_state=current_state,
        target_state=target_state,
        timestep=timestep,
        clip_norm=clip_norm,
    )
    ot_contribution = float(strength) * (ot_direction - v_base)
    if respect_model_norm:
        v_norm = _otip_feature_norm(v_base).clamp_min(1e-8)
        ot_norm = _otip_feature_norm(ot_direction).clamp_min(1e-8)
        ot_contribution = ot_contribution * (v_norm / ot_norm)
    guided = v_base + ot_contribution
    return guided.to(dtype=v_base.dtype), float(strength), float(ot_direction.detach().abs().mean().item())
def _rf_gamma_for_mode(
    mode: str,
    gamma: float,
    sigma_prev: float,
    sigma_cur: float,
) -> float:
    base_mode = _rf_base_mode(mode)
    if base_mode == 'linear':
        return 0.0
    return float(gamma)

def _rf_linear_target(ref_clean: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(0.0, min(1.0, float(sigma)))
    return (1.0 - sigma) * ref_clean + sigma * eps


def _flowturbo_pc_internal_sigmas(sigmas: List[float]) -> List[float]:
    """Build a deterministic FlowTurbo-style pseudo-corrector subgrid.
    Pseudo-corrector Heun reuses the endpoint model evaluation from the
    previous interval as the next interval's start velocity. With ``n`` sampler
    intervals, Endpoint-Heun costs ``2*n`` model calls. This helper creates
    ``2*n - 1`` internal intervals so pseudo-corrector uses the same budget:
    one initial model call plus one endpoint call per internal interval.
    """
    clean: List[float] = []
    for s in sigmas or []:
        try:
            sf = max(0.0, min(1.0, float(s)))
        except Exception:
            continue
        if not clean or abs(sf - clean[-1]) > 1e-8:
            clean.append(sf)
    clean = sorted(clean)
    if len(clean) <= 2:
        return clean

    intervals = []
    for i in range(len(clean) - 1):
        a, b = clean[i], clean[i + 1]
        d = b - a
        if d > 1e-8:
            intervals.append((d, i, a, b))

    extra_count = max(0, min(len(intervals) - 1, len(clean) - 2))
    split_indices = {i for _, i, _, _ in sorted(intervals, reverse=True)[:extra_count]}

    grid: List[float] = [clean[0]]
    for i in range(len(clean) - 1):
        a, b = clean[i], clean[i + 1]
        if i in split_indices:
            mid = 0.5 * (a + b)
            if mid - grid[-1] > 1e-8 and b - mid > 1e-8:
                grid.append(mid)
        grid.append(b)

    out: List[float] = []
    for s in grid:
        if not out or abs(float(s) - out[-1]) > 1e-8:
            out.append(float(s))
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# PMI — Proximal-Mean Inversion (Wang et al., ICLR 2026)
# "Free Lunch for Stabilizing Rectified Flow Inversion"
# https://arxiv.org/abs/2602.11850
# ═══════════════════════════════════════════════════════════════════════════════

class _PMIState:
    def __init__(self, pmi_dim: int = 22, eps: float = 1e-12) -> None:
        self.mean_velocity: Optional[torch.Tensor] = None
        self.prev_corrected_velocity: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.pmi_dim: int = max(1, int(pmi_dim))
        self.eps: float = float(eps)

    def reset(self) -> None:
        self.mean_velocity = None
        self.prev_corrected_velocity = None
        self.step_count = 0

    def _grad_norm(self, grad: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, grad.ndim))
        if not dims:
            return grad.detach().abs().clamp_min(self.eps)
        return torch.linalg.vector_norm(
            grad.detach().float(), ord=2, dim=dims, keepdim=True
        ).to(dtype=grad.dtype).clamp_min(self.eps)

    def update_and_correct(
        self,
        v_model: torch.Tensor,
        delta_t: float,
        t_next: float,
        strength: float = 1.0,
        post_update_corrected: bool = False,
    ) -> torch.Tensor:
        """
        Paper Algorithm 1 uses xdist as a running sum of the *raw predicted*
        velocities up to the current step:

            xdist += Δt * v_t
            v_mean = xdist / (t_next - t0)
            v_hat = v_t - r_t * grad F(v_t) / ||grad F(v_t)||
            z_next = z + Δt * v_hat

        The corrected velocity is used for the RF state update and the
        previous-velocity consistency term.
        """
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0:
            return v_model

        dt = float(delta_t)
        if not math.isfinite(dt) or abs(dt) <= self.eps:
            return v_model

        t_next_f = float(t_next)
        if not math.isfinite(t_next_f):
            return v_model

        device = v_model.device
        dtype = v_model.dtype
        v_detached = v_model.detach()

        raw_increment = (dt * v_detached).to(device=device, dtype=dtype)
        if self.mean_velocity is None:
            self.mean_velocity = raw_increment.clone()
        else:
            self.mean_velocity = self.mean_velocity.to(device=device, dtype=dtype) + raw_increment

        denom = t_next_f if abs(t_next_f) > self.eps else (self.eps if t_next_f >= 0.0 else -self.eps)
        pred_mean = (self.mean_velocity.to(device=device, dtype=dtype) / denom).detach()

        # PMI objective gradient:
        #   grad 0.5||v - v_mean||_2^2 = v - v_mean
        #   grad ||v - v_prev||_1       = sign(v - v_prev)
        pred = v_detached
        grad = (pred.float() - pred_mean.float()).to(dtype=dtype)
        if self.prev_corrected_velocity is not None:
            prev = self.prev_corrected_velocity.to(device=device, dtype=dtype).detach()
            grad = grad + (pred.float() - prev.float()).sign().to(dtype=dtype)

        radius = math.sqrt(2.0 * self.pmi_dim + 3.0 * math.sqrt(2.0 * self.pmi_dim)) * abs(dt) * strength
        corrected = (pred - radius * (grad / self._grad_norm(grad))).to(dtype=dtype)

        self.prev_corrected_velocity = corrected.detach().clone()
        self.step_count += 1

        return corrected

# ═══════════════════════════════════════════════════════════════════════════════
# Main RF trajectory builder
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_build_cache_from_sampler_sigmas(
    ref_clean:      torch.Tensor,
    sampler_sigmas: List[float],
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    gamma:          float = 0.5,
    seed:           int   = 0,
    stats:          Optional[vp._RuntimeStats] = None,
    eps:            Optional[torch.Tensor] = None,
    rf_mode:        str   = 'rf_gamma',
    pmi_alpha:      float = 0.4,
    otip_strength:   float = 0.0,
    otip_phase:      float = 1.0,
    otip_clip_norm:  float = 10.0,
    otip_respect_model_norm: bool = False,
    preview_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None,
) -> Tuple[Dict[float, torch.Tensor], torch.Tensor, List[float]]:
    """
    Build reference x_sigma latents on the actual sampler sigma grid.
    """
    mode = str(rf_mode or 'rf_gamma')
    valid_modes = {'linear', 'rf_gamma', 'rf_gamma_rk2', 'rf_solver_2', 'endpoint_heun', 'fireflow', 'flowturbo_pc'}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid rf_mode={mode!r}. Expected one of {sorted(valid_modes)}."
        )
    base_mode = _rf_base_mode(mode)
    otip_strength_eff = _coerce_otip_strength(otip_strength)
    otip_phase_eff = _coerce_otip_phase(otip_phase)
    otip_clip_norm_eff = _coerce_otip_clip_norm(otip_clip_norm)
    # OTIP respect_model_norm is intentionally hardcoded off.
    otip_respect_model_norm_eff = False
    use_otip = otip_strength_eff > 0.0

    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    device = ref_clean.device
    dtype  = ref_clean.dtype

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=rng)

    if sampler_sigmas is None:
        raise RuntimeError('RF trajectory build failed: sampler_sigmas is missing.')

    # Build sorted unique sigma grid starting from 0. Invalid entries are fatal.
    sigmas: List[float] = [0.0]
    for idx, s in enumerate(sampler_sigmas):
        try:
            sf = float(s)
        except Exception as exc:
            raise RuntimeError(
                f'RF trajectory build failed: sampler sigma at index {idx} is not numeric: {s!r}.'
            ) from exc
        if not math.isfinite(sf):
            raise RuntimeError(
                f'RF trajectory build failed: sampler sigma at index {idx} is not finite: {s!r}.'
            )
        sf = max(0.0, min(1.0, sf))
        if all(abs(sf - existing) > 1e-6 for existing in sigmas):
            sigmas.append(sf)
    sigmas = sorted(sigmas)
    if len(sigmas) <= 1:
        raise RuntimeError(
            'RF trajectory build failed: sampler sigma schedule did not contain any usable non-zero steps.'
        )

    z = ref_clean.clone()
    prev = 0.0
    cache: Dict[float, torch.Tensor] = {0.0: z.detach().clone()}
    model_ok = 0
    failures = 0
    vm_sum = 0.0
    vp_sum = 0.0
    # FireFlow state: stores midpoint velocity from previous step for reuse.
    next_step_velocity: Optional[torch.Tensor] = None

    # PMI state. pmi_alpha is the on/off control: <= 0 means disabled.
    pmi_alpha_eff = max(0.0, min(1.0, float(pmi_alpha)))
    use_pmi = pmi_alpha_eff > 0.0
    pmi_state = _PMIState()

    total_preview_steps = max(1, len(sigmas) - 1)
    previewed_steps: set = set()

    def _preview_once(step_index: int, denoised_preview: Optional[torch.Tensor], x_current: Optional[torch.Tensor]) -> None:
        if preview_callback is None or not torch.is_tensor(denoised_preview):
            return
        step_index = max(0, min(total_preview_steps - 1, int(step_index)))
        if step_index in previewed_steps:
            return
        previewed_steps.add(step_index)
        _rf_emit_preview(preview_callback, step_index, denoised_preview, x_current, total_preview_steps)

    vp._rf_vprint(stats,
        f'{vp._rf_prefix(stats)}   RF trajectory mode: {mode}'
        f'{f"  base={base_mode}" if use_otip else ""}  gamma={gamma:.4f}  '
        f'parameterization={parameterization}\n'
        f'{vp._rf_prefix(stats)}   pmi_alpha={pmi_alpha_eff:.3f}  '
        f'PMI={"on" if use_pmi else "off"}  '
        f'OTIP={"on" if use_otip else "off"}  '
        f'otip_strength={otip_strength_eff:.3f}  '
        f'otip_phase={otip_phase_eff:.3f}  '
        f'otip_clip={otip_clip_norm_eff:.3f}'
    )

    # Print persistent RF inversion progress snapshots. This keeps every RF step
    rf_total_steps = max(1, len(sigmas) - 1)
    rf_progress_start_time = time.time()

    # RF step quality diagnostics.
    path_prev_speed: Optional[float] = None

    def _call_model_as_velocity_shared(z_in, sigma_val, label=''):
        nonlocal model_ok, vm_sum
        t_tensor = torch.full((z_in.shape[0],), sigma_val, device=device, dtype=dtype)
        with torch.no_grad():
            try:
                raw = apply_model_fn(z_in, t_tensor, **base_model_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f'RF trajectory build failed during model call at σ={sigma_val:.6f} '
                    f'mode={mode}{label}.'
                ) from exc

        model_ok += 1
        v = _velocity_from_pred(z_in, raw, sigma_val, parameterization)
        denoised_preview = _flow_denoised_preview_from_raw_velocity(z_in, v, sigma_val)
        vm_sum += float(v.abs().mean().item())
        return v, True, denoised_preview

    def _apply_pmi_with_delta_if_enabled(
        v: torch.Tensor,
        delta_t: float,
        pmi_time: float,
        *,
        post_update_corrected: bool = False,
    ) -> torch.Tensor:
        if not use_pmi:
            return v
        return pmi_state.update_and_correct(
            v,
            delta_t=delta_t,
            t_next=pmi_time,
            strength=pmi_alpha_eff,
            post_update_corrected=post_update_corrected,
        )

    if base_mode == 'flowturbo_pc':
        # FlowTurbo-style pseudo-corrector Heun.
        # Internal substeps are integrated silently; verbose logs/progress are
        # emitted only when an original sampler sigma endpoint is reached.
        internal_sigmas = _flowturbo_pc_internal_sigmas(sigmas)
        internal_total_steps = max(1, len(internal_sigmas) - 1)
        original_sigma_keys = {round(float(sv), 6) for sv in sigmas}
        cached_macro_steps = 0
        macro_total_steps = max(1, len(sigmas) - 1)

        vp._rf_vprint(stats,
            f'{vp._rf_prefix(stats)}   FlowTurbo-PC internal grid: '
            f'sampler_steps={len(sigmas)-1}  internal_steps={internal_total_steps}  '
            f'target_model_evals={internal_total_steps + 1}  logging=grid_endpoints'
        )

        v_start_model, ok_start, raw_preview_start = _call_model_as_velocity_shared(
            z, internal_sigmas[0], ' flowturbo_pc start'
        )
        vm_abs = float(v_start_model.abs().mean().item())

        macro_prev_sigma = float(sigmas[0])
        macro_prev_z = z.detach().clone()
        macro_vm_sum = 0.0
        macro_vp_sum = 0.0
        macro_internal_count = 0
        macro_first_internal_sigma = float(internal_sigmas[0])
        macro_last_vm_start = vm_abs
        macro_last_vm_end = vm_abs
        macro_last_otip_extra = ''

        # Keep the integration subgrid hidden, but drive the usual RF progress
        # iterator once per original sampler-grid endpoint.
        macro_progress_iter = iter(vp._rf_step_iterator(macro_total_steps))
        for internal_index in range(internal_total_steps):
            step_i = int(internal_index) + 1
            sigma_prev = float(internal_sigmas[step_i - 1])
            sigma_cur = float(internal_sigmas[step_i])
            delta = float(sigma_cur - sigma_prev)
            gamma_eff = _rf_gamma_for_mode(mode, gamma, sigma_prev, sigma_cur)
            otip_extra = ''
            otip_schedule_index = max(0, internal_total_steps - step_i)

            # Pseudo-corrector predictor: use the reusable start velocity.
            z_pred_next = z + delta * v_start_model
            v_model_end, ok_end, raw_preview_end = _call_model_as_velocity_shared(
                z_pred_next, sigma_cur, ' flowturbo_pc pseudo_end'
            )
            vm_abs_start = float(v_start_model.abs().mean().item())
            vm_abs_end = float(v_model_end.abs().mean().item())

            # Model-only Heun/trapezoid update first.
            z_model_next = z + 0.5 * delta * (v_start_model + v_model_end)
            z_prior_next = _rf_linear_target(ref_clean, eps, sigma_cur)
            vp_abs_target = float((z_prior_next - z).abs().mean().item() / max(abs(delta), 1e-12))
            vp_sum += vp_abs_target

            z_solver_next = gamma_eff * z_model_next + (1.0 - gamma_eff) * z_prior_next
            if abs(delta) > 1e-12:
                v_total = (z_solver_next - z) / delta
                if use_otip:
                    v_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                        v_total, z, eps, sigma_prev, otip_schedule_index, internal_total_steps,
                        otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                        respect_model_norm=otip_respect_model_norm_eff,
                    )
                    otip_extra = f'  OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
                v_total = _apply_pmi_with_delta_if_enabled(
                    v_total, delta, sigma_cur, post_update_corrected=True
                )
                z = z + delta * v_total
            else:
                z = z_solver_next

            # Reuse the endpoint model evaluation as the next start velocity.
            v_start_model = v_model_end.detach().clone()
            vm_abs = vm_abs_end

            macro_internal_count += 1
            macro_vm_sum += 0.5 * (vm_abs_start + vm_abs_end)
            macro_vp_sum += vp_abs_target
            macro_last_vm_start = vm_abs_start
            macro_last_vm_end = vm_abs_end
            macro_last_otip_extra = otip_extra

            sigma_key = round(sigma_cur, 6)
            cache_hit_endpoint = sigma_key in original_sigma_keys
            if cache_hit_endpoint:
                cache[sigma_key] = z.detach().clone()
                macro_step_i = cached_macro_steps + 1
                _preview_once(cached_macro_steps, raw_preview_end, z)

                z_mean = float(z.mean().item())
                z_std = float(z.std().item())
                z_min = float(z.min().item())
                z_max = float(z.max().item())
                macro_delta = float(sigma_cur - macro_prev_sigma)
                macro_dz_abs = float((z - macro_prev_z).abs().mean().item())
                macro_model_abs = macro_vm_sum / max(1, macro_internal_count)
                macro_prior_abs = macro_vp_sum / max(1, macro_internal_count)

                extra = (
                    f'FlowTurbo-PC reuse  substeps={macro_internal_count}  '
                    f'|v_start|={macro_last_vm_start:.5f}  '
                    f'|v_end|={macro_last_vm_end:.5f}  '
                    f'|prior_avg|={macro_prior_abs:.5f}'
                )
                if macro_last_otip_extra:
                    extra += macro_last_otip_extra
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'

                try:
                    next(macro_progress_iter)
                except StopIteration:
                    pass

                vp._rf_vprint(stats,
                    f'{vp._rf_prefix(stats)}     z_sigma step {macro_step_i:02d}/{macro_total_steps}: '
                    f'mode={mode}  γ_eff={gamma_eff:.4f}  '
                    f'σ_prev={macro_prev_sigma:.6f} -> σ={sigma_cur:.6f}  Δσ={macro_delta:.6f}  '
                    f'|model|={macro_model_abs:.5f}  |prior|={macro_prior_abs:.5f}  '
                    f'|Δz|={macro_dz_abs:.5f}  {extra}\n'
                    f'{vp._rf_prefix(stats)}       z_σ mean={z_mean:.4f}  std={z_std:.4f}  '
                    f'min={z_min:.4f}  max={z_max:.4f}'
                )
                path_prev_speed = vp._rf_print_step_quality(
                    stats,
                    ref_clean=ref_clean,
                    eps=eps,
                    z=z,
                    sigma_cur=sigma_cur,
                    delta=macro_delta,
                    dz_abs=macro_dz_abs,
                    path_prev_speed=path_prev_speed,
                )
                vp._rf_progress_snapshot(
                    macro_step_i,
                    macro_total_steps,
                    rf_progress_start_time,
                    persistent=vp._coerce_bool(getattr(stats, 'rf_verbose', False)),
                )

                cached_macro_steps += 1
                macro_prev_sigma = sigma_cur
                macro_prev_z = z.detach().clone()
                macro_vm_sum = 0.0
                macro_vp_sum = 0.0
                macro_internal_count = 0
                macro_first_internal_sigma = sigma_cur
                macro_last_otip_extra = ''

        # Ensure every original sampler endpoint has an exact cache entry.
        missing_keys = [round(float(sv), 6) for sv in sigmas if round(float(sv), 6) not in cache]
        if missing_keys:
            raise RuntimeError(
                f'FlowTurbo-PC trajectory build failed: missing original sampler cache keys {missing_keys}.'
            )

        steps = max(1, len(sigmas) - 1)
        vp._rf_vprint(stats,
            f'{vp._rf_prefix(stats)}   RF schedule build: mode={mode}  sampler_sigmas={len(sampler_sigmas)}  '
            f'unique={len(sigmas)}  rf_steps={len(sigmas)-1}  internal_steps={internal_total_steps}  '
            f'model_ok={model_ok}  failures={failures}\n'
            f'{vp._rf_prefix(stats)}     sigma_range=[{sigmas[0]:.6f}, {sigmas[-1]:.6f}]  '
            f'|model|={vm_sum/max(1, model_ok):.5f}  |prior|={vp_sum/max(1, internal_total_steps):.5f}  '
            f'z_final std={z.std().item():.4f}  parameterization={parameterization}'
        )
        return cache, eps, sigmas

    for step_index in vp._rf_step_iterator(rf_total_steps):
        step_i = int(step_index) + 1
        s = sigmas[step_i]
        sigma_prev = float(prev)
        sigma_cur  = float(s)
        delta      = float(sigma_cur - sigma_prev)
        z_prev     = z.detach().clone()
        gamma_eff  = _rf_gamma_for_mode(mode, gamma, sigma_prev, sigma_cur)
        otip_extra = ''
        otip_schedule_index = max(0, rf_total_steps - step_i)

        vm_abs = 0.0
        vp_abs = 0.0
        extra  = ''

        # ── Helper: run model and convert output to velocity ─────────────────
        def _call_model_as_velocity(z_in, sigma_val, label=''):
            nonlocal model_ok, vm_sum
            t_tensor = torch.full((z_in.shape[0],), sigma_val, device=device, dtype=dtype)
            with torch.no_grad():
                try:
                    raw = apply_model_fn(z_in, t_tensor, **base_model_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        f'RF trajectory build failed during model call at σ={sigma_val:.6f} '
                        f'mode={mode}{label}.'
                    ) from exc

            model_ok += 1
            v = _velocity_from_pred(z_in, raw, sigma_val, parameterization)
            denoised_preview = _flow_denoised_preview_from_raw_velocity(z_in, v, sigma_val)
            vm_sum += float(v.abs().mean().item())
            return v, True, denoised_preview

        def _apply_pmi_if_enabled(
            v: torch.Tensor,
            pmi_time: float,
            *,
            post_update_corrected: bool = False,
        ) -> torch.Tensor:
            if not use_pmi:
                return v
            return pmi_state.update_and_correct(
                v,
                delta_t=delta,
                t_next=pmi_time,
                strength=pmi_alpha_eff,
                post_update_corrected=post_update_corrected,
            )


        if mode == 'linear':
            z = _rf_linear_target(ref_clean, eps, sigma_cur)
            extra = 'linear_target'

        elif base_mode == 'fireflow':
            # ── (Deng et al., ICML 2025) ─────────
            if next_step_velocity is None:
                v_model_pred, ok, raw_preview = _call_model_as_velocity(z, sigma_prev, ' fresh')
                denom_prev = max(1.0 - sigma_prev, 1e-7)
                v_prior_pred = (eps - z) / denom_prev
                v_pred = gamma_eff * v_model_pred + (1.0 - gamma_eff) * v_prior_pred
                vm_abs = float(v_model_pred.abs().mean().item())
                pred_source = 'fresh_blend'
            else:
                v_pred = next_step_velocity.to(device=device, dtype=dtype)
                vm_abs = float(v_pred.abs().mean().item())
                pred_source = 'reused_blend'

            z_mid      = z + 0.5 * delta * v_pred
            sigma_mid  = sigma_prev + 0.5 * delta
            v_mid, ok, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
            vm_abs_mid = float(v_mid.abs().mean().item())

            denom_mid = max(1.0 - sigma_mid, 1e-7)
            v_prior_mid = (eps - z_mid) / denom_mid
            vp_abs = float(v_prior_mid.detach().abs().mean().item())
            vp_sum += vp_abs

            v_mid_total = gamma_eff * v_mid + (1.0 - gamma_eff) * v_prior_mid
            if use_otip:
                v_mid_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                    v_mid_total, z, eps, sigma_prev, otip_schedule_index, rf_total_steps,
                    otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                    respect_model_norm=otip_respect_model_norm_eff,
                )
                otip_extra = f'  OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
            v_mid_total = _apply_pmi_if_enabled(v_mid_total, sigma_cur, post_update_corrected=False)
            next_step_velocity = v_mid_total.detach().clone()
            z = z + delta * v_mid_total
            _preview_once(step_i - 1, raw_preview_mid, z)
            extra = (
                f'FireFlow pred={pred_source}  σ_mid={sigma_mid:.6f}  '
                f'|v_pred|={vm_abs:.5f}  |v_mid|={vm_abs_mid:.5f}  |prior_mid|={vp_abs:.5f}'
            )
            if otip_extra:
                extra += otip_extra
            if use_pmi:
                extra += f'  PMI step={pmi_state.step_count}'

        else:
            # ── RF-style velocity  ─
            v_model, ok, raw_preview = _call_model_as_velocity(z, sigma_prev)
            vm_abs = float(v_model.abs().mean().item())

            denom   = max(1.0 - sigma_prev, 1e-7)
            v_prior = (eps - z) / denom
            vp_abs  = float(v_prior.abs().mean().item())
            vp_sum += vp_abs

            if base_mode == 'rf_solver_2':
                # ── RF-Solver-2 second-order Taylor step ───────────────
                z_mid = z + 0.5 * delta * v_model
                sigma_mid = sigma_prev + 0.5 * delta
                v_model_mid, ok_mid, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' rf_solver_2 mid')
                vm_abs_mid = float(v_model_mid.abs().mean().item())

                half_delta = 0.5 * delta
                if abs(half_delta) > 1e-12:
                    first_order = (v_model_mid - v_model) / half_delta
                    z_model_next = z + delta * v_model + 0.5 * (delta ** 2) * first_order
                else:
                    z_model_next = z.detach().clone()

                z_prior_next = _rf_linear_target(ref_clean, eps, sigma_cur)
                vp_abs_target = float((z_prior_next - z).abs().mean().item() / max(abs(delta), 1e-12))
                vp_sum += vp_abs_target

                z_solver_next = gamma_eff * z_model_next + (1.0 - gamma_eff) * z_prior_next
                if abs(delta) > 1e-12:
                    v_total = (z_solver_next - z) / delta
                    if use_otip:
                        v_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                            v_total, z, eps, sigma_prev, otip_schedule_index, rf_total_steps,
                            otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                            respect_model_norm=otip_respect_model_norm_eff,
                        )
                        otip_extra = f'  OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
                    v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
                    z = z + delta * v_total
                else:
                    z = z_solver_next
                _preview_once(step_i - 1, raw_preview_mid, z)
                extra = (
                    f'RF-Solver-2 exact  |v_model_mid|={vm_abs_mid:.5f}  '
                    f'|prior_target|={vp_abs_target:.5f}'
                )
                if otip_extra:
                    extra += otip_extra
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'

            elif base_mode == 'endpoint_heun':
                # ── Endpoint Heun / explicit trapezoid step ─────────────
                # Predictor: advance with the start velocity to the actual next sigma.
                # Corrector: query the model at that predicted endpoint and average
                # start/end velocities. At gamma=1 this is a pure model trajectory
                # with exactly two model evaluations per RF step.
                z_pred_next = z + delta * v_model
                v_model_end, ok_end, raw_preview_end = _call_model_as_velocity(
                    z_pred_next, sigma_cur, ' endpoint_heun end'
                )
                vm_abs_end = float(v_model_end.abs().mean().item())

                z_model_next = z + 0.5 * delta * (v_model + v_model_end)
                z_prior_next = _rf_linear_target(ref_clean, eps, sigma_cur)
                vp_abs_target = float((z_prior_next - z).abs().mean().item() / max(abs(delta), 1e-12))
                vp_sum += vp_abs_target

                z_solver_next = gamma_eff * z_model_next + (1.0 - gamma_eff) * z_prior_next
                if abs(delta) > 1e-12:
                    v_total = (z_solver_next - z) / delta
                    if use_otip:
                        v_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                            v_total, z, eps, sigma_prev, otip_schedule_index, rf_total_steps,
                            otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                            respect_model_norm=otip_respect_model_norm_eff,
                        )
                        otip_extra = f'  OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
                    v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
                    z = z + delta * v_total
                else:
                    z = z_solver_next
                _preview_once(step_i - 1, raw_preview_end, z)
                extra = (
                    f'Endpoint-Heun  |v_model_end|={vm_abs_end:.5f}  '
                    f'|prior_target|={vp_abs_target:.5f}'
                )
                if otip_extra:
                    extra += otip_extra
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'

            elif base_mode == 'rf_gamma_rk2':
                v1    = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                z_mid = z + 0.5 * delta * v1
                sigma_mid = sigma_prev + 0.5 * delta
                v_model_mid, ok_mid, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
                vm_abs_mid = float(v_model_mid.abs().mean().item())

                denom_mid = max(1.0 - sigma_mid, 1e-7)
                v_prior_mid = (eps - z_mid) / denom_mid
                vp_abs_mid  = float(v_prior_mid.abs().mean().item())
                vp_sum += vp_abs_mid

                v_total = gamma_eff * v_model_mid + (1.0 - gamma_eff) * v_prior_mid
                if use_otip:
                    v_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                        v_total, z, eps, sigma_prev, otip_schedule_index, rf_total_steps,
                        otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                        respect_model_norm=otip_respect_model_norm_eff,
                    )
                    otip_extra = f'  OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
                v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview_mid, z)
                extra = f'mid |v_model_mid|={vm_abs_mid:.5f}'
                if otip_extra:
                    extra += otip_extra
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'
            else:
                v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                if use_otip:
                    v_total, ot_strength, ot_abs = _otip_apply_velocity_guidance(
                        v_total, z, eps, sigma_prev, otip_schedule_index, rf_total_steps,
                        otip_strength_eff, otip_phase_eff, otip_clip_norm_eff,
                        respect_model_norm=otip_respect_model_norm_eff,
                    )
                    otip_extra = f'OTIP λ={ot_strength:.4f} |ot|={ot_abs:.5f}'
                v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview, z)
                if otip_extra:
                    extra = otip_extra
                if use_pmi:
                    extra = (extra + '  ' if extra else '') + f'PMI step={pmi_state.step_count}'


        prev  = sigma_cur
        z_mean = float(z.mean().item())
        z_std  = float(z.std().item())
        z_min  = float(z.min().item())
        z_max  = float(z.max().item())
        dz_abs = float((z - z_prev).abs().mean().item())

        cache[round(sigma_cur, 6)] = z.detach().clone()

        vp._rf_vprint(stats,
            f'{vp._rf_prefix(stats)}     z_sigma step {step_i:02d}/{len(sigmas)-1}: '
            f'mode={mode}  γ_eff={gamma_eff:.4f}  '
            f'σ_prev={sigma_prev:.6f} -> σ={sigma_cur:.6f}  Δσ={delta:.6f}  '
            f'|model|={vm_abs:.5f}  |prior|={vp_abs:.5f}  |Δz|={dz_abs:.5f}  {extra}\n'
            f'{vp._rf_prefix(stats)}       z_σ mean={z_mean:.4f}  std={z_std:.4f}  '
            f'min={z_min:.4f}  max={z_max:.4f}'
        )
        path_prev_speed = vp._rf_print_step_quality(
            stats,
            ref_clean=ref_clean,
            eps=eps,
            z=z,
            sigma_cur=sigma_cur,
            delta=delta,
            dz_abs=dz_abs,
            path_prev_speed=path_prev_speed,
        )

        vp._rf_progress_snapshot(
            step_i,
            rf_total_steps,
            rf_progress_start_time,
            persistent=vp._coerce_bool(getattr(stats, 'rf_verbose', False)),
        )

    steps = max(1, len(sigmas) - 1)
    vp._rf_vprint(stats,
        f'{vp._rf_prefix(stats)}   RF schedule build: mode={mode}  sampler_sigmas={len(sampler_sigmas)}  '
        f'unique={len(sigmas)}  rf_steps={len(sigmas)-1}  '
        f'model_ok={model_ok}  failures={failures}\n'
        f'{vp._rf_prefix(stats)}     sigma_range=[{sigmas[0]:.6f}, {sigmas[-1]:.6f}]  '
        f'|model|={vm_sum/max(1, model_ok):.5f}  |prior|={vp_sum/steps:.5f}  '
        f'z_final std={z.std().item():.4f}  parameterization={parameterization}'
    )
    return cache, eps, sigmas

def _find_sigma_schedule(obj: Any, depth: int = 0) -> Optional[List[float]]:
    if depth > 6 or obj is None:
        return None

    if isinstance(obj, dict):
        preferred = (
            'sample_sigmas', 'sampler_sigmas', 'sigmas', 'scheduler_sigmas',
            'denoise_sigmas', 'noise_sigmas', 'timesteps', 'timestep_schedule',
        )
        for key in preferred:
            if key in obj:
                seq = vp._coerce_sigma_sequence(obj.get(key))
                if seq is not None:
                    return seq
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(word in key_l for word in ('sigma', 'timestep', 'schedule')):
                seq = vp._coerce_sigma_sequence(value)
                if seq is not None:
                    return seq
            if isinstance(value, dict):
                found = _find_sigma_schedule(value, depth + 1)
                if found is not None:
                    return found
            elif isinstance(value, (list, tuple)) and any(
                word in key_l for word in ('sigma', 'timestep', 'schedule')
            ):
                found = _find_sigma_schedule(value, depth + 1)
                if found is not None:
                    return found

    if isinstance(obj, (list, tuple)):
        seq = vp._coerce_sigma_sequence(obj)
        if seq is not None:
            return seq
        for item in obj:
            if isinstance(item, dict):
                found = _find_sigma_schedule(item, depth + 1)
                if found is not None:
                    return found
    return None
def _sigma_from_timestep(timestep: torch.Tensor) -> float:
    if not torch.is_tensor(timestep):
        raise RuntimeError('Sigma conversion failed: timestep is not a tensor.')
    try:
        val = float(timestep.detach().float().mean().item())
    except Exception as exc:
        raise RuntimeError('Sigma conversion failed: timestep could not be converted to float.') from exc
    if not math.isfinite(val):
        raise RuntimeError(f'Sigma conversion failed: timestep is not finite: {val!r}.')
    if 0.0 <= val <= 1.0:
        return max(0.0, min(1.0, val))
    if 1.0 < val <= 1000.0:
        return max(0.0, min(1.0, val / 1000.0))
    raise RuntimeError(f'Sigma conversion failed: unsupported timestep value {val!r}.')

def _sigma_to_progress(timestep: torch.Tensor, sampler_sigmas: List[float]) -> float:
    sigma = _sigma_from_timestep(timestep)

    if sampler_sigmas is None:
        raise RuntimeError('Progress conversion failed: sampler_sigmas is missing.')

    active: List[float] = []
    for idx, s in enumerate(sampler_sigmas):
        try:
            sf = float(s)
        except Exception as exc:
            raise RuntimeError(
                f'Progress conversion failed: sampler sigma at index {idx} is not numeric: {s!r}.'
            ) from exc
        if not math.isfinite(sf):
            raise RuntimeError(
                f'Progress conversion failed: sampler sigma at index {idx} is not finite: {s!r}.'
            )
        active.append(max(0.0, min(1.0, sf)))

    # The sampler schedule normally contains a terminal sigma=0 endpoint, but
    # the model is not evaluated there. Progress must span only real model calls
    # so *_end values are reached on the last denoising call.
    while active and active[-1] <= 1e-8:
        active.pop()

    if len(active) < 2:
        raise RuntimeError(
            'Progress conversion failed: sampler_sigmas did not contain at least two active model-call sigmas.'
        )

    idx = min(range(len(active)), key=lambda i: abs(active[i] - sigma))
    return max(0.0, min(1.0, idx / max(1, len(active) - 1)))

def _rf_record_probe_sigma(state: Dict[str, Any], debug_store: Optional[Dict[str, Any]], sigma: float) -> float:
    """Record the exact sigma value a sampler actually passed to the model during a cheap probe pass."""
    sigma_key = round(float(sigma), 6)
    probe = state.setdefault('probe_sigmas', [])
    if not isinstance(probe, list):
        probe = []
        state['probe_sigmas'] = probe
    # Keep model-call order, but avoid repeated entries from chunked cond/uncond calls at the same sigma.
    if not probe or round(float(probe[-1]), 6) != sigma_key:
        probe.append(sigma_key)
    state['last_probe_sigma'] = sigma_key
    if debug_store is not None:
        debug_store['probe_sigmas'] = list(probe)
        debug_store['last_probe_sigma'] = sigma_key
    return sigma_key

def _rf_cache_lookup(cache: Dict[float, torch.Tensor], sigma: float, *, allow_nearest: bool = True):
    """
    Exact cache lookup first; optional nearest fallback for adaptive samplers whose
    real path can differ from the zero-denoiser probe path.
    """
    sigma_key = round(float(sigma), 6)
    if not isinstance(cache, dict) or not cache:
        return None, sigma_key, 'missing'

    cached = cache.get(sigma_key, None)
    if cached is not None:
        return cached, sigma_key, 'exact'

    if not allow_nearest:
        return None, sigma_key, 'missing'

    try:
        nearest_key = min(cache.keys(), key=lambda k: abs(float(k) - sigma_key))
    except Exception:
        return None, sigma_key, 'missing'

    return cache.get(nearest_key, None), float(nearest_key), 'nearest'

def _rf_new_debug_store() -> Dict[str, Any]:
    """Reset and return the module-level RF debug store used by RFInversion runtime."""
    debug_store: Dict[str, Any] = _RF_LAST_DEBUG_STORE
    debug_store.clear()
    debug_store.update({
        'cache': {},
        'xhat_cache': {},
        'pred_cache': {},
        'xhat_plus_cache': {},
        'sampler_sigmas': None,
        'built_sigmas': None,
        'run_count': 0,
        'persistent_cache_key': None,
        'persistent_cache_hit': False,
        'parameterization': 'unknown',
        'apply_model_output': 'raw_transformer_velocity',
        'model_info': {},
        'wrapper_calls': 0,
        'last_sigma': None,
        'last_cond_mode': None,
        'last_cache_lookup': None,
        'last_error': None,
    })
    return debug_store

def _rf_make_preview_callback(model_for_preview: Any, total_steps: int) -> Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]]:
    """Create a ComfyUI-style latent preview callback for the RF trajectory."""
    total_steps = max(1, int(total_steps))
    try:
        return latent_preview.prepare_callback(model_for_preview, total_steps)
    except Exception as exc:
        raise RuntimeError('RF preview callback creation failed.') from exc

def _rf_emit_preview(
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]],
    step: int,
    denoised_preview: Optional[torch.Tensor],
    x_current: Optional[torch.Tensor],
    total_steps: int,
) -> None:
    """Emit one RF denoised/x0 prediction preview frame."""
    if callback is None:
        raise RuntimeError('RF preview failed: callback is missing.')
    if not torch.is_tensor(denoised_preview):
        raise RuntimeError('RF preview failed: denoised_preview is not a tensor.')
    try:
        preview_latent = denoised_preview[:1].detach()
        current = x_current[:1].detach() if torch.is_tensor(x_current) else preview_latent
        callback(int(step), preview_latent, current, int(total_steps))
    except Exception as exc:
        raise RuntimeError(f'RF preview frame failed at step {int(step) + 1}.') from exc
def _rf_latent_get_config(rf_inversion: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[torch.Tensor], Optional[Any], str]:
    """Read RFInversion's LATENT metadata without exposing a custom Comfy type."""
    if not isinstance(rf_inversion, dict):
        return False, {}, {}, None, None, 'not-connected'
    cfg = rf_inversion.get('untwist_rf_config', None)
    state = rf_inversion.get('untwist_rf_state', None)
    ref_clean = rf_inversion.get('untwist_ref_clean', None)
    ref_conditioning = rf_inversion.get('untwist_ref_conditioning', None)
    if not isinstance(cfg, dict):
        return False, {}, {}, None, None, 'missing-config'
    if not isinstance(state, dict):
        state = {}
        rf_inversion['untwist_rf_state'] = state
    if not torch.is_tensor(ref_clean):
        return False, cfg, state, None, ref_conditioning, 'missing-ref-clean'
    return True, cfg, state, ref_clean, ref_conditioning, 'RFInversion LATENT'
def _append_conditioning_status(mode: str, status: str) -> str:
    if status and status != 'not-applicable':
        return f'{mode};{status}'
    return mode



class RFInversion:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('MODEL', 'LATENT')
    RETURN_NAMES = ('model', 'rf_inversion')
    FUNCTION = 'build'
    DESCRIPTION = (
        'Stores RF inversion settings/reference data in a normal LATENT and captures '
        'the sampler sigma schedule internally. No SIGMAS input is required.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'reference_latent': ('LATENT',),
                'ref_conditioning': ('CONDITIONING',),
                'rf_mode': (['linear', 'rf_gamma', 'rf_gamma_rk2', 'rf_solver_2', 'endpoint_heun', 'fireflow', 'flowturbo_pc'], {
                    'default': 'rf_gamma',
                    'tooltip': (
                        'Selects the ODE solver used to build the noisy reference trajectory. '
                        'Set otip_strength > 0 to add OTIP transport guidance to the selected solver.'
                    ),
                }),
                'gamma': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Blends weight between model velocity and prior velocity (0 = pure prior / straight path, 1 = pure model).'
                }),
                'pmi_alpha': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Proximal-Mean Inversion: 0 disables PMI, 1.0 matches the official radius. Applies to RF gamma, RK2, and FireFlow.'
                }),
                'otip_strength': ('FLOAT', {
                    'default': 0.35,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Optimal Transport for Rectified Flow Image Editing. 0 disables it.'
                }),
                'otip_clip_norm': ('FLOAT', {
                    'default': 20.0,
                    'min': 0.0,
                    'max': 100.0,
                    'step': 0.01,
                    'tooltip': 'OTIP Clipping threshold for the closed-form Wasserstein-2 transport direction.'
                }),
                'verbose': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Enable verbose logging.'
                }),
            },
        }

    def build(
        self,
        model,
        reference_latent,
        rf_mode='rf_gamma_rk2',
        gamma=0.5,
        pmi_alpha=0.0,
        otip_strength=0.0,
        otip_clip_norm=10.0,
        verbose=False,
        ref_conditioning=None,
    ):
        rf_mode = str(rf_mode or 'rf_gamma')
        otip_strength = _coerce_otip_strength(otip_strength)
        # OTIP phase is intentionally hardcoded to 1.0.
        otip_phase = 1.0
        otip_clip_norm = _coerce_otip_clip_norm(otip_clip_norm)
        # OTIP respect_model_norm is intentionally hardcoded off.
        otip_respect_model_norm = False
        verbose_flag = vp._coerce_bool(verbose)

        if not isinstance(reference_latent, dict) or 'samples' not in reference_latent:
            raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with 'samples'.")

        ref_clean = reference_latent['samples'].detach().clone()
        ref_clean = model.model.process_latent_in(ref_clean)

        model_info = vp._rf_model_identity(model)
        adapter = _select_model_adapter(model, model_info)
        detected_param = 'raw_velocity'
        dm_for_ref = None
        try:
            dm_for_ref = _safe_get_diffusion_model(model, adapter)
        except Exception as exc:
            raise RuntimeError(
                'RFInversion failed: could not access diffusion model for reference conditioning preprocessing.'
            ) from exc

        cfg: Dict[str, Any] = {
            'rf_mode': str(rf_mode),
            'gamma': float(gamma),
            'pmi_alpha': float(pmi_alpha),
            'otip_strength': float(otip_strength),
            'otip_phase': float(otip_phase),
            'otip_clip_norm': float(otip_clip_norm),
            'otip_respect_model_norm': False,
            'seed': 42,
            'verbose': verbose_flag,
            'apply_model_output': 'raw_transformer_velocity',
            'model_info': model_info,
        }
        state: Dict[str, Any] = {
            'cache': {0.0: ref_clean.detach().to(device='cpu').clone()},
            'eps': None,
            'prev_z': None,
            'prev_sigma': None,
            'run_count': 0,
            'sampler_sigmas': None,
            'schedule_built': False,
            'schedule_sorted': None,
            'persistent_cache_key': None,
            'persistent_cache_hit': False,
            'preview_callback': None,
            'wrapper_calls': 0,
            'model_info': model_info,
            'last_sigma': None,
            'last_cond_mode': None,
            'last_cache_lookup': None,
            'last_error': None,
            'sigma_probe_active': False,
            'probe_sigmas': [],
            'probe_model_calls': 0,
        }
        debug_store = _rf_new_debug_store()
        debug_store['cache'] = state['cache']
        debug_store['parameterization'] = detected_param
        debug_store['apply_model_output'] = cfg['apply_model_output']
        debug_store['model_info'] = model_info
        debug_store['probe_sigmas'] = []
        debug_store['probe_model_calls'] = 0

        # Normal LATENT output: samples stay a latent tensor; extra keys carry RF metadata.
        rf_latent: Dict[str, Any] = dict(reference_latent)
        rf_latent['samples'] = reference_latent['samples']
        rf_latent['untwist_rf_config'] = cfg
        rf_latent['untwist_rf_state'] = state
        rf_latent['untwist_rf_cache'] = state['cache']
        rf_latent['untwist_rf_sigmas'] = None
        rf_latent['untwist_rf_mode'] = str(rf_mode)
        rf_latent['untwist_rf_seed'] = 42
        rf_latent['untwist_rf_parameterization'] = detected_param
        rf_latent['untwist_rf_apply_model_output'] = cfg['apply_model_output']
        rf_latent['untwist_rf_model_info'] = model_info
        rf_latent['untwist_ref_clean'] = ref_clean.detach().to(device='cpu').clone()
        rf_latent['untwist_ref_conditioning'] = ref_conditioning

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        setattr(model_clone, '_untwisting_rope_rf_state', state)
        setattr(model_clone, '_untwisting_rope_rf_config', cfg)

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            found = vp._coerce_sigma_sequence(sigmas)
            if found is not None:
                state['sampler_sigmas'] = list(found)
                state['schedule_built'] = False
                state['schedule_sorted'] = None
                state['persistent_cache_key'] = None
                state['persistent_cache_hit'] = False
                state['cache'] = {0.0: ref_clean.detach().to(device='cpu').clone()}
                state['eps'] = None
                state['run_count'] = int(state.get('run_count', 0)) + 1
                state['preview_callback'] = _rf_make_preview_callback(model_clone, max(1, len(found) - 1))
                state['probe_sigmas'] = []
                state['probe_model_calls'] = 0

                rf_latent['untwist_rf_cache'] = state['cache']
                rf_latent['untwist_rf_sigmas'] = list(found)
                rf_latent['untwist_rf_state'] = state

                debug_store['cache'] = state['cache']
                debug_store['sampler_sigmas'] = list(found)
                debug_store['probe_sigmas'] = []
                debug_store['probe_model_calls'] = 0
                debug_store['built_sigmas'] = None
                debug_store['run_count'] = int(state['run_count'])
                debug_store['persistent_cache_key'] = None
                debug_store['persistent_cache_hit'] = False
                debug_store['parameterization'] = rf_latent.get('untwist_rf_parameterization', 'unknown')

                # Cheap sigma preflight
                probe_noise = noise.detach().clone() if torch.is_tensor(noise) else noise
                probe_latent_image = latent_image.detach().clone() if torch.is_tensor(latent_image) else latent_image
                probe_denoise_mask = denoise_mask.detach().clone() if torch.is_tensor(denoise_mask) else denoise_mask
                probe_extra_args = extra_args.copy() if isinstance(extra_args, dict) else extra_args

                state['sigma_probe_active'] = True
                try:
                    executor(model_wrap, sigmas, probe_extra_args, None, probe_noise, probe_latent_image, probe_denoise_mask, True)
                finally:
                    state['sigma_probe_active'] = False

                probe_sigmas = list(state.get('probe_sigmas') or [])
                if probe_sigmas:
                    state['sampler_sigmas'] = probe_sigmas
                    state['schedule_built'] = False
                    state['schedule_sorted'] = None
                    state['persistent_cache_key'] = None
                    state['persistent_cache_hit'] = False
                    state['cache'] = {0.0: ref_clean.detach().to(device='cpu').clone()}
                    state['eps'] = None
                    state['preview_callback'] = _rf_make_preview_callback(model_clone, max(1, len(probe_sigmas) - 1))

                    rf_latent['untwist_rf_cache'] = state['cache']
                    rf_latent['untwist_rf_sigmas'] = list(probe_sigmas)
                    rf_latent['untwist_rf_state'] = state

                    debug_store['cache'] = state['cache']
                    debug_store['sampler_sigmas'] = list(probe_sigmas)
                    debug_store['probe_sigmas'] = list(probe_sigmas)
                    debug_store['built_sigmas'] = None
                    debug_store['persistent_cache_key'] = None
                    debug_store['persistent_cache_hit'] = False
                    vp._rf_print_sampler_capture(verbose_flag, probe_sigmas, state["run_count"])
                else:
                    vp._rf_print_sampler_capture(verbose_flag, found, state["run_count"])

            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            sampler_sample_wrapper,
            model_clone.model_options,
            is_model_options=True,
        )

        old_model_function_wrapper = model_clone.model_options.get('model_function_wrapper', None)
        rf_runtime_stats = vp._RuntimeStats(verbose=False, rf_verbose=verbose_flag)
        rf_runtime_stats.rf_prefix = vp._RF_PREFIX
        rf_runtime_stats.parameterization = detected_param

        def rf_model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            state['wrapper_calls'] = int(state.get('wrapper_calls', 0)) + 1
            call_n = int(state['wrapper_calls'])
            debug_store['wrapper_calls'] = call_n

            input_x = args.get('input', None)
            timestep = args.get('timestep', None)
            c_in = args.get('c', {})
            c = c_in.copy() if isinstance(c_in, dict) else {}
            to = c.get('transformer_options', {}).copy()
            _maybe_install_untwist_attention_override(to)
            c['transformer_options'] = to
            sigma = _sigma_from_timestep(timestep) if torch.is_tensor(timestep) else 1.0
            sigma_key = round(float(sigma), 6)
            state['last_sigma'] = sigma_key
            debug_store['last_sigma'] = sigma_key

            if state.get('sigma_probe_active', False):
                if not torch.is_tensor(input_x):
                    raise RuntimeError('RFInversion sigma probe received a non-tensor input.')
                state['probe_model_calls'] = int(state.get('probe_model_calls', 0)) + 1
                debug_store['probe_model_calls'] = int(state['probe_model_calls'])
                _rf_record_probe_sigma(state, debug_store, sigma)
                return torch.zeros_like(input_x)

            try:
                if not torch.is_tensor(input_x):
                    raise RuntimeError('RFInversion wrapper received a non-tensor input.')

                target_b = int(input_x.shape[0])
                rf_ref_clean = _repeat_to_batch(ref_clean.to(device=input_x.device, dtype=input_x.dtype), target_b)
                sampler_sigmas = state.get('sampler_sigmas', None)

                # Build the full sampler-grid RF trajectory once per sampler run.
                if not state.get('schedule_built', False) and sampler_sigmas is not None:
                    effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                        adapter, ref_conditioning, dm_for_ref, input_x.device,
                        c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                        rf_runtime_stats, label='RFInversion',
                    )
                    rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                    rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                    state['last_cond_mode'] = rf_cond_mode
                    debug_store['last_cond_mode'] = rf_cond_mode

                    built_cache, eps, sorted_sigmas, cache_key, _persistent_hit = _rf_ensure_trajectory_cache(
                        rf_inversion=rf_latent,
                        rf_state=state,
                        rf_cfg=cfg,
                        ref_clean_cpu=ref_clean,
                        ref_clean_for_build=rf_ref_clean,
                        ref_conditioning=ref_conditioning,
                        sampler_sigmas=list(sampler_sigmas),
                        target_b=target_b,
                        rf_cond_mode=rf_cond_mode,
                        apply_model_fn=_make_raw_velocity_apply_model_fn(apply_model),
                        base_model_kwargs=rf_kwargs,
                        device=input_x.device,
                        dtype=input_x.dtype,
                        stats=rf_runtime_stats,
                        preview_callback_factory=lambda: _rf_make_preview_callback(model_clone, max(1, len(list(sampler_sigmas)) - 1)),
                        verbose_flag=verbose_flag,
                        debug_store=debug_store,
                        parameterization=detected_param,
                    )
                    debug_store['apply_model_output'] = cfg['apply_model_output']
                    debug_store['model_info'] = model_info

                elif not state.get('schedule_built', False) and sampler_sigmas is None:
                    raise RuntimeError(
                        'RFInversion failed: sampler sigma schedule was not captured. '
                        'SAMPLER_SAMPLE did not run before the RF model wrapper was called.'
                    )

                cache = state.get('cache') if isinstance(state.get('cache'), dict) else {}
                cached, used_sigma_key, cache_lookup = _rf_cache_lookup(cache, sigma_key, allow_nearest=True)
                if cached is None:
                    raise RuntimeError(
                        f'RFInversion failed: no RF cache entry for sigma={sigma_key:.6f}.'
                    )
                state['last_cache_lookup'] = cache_lookup
                state['last_cache_key'] = float(used_sigma_key)
                debug_store['last_cache_lookup'] = cache_lookup
                debug_store['last_cache_key'] = float(used_sigma_key)


            except Exception as exc:
                state['last_error'] = repr(exc)
                debug_store['last_error'] = repr(exc)
                vp._rf_print_traceback(True, traceback.format_exc())
                raise RuntimeError('RFInversion standalone wrapper failed in.') from exc

            if old_model_function_wrapper is not None:
                return old_model_function_wrapper(apply_model, args)
            return apply_model(args['input'], args['timestep'], **args['c'])

        model_clone.set_model_unet_function_wrapper(rf_model_function_wrapper)


        return (model_clone, rf_latent)
