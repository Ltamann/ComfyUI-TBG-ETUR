from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch

_PREFIX = '[UntwistingRoPE]'
_RF_PREFIX = '[RFInversion]'

def _rf_prefix(stats: Optional[Any] = None) -> str:
    if stats is None:
        return _RF_PREFIX
    prefix = getattr(stats, 'rf_prefix', None)
    if prefix is None:
        return _RF_PREFIX
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError(f'Invalid RF prefix on stats: {prefix!r}')
    return prefix

def _coerce_bool(value: Any) -> bool:
    """Robust boolean parsing for ComfyUI values that may arrive as bools or strings."""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')
    return bool(value)

class _RuntimeStats:
    def __init__(self, verbose: bool = False, rf_verbose: bool = False) -> None:
        # verbose controls UntwistingRoPE patch/attention logs.
        # rf_verbose controls RFInversion trajectory/wrapper logs.
        self.verbose: bool = _coerce_bool(verbose)
        self.rf_verbose: bool = _coerce_bool(rf_verbose)
        self.rf_prefix: str = _RF_PREFIX
        self.wrapper_calls:  int = 0
        self.patchify_calls: int = 0
        self.attn_calls:     int = 0
        self.context_refiner_calls: int = 0
        self.adapter_attn_calls: int = 0
        self.adapter_attn_failures: int = 0

        self.rf_sigma_cache: Dict[float, torch.Tensor] = {}
        self.rf_eps: Optional[torch.Tensor] = None
        self.rf_prev_z: Optional[torch.Tensor] = None
        self.rf_prev_sigma: Optional[float] = None
        self.rf_step_count: int = 0
        self.rf_run_count: int = 0
        self.rf_sampler_sigmas: Optional[List[float]] = None
        self.rf_schedule_built: bool = False

        self.fixed_noise: Optional[torch.Tensor] = None

        self.scale_vec_logged:  bool = False
        self.joint_mask_logged: bool = False
        self.last_printed_sigma: Optional[float] = None

        # Parameterization detection: tracks whether apply_model is x0 or velocity
        self.parameterization: str = 'unknown'

def _vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'verbose', False)):
        print(*args, **kwargs)

def _rf_vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'rf_verbose', False)):
        print(*args, **kwargs)

def _rf_tensor_summary(name: str, value: Any) -> str:
    """Compact tensor diagnostic string safe for dtype/device/empty tensors."""
    if not torch.is_tensor(value):
        return f'{name}=<{type(value).__name__}>'
    try:
        shape = tuple(int(v) for v in value.shape)
        base = f'{name}: shape={shape} dtype={value.dtype} device={value.device}'
        if value.numel() == 0:
            return base + ' empty'
        vf = value.detach().float()
        return (
            f'{base} mean={float(vf.mean().item()):.6f} '
            f'std={float(vf.std(unbiased=False).item()):.6f} '
            f'min={float(vf.min().item()):.6f} max={float(vf.max().item()):.6f}'
        )
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} tensor summary failed for {name}; strict mode refuses to hide diagnostic failure: {exc}') from exc

def _rf_sequence_summary(name: str, seq: Any, max_items: int = 8) -> str:
    values = _coerce_sigma_sequence(seq)
    if values is None:
        return f'{name}=<none/invalid>'
    head = ', '.join(f'{v:.6f}' for v in values[:max_items])
    tail = ', '.join(f'{v:.6f}' for v in values[-max_items:])
    if len(values) <= max_items * 2:
        body = ', '.join(f'{v:.6f}' for v in values)
    else:
        body = f'{head}, ..., {tail}'
    return f'{name}: count={len(values)} min={min(values):.6f} max={max(values):.6f} values=[{body}]'

def _rf_brief_obj(obj: Any, depth: int = 0) -> str:
    """Small structural summary for conditioning/debug dictionaries."""
    if torch.is_tensor(obj):
        return f'Tensor{tuple(obj.shape)}:{obj.dtype}:{obj.device}'
    if obj is None:
        return 'None'
    if depth >= 2:
        return type(obj).__name__
    if isinstance(obj, dict):
        items = []
        for idx, (k, v) in enumerate(obj.items()):
            if idx >= 12:
                items.append('...')
                break
            items.append(f'{k}={_rf_brief_obj(v, depth + 1)}')
        return '{' + ', '.join(items) + '}'
    if isinstance(obj, (list, tuple)):
        items = []
        for idx, v in enumerate(obj):
            if idx >= 8:
                items.append('...')
                break
            items.append(_rf_brief_obj(v, depth + 1))
        return f'{type(obj).__name__}[{len(obj)}](' + ', '.join(items) + ')'
    return f'{type(obj).__name__}({repr(obj)[:80]})'

def _rf_tensor_stats(value: Any) -> Dict[str, Any]:
    """Numerical health stats used only for RF diagnostics."""
    out: Dict[str, Any] = {
        'is_tensor': torch.is_tensor(value),
        'finite': False,
        'numel': 0,
        'nan_count': None,
        'inf_count': None,
        'mean': None,
        'std': None,
        'min': None,
        'max': None,
        'max_abs': None,
    }
    if not torch.is_tensor(value):
        return out
    try:
        x = value.detach().float()
        out['numel'] = int(x.numel())
        if x.numel() == 0:
            out['finite'] = True
            return out
        finite = torch.isfinite(x)
        out['finite'] = bool(finite.all().item())
        out['nan_count'] = int(torch.isnan(x).sum().item())
        out['inf_count'] = int(torch.isinf(x).sum().item())
        xf = x[finite]
        if xf.numel() == 0:
            return out
        out['mean'] = float(xf.mean().item())
        out['std'] = float(xf.std(unbiased=False).item())
        out['min'] = float(xf.min().item())
        out['max'] = float(xf.max().item())
        out['max_abs'] = float(xf.abs().max().item())
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} tensor stats failed; strict mode refuses to hide diagnostic failure: {exc}') from exc
    return out

def _rf_scalar_fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None:
            return 'n/a'
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f'{value:.{digits}f}'
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} scalar formatting failed for value={value!r}: {exc}') from exc

def _rf_print_step_quality(
    stats: Optional[Any],
    ref_clean: torch.Tensor,
    eps: torch.Tensor,
    z: torch.Tensor,
    sigma_cur: float,
    delta: float,
    dz_abs: float,
    path_prev_speed: Optional[float],
) -> Optional[float]:
    """Print cheap per-step RF trajectory diagnostics and return updated speed state.

    This lives in verbose_prints.py on purpose: the RF builder should only own
    trajectory construction, while diagnostic reductions/formatting stay here.
    The function performs tensor reductions only; it does not call the model and
    does not change sampling math.
    """
    if not _coerce_bool(getattr(stats, 'rf_verbose', False)):
        return path_prev_speed

    try:
        with torch.no_grad():
            sigma_f = max(0.0, min(1.0, float(sigma_cur)))
            target_step = ((1.0 - sigma_f) * ref_clean.detach().float()) + (sigma_f * eps.detach().float())
            z_step = z.detach().float()
            diff_step = z_step - target_step

            step_linear_mae = float(diff_step.abs().mean().item())
            step_linear_rmse = float(diff_step.pow(2).mean().sqrt().item())

            flat_z = z_step.flatten()
            flat_t = target_step.flatten()
            cos_denom = flat_z.norm() * flat_t.norm()
            step_linear_cos = (
                float(torch.dot(flat_z, flat_t).div(cos_denom).item())
                if float(cos_denom.item()) > 1e-12 else float('nan')
            )

            step_speed = float(dz_abs) / max(abs(float(delta)), 1e-12)
            step_rough = (
                abs(step_speed - float(path_prev_speed))
                if path_prev_speed is not None else 0.0
            )

            step_tail_ratio = float(z_step.abs().max().item()) / max(float(z_step.std().item()), 1e-12)

            dims = tuple(range(1, z_step.ndim))
            if dims:
                z_step_mean = z_step.mean(dim=dims)
                t_step_mean = target_step.mean(dim=dims)
                step_mean_drift = float((z_step_mean - t_step_mean).abs().mean().item())

                z_step_std = z_step.std(dim=dims, unbiased=False)
                t_step_std = target_step.std(dim=dims, unbiased=False).clamp_min(1e-12)
                step_std_ratio_drift = float((z_step_std / t_step_std - 1.0).abs().mean().item())
            else:
                step_mean_drift = abs(float(z_step.mean().item()) - float(target_step.mean().item()))
                step_std_ratio_drift = 0.0
    except Exception as exc:
        raise RuntimeError(
            f'{_rf_prefix(stats)} RF step quality diagnostic failed at σ={float(sigma_cur):.6f}; '
            f'strict mode refuses to hide diagnostic failure: {exc}'
        ) from exc

    _rf_vprint(
        stats,
        f'{_rf_prefix(stats)}       step_quality '
        f'speed={step_speed:.6f}  rough={step_rough:.6f}  '
        f'linear_mae={step_linear_mae:.6f}  linear_rmse={step_linear_rmse:.6f}  '
        f'linear_cos={step_linear_cos:.6f}  tail={step_tail_ratio:.6f}  '
        f'mean_drift={step_mean_drift:.6f}  std_ratio_drift={step_std_ratio_drift:.6f}'
    )
    return step_speed


def _rf_model_identity(model_patcher: Any) -> Dict[str, Any]:
    """Best-effort model identity diagnostics; never used for math decisions."""
    base = getattr(model_patcher, 'model', model_patcher)
    diffusion_model = getattr(base, 'diffusion_model', None)
    model_config = getattr(base, 'model_config', None)
    unet_config = getattr(model_config, 'unet_config', None)
    if not isinstance(unet_config, dict):
        unet_config = {}
    model_type = getattr(base, 'model_type', None)
    model_sampling = getattr(base, 'model_sampling', None)
    latent_format = getattr(base, 'latent_format', None)
    info = {
        'base_class': type(base).__name__ if base is not None else 'None',
        'diffusion_class': type(diffusion_model).__name__ if diffusion_model is not None else 'None',
        'diffusion_module': getattr(type(diffusion_model), '__module__', '') if diffusion_model is not None else '',
        'model_type': getattr(model_type, 'name', str(model_type)),
        'model_sampling_class': type(model_sampling).__name__ if model_sampling is not None else 'None',
        'latent_format_class': type(latent_format).__name__ if latent_format is not None else 'None',
        'image_model': unet_config.get('image_model', None),
        'in_channels': unet_config.get('in_channels', None),
        'out_channels': unet_config.get('out_channels', None),
    }
    return info

def _rf_print_model_identity(prefix: str, info: Dict[str, Any]) -> None:
    print(
        f'{prefix} model_info: base={info.get("base_class")} '
        f'diffusion={info.get("diffusion_module")}.{info.get("diffusion_class")} '
        f'image_model={info.get("image_model")} adapter={info.get("architecture_name", info.get("architecture", "unknown"))}\n'
        f'{prefix} model_type={info.get("model_type")} '
        f'sampling={info.get("model_sampling_class")} '
        f'latent_format={info.get("latent_format_class")} '
        f'in_channels={info.get("in_channels")} out_channels={info.get("out_channels")}'
    )

def _rf_step_iterator(num_steps: int):
    """Plain RF step iterator.

    Do not use tqdm/model_trange here because that refreshes a single terminal
    line. RF inversion wants persistent per-step console lines, while still
    keeping the preview callback independent.
    """
    return range(max(0, int(num_steps)))

def _rf_format_duration(seconds: float) -> str:
    seconds_i = int(max(0, round(float(seconds))))
    minutes, seconds_i = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds_i:02d}'
    return f'{minutes:02d}:{seconds_i:02d}'

def _rf_progress_snapshot(step_i: int, total_steps: int, start_time: float, persistent: bool = False) -> None:
    total_steps = max(1, int(total_steps))
    step_i = max(0, min(int(step_i), total_steps))

    elapsed = max(0.0, time.time() - float(start_time))
    frac = step_i / total_steps

    bar_width = 70
    filled = int(round(bar_width * frac))
    bar = '█' * filled + ' ' * (bar_width - filled)

    percent = int(round(frac * 100.0))
    rate = step_i / elapsed if elapsed > 1e-9 else 0.0
    remaining = max(0.0, (total_steps - step_i) / rate) if step_i > 0 and rate > 1e-9 else 0.0
    rate_text = f'{rate:.2f}it/s' if rate >= 1.0 else f'{(1.0 / max(rate, 1e-9)):.2f}s/it'

    line = (
        f'RF inversion: {percent:3d}%|{bar}| '
        f'{step_i}/{total_steps} '
        f'[{_rf_format_duration(elapsed)}<{_rf_format_duration(remaining)}, {rate_text}]'
    )

    end = '\n' if persistent or step_i >= total_steps else '\r'
    print(line, end=end, flush=True)



def _normalize_sigma_float(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        v = float(value.detach().float().mean().item())
    else:
        v = float(value)
    if not math.isfinite(v):
        raise ValueError(f'Invalid sigma/timestep value {value!r}: not finite.')
    if 0.0 <= v <= 1.0:
        return max(0.0, min(1.0, v))
    if 1.0 < v <= 1000.0:
        return max(0.0, min(1.0, v / 1000.0))
    raise ValueError(f'Invalid sigma/timestep value {value!r}; expected [0,1] sigma or [1,1000] timestep.')

def _coerce_sigma_sequence(value: Any) -> Optional[List[float]]:
    """Convert a scheduler sigma/timestep list into normalized [0,1] floats.

    Strict mode: malformed schedules raise instead of returning None. A missing
    value still returns None so callers can distinguish "not provided" from
    "provided but broken".
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.detach().float().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            if torch.is_tensor(item):
                flat.extend(item.detach().float().flatten().tolist())
            elif isinstance(item, (int, float)):
                flat.append(float(item))
            else:
                raise ValueError(f'Invalid sigma sequence item {item!r}; expected tensor/int/float.')
    else:
        raise ValueError(f'Invalid sigma sequence type {type(value).__name__}; expected tensor/list/tuple.')

    out: List[float] = [_normalize_sigma_float(item) for item in flat]
    if len(out) < 2:
        raise ValueError(f'Invalid sigma sequence; expected at least 2 values, got {len(out)}.')
    dedup: List[float] = []
    for sigma in out:
        if not dedup or abs(dedup[-1] - sigma) > 1e-6:
            dedup.append(sigma)
    if len(dedup) < 2:
        raise ValueError(f'Invalid sigma sequence; deduplicated length is {len(dedup)}.')
    return dedup

def _rf_print_sampler_capture(verbose_flag: Any, found: Any, run_count: Any) -> None:
    if not _coerce_bool(verbose_flag):
        return

def _rf_print_persistent_cache_hit(verbose_flag: Any, cache_key: str, built_cache: Any) -> None:
    if _coerce_bool(verbose_flag):
        print(f'{_RF_PREFIX}   RF persistent cache HIT key={str(cache_key)[:12]} cache_items={len(built_cache)}')

def _rf_print_persistent_cache_miss(verbose_flag: Any, cache_key: str) -> None:
    if _coerce_bool(verbose_flag):
        print(f'{_RF_PREFIX}   RF persistent cache MISS key={str(cache_key)[:12]} → building now')

def _rf_print_direct_fallback(verbose_flag: Any, sigma: float) -> None:
    raise RuntimeError(
        f'{_RF_PREFIX} No sampler sigmas captured; direct one-step RF path is disabled for σ={float(sigma):.6f}'
    )

def _rf_print_traceback(verbose_flag: Any, trace_text: str) -> None:
    if _coerce_bool(verbose_flag) and trace_text:
        print(trace_text)


def _rf_print_prepared(
    verbose_flag: Any,
    rf_mode: str,
    gamma: float,
    gamma_curve: float,
    norm_strength: float,
    pmi_alpha: float,
    model_info: Dict[str, Any],
) -> None:
    if not _coerce_bool(verbose_flag):
        return
    print(f'\n{_RF_PREFIX} ═══════════════════════════════════════')
    print(f'{_RF_PREFIX} RF INVERSION PREPARED')
    print(f'{_RF_PREFIX} ═══════════════════════════════════════')
    print(f'{_RF_PREFIX}   mode          : {rf_mode}')
    print(f'{_RF_PREFIX}   gamma         : {float(gamma):.4f}')
    print(f'{_RF_PREFIX}   gamma_curve   : {float(gamma_curve):.3f}')
    print(f'{_RF_PREFIX}   norm_strength : {float(norm_strength):.3f}')
    print(f'{_RF_PREFIX}   pmi_alpha     : {float(pmi_alpha):.3f}')
    print(f'{_RF_PREFIX}   seed          : 42 (internal fixed noise seed)')
    print(f'{_RF_PREFIX}   schedule      : captured from sampler at runtime; no SIGMAS input')
    print(f'{_RF_PREFIX}   output        : normal LATENT with RF metadata')
    print(f'{_RF_PREFIX}   wrapper       : standalone RF cache builder installed on MODEL')
    print(f'{_RF_PREFIX}   diagnostics   : verbose=True prints per-call/cache/conditioning details')
    _rf_print_model_identity(f'{_RF_PREFIX}   RFInversion', model_info)
    print(f'{_RF_PREFIX} ═══════════════════════════════════════\n')


def _untwist_format_scale_value(value: Any) -> str:
    """Format one scale value for compact debug logs without noisy trailing zeros."""
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f'Invalid RoPE scale value {value!r}: not finite.')
    text = f'{value_f:.6f}'.rstrip('0').rstrip('.')
    return text if text else '0'


def _untwist_scale_range(values: Any) -> str:
    """Return a compact [first ... last] summary from a 1D scale tensor/list."""
    if torch.is_tensor(values):
        if values.numel() <= 0:
            raise ValueError('Cannot summarize empty RoPE scale tensor.')
        flat = values.detach().float().flatten().to(device='cpu')
    else:
        flat = torch.as_tensor(values, dtype=torch.float32).flatten()
        if flat.numel() <= 0:
            raise ValueError('Cannot summarize empty RoPE scale values.')
    first = _untwist_format_scale_value(float(flat[0].item()))
    last = _untwist_format_scale_value(float(flat[-1].item()))
    return f'[{first} ... {last}]'


_AXIS0_ROPE_MODES = {'default', 'match_axes', 'constant'}

def _untwist_coerce_axis0_rope_mode(value: Any = None, legacy_scale: Any = None) -> str:
    """Mirror the main node's axis-0 RoPE mode normalization for debug output."""
    if value is None:
        if legacy_scale is not None:
            legacy_f = float(legacy_scale)
            return 'default' if legacy_f < 0.0 else 'constant'
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
        raise ValueError(f'Invalid axis0_rope_mode={value!r}; expected one of {sorted(_AXIS0_ROPE_MODES)}.')
    return mode

def _untwist_coerce_axis0_rope_scale(value: Any, default: float = 0.0) -> float:
    """Debug-side formatting clamp: axis0_rope_scale is non-negative now."""
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f'Invalid axis0_rope_scale={value!r}: not finite.')
    if value_f < 0.0:
        raise ValueError(f'Invalid axis0_rope_scale={value_f!r}; expected non-negative value.')
    return value_f


def _untwist_print_rope_scale_debug(
    stats: Optional[_RuntimeStats],
    cfg: Dict[str, Any],
    module_name: str,
    scale_vec: Any,
) -> None:
    """Print one compact RoPE scale-vector snapshot per UntwistingRoPE model call."""
    if not _coerce_bool(getattr(stats, 'verbose', False)):
        return
    if not isinstance(cfg, dict) or cfg.get('_rope_scale_debug_printed', False):
        return
    if not cfg.get('enabled', False) or int(cfg.get('cross_batch_target_batch', 0)) <= 0:
        return

    sigma = float(cfg.get('sigma', 0.0))
    if stats is not None:
        last_sigma = getattr(stats, 'last_printed_sigma', None)
        if last_sigma is not None and abs(last_sigma - sigma) < 1e-6:
            cfg['_rope_scale_debug_printed'] = True
            return
        stats.last_printed_sigma = sigma

    cfg['_rope_scale_debug_printed'] = True

    try:
        if not torch.is_tensor(scale_vec):
            scale_vec = torch.as_tensor(scale_vec)
        flat = scale_vec.detach().flatten()
        head_dim = int(flat.numel())
        axes_dims = [int(x) for x in (cfg.get('axes_dims') or [])]
        if not axes_dims or sum(axes_dims) != head_dim:
            axes_dims = [head_dim]

        if len(axes_dims) >= 2:
            axis0_dim = max(0, min(int(axes_dims[0]), head_dim))
            axis0 = flat[:axis0_dim]
            axis1_plus = flat[axis0_dim:]
        else:
            axis0 = flat
            axis1_plus = flat.new_empty((0,))

        progress = float(cfg.get('progress', 0.0))
        low_scale = float(cfg.get('_debug_low_scale', 0.0))
        high_scale = float(cfg.get('_debug_high_scale', 0.0))
        axis0_rope_mode = _untwist_coerce_axis0_rope_mode(
            cfg.get('axis0_rope_mode', None),
            legacy_scale=cfg.get('axis0_rope_scale', None),
        )
        axis0_rope_scale = _untwist_coerce_axis0_rope_scale(
            cfg.get('axis0_rope_scale', 0.0), default=0.0
        )

        print(f'{_PREFIX}   progress={progress:.6f}  sigma={sigma:.6f}')
        print(f'{_PREFIX}   low_scale={low_scale:.6f}  high_scale={high_scale:.6f}')
        print(f'{_PREFIX}   axis0={_untwist_scale_range(axis0)}')
        print(f'{_PREFIX}   axis1+={_untwist_scale_range(axis1_plus)}')
    except Exception as exc:
        raise RuntimeError(f'{_PREFIX} RoPE scale debug print failed; strict mode refuses to hide debug failure: {exc}') from exc


def _untwist_print_rope_scale_debug_from_cfg(
    stats: Optional[_RuntimeStats],
    cfg: Dict[str, Any],
    module_name: str,
    device: Any,
    dtype: Any,
    build_frequency_scale_vector: Any,
) -> None:
    """Strict RoPE debug print for adapter attention paths.

    The actual scale-vector builder stays in the main module and is passed in here
    so verbose_prints owns the formatting/printing without creating import cycles.
    """
    if not _coerce_bool(getattr(stats, 'verbose', False)):
        return
    if not isinstance(cfg, dict) or cfg.get('_rope_scale_debug_printed', False):
        return
    if not cfg.get('enabled', False) or int(cfg.get('cross_batch_target_batch', 0)) <= 0:
        return

    sigma = float(cfg.get('sigma', 0.0))
    if stats is not None:
        last_sigma = getattr(stats, 'last_printed_sigma', None)
        if last_sigma is not None and abs(last_sigma - sigma) < 1e-6:
            cfg['_rope_scale_debug_printed'] = True
            return

    if not callable(build_frequency_scale_vector):
        raise RuntimeError('RoPE scale debug requires callable build_frequency_scale_vector.')
    try:
        head_dim = int(cfg.get('head_dim', 0))
        if head_dim <= 0:
            raise RuntimeError(f'RoPE scale debug requires positive head_dim, got {head_dim}.')

        progress = float(cfg.get('progress', 0.0))
        high_start = float(cfg['high_scale_start'])
        high_end = float(cfg['high_scale_end'])
        low_start = float(cfg['low_scale_start'])
        low_end = float(cfg['low_scale_end'])
        high_scale = high_start + (high_end - high_start) * progress
        low_scale = low_start + (low_end - low_start) * progress
        beta = float(cfg.get('beta', 2.0))

        cfg['_debug_high_scale'] = float(high_scale)
        cfg['_debug_low_scale'] = float(low_scale)

        scale_vec = build_frequency_scale_vector(
            head_dim,
            cfg.get('axes_dims') or [],
            high_scale,
            low_scale,
            beta,
            device,
            dtype,
            runtime_cfg=cfg,
        )
        _untwist_print_rope_scale_debug(stats, cfg, module_name, scale_vec)
    except Exception as exc:
        raise RuntimeError(f'{_PREFIX} RoPE scale debug failed; strict mode refuses to hide failure: {exc}') from exc

def _untwist_print_patch_complete(stats: Optional[_RuntimeStats], rf_active: bool, adapter: Any) -> None:
    _vprint(stats, f'\n{_PREFIX} ═══════════════════════════════════════')
    _vprint(stats, f'{_PREFIX} PATCH COMPLETE')
    if rf_active:
        _vprint(stats, f'{_PREFIX}   RF input      : LATENT from RFInversion')
        _vprint(stats, f'{_PREFIX}   RF schedule   : captured internally by RFInversion model wrapper')
        _vprint(stats, f'{_PREFIX}   RF preview    : emitted while building inversion trajectory')
        uses_kv = getattr(adapter, 'uses_reference_branch_kv', lambda: False)
        if bool(uses_kv()):
            _vprint(stats, f'{_PREFIX}   K/V           : reference branch contributes K and V; only K is untwisted')
    else:
        _vprint(stats, f'{_PREFIX}   RF input      : not connected')
        _vprint(stats, f'{_PREFIX}   Mode          : target-only attention patch')
    _vprint(stats, f'{_PREFIX}   Output: target prediction returned unchanged')
    _vprint(stats, f'{_PREFIX} ═══════════════════════════════════════\n')

__all__ = [
    '_PREFIX',
    '_RF_PREFIX',
    '_rf_prefix',
    '_coerce_bool',
    '_RuntimeStats',
    '_vprint',
    '_rf_vprint',
    '_rf_tensor_summary',
    '_rf_sequence_summary',
    '_rf_brief_obj',
    '_rf_tensor_stats',
    '_rf_scalar_fmt',
    '_rf_model_identity',
    '_rf_print_model_identity',
    '_rf_print_step_quality',
    '_rf_step_iterator',
    '_rf_format_duration',
    '_rf_progress_snapshot',
    '_normalize_sigma_float',
    '_coerce_sigma_sequence',
    '_rf_print_sampler_capture',
    '_rf_print_persistent_cache_hit',
    '_rf_print_persistent_cache_miss',
    '_rf_print_direct_fallback',
    '_rf_print_traceback',
    '_rf_print_prepared',
    '_untwist_format_scale_value',
    '_untwist_scale_range',
    '_untwist_coerce_axis0_rope_mode',
    '_untwist_coerce_axis0_rope_scale',
    '_untwist_print_rope_scale_debug',
    '_untwist_print_rope_scale_debug_from_cfg',
    '_untwist_print_patch_complete',
]
