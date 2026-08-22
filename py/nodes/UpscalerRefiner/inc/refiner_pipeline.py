"""Shared contracts for the ETUR refiner pipeline.

This module deliberately contains no ComfyUI node or model code.  It provides
the small amount of state needed to keep model-specific sampling separate from
the image, mask, and compositing stages shared by every refiner.
"""

from dataclasses import dataclass, field
from typing import Any
from . import sampler_adapters
from .tile_geometry import TileGeometry


STAGE_IDS = (
    "tile_preparation",
    "geometry_normalization",
    "segment_mask_preparation",
    "neighbor_fusion",
    "segment_fusion",
    "conditioning",
    "latent_encoding",
    "model_sampling",
    "vae_decode",
    "decoded_rgb_normalization",
    "decoded_size_normalization",
    "sift_drift_correction",
    "border_correction",
    "tile_color_correction",
    "segment_post_decode_correction",
    "tile_compositing",
    "segment_compositing",
    "final_color_correction",
    "final_rebuild",
)


RGB_STAGE_ORDER = (
    "decoded_rgb_normalization",
    "geometry_normalization",
    "neighbor_fusion",
    "segment_fusion",
    "tile_color_correction",
    "sift_drift_correction",
    "border_correction",
    "segment_post_decode_correction",
    "tile_compositing",
    "segment_compositing",
    "final_color_correction",
    "final_rebuild",
)


LEGACY_STAGE_ALIASES = {
    "03_Flux2_PID_NormalVAE_Reference": "vae_decode",
    "04_Flux2_PID_AfterPiDVAE_ColorMatch": "tile_color_correction",
    "05_Flux2_PID_PostTone_ColorMatch": "tile_color_correction",
    "07_Tile_AfterFusion_ColorMatch": "tile_color_correction",
    "08_Segment_PostVAE_ColorMatch": "segment_post_decode_correction",
    "10_Final_TileOnly_ColorCorrection": "final_color_correction",
    "11_Final_SegmentAware_ColorBase": "segment_compositing",
    "12_PID_Final_ColorMatch_4x": "final_color_correction",
    "13_Final_PerArea_SegmentOverrides": "final_color_correction",
    "14_Final_Global_ColorMode": "final_color_correction",
    "15_Segment_Background_Harmonization": "segment_post_decode_correction",
}


MODEL_TYPES = (
    "FLUX1",
    "FLUX2",
    "FLUX1 Kontext",
    "Qwen Image",
    "Qwen Image Edit",
    "Krea2",
    "SDXL",
    "SD3",
    "Z-Image",
    "Ideogram4",
)


@dataclass
class RGBConfig:
    """Normalized RGB policy used after model decode."""

    tile_color_correction: bool = True
    pid_normal_vae_reference: bool = False
    segment_harmonization: bool = True
    final_color_correction: bool = True
    detail_aware_stitch: bool = True
    color_match_method: str = "none"
    color_match_strength: float = 1.0
    enabled_stages: list[str] = field(default_factory=lambda: list(STAGE_IDS))
    override_normal_gates: bool = False


@dataclass
class SamplerResult:
    """Normalized sampler output shared by every model adapter."""

    latent: Any
    source_type: str = "sampler"
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_rgb_config(value=None, defaults=None):
    """Translate workflow-era RGB settings into one runtime policy."""
    source = value if isinstance(value, dict) else {}
    base = defaults if isinstance(defaults, RGBConfig) else RGBConfig()

    def get_bool(name, legacy_name, default):
        if name in source:
            return bool(source[name])
        if legacy_name in source:
            return bool(source[legacy_name])
        return bool(default)

    method = source.get("color_match_method", source.get("Color_Match", base.color_match_method))
    try:
        strength = float(source.get("color_match_strength", source.get("Color_Match_Str", base.color_match_strength)))
    except (TypeError, ValueError):
        strength = base.color_match_strength

    enabled = source.get("enabled_stages", source.get("enabled"))
    if not isinstance(enabled, (list, tuple)):
        enabled = list(STAGE_IDS)
    enabled = [
        LEGACY_STAGE_ALIASES.get(str(item), str(item))
        for item in enabled
        if LEGACY_STAGE_ALIASES.get(str(item), str(item)) in STAGE_IDS
    ]
    return RGBConfig(
        tile_color_correction=get_bool(
            "RGB_Tile_Color_Correction", "Flux2_Tile_Color_Correction", base.tile_color_correction
        ),
        pid_normal_vae_reference=get_bool(
            "RGB_PID_Normal_VAE_Color_Reference",
            "Flux2_PiD_Normal_VAE_Color_Match",
            base.pid_normal_vae_reference,
        ),
        segment_harmonization=get_bool(
            "Segment_Background_Harmonization",
            "Segment_Background_Harmonization",
            base.segment_harmonization,
        ),
        final_color_correction=get_bool(
            "Final_Color_Correction", "Final_Color_Correction", base.final_color_correction
        ),
        detail_aware_stitch=get_bool(
            "Detail_Aware_Stitch", "Detail_Aware_Stitch", base.detail_aware_stitch
        ),
        color_match_method=str(method),
        color_match_strength=max(0.0, min(1.0, strength)),
        enabled_stages=enabled,
        override_normal_gates=bool(
            source.get("Override_Normal_Gates", source.get("override", base.override_normal_gates))
        ),
    )


def normalize_stage_config(value):
    """Normalize the new list form and the legacy ColorMatch dict form."""
    if not isinstance(value, dict):
        return {
            "_connected": False,
            "enabled": list(STAGE_IDS),
            "override": False,
            "aliases": dict(LEGACY_STAGE_ALIASES),
        }

    override = bool(value.get("Override_Normal_Gates", value.get("override", False)))
    enabled = value.get("enabled")
    if isinstance(enabled, (list, tuple)):
        enabled = [
            LEGACY_STAGE_ALIASES.get(str(item), str(item))
            for item in enabled
            if LEGACY_STAGE_ALIASES.get(str(item), str(item)) in STAGE_IDS
        ]
    else:
        enabled = list(STAGE_IDS)

    # Existing workflows contain one boolean per ColorMatch stage.  A false
    # legacy gate disables its corresponding shared stage; true gates remain
    # compatible with the old default behavior.
    for legacy_name, stage_id in LEGACY_STAGE_ALIASES.items():
        if legacy_name in value and not bool(value[legacy_name]):
            enabled = [item for item in enabled if item != stage_id]

    return {
        "_connected": bool(value.get("_connected", True)),
        "enabled": enabled,
        "override": override,
        "aliases": dict(LEGACY_STAGE_ALIASES),
    }


@dataclass
class StageEvent:
    stage_id: str
    status: str
    tile_index: int | None = None
    segment_index: int | None = None
    model_type: str | None = None
    input_space: str | None = None
    output_space: str | None = None
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class StageRegistry:
    """Per-run stage configuration and applied-stage diagnostics."""

    def __init__(self, config=None, model_type=None):
        self.config = normalize_stage_config(config)
        self.model_type = model_type
        self.events = []

    def enabled(self, stage_id, normal_active=True):
        if stage_id not in STAGE_IDS:
            raise KeyError(f"Unknown ETUR stage: {stage_id}")
        if not self.config["_connected"]:
            return bool(normal_active)
        if stage_id in self.config["enabled"]:
            return True if self.config["override"] else bool(normal_active)
        return False

    def record(self, stage_id, status, tile_index=None, segment_index=None,
               input_space=None, output_space=None, reason=None, **metrics):
        event = StageEvent(
            stage_id=stage_id,
            status=status,
            tile_index=tile_index,
            segment_index=segment_index,
            model_type=self.model_type,
            input_space=input_space,
            output_space=output_space,
            reason=reason,
            metrics=metrics,
        )
        self.events.append(event)
        return event

    def mark(self, stage_id, normal_active=True, tile_index=None,
             segment_index=None, input_space=None, output_space=None,
             reason=None, **metrics):
        active = self.enabled(stage_id, normal_active)
        self.record(
            stage_id,
            "applied" if active else "skipped",
            tile_index=tile_index,
            segment_index=segment_index,
            input_space=input_space,
            output_space=output_space,
            reason=reason if active else (reason or "disabled"),
            **metrics,
        )
        return active

    def export(self):
        return [
            {
                "stage_id": event.stage_id,
                "status": event.status,
                "tile_index": event.tile_index,
                "segment_index": event.segment_index,
                "model_type": event.model_type,
                "input_space": event.input_space,
                "output_space": event.output_space,
                "reason": event.reason,
                "metrics": dict(event.metrics),
            }
            for event in self.events
        ]


@dataclass
class TileExecutionContext:
    tile_index: int
    segment_index: int | None
    model_type: str
    tile: Any = None
    inpaint_mask: Any = None
    complexity_mask: Any = None
    border_mask: Any = None
    segment_inpaint_mask: Any = None
    segment_compositing_mask: Any = None
    latent: Any = None
    decoded: Any = None
    sampler_metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_spaces: dict[str, Any] = field(default_factory=dict)
    rgb: Any = None
    source_tile: Any = None
    masks: dict[str, Any] = field(default_factory=dict)
    geometry: Any = None
    conditioning: Any = None
    sampler_result: SamplerResult | None = None
    rgb_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RGBImageContract:
    """Normalized RGB output handed from model decode to shared ETUR stages."""

    image: Any
    tile_index: int
    segment_index: int | None
    source_type: str
    coordinate_space: str
    reference: Any = None
    border_mask: Any = None
    fusion_mask: Any = None
    inpaint_mask: Any = None
    compositing_mask: Any = None
    native_size: tuple[int, int] | None = None
    sampling_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_rgb_contract(image, tile_index, segment_index=None,
                           source_type="vae", coordinate_space="native_tile",
                           reference=None, border_mask=None, fusion_mask=None,
                           inpaint_mask=None, compositing_mask=None,
                           native_size=None, sampling_size=None, **metadata):
    """Normalize a decoded image without consulting model type."""
    if image is None or not hasattr(image, "ndim"):
        raise ValueError("ETUR RGB contract requires a tensor image")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or int(image.shape[-1]) != 3:
        raise ValueError(f"ETUR RGB contract requires BHWC RGB, got {tuple(image.shape)}")
    image = image.clamp(0.0, 1.0)
    if native_size is None:
        native_size = (int(image.shape[2]), int(image.shape[1]))
    if sampling_size is None:
        sampling_size = native_size
    return RGBImageContract(
        image=image,
        tile_index=int(tile_index),
        segment_index=None if segment_index is None else int(segment_index),
        source_type=str(source_type),
        coordinate_space=str(coordinate_space),
        reference=reference,
        border_mask=border_mask,
        fusion_mask=fusion_mask,
        inpaint_mask=inpaint_mask,
        compositing_mask=compositing_mask,
        native_size=tuple(native_size),
        sampling_size=tuple(sampling_size),
        metadata=dict(metadata),
    )


class RGBPostProcessPipeline:
    """Model-independent RGB stage order and execution bookkeeping."""

    stage_order = RGB_STAGE_ORDER

    @classmethod
    def run(cls, registry, contract, callbacks=None, active_stages=None):
        """Run registered RGB callbacks in the canonical order."""
        callbacks = callbacks or {}
        active_stages = active_stages or {}
        for stage_id in cls.stage_order:
            callback = callbacks.get(stage_id)
            normal_active = active_stages.get(stage_id, True)
            try:
                contract = cls.run_stage(
                    registry,
                    stage_id,
                    contract,
                    callback=callback,
                    normal_active=normal_active,
                    reason="shared RGB pipeline",
                )
            except Exception as exc:
                if registry is not None:
                    registry.record(
                        stage_id,
                        "failed",
                        tile_index=contract.tile_index,
                        segment_index=contract.segment_index,
                        input_space=contract.coordinate_space,
                        output_space=contract.coordinate_space,
                        reason=str(exc),
                    )
                raise
        return contract

    @staticmethod
    def run_stage(registry, stage_id, contract, callback=None, normal_active=True,
                  output_space=None, reason=None):
        if registry is not None:
            active = registry.mark(
                stage_id,
                normal_active=normal_active,
                tile_index=contract.tile_index,
                segment_index=contract.segment_index,
                input_space=contract.coordinate_space,
                output_space=output_space or contract.coordinate_space,
                reason=reason,
            )
        else:
            active = bool(normal_active)
        if not active or callback is None:
            return contract
        result = callback(contract)
        if result is None:
            return contract
        if isinstance(result, RGBImageContract):
            return result
        contract.image = result
        if output_space is not None:
            contract.coordinate_space = output_space
        return contract


class ModelAdapter:
    """Boundary for model-specific conditioning, latent, sampling, and decode."""

    model_types = MODEL_TYPES

    def __init__(self, model_type):
        self.model_type = model_type
        self.sampler_adapter = sampler_adapters.SamplerAdapter()

    def supports(self, stage_id):
        return stage_id in STAGE_IDS

    def prepare_conditioning(self, owner, *args, **kwargs):
        return owner._conditioning_model_specific(*args, **kwargs)

    def encode_tile(self, encode):
        return encode()

    def prepare_latent(self, latent, mask, source, set_mask=None, hook_active=False):
        if mask is not None and "mask" not in source.lower():
            latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            source = "VAEEncode_noise_mask"
        return latent, source

    def resolve_sampler_path(self, sampler, direct_sampler_name=None):
        return False, False

    def sample(self, latent_output):
        return SamplerResult(
            latent=latent_output,
            source_type="sampler",
            metadata={"adapter": self.__class__.__name__},
        )

    def sample_tile(self, owner, **kwargs):
        return self.sampler_adapter.sample(owner, **kwargs)

    def decode(self, decode):
        return decode()


class Flux2Adapter(ModelAdapter):
    model_types = ("FLUX2",)

    def __init__(self, model_type):
        super().__init__(model_type)
        self.sampler_adapter = sampler_adapters.Flux2SamplerAdapter()

    def resolve_sampler_path(self, sampler, direct_sampler_name=None):
        hook_active = bool(getattr(sampler, "Flux2_Sampler_Hook", False))
        direct = (
            hook_active
            and getattr(sampler, "sampler_input", None) is None
            and getattr(sampler, "sampler_name", None) == direct_sampler_name
        )
        return hook_active, direct

    def prepare_latent(self, latent, mask, source, set_mask=None, hook_active=False):
        if hook_active:
            latent["_flux2_inpaint_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            from .flux2_differential import DEFAULT_CONFIG
            latent["_flux2_differential"] = dict(DEFAULT_CONFIG)
            return latent, "VAEEncode_private_flux2_mask"
        if mask is not None and set_mask is not None:
            latent = set_mask(latent, mask)
        return latent, "Flux2_VAEEncode_SetLatentNoiseMask"

    def sample(self, latent_output):
        return SamplerResult(
            latent=latent_output,
            source_type="flux2_sampler",
            metadata={"adapter": "Flux2Adapter"},
        )

    def sample_tile(self, owner, **kwargs):
        return self.sampler_adapter.sample(owner, **kwargs)


class FluxAdapter(ModelAdapter):
    model_types = ("FLUX1", "FLUX1 Kontext")

    def __init__(self, model_type):
        super().__init__(model_type)
        self.sampler_adapter = sampler_adapters.FluxSamplerAdapter()


class QwenAdapter(ModelAdapter):
    model_types = ("Qwen Image", "Qwen Image Edit")

    def __init__(self, model_type):
        super().__init__(model_type)
        self.sampler_adapter = sampler_adapters.QwenSamplerAdapter()


class Krea2Adapter(ModelAdapter):
    model_types = ("Krea2",)

    def __init__(self, model_type):
        super().__init__(model_type)
        self.sampler_adapter = sampler_adapters.Krea2SamplerAdapter()


class GenericAdapter(ModelAdapter):
    model_types = ("SDXL", "SD3", "Z-Image", "Ideogram4")

    def __init__(self, model_type):
        super().__init__(model_type)
        self.sampler_adapter = (
            sampler_adapters.IdeogramSamplerAdapter()
            if model_type == "Ideogram4"
            else sampler_adapters.GenericSamplerAdapter()
        )

    def sample_tile(self, owner, **kwargs):
        return self.sampler_adapter.sample(owner, **kwargs)


ADAPTERS = {
    model_type: adapter
    for adapter in (Flux2Adapter, FluxAdapter, QwenAdapter, Krea2Adapter, GenericAdapter)
    for model_type in adapter.model_types
}


def get_model_adapter(model_type):
    adapter = ADAPTERS.get(model_type, GenericAdapter)
    return adapter(model_type)
