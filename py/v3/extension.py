from __future__ import annotations

import inspect
import re
from typing import Any

from typing_extensions import override

from comfy_api.v0_0_2 import ComfyExtension, io, ui

from ..nodes.UpscalerRefiner.TBG_Helpers import (
    AISourcePatternCleanup,
    FluxKontextResolution,
    ImageComplexityMap,
    IncrementBatch,
    MaskConditionalPaint,
    MaskGrayValueScaler,
    MaskMultiplyAdvanced,
    MaskMultiplyZeroWins,
    MaskToDenoiseInterpolator,
    MaskUpdateInpaintFromConditioning,
    QwenEditResolution,
    TBG_ETUR_SIFT_Drift_Correction,
)
from ..nodes.UpscalerRefiner.TBG_Nodes_CE import (
    TBG_ETUR_Refiner_CE,
    TBG_ETUR_Upscaler_and_Tile_Generator_CE,
)
from ..nodes.UpscalerRefiner.TBG_Nodes_PRO import (
    TBG_ETUR_ColorCorrection,
    TBG_ETUR_ColorMatch_Debug,
    TBG_ETUR_Labs_Refiner,
    TBG_ETUR_Labs_Upscaler,
    TBG_ETUR_Refiner_PRO,
    TBG_ETUR_Upscaler_and_Tile_Generator_PRO,
    TBG_Model_Agnostic_Color_Anchor,
    TBG_Model_Agnostic_Latent_Anchor,
)
from ..nodes.UpscalerRefiner.TBG_Pipes import (
    TBG_ControlNetPipeline,
    TBG_RF_UntwistingRoPE_Pipe,
    TBG_TilePrompter_v1,
    TBG_enrichment_pipe,
)
from ..nodes.UpscalerRefiner.TBG_PID_Upscale import TBG_ETUR_Download_PID_Model, TBG_ETUR_PID_Tile_Upscale_Rebuild
from ..nodes.UpscalerRefiner.TBG_Flux2_Sampler import TBGFlux2Sampler
from ..nodes.UpscalerRefiner.inc.pid_sde_sampler_registry import TBG_PiD_Creative_SDE_Sampler


_BUILTIN_TYPES = {
    "STRING": io.String,
    "INT": io.Int,
    "FLOAT": io.Float,
    "BOOLEAN": io.Boolean,
    "IMAGE": io.Image,
    "MASK": io.Mask,
    "MODEL": io.Model,
    "CLIP": io.Clip,
    "VAE": io.Vae,
    "STYLE_MODEL": io.StyleModel,
    "CLIP_VISION": io.ClipVision,
    "CONTROL_NET": io.ControlNet,
    "MODEL_PATCH": io.ModelPatch,
    "UPSCALE_MODEL": io.UpscaleModel,
    "SEGS": io.SEGS,
    "SAMPLER": io.Sampler,
    "SIGMAS": io.Sigmas,
    "CONDITIONING": io.Conditioning,
    "LATENT": io.Latent,
}

_HIDDEN_MAP = {
    "UNIQUE_ID": io.Hidden.unique_id,
    "PROMPT": io.Hidden.prompt,
    "EXTRA_PNGINFO": io.Hidden.extra_pnginfo,
}


def _snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    if not s:
        s = "input"
    if s[0].isdigit():
        s = f"n_{s}"
    return s.lower()


def _call_input_types(legacy_cls: type) -> dict[str, Any]:
    fn = getattr(legacy_cls, "INPUT_TYPES", None)
    if fn is None:
        return {"required": {}, "optional": {}, "hidden": {}}
    try:
        return fn()
    except TypeError:
        return fn(legacy_cls)


def _resolve_io_type(type_name: str):
    return _BUILTIN_TYPES.get(type_name, io.Custom(type_name))

def _extract_hidden_marker(hidden_spec: Any):
    # Legacy hidden specs can be: "UNIQUE_ID", ("STRING", {...}), or dict metadata.
    if isinstance(hidden_spec, str):
        return hidden_spec
    if isinstance(hidden_spec, (list, tuple)) and hidden_spec:
        first = hidden_spec[0]
        if isinstance(first, str):
            return first
    if isinstance(hidden_spec, dict):
        for key in ("type", "hidden", "kind"):
            value = hidden_spec.get(key)
            if isinstance(value, str):
                return value
    return None


def _extract_default_value(spec: Any):
    if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict):
        if "default" in spec[1]:
            return spec[1]["default"]
    if isinstance(spec, dict) and "default" in spec:
        return spec["default"]
    return None


def _build_input(io_type, input_id: str, optional: bool, options: dict[str, Any]):
    display_name = options.get("label")
    tooltip = options.get("tooltip") or options.get("tooltips")

    input_cls = io_type.Input
    accepted = set(inspect.signature(input_cls).parameters.keys())

    kwargs = {
        "display_name": display_name,
        "optional": optional,
        "tooltip": tooltip,
    }

    passthrough = {}
    for key, value in options.items():
        if key in {"label", "tooltip", "tooltips"}:
            continue
        if key == "control_after_generate" and "control_after_generate" in accepted:
            kwargs["control_after_generate"] = value
            continue
        if key in accepted:
            kwargs[key] = value
        else:
            passthrough[key] = value

    if "extra_dict" in accepted and passthrough:
        kwargs["extra_dict"] = passthrough

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return input_cls(input_id, **kwargs)


def _legacy_input_id(input_name: str) -> str:
    if isinstance(input_name, str) and input_name.strip():
        return input_name
    return _snake(str(input_name))


def _parse_input_spec(input_name: str, spec: Any, optional: bool):
    options = {}

    if isinstance(spec, tuple):
        type_part = spec[0]
        if len(spec) > 1 and isinstance(spec[1], dict):
            options = dict(spec[1])
    else:
        type_part = spec

    input_id = _legacy_input_id(input_name)

    if isinstance(type_part, (list, tuple)):
        return input_id, input_name, _build_input(io.Combo, input_id, optional, {**options, "options": list(type_part)})

    if isinstance(type_part, str):
        io_type = _resolve_io_type(type_part)
        return input_id, input_name, _build_input(io_type, input_id, optional, options)

    return input_id, input_name, _build_input(io.String, input_id, optional, options)


def _parse_outputs(legacy_cls: type):
    return_types = list(getattr(legacy_cls, "RETURN_TYPES", ()) or ())
    return_names = list(getattr(legacy_cls, "RETURN_NAMES", ()) or ())
    output_is_list = list(getattr(legacy_cls, "OUTPUT_IS_LIST", ()) or ())

    outputs = []
    expected_count = len(return_types)

    for idx, type_name in enumerate(return_types):
        io_type = _resolve_io_type(type_name)
        out_name = return_names[idx] if idx < len(return_names) else f"output_{idx + 1}"
        out_id = _snake(out_name)
        is_output_list = bool(output_is_list[idx]) if idx < len(output_is_list) else False
        outputs.append(io_type.Output(out_id, display_name=out_name, is_output_list=is_output_list))

    return outputs, expected_count


def _normalize_legacy_result(raw_result: Any, expected_count: int):
    ui_payload = None
    value = raw_result

    if isinstance(value, dict) and "result" in value:
        ui_payload = value.get("ui")
        value = value.get("result")

    if isinstance(value, io.NodeOutput):
        return value

    if isinstance(value, list):
        value = tuple(value)
    elif not isinstance(value, tuple):
        value = (value,)

    if expected_count == 2 and len(value) == 1:
        info = value[0]
        value = (value[0], "" if info is None else str(info))

    if expected_count > 0:
        if len(value) < expected_count:
            value = value + (None,) * (expected_count - len(value))
        elif len(value) > expected_count:
            value = value[:expected_count]

    return io.NodeOutput(*value, ui=ui_payload)


def _build_v3_wrapper(legacy_cls: type, node_id: str, display_name: str):
    input_types = _call_input_types(legacy_cls)
    required = input_types.get("required", {}) or {}
    optional = input_types.get("optional", {}) or {}
    hidden = input_types.get("hidden", {}) or {}

    inputs = []
    key_map = {}
    seen_ids = set()

    def _add_input(name: str, spec: Any, is_optional: bool):
        normalized, original, node_input = _parse_input_spec(name, spec, is_optional)
        unique = normalized
        counter = 2
        while unique in seen_ids:
            unique = f"{normalized}_{counter}"
            counter += 1
        if unique != normalized:
            if hasattr(node_input, "id"):
                node_input.id = unique
        seen_ids.add(unique)
        key_map[unique] = original
        inputs.append(node_input)

    optional_first = ()
    if legacy_cls is TBG_ETUR_PID_Tile_Upscale_Rebuild:
        optional_first = ("image",)

    for name in optional_first:
        if name in optional:
            _add_input(name, optional[name], True)
    for name, spec in required.items():
        _add_input(name, spec, False)
    for name, spec in optional.items():
        if name in optional_first:
            continue
        _add_input(name, spec, True)

    hidden_items = []
    hidden_key_map = {}
    hidden_defaults = {}
    for name, hidden_spec in hidden.items():
        hidden_marker = _extract_hidden_marker(hidden_spec)
        hidden_type = _HIDDEN_MAP.get(hidden_marker)
        if hidden_type is not None:
            hidden_items.append(hidden_type)
            hidden_key_map[hidden_marker] = name
        else:
            # Keep non-core legacy hidden values internal; inject defaults at execute time.
            hidden_defaults[name] = _extract_default_value(hidden_spec)

    outputs, expected_count = _parse_outputs(legacy_cls)
    category = getattr(legacy_cls, "CATEGORY", "TBG/ETUR")
    description = getattr(legacy_cls, "DESCRIPTION", "")
    is_output_node = bool(getattr(legacy_cls, "OUTPUT_NODE", False))
    function_name = getattr(legacy_cls, "FUNCTION", "execute")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=node_id,
            display_name=display_name,
            category=category,
            description=description,
            inputs=inputs,
            outputs=outputs,
            hidden=hidden_items,
            is_output_node=is_output_node,
        )

    @classmethod
    def execute(cls, **kwargs):
        legacy_kwargs = {}
        for normalized_key, value in kwargs.items():
            original_key = cls._key_map.get(normalized_key, normalized_key)
            legacy_kwargs[original_key] = value

        if cls.hidden.unique_id and "UNIQUE_ID" in cls._hidden_key_map:
            legacy_kwargs[cls._hidden_key_map["UNIQUE_ID"]] = cls.hidden.unique_id
        if cls.hidden.prompt and "PROMPT" in cls._hidden_key_map:
            legacy_kwargs[cls._hidden_key_map["PROMPT"]] = cls.hidden.prompt
        if cls.hidden.extra_pnginfo and "EXTRA_PNGINFO" in cls._hidden_key_map:
            legacy_kwargs[cls._hidden_key_map["EXTRA_PNGINFO"]] = cls.hidden.extra_pnginfo

        for hidden_name, hidden_default in cls._hidden_defaults.items():
            if hidden_name not in legacy_kwargs and hidden_default is not None:
                legacy_kwargs[hidden_name] = hidden_default

        legacy_obj = cls._legacy_cls()
        legacy_fn = getattr(legacy_obj, cls._function_name)
        result = legacy_fn(**legacy_kwargs)
        return _normalize_legacy_result(result, cls._expected_count)

    wrapper_name = f"V3_{legacy_cls.__name__}"
    return type(
        wrapper_name,
        (io.ComfyNode,),
        {
            "__module__": __name__,
            "_legacy_cls": legacy_cls,
            "_function_name": function_name,
            "_key_map": key_map,
            "_hidden_key_map": hidden_key_map,
            "_hidden_defaults": hidden_defaults,
            "_expected_count": expected_count,
            "define_schema": define_schema,
            "execute": execute,
        },
    )


_NODE_DEFS = [
    (TBG_ETUR_Labs_Upscaler, "TBG ETUR Labs Upscaler", "TBG ETUR Labs Upscaler"),
    (TBG_ETUR_Refiner_PRO, "TBG ETUR Refiner PRO", "TBG ETUR Refiner PRO"),
    (TBG_ControlNetPipeline, "TBG ETUR Control Net Pipeline", "TBG ETUR ControlNet Pipeline"),
    (TBG_RF_UntwistingRoPE_Pipe, "TBG ETUR RF UntwistingRoPE Pipe", "TBG ETUR RF UntwistingRoPE Pipe"),
    (TBG_TilePrompter_v1, "TBG ETUR Tile Overrides", "TBG ETUR Tile Overrides"),
    (TBG_enrichment_pipe, "TBG ETUR enrichment pipe", "TBG ETUR Enrichment Pipe"),
    (TBG_ETUR_Upscaler_and_Tile_Generator_PRO, "TBG ETUR Upscaler and Tile Generator PRO", "TBG ETUR Upscaler and Tile Generator PRO"),
    (TBG_ETUR_Refiner_CE, "TBG ETUR Refiner CE", "TBG ETUR Refiner CE"),
    (TBG_ETUR_Upscaler_and_Tile_Generator_CE, "TBG ETUR Upscaler and Tile Generator CE", "TBG ETUR Upscaler and Tile Generator CE"),
    (TBG_ETUR_ColorCorrection, "TBG Color Correction", "TBG Color Correction"),
    (TBG_Model_Agnostic_Color_Anchor, "TBG Model Agnostic Color Anchor", "TBG Model Agnostic Color Anchor"),
    (TBG_Model_Agnostic_Latent_Anchor, "TBG Model Agnostic Latent Anchor", "TBG Model Agnostic Latent Anchor"),
    (TBG_ETUR_ColorMatch_Debug, "TBG ETUR ColorMatch Debug Gates", "TBG ETUR ColorMatch Debug Gates"),
    (TBG_ETUR_Labs_Refiner, "TBG ETUR Labs for Refiner", "TBG ETUR Labs for Refiner"),
    (TBG_ETUR_Download_PID_Model, "TBG ETUR Download PiD Model", "TBG ETUR Download PiD Model"),
    (TBG_ETUR_PID_Tile_Upscale_Rebuild, "TBG ETUR PID Tile Upscale Rebuild", "TBG ETUR tiled Nvidia PID Image Upscale"),
    (TBG_PiD_Creative_SDE_Sampler, "TBG PiD Creative SDE Sampler", "TBG PiD Creative SDE Sampler"),
    (TBG_ETUR_SIFT_Drift_Correction, "TBG SIFT+ Drift Correction", "TBG SIFT+ Drift Correction"),
    (TBGFlux2Sampler, "TBGFlux2Sampler", "TBG Flux2 Differential Diffusion Inpainting"),
    (AISourcePatternCleanup, "AISourcePatternCleanup", "AI Source Pattern Cleanup"),
    (QwenEditResolution, "QwenEditResolution", "Qwen Edit Resolution"),
    (FluxKontextResolution, "FluxKontextResolution", "Flux Kontext Resolution"),
    (IncrementBatch, "IncrementBatch", "IncrementBatch"),
    (ImageComplexityMap, "ImageComplexityMap", "Image Complexity Map"),
    (MaskMultiplyZeroWins, "MaskMultiplyZeroWins", "Mask Multiply Zero Wins"),
    (MaskMultiplyAdvanced, "MaskMultiplyAdvanced", "MaskMultiplyAdvanced"),
    (MaskUpdateInpaintFromConditioning, "MaskUpdateInpaintFromConditioning", "Mask Hard Blend"),
    (MaskConditionalPaint, "MaskConditionalPaint", "MaskConditionalPaint"),
    (MaskToDenoiseInterpolator, "MaskToDenoiseInterpolator", "Complexity Mask to Denoise"),
    (MaskGrayValueScaler, "MaskGrayValueScaler", "Complexity Mask to Denoise Min-Max"),
]
V3_NODES = [_build_v3_wrapper(*node_def) for node_def in _NODE_DEFS]


class TBGV3Extension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        print(f"[TBG][v3] comfy_api.v0_0_2 loaded; registered {len(V3_NODES)} nodes")

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return V3_NODES


async def comfy_entrypoint() -> TBGV3Extension:
    return TBGV3Extension()


