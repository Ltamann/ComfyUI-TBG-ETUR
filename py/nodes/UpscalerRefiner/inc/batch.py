import copy
import gc
from types import SimpleNamespace

import torch
from comfy import model_management

from .....TBG.CALLBACKS.constants import reset_tbg


BATCH_MODE_ATTR = "_tbg_batch_mode"
BATCH_TILER_KWARGS_ATTR = "_tbg_batch_tiler_kwargs"
BATCH_OVERRIDE_KWARGS_ATTR = "_tbg_batch_tile_override_kwargs"


def is_batch_image(image):
    return bool(
        image is not None
        and hasattr(image, "shape")
        and len(image.shape) >= 4
        and int(image.shape[0]) > 1
    )


def is_batch_pipe(pipe):
    if not isinstance(pipe, (tuple, list)) or len(pipe) < 12:
        return False
    inputs = pipe[0]
    params = pipe[1]
    prompter = pipe[7]
    return bool(
        getattr(inputs, BATCH_MODE_ATTR, False)
        or getattr(params, BATCH_MODE_ATTR, False)
        or getattr(prompter, BATCH_MODE_ATTR, False)
    )


def _copy_kwargs(kwargs):
    return dict(copy.copy(kwargs or {}))


def _light_namespace(**values):
    ns = SimpleNamespace()
    for key, value in values.items():
        setattr(ns, key, value)
    return ns


def make_batch_pipe(tbg, kwargs, node_id, tiler_id):
    image_batch = kwargs.get("image", None)
    tiler_kwargs = _copy_kwargs(kwargs)

    inputs = _light_namespace(image=image_batch)
    params = copy.copy(tbg.PARAMS)
    outputs = _light_namespace(
        upscaled_image=None,
        orig_grid_images_all=[],
        grid_images_all=[],
        persistent_generated_tiles=[],
        last_final_image=None,
    )
    segments = copy.copy(tbg.SEGMENTS)
    size = copy.copy(tbg.SIZE)
    api = copy.copy(tbg.API)
    prompter = _light_namespace(
        tiler_prompts=[],
        output_prompts=[],
        output_denoises=[],
        output_seeds_js=[],
        output_cnet_js=[],
        output_cfg_js=[],
        output_model_js=[],
        output_cnetpipe_js=[],
        output_color_match_js=[],
        output_ignore_general_prompt_js=[],
        model_overrides=None,
        model_override_key=None,
        cnetpipe_overrides=None,
        cnetpipe_override_key=None,
        tiles_to_process=None,
        cache_key=None,
        Prompt_Selected_Tiles_Only=None,
        Prompt_Selected_Tiles_By_Numbers=None,
        Rebuild_only_modified_tiles=False,
    )

    for obj in (inputs, params, outputs, segments, size, api, prompter):
        setattr(obj, BATCH_MODE_ATTR, True)
    setattr(params, BATCH_TILER_KWARGS_ATTR, tiler_kwargs)
    setattr(params, "_tbg_batch_size", int(image_batch.shape[0]) if image_batch is not None else 0)
    setattr(params, "_tbg_batch_base_tiler_id", tiler_id)
    setattr(params, "_tbg_batch_tiler_node_id", node_id)
    setattr(params, "_tbg_batch_image_shape", tuple(int(v) for v in image_batch.shape))

    return (
        inputs,
        params,
        tbg.KSAMPLER,
        outputs,
        segments,
        size,
        api,
        prompter,
        "Batch Mode",
        node_id,
        tiler_id,
        getattr(api, "info_url", None),
    )


def attach_tile_overrides_to_batch_pipe(pipe, kwargs, override_node_id):
    if not is_batch_pipe(pipe):
        return pipe
    (
        inputs,
        params,
        ksampler,
        outputs,
        segments,
        size,
        api,
        prompter,
        current_credits,
        node_id,
        tiler_id,
        info_url,
    ) = pipe

    prompter = copy.copy(prompter)
    override_kwargs = _copy_kwargs(kwargs)
    override_kwargs.pop("TBG_Pipe", None)
    setattr(prompter, BATCH_MODE_ATTR, True)
    setattr(prompter, BATCH_OVERRIDE_KWARGS_ATTR, override_kwargs)
    setattr(prompter, "_tbg_batch_override_node_id", override_node_id)
    setattr(prompter, "Rebuild_only_modified_tiles", kwargs.get("Rebuild_only_modified_tiles"))

    return (
        inputs,
        params,
        ksampler,
        outputs,
        segments,
        size,
        api,
        prompter,
        current_credits,
        override_node_id,
        tiler_id,
        info_url,
    )


def _internal_id(base_id, suffix, index):
    return f"{base_id}__batch_{index}_{suffix}"


def _slice_matching_batch_tensors(kwargs, index, batch_size):
    sliced = {}
    for key, value in kwargs.items():
        if (
            key != "image"
            and isinstance(value, torch.Tensor)
            and len(value.shape) >= 1
            and int(value.shape[0]) == batch_size
        ):
            sliced[key] = value[index:index + 1]
        else:
            sliced[key] = value
    return sliced


def _cat_same_shape(outputs, name):
    if not outputs:
        return None
    if any(image is None for image in outputs):
        raise ValueError(f"ETUR batch output '{name}' is missing one or more per-image results.")
    shapes = [tuple(int(v) for v in image.shape[1:]) for image in outputs if image is not None]
    if not shapes:
        return None
    first = shapes[0]
    if any(shape != first for shape in shapes):
        raise ValueError(
            f"ETUR batch output '{name}' has mixed image sizes. "
            "Use same-size inputs, or resize/pad before ETUR, so the refiner can return one IMAGE batch."
        )
    return torch.cat(outputs, dim=0)


def _drop_single_run_state(tiler_id):
    try:
        reset_tbg(str(tiler_id))
    except Exception:
        pass
    try:
        model_management.soft_empty_cache()
    except Exception:
        pass
    gc.collect()


def run_batch_refiner(refiner_cls, tiler_cls, tile_prompter_cls, **kwargs):
    batch_pipe = kwargs.get("TBG_Pipe")
    if not is_batch_pipe(batch_pipe):
        return refiner_cls.fn(**kwargs)

    inputs, params, _, _, _, _, _, prompter, _, _, tiler_id, _ = batch_pipe
    image_batch = getattr(inputs, "image", None)
    if image_batch is None:
        raise ValueError("ETUR batch mode received a batch pipe without an image batch.")

    batch_size = int(image_batch.shape[0])
    base_tiler_id = str(getattr(params, "_tbg_batch_base_tiler_id", tiler_id))
    tiler_kwargs_template = _copy_kwargs(getattr(params, BATCH_TILER_KWARGS_ATTR, {}))
    override_kwargs_template = _copy_kwargs(getattr(prompter, BATCH_OVERRIDE_KWARGS_ATTR, {}))
    override_node_id = str(getattr(prompter, "_tbg_batch_override_node_id", "TileOverrides"))
    refiner_node_id = str(kwargs.get("id", "Refiner"))

    print(f"[TBG Batch] Starting memory-safe ETUR batch: {batch_size} images")

    collected = [[] for _ in range(5)]
    for index in range(batch_size):
        one_based = index + 1
        single_tiler_id = _internal_id(base_tiler_id, "tiler", index)
        single_override_id = _internal_id(override_node_id, "override", index)
        single_refiner_id = _internal_id(refiner_node_id, "refiner", index)
        print(
            f"[TBG Batch] Image {one_based}/{batch_size}: "
            f"tiler={single_tiler_id} override={single_override_id} refiner={single_refiner_id}"
        )

        tiler_kwargs = _slice_matching_batch_tensors(_copy_kwargs(tiler_kwargs_template), index, batch_size)
        tiler_kwargs["image"] = image_batch[index:index + 1]
        tiler_kwargs["id"] = single_tiler_id
        tiler_result = tiler_cls.fn(**tiler_kwargs)
        single_pipe = tiler_result[0]

        override_kwargs = _copy_kwargs(override_kwargs_template)
        override_kwargs["TBG_Pipe"] = single_pipe
        override_kwargs["id"] = single_override_id
        override_result = tile_prompter_cls().fn(**override_kwargs)
        single_pipe = override_result["result"][0]

        refiner_kwargs = _copy_kwargs(kwargs)
        refiner_kwargs["TBG_Pipe"] = single_pipe
        refiner_kwargs["id"] = single_refiner_id
        single_result = refiner_cls.fn(**refiner_kwargs)

        if isinstance(single_result, dict):
            single_result = single_result.get("result", ())
        for output_index in range(5):
            collected[output_index].append(single_result[output_index])

        _drop_single_run_state(single_tiler_id)

    result = tuple(
        _cat_same_shape(collected[output_index], name)
        for output_index, name in enumerate(
            ("Refined", "Refined without Segs", "Refined without ColorCorrection", "Original Upscaled", "Original")
        )
    )

    print(f"[TBG Batch] Completed memory-safe ETUR batch: {batch_size} images")
    return result + (result[0],)
