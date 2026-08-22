"""Vision-Language (VL) tile conditioning for Qwen models.

Encodes the full upscaled canvas ONCE through the VL-capable CLIP text encoder,
then slices per-tile by coordinate. Each tile's positive conditioning is a row
slice of the global encode: the vision-grid cells its crop region covers, plus
the delimiter and template-tail rows.

Pixel alignment uses the 32px merged-cell grid (patch 16 x spatial merge 2),
matching ContextAnchoredTileRefine vl.py exactly.

Module scope is torch-only; comfy is imported lazily inside functions.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Qwen3-VL / Krea2Tokenizer image block (plain ASCII, cannot corrupt).
# Expands to one <|image_pad|> token per merged patch of the attached image.
VISION_BLOCK_QWEN3VL = "<|vision_start|><|image_pad|><|vision_end|>"

# Krea 2's conditioning template (comfy/text_encoders/krea2.py KREA2_TEMPLATE).
# Krea2TEModel strips the system and user-opening prefix by scanning for the two
# <|im_start|> tokens, so this must remain the native chat template.
# <|im_start|> tokens, so this must remain the native chat template.

KREA2_TEMPLATE_QWEN3VL_V1 = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# long tbg test promt

KREA2_TEMPLATE_QWEN3VL_V2 = "<|im_start|>system\nDescribe the image in detailed, spatially precise terms. For every important element, describe: appearance, color, shape, size, texture, quantity, exact position in the frame, position relative to other elements, and whether it is cropped by an image border. For cropped elements, identify the border(s) where cropping occurs and describe the visible portion. Describe the foreground, middle ground, background, overall composition, lighting, and visible text. Prioritize accurate spatial relationships and framing over unnecessary verbosity.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# Long Inpainting promt

KREA2_TEMPLATE_QWEN3VL_V3 = "<|im_start|>system\nDescribe the image in detailed, spatially precise terms for use as an image-inpainting generation prompt. For every important element, describe: appearance, color, shape, size, texture, quantity, exact position in the frame, position relative to other elements, depth, perspective, orientation, position relative to surrounding elements, and whether it is cropped by an image border. For cropped elements, identify the border(s) where cropping occurs and describe the visible portion. Describe the foreground, middle ground, background, overall composition, lighting, shadows, reflections, atmospheric effects, and visible text.\n\nIMPORTANT INPAINTING INTEGRATION: The description will be used to generate pixels inside an existing image. Describe every element as a naturally integrated part of the existing scene, never as an isolated object, cutout, overlay, sticker, or pasted element. Pay particular attention to how each element transitions into and interacts with its surroundings.\n\nFor boundaries between elements, describe the physical reason for the boundary whenever possible, such as natural object contours, occlusion, contact, depth, shadows, reflections, transparency, atmospheric perspective, material transitions, or depth of field. Describe contact points, overlaps, occlusions, and areas where foreground and background elements naturally intersect. Preserve visual continuity between neighboring regions in color, texture, lighting, sharpness, perspective, noise, grain, reflections, and atmospheric effects.\n\nDo not introduce artificial outlines, halos, borders, cutout edges, sticker-like edges, masking artifacts, or visible segmentation boundaries unless they are genuinely present in the image. Do not describe objects as floating or isolated when they physically interact with surrounding surfaces or objects. Instead, describe how they connect, overlap, touch, occlude, cast shadows onto, or are partially obscured by nearby elements.\n\nWhen describing an element near an inpainting boundary, prioritize continuity with the surrounding image. Describe how texture, color, illumination, shadows, reflections, and atmospheric effects continue naturally across the transition. Avoid language that implies a hard pasted boundary. The generated description should help an inpainting model create pixels that visually fuse with the existing surrounding pixels rather than producing a visible seam.\n\nFor cropped elements, identify the border(s) where cropping occurs and describe the visible portion. Prioritize accurate spatial relationships, scene continuity, boundary integration, and framing over unnecessary verbosity.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# Micro textures

KREA2_TEMPLATE_QWEN3VL_V4 = "<|im_start|>system\nDescribe the image precisely for photorealistic image reconstruction. Focus on visible content, materials, surface detail, spatial accuracy, and framing.\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# Micro textures + Body Surfaces Details

KREA2_TEMPLATE_QWEN3VL_V5 = """<|im_start|>system
Describe this image for photorealistic reconstruction. Be precise, concrete, and only describe details that are visible.

For every important object or person, describe shape, size, color, position, orientation, proportions, and relationship to nearby elements. Inspect all visible SURFACE and SMALL DETAILS, not just the main features.

FOR PEOPLE, carefully describe:
- FACE: face shape, proportions, skin tone, undertone, complexion, pores, fine lines, freckles, moles, blemishes, wrinkles, facial hair, and other visible skin variation
- EYES: eye color, shape, size, spacing, iris pattern, pupil, eyelids, eyelashes, eyebrows, and catchlights
- NOSE + EARS: shape, size, proportions, contours, and visible texture/details
- MOUTH: lip shape, size, proportions, natural lip texture, color, moisture/gloss, and surrounding skin
- HAIR: color, length, density, thickness, texture, curl/wave/straightness, individual strands, flyaways, parting, volume, hairstyle, roots, and highlights
- HANDS: finger count and position, finger proportions, knuckles, joints, skin texture, veins, creases, fingertips, and gestures
- NAILS: shape, length, size, color, natural texture, polish, shine, and imperfections
- BODY/SKIN: visible skin tone and variation, pores, fine hairs, wrinkles, creases, folds, muscle definition, natural asymmetry, and other visible anatomical details
- ACCESSORIES: jewelry, rings, earrings, necklaces, bracelets, watches, piercings, glasses, and their material, color, shape, size, position, reflections, and small details
- CLOTHING: material, weave, fibers, seams, stitching, wrinkles, folds, compression, stretching, stains, fading, wear, pilling, fraying, and other imperfections

For ALL materials, explicitly describe fine MICROTEXTURE and larger physical texture: fibers, weave, pores, grain, wrinkles, creases, folds, dents, scratches, cracks, chips, scuffs, stains, dirt, dust, fingerprints, fading, discoloration, fraying, peeling, oxidation, and other irregularities. State WHERE details occur, their approximate SCALE, DIRECTION, DENSITY, and whether they are regular or irregular.

Describe surface response to light: matte/glossy, rough/smooth, reflective/translucent, wet/moist, highlights, shadows, and local variations in reflectivity. Describe wear, aging, damage, deformation, and contact/compression where visible.

Never replace visible physical details with vague words such as "beautiful," "detailed," "textured," "realistic," or "rough." Name the actual visual characteristics. Preserve natural asymmetry and small imperfections rather than idealizing the subject.

Also describe foreground, middle ground, background, composition, camera viewpoint, perspective, framing, cropping at each image border, depth, occlusion, lighting direction and quality, shadows, reflections, focus, depth of field, atmospheric effects, and visible text.

PRIORITY: facial identity and proportions, eyes, hair, skin, hands and fingers, nails, lips, jewelry/accessories, clothing, MATERIALS, MICROTEXTURE, IMPERFECTIONS, spatial relationships, and FRAMING.
<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


KREA2_TEMPLATE_QWEN3VL = """<|im_start|>system
Describe the image in detailed, spatially precise terms for use as an image-inpainting generation prompt. Analyze the entire image carefully and describe all important visual elements, but give SPECIAL PRIORITY to the image borders and to visual information that enters the frame from the TOP and LEFT sides.

For every important element, describe: appearance, color, shape, size, texture, material, quantity, exact position in the frame, position relative to other elements, depth, perspective, orientation, lighting, shadows, reflections, and whether it is cropped by an image border. For cropped elements, identify the exact border(s) where cropping occurs and describe the visible portion.

BORDER AND EDGE ANALYSIS IS CRITICAL FOR INPAINTING: Carefully inspect the TOP and LEFT portions of the image and identify every significant color, material, texture, pattern, surface, object, structural feature, shadow, highlight, reflection, or visual detail that enters or continues from those borders. Describe what is visible directly along the top edge and left edge, what lies immediately inside those borders, and how those features change as they extend farther into the image. Pay special attention to textures and materials that should naturally continue into an inpainted area.

For the TOP BORDER, explicitly describe the incoming visual information from left to right: dominant colors, materials, textures, patterns, surfaces, objects, lighting, shadows, gradients, and any structural features touching or approaching the top edge. Explain how each feature continues downward into the image and its spatial relationship to nearby elements.

For the LEFT BORDER, explicitly describe the incoming visual information from top to bottom: dominant colors, materials, textures, patterns, surfaces, objects, lighting, shadows, gradients, and any structural features touching or approaching the left edge. Explain how each feature continues inward toward the center of the image and its spatial relationship to nearby elements.

When a material or texture touches an image border, describe its visual characteristics precisely, including color, hue, brightness, saturation, roughness, smoothness, grain, pattern, scale, directionality, repetition, irregularity, and local variation when visible. Describe whether the surface is wood, stone, metal, fabric, glass, skin, vegetation, wall, floor, sky, water, concrete, plastic, paper, or another material when identifiable. Describe how its texture, color, illumination, and pattern continue away from the border.

Treat border information as CONTINUOUS SCENE CONTEXT rather than as a separate frame or boundary. The top and left edges provide important visual evidence for reconstructing missing regions. When an inpainted region is adjacent to these areas, describe the incoming colors, materials, textures, patterns, lighting, shadows, and structural features that the generated pixels should naturally continue.

IMPORTANT INPAINTING INTEGRATION: Describe every element as a naturally integrated part of the existing scene, never as an isolated object, cutout, overlay, sticker, or pasted element. For boundaries between elements, describe the physical reason for the transition whenever possible, such as natural object contours, occlusion, contact, depth, shadows, reflections, transparency, atmospheric perspective, material transitions, or depth of field.

Do not introduce artificial outlines, halos, borders, cutout edges, sticker-like edges, masking artifacts, or visible segmentation boundaries unless they are genuinely present in the image. Avoid describing objects as isolated when they physically interact with surrounding surfaces or objects. Describe how objects connect, overlap, touch, occlude, cast shadows onto, reflect light from, or are partially obscured by nearby elements.

Describe the foreground, middle ground, background, overall composition, perspective, lighting, shadows, reflections, atmospheric effects, and visible text. Preserve continuity in color, material, texture, illumination, sharpness, perspective, noise, grain, reflections, and atmospheric effects between existing pixels and regions that may be generated.

Prioritize accurate spatial relationships and framing, but when describing the image for inpainting, prioritize BORDER CONTINUITY, INCOMING TEXTURES, MATERIALS, COLORS, PATTERNS, AND STRUCTURAL FEATURES over unnecessary verbosity. The description should provide enough information for an inpainting model to continue the visual information entering from the TOP and LEFT borders naturally into the missing region without creating hard seams or artificial boundaries.
<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
# Qwen Image's native image-conditioning template. QwenImageTokenizer replaces
# the image_pad token with the attached image embedding.
QWEN25_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"


def _get_vl_template(model_type):
    """Return (llama_template, image_block) for the given model type.

    Qwen3-VL / Krea2Tokenizer uses <|vision_start|><|image_pad|><|vision_end|>.
    Qwen2.5-VL / QwenImageTokenizer uses the same image block with its native template.
    """
    if model_type in ("Krea2", "Qwen3-VL 4B", "Qwen3-VL 8B"):
        return KREA2_TEMPLATE_QWEN3VL, VISION_BLOCK_QWEN3VL
    # Qwen2.5-VL / Qwen Image / Qwen Image Edit
    return QWEN25_TEMPLATE, VISION_BLOCK_QWEN3VL

# One merged patch covers this many encode-side pixels (patch 16 x spatial merge 2).
MERGED_CELL = 32
QWEN25_MERGED_CELL = 28


def _merged_cell(model_type):
    return QWEN25_MERGED_CELL if model_type in ("Qwen Image", "Qwen Image Edit") else MERGED_CELL

# Encode budget (total pixels, aspect preserved, snapped to /MERGED_CELL). Sized so a
# 2x2 grid keeps the per-tile row density proven sufficient in per-tile-encode A/Bs.
# The tower's 48x48 num_position_embeddings table (QWEN35_VISION_DEFAULTS: patch 16,
# merge 2, 2304 embeddings) is indexed on the UNMERGED 16px patch lattice, not the 32px
# merged-cell lattice this module reasons in, so staying inside it would mean <= 768x768
# px total; at this budget a square canvas is 28x28 merged cells = 56x56 patches and the
# tower interpolates its position embeddings (fast_pos_embed_interpolate) by design.
GLOBAL_SLICE_BUDGET = 768 * 1024


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_for_global(source, merged_cell=MERGED_CELL):
    """Area-resample to the fixed global budget, snapped to /MERGED_CELL grid.

    Returns ``(resampled_tensor, height, width)`` where height and width are
    the snapped dimensions (multiples of MERGED_CELL). This copy is conditioning-side
    only — the sampled tiles are never resampled (prime directive 1).
    """
    import comfy.utils

    samples = source.movedim(-1, 1)
    source_pixels = samples.shape[3] * samples.shape[2]
    scale = math.sqrt(GLOBAL_SLICE_BUDGET / source_pixels)
    width = max(merged_cell, round(samples.shape[3] * scale / merged_cell) * merged_cell)
    height = max(merged_cell, round(samples.shape[2] * scale / merged_cell) * merged_cell)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3], height, width


def parse_krea2_layer_weights(value):
    if value is None or value == "":
        return None
    try:
        weights = [float(part.strip()) for part in str(value).replace(";", ",").split(",") if part.strip()]
    except ValueError:
        return None
    if len(weights) != 12:
        return None
    return weights


def apply_krea2_vl_mask(
    conditioning,
    vision_weights,
    strength,
    layer_weights=None,
    layer_multiplier=1.0,
    reference_rebalance=False,
):
    """Apply the selected legacy or Rebalance-style Krea2 layer scaling."""
    if conditioning is None:
        return conditioning

    weights = None
    if vision_weights is not None:
        weights = torch.as_tensor(
            vision_weights, dtype=torch.float32
        ).clamp(0.0, 1.0)
    result = []
    for entry in conditioning:
        tensor, extras = entry[0], dict(entry[1])
        scaled = tensor
        if weights is not None:
            vision_count = min(int(weights.numel()), max(0, int(tensor.shape[1]) - 2))
        else:
            vision_count = 0
        if vision_count:
            scaled = tensor.clone()
            token_weights = (
                weights[:vision_count]
                * float(strength)
            ).to(
                device=tensor.device, dtype=tensor.dtype
            ).view(1, vision_count, 1)
            visual = scaled[:, 1:1 + vision_count] * token_weights
            if not reference_rebalance and layer_weights is not None and visual.shape[-1] % len(layer_weights) == 0:
                layer_gains = torch.tensor(
                    layer_weights, device=tensor.device, dtype=tensor.dtype
                ).view(1, 1, len(layer_weights), 1)
                visual = visual.reshape(
                    visual.shape[0], vision_count, len(layer_weights), -1
                ) * layer_gains
                visual = visual.reshape(visual.shape[0], vision_count, -1)
            scaled[:, 1:1 + vision_count] = visual

        if reference_rebalance and layer_weights is not None and scaled.shape[-1] % len(layer_weights) == 0:
            original_dtype = scaled.dtype
            layer_count = len(layer_weights)
            layer_dim = scaled.shape[-1] // layer_count
            rebalance = scaled.float().reshape(
                *scaled.shape[:-1], layer_count, layer_dim
            )
            layer_gains = torch.tensor(
                layer_weights,
                dtype=rebalance.dtype,
                device=rebalance.device,
            ).view(*([1] * (rebalance.dim() - 2)), layer_count, 1)
            rebalance = rebalance * layer_gains
            scaled = rebalance.reshape(*scaled.shape[:-1], -1).to(original_dtype)
            scaled = scaled * float(layer_multiplier)
        elif layer_multiplier != 1.0:
            if reference_rebalance:
                scaled = scaled * float(layer_multiplier)

        result.append([scaled, extras])
    return result


# ---------------------------------------------------------------------------
# Slice index math
# ---------------------------------------------------------------------------

def slice_indices(crop_x0, crop_y0, crop_w, crop_h, canvas_h, canvas_w, enc_h, enc_w, expected_seq, merged_cell=MERGED_CELL):
    """Compute token row indices for a tile's crop region in the global encode.

    The stripped encode's layout is:
      [0]      = vision_start
      [1..N]   = grid rows in raster order  (cell(r,c) = 1 + r*grid_w + c)
      [N+1]    = vision_end
      [N+2..]  = template tail

    The tile rect maps to the merged-cell range by intersection. Boundary cells
    that are only partly covered are included so neighboring tiles share them —
    the row-space analogue of the overlap band.

    Parameters
    ----------
    crop_x0, crop_y0 : int
        Top-left corner of the crop in canvas pixel coordinates.
    crop_w, crop_h : int
        Width and height of the crop.
    canvas_h, canvas_w : int
        Full canvas dimensions.
    enc_h, enc_w : int
        Encoded (resampled) canvas dimensions.
    expected_seq : int
        Total sequence length of the global encode.

    Returns
    -------
    list[int]
        Token row indices to keep for this tile.
    """
    grid_h = enc_h // merged_cell
    grid_w = enc_w // merged_cell
    n_rows = grid_h * grid_w

    cx0 = max(0, math.floor((crop_x0) * enc_w / canvas_w / merged_cell))
    cx1 = min(grid_w, math.ceil((crop_x0 + crop_w) * enc_w / canvas_w / merged_cell))
    cy0 = max(0, math.floor((crop_y0) * enc_h / canvas_h / merged_cell))
    cy1 = min(grid_h, math.ceil((crop_y0 + crop_h) * enc_h / canvas_h / merged_cell))

    rows = [1 + r * grid_w + c for r in range(cy0, cy1) for c in range(cx0, cx1)]
    result = [0] + rows + [1 + n_rows] + list(range(1 + n_rows + 1, expected_seq))
    # Debug: verify per-tile cropping
    import sys
    n_vision = len(rows)
    n_tail = expected_seq - (1 + n_rows + 1)
    print(
        f"[VL DEBUG] tile crop=({crop_x0},{crop_y0},{crop_w}x{crop_h}) "
        f"grid_cells=[{cx0}:{cx1},{cy0}:{cy1}] vision_rows={n_vision} "
        f"tail_rows={n_tail} total_indices={len(result)}",
        file=sys.stderr,
    )
    return result


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_canvas(clip, canvas_copy, grid_h, grid_w, model_type):
    """Encode the full resampled canvas ONCE through the VL-capable CLIP.

    Uses the correct image-conditioning template for the model type.

    Passes llama_template explicitly so the tokenizer uses the known conditioning
    prompt ("Describe the image by detailing …") rather than the default text-gen
    template — without it the row layout that build_vl_conditioning slices by is wrong.

    Validates output sequence length against the expected layout so a core
    template change fails fast instead of silently scrambling every slice.

    Returns
    -------
    tuple[list[list], int]
        (encoded_conditioning, expected_sequence_length)
    """
    n_rows = grid_h * grid_w
    llama_template, image_block = _get_vl_template(model_type)

    try:
        # Pass llama_template explicitly; without it the tokenizer picks a default
        # template whose prefix survives the strip logic and shifts the row layout
        # that the per-tile slicing depends on.
        tokens = clip.tokenize(image_block, images=[canvas_copy], llama_template=llama_template)
    except TypeError as error:
        raise RuntimeError(
            "VL encode: this CLIP's tokenizer does not accept images. The node needs a "
            "vision-language text encoder (Qwen25-7B-VLI, Qwen3-VL, or similar). ({})".format(error)
        ) from error

    # Derive the expected sequence length from the token stream.
    key = next(iter(tokens))
    token_list = tokens[key][0]
    ids = [t[0] for t in token_list]

    pad_pos = next((i for i, v in enumerate(ids) if isinstance(v, dict)), None)
    if pad_pos is None and model_type == "Krea2":
        # Krea2's token-weight pairs use integer tuples, not dict markers.
        # tail_len will be derived from the actual encoder output anyway.
        tail_len = 0
    else:
        if pad_pos is None:
            raise RuntimeError(
                "VL encode: the tokenizer produced no image marker for model {}."
                .format(model_type)
            )
        tail_len = len(ids) - pad_pos - 2
    token_expected_seq = 1 + n_rows + 1 + tail_len

    encoded = clip.encode_from_tokens_scheduled(tokens)
    actual_seq = int(encoded[0][0].shape[1])

    minimum_seq = 1 + n_rows + 1
    if actual_seq < minimum_seq:
        raise RuntimeError(
            "VL encode: encoded conditioning has {} rows, but at least {} "
            "are required for vision grid {}x{} plus start/end rows."
            .format(actual_seq, minimum_seq, grid_h, grid_w)
        )

    if actual_seq != token_expected_seq:
        print(
            "[VL INFO] Encoder sequence differs from token estimate: "
            "actual={}, token_estimate={}, grid={}x{}, tail_estimate={}. "
            "Using actual sequence length."
            .format(
                actual_seq,
                token_expected_seq,
                grid_h,
                grid_w,
                tail_len,
            )
        )

    return encoded, actual_seq


def _encode_batch(clip, canvas_copy, grid_h, grid_w, model_type):
    """One encode per batch row, concatenated on the batch axis.

    The core tokenizer attaches images[0] alone, so handing over a whole
    [B,H,W,3] canvas would condition EVERY image on row 0's picture.
    Every row is the same size by construction, so _encode_canvas' own
    seq fail-fast covers layout drift and the rows always cat.
    """
    batch = int(canvas_copy.shape[0])
    if batch == 1:
        return _encode_canvas(clip, canvas_copy, grid_h, grid_w, model_type)

    per_row = []
    expected_seq = None
    for b in range(batch):
        encoded, expected_seq = _encode_canvas(clip, canvas_copy[b:b + 1], grid_h, grid_w, model_type)
        per_row.append(encoded)

    merged = []
    for entries in zip(*per_row):
        extras = dict(entries[0][1])
        pooled = extras.get("pooled_output")
        if isinstance(pooled, torch.Tensor):
            extras["pooled_output"] = torch.cat(
                [entry[1]["pooled_output"] for entry in entries], dim=0
            )
        merged.append([torch.cat([entry[0] for entry in entries], dim=0), extras])
    return merged, expected_seq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vl_conditioning(
    clip,
    full_image,
    grid_specs,
    canvas_h,
    canvas_w,
    model_type,
    vl_strength=0.5,
    tile_indices=None,
):
    """Build per-tile VL conditioning from a single full-canvas encode.

    Parameters
    ----------
    clip : CLIP model
        VL-capable text encoder (Qwen25-7B-VLI, Qwen3-VL, or similar) with ``tokenize(images=...)``
        and ``encode_from_tokens_scheduled()`` methods.
    full_image : torch.Tensor
        Full upscaled canvas, shape ``(1, H, W, 3)`` or ``(B, H, W, 3)``.
    grid_specs : list[tuple]
        Tile grid specs, each tuple: ``(a, b, c, x, y, width, height)``.
        The x, y, width, height fields (indices 3-6) define the tile crop
        in canvas pixel coordinates.
    canvas_h, canvas_w : int
        Full canvas dimensions.
    model_type : str
        Model type for selecting the correct image placeholder token.
        One of: "Qwen Image", "Qwen Image Edit", "Krea 2", "Qwen3-VL 4B", "Qwen3-VL 8B".
    vl_strength : float
        Multiplier applied to VL conditioning tokens before txtfusion.
        Reduces global semantic bleed from the Qwen3-VL encoder's
        global self-attention. Range 0.0–1.0; default 0.5.
        Values < 0.3 may weaken prompt adherence; values > 0.8 may
        cause full-image semantics to dominate tile-local content.
    tile_indices : iterable[int] or None
        Optional tile indices to materialize. ``None`` keeps the full-grid
        behavior used by normal runs.

    Returns
    -------
    dict[int, list]
        ``{tile_index: vl_conditioning}`` for each tile in grid_specs.
    """
    # Step 1: resample full canvas to global budget.
    merged_cell = _merged_cell(model_type)
    canvas_copy, enc_h, enc_w = resample_for_global(full_image, merged_cell)

    grid_h = enc_h // merged_cell
    grid_w = enc_w // merged_cell

    # Step 2: encode the full canvas ONCE (model-type-aware template in _encode_canvas).
    encoded, expected_seq = _encode_batch(clip, canvas_copy, grid_h, grid_w, model_type)

    tile_positives = {}
    selected_indices = None if tile_indices is None else {int(i) for i in tile_indices}
    for tile_idx, spec in enumerate(grid_specs):
        if selected_indices is not None and tile_idx not in selected_indices:
            continue
        # Extract crop coordinates from grid_specs tuple.
        try:
            _, _, _, crop_x0, crop_y0, crop_w, crop_h = spec
        except (ValueError, TypeError):
            continue

        indices = slice_indices(
            crop_x0, crop_y0, crop_w, crop_h,
            canvas_h, canvas_w, enc_h, enc_w, expected_seq, merged_cell
        )

        sliced = []
        for entry in encoded:
            tensor, extras = entry[0], dict(entry[1])
            # A full-canvas attention mask is wrong for a slice; drop it.
            extras.pop("attention_mask", None)
            index_tensor = torch.tensor(indices, device=tensor.device)
            sliced_tensor = tensor.index_select(1, index_tensor)
            n_tail = expected_seq - (1 + (enc_h // merged_cell) * (enc_w // merged_cell) + 1)
            n_vision = max(0, len(indices) - 2 - n_tail)
            extras["tbg_vl_vision_ranges"] = [(1, 1 + n_vision)]
            # Scale only Qwen2.5 visual rows. Keep the template tail at its
            # native strength so lowering VL does not remove the instruction.
            if model_type in ("Qwen Image", "Qwen Image Edit") and vl_strength != 1.0:
                sliced_tensor = sliced_tensor.clone()
                sliced_tensor[:, 1:1 + n_vision] *= vl_strength
            elif vl_strength != 1.0:
                # Preserve the established Krea2/Qwen3 behavior.
                sliced_tensor = sliced_tensor * vl_strength
            sliced.append([sliced_tensor, extras])

        # Return raw ComfyUI conditioning format: list of [tensor, dict] entries.
        # Do NOT use convert_cond — it transforms the format and downstream
        # InpaintModelConditioningNode.encode expects (tensor, dict) pairs.
        tile_positives[tile_idx] = sliced

    return tile_positives


def encode_tile(
    clip,
    tile_image,
    model_type,
    vl_strength=1.0,
    layer_weights=None,
    layer_multiplier=1.0,
    reference_rebalance=False,
):
    """Encode a single tile's upscaled crop through the VL encoder.

    For Tile Embedding mode: each tile's crop is encoded individually,
    so the vision tokens describe exactly what that tile contains.

    Parameters
    ----------
    clip : CLIP model
        VL-capable text encoder with ``tokenize(images=...)`` and
        ``encode_from_tokens_scheduled()`` methods.
    tile_image : torch.Tensor
        Single tile's upscaled crop, shape ``(1, H, W, 3)``.
    model_type : str
        Model type for selecting the correct image placeholder token.

    Returns
    -------
    list
        Conditioning in ComfyUI format for this tile.
    """
    merged_cell = _merged_cell(model_type)
    canvas_copy, enc_h, enc_w = resample_for_global(tile_image, merged_cell)
    grid_h = enc_h // merged_cell
    grid_w = enc_w // merged_cell

    # Encode this tile's crop (model-type-aware template).
    encoded, _ = _encode_batch(clip, canvas_copy, grid_h, grid_w, model_type)

    # No slicing needed — the entire encode belongs to this tile.
    # Drop the full-canvas attention mask (not meaningful for a single tile).
    cleaned = []
    for entry in encoded:
        tensor, extras = entry[0], dict(entry[1])
        extras.pop("attention_mask", None)
        extras["tbg_vl_vision_ranges"] = [(1, 1 + grid_h * grid_w)]
        if model_type in ("Qwen Image", "Qwen Image Edit") and vl_strength != 1.0:
            tensor = tensor.clone()
            tensor[:, 1:1 + grid_h * grid_w] *= vl_strength
        cleaned.append([tensor, extras])

    if model_type == "Krea2" and (
        vl_strength != 1.0
        or layer_weights is not None
        or (reference_rebalance and layer_multiplier != 1.0)
    ):
        cleaned = apply_krea2_vl_mask(
            cleaned,
            torch.ones(grid_h * grid_w, dtype=torch.float32),
            vl_strength,
            layer_weights,
            layer_multiplier,
            reference_rebalance,
        )

    # Return raw ComfyUI conditioning format: list of [tensor, dict] entries.
    # Do NOT use convert_cond — it transforms the format and downstream
    # InpaintModelConditioningNode.encode expects (tensor, dict) pairs.
    return cleaned


def combine_tile_and_global(tile_conditioning, global_conditioning, model_type=None):
    """Append a global tile slice to the local tile VL tokens.

    Krea2/Qwen3 keep their established appended-token behavior. Qwen Image uses
    one native image sequence: its local visual rows are blended with the
    corresponding global rows while its delimiters and template tail remain
    single and unchanged.
    """
    if tile_conditioning is None:
        return global_conditioning
    if global_conditioning is None:
        return tile_conditioning

    combined = []
    for tile_entry, global_entry in zip(tile_conditioning, global_conditioning):
        tile_tensor, tile_extras = tile_entry[0], dict(tile_entry[1])
        global_tensor = global_entry[0]
        tile_extras.pop("attention_mask", None)
        if model_type in ("Qwen Image", "Qwen Image Edit"):
            tile_ranges = list(tile_extras.get("tbg_vl_vision_ranges", ()))
            global_ranges = list(global_entry[1].get("tbg_vl_vision_ranges", ()))
            if not tile_ranges or not global_ranges:
                combined.append([tile_tensor, tile_extras])
                continue

            tile_start, tile_end = tile_ranges[0]
            global_start, global_end = global_ranges[0]
            tile_visual = tile_tensor[:, tile_start:tile_end]
            global_visual = global_tensor[:, global_start:global_end]
            if tile_visual.shape[1] != global_visual.shape[1]:
                global_visual = F.interpolate(
                    global_visual.transpose(1, 2),
                    size=tile_visual.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            fused_tensor = tile_tensor.clone()
            fused_tensor[:, tile_start:tile_end] = tile_visual * 0.75 + global_visual * 0.25
            tile_extras["tbg_vl_vision_ranges"] = tile_ranges
            combined.append([fused_tensor, tile_extras])
            continue

        tile_length = int(tile_tensor.shape[1])
        tile_ranges = list(tile_extras.get("tbg_vl_vision_ranges", ()))
        global_ranges = [
            (int(row_start) + tile_length, int(row_end) + tile_length)
            for row_start, row_end in global_entry[1].get("tbg_vl_vision_ranges", ())
        ]
        tile_extras["tbg_vl_vision_ranges"] = tile_ranges + global_ranges
        combined.append([torch.cat((tile_tensor, global_tensor), dim=1), tile_extras])
    return combined
