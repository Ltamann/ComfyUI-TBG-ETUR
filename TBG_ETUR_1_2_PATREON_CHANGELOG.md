# TBG ETUR Version 1.2 + NVIDIA PiD

This release is a major feature update over the public 1.1.18 version. It adds NVIDIA PixelDiT/PiD support, new Flux 2 inpaint handling, tile override improvements, batch support, stronger color and geometry stabilization, new helper nodes, frontend improvements, and many stability fixes.

It is also the first TBG ETUR version developed with a full coding-model-assisted workflow. That made it possible to touch many connected parts of ETUR at once: tiling, cache logic, sampler behavior, PiD VAE decoding, segment rebuilding, frontend state, and worker cleanup.

## Highlights

- NVIDIA PiD can now be used as a dedicated tiled image/latent upscaler.
- PiD can be selected as the ETUR refiner VAE/decode mode.
- PiD pre-upscaling is available directly inside the Tiler/Upscaler.
- New PiD model download/load node.
- Flux 2 gets a new safer inpaint path and an optional Flux2 sampler hook.
- Tile Overrides now support per-tile CFG, model, ControlNet pipe, color-match behavior, and "ignore general prompt".
- Batch IMAGE inputs can now be processed safely one image at a time.
- New tile-aware color correction and geometry drift correction were added.
- Segment processing, segment cache invalidation, and final PiD segment compositing were heavily improved.
- Qwen VLM / thinking-model output handling was improved.
- Worker startup/shutdown and stale worker cleanup were made more robust.

## New Nodes

### TBG ETUR Download PiD Model

A new PiD loader node downloads and loads supported PixelDiT/PiD model files from the Comfy-Org/PixelDiT repository.

The PiD bundle contains two model files:

- the PiD diffusion model, stored in ComfyUI's `models/diffusion_models`
- the PiD text encoder, stored in ComfyUI's `models/text_encoders`

The node has a `Load PiD CLIP` switch. When this is enabled, the node also loads the PiD text encoder as a PixelDiT CLIP and outputs it.

It outputs:

- PID Model
- PID CLIP, if `Load PiD CLIP` is enabled
- PID Model Info, as metadata/compatibility output

The `PID Model` output can be connected to the standalone PiD upscaler, the Tiler Labs node, or the Refiner Labs node. `PID CLIP` is optional; downstream PiD nodes also have optional `PID_CLIP` inputs, so advanced users can connect their own compatible CLIP/text encoder instead of using the loader output.

### TBG ETUR PID Tile Upscale Rebuild

A new standalone PiD tiled upscaler node was added.

It accepts either an IMAGE input or a source LATENT input. This means latent workflows can avoid unnecessary VAE encode/decode cycles when possible, helping preserve image quality during PiD upscale workflows.

This node uses ETUR's shared PiD tiled upscale runtime. It calls the same `run_pid_tiled_upscale` path used by ETUR's PiD pre-upscale stage, and the final tile rebuild/stitching uses the shared GPU-accelerated PiD rebuild code when available.

The Refiner's "Nvidia PiD 4x" VAE mode uses the same PiD model family and the same GPU rebuild/stitching layer, but it enters through an ETUR-aware latent-decode path after tile sampling. That refiner path has to keep tile masks, segment handling, final rebuild state, and ETUR color/stabilization stages intact.

Main controls include:

- PID_VAE_Compatible_Model
- upscale_by
- steps
- cfg
- sampler_name
- scheduler
- denoise
- degrade_sigma
- prompt
- optional sampler override
- optional PiD CLIP override
- optional PiD source VAE override

It returns:

- final rebuilt PiD image
- PiD tile previews

### TBG Flux2 Differential Diffusion Inpainting

A new standalone Flux 2 sampler hook node was added under TBG/Sampler.

It stores the inpaint mask privately and can wrap a sampler so Flux 2 mask correction happens later in the sampling process, around the tested Flux 2 correction window. This is useful for Flux 2 inpaint workflows where normal noise-mask behavior can become too aggressive or get stuck.

Inputs include:

- model
- latent
- inpaint_mask
- denoise
- correction_start_sigma
- optional sampler

Outputs:

- model
- latent
- sampler

### TBG Color Correction

A standalone color correction node was added for workflows that need ETUR color stabilization without running the full refiner.

It supports:

- detail-preserving global RGB/luma stabilization
- optional geometry drift correction
- Color Match Strength
- optional mask support internally

### TBG SIFT+ Drift Correction

A standalone geometry drift correction node was added.

It uses the same feature-alignment family as the system behind PixelDriftFix, but ETUR does not align directly from the raw image. It first builds high-pass and edge-focused detail images so edges, texture, and structural lines are easier to match.

If the primary SIFT alignment cannot find a reliable match, ETUR automatically tries a fallback cascade: high-pass SIFT, wide high-pass SIFT, AKAZE/MLDB matching, RootSIFT matching, border-only matching, contour/edge matching, and a weighted translation search. In the strongest internal modes, ETUR also has an optical-flow fallback. This helps recover small shifts even when the image has changed too much for the first SIFT pass.

### AI Source Pattern Cleanup

A new preprocessing node reduces high-frequency source micro-grid patterns often found in direct AI-generated source images.

It is intended for 1:1 source/reference images before ETUR processing, not as a final image beauty filter.

Outputs:

- cleaned image
- optional difference/debug image

### TBG Model Agnostic Color Anchor

A new model-agnostic color-stability sampler hook was added.

This node is inspired by the Flux 2 Klein Color Anchor idea, but it does not need a separate reference-image input. Instead, it captures the VAE-encoded image latent that is already fed into the sampler, then preserves the per-channel latent color means during sampling.

Because it works from the sampler input latent, it remains model-agnostic and can still work inside complex tiled sampling paths where ETUR does not have a simple direct hook into each tile's reference image.

It behaves more like a ControlNet-style timing control than a simple on/off correction. Users can set:

- strength
- start_percent
- end_percent
- ramp_curve

This timing control was important in testing. A full-time color anchor could disturb Flux 2 detail creation, so ETUR lets the anchor act mainly during the early color-forming part of sampling and then lets the model freely create new details later.

Important note: this only works as a real source-image color anchor when the sampler receives a real VAE-encoded source/input image latent. If the sampler starts from an Empty Latent, there is no source-image color information to preserve.

### TBG Model Agnostic Latent Anchor

A stronger model-agnostic source-latent preservation sampler hook was added.

This node follows the same ETUR idea as the Color Anchor, but instead of preserving only the per-channel color means, it captures the full VAE-encoded image latent that is fed into the sampler. During sampling, it can gently pull the denoised latent back toward that original source latent.

That makes it fully model-agnostic. It does not need a separate reference-image connection, and it can still work in tiled or complex ETUR sampling paths where the sampler only sees the current tile latent internally.

Latent Anchor is designed for stronger preservation than Color Anchor. It can help keep:

- structure
- texture
- style
- local layout
- color

It also uses ControlNet-style timing controls:

- strength
- start_percent
- end_percent
- ramp_curve

This timing is important because a full latent lock can become too restrictive if it is active for the whole sampling process. In practice, users can apply it only during the part of sampling where they want source preservation, then let the model continue generating new details more freely.

Important note: this node is much stronger than Color Anchor. High strength can preserve too much of the source image, reduce edit freedom, or create ghosting. It should be treated as an advanced preservation tool, especially for Flux 2 and other highly editable model workflows.

### TBG ETUR ColorMatch Debug Gates

A developer/debug node was added because Flux 2 and PiD introduced several different color-drift correction stages. We needed direct UI control for testing, so each important color and stabilization stage can now be isolated.

The node exposes an `Override_Normal_Gates` switch and a numbered set of stage switches. With override off, the switches only disable stages that ETUR would normally run. With override on, the switches can force supported stages on or off for deeper debugging.

Current wired debug gates are:

- Flux 2 PiD normal-VAE reference creation
- Flux 2 PiD after-PiD-VAE detail-preserving color correction
- Flux 2 PiD post-tone and seam-local low-frequency stabilization
- Segment post-VAE/PiD color matching
- Final tile-only color correction before segment rebuild
- Final segment-aware color-base generation before PiD final matching
- PiD final 4x color-match container
- Final per-area tile/segment override path for Protect / Origin / Off color behavior
- Final global TBG detail-preserving color mode

This node is mainly for development and troubleshooting. It now only exposes stages that are actually wired in the current refiner code.

## NVIDIA PiD Integration

PiD integration is the biggest feature in this release.

TBG ETUR now supports PiD in three main ways:

1. As a standalone tiled PiD upscaler.
2. As a pre-upscale option inside the Tiler/Upscaler.
3. As a refiner VAE/decode mode using the new "Nvidia PiD 4x" option.

### PiD as a Refiner VAE

The old VAE encode switch has been expanded into a three-option selector:

- tiled slow
- tbg Color-preserving fast
- Nvidia PiD 4x

When "Nvidia PiD 4x" is selected, ETUR uses PiD as the 4x decode/upscale stage for supported models.

Supported refiner model types:

- FLUX1
- FLUX2
- FLUX1 Kontext
- Qwen Image
- Qwen Image Edit
- SDXL
- SD3
- Z-Image

For native 1024x1024 regions, ETUR can use the faster PiD path. For other tile or segment sizes, it switches to tiled PiD latent decoding so unusual aspect ratios and larger regions can still be processed.

### PiD Pre-Upscale in the Tiler

PiD model options are now available in the Tiler/Upscaler model list. When selected, ETUR prepares the image, runs PiD tile upscale, rebuilds the image, and then continues into normal ETUR tiling/refinement.

This makes PiD usable on practical image sizes instead of only ideal fixed-size inputs.

### PiD Model Loading

PiD can be loaded automatically from the selected PiD model option, or manually supplied through the Labs inputs:

- PID_Model
- PID_VAE_Compatible_Model
- PID_CLIP
- PID_Source_VAE
- PID_degrade_sigma

`PID_CLIP` is optional. If it is not connected, ETUR loads the text encoder defined by the selected PiD model spec when the file is available locally. The Download PiD Model node is the easiest way to download that text encoder and output it as `PID CLIP`.

### PiD Sampling

The PiD default sampler is `pid_sde`. ETUR now registers this sampler directly from inside the TBG ETUR custom node, so automated PiD workflows no longer need a separate PiD sampler custom node installed.

ETUR also registers `pid_creative_sde` and includes a TBG PiD Creative SDE Sampler node for users who want to experiment with a more creative PiD sampling path.

ComfyUI still provides the core PixelDiT/PiD model architecture, latent format, and PixelDiT text-encoder support. TBG ETUR provides the PiD workflow integration, model download/load node, tiled rebuild logic, and PiD SDE sampler registration.

## Flux 2 Improvements

### New TBG Flux2 Sampler

The release registers a new sampler name:

- TBG Flux2 Sampler

This sampler is designed for Flux 2 inpainting. It delays latent correction until the configured Flux 2 correction region instead of applying the mask too aggressively from the beginning.

### Flux2 Sampler Hook Switch

The Labs Refiner node now includes:

- Flux2 Sampler Hook

Default behavior is OFF.

OFF uses the safer path:

- normal VAE encode
- ComfyUI Set Latent Noise Mask

ON uses the private/legacy Flux2 differential sampler hook path.

This makes Flux 2 workflows safer by default while still allowing experimental users to enable the hook path.

### Differential Diffusion Defaults

Differential Diffusion is enabled by default in the refiner paths.

ETUR now adjusts it automatically depending on fusion mode:

- Neuro Generative Tile Fusion and NGTF Flux Kontext force Differential Diffusion ON.
- Soft Merge forces Differential Diffusion OFF.

### Flux 2 Reference / Negative Conditioning Fixes

Flux 2 reference behavior was corrected so crop-aware conditioning can include both positive and negative conditioning. The Labs Refiner supports:

- cropped_positive
- cropped_negative

These are full-image conditioning inputs. ETUR crops the matching region per tile and appends it to the tile conditioning instead of overwriting the normal tile prompt conditioning.

This helps Flux 2 behave more consistently, especially at high CFG or high restoration strength.

### Flux 2 Tile Color Correction

Flux 2 color correction is now controlled through the Labs Refiner:

- Tile Color Correction

If Labs is not connected, ETUR keeps this enabled by default.

Flux 2 PiD also gained dedicated color stages for:

- normal VAE reference creation
- after-PiD VAE color correction
- post-tone correction
- local tile/context color drift correction

## Tiler and Upscaler Changes

### Optimize Upscale Factor for Tile Use

The Tiler/Upscaler now has:

- Optimize upscale factor for optimal tile use

When enabled, ETUR:

- reads the requested `upscale_by`
- checks the tile grid it would create
- keeps the same practical tile count
- slightly lowers the effective upscale factor when needed
- avoids unnecessary overlap waste
- preserves aspect ratio
- uses one shared scale value
- never stretches the image

The goal is cleaner tile usage, more predictable caching, and better refiner behavior.

### PiD Upscale Options in the Tiler

PiD options are now included in the Tiler/Upscaler upscale model list. When used there, PiD runs before final ETUR tile generation.

### Batch Pipe Creation

The Tiler now detects IMAGE batches. Instead of generating all tiles for all images at once, it returns a lightweight batch pipe that the Refiner processes sequentially.

This prevents large batch jobs from exploding memory use.

### Better Tiler Cache Fingerprinting

The tiler cache now fingerprints more of the real tiler state, including segment mask contents. Changing a segment mask now invalidates stale cached tiles.

## Refiner Changes

### PiD VAE Decode in the Refiner

The Refiner can now decode/refine with PiD through the "Nvidia PiD 4x" VAE mode.

ETUR handles:

- PiD-compatible model type validation
- 1024x1024 fast PiD path
- tiled PiD latent decode for non-standard tile sizes
- shared GPU-accelerated PiD tile rebuild/stitching
- PiD final image rebuild
- PiD final segment composite
- PiD 4x color matching
- PiD segment-aware color base generation
- PiD debug image output in Dev mode

### Selected Tile Regeneration

Selected-tile workflows were improved.

When only selected tiles are regenerated, ETUR can use the latest final image as the canvas, cut it back into tile inputs, regenerate only the selected tiles, blend those selected tiles back in, and save the result as the new latest final image.

Important behavior:

- If the tiler context changes, ETUR clears the generated tile cache and `last_final_image`.
- After that, it rebuilds from the current upscaled input instead of reusing stale tile data.

### Tile-Aware Color Stabilization

The refiner now has a new detail-preserving tile-aware color stabilizer.

The default Color Match mode in PRO is now the TBG ETUR Detail-Preserving Tile Stabilizer. It performs color stabilization with awareness of tile borders and generated neighbor context. When tile-aware conditions are unavailable, it falls back to the global TBG Detail-Preserving Color Stabilizer.

### Geometry Drift Correction

The refiner now includes SIFT-based geometry drift correction.

This aligns generated tiles back to their reference tiles after sampling. It is especially useful when a model slightly shifts anatomy, edges, objects, or texture blocks during upscale/refine.

### Batch Refiner

The Refiner now detects batch pipes and processes each image separately:

1. Tiler runs for one image.
2. Tile Overrides are applied for that image.
3. Refiner processes that image.
4. Results are collected and concatenated into a final IMAGE batch.

Important behavior:

- Single-image workflows stay on the normal path.
- Batched image inputs are refined one at a time.
- Batched masks and similar tensors are sliced per image when their batch dimension matches.
- Mixed output sizes raise a clear error asking the user to resize or pad before ETUR.
- In batch mode, the Tiles output stays lightweight instead of accumulating every preview tile from every image.

## Tile Override System

The Tile Overrides node is much stronger in 1.2.

Per-tile controls now include:

- prompt override
- denoise override
- seed override
- ControlNet strength override
- CFG override
- model override
- ControlNet pipe override
- color match override
- ignore general prompt

### Per-Tile Model Override

You can now connect up to three model override inputs and select them per tile.

This is useful when one tile needs a consistency model/LoRA setup and another tile needs a more editable model/LoRA setup.

### Per-Tile ControlNet Pipe Override

You can now connect up to three ControlNet pipe override inputs and select them per tile.

This is useful when one tile needs a strong reference image and another tile needs a softer consistency reference.

### Per-Tile CFG Override

CFG can now be overridden per tile.

This is especially useful for Flux 2 workflows where CFG often behaves like restoration/edit strength. Some tiles may need a lower value to preserve structure, while others may need a higher value to repair or regenerate details.

### Per-Tile Color Match Override

Each tile or segment can choose its final color behavior:

- Preset Color Match: use the global refiner Color Match setting.
- Protect New Generated Content: protect the tile/segment from aggressive full-origin correction.
- Color Match From Origin: allow that tile/segment area into full-origin matching.
- Color Match Off: exclude that tile/segment area from color correction.

This is especially important for PiD final color matching, where different segments may need different color authority.

### Ignore General Prompt

Tiles can now ignore the global/general prompt. This helps when a local tile prompt needs to be clean and specific.

### Override Change Detection Fix

Tile Overrides now participates more correctly in ComfyUI change detection.

The JSON override data is fingerprinted so changing per-tile settings can requeue the Refiner instead of silently reusing old outputs.

This fixes the bug where Tile Overrides could fail to trigger a Refiner rerun.

### Tiler Context Safety

Tile override JSON now tracks the upstream tiler context. If the tiler grid or image context changes, stale tile override data is ignored instead of being applied to the wrong tile layout.

## Segment Processing

### Segment Mask Hashing

Segment masks are now fingerprinted by tensor contents.

This means changing the actual mask invalidates stale cached tiles instead of only reacting to object identity or high-level settings.

### Sequential Segment Reference Behavior

Segment processing now has better sequential canvas behavior:

- Segment 1 starts from the tile-only final image.
- Later segments can see earlier processed/cached segments.
- Cached segments update the sequential reference canvas.
- The final segment rebuild can reuse the sequential canvas instead of double-compositing.

### PiD Segment Support

PiD segment handling was heavily improved.

ETUR can now:

- run segment PiD decode
- use tiled PiD latent decode for non-1024 segment regions
- build 4x segment-aware final color references
- composite PiD segments into the final 4x image
- widen final color-match masks around segment areas to avoid hard color transitions
- preserve native PiD geometry before color lock when needed

### Segment Background Harmonization

The Labs Refiner now includes:

- Segment Background Harmonization

This applies low-frequency color/tint harmonization around generated segments while protecting the object/core mask.

### Segment Transform and Debug Plumbing

Segment processing now carries more internal state:

- segment sampling transforms
- segment binary masks
- cropped masks
- inpainting masks
- compositing masks
- denoise mask tiles
- segment crop regions
- NGTF/debug mask output in Dev mode

## Color and Drift Features

### Detail-Preserving Color Stabilizer

A new non-destructive color correction system was added.

It is designed to stabilize color and tone without destroying newly generated details. The CE/global version uses RGB and luma correction for a safer full-image match. For API members, the PRO tile-aware version can use a smarter NGTF-based technique that also considers local tile borders and already generated neighbor context.

ETUR switches between these behaviors automatically in the background depending on the available CE or PRO feature path.

### Tile-Aware ETUR Color Correction

Inside ETUR, color correction can now happen at multiple stages:

- after tile fusion
- after normal tile VAE decode
- after segment VAE/PiD decode
- after segment fusion
- after tile-only final rebuild
- during final PiD color matching

### SIFT+ Geometry Drift Protection

SIFT drift protection is now available both as a Refiner switch and as a standalone helper node.

This helps correct small position and geometry shifts introduced during sampling, especially in tiled workflows where small shifts can become visible at tile borders.

### Model-Agnostic Stabilizers

Two model-agnostic stabilizer hooks were added:

- TBG Model Agnostic Color Anchor
- TBG Model Agnostic Latent Anchor

These are not limited to one model family and can help with color-only or full latent stabilization.

## VLM and Prompting Updates

### More Qwen VLM Options

The VLM model list now includes additional Qwen3-VL options, including instruct, thinking, FP8, and GGUF aliases.

### OpenAI-Compatible VLM Server

The Tiler can now route VLM requests to an OpenAI-compatible local/server backend through Labs Upscaler settings:

- VLM_Server_Base_URL
- VLM_Server_Model

The API key is read from:

- TBG_ETUR_OPENAI_API_KEY

### Qwen Thinking Output Cleanup

Qwen thinking-model responses are now cleaned before being written into tile prompts.

Thinking text is stripped and logged, while only the final cleaned answer is used as the tile prompt. This prevents reasoning text from being inserted into image prompts.

## Worker and Runtime Improvements

Worker handling was improved in several areas:

- worker pidfile tracking
- stale worker reaping
- safer worker shutdown
- safer cleanup at ComfyUI exit
- worker device environment handling
- better diagnostics
- more robust OpenAI/VLM server manager state handling

This should reduce cases where an old worker process keeps running or the wrong worker state is reused.

## Frontend Updates

### Tile Overrides Frontend

The Tile Overrides frontend JavaScript was updated for the new override fields:

- CFG override
- model override
- ControlNet pipe override
- color match override
- ignore general prompt

It also improves:

- requeue widget handling
- tile override refresh detection
- tile override change detection
- session/cache behavior
- context-key validation
- mouse wheel behavior inside custom fields

### Main ETUR Frontend

The main ETUR frontend JavaScript was updated for compatibility with the new node schema and runtime behavior.

### Node Info Text Fix

Node info text now stays inside nodes more reliably.

## Workflow and Packaging Updates

The example workflow set was updated around the 1.1.19 development point and now includes new or updated workflows for:

- Flux1 DEV PRO
- Flux2 Klein PRO
- CE workflows
- TBG ETUR + PiD Flux1
- TBG ETUR + PiD Flux2
- PiD Stand-Alone

Compatibility JavaScript and packaged application/runtime files were also updated.

## Other Technical Additions

### RF Inversion + UntwistingRoPE Pipe

A new optional `TBG ETUR RF UntwistingRoPE Pipe` was added for users experimenting with RF Inversion and UntwistingRoPE workflows inside ETUR.

This is an integration pipe, not a replacement for the UntwistingRoPE node itself. The UntwistingRoPE custom node/runtime must be installed for this feature path to work.

The pipe is disabled by default and exposes advanced per-tile/segment adapter settings. When enabled, ETUR can pass the RF / UntwistingRoPE settings through its tiled and segment workflows so advanced users can test RF inversion behavior without manually rebuilding the whole ETUR graph.

### Cropped Conditioning

The Labs Refiner supports full-image cropped conditioning:

- cropped_positive
- cropped_negative

ETUR crops the correct area for each tile and appends it to the tile's normal conditioning.

### ColorMatch Debug Stages

The new debug gate node exposes detailed stage names for development and diagnosis. This makes it easier to isolate exactly where color changes are happening.

## Bug Fixes

### Tile Overrides Did Not Trigger Refiner Rerun

Fixed a bug where changing Tile Overrides could fail to trigger a Refiner rerun.

The override JSON is now included in change detection, and stale override cache is cleared when empty or invalid.

### Stale Tile Cache After Tiler Changes

Fixed stale generated tile reuse when the tiler context changes.

ETUR now clears generated tile cache and `last_final_image` when tiler-relevant settings change.

### Segment Mask Cache Bug

Changing a Segment Mask now invalidates cached tiler data correctly.

### Flux 2 Reference / Negative Conditioning

Flux 2 crop-aware reference conditioning now supports the negative path instead of only relying on positive/reference information.

This improves high-CFG Flux 2 workflows and makes reference behavior more predictable.

### Qwen Thinking Models

Thinking output from Qwen thinking models is stripped from tile prompts and written to logs instead.

### VLM / LLM None Handling

Missing VLM/LLM values are handled more safely around TBG nodes.

### Worker Lifecycle Fixes

Worker shutdown, stale worker detection, pidfile handling, and cleanup were improved.

### PiD Final Rebuild Stability

PiD final rebuild now has safer GPU/worker paths and better fallback behavior when rebuilding tile-only and segment-aware final images.

### Mixed Batch Output Size Error

Batch mode now raises a clear error if the refined images do not share the same output size.

### Tile Override Mouse Wheel Zoom

Mouse wheel behavior inside override fields was improved so it no longer accidentally breaks the normal ComfyUI canvas zoom behavior.

## Accuracy Notes Compared to the Draft

The actual code confirms the PiD default sampler is `pid_sde`, but it does not add separate public sampler names called "PiD_SDE Sampler" and "PiD_SDE Creative Sampler".

The current package metadata still reports `pyproject.toml` version `1.1.18` and `py/utils/version.py` as `beta 1.1.0`. If this post is for the official 1.2 release, those version strings should be updated before publishing.

## Short Summary

TBG ETUR 1.2 is a large upgrade focused on practical high-quality tiled refinement. The biggest change is full PiD integration across standalone upscaling, tiler pre-upscale, and refiner VAE decode. Flux 2 workflows are safer and more controllable, tile overrides are much more powerful, batch workflows are memory-safe, and color/geometry stabilization has been rebuilt around more detail-preserving methods.
