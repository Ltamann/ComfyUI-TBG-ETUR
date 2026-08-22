"""
TBG VRAM Optimization Module - Unified Conditioning (minimal)
"""

import numpy as np
import torch
import nodes

from ....utils.log import log
from .cnet import apply_controlnets_from_pipe, normalize_controlnet_mode, prepare_krea2_depth_control
from .redux import FluxRedux_ForTiles
from comfy import model_management

device = model_management.get_torch_device()

class VRAMOptimizer:
    """
    Unified per-tile conditioning cache:
    - precompute_all_embeddings(): CLIP → optional Redux → optional CNet (CPU)
    - unified_condition_to_gpu(tile_idx): return (pos, neg) on GPU from cache
    """

    def __init__(self, SELF):
        self.SELF = SELF
        self.INFO = SELF.INFO
        self.KSAMPLER = SELF.KSAMPLER
        self.INPUTS = SELF.INPUTS
        self.OUTPUTS = SELF.OUTPUTS
        self.PARAMS = SELF.PARAMS
        self.PROMPTER = SELF.PROMPTER
        self.API = SELF.API
        self.node_id = SELF.INFO.id
        self.DUALMODELvae = SELF.DUALMODEL.vae
        self.DUALMODELmodel = SELF.DUALMODEL.model
        self.DUALMODELclip = SELF.DUALMODEL.clip
        self.DUALMODELGeneral_Prompt = SELF.DUALMODEL.General_Prompt
        self.DUALMODELGeneral_Prompt_Negative = SELF.DUALMODEL.General_Prompt_Negative
        self.lowvram = SELF.lowvram
        self.vram_profile = getattr(
            self.KSAMPLER,
            "vram_profile",
            "Low VRAM Cache (Unload Models)",
        )
        #self.encode_all_text_prompts()
        self.grid_prompts = self.PROMPTER.output_prompts
        self.ignore_general_prompts = list(
            getattr(self.PROMPTER, "output_ignore_general_prompt_js", []) or []
        )

        #self.orig_grid_images = self.OUTPUTS.orig_grid_images_all
        self.orig_grid_images = self.OUTPUTS.orig_grid_images_all
        self.image = self.OUTPUTS.upscaled_image

        self.clip_model = self.KSAMPLER.clip
        self.general_prompt = self.KSAMPLER.General_Prompt
        self.general_prompt_negative = self.KSAMPLER.General_Prompt_Negative

        self.redux_enabled = bool(
            self.PARAMS.Redux_Clip_Vision
            and self.PARAMS.Redux_Style_Model
            and self.PARAMS.Redux_strength > 0.01
        )
        self.redux_strength = self.PARAMS.Redux_strength
        self.redux_style_model = self.PARAMS.Redux_Style_Model
        self.redux_clip_vision = self.PARAMS.Redux_Clip_Vision

        self.text_embeddings_cache = {}  # tile_idx -> {"conditioning","negative","fingerprint"}
        self.text_embeddings_cache_low = {}  # Low model (optional)
        self.vl_conditioning_cache = {}  # tile_idx -> vl_conditioning (for Qwen models)
        self.krea2_control_latents = {}
        self.device = device
        self.last_timestamp = getattr(self.PARAMS, "timestamp", 0)

        self.precompute_all_embeddings(SELF)
        print(f"DEBUG_INIT: about to call precompute_vl_conditioning, VL_Encoding_Mode={getattr(self.PARAMS, 'VL_Encoding_Mode', 'NOT_SET')}")
        self.precompute_vl_conditioning(SELF)

    def _copy_embedding_entry(self, emb):
        if not isinstance(emb, dict):
            return emb
        return {
            "conditioning": emb.get("conditioning"),
            "negative": emb.get("negative"),
            "fingerprint": emb.get("fingerprint"),
        }

    def _filtered_controlnet_pipe(self):
        pipe = getattr(self.KSAMPLER, "Controlnet_Pipe", None) or []
        filtered = []
        for p in pipe:
            mode = normalize_controlnet_mode(p)
            if mode != "ControlNet":
                continue
            cn = (p.get('controlnet') or getattr(p, 'controlnet', None) or getattr(p, 'model', None))
            if cn is None:
                continue
            filtered.append(p)
        return filtered

    def is_ultra_low(self):
        return self.vram_profile == "Ultra Low Memory (Per-Tile Streaming)"

    def _unload_vl_encoder(self, stage):
        """Release only the VL CLIP when the profile does not keep models cached."""
        if self.vram_profile == "Fast Cache (Max Speed)":
            return

        patcher = getattr(self.clip_model, "patcher", None)
        if patcher is None or model_management.is_device_cpu(patcher.load_device):
            return
        if patcher.loaded_size() <= 0:
            return

        loaded_models = model_management.current_loaded_models
        if not any(loaded.model is patcher for loaded in loaded_models):
            return

        keep_loaded = [loaded for loaded in loaded_models if loaded.model is not patcher]
        unloaded = model_management.free_memory(
            1e30,
            patcher.load_device,
            keep_loaded,
        )
        if unloaded:
            model_management.soft_empty_cache()
            log(
                f"[TBG][{self.vram_profile}] unloaded VL encoder after {stage}",
                None,
                None,
                f"Node {self.node_id}",
            )

    def _ultra_stage_unload(self, stage):
        if not self.is_ultra_low():
            return
        before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        log(
            f"[TBG][UltraLow] stage unload {stage}: {before:.2f}GB -> {after:.2f}GB",
            None,
            None,
            f"Node {self.node_id}",
        )

    # ------------------------------------------------------------------ utils

    def _nuclear_gpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device).float().contiguous()
        if isinstance(obj, dict):
            return {k: self._nuclear_gpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._nuclear_gpu(x) for x in obj)
        return obj

    def _nuclear_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().half().contiguous()
        if isinstance(obj, dict):
            return {k: self._nuclear_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._nuclear_cpu(x) for x in obj)
        return obj

    # -------------------------------------------------------- fingerprint/hash

    def _get_cnet_pipe_hash(self):
        pipe = getattr(self.KSAMPLER, "Controlnet_Pipe", None)
        if not pipe:
            return 0
        parts = []
        for p in pipe:
            mode = normalize_controlnet_mode(p)
            if mode == "Model_Patch":
                model_obj = p.get("model_patch")
            else:
                model_obj = p.get("controlnet")
            model_name = model_obj.__class__.__name__ if model_obj is not None else "none"
            pre = str(p.get("preprocessor", "None"))
            s = p.get("strength", 1.0)
            st = p.get("start", 0.0)
            en = p.get("end", 1.0)
            parts.append(f"mode={mode}|m={model_name}|p={pre}|s={s}|st={st}|en={en}")
        return hash(tuple(parts))

    def _get_redux_hash(self):
        style = self.redux_style_model.__class__.__name__ if self.redux_style_model else "none"
        vision = self.redux_clip_vision.__class__.__name__ if self.redux_clip_vision else "none"
        return hash(f"{style}_{vision}")

    def _hash_tile_image(self, tile_idx):
        tile_image = self.get_tile_image(tile_idx)
        if tile_image is None:
            return 0
        try:
            if isinstance(tile_image, torch.Tensor):
                pixels = (tile_image.cpu().float().clamp(0, 1) * 255).byte().numpy()
            else:
                pixels = np.asarray(tile_image).astype(np.uint8)
            h, w = pixels.shape[-2:]
            if h > 64 or w > 64:
                sh = max(1, h // 64)
                sw = max(1, w // 64)
                pixels = pixels[:, ::sh, ::sw] if pixels.ndim == 4 else pixels[::sh, ::sw]
            return hash(pixels.tobytes())
        except Exception as e:
            log(f"Pixel hash error: {e}", None, None, f"Node {self.node_id}")
            return hash(f"error_{tile_idx}")

    def _get_tile_fingerprint(self, tile_index):
        if tile_index >= len(self.grid_prompts):
            return None
        combined_prompt = self._tile_prompt(tile_index)
        prompt_hash = hash(str(combined_prompt))
        redux_hash = self._get_redux_hash() if self.redux_enabled else None
        cnet_hash = self._get_cnet_pipe_hash() if getattr(self.KSAMPLER, "Controlnet_Pipe", None) else None
        image_hash = self._hash_tile_image(tile_index)
        ts = getattr(self.PARAMS, "timestamp", 0)
        denoise = getattr(self.KSAMPLER, "denoise", None)
        # --- DEV: Trace denoise + custom_sigmas in fingerprint ---
        if self.API.status == "Dev" or getattr(self.API, "dev_debug_enabled", False):
            try:
                _cs = getattr(self.KSAMPLER, "custom_sigmas", None)
                if torch.is_tensor(_cs):
                    _sigsig = tuple(round(float(v), 6) for v in _cs.detach().float().cpu().view(-1).tolist()[:8])
                    _siglen = int(_cs.numel())
                elif _cs is None:
                    _sigsig = None
                    _siglen = 0
                else:
                    _vals = [round(float(v), 6) for v in list(_cs)]
                    _sigsig = tuple(_vals[:8])
                    _siglen = len(_vals)
            except Exception:
                _sigsig = "err"
                _siglen = -1
            try:
                log(f"[TBG DevDenoise] VRAM fingerprint tile={tile_index}: denoise={denoise} redux={self.redux_strength} cs_len={_siglen} cs_head={_sigsig}", None, None, f"Node {self.node_id}")
            except Exception:
                pass
        # --- END DEV ---
        return hash(
            f"t{tile_index}_p{prompt_hash}_img{image_hash}_c{cnet_hash}_r{redux_hash}"
            f"_s{self.redux_strength}_d{denoise}_ts{ts}"
        )

    def _ignore_general_prompt_for_tile(self, tile_idx):
        try:
            if tile_idx < len(self.ignore_general_prompts):
                return bool(self.ignore_general_prompts[tile_idx])
        except Exception:
            pass
        return False

    def _tile_prompt(self, tile_idx, general_prompt=None):
        tile_prompt = ""
        try:
            if tile_idx < len(self.grid_prompts):
                tile_prompt = self.grid_prompts[tile_idx] or ""
        except Exception:
            tile_prompt = ""
        if self._ignore_general_prompt_for_tile(tile_idx):
            return str(tile_prompt).strip()
        general = self.general_prompt if general_prompt is None else general_prompt
        return " ".join(str(part).strip() for part in (general, tile_prompt) if str(part or "").strip())

    # ------------------------------------------------------------ tile access

    def get_tile_image(self, tile_idx):
        if (
            hasattr(self, "orig_grid_images")
            and self.orig_grid_images
            and len(self.orig_grid_images) > tile_idx
        ):
            return self.orig_grid_images[tile_idx]
        return None

    # ------------------------------------------------------------ CLIP encode

    def encode_single_prompt(self, combined_prompt):
        pos = nodes.CLIPTextEncode().encode(self.clip_model, combined_prompt)[0]
        neg = nodes.CLIPTextEncode().encode(self.clip_model, self.general_prompt_negative)[0]
        return {
            "conditioning": self._nuclear_cpu(pos),
            "negative": self._nuclear_cpu(neg),
        }

    def encode_single_prompt_low(self, combined_prompt):
        pos = nodes.CLIPTextEncode().encode(self.DUALMODELclip , combined_prompt)[0]
        neg = nodes.CLIPTextEncode().encode(self.DUALMODELclip , self.DUALMODELGeneral_Prompt_Negative)[0]
        return {
            "conditioning": self._nuclear_cpu(pos),
            "negative": self._nuclear_cpu(neg),
        }

    def encode_all_text_prompts(self):

        prompt_cache = {}
        tile_cache = {}
        cached_used = 0
        new_cached = 0

        total_tiles = len(self.orig_grid_images)
        total_tiles = len(self.grid_prompts)

        for tile_idx in range(total_tiles):

            if self.PARAMS.tiles_to_process and tile_idx not in self.PARAMS.tiles_to_process:
                continue

            combined_prompt = self._tile_prompt(tile_idx)
            ph = hash(str(combined_prompt))


            if ph in prompt_cache:
                tile_cache[tile_idx] = self._copy_embedding_entry(prompt_cache[ph])
                cached_used += 1

            else:
                emb = self.encode_single_prompt(combined_prompt)
                prompt_cache[ph] = emb
                tile_cache[tile_idx] = self._copy_embedding_entry(emb)
                new_cached += 1


        self.text_embeddings_cache = tile_cache
        uniq = len(prompt_cache)
        savings = total_tiles - uniq
        allc = cached_used + new_cached
        log(
            f"CLIP: {uniq} unique → {savings} saved / "
            f"reused={cached_used} / new={new_cached} / total={allc}",
            None,
            None,
            f"Node {self.node_id}",
        )

    def encode_all_text_prompts_low(self):
        total_tiles = len(self.grid_prompts)
        prompt_cache_low = {}
        tile_cache = {}
        cached_used = 0
        new_cached = 0

        for tile_idx in range(total_tiles):
            if self.PARAMS.tiles_to_process and tile_idx not in self.PARAMS.tiles_to_process:
                continue
            combined_prompt = self._tile_prompt(tile_idx, self.DUALMODELGeneral_Prompt)

            ph = hash(str(combined_prompt))
            log(
                f"TBG[Node {self.node_id}] Prompt: {combined_prompt} Tile → {tile_idx + 1} ",
                None,
                None,
                f"Node {self.node_id}",
            )
            if ph in prompt_cache_low:
                tile_cache[tile_idx] = self._copy_embedding_entry(prompt_cache_low[ph])
                cached_used += 1

            else:
                emb = self.encode_single_prompt_low(combined_prompt)
                prompt_cache_low[ph] = emb
                tile_cache[tile_idx] = self._copy_embedding_entry(emb)
                new_cached += 1
        self.text_embeddings_cache_low = tile_cache

        uniq = len(prompt_cache_low)
        savings = total_tiles - uniq
        allc = cached_used + new_cached
        log(
            f"TBG[Node {self.node_id}] CLIP: {uniq} unique → {savings} saved / "
            f"reused={cached_used} / new={new_cached} / total={allc}",
            None,
            None,
            f"Node {self.node_id}",
        )

    def precompute_vl_conditioning(self, SELF):
        """Precompute VL conditioning for Global Embedding mode (Qwen only).

        Tile Embedding mode encodes on-the-fly from tile_to_process in conditioning().
        """
        print(f"DEBUG_VL: precompute_vl_conditioning ENTERING node={self.node_id}")
        from .vl_encode import build_vl_conditioning

        # Debug: log the VL mode being used
        vl_mode = getattr(self.PARAMS, 'VL_Encoding_Mode', 'Off')
        log(
            f"VL: precompute mode check = '{vl_mode}' (PARAMS keys={list(self.PARAMS.__dict__.keys())})",
            None, None, f"Node {self.node_id}",
        )

        # Only run for Qwen / Krea2 VL models
        model_type = getattr(self.KSAMPLER, 'model_type', None)
        if model_type not in ("Qwen Image", "Qwen Image Edit", "Krea2"):
            return

        # Check VL mode
        if vl_mode not in ("Global Embedding", "Tile + Global Embedding", "Inner Tile + Global Embedding"):
            return

        if self.clip_model is None:
            return

        full_image = getattr(self, 'image', None)
        if full_image is None or full_image.shape[0] != 1:
            return

        grid_specs = list(getattr(self.PARAMS, "grid_specs", []) or [])
        if not grid_specs:
            return

        canvas_h = int(full_image.shape[1])
        canvas_w = int(full_image.shape[2])

        vl_strength = float(getattr(self.PARAMS, 'VL_Strength', 0.5))
        if not (0.0 <= vl_strength <= 1.0):
            vl_strength = 0.5
        log(
            f"VL: Global Embedding mode — encoding canvas, "
            f"vl_strength={vl_strength:.2f} (model={model_type})",
            None, None, f"Node {self.node_id}",
        )
        try:
            selected_tiles = getattr(self.PARAMS, "tiles_to_process", None)
            if not selected_tiles:
                selected_tiles = None
            self.vl_conditioning_cache = build_vl_conditioning(
                clip=self.clip_model,
                full_image=full_image,
                grid_specs=grid_specs,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                model_type=model_type,
                vl_strength=vl_strength,
                tile_indices=selected_tiles,
            )
        finally:
            self._unload_vl_encoder("global VL encoding")
        log(
            f"VL: Global Embedding mode — {len(self.vl_conditioning_cache)} tiles sliced, "
            f"vl_strength={vl_strength:.2f} (model={model_type})",
            None, None, f"Node {self.node_id}",
        )

    def get_tile_vl_conditioning(self, tile_idx, tile_image):
        """Encode a single tile for Tile Embedding mode (on-the-fly).

        Parameters
        ----------
        tile_idx : int
            Tile index (for logging).
        tile_image : torch.Tensor
            The tile's upscaled crop (tile_to_process), shape ``(1, H, W, 3)``.

        Returns
        -------
        list or None
            Conditioning for this tile, or None if VL is not enabled/not Qwen.
        """
        from .vl_encode import encode_tile

        vl_mode = getattr(self.PARAMS, 'VL_Encoding_Mode', 'Off')
        if vl_mode not in ("Tile Embedding", "Tile + Global Embedding", "Inner Tile + Global Embedding"):
            return None

        model_type = getattr(self.KSAMPLER, 'model_type', None)
        if model_type not in ("Qwen Image", "Qwen Image Edit", "Krea2"):
            return None

        if self.clip_model is None:
            return None

        if tile_image is None or tile_image.shape[0] != 1:
            return None

        tile_vl_strength = float(getattr(self.PARAMS, "VL_Strength", 0.5))
        try:
            vl_cond = encode_tile(
                self.clip_model,
                tile_image,
                model_type,
                vl_strength=tile_vl_strength,
                layer_weights=None,
                layer_multiplier=1.0,
                reference_rebalance=False,
            )
        finally:
            self._unload_vl_encoder(f"tile {tile_idx + 1} VL encoding")
        log(
            f"VL: Tile Embedding mode — tile {tile_idx + 1} encoded "
            f"(model={model_type} strength={tile_vl_strength:.3f})",
            None, None, f"Node {self.node_id}",
        )
        return vl_cond

    # ---------------------------------------------------------- ControlNet/CN

    def _compute_single_controlnet(self, SELF, tile_idx):
        pipe = self._filtered_controlnet_pipe()
        all_pipe = getattr(self.KSAMPLER, "Controlnet_Pipe", None) or []
        tile_image = self.get_tile_image(tile_idx)
        self.krea2_control_latents[tile_idx] = prepare_krea2_depth_control(
            self, all_pipe, tile_image, self.KSAMPLER.vae
        )
        if not pipe:
            return None, None

        chosen = [str(p.get("preprocessor", "None")) for p in pipe]
        log(
            f"[TBG][ControlNet] Tile {tile_idx + 1}: selected {len(pipe)} entries for precompute -> {chosen}",
            None,
            None,
            f"Node {self.node_id}",
        )

        base = self.text_embeddings_cache.get(tile_idx)
        if base is None:
            return None, None
        pos, neg = apply_controlnets_from_pipe(
            self,
            SELF,
            pipe,
            self._nuclear_gpu(base["conditioning"]),
            self._nuclear_gpu(base["negative"]),
            self.OUTPUTS.upscaled_image,
            tile_image,
            self.KSAMPLER.vae,
            tile_idx,
        )
        return self._nuclear_cpu(pos), self._nuclear_cpu(neg)

    def combine_conditioning_atomic(self, *conditionings):
        valid = [c for c in conditionings if c is not None]
        if not valid:
            return None
        all_gpu = [self._nuclear_gpu(c) for c in valid]
        combined = []
        max_len = max(len(c) for c in all_gpu)
        for i in range(max_len):
            tensors = []
            merged = {}
            for cond in all_gpu:
                if i < len(cond) and isinstance(cond[i], list) and len(cond[i]) >= 2:
                    t, d = cond[i][0], cond[i][1]
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
                    merged.update(d)
            if tensors:
                combined_tensor = torch.cat(tensors, dim=1)
                combined.append([combined_tensor, merged])
        return self._nuclear_cpu(combined)

    # ----------------------------------------------------- full precomputation

    def _combine_all_layers_for_tile(self, SELF, tile_idx):
        base = self.text_embeddings_cache.get(tile_idx)
        if base is None:
            return
        pos = base["conditioning"]
        neg = base["negative"]

        # Redux
        if self.redux_enabled and self.redux_strength > 1e-4:
            try:

                tile_image = self.get_tile_image(tile_idx)
                redux_cond, = FluxRedux_ForTiles(
                    self.SELF,
                    conditioning=pos,
                    image=tile_image,
                )
                if redux_cond is not None:
                    pos = self._nuclear_cpu(redux_cond)
                log(
                    f"TBG[Tile {tile_idx}] Redux applied (strength={self.redux_strength})",
                    None,
                    None,
                    f"Node {self.node_id}",
                )
            except Exception as e:
                log(
                    f"TBG[Tile {tile_idx}] Redux failed → RAW ({e})",
                    None,
                    None,
                    f"Node {self.node_id}",
                )

        # ControlNet
        if getattr(self.KSAMPLER, "Controlnet_Pipe", None):
            c_pos, c_neg = self._compute_single_controlnet(SELF,tile_idx)
            if c_pos is not None:
                pos = self.combine_conditioning_atomic(pos, c_pos)
            if c_neg is not None and neg is not None:
                neg = self.combine_conditioning_atomic(neg, c_neg)

        fp = self._get_tile_fingerprint(tile_idx)
        self.text_embeddings_cache[tile_idx] = {
            "conditioning": pos,
            "negative": neg,
            "fingerprint": fp,
        }

    def precompute_text_only(self):
        log(
            "Starting VRAM Optimization (text-only cache for Ultra Low mode)",
            None,
            None,
            f"Node {self.node_id}",
        )
        self.encode_all_text_prompts()
        if self.DUALMODELmodel is not None and self.DUALMODELclip is not None and self.DUALMODELvae is not None:
            self.encode_all_text_prompts_low()
        log(
            "VRAM Optimization Complete (text-only cache)",
            None,
            None,
            f"Node {self.node_id}",
        )

    def precompute_full_layers(self, SELF):
        log("Starting VRAM Optimization (unified conditioning)", None, None, f"Node {self.node_id}")
        self.encode_all_text_prompts()  # CLIP
        if self.DUALMODELmodel is not None and self.DUALMODELclip is not None and self.DUALMODELvae is not None:
            self.encode_all_text_prompts_low()
        for tile_idx in range(len(self.grid_prompts)):
            if self.PARAMS.tiles_to_process and tile_idx not in self.PARAMS.tiles_to_process:
                continue
            self._combine_all_layers_for_tile(SELF, tile_idx)
        log("VRAM Optimization Complete (unified conditioning precomputed)", None, None, f"Node {self.node_id}")

    def precompute_fast(self, SELF):
        self.precompute_full_layers(SELF)

    def precompute_low(self, SELF):
        self.precompute_full_layers(SELF)

    def precompute_ultra(self):
        self.precompute_text_only()

    def precompute_all_embeddings(self, SELF):
        if self.vram_profile == "Fast Cache (Max Speed)":
            self.precompute_fast(SELF)
        elif self.vram_profile == "Low VRAM Cache (Unload Models)":
            self.precompute_low(SELF)
        else:
            self.precompute_ultra()

    # ----------------------------------------------------- public sampling API

    def unified_condition_to_gpu(self, tile_index, model_type="high"):
        # NEW:
        try:
            """
                   Return (positive, negative) unified conditioning on GPU.
                   No recompute here; assumes precompute_all_embeddings ran.
                   """
            if model_type == "low":
                cached = self.text_embeddings_cache_low.get(tile_index)
            else:
                cached = self.text_embeddings_cache.get(tile_index)
            if not isinstance(cached, dict):
                return None, None
            pos = cached.get("conditioning")
            neg = cached.get("negative")
            if pos is None or neg is None:
                return None, None
            return self._nuclear_gpu(pos), self._nuclear_gpu(neg)

        except AttributeError as e:
            if "'NoneType' object has no attribute 'unified_condition_to_gpu'" in str(e):
                raise RuntimeError(
                    f"TBG Node {self.INFO.id}: Cannot apply ControlNet conditioning. "
                    f"VRAM optimizer is None. Likely causes: "
                    f"1) ControlNet model failed to load, "
                    f"2) Preprocessor model is missing, "
                    f"3) VRAM optimizer crashed during init. "
                    f"Check earlier logs for 'VRAMOptimizer init failed'."
                ) from e
            else:
                raise  # Re-raise if it's a different AttributeError

    def build_tile_conditioning_streaming(self, tile_idx, model_type="high"):
        if model_type == "low":
            base = self.text_embeddings_cache_low.get(tile_idx)
        else:
            base = self.text_embeddings_cache.get(tile_idx)
        if base is None:
            return None, None

        pos = self._nuclear_gpu(base["conditioning"])
        neg = self._nuclear_gpu(base["negative"])
        log(
            f"[TBG][UltraLow] Tile {tile_idx + 1}: streaming conditioning start ({model_type})",
            None,
            None,
            f"Node {self.node_id}",
        )

        if model_type != "low":
            if self.redux_enabled and self.redux_strength > 1e-4:
                try:
                    tile_image = self.get_tile_image(tile_idx)
                    redux_cond, = FluxRedux_ForTiles(
                        self.SELF,
                        conditioning=pos,
                        image=tile_image,
                    )
                    if redux_cond is not None:
                        pos = self._nuclear_gpu(redux_cond)
                    log(
                        f"[TBG][UltraLow] Tile {tile_idx + 1}: Redux applied",
                        None,
                        None,
                        f"Node {self.node_id}",
                    )
                    self._ultra_stage_unload("redux")
                except Exception as e:
                    log(
                        f"[TBG][UltraLow] Tile {tile_idx + 1}: Redux failed ({e})",
                        None,
                        None,
                        f"Node {self.node_id}",
                    )

            pipe = self._filtered_controlnet_pipe()
            if pipe:
                tile_image = self.get_tile_image(tile_idx)
                pos, neg = apply_controlnets_from_pipe(
                    self,
                    self.SELF,
                    pipe,
                    pos,
                    neg,
                    self.OUTPUTS.upscaled_image,
                    tile_image,
                    self.KSAMPLER.vae,
                    tile_idx,
                )
                pos = self._nuclear_gpu(pos)
                neg = self._nuclear_gpu(neg)
                log(
                    f"[TBG][UltraLow] Tile {tile_idx + 1}: ControlNet applied",
                    None,
                    None,
                    f"Node {self.node_id}",
                )
                self._ultra_stage_unload("controlnet")

        log(
            f"[TBG][UltraLow] Tile {tile_idx + 1}: streaming conditioning done ({model_type})",
            None,
            None,
            f"Node {self.node_id}",
        )
        return pos, neg





    # --------------------------------------------------------------- cleanup

    def cleanup(self):
        self.text_embeddings_cache.clear()
        self.text_embeddings_cache_low.clear()
        self.vl_conditioning_cache.clear()
        self.last_timestamp = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(
            f"TBG[Node {self.node_id}] COMPLETE CACHE CLEANUP - "
            f"{torch.cuda.memory_allocated() / 1e9:.1f}GB freed",
            None,
            None,
            f"Node {self.node_id}",
        )
