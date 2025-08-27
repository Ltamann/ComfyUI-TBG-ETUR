#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .configuration import Configuration as _CONF

class Node:
    #INPUT_QTY = 1

    # Original UI scaffolding keys (e.g., "tile 00", "tile 01", ...)
    #INPUT_NAMES = tuple(f"tile {i:02d}" for i in range(INPUT_QTY))
    #INPUT_TYPES = ("STRING",) * len(INPUT_NAMES)

    # For Python-side inputs (used by Comfy)
    #ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)

    # For JS-side (raw type, no options)
    #ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')

    # New hidden keys that mirror node.properties set by the UI (you can keep these if needed)
    #PROMPT_KEYS = tuple(f"prompt_{i}" for i in range(INPUT_QTY))
    #DENOISE_KEYS = tuple(f"denoise_{i}" for i in range(INPUT_QTY))
    #SEED_KEYS = tuple(f"seed_{i}" for i in range(INPUT_QTY))
    #CNET_KEYS = tuple(f"cnet_strength_{i}" for i in range(INPUT_QTY))

    #PROMPT_TYPES = ("STRING",) * INPUT_QTY
    #DENOISE_TYPES = ("STRING",) * INPUT_QTY
    #SEED_TYPES = ("STRING",) * INPUT_QTY
    #CNET_TYPES = ("STRING",) * INPUT_QTY

    #ENTRIES_PROMPT = _CONF.generate_entries(PROMPT_KEYS, PROMPT_TYPES)
    #ENTRIES_DENOISE = _CONF.generate_entries(DENOISE_KEYS, DENOISE_TYPES)
    #ENTRIES_SEED = _CONF.generate_entries(SEED_KEYS, SEED_TYPES)
    #ENTRIES_CNET = _CONF.generate_entries(CNET_KEYS, CNET_TYPES)

    # Single JSON field that carries all arrays from JS -> Python
    CONFIG_KEYS = ("tile_edits_json",)
    CONFIG_TYPES = ("STRING",)
    ENTRIES_CONFIG = _CONF.generate_entries(CONFIG_KEYS, CONFIG_TYPES)
