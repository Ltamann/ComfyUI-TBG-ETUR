#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from .configuration import Configuration as _CONF
class Node:
    # Single JSON field that carries all arrays from JS -> Python
    CONFIG_KEYS = ("tile_edits_json",)
    CONFIG_TYPES = ("STRING",)
    ENTRIES_CONFIG = _CONF.generate_entries(CONFIG_KEYS, CONFIG_TYPES)
