#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# from ComfyUI/custom_nodes/TBG_upscaler/py/vendor/ComfyUI_MaraScott_Nodes check licence here
###

from .configuration import Configuration as _CONF

class Node:
    
    INPUT_QTY = 254

    INPUT_NAMES = tuple("tile {:02d}".format(i) for i in range(0, INPUT_QTY))
    
    INPUT_TYPES = ("STRING",) * len(INPUT_NAMES)

    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')    