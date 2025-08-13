#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# from ComfyUI/custom_nodes/TBG_upscaler/py/vendor/ComfyUI_MaraScott_Nodes check licence here
###
from ....utils.constants import get_category
class Configuration:
    
    OUTPUT_NODE = False
    CATEGORY = get_category("TBG")
    
    @classmethod
    def generate_entries(self, input_names, input_types, code = 'py'):
        entries = {}
        for name, type_ in zip(input_names, input_types):
            # Handle special cases where additional parameters are needed
            if code == 'js':
                entries[name] = type_
            else:
                entries[name] = (type_, {"multiline": True})
        return entries
    
