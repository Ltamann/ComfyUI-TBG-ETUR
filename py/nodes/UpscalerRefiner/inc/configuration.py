#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ....utils.constants import get_category

class Configuration:
    OUTPUT_NODE = False
    CATEGORY = get_category("TBG")

    @classmethod
    def generate_entries(self, input_names, input_types, code = 'py'):
        entries = {}
        for name, type_ in zip(input_names, input_types):
            if code == 'js':
                entries[name] = type_
            else:
                entries[name] = (type_, {"multiline": True})
        return entries
