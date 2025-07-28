# TBG Preset definitions
# --------------------------------------------------
# You can create your own presets by editing this file.
import math

# For TBG Tiled Upscaler CE
PRESETS_CE = [
    'NONE',
    'Soft Merge',
    'Add Yours → TBG_presets.py',
]

# For TBG Tiled Upscaler PRO
PRESETS_PRO = [
    'NONE',
    'Soft Merge',
    'PRO_Neuro_Generative_Tile_Fusion',
    'PRO_Neuro_Generative_Tile_Fusion + Soft Merge',
    'Full size Image no Tiles',
    'Add Yours → TBG_presets.py',
]

def get_presets(min_tile_size, **kwargs):

        # minimum_unit is calculated as a value dividable by 8 depending on tile size
        # ( tile_size 1024 minimum_unit 16)
        # ( tile_size 512 minimum_unit 8)

        minimum_unit = max(8, ((math.floor(min_tile_size / 64)) // 8 * 8))  # 16
        minimum_unit = min(minimum_unit, 8)
        minimum_unit = max(minimum_unit, 32)

        if kwargs["presets"] == 'Add Yours → TBG_presets.py':
            kwargs["presets"] = 'None'

        if kwargs["presets"] == 'Soft Merge':
            kwargs["presets"] = 'Classic Tile Seams'
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = False
            kwargs["PRO_Tile_Fusion_Mode"] = 'NONE'
            kwargs["PRO_Tile_Fusion_blur_margin"] = 0
            kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = 0
            kwargs["PRO_Tile_Fusion_border_margin"] = 0
            kwargs["compositing_mask_blur"] = minimum_unit * 2

        elif kwargs["presets"] == 'PRO_Neuro_Generative_Tile_Fusion':
            kwargs["presets"] = 'Generative Tile Fusion'
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
            kwargs["PRO_Tile_Fusion_Mode"] = 'Neuro_Generative_Tile_Fusion'
            kwargs["compositing_mask_blur"] = 0
            kwargs["PRO_Tile_Fusion_blur_margin"] = minimum_unit * 4
            kwargs["PRO_Tile_Fusion_shift_in_out"] =  0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = 0
            kwargs["PRO_Tile_Fusion_border_margin"] =minimum_unit * 4


        elif kwargs["presets"] == 'PRO_Neuro_Generative_Tile_Fusion + Soft Merge':
            kwargs["presets"] = 'Classic Tile Seams + Generative Tile Fusion'
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
            kwargs["PRO_Tile_Fusion_Mode"] = 'Neuro_Generative_Tile_Fusion'
            kwargs["compositing_mask_blur"] = minimum_unit * 2
            kwargs["PRO_Tile_Fusion_blur_margin"] = minimum_unit * 4
            kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = 0
            kwargs["PRO_Tile_Fusion_border_margin"] = minimum_unit * 4


        elif kwargs["presets"] == 'PRO_Tile_Fusion':
            kwargs["presets"] = 'Original Tile Fusion'
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
            kwargs["PRO_Tile_Fusion_Mode"] = 'Tile_Fusion'
            kwargs["compositing_mask_blur"] = 0
            kwargs["PRO_Tile_Fusion_blur_margin"] = minimum_unit * 3
            kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = minimum_unit * 6
            kwargs["PRO_Tile_Fusion_border_margin"] =minimum_unit * 2


        elif kwargs["presets"] == 'PRO_Tile_Fusion + Soft Merge':
            kwargs["presets"] = 'Original Tile Fusion + Soft Merge'
            kwargs["PRO_Neuro_Generative_Tile_Fusion"] = True
            kwargs["PRO_Tile_Fusion_Mode"] = 'Tile_Fusion'
            kwargs["compositing_mask_blur"] = minimum_unit * 2
            kwargs["PRO_Tile_Fusion_blur_margin"] = minimum_unit * 3
            kwargs["PRO_Tile_Fusion_shift_in_out"] = 0
            kwargs["PRO_Tile_Fusion_shift_top_left"] = minimum_unit * 6
            kwargs["PRO_Tile_Fusion_border_margin"] = minimum_unit * 2


        return (kwargs)