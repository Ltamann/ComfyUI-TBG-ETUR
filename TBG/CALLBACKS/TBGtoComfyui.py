from py.vendor.ComfyUI_Unload_Models_main.py.unload_one_model import UnloadOneModelNode
from py.vendor.ComfyUI_QwenVL.AILab_QwenVL import QwenVLBase
from py.vendor.ComfyUI_QwenVL.nodes import Qwen2VL_TBG
from py.vendor.ComfyUI_Florence2.nodes import DownloadAndLoadFlorence2Model,Florence2Run
from py.vendor.seedvr2_videoupscaler.src.interfaces.video_upscaler import TBG_SeedVR2VideoUpscaler
from py.vendor.flashvsr_ultra_fast.nodes import refine_tile
from py.vendor.ComfyUI_Impact_Pack.masktoseg import MaskToSEGS, combine_segs
from TBG.CALLBACKS.constants import tbg
from TBG.SERVERS.COMFYUI_server import register_main_class

@register_main_class
class LLM:
    @staticmethod
    def get_prompts():
        if not tbg.LLM.model == "NONE":
            if "Sky" in tbg.LLM.model.lower():
                qwen = Qwen2VL_TBG()
            if "Qwen" in tbg.LLM.model.lower():
                QwenVL = QwenVLBase()
            if "florence" in tbg.LLM.model.lower():
                florence_loader = DownloadAndLoadFlorence2Model()
                florence_run = Florence2Run()
                if "large" in tbg.LLM.model.lower():
                    florence_task = "prompt_gen_mixed_caption"
                else:
                    florence_task = "detailed_caption"
