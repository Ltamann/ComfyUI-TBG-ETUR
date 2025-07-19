# # from ComfyUI/custom_nodes/TBG_upscaler/py/vendor/ComfyUI_MaraScott_Nodes check licence here
import re

APIURL ='https://api.ylab.es/'
#APIURL ='http://localhost:8080/'
NAMESPACE='TBG'
ICON=''



def get_apiurl(classOject):
    return APIURL

def _get_version(classOject):
    version_match = re.search(r'_v(\d+)', classOject.__name__)
    version = version_match.group(1) if version_match else None
    return version

def get_name(classOject, name, shortcut = "", vendor = ""):
    
    version = _get_version(classOject)
    v = f" - V{version}" if version is not None else ""
    s = f" /{shortcut}" if shortcut != "" else ""
    vd = f" (from {vendor})" if vendor != "" else ""
    
    return '{} {}{}{}{}'.format(ICON, name, v, s, vd)

def get_category(sub_dirs = None):
    if sub_dirs is None:
        return NAMESPACE
    else:
        return "{}/{}".format(f"{ICON} {NAMESPACE}", sub_dirs)
