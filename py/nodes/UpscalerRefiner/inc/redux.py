import nodes

def FluxRedux_ForTiles(self, conditioning, image):
    clip_vision = self.PARAMS.Redux_Clip_Vision
    style_model = self.PARAMS.Redux_Style_Model
    Redux_strength = min(self.PARAMS.Redux_strength,1)
    clip_vision_output, = nodes.CLIPVisionEncode.encode(self, self.PARAMS.Redux_Clip_Vision, image, "none")
    IPconditioning, = nodes.StyleModelApply.apply_stylemodel(self, conditioning,  self.PARAMS.Redux_Style_Model, clip_vision_output, Redux_strength, "multiply")
    return (IPconditioning,)



