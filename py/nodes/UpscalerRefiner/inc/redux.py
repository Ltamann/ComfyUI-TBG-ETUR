import nodes

def FluxRedux_ForTiles(self, conditioning, image):
    clip_vision = self.PARAMS.Redux_Clip_Vision
    style_model = self.PARAMS.Redux_Style_Model
    Redux_strength = self.PARAMS.Redux_strength
    clip_vision_output, = nodes.CLIPVisionEncode.encode(self, clip_vision, image, "none")
    IPconditioning, = nodes.StyleModelApply.apply_stylemodel(self, conditioning, style_model, clip_vision_output, Redux_strength, "multiply")
    return (IPconditioning,)

    result = { "ui": { "a_images":[], "b_images": [] } }
    if image_a is not None and len(image_a) > 0:
      result['ui']['a_images'] =  self.save_images(self, image_a, filename_prefix="tbg", prompt=None, extra_pnginfo=None)['ui']['images']

    if image_b is not None and len(image_b) > 0:
      result['ui']['b_images'] =  self.save_images(self, image_a, filename_prefix="tbg", prompt=None, extra_pnginfo=None)['ui']['images']

    return result

