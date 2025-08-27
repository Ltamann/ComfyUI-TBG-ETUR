import torch


class RegionalStyleModelApplyNode:


    def apply_stylemodel(self, clip_vision_output, style_model, strength, conditioning=None):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0) * strength
        c = []
        if conditioning is not None:
            for t in conditioning:
                n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
                c.append(n)
        else:
            c.append([cond, None])
        return (c, )
