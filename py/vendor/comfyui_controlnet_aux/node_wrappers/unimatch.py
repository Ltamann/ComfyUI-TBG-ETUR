import comfy.model_management as model_management
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class Unimatch_OptFlowPreprocessor:

    def estimate(self, image, ckpt_name, backward_flow=False, bidirectional_flow=False):
        assert len(image) > 1, "[Unimatch] Requiring as least two frames as an optical flow estimator. Only use this node on video input."    
        from ..src.custom_controlnet_aux.unimatch import UnimatchDetector
        tensor_images = image
        model = UnimatchDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        flows, vis_flows = [], []
        for i in range(len(tensor_images) - 1):
            image0, image1 = np.asarray(image[i:i+2].cpu() * 255., dtype=np.uint8)
            flow, vis_flow = model(image0, image1, output_type="np", pred_bwd_flow=backward_flow, pred_bidir_flow=bidirectional_flow)
            flows.append(torch.from_numpy(flow).float())
            vis_flows.append(torch.from_numpy(vis_flow).float() / 255.)
        del model
        return (torch.stack(flows, dim=0), torch.stack(vis_flows, dim=0))

class MaskOptFlow:

    def mask_opt_flow(self, optical_flow, mask):
        from ..src.custom_controlnet_aux.unimatch import flow_to_image
        assert len(mask) >= len(optical_flow), f"Not enough masks to mask optical flow: {len(mask)} vs {len(optical_flow)}"
        mask = mask[:optical_flow.shape[0]]
        mask = F.interpolate(mask, optical_flow.shape[1:3])
        mask = rearrange(mask, "n 1 h w -> n h w 1")
        vis_flows = torch.stack([torch.from_numpy(flow_to_image(flow)).float() / 255. for flow in optical_flow.numpy()], dim=0)
        vis_flows *= mask
        optical_flow *= mask
        return (optical_flow, vis_flows)

        
