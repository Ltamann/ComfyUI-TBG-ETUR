import torch
from comfy.samplers import KSAMPLER
from tqdm import trange
from ..utils.attn_bank import AttentionBank
from ..utils.const import DEFAULT_DOUBLE_LAYERS, DEFAULT_SINGLE_LAYERS


def get_sample_forward(attn_bank, save_steps, single_layers, double_layers, order="second"):
    @torch.no_grad()
    def sample_forward(model, x, sigmas, extra_args=None, callback=None, disable=None):
        attn_bank.clear()
        attn_bank['save_steps'] = save_steps

        extra_args = {} if extra_args is None else extra_args

        model_options = extra_args.get('model_options', {})
        model_options = {**model_options}
        transformer_options = model_options.get('transformer_options', {})
        transformer_options = {**transformer_options}
        model_options['transformer_options'] = transformer_options
        extra_args['model_options'] = model_options
        prev_pred = None
        N = len(sigmas)-1
        s_in = x.new_ones([x.shape[0]])
        for i in trange(N, disable=disable):
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            if N-i-1 < save_steps:
                attn_bank[N-i-1] = {
                    'first': {},
                    'mid': {}
                }

            transformer_options['rfedit'] = {
                'step': N-i-1,
                'process': 'forward' if N-i-1 < save_steps else None,
                'pred': 'first',
                'bank': attn_bank,
                'single_layers': single_layers,
                'double_layers': double_layers,
            }

            if order == 'fireflow' and prev_pred is not None:
                pred = prev_pred
            else:
                pred = model(x, s_in * sigma, **extra_args)

            transformer_options['rfedit'] = {
                'step': N-i-1,
                'process': 'forward' if N-i-1 < save_steps else None,
                'pred': 'mid',
                'bank': attn_bank,
                'single_layers': single_layers,
                'double_layers': double_layers,
            }
            
            img_mid = x + (sigma_next- sigma) / 2 * pred
            sigma_mid = (sigma + (sigma_next - sigma) / 2)
            pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)
            if order == 'fireflow':
                prev_pred = pred_mid
                x = x + (sigma_next - sigma) * pred_mid
            else:
                first_order = (pred_mid - pred) / ((sigma_next - sigma) / 2)
                x = x + (sigma_next - sigma) * pred + 0.5 * (sigma_next - sigma) ** 2 * first_order

            if callback is not None:
                callback({'x': x, 'denoised': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return x

    return sample_forward

def get_sample_reverse(attn_bank, inject_steps, single_layers, double_layers, order="second"):
    @torch.no_grad()
    def sample_reverse(model, x, sigmas, extra_args=None, callback=None, disable=None):
        if inject_steps > attn_bank['save_steps']:
            raise ValueError(f'You must save at least as many steps as you want to inject. save_steps: {attn_bank["save_steps"]}, inject_steps: {inject_steps}')
        
        extra_args = {} if extra_args is None else extra_args

        model_options = extra_args.get('model_options', {})
        model_options = {**model_options}
        transformer_options = model_options.get('transformer_options', {})
        transformer_options = {**transformer_options}
        model_options['transformer_options'] = transformer_options
        extra_args['model_options'] = model_options

        prev_pred = None
        N = len(sigmas)-1
        s_in = x.new_ones([x.shape[0]])
        for i in trange(N, disable=disable):
            sigma = sigmas[i]
            sigma_prev = sigmas[i+1]

            transformer_options['rfedit'] = {
                'step': i,
                'process': 'reverse' if i < inject_steps else None,
                'pred': 'first',
                'bank': attn_bank,
                'single_layers': single_layers,
                'double_layers': double_layers,
            }

            if order == "fireflow" and prev_pred is not None:
                pred = prev_pred
            else:
                pred = model(x, s_in * sigma, **extra_args)

            transformer_options['rfedit'] = {
                'step': i,
                'process': 'reverse' if i < inject_steps else None,
                'pred': 'mid',
                'bank': attn_bank,
                'single_layers': single_layers,
                'double_layers': double_layers,
            }
            
            img_mid = x + (sigma_prev- sigma) / 2 * pred
            sigma_mid = (sigma + (sigma_prev - sigma) / 2)
            pred_mid = model(img_mid, s_in * sigma_mid, **extra_args)
            if order == "fireflow":
                prev_pred = pred_mid
                x = x + (sigma_prev - sigma) * pred_mid
            else:
                first_order = (pred_mid - pred) / ((sigma_prev - sigma) / 2)
                x = x + (sigma_prev - sigma) * pred + 0.5 * (sigma_prev - sigma) ** 2 * first_order

            if callback is not None:
                callback({'x': x, 'denoised': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return x

    return sample_reverse


class FlowEditForwardSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "save_steps": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
        },           
            "optional": {
                "single_layers": ("SINGLE_LAYERS",),
                "double_layers": ("DOUBLE_LAYERS",),
                "order": (["second", "fireflow",],),
        }}
    RETURN_TYPES = ("SAMPLER","ATTN_INJ")
    FUNCTION = "build"

    CATEGORY = "fluxtapoz"

    def build(self, save_steps, single_layers=DEFAULT_SINGLE_LAYERS, double_layers=DEFAULT_DOUBLE_LAYERS, order="second"):
        attn_bank = AttentionBank()
        sampler = KSAMPLER(get_sample_forward(attn_bank, save_steps, single_layers, double_layers, order))

        return (sampler, attn_bank)


class FlowEditReverseSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "attn_inj": ("ATTN_INJ",),
            "inject_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
        },           
            "optional": {
                "single_layers": ("SINGLE_LAYERS",),
                "double_layers": ("DOUBLE_LAYERS",),
                "order": (["second", "fireflow",],),
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "fluxtapoz"

    def build(self, attn_inj, inject_steps, single_layers=DEFAULT_SINGLE_LAYERS, double_layers=DEFAULT_DOUBLE_LAYERS, order="second"):
        sampler = KSAMPLER(get_sample_reverse(attn_inj, inject_steps, single_layers, double_layers, order))
        return (sampler, )


class PrepareAttnBankNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "latent": ("LATENT",),
            "attn_inj": ("ATTN_INJ",),
        }
        }

    RETURN_TYPES = ("LATENT", "ATTN_INJ")
    FUNCTION = "prepare"

    CATEGORY = "fluxtapoz"

    def prepare(self, latent, attn_inj):
        # Hack to force order of operations in ComfyUI graph
        return (latent, attn_inj)

