
default_attn = {
    'inputs': [True] * 10,
    'input_idxs': list(range(10)),
    'middle_0': True,
    'outputs': [True] * 12,
    'output_idxs': list(range(12))
}


class ApplyFluxRaveAttentionNode:

    def apply(self, model, grid_size, seed, attn_override=default_attn):
        model = model.clone()

        transformer_options = {**model.model_options.get('transformer_options', {})}
        model.model_options['transformer_options'] = transformer_options

        transformer_options['RAVE'] = {
            "grid_size": grid_size,
            "seed": seed,
        }

        return (model, )