class AddFluxFlowNode:

    def patch(self, model, flow):
        m = model.clone()
        model_options = {**model.model_options}
        model.model_options = model_options
        transformer_options = {**model.model_options.get('transformer_options', {})}
        model.model_options['transformer_options'] = transformer_options

        transformer_options['FLOW'] = flow

        return (m, )

