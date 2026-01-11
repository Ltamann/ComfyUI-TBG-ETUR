from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT


class Shuffle_Preprocessor:

    def preprocess(self, image, resolution=512, seed=0):
        from ..src.custom_controlnet_aux.shuffle import ContentShuffleDetector

        return (common_annotator_call(ContentShuffleDetector(), image, resolution=resolution, seed=seed), )

