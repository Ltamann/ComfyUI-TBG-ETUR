"""PiD decode policy and normalized output boundary."""

from .refiner_pipeline import normalize_rgb_contract


class PIDPipeline:
    """Small PiD boundary used by the refiner until the decode body is moved here."""

    @staticmethod
    def active(ksampler):
        return bool(getattr(ksampler, "pid_vae_decode", False)) and (
            str(getattr(ksampler, "vae_encode_type", "") or "") == "Nvidia PiD 4x"
        )

    @staticmethod
    def source_type(ksampler, segment_index):
        if not PIDPipeline.active(ksampler):
            return "tiled_vae" if bool(getattr(ksampler, "tiled", False)) else "vae"
        return "segment_pid" if segment_index is not None else "pid"

    @staticmethod
    def normalize_output(image, tile_index, segment_index, **kwargs):
        return normalize_rgb_contract(
            image,
            tile_index=tile_index,
            segment_index=segment_index,
            source_type=kwargs.pop("source_type", "pid"),
            coordinate_space=kwargs.pop("coordinate_space", "pid_tile_4x"),
            **kwargs,
        )
