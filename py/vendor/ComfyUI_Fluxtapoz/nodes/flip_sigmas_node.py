
class InFluxFlipSigmasNode:

    def flip(self, sigmas):
        sigmas = sigmas.flip()
        if sigmas[0] == 0:
            sigmas[0] = 1e-3
        return (sigmas, )
