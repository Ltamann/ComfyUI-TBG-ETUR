import torch


def mix(latent_image, noise_image, mix_percent):
    return ((1 - mix_percent) * latent_image + mix_percent *
                          noise_image) / ((mix_percent**2 + (1-mix_percent)**2) ** 0.5)

            


class FluxNoiseMixerNode:


    def append(self, latent, noise, mix_percent, random_noise, mix_type, random_mix_type, take_diff):
        latent_image = latent.copy()
        noise = noise['samples']
        latent = latent_image['samples']

        random_noise_latent = torch.randn_like(noise)
        if random_mix_type == 'mix':
            noise = mix(noise, random_noise_latent, random_noise)
            # noise = (noise * (1-random_noise) + random_noise_latent * (random_noise))
        elif random_mix_type == 'add':
            noise += random_noise_latent * random_noise

        if mix_type == 'mix':
            new_latent = mix(latent, noise, mix_percent)
            # new_latent = (latent * (1-mix_percent) + noise * (mix_percent))
        elif mix_type == 'add':
            new_latent = latent + noise * mix_percent

        if take_diff:
            new_latent = new_latent - latent * mix_percent
        latent_image['samples'] = new_latent
        return (latent_image, )