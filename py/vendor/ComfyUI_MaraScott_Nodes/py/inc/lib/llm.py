from ...utils.log import log

class MS_Llm():


    @classmethod
    def generate_tile_prompt(self, image, prompt_context, seed=None):
        prompt_tile = self.vision_llm.generate_prompt(image)
        _prompt = self.get_grok_prompt(prompt_context, prompt_tile)
        prompt = _prompt
        log(prompt, None, None, self.vision_llm.name)
        return prompt

        
    @classmethod
    def get_grok_prompt(self, prompt_context, prompt_tile):
        prompt = [
            f"tile_prompt: \"{prompt_tile}\".",
            f"full_image_prompt: \"{prompt_context}\".",
            "tile_prompt is part of full_image_prompt.",
            "If tile_prompt is describing something different than the full image, correct tile_prompt to match full_image_prompt.",
            "if you don't need to change the tile_prompt return the tile_prompt.",
            "your answer will strictly and only return the tile_prompt string without any decoration like markdown syntax."
        ]
        return " ".join(prompt)
    

        return completion.choices[0].message.content
