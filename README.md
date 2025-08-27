# This is only a test version we are working on right now 
The tile promter got a major update
- new UI 
- placeholder now hidden
- more settings per tile we add prompt,denoise,seed,cnet-strength 
    now you can test 1 tile in preview mode and if you like the result add the seed, denoise and cnet-strength and prompt to the tile
    i was missing the seed function for the segments - now it's like a perfect mix of tiled upscaling and inpainting.
- input values like prompt,denoise,seed,and,cnet-strength are saved in workflow json, and you can reload the browser without loosing the inputs, visible after first run as tiles are recreated.
- smaler tile preview

- New VL  Qwen 2.5 VL + Skycaptioner V1 support
- New seed selected in Refiner: copy last used seed  , the seed shown if random is for the next generation not the used one.
- New helper nodes 
- New Qwen Image Edit support
- Nunchaku support 
- Flux Krea support 

- Bugfix edge cases 1 row, 1 col, 1 tile for tile fusion