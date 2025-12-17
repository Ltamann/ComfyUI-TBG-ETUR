import comfy_extras
from ....vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch

# to do add a color correction to the fusion process

def fill_missing_tiles_from_background_rebuild(self, background, output_images):
        # Build Tiles from Refine_Alternative_Image
    images = self.OUTPUTS.grid_images_all
    for index, output_image in enumerate(output_images):
        if output_image is not None:
            if self.PARAMS.color_match_method != 'none':
                images[index] = ColorMatch().colormatch(self.OUTPUTS.grid_images_all[index], output_images[index], self.PARAMS.color_match_method)[0]
            else:      
                images[index] =  output_images[index]
    return images


def crop(image, width, height, x, y):
    x = min(x, image.shape[2] - 1)
    y = min(y, image.shape[1] - 1)
    to_x = width + x
    to_y = height + y
    img = image[:, y:to_y, x:to_x, :]
    return (img,)


def fill_missing_tiles_from_background(self, background, output_images):
            # Build Tiles from Refine_Alternative_Image
    Alternative_Images=[]
    grid_spec_tile = [item for item in self.PARAMS.grid_specs if item[2] < 9000]
    grid_spec_custom = [item for item in self.PARAMS.grid_specs if item[2] >= 9000]
    # for Tiles
    for index, grid_spec in enumerate(grid_spec_tile):
        row, col, order, x_start, y_start, width_inc, height_inc = grid_spec
        Alternative_Image = crop(background, width_inc, height_inc, x_start, y_start)[0]
        Alternative_Images.append(Alternative_Image)
    return Alternative_Images

