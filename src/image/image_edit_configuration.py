

class ImageEditConfiguration:
    def __init__(self) -> None:
        self.random_rotation_chance = 0.5
        self.random_flip_chance = 0.5
        self.random_draw_chance = 0.5
        self.random_crop_chance = 0.5
        self.random_shear_chance = 0.3

        self.random_color_rotation_chance = 0.2
        self.random_scale_chance = 0.3
        self.random_noise_chance = 0.7
        self.random_contrast_chance = 0.2
        self.random_brightness_chance = 0.1
        self.random_saturation_chance = 0.4
        self.random_hue_chance = 0.5
        self.random_blur_chance = 0.3
        self.random_sharpen_chance = 0.7
        self.random_invert_chance = 0.2
        self.random_sepia_chance = 0.1
        self.random_grayscale_chance = 0.5
        self.random_pixelate_chance = 0.3
        
        # Texture-based drawing configuration
        self.texture_draw_probability = 0.6  # Probability of using texture-based drawing vs solid colors

    def set_from_dict(self, config: dict):
        self.__dict__.update(config)
