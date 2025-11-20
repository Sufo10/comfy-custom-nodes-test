import node_helpers
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import torch
import os

class LoadImageCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_path": ("STRING", {"default": "", "tooltip": "Path to the image"})},
                }
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, image_path):
        image_path = os.path.abspath(image_path)
        img = node_helpers.pillow(Image.open, image_path)
        output_images = []
        w, h = None, None
        excluded_formats = ['MPO']
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            if image.size[0] != w or image.size[1] != h:
                continue
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]
        return (output_image,)
