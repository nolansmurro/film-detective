from PIL import Image

def crop_resize(image, size):
    target_ratio = size[0] / size[1]
    image_ratio = image.width / image.height

    if image_ratio > target_ratio:
        new_width = int(target_ratio * image.height)
        left = (image.width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = image.height
    else:
        new_height = int(image.width / target_ratio)
        left = 0
        top = (image.height - new_height) / 2
        right = image.width
        bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    image = image.resize(size, Image.LANCZOS)
    
    return image