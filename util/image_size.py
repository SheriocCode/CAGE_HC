import numpy as np


def image_size_to_hw(image_size):
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a (height, width) pair")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def coord_scale_values(image_size, coord_dim=2, dtype=np.float32):
    if coord_dim % 2 != 0:
        raise ValueError("coord_dim must contain x/y pairs")

    height, width = image_size_to_hw(image_size)
    xy_scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=dtype)
    return np.tile(xy_scale, coord_dim // 2)
