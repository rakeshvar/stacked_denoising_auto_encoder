import numpy as np
from math import ceil
from scipy.ndimage.interpolation import zoom as zoom_fn

def normalize(img, make_white):
    maxx, minn = img.max(), img.min()
    img -= minn
    img /= maxx - minn
    if make_white and np.mean(img) < .5:
        img = 1 - img
    return img


def tile_raster_images(images,
                       zoom=1,
                       margin=1,
                       make_white=True,
                       global_normalize=True):
    """

    :param images: A 3D stack of images
    :param zoom: Amount of zoom to be applied to each image
    :param margin: Width of margin between tiled images
    :param make_white: Invert image so that it may look whiter
    :param global_normalize: Apply normalization over the entire image-stack
    :return: tiled image
    """
    n_images = images.shape[0]
    im_per_row = n_images // int(np.sqrt(n_images))
    im_per_col = ceil(float(n_images) / im_per_row)
    h, w = images.shape[1], images.shape[2]

    out_shape = ((h*zoom + margin) * im_per_col - margin,
                 (w*zoom + margin) * im_per_row - margin)
    out_array = np.full(out_shape, 255 if make_white else 0, dtype='uint8')

    if global_normalize:
        images = normalize(images, make_white)

    for i in range(n_images):
        img = images[i]
        if not global_normalize:
            img = normalize(img, make_white)

        img = zoom_fn(img, zoom=zoom, order=0)
        tile_row, tile_col = i // im_per_row, i % im_per_row
        out_array[
        tile_row * (h*zoom + margin): tile_row * (h*zoom + margin) + h*zoom,
        tile_col * (w*zoom + margin): tile_col * (w*zoom + margin) + w*zoom
        ] = 255 * img

    return out_array
