import numpy as np
from PIL import Image

# Dimensions that the images will be resized to for analysis
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 250


def read_image(path: str) -> np.ndarray:
    """Reads the image at the path provided, resizes it to IMAGE_WIDTHxIMAGE_HEIGHT, converts it to greyscale and turns it into a flattened ndarray

    :param path: path of the image
    :type path: str
    :return: flattened ndarray where each element contains info on a pixel of the original image
    :rtype: numpy.ndarray
    """
    # Reads image at path
    image = Image.open(path)
    # Resizes it
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Converts to greyscale
    image = image.convert("L")
    # Converts to ndarray
    image = np.array(image)
    # Converts to an 1D array and normalizes it
    image = image.flatten() / 255
    return image