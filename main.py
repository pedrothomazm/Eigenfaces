from typing import Iterable
import numpy as np
import glob
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


def read_dataset(path: str, extension: str) -> Iterable[np.ndarray]:
    """Reads all images of the specified format in the specified folder

    :param path: Path to the folder
    :type path: str
    :param extension: Extension of the images being read
    :type extension: str
    :return: Iterable containing the images converted to ndarrays using read_image
    :rtype: Iterable[numpy.ndarray]
    """
    pathname_pattern = f"{path}/*.{extension}"
    path_list = glob.glob(pathname_pattern)
    map_iter = map(read_image, path_list)
    return map_iter