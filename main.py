from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
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


def visualize_image(array: np.ndarray, name: str):
    """Shows the image represented by the array

    :param array: array of size IMAGE_WIDTH*IMAGE_HEIGHT
    :type array: numpy.ndarray
    """
    resized = np.resize(array, (IMAGE_HEIGHT, IMAGE_WIDTH))
    plot = plt.imshow(resized, cmap="gray")
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    plt.show


if __name__ == "__main__":
    faces_matrix = np.row_stack(tuple(read_dataset("data", "jpg")))
    avg_face = np.mean(faces_matrix, axis=0)
    diff_matrix = faces_matrix - avg_face
    # Using full_matrices=False to avoid doing unnecessary calculations
    _, _, eigenfaces = np.linalg.svd(diff_matrix, full_matrices=False)
    visualize_image(avg_face, "average_face")
    visualize_image(eigenfaces[0], "eigenface_1")