import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from skimage.draw import polygon2mask  # conda install scikit-image


def color_2_coordinates(img, color):
    """
    function to get the (x,y) positions of pixel with custom image. (pixels of the corners of our polygon)
    :param img: image with RGB pixels
    :param color: RBG color [R, B, G] of pixels to find
    :return: list of (x, y) coordinates of pixels with the color values
    """
    arr = np.array(img)
    coordinates = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if arr[x, y].tolist() == color:
                coordinates.append([x, y])
    # fixing the order of coordinates to form a polygon
    try:
        coordinates[-1], coordinates[-2] = coordinates[-2], coordinates[-1]
    except ValueError:
        print("Empty list")

    return coordinates


def corners_2_mask(img, coordinates):
    """
    create a mask from corner coordinates using :
    skimage.draw.polygon2mask(image_shape, polygon) : directly returns a bool-type numpy.array where True means the point is inside the polygon.
    :param coordinates: list of (x, y) coordinates corners
    :return: boolean array with same size of the image where the pixels inside the polygon have True values.
    """
    image_shape = img.size
    polygon = np.array(coordinates)
    mask = polygon2mask(image_shape, polygon)

    return mask


def mask_2_coordinates(mask):
    """
    Get (x,y) position of the pixels inside the mask
    :param mask: boolean array where the pixels inside the polygon have True values
    :return: list of coordinates of pixels inside the mask
    """
    pixels = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == True:
                pixels.append([x, y])

    return pixels


if __name__ == '__main__':
    # open the image
    image = PIL.Image.open('capture.png').convert('RGB')
    # my corners pixels have red color [237, 28, 36]
    red = [237, 28, 36]
    # Detect position of corners
    corner = color_2_coordinates(image, red)
    # convert corners to mask
    mask = corners_2_mask(image, corner)
    # Get position of pixels inside the mask
    my_pixels = mask_2_coordinates(mask)
    print(my_pixels)

    # testing : change the color of pixels founded in a copy of our image
    arr2 = np.array(image).copy()
    arr2[mask] = [237, 28, 36]

    plt.imshow(arr2)
    plt.show()
