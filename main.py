import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask #conda install scikit-image
import cv2


def color2coordinates(img, color):
    """
    function to get the (x,y) positions of pixel with custom image. (pixels of the corners of our polygon)
    :param img: image with RGB pixels
    :param color: RBG color [R, B, G] of pixels to find
    :return: list of (x, y) coordinates of pixels with the color values
    """
    arr = np.array(img)
    X, Y = np.where(np.all(arr==color,axis=2))
    keycoordinates = list(zip(X,Y))

    try:
        keycoordinates[-1], keycoordinates[-2] = keycoordinates[-2], keycoordinates[-1] # fixing the order of keypoints
    except ValueError:
        print("Empty list, color value not founded !")

    return keycoordinates


def corners2mask(img, coordinates):
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


def mask_pixels(mask):
    """
    Get (x,y) position of the pixels inside the mask
    :param mask: boolean array where the pixels inside the polygon have True values
    :return: list of coordinates of pixels inside the mask
    """
    a, b = np.where(mask==True)
    pixels = list(zip(a,b))
    return pixels


def mask_img(img, coord):
    """
    Display mask on top of an image
    :param img: original image
    :param coord: coordinates of the pixels inside the mask
    :return: array with mask in top of the image
    """
    maskk = np.ones_like(img) * 255
    points = np.fliplr(np.array(coord, np.int32))

    cv2.fillPoly(maskk, [points], (0, 255, 255))
    img_with_mask = cv2.normalize(np.int64(img) * maskk, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return img_with_mask

if __name__ == '__main__':
    # open the image
    image = PIL.Image.open('capture.png').convert('RGB')
    # my corners pixels have red color [237, 28, 36]
    red = [237, 28, 36]
    # Detect position of corners
    corner = color2coordinates(image, red)
    # convert corners to mask
    mask = corners2mask(image, corner)
    # Get position of pixels inside the mask
    my_pixels = mask_pixels(mask)
    print(my_pixels)

    # testing : display our pixels founded on top of our image
    img_with_mask = mask_img(image, my_pixels)

    plt.imshow(img_with_mask)
    plt.show()
