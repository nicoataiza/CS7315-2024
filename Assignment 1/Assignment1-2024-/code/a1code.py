### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io


def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = io.imread(img_path)
    # convert to 0.0 to 1.0
    return out / 255


def print_stats(image):
    """Prints the height, width and number of channels in an image.

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).

    Returns: none
    """
    if len(image.shape) == 3:
        width, height, channels = image.shape
    else:
        width, height = image.shape
        channels = 1
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"channels: {channels}")
    # return image.shape


def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index
        start_col (int): The starting column index
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    out = image[start_row : start_row + num_rows, start_col : start_col + num_cols, :]
    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0
    If you are using values 0-255, change 0.5 to 128.
    Pixel values will be within range.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = np.minimum(factor * (image - 0.5) + 0.5, 1)
    out = np.maximum(out, 0)
    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (float): Scaled percentage of desired size of the rows of the output image
        output_cols (float): Scaled percentage of desired size of the columns of the output image

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    original_size = input_image.shape
    target_row, target_column = int(original_size[0] * output_rows), int(
        original_size[1] * output_cols
    )
    out = np.empty((target_row, target_column, 3))
    # iterate through each pixel
    for channel in range(3 if original_size[2] == 3 else 1):
        for i in range(target_row):
            for j in range(target_column):
                out[i][j][channel] = input_image[int(i / output_rows)][
                    int(j / output_cols)
                ][channel]
    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale.
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """

    return np.mean(input_image, axis=2)


def binary(input_image, t):
    """Creates a binary mask given value t.
    Returns a black and white image where the value is black if pixel value is >= t.

    Inputs:
        input_image: numpy array of shape (Hi, Wi).
        t: float [0, 1]

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    return (input_image >= t).astype(np.uint8)


def conv2D(image, kernel):
    """Convolution of a 2D image with a 2D kernel.
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    # target size is defined to be (W + kW - 1) * (H + kH - 1)
    padded_size = (
        image.shape[0] + kernel.shape[0] - 1,
        image.shape[1] + kernel.shape[1] - 1,
    )
    # pad the image with zeroes
    padded_image = np.zeros(padded_size)

    # go through each pixel; offset from padded to output is 1 pixel shifted right
    if kernel.shape[0] * kernel.shape[1] != 1:
        padded_image[
            kernel.shape[0] // 2 : kernel.shape[0] // 2 + image.shape[0],
            kernel.shape[1] // 2 : kernel.shape[1] // 2 + image.shape[1],
        ] = image
    else:
        padded_image = np.copy(image)
    out = np.zeros_like(image)
    flipped_kernel = np.flip(kernel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            section = padded_image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            out[i][j] = np.sum(section * flipped_kernel)
    return out


def test_conv2D():
    """A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.

    Returns:
        None
    """

    # Test code written by
    # Simple convolution kernel.
    kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])
    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1
    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)
    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3
    # Test if the output matches expected output
    assert (
        np.max(test_output - expected_output) < 1e-10
    ), "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = np.copy(image)
    if len(image.shape) == 3:
        for channel in range(3):
            out[:, :, channel] = conv2D(out[:, :, channel], kernel)
    else:
        out = conv2D(out, kernel)
    return out


def gauss2D(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.

    Args:
        size: filter height and width
        sigma: std deviation of Gaussian

    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


# def corr(image, kernel):
#     """Cross correlation of a RGB image with a 2D kernel

#     Args:
#         image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
#         kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

#     Returns:
#         out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
#     """
#     out = None
#     ### YOUR CODE HERE

#     return out
