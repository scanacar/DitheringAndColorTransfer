import numpy as np
# from statistics import mean


#  This function allows us to keep the values between 0 and 255
#  if the pixel values are not between 0 and 255.
def min_max_value(value):
    if value > 255:
        value = 255
    elif value < 0:
        value = 0
    else:
        value = value

    return value


#  This function allows us to find the quantized value.
def find_quantized_value(old_pixel, q):
    return np.round(old_pixel / q) * q


#  Floyd Steinberg Dithering Function
def FloydSteinberg(image, q):
    height, width = image.shape[0], image.shape[1]
    pixel = image.copy()
    # quantization_errors = []  # For holding quantization errors

    for x in range(0, height - 1):
        for y in range(0, width - 1):
            old_pixel = pixel[x][y]
            new_pixel = find_quantized_value(old_pixel, q)
            pixel[x][y] = new_pixel
            quant_error = old_pixel - new_pixel
            # quantization_errors.append(quant_error)
            pixel[x + 1][y] = min_max_value(pixel[x + 1][y] + quant_error * 7 / 16)
            pixel[x - 1][y + 1] = min_max_value(pixel[x - 1][y + 1] + quant_error * 3 / 16)
            pixel[x][y + 1] = min_max_value(pixel[x][y + 1] + quant_error * 5 / 16)
            pixel[x + 1][y + 1] = min_max_value(pixel[x + 1][y + 1] + quant_error * 1 / 16)

    # print(mean(quantization_errors))  # Averaging quantization errors

    return pixel
