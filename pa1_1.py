import cv2 as cv2
import numpy as np
from pa1_2 import FloydSteinberg

if __name__ == "__main__":

    image = cv2.imread("dithering/image4.jpg")  # Reading an image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray scale
    grayscale_image_for_dithering = grayscale_image.copy()
    q = 4  # Quantization parameter
    q = 255 / q
    #  Quantization implementation
    for i in range(grayscale_image.shape[0]):
        for j in range(grayscale_image.shape[1]):
            grayscale_image[i][j] = np.round(grayscale_image[i][j] / q) * q
    #  Showing Quantized Image
    cv2.imshow("Quantized Image", grayscale_image)
    #  Calling Floyd Steinberg Dithering function with Quantized Image and q parameter
    dithered_image = FloydSteinberg(grayscale_image_for_dithering, q)
    #  Showing Dithered Image
    cv2.imshow("Dithered Image", dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
