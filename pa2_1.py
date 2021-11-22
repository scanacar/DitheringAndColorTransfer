import cv2 as cv2
from pa2_2 import colorTransfer

if __name__ == "__main__":

    #  Reading Source and Target Images
    source = cv2.imread("colorTransfer/woods.jpg")
    target = cv2.imread("colorTransfer/storm.jpg")

    #  Result of Color Transfer
    result_of_color_transfer = colorTransfer(source, target)

    #  Showing Source-Target-Result Images
    cv2.imshow("Source Image", source)
    cv2.imshow("Target Image", target)
    cv2.imshow("Result", result_of_color_transfer)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
