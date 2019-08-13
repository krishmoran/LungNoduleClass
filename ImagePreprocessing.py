import numpy as np
import cv2
import os

# Loads and processes a slice of a CT image given its filepath
def load_image(fpath):

    # loads a gray slice given its path and reduces pixel intensity to obtain original HU values
    if os.path.exists(fpath):
        img_gray = cv2.imread(fpath)
        img_gray = img_gray.astype(np.float)-32768
    else:
        raise FileNotFoundError

    # converts grayscale image to color image format (2 array -> BGR)
    # img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Normalizes the image pixels within the range [0,1]
    normalizedImg = np.zeros((512, 512))
    normalizedImg = cv2.normalize(img_gray,  normalizedImg, 0, 1, cv2.NORM_MINMAX)

    # returns the matrix of the image
    print(normalizedImg)

    # creates window and displays given image to test scaling 
    window = cv2.namedWindow('Display', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, normalizedImg)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

    # return(normalizedImg)
load_image('/Users/krishmoran/Documents/Lung Classification/Images_png/004408_01_02/110.png')





