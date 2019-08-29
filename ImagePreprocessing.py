import numpy as np
import cv2
import os
import time, sys


# Hidden File Ignore
# <editor-fold desc="Hidden File Ignore">
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


# Loads and processes a slice of a CT image given its filepath
def load_image(fpath):
    # loads a gray slice given its path and reduces pixel intensity to obtain original HU values
    if os.path.exists(fpath):
        img_gray = cv2.imread(fpath)
        img_gray = img_gray.astype(np.float) - 32768
    else:
        raise FileNotFoundError

    # converts grayscale image to color image format (2 array -> BGR)
    # img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Normalizes the image pixels within the range [0,1]
    normalizedImg = np.zeros((512, 512))
    normalizedImg = cv2.normalize(img_gray, normalizedImg, 0, 1, cv2.NORM_MINMAX)

    # creates window and displays given image to test scaling x
    # cv2.imshow('CT Slice', normalizedImg)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()

    # returns the matrix of the image 
    return normalizedImg

# given the folder path and filename (e.g. '000.png'), returns the full filepath 
def get_path(folder, filename):
    return folder + '/' + filename

# to load and stack 5 slices, including the indicated key slice and the 2 indexed above and below
# key_slice is inputted as the index of the image 
def stack_slices(patient_folder, key_slice):
    patient1_slices = []

    slices = list(listdir_nohidden(patient_folder))
    slices.sort()

    # creates a new list of the corresponding slice indices for easier searcing
    sl_indices = []
    for str in slices:
        sl_indices.append(int(str[:3]))

    # determines the first (lowest indexed) slice to be added
    infslice = key_slice - 2
    # superior slice = key_slice + 2

    for i in range(len(slices)):
        if sl_indices[i] == infslice:
            x = load_image(get_path(patient_folder, slices[i]))
            patient1_slices.append(x)
            print("Slice Extracted:", sl_indices[i])

            # once the inferior slice is found, the subsequent 4 slices are added to the list
            for j in range(i + 1, i + 5):
                x = load_image(get_path(patient_folder, slices[j]))
                print("Slice Extracted:", sl_indices[j])
                patient1_slices.append(x)
                
    # ims = [im.astype(float) for im in patient1_slices]
    pat = patient1_slices
    im = np.concatenate(patient1_slices)
    # final_img = np.stack(patient1_slices)
    print(im)

stack_slices('/Users/krishmoran/Documents/LungClassification/Krish/SamplePatients/Patient_4274', 126)


# DeepLesion method for combining slices 
#     imgs = [im.astype(float) for im in imgs]
#     im = cv2.merge(imgs)
#     im = im.astpype(np.float32, False) - 32768





