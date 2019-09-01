import numpy as np
import cv2
import os
import time, sys
import pandas as pd

# Constants
input_patient_folder = '/Users/krishmoran/Documents/LungClassification/Krish/SamplePatients/Patient_4274'
slice_info = '/Users/krishmoran/Downloads/DL_info.csv'

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

    # Normalizes the image pixels within the range [0,1]
    # and converts grayscale image to color image format (2 array -> BGR)
    normalizedImg = np.zeros((512, 512))
    normalizedImg = cv2.normalize(img_gray, normalizedImg, 0, 1, cv2.NORM_MINMAX)

    # creates window and displays given image to test scaling x
    # cv2.imshow('CT Slice', normalizedImg)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()

    # returns the matrix of the image 
    return normalizedImg

# given the folder path and filename (e.g. '047.png'), returns the full filepath (OS X Dir)
def get_file_path(folder, filename):
    return folder + '/' + filename

# to load and stack 5 slices, including the indicated key slice and the 2 indexed above and below
# key_slice is inputted as the index of the image 
def stack_slices(patient_folder, key_slice):
    patient1_slices = []

    slices = list(listdir_nohidden(patient_folder))
    slices.sort()

    # creates a new list of the corresponding slice indices (ints for easier searcing
    sl_indices = []
    for str in slices:
        sl_indices.append(int(str[:3]))

    # determines the first (lowest indexed) slice to be added
    # slice = 0 (top/superior)
    # inferior slice = key_slice + 2
    sup_slice = key_slice - 2

    # divides print info for each patient when displaying all of them
    print('--------------------------')
    f = patient_folder[-10:]
    print('PATIENT' + " " + f[:4])

    for i in range(len(slices)):
        if sl_indices[i] == sup_slice:
            x = load_image(get_file_path(patient_folder, slices[i]))
            patient1_slices.append(x)
            print("Slice Extracted:", sl_indices[i])

            # once the superior slice is found, the subsequent 4 slices are added to the list
            for j in range(i + 1, i + 5):
                x = load_image(get_file_path(patient_folder, slices[j]))
                print("Slice Extracted:", sl_indices[j])
                patient1_slices.append(x)
                

    # im = np.concatenate(patient1_slices)
    final_img = np.stack(patient1_slices)
    return final_img

# stack_slices('/Users/krishmoran/Documents/LungClassification/Krish/SamplePatients/Patient_4274', 126)
# stack_slices('/Users/krishmoran/Documents/LungClassification/Images2_png/004264_01_01/', 86)\

# Retrieves the list of lung patients from CSV and creates a 
# 2D array with their folder name and the corresponding key slice index
def get_patient_list():

    # Loads the patient scan info CSV 
    df = pd.read_csv(slice_info)

    # Sorts the dataset by only lung scans
    df1 = df.loc[(df['Coarse_lesion_type'] == 5)]

    patient_indices = [] # used to keep frequency of each patient folder = 1
    unique_patients = [] # list of exact folder names without duplicates of patients
    key_slices = [] # list of the corresponding key slice indices
    file_names = df1.File_name
    for fname in file_names:
        if fname[:6] not in patient_indices: 
            patient_indices.append(fname[:6]) # adds patient index to freq array, prevents duplicates 
            unique_patients.append(fname[:12]) # adds the part of the filename which represents the folder name
            key_slices.append(int(fname[13:16])) # adds the key slice from the filename into an array
    
    # stacks the two lists(patient list and corresponding key slices) to make a 2D array
    pats_w_key_slices = np.stack((unique_patients, key_slices), axis = -1)
    return(pats_w_key_slices)


# returns the full folder path given the main path and the folder name 
def get_folder_path(ultimate_path, folder_name):
    return ultimate_path + '/' + folder_name

# loads and processes all lung patients stated in the info CSV
def load_all_patients(all_patients_path):
    patient_list = get_patient_list()
    preprocessed_patients = []
    for i in range (len(patient_list)):
        pat_folder = get_folder_path(all_patients_path, patient_list[i, 0])
        key_slice = int(patient_list[i, 1])
        if os.path.exists(pat_folder):
            preprocessed_patients.append(stack_slices(pat_folder, key_slice))

load_all_patients('/Users/krishmoran/Documents/LungClassification/Images2_png')







