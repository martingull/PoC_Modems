import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

DATA_ROOT = './raw_pictures'


def load_image(infile_name):
    img = Image.open(infile_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def count_instances(root_dirname):
    n_images = 0
    folders = [folder for folder in os.listdir(root_dirname)]
    n_folder = len(folders)
    for folder in folders:
        n_images += len(os.listdir(root_dirname + '/' + folder))
    return n_images, n_folder

folders = [folder for folder in os.listdir(DATA_ROOT)]
# Make dict of location and label name, use later when printing
n_images, n_folder = count_instances(DATA_ROOT)
data = np.zeros((n_images, 227, 227, 3), dtype=np.float32)
target = np.zeros((n_images, n_folder), dtype=np.float32)
i_image = 0
i_folder = -1
for folder in folders:
    i_folder += 1
    files = [file for file in os.listdir(DATA_ROOT + '/' + folder)]
    for file in files:
        # Parse images
        img = load_image(DATA_ROOT + '/' + folder + '/' + file)
        data[i_image, :, :, :] = img
        target[i_image, i_folder] = 1.0
        i_image += 1

#

