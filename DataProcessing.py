import os
from PIL import Image
import tflearn
import numpy as np
import av
import tensorflow as tf


# TODO: video and data root | and 'rawpictures'
# TODO: sample every 5th or so image
# TODO: v2image should write to a folder using resize pics in title. to avoid retraining etc.
# TODO: Add data augmentation

class PreProcessor:

    def __init__(self):
        self.size_pics = (256, 256)

    def create_data(self):
        self.video_to_images(size_pics=self.size_pics)

    def augment_data(self, image_path='./raw_pictures', augment_path='./processed_data'):

        # check if destination folder present
        if not os.path.exists(augment_path):
            os.makedirs(augment_path)

        folders = [f.name for f in os.scandir(image_path) if f.is_dir()]
        for folder in folders:
            files = [f.path for f in os.scandir(image_path + '/' + folder)]

            # check if destination folder present
            if not os.path.exists(augment_path + '/' + folder):
                os.makedirs(augment_path + '/' + folder)

            for file in files:
                tf.image.decode_jpeg(file)

    @staticmethod
    def video_to_images(image_path='./raw_pictures', video_path='./Pictures KPN', size_pics=(256, 256)):

        # check if destination folder present
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        folders = [folder for folder in os.listdir(video_path)]
        for folder in folders:
            files = [file for file in os.listdir(video_path + '/' + folder)]

            # check if destination folder present
            if not os.path.exists(image_path + '/' + folder):
                os.makedirs(image_path + '/' + folder)

            for file in files:
                container = av.open(video_path + '/' + folder + '/' + file)

                # resize and save frame to disk
                for frame in container.decode(video=0):
                    frame = frame.reformat(width=size_pics[0], height=size_pics[1])
                    img = frame.to_image()
                    head = ''.join(file.split('.')[:1])  # remove everything after punctuation
                    img.save(image_path + '/' + folder + '/' + head + 'frame-%04d.jpg' % frame.index)

    @staticmethod
    def load_image(infile_name):
        img = Image.open(infile_name)
        img.load()
        return np.asarray(img, dtype="int32")

    @staticmethod
    def count_instances(root_dirname):
        n_images = 0
        folders = [f.path for f in os.scandir(root_dirname) if f.is_dir()]
        n_folder = len(folders)
        for folder in folders:
            n_images += len(os.listdir(folder))
        return n_images, n_folder

    @staticmethod
    def load_data(root_directory, size_pic=(256, 256)):

        folders = [f.path for f in os.scandir(root_directory) if f.is_dir()]
        n_images, n_folder = PreProcessor.count_instances(root_directory)
        data = np.zeros((n_images, size_pic[0], size_pic[1], 3), dtype=np.float32)
        target = np.zeros((n_images, n_folder), dtype=np.float32)
        i_image = 0
        i_folder = -1
        for folder in folders:
            i_folder += 1
            files = [f.path for f in os.scandir(folder)]
            for file in files:
                # Parse images
                img = PreProcessor.load_image(file)
                data[i_image, :, :, :] = img
                target[i_image, i_folder] = 1.0
                i_image += 1

        return data, target
