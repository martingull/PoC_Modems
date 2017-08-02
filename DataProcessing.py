import os
from PIL import Image
import numpy as np
import av
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# TODO: video and data root | and 'rawpictures'
# TODO: v2image should write to a folder using resize pics in title. to avoid retraining etc.

class PreProcessor:

    def __init__(self, size_pics=(256, 256)):
        self.size_pics = size_pics

    def create_data(self):
        self.video_to_images(size_pics=self.size_pics)
        self.augment_data()

    def augment_data(self, image_path='./raw_pictures', augment_path='./processed_data', batch_size=10):

        # Define data augmentor
        dataaug = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # check if destination folder present
        if not os.path.exists(augment_path):
            os.makedirs(augment_path)

        folders = [f.name for f in os.scandir(image_path) if f.is_dir()]
        for folder in folders:
            files = [f.path for f in os.scandir(image_path + '/' + folder)]

            # check if destination folder present
            if not os.path.exists(augment_path + '/' + folder):
                os.makedirs(augment_path + '/' + folder)

            # for each raw image
            for file in files:
                img = load_img(file)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)  # dim: (1, height, width, rbg)
                i = 0
                for batch in dataaug.flow(x, batch_size=1, save_to_dir=augment_path + '/' + folder, save_prefix='aug',
                                          save_format='jpeg'):
                    i += 1
                    if i > batch_size:
                        break

    @staticmethod
    def video_to_images(image_path='./raw_pictures', video_path='./Pictures KPN', size_pics=(256, 256), frame_spacing=3):

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
                c = 0
                for frame in container.decode(video=0):
                    if c % frame_spacing == 0:
                        frame = frame.reformat(width=size_pics[0], height=size_pics[1])
                        img = frame.to_image()
                        head = ''.join(file.split('.')[:1])  # remove everything after punctuation
                        img.save(image_path + '/' + folder + '/' + head + 'frame-%04d.jpg' % frame.index)
                    c += 1

    @staticmethod
    def load_image(file):
        img = load_img(file)
        x = img_to_array(img)
        return x
        # return np.expand_dims(x, axis=0)  # dim: (1, height, width, rbg)

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
                x = PreProcessor.load_image(file)
                data[i_image, :, :, :] = x
                target[i_image, i_folder] = 1.0
                i_image += 1

        return data, target
