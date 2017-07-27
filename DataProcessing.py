import os
from PIL import Image
import numpy as np
import av

VIDEO_ROOT = './Pictures KPN'
DATA_ROOT = './raw_pictures'


class PreProcessor:

    def __init__(self, resize_pics=(227, 227)):
        self.video_root = VIDEO_ROOT
        self.picture_root = DATA_ROOT
        self.resize_pics = resize_pics

    def video_to_images(self):

        # check if destination folder present
        if not os.path.exists('./raw_pictures'):
            os.makedirs('./raw_pictures')

        folders = [folder for folder in os.listdir(VIDEO_ROOT)]
        for folder in folders:
            files = [file for file in os.listdir(VIDEO_ROOT + '/' + folder)]
            for file in files:
                container = av.open(VIDEO_ROOT + '/' + folder + '/' + file)

                # check if destination folder present
                if not os.path.exists('./raw_pictures/' + folder):
                    os.makedirs('./raw_pictures/' + folder)

                # resize and save frame to disk
                for frame in container.decode(video=0):
                    frame = frame.reformat(width=self.resize_pics[0], height=self.resize_pics[1])
                    img = frame.to_image()
                    head = ''.join(file.split('.')[:1])  # remove everything after punctuation
                    img.save('./raw_pictures/' + folder + '/' + head + 'frame-%04d.jpg' % frame.index)

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

    def load_data(self):

        folders = [f.path for f in os.scandir(DATA_ROOT) if f.is_dir()]
        n_images, n_folder = self.count_instances(DATA_ROOT)
        data = np.zeros((n_images, self.resize_pics[0], self.resize_pics[1], 3), dtype=np.float32)
        target = np.zeros((n_images, n_folder), dtype=np.float32)
        i_image = 0
        i_folder = -1
        for folder in folders:
            i_folder += 1
            files = [f.path for f in os.scandir(folder)]
            for file in files:
                # Parse images
                img = self.load_image(file)
                data[i_image, :, :, :] = img
                target[i_image, i_folder] = 1.0
                i_image += 1

        return data, target
