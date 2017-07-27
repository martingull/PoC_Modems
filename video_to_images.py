import av
import os

DATA_ROOT = './Pictures KPN'

def video_to_images():

    # check if destination folder present
    if not os.path.exists('./raw_pictures'):
        os.makedirs('./raw_pictures')

    folders = [folder for folder in os.listdir(DATA_ROOT)]
    for folder in folders:
        files = [file for file in os.listdir(DATA_ROOT + '/' + folder)]
        for file in files:
            container = av.open(DATA_ROOT + '/' + folder + '/' + file)

            # check if destination folder present
            if not os.path.exists('./raw_pictures/' + folder):
                os.makedirs('./raw_pictures/' + folder)

            # resize and save frame to disk
            for frame in container.decode(video=0):
                frame = frame.reformat(width=227, height=227)
                img = frame.to_image()
                head = ''.join(file.split('.')[:1])  # remove everything after punctuation
                img.save('./raw_pictures/' + folder + '/' + head + 'frame-%04d.jpg' % frame.index)

if __name__ == '__main__':
    video_to_images()
