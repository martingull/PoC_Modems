

from __future__ import division, print_function, absolute_import

import tflearn
from DataProcessing import PreProcessor
from NetworkGenerator import NetworkGenerator
# import tflearn.datasets.oxflower17 as oxflower17

mode = 'TRAIN'  # Set to 'TRAIN' or 'TEST'
pic_size = (227, 227)

dataset = PreProcessor()
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
# dataset.video_to_images()
dataset.augment_data()
X, Y = dataset.load_data('./processed_data')
n_labels = Y.shape[1]

# Get Architecture
net_gen = NetworkGenerator()
network, net_name = net_gen.get_alex_net(pic_size)

# Training
model = tflearn.DNN(network, checkpoint_path='model_' + net_name,
                    max_checkpoints=1, tensorboard_verbose=2)
if mode == 'TRAIN':
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id=net_name + '_' + 'oxflowers17')
    model.save('./model.tflearn')
elif mode == 'TEST':
    model.load('./model.tflearn')
    # model.predict()

    # Perform post-analysis
