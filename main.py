

from __future__ import division, print_function, absolute_import

import tflearn
from DataProcessing import PreProcessor
from NetworkGenerator import NetworkGenerator
# import tflearn.datasets.oxflower17 as oxflower17

mode = 'TRAIN'  # Set to 'TRAIN' or 'TEST'

dataset = PreProcessor()
# dataset.video_to_images()
# dataset.augment_data()
X, Y = dataset.load_data('./processed_data')
n_labels = Y.shape[1]
size_pics = dataset.size_pics

# Get Architecture
net_gen = NetworkGenerator()
network, net_name = net_gen.get_alex_net(size_pics)

# Training
model = tflearn.DNN(network, checkpoint_path='model_' + net_name,
                    max_checkpoints=1, tensorboard_verbose=2)
if mode == 'TRAIN':
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id=net_name + '_' + 'kpn')

    model.save('./model.tflearn')

# Testing
elif mode == 'TEST':
    model.load('./model.tflearn')
    # model.predict()

    # Perform post-analysis
