class NetworkGenerator:

    def __init__(self):
        self.author = 'ACN'

    def get_alex_net(self, pic_size):

        """ AlexNet.
        Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
        References:
            - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
            Classification with Deep Convolutional Neural Networks. NIPS, 2012.
            - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
        Links:
            - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
            - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
        """

        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.normalization import local_response_normalization
        from tflearn.layers.estimator import regression

        # Building 'AlexNet'
        network = input_data(shape=[None, pic_size[0], pic_size[1], 3])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network, 'AlexNet'

    def get_convnet(self, pic_size):

        """ Convolutional network applied to CIFAR-10 dataset classification task.
        References:
            Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
        Links:
            [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
        """

        import tflearn
        from tflearn.data_utils import shuffle, to_categorical
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.estimator import regression
        from tflearn.data_preprocessing import ImagePreprocessing
        from tflearn.data_augmentation import ImageAugmentation

        # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

        # Convolutional network building
        network = input_data(shape=[None, pic_size[0], pic_size[1], 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

        return network, 'ConvNet'

    def get_highway(self, pic_size):

        """ Convolutional Neural Network for MNIST dataset classification task.
        References:
            Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
            learning applied to document recognition." Proceedings of the IEEE,
            86(11):2278-2324, November 1998.
        Links:
            [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
        """

        from __future__ import division, print_function, absolute_import

        import tflearn
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import highway_conv_2d, max_pool_2d
        from tflearn.layers.normalization import local_response_normalization, batch_normalization
        from tflearn.layers.estimator import regression

        # Building convolutional network
        network = input_data(shape=[None, 28, 28, 1], name='input')
        # highway convolutions with pooling and dropout
        for i in range(3):
            for j in [3, 2, 1]:
                network = highway_conv_2d(network, 16, j, activation='elu')
            network = max_pool_2d(network, 2)
            network = batch_normalization(network)

        network = fully_connected(network, 128, activation='elu')
        network = fully_connected(network, 256, activation='elu')
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')

        return network, 'highway'
