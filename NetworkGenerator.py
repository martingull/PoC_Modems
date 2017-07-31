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
