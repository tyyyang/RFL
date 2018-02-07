from rfl_net.network import Network
import config

class XZNet(Network):

    def setup(self):
        (self.feed('input')
         .conv(11, 11, 96, 2, 2, padding='VALID', name='conv1', relu=False)
         .bn(is_train=self.is_train, relu=True, name='bn1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 256, 1, 1, padding='VALID', name='conv2', relu=False)
         .bn(is_train=self.is_train, relu=True, name='bn2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 384, 1, 1, padding='VALID', name='conv3', relu=False)
         .bn(is_train=self.is_train, relu=True, name='bn3')
         .conv(3, 3, 384, 1, 1, padding='VALID', name='conv4', relu=False)
         .bn(is_train=self.is_train, relu=True, name='bn4')
         .conv(3, 3, 256, 1, 1, padding='VALID', name='conv5', relu=False)
         .bn(is_train=self.is_train, relu=False, name='bn5'))

class FilterNet(Network):
    def setup(self):
        (self.feed('output')
         .conv(1, 1, config.output_size, 1, 1, padding='SAME', name='conv6', relu=False))

class ConvXZNet(Network):

    def setup(self):
        (self.feed('z_gf', 'x_output')
         .batch_conv(name='response'))