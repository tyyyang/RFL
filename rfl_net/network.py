import numpy as np
import tensorflow as tf
import os
from tensorflow.python.util import nest
from tensorflow.python.training import moving_averages


DEFAULT_PADDING = 'SAME'
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'batch_normalization'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, is_train):
        # The input nodes for this network
        self.inputs = inputs
        self.is_train = is_train
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)

        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False, var_scope = None):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='bytes').item()
        for op_name in data_dict:
            if var_scope is not None:
                variable_sco = os.path.join(var_scope, op_name)
            else:
                variable_sco = op_name
            with tf.variable_scope(variable_sco, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name.decode('utf-8'))
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, initializer=None, trainable=True):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             trainable=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o], trainable=trainable)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
                kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(axis=3, values=output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], trainable=trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)


    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def hiden_size(self, input, name):
        input_shape = input.get_shape().as_list()
        return input_shape[3]


    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def batch_conv(self, inputs, name):
        z_conv = inputs[0]
        x_conv = inputs[1]

        batch_size = z_conv.get_shape().as_list()[0]
        chans = []
        for i in range(batch_size):
            input_single = tf.expand_dims(tf.gather(x_conv, i), 0)
            filter = tf.expand_dims(tf.gather(z_conv, i),3)
            chan = tf.nn.conv2d(input_single, filter, [1, 1, 1, 1], 'VALID')
            chans.append(tf.squeeze(chan))
        response = tf.stack(chans, 0)

        return response

    @layer
    def fuse_lstm_state(self, inputs, name):
        c = inputs[0]
        h = inputs[1]
        state_size = inputs[2]
        return nest.pack_sequence_as(structure=state_size,
                              flat_sequence=[c,h])

    @layer
    def bn(self, input, name, is_train, is_use_bias=False, relu=False):

        with tf.variable_scope(name):
            input_shape = input.get_shape()
            params_shape = input_shape[-1:]

            if is_use_bias:
                bias = self.make_var('bias', params_shape,
                                     initializer=tf.zeros_initializer())
                return input + bias

            axis = list(range(len(input_shape) - 1))

            beta = self.make_var('beta',
                                 params_shape,
                                 initializer=tf.zeros_initializer())
            gamma = self.make_var('gamma',
                                  params_shape,
                                  initializer=tf.ones_initializer())

            if is_train:
                # These ops will only be preformed when training.
                moving_mean = tf.get_variable('moving_mean',
                                            params_shape,
                                            trainable=False,
                                            initializer=tf.zeros_initializer())
                moving_variance = tf.get_variable('moving_variance',
                                                params_shape,
                                                trainable=False,
                                                initializer=tf.ones_initializer())

                mean, variance = tf.nn.moments(input, axis)
                update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                           mean, BN_DECAY)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, BN_DECAY)
                tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
                tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                # tf.histogram_summary(mean.op.name, mean)
                # tf.histogram_summary(variance.op.name, variance)

            output = tf.nn.batch_normalization(input, mean, variance, beta, gamma, BN_EPSILON, name = name)
            if relu:
                output = tf.nn.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
