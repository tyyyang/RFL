import tensorflow as tf
from rfl_net.network import Network
import config

class InitLSTMSate(Network):

    def setup(self):

        (self.feed('input')
         .conv(config.conv_filter_size, config.conv_filter_size, config.hidden_size, 1, 1, padding='SAME', name='conv_c', relu=False)
         .tanh(name='c_state'))

        (self.feed('input')
         .conv(config.conv_filter_size, config.conv_filter_size, config.hidden_size, 1, 1, padding='SAME', name='conv_h', relu=False)
         .tanh(name='h_state'))

        (self.feed('c_state', 'h_state', 'state_size')
         .fuse_lstm_state(name='lstm_state'))

class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):
    """Basic Conv LSTM recurrent network cell.
    """

    def __init__(self, shape, filter_size, num_features, is_train, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tf.nn.tanh, max_pool_ss=None, padding='SAME'):
        """Initialize the basic Conv LSTM cell."""
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._max_pool_ss = max_pool_ss
        self._padding = padding
        self._is_train = is_train

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(tf.TensorShape(self.output_size),
                                              tf.TensorShape(self.output_size))
                if self._state_is_tuple else 2 * self.shape[0] * self.shape[1] * self.num_features)

    @property
    def output_size(self):
        return self.shape + [self.num_features]

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
            if self._max_pool_ss is not None:
                inputs = tf.nn.max_pool(inputs, [1] + self._max_pool_ss[0:2] + [1],
                                        [1] + self._max_pool_ss[2:4] + [1], self._padding)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)
            estimated_cell = self._activation(j)
            input_gate = tf.nn.sigmoid(i)
            forget_gate = tf.nn.sigmoid(f + self._forget_bias)
            output_gate = tf.nn.sigmoid(o)

            new_c = (c * forget_gate + input_gate * estimated_cell)
            new_h = self._activation(new_c) * output_gate

            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state, input_gate, forget_gate, output_gate

def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "weights", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "biases", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term