import functools
import numpy as np
import tensorflow as tf
import config
from rfl_net.network import UPDATE_OPS_COLLECTION
from rfl_net.rnn import rnn, DropoutWrapper
from data_input.prepare_targets import create_labels_overlap
from rfl_net.utils import activation_summary

from rfl_net.xz_net import XZNet, ConvXZNet, FilterNet
from rfl_net.conv_lstm import BasicConvLSTMCell, InitLSTMSate


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class RFLNet():
    def __init__(self, is_train, z_examplar=None, x_crops=None, y_crops=None, init_z_exemplar=None):

        self._is_train = is_train
        input_shape = z_examplar.get_shape().as_list()

        self._batch_size = input_shape[0]
        self._time_steps = input_shape[1]
        x_shape = x_crops.get_shape().as_list()
        self._z_examplar = tf.reshape(z_examplar, [-1, config.z_exemplar_size, config.z_exemplar_size, 3])
        self._x_crops = tf.reshape(x_crops, [-1]+ x_shape[2:])
        self._y_crops = y_crops
        self._response_size = config.response_size-int(2*8/config.stride) if config.is_augment and is_train else config.response_size
        self._gt_pos = tf.convert_to_tensor(np.floor([self._response_size/2, self._response_size/2]), tf.float32)
        if init_z_exemplar is not None:
            self.init_z_exemplar = tf.reshape(init_z_exemplar, [-1, config.z_exemplar_size, config.z_exemplar_size, 3])

        self.filter
        self.response
        if y_crops is not None:
            self.loss
            self.dist_error
        else:
            self.init_state_filter
        if is_train:

            self._global_step = tf.get_variable('global_step', [], tf.int64, initializer=tf.constant_initializer(0),
                                          trainable=False)
            self._lr = tf.train.exponential_decay(config.learning_rate, self._global_step, config.decay_circles,
                                                  config.lr_decay, staircase=True)
            tf.summary.scalar('learning_rate', self._lr)
            self.optimize

        self._summary = tf.summary.merge_all()
        self._saver = tf.train.Saver(tf.global_variables())

    @lazy_property
    def init_state_filter(self):

        with tf.variable_scope('z_net', reuse=True):
            z_net = XZNet({'input': self.init_z_exemplar}, self._is_train)

        with tf.variable_scope('init_state'):
            init_state_net = InitLSTMSate({'input': z_net.get_output(),'state_size':self._state_size}, self._is_train)

        init_state = init_state_net.get_output()

        with tf.variable_scope('z_filter', reuse=True):
            first_output = init_state[1]
            init_filter_net = FilterNet({'output': first_output}, self._is_train)
        init_filter = init_filter_net.get_output()

        return tuple([init_state, init_filter])

    @lazy_property
    def filter(self):

        # build z_net for reference image
        with tf.variable_scope('z_net'):
            z_net = XZNet({'input': self._z_examplar}, self._is_train)

        # build rnn for filter generation
        z_output = z_net.get_output()

        gf_shape = z_output.get_shape().as_list()
        # building rnn cell
        rnn_cell = BasicConvLSTMCell(gf_shape[1:3], [config.conv_filter_size, config.conv_filter_size],
                                         config.hidden_size, self._is_train,
                                         forget_bias=1.0, activation=tf.nn.tanh)

        if self._is_train and config.keep_prob < 1:
            rnn_cell = DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)
        # cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_rnn_layers)
        cell = rnn_cell
        self._state_size = cell.state_size

        # reorganize rnn input
        rnn_inputs = tf.reshape(z_output, [self._batch_size, self._time_steps] + gf_shape[1:4])
        if self._is_train and config.keep_prob < 1:
            rnn_inputs = tf.nn.dropout(rnn_inputs, config.keep_prob)
        rnn_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=self._time_steps, value=rnn_inputs)]

        if self._is_train or self._y_crops is not None:
            with tf.variable_scope('init_state'):
                init_state_net = InitLSTMSate({'input':rnn_inputs[0],'state_size':self._state_size}, self._is_train)
            initial_state = init_state_net.get_output()
            rnn_inputs_new = rnn_inputs[1:self._time_steps]
            outputs, final_state, input_gates, forget_gates, output_gates \
                = rnn(cell, rnn_inputs_new, initial_state=initial_state)
            first_output = initial_state[1]
            outputs = [first_output] + outputs
        else:
            self._initial_state = cell.zero_state(self._batch_size, tf.float32)
            outputs, final_state, input_gates, forget_gates, output_gates \
                = rnn(cell, rnn_inputs, initial_state=self._initial_state)

        outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1]+ gf_shape[1:3]+[config.hidden_size])
        self._final_state = final_state

        with tf.variable_scope('z_filter'):
            f_net = FilterNet({'output': outputs}, self._is_train)
        z_gf = f_net.get_output()

        activation_summary(z_output, 'activation/z_output')
        activation_summary(z_gf, 'activation/z_gf')
        activation_summary(final_state, 'activation/cell_state')
        activation_summary(input_gates, 'gates/input')
        activation_summary(forget_gates, 'gates/forget')
        activation_summary(output_gates, 'gates/output')

        return z_gf

    @lazy_property
    def response(self):

        if self._is_train or self._y_crops is not None:
            self._z_gf = self.filter
        else:
            self._z_gf = tf.placeholder(tf.float32, [config.num_scale, 6, 6, 256])

        # build x_net for test image
        if config.share_param:
            with tf.variable_scope('z_net', reuse=True):
                x_net = XZNet({'input': self._x_crops}, self._is_train)
        else:
            with tf.variable_scope('x_net'):
                x_net = XZNet({'input': self._x_crops}, self._is_train)

        # convolve filter with test image
        x_output = x_net.get_output()
        conv_xz = ConvXZNet({'z_gf': self._z_gf,'x_output': x_output}, self._is_train)

        activation_summary(x_output, 'activation/x_output')

        return conv_xz.get_output()

    @lazy_property
    def loss(self):
        response = self.response
        labels, weights = create_labels_overlap(np.array([self._response_size, self._response_size]), self._y_crops)
        labels = tf.reshape(labels, [-1])
        weights = tf.reshape(weights, [-1])
        response = tf.reshape(response,[-1])
        keep = tf.where(tf.not_equal(labels, -1))[:, 0]
        logits = tf.gather(response, keep)
        labels = tf.gather(labels, keep)
        weights = tf.gather(weights, keep)
        loss = tf.reduce_sum(weights*tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))/(self._batch_size*self._time_steps)
        if self._is_train:
            tf.summary.scalar('loss/cross_entropy', loss)

        return loss

    @lazy_property
    def dist_error(self):
        response = self.response
        response_shape = response.get_shape().as_list()
        max_idx = tf.argmax(tf.reshape(response, [response_shape[0], -1]), 1)
        esti_pos = tf.cast(tf.stack([max_idx % self._response_size, max_idx // self._response_size], 1), tf.float32)
        dist_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(esti_pos - tf.expand_dims(self._gt_pos, 0)), 1)))
        if self._is_train:
            tf.summary.scalar('loss/dist_error', dist_error)

        return dist_error

    @lazy_property
    def optimize(self):

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        apply_gradient_op = optimizer.apply_gradients(zip(grads, tvars), self._global_step)

        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        return train_op

    @property
    def saver(self):
        return self._saver

    @property
    def global_step(self):
        return self._global_step

    @property
    def lr(self):
        return self._lr

    @property
    def summary(self):
        return self._summary

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def z_gf(self):
        return self._z_gf
