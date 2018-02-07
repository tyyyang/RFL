import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import config
from rfl_net.rfl_net import RFLNet
import math
import collections
LSTMState = collections.namedtuple('LSTMState',['c','h'])


class RFLearner():
    def __init__(self, sess):

        self.z_file_init = tf.placeholder(tf.string, [], name='z_filename_init')
        self.z_roi_init = tf.placeholder(tf.float32, [1, 4], name='z_roi_init')
        self.z_file = tf.placeholder(tf.string, [], name='z_filename')
        self.z_roi = tf.placeholder(tf.float32, [1, 4], name='z_roi')
        self.x_file = tf.placeholder(tf.string, [], name='x_filename')
        self.x_roi = tf.placeholder(tf.float32, [config.num_scale, 4], name='x_roi')

        init_z_exemplar,_ = self._read_and_crop_image(self.z_file_init, self.z_roi_init, [config.z_exemplar_size, config.z_exemplar_size])
        init_z_exemplar = tf.reshape(init_z_exemplar, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        z_exemplar,_ = self._read_and_crop_image(self.z_file, self.z_roi, [config.z_exemplar_size, config.z_exemplar_size])
        z_exemplar = tf.reshape(z_exemplar, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        self.x_instances, self.image = self._read_and_crop_image(self.x_file, self.x_roi, [config.x_instance_size, config.x_instance_size])
        self.x_instances = tf.reshape(self.x_instances, [config.num_scale, 1, config.x_instance_size, config.x_instance_size, 3])

        self._rfl_net = RFLNet(False, z_examplar=z_exemplar, x_crops=self.x_instances, init_z_exemplar=init_z_exemplar)

        up_response_size = config.response_size * config.response_up
        self.up_response = tf.squeeze(tf.image.resize_images(tf.expand_dims(self._rfl_net.response, -1),
                                                             [up_response_size, up_response_size],
                                                             method=tf.image.ResizeMethod.BICUBIC,
                                                             align_corners=True), -1)

        ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self._rfl_net.saver.restore(sess, ckpt.model_checkpoint_path)
            self._sess = sess

    def _read_and_crop_image(self, filename, roi, model_sz):
        image_file = tf.read_file(filename)
        # Decode the image as a JPEG file, this will turn it into a Tensor
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        frame_sz = tf.shape(image)
        # used to pad the crops
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')
        # pad with if necessary
        frame_padded, npad = self._pad_frame(image, frame_sz, roi, avg_chan)
        frame_padded = tf.cast(frame_padded, tf.float32)
        crop_patch = self._crop_image(frame_padded, npad, frame_sz, roi, model_sz)
        return crop_patch, image

    def _pad_frame(self, im, frame_sz, roi, avg_chan):
        pos_x = tf.reduce_max(roi[:, 0], axis=0)
        pos_y = tf.reduce_max(roi[:, 1], axis=0)
        patch_sz = tf.reduce_max(roi[:, 2:4], axis=0)
        c = patch_sz / 2
        xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c[0]), tf.int32))
        ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c[1]), tf.int32))
        xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c[0]), tf.int32) - frame_sz[1])
        ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c[1]), tf.int32) - frame_sz[0])
        npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
        paddings = [[npad, npad], [npad, npad], [0, 0]]
        im_padded = im
        if avg_chan is not None:
            im_padded = im_padded - avg_chan
        im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
        if avg_chan is not None:
            im_padded = im_padded + avg_chan
        return im_padded, npad

    def _crop_image(self, im, npad, frame_sz, rois, model_sz):
        radius = (rois[:, 2:4]-1) / 2
        c_xy = rois[:, 0:2]
        self.pad_frame_sz = pad_frame_sz = tf.cast(tf.expand_dims(frame_sz[0:2]+2*npad,0), tf.float32)
        npad = tf.cast(npad, tf.float32)
        xy1 = (npad + c_xy - radius)
        xy2 = (npad + c_xy + radius)
        norm_rect = tf.stack([xy1[:,1], xy1[:,0], xy2[:,1], xy2[:,0]], axis=1)/tf.concat([pad_frame_sz, pad_frame_sz],1)
        crops = tf.image.crop_and_resize(tf.expand_dims(im, 0), norm_rect, tf.zeros([tf.shape(rois)[0]],tf.int32), model_sz, method='bilinear')

        return crops


class RFLTracker():

    def __init__(self, rfl):

        self._rfl_net = rfl._rfl_net
        self._rfl_leaner = rfl
        self._sess = rfl._sess
        self.idx = 1
        self.state_damp_factor = config.state_damp

        # prepare constant things for tracking
        scale_steps = list(range(math.ceil(config.num_scale / 2) - config.num_scale, math.floor(config.num_scale / 2) + 1))
        self.scales = np.power(config.scale_multipler, scale_steps)

        up_response_size = config.response_size * config.response_up
        if config.window == 'cosine':
            window = np.matmul(np.expand_dims(np.hanning(up_response_size), 1),
                               np.expand_dims(np.hanning(up_response_size), 0)).astype(np.float32)
        else:
            window = np.ones([up_response_size, up_response_size], dtype=np.float32)
        self.window = window / np.sum(window)

    def estimate_bbox(self, responses, x_roi_size_origs, target_pos, target_size):

        up_response_size = config.response_size * config.response_up
        current_scale_idx = math.floor(config.num_scale / 2)
        best_scale_idx = current_scale_idx
        best_peak = -math.inf

        for s_idx in range(config.num_scale):
            this_response = responses[s_idx].copy()

            # penalize the change of scale
            if s_idx != current_scale_idx:
                this_response *= config.scale_penalty
            this_peak = np.max(this_response)
            if this_peak > best_peak:
                best_peak = this_peak
                best_scale_idx = s_idx
        response = responses[best_scale_idx]

        x_roi_size_orig = x_roi_size_origs[best_scale_idx]

        # make response sum to 1
        response -= np.min(response)
        response /= np.sum(response)

        # apply window
        response = (1 - config.win_weights) * response + config.win_weights * self.window

        max_idx = np.argsort(response.flatten())
        max_idx = max_idx[-config.avg_num:]

        x = max_idx % up_response_size
        y = max_idx // up_response_size
        position = np.vstack([x, y]).transpose()

        shift_center = position - up_response_size / 2
        shift_center_instance = shift_center * config.stride / config.response_up
        shift_center_orig = shift_center_instance * np.expand_dims(x_roi_size_orig, 0) / config.x_instance_size
        target_pos = np.mean(target_pos + shift_center_orig, 0)

        target_size_new = target_size * self.scales[best_scale_idx]
        target_size = (1 - config.scale_damp) * target_size + config.scale_damp * target_size_new

        return target_pos, target_size, best_scale_idx

    def initialize(self, init_frame_file, init_box):
        bbox = np.array(init_box)
        self.target_pos = bbox[0:2] + bbox[2:4] / 2
        self.target_size = bbox[2:4]

        self.z_roi_size = calc_z_size(self.target_size)
        self.x_roi_size = calc_x_size(self.z_roi_size)
        z_roi = np.concatenate([self.target_pos, self.z_roi_size], 0)
        next_state, init_gf = self._sess.run(self._rfl_net.init_state_filter,
                                             {self._rfl_leaner.z_file_init: init_frame_file,
                                              self._rfl_leaner.z_roi_init: [z_roi]})
        self.next_state = next_state

        self.z_gf = np.tile(init_gf, [config.num_scale, 1, 1, 1])

    def track(self, cur_frame_file):
        # build pyramid of search images
        sx_roi_size = np.round(np.expand_dims(self.x_roi_size, 0) * np.expand_dims(self.scales, 1))
        target_poses = np.tile(np.expand_dims(self.target_pos,axis=0), [config.num_scale,1])
        x_rois = np.concatenate([target_poses, sx_roi_size], axis=1)
        responses, cur_frame, x_instances = self._sess.run([self._rfl_leaner.up_response, self._rfl_leaner.image, self._rfl_leaner.x_instances],
                                                           {self._rfl_leaner.x_file: cur_frame_file,
                                    self._rfl_leaner.x_roi: x_rois,
                                    self._rfl_net._z_gf: self.z_gf})

        # estimate position and size
        self.target_pos, self.target_size, best_scale_idx = self.estimate_bbox(responses, sx_roi_size, self.target_pos, self.target_size)
        bbox = np.hstack([self.target_pos - self.target_size / 2, self.target_size])

        # crop exemplar to generate new filter
        self.z_roi_size = calc_z_size(self.target_size)
        self.x_roi_size = calc_x_size(self.z_roi_size)

        z_roi = np.concatenate([self.target_pos, self.z_roi_size], 0)
        z_gf, next_state_new = self._sess.run([self._rfl_net.filter,
                                               self._rfl_net.final_state],
                                              { self._rfl_leaner.z_file: cur_frame_file,
                                                self._rfl_leaner.z_roi: [z_roi],
                                                self._rfl_net.initial_state: self.next_state})
        # damp state
        self.next_state = damp_state(self.next_state, next_state_new, self.state_damp_factor)

        self.z_gf = np.tile(z_gf, [config.num_scale, 1, 1, 1])

        return bbox, cur_frame


def calc_z_size(target_size):
    # calculate roi region
    if config.fix_aspect:
        extend_size = target_size + config.context_amount * (target_size[0] + target_size[1])
        z_size = np.sqrt(np.prod(extend_size))
        z_size = np.repeat(z_size, 2, 0)
    else:
        z_size = target_size * config.z_scale

    return z_size

def calc_x_size(z_roi_size):
    # calculate roi region
    z_scale = config.z_exemplar_size / z_roi_size
    delta_size = config.x_instance_size - config.z_exemplar_size
    x_size = delta_size / z_scale + z_roi_size

    return x_size

def damp_state(next_state, next_state_new, state_damp_factor):
    c = next_state[0]
    h = next_state[1]
    c_n = next_state_new[0]
    h_n = next_state_new[1]
    c = (1 - state_damp_factor) * c + state_damp_factor * c_n
    h = (1 - state_damp_factor) * h + state_damp_factor * h_n
    one_state = tuple([c, h])
    return one_state