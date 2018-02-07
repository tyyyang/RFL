import tensorflow as tf
import config
import os
import glob
import numpy as np

class DataInput():
    def __init__(self, batch_size, time_steps, is_train):
        self._batch_size = batch_size
        self._time_steps = time_steps
        self._is_train = is_train
        np.random.seed(1234)
        tf.set_random_seed(1234)
        self.shift = int((config.x_instance_size - config.z_exemplar_size) / 2)

    def next_batch(self):
        with tf.device('/cpu:0'):
            return self.batch_input()

    def batch_input(self):

        if self._is_train:
            tf_files = glob.glob(os.path.join(config.tfrecords_path, 'train-*.tfrecords'))
            filename_queue = tf.train.string_input_producer(tf_files, shuffle=True, capacity=16)

            min_queue_examples = config.min_queue_examples
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * self._batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
            enqueue_ops = []
            for _ in range(config.num_readers):
                _, value = tf.TFRecordReader().read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.add_queue_runner(
                tf.train.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            tf_files = sorted(glob.glob(os.path.join(config.tfrecords_path, 'val-*.tfrecords')))
            filename_queue = tf.train.string_input_producer(tf_files, shuffle=False, capacity=8)
            _, example_serialized = tf.TFRecordReader().read(filename_queue)
            # example_serialized = next(tf.python_io.tf_record_iterator(self._tf_files[0]))
        images_and_labels = []
        for thread_id in range(config.num_preprocess_threads):
            sequence, context = self.parse_example_proto(example_serialized)
            image_buffers = sequence['images']
            bboxes = sequence['bboxes']
            seq_len = tf.cast(context['seq_len'][0], tf.int32)
            z_exemplars, x_crops, y_crops = self.process_images(image_buffers, bboxes, seq_len, thread_id)
            images_and_labels.append([z_exemplars, x_crops, y_crops])

        batch_z, batch_x, batch_y = tf.train.batch_join(images_and_labels,
                                                             batch_size=self._batch_size,
                                                             capacity=2 * config.num_preprocess_threads * self._batch_size)
        if self._is_train:
            tf.summary.image('exemplars', batch_z[0], 5)
            tf.summary.image('crops', batch_x[0], 5)

        return batch_z, batch_x, batch_y

    def process_images(self, image_buffers, bboxes, seq_len, thread_id):
        if config.is_limit_search:
            search_range = tf.minimum(config.max_search_range, seq_len-1)
        else:
            search_range = seq_len-1
        rand_start_idx = tf.random_uniform([], 0, seq_len-search_range, dtype=tf.int32)
        frame_idxes = tf.range(rand_start_idx, rand_start_idx+search_range)
        shuffle_idxes = tf.random_shuffle(frame_idxes)
        selected_len = self._time_steps+1
        selected_idxes = shuffle_idxes[0:selected_len]
        selected_idxes, _ = tf.nn.top_k(selected_idxes, selected_len)
        selected_idxes = selected_idxes[::-1]

        z_exemplars, y_exemplars, x_crops, y_crops = [], [], [], []
        for i in range(selected_len):
            idx = selected_idxes[i]
            image_buffer = tf.gather(image_buffers, idx)
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image.set_shape([config.x_instance_size, config.x_instance_size, 3])

            # Randomly distort the colors.
            if self._is_train:
                image = self.distort_color(image, thread_id)

            if i < self._time_steps:
                # if self._is_train:
                exemplar = tf.image.crop_to_bounding_box(image, self.shift, self.shift, config.z_exemplar_size,
                                                         config.z_exemplar_size)
                if config.is_augment and i>0:
                    exemplar = self.translate_and_strech(exemplar,
                                                         [config.z_exemplar_size, config.z_exemplar_size],
                                                         config.max_strech_z)
                z_exemplars.append(exemplar)
            if i > 0:
                bbox = tf.gather(bboxes, idx)
                if self._is_train and config.is_augment:
                    image, bbox = self.translate_and_strech(image, [config.x_instance_size-2*8, config.x_instance_size-2*8],
                                                            config.max_strech_x, config.max_translate_x, bbox)
                x_crops.append(image)
                y_crops.append(bbox)
        x_crops = tf.stack(x_crops, 0)
        y_crops = tf.stack(y_crops, 0)
        z_exemplars = tf.stack(z_exemplars, 0)
        return z_exemplars, x_crops, y_crops

    def translate_and_strech(self, image, m_sz, max_strech, max_translate=None, bbox=None, rgb_variance=None):

        m_sz_f = tf.convert_to_tensor(m_sz, dtype=tf.float32)
        img_sz = tf.convert_to_tensor(image.get_shape().as_list()[0:2],dtype=tf.float32)
        scale = 1+max_strech*tf.random_uniform([2], -1, 1, dtype=tf.float32)
        scale_sz = tf.round(tf.minimum(scale*m_sz_f, img_sz))

        if max_translate is None:
            shift_range = (img_sz - scale_sz) / 2
        else:
            shift_range = tf.minimum(float(max_translate), (img_sz-scale_sz)/2)

        start = (img_sz - scale_sz)/2
        shift_row = start[0] + tf.random_uniform([1], -shift_range[0], shift_range[0], dtype=tf.float32)
        shift_col = start[1] + tf.random_uniform([1], -shift_range[1], shift_range[1], dtype=tf.float32)

        x1 = shift_col/(img_sz[1]-1)
        y1 = shift_row/(img_sz[0]-1)
        x2 = (shift_col + scale_sz[1]-1)/(img_sz[1]-1)
        y2 = (shift_row + scale_sz[0]-1)/(img_sz[0]-1)
        crop_img = tf.image.crop_and_resize(tf.expand_dims(image,0),
                                            tf.expand_dims(tf.concat(axis=0, values=[y1, x1, y2, x2]), 0),
                                            [0], m_sz)
        crop_img = tf.squeeze(crop_img)
        if rgb_variance is not None:
            crop_img = crop_img + rgb_variance*tf.random_normal([1,1,3])

        if bbox is not None:
            new_bbox = bbox - tf.concat(axis=0, values=[shift_col, shift_row, shift_col, shift_row])
            scale_ratio = m_sz_f/tf.reverse(scale_sz, [0])
            new_bbox = new_bbox*tf.tile(scale_ratio,[2])
            return crop_img, new_bbox
        else:
            return crop_img

    def distort_color(self, image, thread_id=0):
        """Distort the color of the image.
        """
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def parse_example_proto(self, example_serialized):

        context_features = {
            'seq_name': tf.FixedLenFeature([], dtype=tf.string),
            'seq_len': tf.FixedLenFeature(1, dtype=tf.int64),
            'trackid': tf.FixedLenFeature(1, dtype=tf.int64),
        }
        sequence_features = {
            'images': tf.FixedLenSequenceFeature([],dtype=tf.string),
            'bboxes': tf.FixedLenSequenceFeature([4],dtype=tf.float32)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(example_serialized, context_features, sequence_features)

        return sequence_parsed, context_parsed

