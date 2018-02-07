import tensorflow as tf
import config
import os
from data_input.data_input import DataInput
from rfl_net.rfl_net import RFLNet
from rfl_net.utils import print_and_log

class EvalNet():
    def __init__(self):
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with tf.Graph().as_default() as g:
            # build input graph
            dataset_val = DataInput(config.batch_size_v, config.time_steps_v, False)
            batch_z, batch_x, batch_y = dataset_val.next_batch()

            self.rfl_net = RFLNet(False, batch_z, batch_x, batch_y)

            self.summary_writer = tf.summary.FileWriter(config.summaries_dir+'val', g)
            self.summary_op = tf.summary.merge_all()
            self.graph = g

    def eval_once(self):

        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with self.graph.as_default(), tf.Session(config=config_proto) as sess:
            tf.set_random_seed(1234)
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.rfl_net.saver.restore(sess, ckpt.model_checkpoint_path)
                print_and_log('Checkpoint restored from %s' % (config.checkpoint_dir))
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            coord = tf.train.Coordinator()
            enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

            totoal_dist_error = 0
            i = 0
            print_and_log('Starting validate current network......')
            while i < config.num_iterations_val:
                dist_error = sess.run(self.rfl_net.dist_error)
                totoal_dist_error += dist_error
                i += 1
                print_and_log('Examples %d dist error: %f' % (i, dist_error))

            coord.request_stop()
            coord.join(enqueue_threads)
            avg_dist_error = totoal_dist_error / i
            print_and_log('val_dist_error: %f' % (avg_dist_error))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(self.summary_op))
            summary.value.add(tag='val/dist_error', simple_value=avg_dist_error)
            self.summary_writer.add_summary(summary, global_step)

            coord.request_stop()
            coord.join(enqueue_threads)

if __name__ == "__main__":
    if tf.gfile.Exists(config.summaries_dir+'val'):
        tf.gfile.DeleteRecursively(config.summaries_dir+'val')
    tf.gfile.MakeDirs(config.summaries_dir+'val')
    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    eval_net = EvalNet()
    eval_net.eval_once()
    eval_net.summary_writer.close()
