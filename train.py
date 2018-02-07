import tensorflow as tf
import config
from data_input.data_input import DataInput
from rfl_net.rfl_net import RFLNet
from eval import EvalNet
from datetime import datetime
import time
import os
from rfl_net.utils import print_and_log

def train():
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        tf.set_random_seed(1234)

        # build input graph
        dataset_train = DataInput(config.batch_size, config.time_steps, True)
        batch_z, batch_x, batch_y = dataset_train.next_batch()

        rfl_net = RFLNet(True, batch_z, batch_x, batch_y)

        eval_net = EvalNet()

        summary_writer = tf.summary.FileWriter(config.summaries_dir+'train', sess.graph)

        tf.global_variables_initializer().run()


        # load previous checkpoints if any
        ckpt = tf.train.get_checkpoint_state(config.pretrained_model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            rfl_net.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Pretrained checkpoint restored from %s' %
                  (config.pretrained_model_checkpoint_path))

        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

        idx = sess.run(rfl_net.global_step) + 1
        while not coord.should_stop() and idx <= config.max_iterations:
            start_time = time.time()
            print_and_log("%s\nCycle: %d Learning rate: %.2e" %
                          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), idx, sess.run(rfl_net.lr)))

            if idx % config.summary_save_step_train == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_t, loss, dist_error, _ = sess.run(
                    [rfl_net.summary, rfl_net.loss, rfl_net.dist_error, rfl_net.optimize],
                    options=run_options,
                    run_metadata=run_metadata)

                summary_writer.add_run_metadata(run_metadata, 'step%03d' % idx)
                summary_writer.add_summary(summary_t, idx)
                print_and_log('Adding run metadata for', idx)
            else:
                loss, dist_error, _ = sess.run([rfl_net.loss, rfl_net.dist_error, rfl_net.optimize])

            print_and_log("Loss: %f, Dist error: %f  Speed: %.0f examples per second" %
                              (loss, dist_error, config.batch_size * config.time_steps / (time.time() - start_time)))

            if idx % config.model_save_step == 0 or idx == config.max_iterations or idx % config.validate_step == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, 'model.ckpt')
                rfl_net.saver.save(sess, checkpoint_path, global_step=idx, write_meta_graph=False)
                print_and_log('Save to checkpoint at step %d' % (idx))

            if idx % config.validate_step == 0:
                eval_net.eval_once()

            idx = sess.run(rfl_net.global_step) + 1

        summary_writer.close()
        eval_net.summary_writer.close()
        coord.request_stop()
        coord.join(enqueue_threads)

if __name__ == "__main__":

    if tf.gfile.Exists(config.summaries_dir):
        tf.gfile.DeleteRecursively(config.summaries_dir)
    tf.gfile.MakeDirs(config.summaries_dir)
    if tf.gfile.Exists('output/log.txt'):
        tf.gfile.Remove('output/log.txt')
    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    train()

