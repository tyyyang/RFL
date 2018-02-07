import os

#================= data preprocessing ==========================
# set proper paths
root_path = '/home/Data/ILSVRC'
tfrecords_path = '/home/Data/ILSVRC-TF'
otb_data_dir = '/home/Data/Benchmark/OTB'

data_path_t = os.path.join(root_path, 'Data/VID/train')
data_path_v = os.path.join(root_path, 'Data/VID/val')
anno_path_t = os.path.join(root_path, 'Annotations/VID/train/')
anno_path_v = os.path.join(root_path, 'Annotations/VID/val/')

vid_info_t = './VID_Info/vid_info_train.txt'
vid_info_v = './VID_Info/vid_info_val.txt'
vidb_t = './VID_Info/vidb_train.pk'
vidb_v = './VID_Info/vidb_val.pk'

max_trackid = 50
min_frames = 20

num_threads_t = 16
num_threads_v = 8


fix_aspect = False
if fix_aspect:
    context_amount = 0.5
else:
    z_scale = 2

#========================== data input ============================
min_queue_examples = 2500
num_readers = 4
num_preprocess_threads = 8

is_limit_search = False
max_search_range = 200

is_augment = True
max_strech_x = 0.05
max_translate_x = 4
max_strech_z = 0.5
max_translate_z = 4

overlap_thre = 0.7


#========================== RFL net ===============================
share_param = False

hidden_size = 1024
output_size = 256
num_rnn_layers = 1
max_grad_norm = 10
keep_prob = 1
conv_filter_size = 3

z_exemplar_size = 127
x_instance_size = 255

#========================== train =================================
batch_size = 10
time_steps = 10

decay_circles = 10000
lr_decay = 0.8
learning_rate = 0.0001

summaries_dir = 'output/summary/'
checkpoint_dir = 'output/models/'
pretrained_model_checkpoint_path = 'output/pre_models/'

summary_save_step_train = 500
model_save_step = 5000
validate_step = 5000
max_iterations = 100000

#========================== validation ==================================
batch_size_v = 10
time_steps_v = 10
num_example_epoch_val = 100000
num_iterations_val = int(num_example_epoch_val/batch_size_v/time_steps_v)

#========================== tracking ====================================
num_scale = 3
scale_multipler = 1.03
scale_penalty = 0.97
scale_damp = 0.6

response_up = 16
response_size = 17
window = 'cosine'
win_weights = 0.11
stride = 8
avg_num = 5

state_damp = 0.06

is_save = False
save_path = './results'


#===========special setting for cpu machine to save memory===============
if True:
    batch_size = 3
    min_queue_examples = 50
    summary_save_step_train = 1
    model_save_step = 1
    validate_step = 2
    batch_size_v = 1
    time_steps_v = 2
    min_queue_examples = 50
    num_readers = 1
    num_preprocess_threads = 1
    # hidden_size = 2
