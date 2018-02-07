import numpy as np
import config
import tensorflow as tf

DEBUG = False

def create_labels_overlap(feat_size, y_crops):
    batch_labels, batch_weights = \
        tf.py_func(create_labels_overlap_py,
                   [feat_size, tf.reshape(y_crops, [-1, 4]), (feat_size - 1)/2],
                   [tf.float32, tf.float32])
    return batch_labels, batch_weights

def create_labels_overlap_py(feat_size, y_crops, orgin, random_select=False):
    orig_size = feat_size*config.stride
    x = np.arange(0, orig_size[0], config.stride)+config.stride/2
    y = np.arange(0, orig_size[1], config.stride)+config.stride/2
    x, y = np.meshgrid(x, y)
    orgin = orgin*config.stride + config.stride/2
    batch_labels, batch_weights, batch_keep  = [], [], []
    for gt_bb_cur in y_crops:
        gt_size_cur = gt_bb_cur[2:4] - gt_bb_cur[0:2] + 1
        gt_bb_cur_new = np.hstack([orgin - (gt_size_cur - 1) / 2, orgin + (gt_size_cur - 1) / 2])
        sample_centers = np.vstack([x.ravel(), y.ravel(), x.ravel(), y.ravel()]).transpose()
        sample_bboxes = sample_centers + np.hstack([-(gt_size_cur-1)/2, (gt_size_cur-1)/2])

        overlaps = bbox_overlaps(sample_bboxes, gt_bb_cur_new)

        pos_idxes = overlaps > config.overlap_thre
        neg_idxes = overlaps < config.overlap_thre
        labels = -np.ones(np.prod(feat_size), dtype=np.float32)
        labels[pos_idxes] = 1
        labels[neg_idxes] = 0
        labels = np.reshape(labels, feat_size)

        num_pos = np.count_nonzero(labels == 1)
        num_neg = np.count_nonzero(labels == 0)

        if DEBUG:
            print(gt_bb_cur)
            print((gt_bb_cur[0:2]+gt_bb_cur[2:4])/2)
            print('Positive samples:', num_pos, 'Negative samples:', num_neg)

        weights = np.zeros(feat_size, dtype=np.float32)
        if num_pos != 0:
            weights[labels == 1] = 0.5 / num_pos
        if num_neg != 0:
            weights[labels == 0] = 0.5 / num_neg
        batch_weights.append(np.expand_dims(weights, 0))
        batch_labels.append(np.expand_dims(labels, 0))

    batch_labels = np.concatenate(batch_labels, 0)
    batch_weights = np.concatenate(batch_weights, 0)
    return batch_labels, batch_weights

def bbox_overlaps(sample_bboxes, gt_bbox):
    lt = np.maximum(sample_bboxes[:, 0:2], gt_bbox[0:2])
    rb = np.minimum(sample_bboxes[:, 2:4], gt_bbox[2:4])
    inter_area = np.maximum(rb - lt + 1, 0)
    inter_area = np.prod(inter_area, 1)
    union_area = np.prod(sample_bboxes[:, 2:4] - sample_bboxes[:, 0:2] + 1, 1) + np.prod(gt_bbox[2:4]-gt_bbox[0:2]+1, 0) - inter_area
    return inter_area / union_area


if __name__ == '__main__':
    feat_size = np.array([255, 255])
    y_bboxes = np.array([[100, 100, 155, 155], [15,15, 50, 100], [15,15, 100, 100]])
    batch_labels, batch_cls_w = create_labels_overlap_py(feat_size, y_bboxes, np.array([128, 128]), True)