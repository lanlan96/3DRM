#!/usr/bin/python3
"""Training and Validation On Classification Task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(__file__))

import math
import random
import argparse
import importlib
from utils.data_utils import *
import pointcnn_feature.pointfly as pf
import tensorflow as tf
from datetime import datetime
import detection_model as model
import time
from utils.utils import *
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', help='Path to data')
    parser.add_argument('--path_val', '-v', help='Path to validation data')
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--path_class_weight', '-c', help='Path to validation data')
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--gpu', '-gpu', help='Setting to use', required='0')
    parser.add_argument('--area', '-a', help='Select area to test', required=True)
    parser.add_argument('--log', '-log', help='')
    parser.add_argument('--batch_size', '-b', help='', type=int)
    parser.add_argument('--debug', '-db', help='', default=False)
    parser.add_argument('--iou_threshold', '-iou', help='', type=float, default=0.5)
    parser.add_argument('--remote', '-remote', help='', default=False)
    parser.add_argument('--load_filelist_mode', help='', default=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.debug = False
    args.remote = False
    DUBUG_LOCAL_DATA = False
    TEST_TRAIN_DATA = False
    SCALE = 0.5
    print('context scale: ', SCALE)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%d_eval_area%d' % (time_string, os.getpid(), int(args.area)))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    # if not args.debug:
    #    sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')

    args.load_ckpt = '../models/4rn/pointcnn_feature_s3dis_x3_l4_rn_2021-01-02-20-23-20_63913_area1_restore/ckpts/loss_min'
    
    print(args.load_ckpt)

    print('PID:', os.getpid())

    print(args)

    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = args.batch_size or setting.batch_sizeF

    PAIR_NUM = 8
    sample_num = setting.sample_num
    point_num = 2048
    rotation_range = setting.rotation_range
    scaling_range = setting.scaling_range
    jitter = setting.jitter
    pool_setting_train = None if not hasattr(setting, 'pool_setting_train') else setting.pool_setting_train

    # Prepare inputs  
    print('{}-Preparing datasets...'.format(datetime.now()))
    sys.stdout.flush()

    path = '../data/split_for_local/'
    read_func = load_local_filelist


    test_txt_path = os.path.join(path, 'val_files_for_Area_' + args.area + '.txt')

    test_f_list = np.array(read_func(test_txt_path))

    TEST_ROOM_DICT = get_room_pkl_dict(test_f_list)

    sample_nums = 100 

    test_weight = np.ones(len(test_f_list))

    num_test = len(test_f_list)

    print("Test: ", num_test)

    sys.stdout.flush()

    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    indices_rn = tf.placeholder(tf.int32, shape=(batch_size * PAIR_NUM, None, 2), name="indices_rn") ###why is 2? what is the use?
    xforms = tf.placeholder(tf.float32, shape=(batch_size, 3, 3), name="xforms")
    xforms_rn = tf.placeholder(tf.float32, shape=(batch_size *PAIR_NUM, 3, 3), name="xforms_rn")
    rotations = tf.placeholder(tf.float32, shape=(batch_size, 3, 3), name="rotations")
    rotations_rn = tf.placeholder(tf.float32, shape=(batch_size *PAIR_NUM, 3, 3), name="rotations_rn")

    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    weight_train_placeholder = tf.placeholder(tf.float32, shape=(None), name="weight")
    ### add weight
    A_data_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size * PAIR_NUM, point_num, 6), name='A_data_train')
    B_data_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size * PAIR_NUM, point_num, 6), name='B_data_train')
    A_data_CLS_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size, point_num, 6), name='data_CLS__train')
    A_data_context_train_placeholder = tf.placeholder(tf.float32, shape=(batch_size, point_num, 6), name='data_context_train')

    # singleobj_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name='singleobj_label_train')
    cls_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name='cls_label_train')
    reg_placeholder = tf.placeholder(tf.float32, shape=(None, 8), name="weight")
    box_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 8), name="boxes")
    rn_0_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name="rn_0_label_train")
    rn_1_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name="rn_1_label_train")
    rn_2_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name="rn_2_label_train")
    rn_3_label_train_placeholder = tf.placeholder(tf.int64, shape=(None), name="rn_3_label_train")

    # Sample points and features
    print("A_data_train_placeholder_shape:",A_data_train_placeholder.shape)
    A_pts_fts_sampled = tf.gather_nd(A_data_train_placeholder, indices=indices_rn, name='pts_fts_sampled')
    A_features_augmented = None
    if setting.data_dim > 3:
        A_points_sampled, A_features_sampled = tf.split(A_pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    A_features_augmented = pf.augment(A_features_sampled, rotations_rn)
                else:
                    normals, rest = tf.split(A_features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations_rn)
                    A_features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                A_features_augmented = A_features_sampled
    else:
        A_points_sampled = A_pts_fts_sampled
    print("A_points_sampled:",A_points_sampled.shape)

    A_points_augmented = pf.augment(A_points_sampled, xforms_rn, jitter_range)
    print("A_points_augmented-shape:",A_points_augmented.shape)

    B_pts_fts_sampled = tf.gather_nd(B_data_train_placeholder, indices=indices_rn, name='pts_fts_sampled')
    B_features_augmented = None
    if setting.data_dim > 3:
        B_points_sampled, B_features_sampled = tf.split(B_pts_fts_sampled,
                                                        [3, setting.data_dim - 3],
                                                        axis=-1,
                                                        name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    B_features_augmented = pf.augment(B_features_sampled, rotations_rn)
                else:
                    normals, rest = tf.split(B_features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations_rn)
                    B_features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                B_features_augmented = B_features_sampled
    else:
        B_points_sampled = B_pts_fts_sampled
    B_points_augmented = pf.augment(B_points_sampled, xforms_rn, jitter_range)


    CLS_A_pts_fts_sampled = tf.gather_nd(A_data_CLS_train_placeholder, indices=indices, name='pts_fts_sampled')
    CLS_A_features_augmented = None
    if setting.data_dim > 3:
        CLS_A_points_sampled, CLS_A_features_sampled = tf.split(CLS_A_pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    CLS_A_features_augmented = pf.augment(CLS_A_features_sampled, rotations)
                else:
                    normals, rest = tf.split(CLS_A_features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    CLS_A_features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                CLS_A_features_augmented = CLS_A_features_sampled
    else:
        CLS_A_points_sampled = CLS_A_pts_fts_sampled
    CLS_A_points_augmented = pf.augment(CLS_A_points_sampled, xforms, jitter_range)

    # Sample points and features with context
    cont_pts_fts_sampled= tf.gather_nd(A_data_context_train_placeholder, indices=indices, name='pts_fts_sampled')
    cont_features_augmented = None
    if setting.data_dim > 3:
        cont_points_sampled, cont_features_sampled = tf.split(cont_pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    cont_features_augmented = pf.augment(cont_features_sampled, rotations)
                else:
                    c_normals, c_rest = tf.split(cont_features_sampled, [3, setting.data_dim - 6])
                    c_normals_augmented = pf.augment(c_normals, rotations)
                    cont_features_augmented = tf.concat([c_normals_augmented, c_rest], axis=-1)
            else:
                cont_features_augmented = cont_features_sampled
    else:
        cont_points_sampled = cont_pts_fts_sampled
    cont_points_augmented = pf.augment(cont_points_sampled, xforms, jitter_range)

    # Build the model
    net = model.Net(points_A=A_points_augmented, features_A=A_features_augmented,
                    points_B=B_points_augmented, features_B=B_features_augmented,
                    points_CLS_A = CLS_A_points_augmented , features_CLS_A =CLS_A_features_augmented, 
                    points_A_with_context=cont_points_augmented ,features_A_with_context=cont_features_augmented, 
                    is_training=is_training, setting=setting, input_box=box_placeholder[:, :8])

    weights_2d = tf.expand_dims(weight_train_placeholder, axis=-1, name='weights_2d')

    cls_logits = net.logits_for_cls
    cls_labels_2d = tf.expand_dims(cls_label_train_placeholder, axis=-1, name='cls_labels_2d')
    cls_labels_tile = tf.tile(cls_labels_2d, (1, tf.shape(cls_logits)[1]), name='cls_labels_tile')  #(batchs, class_num)

    rn_logits_0 = net.logits_0_for_rn
    rn_logits_1 = net.logits_1_for_rn
    rn_logits_2 = net.logits_2_for_rn
    rn_logits_3 = net.logits_3_for_rn
    rn_0_labels_2d = tf.expand_dims(rn_0_label_train_placeholder, axis=-1, name='rn_0_labels_2d')

    rn_1_labels_2d = tf.expand_dims(rn_1_label_train_placeholder, axis=-1, name='rn_1_labels_2d')

    rn_2_labels_2d = tf.expand_dims(rn_2_label_train_placeholder, axis=-1, name='rn_2_labels_2d')

    rn_3_labels_2d = tf.expand_dims(rn_3_label_train_placeholder, axis=-1, name='rn_3_labels_2d')

    rn_0_labels_tile = rn_0_labels_2d
    rn_1_labels_tile = rn_1_labels_2d
    rn_2_labels_tile = rn_2_labels_2d
    rn_3_labels_tile = rn_3_labels_2d

    # Classification and regression
    cls_probs = tf.nn.softmax(net.logits_for_cls, name='cls_obj_probs')
    cls_obj_predictions = tf.argmax(cls_probs, axis=-1, name='cls_obj_predictions', output_type=tf.int32)
    ori_cls_obj_predictions = cls_obj_predictions
    cls_obj_predictions = tf.squeeze(cls_obj_predictions)

    # Reasoning
    rn_probs_0 = net.probs_0_for_rn
    rn_predictions_0 = tf.argmax(rn_probs_0, 1)
    rn_probs_1 = net.probs_1_for_rn
    rn_predictions_1 = tf.argmax(rn_probs_1, 1)
    rn_probs_2 = net.probs_2_for_rn
    rn_predictions_2 = tf.argmax(rn_probs_2, 1)
    rn_probs_3 = net.probs_3_for_rn
    rn_predictions_3 = tf.argmax(rn_probs_3, 1)
    # rn_predictions = tf.squeeze(rn_predictions)

    # Get ops for lr and train
    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)

    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)

    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op)

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()

    # Get saver and write to ckpt or summary
    saver = tf.train.Saver(max_to_keep=None)


    # Calculate the number of parameters
    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))
    sys.stdout.flush()

    with tf.name_scope("cls_and_reg"):
        rn_loss_op, _, _, _, _, rn_acc_op, cls_loss_op, reg_loss_op, loss_op,_,_,_,_ = net.get_loss_op(cls_label=cls_labels_tile,
                                                                    weights_2d=weights_2d, reg_label=reg_placeholder[:, :8], rn0_label=rn_0_labels_tile,
                                                                    rn1_label=rn_1_labels_tile,
                                                                    rn2_label=rn_2_labels_tile,
                                                                    rn3_label=rn_3_labels_tile)
        cls_predictions = cls_obj_predictions
        cls_probs = cls_probs
        cls_acc_op = net.get_acc_op(cls_predictions, cls_label_train_placeholder)

    with tf.name_scope("reasoning"):
        rn_logits_0 = tf.sigmoid(rn_logits_0)
        rn_logits_1 = tf.sigmoid(rn_logits_1)
        rn_logits_2 = tf.sigmoid(rn_logits_2)
        rn_logits_3 = tf.sigmoid(rn_logits_3)


    # Get initialise op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()


    # Create session and start to train/test
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with sess:
        sess.run(init_op)
        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        iou_list = [0.2, 0.3, 0.4, 0.5]

        print("-------------------------------------------------------------------------------")
        print("Start to evaluate:", "       BATCH_SIZE:", batch_size)
        print("-------------------------------------------------------------------------------")
        # Test
        test_cloud_features = []
        cls_confidences_all = []
        sin_confidences_all = []

        single_label_all = []
        ious_all = []
        cls_labels_all = []
        reg_label_all = []
        obbs_all= []
        pts_all = []
        test_f_all = [ ]
        ins_all = []
        reg_pred_all = []

        start_time = time.time()
        test_cls_loss_sum = 0.0
        test_reg_loss_sum = 0.0
        test_sin_loss_sum = 0.0
        test_rn_loss_sum = 0.0

        test_loss_sum = 0.0
        test_total_seen = 0.0
        test_total_sin_correct = 0.0
        test_total_cls_correct = 0.0


        # do not shuttle
        start_time = time.time()
        test_rn_total_correct = 0
        test_rn_total_seen = 0

        # Test for one epoch
        test_batch_idx_total = 1 #ori:0
        test_batch_num_per_epoch = math.floor(num_test / batch_size)
        print("test_batch_num_per_epoch:",test_batch_num_per_epoch)
        for batch_idx_test in range(int(test_batch_num_per_epoch)):

            A_files=[]
            B_files=[]
            ids = batch_idx_test*batch_size
            A_cls_files = test_f_list[ids: ids+batch_size]

            for  A_pkl in A_cls_files:
                room_name = A_pkl.split('/')[3]
                room_pkl_list = TEST_ROOM_DICT[room_name]

                B_of_A_files = np.random.choice(room_pkl_list, size=PAIR_NUM)
                A_pair_files = [A_pkl for i in range(PAIR_NUM)]
                for each_A in A_pair_files:
                    A_files.append(each_A)
                for each_B in B_of_A_files:
                    B_files.append(each_B)


            #CLS
            A_CLS_test_data, _ = load_local_data_and_contdata(A_cls_files, args.iou_threshold, debug=DUBUG_LOCAL_DATA, scale = 0.5)
            #RN
            s1_time = time.time()
            A_test_data = {}
            A_test_data['ori_data'] = []
            A_test_data["pts_data"] = []
            A_test_data['single_object_label'] = []
            A_test_data['pps_sem_label'] = []
            A_test_data['pps_ins_label'] = []
            A_test_data['pps_obbs'] = []

            for k in range(batch_size):
                for j in range(PAIR_NUM):
                    A_test_data['ori_data'].append(A_CLS_test_data["ori_data"][k])
                    A_test_data["pts_data"].append(A_CLS_test_data["pts_data"][k])
                    A_test_data["single_object_label"].append(A_CLS_test_data["single_object_label"][k])
                    A_test_data["pps_sem_label"].append(A_CLS_test_data["pps_sem_label"][k])
                    A_test_data["pps_ins_label"].append(A_CLS_test_data["pps_ins_label"][k])
                    A_test_data["pps_obbs"].append(A_CLS_test_data["pps_obbs"][k])
            A_test_data["ori_data"] = np.array(A_test_data["ori_data"])
            A_test_data["pts_data"] = np.array(A_test_data["pts_data"])
            A_test_data['single_object_label'] = np.array(A_test_data['single_object_label'])
            A_test_data['pps_sem_label'] = np.array(A_test_data["pps_sem_label"])
            A_test_data['pps_ins_label'] = np.array(A_test_data["pps_ins_label"])
            A_test_data['pps_obbs'] = np.array(A_test_data["pps_obbs"])

            B_test_data, p_and_n_batch = load_local_data(B_files, args.iou_threshold, debug=DUBUG_LOCAL_DATA)

            A_test_data['pps_sem_label'][np.where(A_test_data['pps_sem_label'] >=12 ) ] -=1
            A_test_data['pps_sem_label'] -=1
            B_test_data['pps_sem_label'][np.where(B_test_data['pps_sem_label'] >=12 ) ] -=1
            B_test_data['pps_sem_label'] -=1


            A_CLS_test_pts = A_CLS_test_data['pts_data']
            A_CLS_ori_pts = A_CLS_test_data['ori_data']
            A_CLS_f_file = A_cls_files


            A_CLS_test_cls_label = A_CLS_test_data['pps_sem_label']
            A_CLS_test_reg_label = A_CLS_test_data['pps_obbs_reg']
            A_CLS_test_obbs_label = A_CLS_test_data['pps_obbs']

            A_CLS_test_pts_cont = A_CLS_test_data['pts_context_data']
            A_CLS_test_iou_label = A_CLS_test_data['pps_iou']
            A_CLS_test_ins_label = A_CLS_test_data['pps_ins_label']

            A_test_pts = A_test_data['pts_data']

            B_test_pts = B_test_data['pts_data']
            
            #CLS
            test_cls_label = A_CLS_test_cls_label
            test_reg_label = A_CLS_test_reg_label
            test_obbs_label = A_CLS_test_obbs_label
            A_CLS_data_batch = np.array(A_CLS_test_pts)
            A_CLS_data_cont_batch = np.array(A_CLS_test_pts_cont)

            #RN
            A_data_batch = np.array(A_test_pts)
            B_data_batch = np.array(B_test_pts)
            
            cls_label_batch = np.array(test_cls_label)
            cls_label_batch = np.array(test_cls_label)
            cls_label_batch[np.where(cls_label_batch >= 12)]-=1
            cls_label_batch = cls_label_batch-1
            reg_label_batch = np.array(test_reg_label)
            box_label_batch = np.array(test_obbs_label)

            weight_batch = np.array(test_weight[0:len(cls_label_batch)])

            A_test_data["pts_data"] = A_data_batch
            B_test_data["pts_data"] = B_data_batch

            rn_label_batch_0 = compute_rn_label_batch(A_test_data, B_test_data, 0, A_files, B_files) 

            rn_label_batch_1 = compute_rn_label_batch(A_test_data, B_test_data, 1, A_files, B_files) 

            rn_label_batch_2 = compute_rn_label_batch(A_test_data, B_test_data, 2, A_files, B_files)  

            rn_label_batch_3 = compute_rn_label_batch(A_test_data, B_test_data, 3, A_files, B_files) 

            for i in range(batch_size):
                ious_all.append(A_CLS_test_iou_label[i])
                cls_labels_all.append(cls_label_batch[i])
                reg_label_all.append(reg_label_batch[i])
                obbs_all.append(box_label_batch[i])
                pts_all.append(A_CLS_ori_pts[i])
                test_f_all.append(A_CLS_f_file[i])
                ins_all.append(A_CLS_test_ins_label[i])


            offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
            offset = max(offset, -sample_num * setting.sample_num_clip)
            offset = min(offset, sample_num * setting.sample_num_clip)
            sample_num_test = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                    rotation_range=rotation_range,
                                                    scaling_range=scaling_range,
                                                    order=setting.rotation_order)
            xforms_rn_np, rotations_rn_np = pf.get_xforms(batch_size*PAIR_NUM,
                                                    rotation_range=rotation_range,
                                                    scaling_range=scaling_range,
                                                    order=setting.rotation_order)
            # Define the ops and feed_dict
            ops = [ loss_op, cls_predictions, lr_clip_op, merged_summary, cls_acc_op, rn_acc_op,
                cls_loss_op, reg_loss_op, rn_loss_op, rn_predictions_0,rn_predictions_1,rn_predictions_2, global_step,cls_probs]
            feed_dict = {
                A_data_train_placeholder: A_data_batch,
                B_data_train_placeholder: B_data_batch,
                A_data_CLS_train_placeholder: A_CLS_data_batch,
                A_data_context_train_placeholder: A_CLS_data_cont_batch,
                box_placeholder:box_label_batch,
                indices_rn: pf.get_indices(batch_size * PAIR_NUM, sample_num_test,
                                        point_num, pool_setting_train),
                indices: pf.get_indices(batch_size, sample_num_test,
                                        point_num, pool_setting_train),
                xforms: xforms_np,
                rotations: rotations_np,
                xforms_rn: xforms_rn_np,
                rotations_rn: rotations_rn_np,
                jitter_range: np.array([jitter]),
                is_training: True,
                weight_train_placeholder: weight_batch,
                cls_label_train_placeholder: cls_label_batch,
                reg_placeholder: reg_label_batch,
                rn_0_label_train_placeholder: rn_label_batch_0,
                rn_1_label_train_placeholder: rn_label_batch_1,
                rn_2_label_train_placeholder: rn_label_batch_2,
                rn_3_label_train_placeholder: rn_label_batch_3,
            }
            ops.append(net.logits_for_reg)

            # Run the ops and get the results
            return_value = sess.run(ops, feed_dict=feed_dict)
            loss_np = return_value[0]
            cls_predictions_np = return_value[1]
            learningrate = return_value[2]
            test_summary = return_value[3]
            cls_acc_np = return_value[4]
            rn_acc_np = return_value[5]

            cls_loss_np = return_value[6]
            reg_loss_np = return_value[7]
            rn_loss_np = return_value[8]
            rn0_predictions_np = return_value[9]
            rn1_predictions_np = return_value[10]
            rn2_predictions_np = return_value[11]
            g_step = return_value[12]
            cls_probs_np =return_value[13]
            reg_pred_np = return_value[14]
            
            test_cls_loss_sum += cls_loss_np
            test_reg_loss_sum += reg_loss_np
            test_rn_loss_sum += rn_loss_np
            test_loss_sum += loss_np

            correct = np.sum(cls_predictions_np == cls_label_batch)
            rn0_correct = np.sum(rn0_predictions_np == rn_label_batch_0)
            rn1_correct = np.sum(rn1_predictions_np == rn_label_batch_1)
            rn2_correct = np.sum(rn2_predictions_np == rn_label_batch_2)
            rn_correct = (rn0_correct+rn1_correct+rn2_correct)/3

            print("cls_pred :",cls_predictions_np)
            print("cls_label:",cls_label_batch)

            for _i in range(batch_size):
                cls_confidences_all.append(np.squeeze(cls_probs_np[_i]))
                reg_pred_all.append(reg_pred_np[_i])


            test_total_cls_correct += correct
            test_rn_total_correct += rn_correct
            test_total_seen += batch_size
     
            test_rn_total_seen += len(rn_label_batch_0)


            print('{}-[Test {:02d}/{:02d}]  Loss: {:.4f}  cls_loss: {:.4f},  reg_loss: {:.4f}, '
                        'rn_loss: {:.4f},  cls_Acc: {:.4f},  rn_Acc: {:.4f},   lr:{:.8f}'
                        .format(datetime.now(), batch_idx_test + 1, int(test_batch_num_per_epoch),
                                loss_np, cls_loss_np, reg_loss_np, rn_loss_np,
                                cls_acc_np,  rn_acc_np,
                                learningrate))

        # Statistics
        mloss_test = test_loss_sum / float(test_batch_idx_total)
        macc_test = test_total_cls_correct / float(test_total_seen)
        mrn_loss_test = test_rn_loss_sum / float(test_batch_idx_total)
        mrnacc_test = test_rn_total_correct / float(test_rn_total_seen)


        mcls_test_loss = test_cls_loss_sum / float(test_batch_idx_total)
        mreg_test_loss = test_reg_loss_sum / float(test_batch_idx_total)

        end_time = time.time()
        time_for_one_epoch = end_time - start_time

        print('{}-[Test]-Done!  mLoss: {:.4f}  mcls_loss: {:.4f},  mreg_loss: {:.4f}, '
                'rn_loss: {:.4f},  Acc: {:.4f},  rn_Acc: {:.4f},   lr:{:.8f}'
                .format(datetime.now(), mloss_test, mcls_test_loss, mreg_test_loss, mrn_loss_test,
                        macc_test, mrnacc_test, learningrate))

    # Evaluation
    print("-------------------------------------------------------------------------------")
    print("Begin to eval the result ... ")
    print("-------------------------------------------------------------------------------")

    # Save the inputs and outputs to .pkl
    labels_and_scores_path = root_folder + '/labels_and_scores.pkl'

    obbs_all = trans_pred_obb(obbs_all, reg_pred_all)
    l_and_s_dict = {
        'test_f': np.array(test_f_all),
        'pts': np.array(pts_all),
        'obbs': np.array(obbs_all),
        'ious': ious_all,
        'cls_labels': np.array(cls_labels_all),
        'cls_confidences': np.array(cls_confidences_all),
        'ins_label': np.array(ins_all)
    }
    # pickleFile(l_and_s_dict, labels_and_scores_path)

    # Do NMS and eval the 4 categories
    categories = [5, 7, 8, 12]  # chair,board,table,sofa

    # NMS
    _start_time = time.time()
    pts_dict = NMS_on_points_multiprocessing(l_and_s_dict, root_folder, categories, setting.num_class)
    _end_time = time.time()
    print('NMS cost: ', _end_time-_start_time)

    # AP computation
    _start_time = time.time()
    pred_box_file = root_folder + '/pred_boxes.txt'

    for category in categories:
        print("category:",category)
        box2AP2(pts_dict, pred_box_file, category, root_folder, False, int(args.area))

    _end_time = time.time()
    print('AP cost: ', _end_time-_start_time)

    sys.stdout.flush()
    print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
