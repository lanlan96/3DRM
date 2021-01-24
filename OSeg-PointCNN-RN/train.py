#!/usr/bin/python3
"""Training and Validation On Classification Task.

    Train data of one room
"""
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
from utils import *
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


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
    args.debug = (True if "True" in args.debug else False)
    args.remote = (True if "True" in args.remote else False)
    FILTER_DATA_WITH_IOU = False  #ori: True
    ONLY_TRAIN_RN = False
    DUBUG_LOCAL_DATA = False
    LOAD_RELATION_PAIRS = False
    RELATION = 0 # same instance
    global_scale = 0.5

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%s_%d_area%d' % (args.model, args.setting, time_string, os.getpid(), int(args.area)))
    # root_folder = os.path.join(args.save_folder, '%s_%s_%s_area%d' % (args.model, args.setting, time_string,  int(args.area)))

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    args.load_ckpt = None


    if not args.debug:
        sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')

    print('PID:', os.getpid())

    print(args)

    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = args.batch_size or setting.batch_sizeF
    print(batch_size)
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

    train_txt_path = os.path.join(path, 'train_files_for_Area_' + args.area + '.txt')
    test_txt_path = os.path.join(path, 'val_files_for_Area_' + args.area + '.txt')

    train_f_list = np.array(read_func(train_txt_path))
    # train_f_list = train_f_list[:64]
    test_f_list = np.array(read_func(test_txt_path))
    # test_f_list = test_f_list[:64]

    # sem_label_root = "/home/lthpc/lyq/3DRN/data/split_for_local_stage2/" 
    sem_label_root = path
    sem_label_txt = os.path.join(sem_label_root,"train_AUG_sem_label_stage2_Area_"+args.area+".txt")
    test_sem_label_txt = os.path.join(sem_label_root,"test_sem_label_stage2_Area_"+args.area+".txt") #有没有没关系，只要有这个文件就行
    train_sem_labels = np.loadtxt(sem_label_txt , dtype=np.int32)
    test_sem_labels = np.loadtxt(test_sem_label_txt , dtype=np.int32)

    # # 计算cls 类别weight global_weights是按类别分的0:1.0, 1:2.0,...,num_cls-1:2.0
    train_weight, train_global_weights = compute_class_weight(train_sem_labels)

    print("train_global_weights:",train_global_weights)

    if FILTER_DATA_WITH_IOU:
        ious = load_local_data(train_f_list, 0.5, 1, True, debug=DUBUG_LOCAL_DATA)
        index = np.where(ious >= 0.5)[0]
        print("train iou >= 0.5   ", len(index))
        train_f_list = train_f_list[index]

        ious = load_local_data(test_f_list, 0.5, 1, True, debug=DUBUG_LOCAL_DATA)
        index = np.where(ious >= 0.5)[0]
        print("test iou >= 0.5   ", len(index))
        test_f_list = test_f_list[index]


    # train_room_ids, train_uniq_room, objects_in_each_train_room, room_dict = count_rooms(train_f_list)
    # test_room_ids, test_uniq_room, objects_in_each_test_room, test_room_dict = count_rooms(test_f_list)

    ROOM_DICT = get_room_pkl_dict(train_f_list)
    TEST_ROOM_DICT = get_room_pkl_dict(test_f_list)

    test_weight = np.ones(len(test_f_list))

    num_train = len(train_f_list)
    num_test = len(test_f_list)

    print("Train/Test: ", num_train, "/", num_test)
    print('{}-{:d} training samples.'.format(datetime.now(), num_train))
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
    # Calculate batch number for per epoch
    # train_batch_num_per_epoch = np.sum(np.array(objects_in_each_train_room/batch_size).astype(np.int))
    # test_batch_num_per_epoch = np.sum(np.array(objects_in_each_test_room/batch_size).astype(np.int))

    # print('{}-{:d} training batches per_epoch.'.format(datetime.now(), train_batch_num_per_epoch))
    # sys.stdout.flush()

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
    #ori
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

    # Get ops for lr and train
    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)

    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)

    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op)

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()

    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)

    # Get saver and write to ckpt or summary
    saver = tf.train.Saver(max_to_keep=None)

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    # Calculate the number of parameters
    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))
    sys.stdout.flush()

    with tf.name_scope("cls_and_reg"):
        rn_loss_op,rn0_loss_op,rn1_loss_op,rn2_loss_op, rn3_loss_op,rn_acc_op, cls_loss_op, reg_loss_op,loss_op,rn0_acc_op,rn1_acc_op,rn2_acc_op,rn3_acc_op = net.get_loss_op( cls_label=cls_labels_tile,
                                                                    weights_2d=weights_2d, reg_label=reg_placeholder[:, :8], rn0_label=rn_0_labels_tile,
                                                                    rn1_label=rn_1_labels_tile,rn2_label=rn_2_labels_tile,rn3_label=rn_3_labels_tile)
        cls_predictions = cls_obj_predictions
        cls_probs = cls_probs
        cls_acc_op = net.get_acc_op(cls_predictions, cls_label_train_placeholder)

    with tf.name_scope("reasoning"):
        # Get binary focal loss
        rn_logits_0 = tf.sigmoid(rn_logits_0)
        rn_logits_1 = tf.sigmoid(rn_logits_1)
        rn_logits_2 = tf.sigmoid(rn_logits_2)
        rn_logits_3 = tf.sigmoid(rn_logits_3)
    # Get train op
    train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)
    # Get initialise op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Create session and start to train/test
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with tf.Session() as sess:
        sess.run(init_op)
        # Load the model
        if args.load_ckpt is not None:
            ckpt = tf.train.latest_checkpoint(args.load_ckpt)
            saver.restore(sess, ckpt)  # 加载所有的参数
            # saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        print('total-[Train]-Iter: ', num_epochs)
        sys.stdout.flush()

        train_writer = tf.summary.FileWriter(folder_summary + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(folder_summary + '/test', sess.graph)

        train_loss_history = []
        test_loss_history = []

        iou_list = [0.2, 0.3, 0.4, 0.5]
        loss_min = 100000

        print("-------------------------------------------------------------------------------")
        print("Start to train:", "       BATCH_SIZE:", batch_size)
        print("-------------------------------------------------------------------------------")
        for epoch_idx_train in range(num_epochs):
            total_correct = 0
            rn_total_seen = 0
            total_seen = 0
            loss_sum = 0
            cls_loss_sum = 0
            reg_loss_sum = 0
            rn_loss_sum = 0
            rn0_loss_sum = 0
            rn1_loss_sum = 0
            rn2_loss_sum = 0
            rn3_loss_sum = 0

            rn_total_correct = 0
            rn0_total_correct = 0
            rn1_total_correct = 0
            rn2_total_correct = 0
            rn3_total_correct = 0

            start_time = time.time()

            # Do shuttle for room ids
            num_train = len(train_f_list)
            index_ch = np.arange(num_train)
            np.random.shuffle(index_ch)

            # Train for one epoch
            batch_idx_total = 0
            # for room in ROOM_DICT: 
            p_and_n_count = [0, 0]

            #修改train_batch_num_per_epoch
            train_batch_num_per_epoch = math.floor(num_train / batch_size)
            # Train for one epoc
            for batch_idx_train in range(int(train_batch_num_per_epoch)):

                A_files=[]
                B_files=[]
                ids = batch_idx_train*batch_size
                A_cls_files = train_f_list[index_ch[ids: ids+batch_size]]

                for  A_pkl in A_cls_files:
                    room_name = A_pkl.split('/')[3]
                    room_pkl_list = ROOM_DICT[room_name]

                    B_of_A_files = np.random.choice(room_pkl_list, size=PAIR_NUM)
                    A_pair_files = [A_pkl for i in range(PAIR_NUM)]
                    for each_A in A_pair_files:
                        A_files.append(each_A)
                    for each_B in B_of_A_files:
                        B_files.append(each_B)
     
                #CLS
                A_CLS_train_data, _ = load_local_data_and_contdata(A_cls_files, args.iou_threshold, debug=DUBUG_LOCAL_DATA,scale = global_scale)
                #RN
                s1_time = time.time()
                A_train_data = {}
                A_train_data['ori_data'] = []
                A_train_data["pts_data"] = []
                A_train_data['single_object_label'] = []
                A_train_data['pps_sem_label'] = []
                A_train_data['pps_ins_label'] = []
                A_train_data['pps_obbs'] = []

                for k in range(batch_size):
                    for j in range(PAIR_NUM):
                        A_train_data['ori_data'].append(A_CLS_train_data["ori_data"][k])
                        A_train_data["pts_data"].append(A_CLS_train_data["pts_data"][k])
                        A_train_data["single_object_label"].append(A_CLS_train_data["single_object_label"][k])
                        A_train_data["pps_sem_label"].append(A_CLS_train_data["pps_sem_label"][k])
                        A_train_data["pps_ins_label"].append(A_CLS_train_data["pps_ins_label"][k])
                        A_train_data["pps_obbs"].append(A_CLS_train_data["pps_obbs"][k])
                A_train_data["ori_data"] = np.array(A_train_data["ori_data"])
                A_train_data["pts_data"] = np.array(A_train_data["pts_data"])
                A_train_data['single_object_label'] = np.array(A_train_data['single_object_label'])
                A_train_data['pps_sem_label'] = np.array(A_train_data["pps_sem_label"])
                A_train_data['pps_ins_label'] = np.array(A_train_data["pps_ins_label"])
                A_train_data['pps_obbs'] = np.array(A_train_data["pps_obbs"])

                B_train_data, p_and_n_batch = load_local_data(B_files, args.iou_threshold, debug=DUBUG_LOCAL_DATA)

                A_train_data['pps_sem_label'][np.where(A_train_data['pps_sem_label'] >=12 ) ] -=1
                A_train_data['pps_sem_label'] -=1
                B_train_data['pps_sem_label'][np.where(B_train_data['pps_sem_label'] >=12 ) ] -=1
                B_train_data['pps_sem_label'] -=1

                A_CLS_train_pts = A_CLS_train_data['pts_data']
                A_CLS_train_cls_label = A_CLS_train_data['pps_sem_label']
                A_CLS_train_reg_label = A_CLS_train_data['pps_obbs_reg']
                A_CLS_train_obbs_label = A_CLS_train_data['pps_obbs']
                A_CLS_train_pts_cont = A_CLS_train_data['pts_context_data']

                A_train_pts = A_train_data['pts_data']

                B_train_pts = B_train_data['pts_data']
                
                #CLS
                train_cls_label = A_CLS_train_cls_label
                train_reg_label = A_CLS_train_reg_label
                train_obbs_label = A_CLS_train_obbs_label
                A_CLS_data_batch = np.array(A_CLS_train_pts)
                A_CLS_data_cont_batch = np.array(A_CLS_train_pts_cont)

                #RN
                A_data_batch = np.array(A_train_pts)
                B_data_batch = np.array(B_train_pts)

                cls_label_batch = np.array(train_cls_label)
                cls_label_batch = np.array(train_cls_label)
                cls_label_batch[np.where(cls_label_batch >= 12)]-=1
                cls_label_batch = cls_label_batch-1
                reg_label_batch = np.array(train_reg_label)
                box_label_batch = np.array(train_obbs_label)

                weight_batch = np.array(train_weight[index_ch[ids: ids+batch_size]])

                e1_time = time.time()

                rn_label_batch_0 = compute_rn_label_batch(A_train_data, B_train_data, 0, A_files, B_files) 

                rn_label_batch_1 = compute_rn_label_batch(A_train_data, B_train_data, 1, A_files, B_files) 

                rn_label_batch_2 = compute_rn_label_batch(A_train_data, B_train_data, 2, A_files, B_files)  

                rn_label_batch_3 = compute_rn_label_batch(A_train_data, B_train_data, 3, A_files, B_files)  

                p_and_n_count[0] += len(np.where(rn_label_batch_0 == 1)[0])
                p_and_n_count[1] += len(np.where(rn_label_batch_0 == 0)[0])

                offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
                offset = max(offset, -sample_num * setting.sample_num_clip)
                offset = min(offset, sample_num * setting.sample_num_clip)
                sample_num_train = sample_num + offset
                xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                        rotation_range=rotation_range,
                                                        scaling_range=scaling_range,
                                                        order=setting.rotation_order)
                xforms_rn_np, rotations_rn_np = pf.get_xforms(batch_size*PAIR_NUM,
                                                        rotation_range=rotation_range,
                                                        scaling_range=scaling_range,
                                                        order=setting.rotation_order)
                # Define the ops and feed_dict
                ops = [train_op, loss_op, cls_predictions, lr_clip_op, merged_summary, cls_acc_op, rn_acc_op,
                    cls_loss_op, reg_loss_op, rn_loss_op, rn_predictions_0,rn_predictions_1,rn_predictions_2, global_step,cls_probs,
                    rn0_loss_op,rn1_loss_op,rn2_loss_op]
                feed_dict = {
                    A_data_train_placeholder: A_data_batch,
                    B_data_train_placeholder: B_data_batch,
                    A_data_CLS_train_placeholder: A_CLS_data_batch,
                    A_data_context_train_placeholder: A_CLS_data_cont_batch,
                    box_placeholder:box_label_batch,
                    indices_rn: pf.get_indices(batch_size * PAIR_NUM, sample_num_train,
                                            point_num, pool_setting_train),
                    indices: pf.get_indices(batch_size, sample_num_train,
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
                ops.append(net.concat_rn_fc_mean)
                ops.append(net.concat_rn_mlp)
                ops.append(net.xconv_feature)
                ops.append(rn0_acc_op)
                ops.append(rn1_acc_op)
                ops.append(rn2_acc_op)

                ops.append(rn3_loss_op)
                ops.append(rn_predictions_3) 
                ops.append(rn3_acc_op) 
                # Run the ops and get the results
                return_value = sess.run(ops, feed_dict=feed_dict)
                loss_np = return_value[1]
                cls_predictions_np = return_value[2]
                learningrate = return_value[3]
                summary = return_value[4]
                cls_acc_np = return_value[5]
                rn_acc_np = return_value[6]

                cls_loss_np = return_value[7]
                reg_loss_np = return_value[8]
                rn_loss_np = return_value[9]
                rn0_predictions_np = return_value[10]
                rn1_predictions_np = return_value[11]
                rn2_predictions_np = return_value[12]
                g_step = return_value[13]
                cls_probs_np =return_value[14]
                rn0_loss_np = return_value[15]
                rn1_loss_np = return_value[16]
                rn2_loss_np = return_value[17]
                concat_rn_fc_mean = return_value[18]
                concat_rn_mlp = return_value[19]
                xconv_feature = return_value[20]
                rn0_acc_np = return_value[21]
                rn1_acc_np = return_value[22]
                rn2_acc_np = return_value[23]
                rn3_loss_np = return_value[24]
                rn3_predictions_np = return_value[25]
                rn3_acc_np = return_value[26]
                # Count the right nums
                correct = np.sum(cls_predictions_np == cls_label_batch)
                rn0_correct = np.sum(rn0_predictions_np == rn_label_batch_0)
                rn1_correct = np.sum(rn1_predictions_np == rn_label_batch_1)
                rn2_correct = np.sum(rn2_predictions_np == rn_label_batch_2)
                rn3_correct = np.sum(rn3_predictions_np == rn_label_batch_3)
                rn_correct = (rn0_correct+rn1_correct+rn2_correct+rn3_correct)/4

                rn_total_correct += rn_correct
                rn0_total_correct += rn0_correct
                rn1_total_correct += rn1_correct
                rn2_total_correct += rn2_correct
                rn3_total_correct += rn3_correct
                total_correct += correct

                total_seen += batch_size
                rn_total_seen += len(rn_label_batch_0)

                loss_sum += loss_np
                cls_loss_sum += cls_loss_np
                reg_loss_sum += reg_loss_np
                rn_loss_sum += rn_loss_np
                rn0_loss_sum += rn0_loss_np
                rn1_loss_sum += rn1_loss_np
                rn2_loss_sum += rn2_loss_np
                rn3_loss_sum += rn3_loss_np

                # Add to summary
                train_step = epoch_idx_train * train_batch_num_per_epoch + batch_idx_train
                train_writer.add_summary(summary, train_step)


             
                print('{}-[Train {:02d}/{:02d}]-Epoch: {:06d}  Loss: {:.4f}   cls_loss: {:.4f},   reg_loss: {:.4f},   '
                    'rn_loss: {:.4f}, rn0_loss: {:.4f}, rn1_loss: {:.4f}, rn2_loss: {:.4f}, rn3_loss: {:.4f},  cls_Acc: {:.4f},  rn_Acc: {:.4f},rn0_Acc: {:.4f},   rn1_Acc: {:.4f},   rn2_Acc: {:.4f}, rn3_Acc: {:.4f},   lr:{:.8f}'
                    .format(datetime.now(), batch_idx_total+1, int(train_batch_num_per_epoch), epoch_idx_train,
                            loss_np, cls_loss_np, reg_loss_np, rn_loss_np,
                            rn0_loss_np,rn1_loss_np,rn2_loss_np,rn3_loss_np,
                            cls_acc_np,  rn_acc_np,rn0_acc_np,rn1_acc_np,rn2_acc_np,rn3_acc_np,
                            learningrate))
                batch_idx_total = batch_idx_total + 1
            print("[TRAIN] P/N: ", p_and_n_count[0], "/", p_and_n_count[1])
            end_time = time.time()
            time_for_one_epoch = end_time - start_time
            # Compute the mean loss and mean accuracy for one epoch
            mloss = loss_sum / float(batch_idx_total)
            macc = total_correct / float(total_seen)
            mrn_acc = rn_total_correct / float(rn_total_seen)
            mrn0_acc = rn0_total_correct / float(rn_total_seen)
            mrn1_acc = rn1_total_correct / float(rn_total_seen)
            mrn2_acc = rn2_total_correct / float(rn_total_seen)
            mrn3_acc = rn3_total_correct / float(rn_total_seen)

            mcls_loss = cls_loss_sum / float(batch_idx_total)
            mreg_loss = reg_loss_sum / float(batch_idx_total)
            mrn_loss = rn_loss_sum / float(batch_idx_total)
            mrn0_loss = rn0_loss_sum / float(batch_idx_total)
            mrn1_loss = rn1_loss_sum / float(batch_idx_total)
            mrn2_loss = rn2_loss_sum / float(batch_idx_total)
            mrn3_loss = rn3_loss_sum / float(batch_idx_total)
            #save loss_min_ckpt
            if mloss < loss_min:
                filename_ckpt = os.path.join(folder_ckpt, 'loss_min')
                saver.save(sess, filename_ckpt)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))
            #save latest ckpt
            filename_ckpt = os.path.join(folder_ckpt, 'latest')
            saver.save(sess, filename_ckpt)
            print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))
            sys.stdout.flush()

            print('{}-[Train]-Epoch: {:06d}  mLoss: {:.4f}   mCLoss: {:.4f},   mRLoss: {:.4f},   '
                    'mRNLoss: {:.4f}, mRN0Loss: {:.4f},  mRN1Loss: {:.4f},  mRN2Loss: {:.4f},  mRN3Loss: {:.4f},   mAcc: {:.4f},   mrnAcc: {:.4f},   mrn0Acc: {:.4f},   mrn1Acc: {:.4f},   mrn2Acc: {:.4f},  mrn3Acc: {:.4f},  lr:{:.8f}'
                    .format(datetime.now(), epoch_idx_train, mloss, mcls_loss, mreg_loss, mrn_loss,mrn0_loss,mrn1_loss,mrn2_loss, mrn3_loss, macc, mrn_acc, mrn0_acc, mrn1_acc, mrn2_acc, mrn3_acc, learningrate))

            # Statistics the mean loss/acc/cls_loss/reg_loss for draw the loss curve
            train_loss_history.append([mloss, macc, mcls_loss, mreg_loss, mrn_loss, mrn_acc,mrn0_loss,mrn1_loss,mrn2_loss,mrn3_loss,
            mrn0_acc,mrn1_acc,mrn2_acc, mrn3_acc])

            plot_loss_train(train_loss_history, save=True, log_dir=folder_summary)

            print('Time for the epoch: {:.4f}'.format(time_for_one_epoch))
            sys.stdout.flush()

            print("-------------------------------------------------------------------------------")
            print("Start to test", "       BATCH_SIZE:", batch_size)
            print("-------------------------------------------------------------------------------")

            # Test
            if epoch_idx_train % 10 == 0 or epoch_idx_train == num_epochs - 1: #or (args.load_ckpt is not None)

                start_time = time.time()
                test_loss_sum = 0.0
                test_cls_loss_sum = 0.0
                test_reg_loss_sum = 0.0
                test_rn_loss_sum = 0.0
                test_rn0_loss_sum = 0.0
                test_rn1_loss_sum = 0.0
                test_rn2_loss_sum = 0.0
                test_rn3_loss_sum = 0.0

                test_total_seen = 0
                test_total_correct = 0
                test_rn_total_correct = 0
                test_rn0_total_correct = 0
                test_rn1_total_correct = 0
                test_rn2_total_correct = 0
                test_rn3_total_correct = 0

                test_rn_total_seen = 0

                # Test for one epoch
                test_batch_idx_total = 0
                test_batch_num_per_epoch = math.floor(num_test / batch_size)
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
                    A_CLS_test_data, _ = load_local_data_and_contdata(A_cls_files, args.iou_threshold, debug=DUBUG_LOCAL_DATA,scale = global_scale)

                    #RN
                    s1_time = time.time()
                    A_test_data = {}
                    A_test_data["ori_data"] = []
                    A_test_data["pts_data"] = []
                    A_test_data['single_object_label'] = []
                    A_test_data['pps_sem_label'] = []
                    A_test_data['pps_ins_label'] = []
                    A_test_data['pps_obbs'] = []

                    for k in range(batch_size):
                        for j in range(PAIR_NUM):
                            A_test_data["ori_data"].append(A_CLS_test_data["ori_data"][k])
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

                    A_CLS_test_cls_label = A_CLS_test_data['pps_sem_label']
                    A_CLS_test_reg_label = A_CLS_test_data['pps_obbs_reg']
                    A_CLS_test_obbs_label = A_CLS_test_data['pps_obbs']
                    A_CLS_test_pts_cont = A_CLS_test_data['pts_context_data']

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



                    e1_time = time.time()

                    rn_label_batch_0 = compute_rn_label_batch(A_test_data, B_test_data, 0, A_files, B_files) 

                    rn_label_batch_1 = compute_rn_label_batch(A_test_data, B_test_data, 1, A_files, B_files) 

                    rn_label_batch_2 = compute_rn_label_batch(A_test_data, B_test_data, 2, A_files, B_files)  

                    rn_label_batch_3 = compute_rn_label_batch(A_test_data, B_test_data, 3, A_files, B_files)  


                    p_and_n_count[0] += len(np.where(rn_label_batch_0 == 1)[0])
                    p_and_n_count[1] += len(np.where(rn_label_batch_0 == 0)[0])



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
                        cls_loss_op, reg_loss_op, rn_loss_op, rn_predictions_0,rn_predictions_1,rn_predictions_2, global_step,cls_probs,
                        rn0_loss_op,rn1_loss_op,rn2_loss_op]
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
                    ops.append(rn0_acc_op)
                    ops.append(rn1_acc_op)
                    ops.append(rn2_acc_op)
                    ops.append(rn3_loss_op)
                    ops.append(rn_predictions_3)  
                    ops.append(rn3_acc_op)
 
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
                    rn0_loss_np = return_value[14]
                    rn1_loss_np = return_value[15]
                    rn2_loss_np = return_value[16]
                    rn0_acc_np =return_value[17]
                    rn1_acc_np =return_value[18]
                    rn2_acc_np =return_value[19]
                    rn3_loss_np = return_value[20]
                    rn3_predictions_np = return_value[21]
                    rn3_acc_np =return_value[22]

                    test_cls_loss_sum += cls_loss_np
                    test_reg_loss_sum += reg_loss_np
                    test_rn_loss_sum += rn_loss_np
                    test_rn0_loss_sum += rn0_loss_np
                    test_rn1_loss_sum += rn1_loss_np
                    test_rn2_loss_sum += rn2_loss_np
                    test_rn3_loss_sum += rn3_loss_np

                    test_loss_sum += loss_np

                    correct = np.sum(cls_predictions_np == cls_label_batch)
                    rn0_correct = np.sum(rn0_predictions_np == rn_label_batch_0)
                    rn1_correct = np.sum(rn1_predictions_np == rn_label_batch_1)
                    rn2_correct = np.sum(rn2_predictions_np == rn_label_batch_2)
                    rn3_correct = np.sum(rn3_predictions_np == rn_label_batch_3)
                    rn_correct = (rn0_correct+rn1_correct+rn2_correct+rn3_correct)/4

                    test_total_correct += correct
                    test_rn_total_correct += rn_correct
                    test_rn0_total_correct += rn0_correct
                    test_rn1_total_correct += rn1_correct
                    test_rn2_total_correct += rn2_correct
                    test_rn3_total_correct += rn3_correct
                    test_total_seen += batch_size
                    test_rn_total_seen += len(rn_label_batch_0)

                    # Add to summary
                    test_step = epoch_idx_train * test_batch_num_per_epoch + batch_idx_test
                    test_writer.add_summary(test_summary, test_step)

                    print('{}-[Test {:02d}/{:02d}]-Epoch: {:06d}  Loss: {:.4f}  cls_loss: {:.4f},  reg_loss: {:.4f}, '
                            'rn_loss: {:.4f},  cls_Acc: {:.4f},  rn_Acc: {:.4f},   lr:{:.8f}'
                            .format(datetime.now(), batch_idx_test + 1, int(test_batch_num_per_epoch), epoch_idx_train,
                                    loss_np, cls_loss_np, reg_loss_np, rn_loss_np,
                                    cls_acc_np,  rn_acc_np,
                                    learningrate))

                    test_batch_idx_total += 1
                print("[TEST] P/N: ", p_and_n_count[0], "/", p_and_n_count[1])

                # Statistics
                mloss_test = test_loss_sum / float(test_batch_idx_total)
                macc_test = test_total_correct / float(test_total_seen)
                mrn_loss_test = test_rn_loss_sum / float(test_batch_idx_total)
                mrn0_loss_test = test_rn0_loss_sum / float(test_batch_idx_total)
                mrn1_loss_test = test_rn1_loss_sum / float(test_batch_idx_total)
                mrn2_loss_test = test_rn2_loss_sum / float(test_batch_idx_total)
                mrn3_loss_test = test_rn3_loss_sum / float(test_batch_idx_total)

                mrnacc_test = test_rn_total_correct / float(test_rn_total_seen)
                mrnacc0_test = test_rn0_total_correct / float(test_rn_total_seen)
                mrnacc1_test = test_rn1_total_correct / float(test_rn_total_seen)
                mrnacc2_test = test_rn2_total_correct / float(test_rn_total_seen)
                mrnacc3_test = test_rn3_total_correct / float(test_rn_total_seen)

                mcls_test_loss = test_cls_loss_sum / float(test_batch_idx_total)
                mreg_test_loss = test_reg_loss_sum / float(test_batch_idx_total)
                
                for m in range(10):
                    test_loss_history.append([mloss_test, macc_test, mcls_test_loss, mreg_test_loss, mrn_loss_test, mrnacc_test,mrn0_loss_test,  mrn1_loss_test,mrn2_loss_test,mrn3_loss_test,
                        mrnacc0_test,mrnacc1_test,mrnacc2_test,mrnacc3_test])
                
                test_summary = tf.Summary()
                test_summary.value.add(tag="cls_loss",simple_value=mcls_test_loss)
                test_summary.value.add(tag="reg_loss",simple_value=mreg_test_loss)
                test_summary.value.add(tag="total_loss",simple_value=mloss_test)
                test_summary.value.add(tag="mrn_loss_test",simple_value=mrn_loss_test)
                test_summary.value.add(tag="mrn0_loss_test",simple_value=mrn0_loss_test)
                test_summary.value.add(tag="mrn1_loss_test",simple_value=mrn1_loss_test)
                test_summary.value.add(tag="mrn2_loss_test",simple_value=mrn2_loss_test)
                test_summary.value.add(tag="mrn3_loss_test",simple_value=mrn3_loss_test)
                test_summary.value.add(tag="cls_accuracy", simple_value=macc_test)
                test_summary.value.add(tag="mrnacc_test", simple_value=mrnacc_test)

                test_writer.add_summary(test_summary, train_step)



                end_time = time.time()
                time_for_one_epoch = end_time - start_time

                # After traing/testing for one epoch, plot the curves.
                plot_loss(train_loss_history, test_loss_history, save=True, log_dir=folder_summary)

                print('{}-[Test]-Done! -Epoch: {:06d}  mLoss: {:.4f}  mcls_loss: {:.4f},  mreg_loss: {:.4f}, '
                        'rn_loss: {:.4f}, rn0_loss: {:.4f}, rn1_loss: {:.4f}, rn2_loss: {:.4f}, rn3_loss: {:.4f}, Acc: {:.4f},  rn_Acc: {:.4f},   lr:{:.8f}'
                        .format(datetime.now(), epoch_idx_train, mloss_test, mcls_test_loss, mreg_test_loss, mrn_loss_test, 
                        mrn0_loss_test,mrn1_loss_test,mrn2_loss_test,mrn3_loss_test,
                                macc_test, mrnacc_test, learningrate))

                sys.stdout.flush()

        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
