# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using MLCVNet 3D object detector to detect objects from a point cloud.
"""
import open3d

import os
import sys
import numpy as np
import argparse
import importlib
import time
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
parser.add_argument('--scene_name', default='scene0609_02_vh_clean_2.ply',
                    help='Scene name. [default: scene0609_02_vh_clean_2.ply]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.pc_util import random_sampling, read_ply_scannet
from ap_helper import parse_predictions


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    # point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)

    pc = np.concatenate([point_cloud[:, 0:3], np.expand_dims(point_cloud[:, -1], 1)], 1)
    pc = np.expand_dims(pc.astype(np.float32), 0)  # (1,40000,4)
    return pc, point_cloud


def heading2rotmat(heading_angle):
    pass
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat


def convert_oriented_box_to_trimesh_fmt(box):
    ctr = box[:3]
    lengths = box[3:6]
    trns = np.eye(4)
    trns[0:3, 3] = ctr
    trns[3, 3] = 1.0
    trns[0:3, 0:3] = heading2rotmat(box[6])
    box_trimesh_fmt = trimesh.creation.box(lengths, trns)
    return box_trimesh_fmt


label2color = {
        0: [31, 119, 180],  # 3, cabinet
        1: [255, 187, 120],  # 4, bed
        2: [188, 189, 34],  # 5, chair
        3: [140, 86, 75],  # 6, sofa
        4: [255, 152, 150],  # 7, table
        5: [214, 39, 40],  # 8, door
        6: [197, 176, 213],  # 9, window
        7: [148, 103, 189],  # 10, bookshelf
        8: [196, 156, 148],  # 11, picture
        9: [23, 190, 207],  # 12, counter
        10: [247, 182, 210],  # 14, desk
        11: [219, 219, 141],  # 16, curtain
        12: [255, 127, 14],  # 24, refrigerator
        13: [158, 218, 229],  # 28, shower curtain
        14: [44, 160, 44],  # 33, toilet
        15: [112, 128, 144],  # 34, sink
        16: [227, 119, 194],  # 36, bathtub
        17: [82, 84, 163],  # 39, otherfurniture
        18: [0, 0, 255],  # None,
        19: [255, 0, 255]  # special objects
    }


DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal

    all_bbox = []
    objectness_prob = softmax(objectness_scores[0,:,:])[:,1] # (K,)
    # Dump predicted bounding boxes
    if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
        num_proposal = pred_center.shape[1]
        obbs = []
        for j in range(num_proposal):
            obb = config.param2obb(pred_center[0,j,0:3], pred_heading_class[0,j], pred_heading_residual[0,j],
                            pred_size_class[0,j], pred_size_residual[0,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
            # Output boxes according to their semantic labels
            pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1) # B,num_proposal
            pred_sem_cls = pred_sem_cls.detach().cpu().numpy()
            for l in np.unique(pred_sem_cls[0,:]):
                mask = np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[0,:]==1)
                mask = np.logical_and(mask==True,pred_sem_cls[0,:]==l)
                if np.sum(mask)>0:
                    # pc_util.write_oriented_bbox(obbs[mask,:], os.path.join(dump_dir, '%d_pred_confident_nms_bbox.ply'%(l)))
                    for box, pred_cls in zip(obbs[mask,:], pred_sem_cls[0][mask]):
                        bbox_mesh = convert_oriented_box_to_trimesh_fmt(box)
                        bbox = np.array(bbox_mesh.vertices)
                        bbox = np.array([bbox[0], bbox[1], bbox[3], bbox[2], bbox[4], bbox[5], bbox[7], bbox[6], pred_cls])
                        all_bbox.append(bbox)

    return all_bbox

def vis(point_clouds, bboxes, scene_name):

    # bbox for votenet
    count = 0
    all_lines = []
    all_colors = []
    all_bbox = []
    for bbox in bboxes[0]:
        pred_cls = bbox[-1]
        bbox = np.array(list(bbox[0:8]))
        all_bbox.append(bbox)

        lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count],
                 [0 + 8 * count, 4 + 8 * count],
                 [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count],
                 [4 + 8 * count, 5 + 8 * count],
                 [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count],
                 [6 + 8 * count, 7 + 8 * count],
                 [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count],
                 [7 + 8 * count, 4 + 8 * count]]
        all_lines.append(lines)

        # repeat n times for bold the lines
        repeat_num = 20
        for j in range(int(repeat_num / 2)):
            all_bbox.append(bbox - 0.001 * (j + 1))
            all_bbox.append(bbox + 0.001 * (j + 1))

        for n in range(repeat_num):
            count = count + 1
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count],
                     [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count],
                     [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count],
                     [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count],
                     [7 + 8 * count, 4 + 8 * count]]
            all_lines.append(lines)

        _color_rgb = label2color[pred_cls]
        all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
        count += 1


    all_lines = np.array(all_lines).astype(np.int)
    all_lines = np.reshape(all_lines, (-1, 2))

    all_bbox = np.array(all_bbox).astype(np.double)
    all_bbox = np.reshape(all_bbox, (-1, 3))
    all_colors = np.array(all_colors) / 255.0

    line_pcd1 = open3d.geometry.LineSet()
    line_pcd1.lines = open3d.utility.Vector2iVector(all_lines)
    line_pcd1.colors = open3d.utility.Vector3dVector(all_colors)
    line_pcd1.points = open3d.utility.Vector3dVector(all_bbox)


    # bbox for votenet_with_rn
    count = 0
    all_lines = []
    all_colors = []
    all_bbox = []
    for bbox in bboxes[1]:
        pred_cls = bbox[-1]
        bbox = np.array(list(bbox[0:8]))
        all_bbox.append(bbox)

        lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count],
                 [0 + 8 * count, 4 + 8 * count],
                 [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count],
                 [4 + 8 * count, 5 + 8 * count],
                 [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count],
                 [6 + 8 * count, 7 + 8 * count],
                 [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count],
                 [7 + 8 * count, 4 + 8 * count]]
        all_lines.append(lines)

        # repeat n times for bold the lines
        repeat_num = 20
        for j in range(int(repeat_num / 2)):
            all_bbox.append(bbox - 0.001 * (j + 1))
            all_bbox.append(bbox + 0.001 * (j + 1))

        for n in range(repeat_num):
            count = count + 1
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count],
                     [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count],
                     [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count],
                     [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count],
                     [7 + 8 * count, 4 + 8 * count]]
            all_lines.append(lines)

        _color_rgb = label2color[pred_cls]
        all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
        count += 1

    all_lines = np.array(all_lines).astype(np.int)
    all_lines = np.reshape(all_lines, (-1, 2))

    all_bbox = np.array(all_bbox).astype(np.double)
    all_bbox = np.reshape(all_bbox, (-1, 3))
    all_colors = np.array(all_colors) / 255.0

    line_pcd2 = open3d.geometry.LineSet()
    line_pcd2.lines = open3d.utility.Vector2iVector(all_lines)
    line_pcd2.colors = open3d.utility.Vector3dVector(all_colors)
    line_pcd2.points = open3d.utility.Vector3dVector(all_bbox)

    scene_points = np.array(point_clouds[:, :3])
    scene_colors = np.array(point_clouds[:, 3:6]) / 255.0
    scene_pts = open3d.geometry.PointCloud()
    scene_pts.points = open3d.utility.Vector3dVector(scene_points)
    scene_pts.colors = open3d.utility.Vector3dVector(scene_colors)

    vis1 = open3d.Visualizer()
    vis1.create_window(window_name='votenet')
    vis1.add_geometry(scene_pts)
    vis1.add_geometry(line_pcd1)
    vis1.run()

    # _start = time.time()
    # while True:
    #     vis1.update_geometry()
    #     vis1.poll_events()
    #     vis1.capture_screen_image('/home/alala/Desktop/det_results/{}_votenet.png'.format(scene_name))
    #
    #     if time.time() - _start > 1:
    #         break
    vis1.remove_geometry(scene_pts)
    vis1.remove_geometry(line_pcd1)
    vis1.close()
    vis1.destroy_window()



    vis2 = open3d.Visualizer()
    vis2.create_window(window_name='votenetrn')
    vis2.add_geometry(scene_pts)
    vis2.add_geometry(line_pcd2)
    vis2.run()

    # _start = time.time()
    # while True:
    #     vis2.update_geometry()
    #     vis2.poll_events()
    #     vis2.capture_screen_image('/home/alala/Desktop/det_results/{}_votenetrn.png'.format(scene_name))
    #
    #     if time.time() - _start > 1:
    #         break
    vis2.remove_geometry(scene_pts)
    vis2.remove_geometry(line_pcd1)
    vis2.close()
    vis2.destroy_window()


def get_gt_bbox(end_points, dump_dir, config):
    gt_center = end_points['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy()  # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy()  # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy()  # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy()  # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy()  # B,K2,3
    objectness_label = end_points['objectness_label'].detach().cpu().numpy()  # (B,K,)
    objectness_mask = end_points['objectness_mask'].detach().cpu().numpy()  # (B,K,)

    # Dump GT bounding boxes
    all_bbox = []
    obbs = []
    for j in range(gt_center.shape[1]):
        if gt_mask[0, j] == 0: continue
        obb = config.param2obb(gt_center[0, j, 0:3], gt_heading_class[0, j], gt_heading_residual[0, j],
                               gt_size_class[0, j], gt_size_residual[0, j])
        obbs.append(obb)
    if len(obbs) > 0:
        obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
        for box in obbs:
            pred_cls = box[-1]
            box = box[0:-1]
            bbox_mesh = convert_oriented_box_to_trimesh_fmt(box)
            bbox = np.array(bbox_mesh.vertices)
            bbox = np.array([bbox[0], bbox[1], bbox[3], bbox[2], bbox[4], bbox[5], bbox[7], bbox[6], pred_cls])
            all_bbox.append(bbox)
    return all_bbox


if __name__ == '__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import DC  # dataset config

    # Make sure your scannet ply file get transformed first, use rotate_val_scans.py to transform
    # Put any scannet transformed ply file in the demo_files folder and put its file name here
    # Then run demo.py
    for scene_name in open('./scannet/meta_data/scannetv2_val.txt').readlines():
        scene_name = scene_name.split('\n')[0]

        if scene_name not in 'scene0430_00':
            continue

        print("=========================== {} =========================".format(scene_name))
        pc_path = os.path.join('/media/alala/Data2/ScanNet/scannet/', scene_name + '/' + scene_name + '_vh_clean_2.ply')

        eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                            'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                            'conf_thresh': 0.5, 'dataset_config': DC}

        checkpoints = [os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar'),
                       './log_scannet/log_rn8_support_random/checkpoint.tar']

        pred_boxes = []
        # Init the model and optimzier
        for model_name, checkpoint_path in zip(['votenet', 'votenet_with_rn'], checkpoints):
            # model_name = 'votenet'
            # checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
            MODEL = importlib.import_module(model_name)  # import network module
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
                                sampling='seed_fps', num_class=DC.num_class,
                                num_heading_bin=DC.num_heading_bin,
                                num_size_cluster=DC.num_size_cluster,
                                mean_size_arr=DC.mean_size_arr).to(device)
            print('Constructed model.')

            # Load checkpoint
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

            # Load and preprocess input point cloud
            net.eval()  # set model to eval mode (for bn and dp)
            point_cloud = read_ply_scannet(pc_path)
            pc, point_cloud_with_rgb = preprocess_point_cloud(point_cloud)
            print('Loaded point cloud data: %s' % (pc_path))

            # Model inference
            inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
            tic = time.time()
            with torch.no_grad():
                end_points = net(inputs)
            toc = time.time()
            print('Inference time: %f' % (toc - tic))
            end_points['point_clouds'] = inputs['point_clouds']
            end_points['point_clouds_with_rgb'] = inputs['point_clouds'][0]
            pred_map_cls = parse_predictions(end_points, eval_config_dict)
            print('Finished detection. %d object detected.' % (len(pred_map_cls[0])))

            dump_dir = os.path.join(demo_dir, FLAGS.scene_name.split('.')[0])
            if not os.path.exists(dump_dir): os.mkdir(dump_dir)
            boxes = dump_results(end_points, dump_dir, DC, True)
            pred_boxes.append(boxes)
            print('Dumped detection results to folder %s' % (dump_dir))

        # gt_bbox = get_gt_bbox(end_points, dump_dir, DC)
        vis(point_cloud, pred_boxes, scene_name)
