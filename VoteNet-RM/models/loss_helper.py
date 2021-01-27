# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss
from scannet.model_util_scannet import ScannetDatasetConfig
DATASET_CONFIG = ScannetDatasetConfig()

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

# For nearest
# RN_CLS_WEIGHTS = [0.8, 0.2] # put larger weights on negative objectness
# RN_CLS_WEIGHTS1 = [0.14, 0.86] # put larger weights on negative objectness

# For random
# RN_CLS_WEIGHTS = [0.3, 0.7] # scannet
# RN_CLS_WEIGHTS1 = [0.2, 0.8] # scannet
# RN_CLS_WEIGHTS2 = [0.05, 0.95] # scannet

RN_CLS_WEIGHTS = [0.25, 0.75] # sunrgbd
RN_CLS_WEIGHTS1 = [0.1, 0.9] # sunrgbd
RN_CLS_WEIGHTS2 = [0.05, 0.95] # sunrgbd

print("RN_CLS_WEIGHTS: ", RN_CLS_WEIGHTS)
print("RN_CLS_WEIGHTS1: ", RN_CLS_WEIGHTS1)
print("RN_CLS_WEIGHTS2: ", RN_CLS_WEIGHTS2)

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def get_loss_with_rn(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Relation cls loss
    rn_cls_losses = compute_multi_rn_cls_loss(end_points)
    end_points['rn_cls_loss'] = rn_cls_losses[0]

    # Final loss function
    alpha = 1.0
    beta = 1.0
    # gamma = 1.0
    # print("alpha:{},  beta:{}".format(alpha, beta))
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.1*alpha*rn_cls_losses[1] + 0.1*beta*rn_cls_losses[2]
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


def compute_multi_rn_cls_loss(end_points):
    relation_type = end_points['relation_type']
    object_assignment = end_points['object_assignment']
    objectness_label = end_points['objectness_label'].float()
    bboxes = []
    proposal_labels = []

    assert len(relation_type) > 0

    rn_loss_0 = 0.0
    rn_loss_1 = 0.0
    rn_loss_2 = 0.0

    nearest_n_index = end_points['nearest_n_index']
    bs, roi_num, pair_num = nearest_n_index.shape
    nearest_n_index_reshape = nearest_n_index.reshape(bs * roi_num * pair_num)

    _mask_i = objectness_label.unsqueeze(2).repeat(1, 1, pair_num).view(bs, roi_num * pair_num)
    _mask_j = objectness_label.index_select(1, nearest_n_index_reshape)
    _mask_j = _mask_j[torch.arange(bs).cuda().view(bs, -1), torch.arange(bs * roi_num * pair_num).view(bs, -1)]
    rn_mask = (_mask_i.mul(_mask_j)).float()

    if 'semantic' in relation_type:
        """
            Compute loss for relation_0: semantic relationship

        """
        rn_scores_0 = end_points['rn_logits_0']

        proposal_labels = torch.gather(end_points['sem_cls_label'], 1, object_assignment)

        _label_i = proposal_labels.unsqueeze(2).repeat(1, 1, pair_num).view(bs, roi_num * pair_num)
        _label_j = proposal_labels.index_select(1, nearest_n_index_reshape)
        _label_j = _label_j[torch.arange(bs).cuda().view(bs, -1), torch.arange(bs * roi_num * pair_num).view(bs, -1)]
        rn_labels_0 = (_label_i.eq(_label_j).long())

        criterion = nn.CrossEntropyLoss(torch.Tensor(RN_CLS_WEIGHTS).cuda(), reduction='none')
        rn_loss_0 = criterion(rn_scores_0, rn_labels_0)
        rn_loss_0 = torch.sum(rn_loss_0 * rn_mask) / (torch.sum(rn_mask) + 1e-6)

        print("rn0_scores: ", len(np.where(np.argmax(rn_scores_0.cpu().detach().numpy(), 1) == 1)[0]))
        print("rn0 labels: {},   rn_labels_num: {}".format(len(np.where(rn_labels_0.cpu().detach().numpy() == 1)[0]),
                                                           rn_labels_0.cpu().detach().numpy().shape))
        end_points['rn_labels_0'] = rn_labels_0
        end_points['rn_preds_0'] = torch.argmax(rn_scores_0, dim=1)

    if 'spatial' in relation_type:
        """
            Compute loss for relation_1: spatial relationship
        """
        rn_scores_1 = end_points['rn_logits_1']

        bboxes = get_obbs(end_points)
        bs, roi_num, point_num, dim = bboxes.shape
        bboxes_i = bboxes.unsqueeze(2).repeat(1, 1, pair_num, 1, 1).reshape(bs, roi_num * pair_num, point_num, dim)
        bboxes_j = bboxes[torch.arange(bs).unsqueeze(1).repeat(1, roi_num * pair_num).reshape(bs * roi_num * pair_num),
                   nearest_n_index_reshape, :, :].reshape(bs, roi_num * pair_num, point_num, dim)
        rn_labels_1 = compute_spatial_relation(bboxes_i, bboxes_j, 1).long()

        criterion = nn.CrossEntropyLoss(torch.Tensor(RN_CLS_WEIGHTS1).cuda(), reduction='none')
        rn_loss_1 = criterion(rn_scores_1, rn_labels_1)
        rn_loss_1 = torch.sum(rn_loss_1 * rn_mask) / (torch.sum(rn_mask) + 1e-6)

        print("rn1_scores: ", len(np.where(np.argmax(rn_scores_1.cpu().detach().numpy(), 1) == 1)[0]))
        print("rn1 labels: {},   rn_labels_num: {}".format(len(np.where(rn_labels_1.cpu().detach().numpy() == 1)[0]),
                                                           rn_labels_1.cpu().detach().numpy().shape))
        end_points['rn_labels_1'] = rn_labels_1
        end_points['rn_preds_1'] = torch.argmax(rn_scores_1, dim=1)

    rn_loss = rn_loss_0 + rn_loss_1

    # print("rn_mask: ", rn_mask)
    return [rn_loss, rn_loss_0, rn_loss_1]


def compute_ins_relation(bboxes_i, bboxes_j, cls_i, cls_j, thr=0.25):
    """
    :param bboxes_i: [batchsize, N, 8, 3]
    :param bboxes_j: [batchsize, N, 8, 3]
    :param cls_i: [batchsize, N]
    :param cls_j: [batchsize, N]
    :return:
    """

    bs, proposal_num, point_num, dim = bboxes_i.shape
    obbs_i = torch.reshape(bboxes_i, (bs * proposal_num, point_num, dim))
    obbs_j = torch.reshape(bboxes_j, (bs * proposal_num, point_num, dim))
    cls_i = torch.reshape(cls_i, (bs * proposal_num, 1))
    cls_j = torch.reshape(cls_j, (bs * proposal_num, 1))
    box_num, _, _ = obbs_i.shape

    # obbs_i = obbs_i[:, :, [0, 2, 1]]
    # obbs_j = obbs_j[:, :, [0, 2, 1]]
    # obbs_i[:, :, 2] *= -1
    # obbs_j[:, :, 2] *= -1

    boxes = torch.stack([obbs_i, obbs_j], 0).squeeze(1) # [2, bs*proposal_num, 8, 3]

    x1 = boxes[:, :, :, 0].min(2)[0]
    y1 = boxes[:, :, :, 1].min(2)[0]
    z1 = boxes[:, :, :, 2].min(2)[0]
    x2 = boxes[:, :, :, 0].max(2)[0]
    y2 = boxes[:, :, :, 1].max(2)[0]
    z2 = boxes[:, :, :, 2].max(2)[0]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    xx1 = x1.max(0)[0]
    yy1 = y1.max(0)[0]
    zz1 = z1.max(0)[0]
    xx2 = x2.min(0)[0]
    yy2 = y2.min(0)[0]
    zz2 = z2.min(0)[0]

    l = np.maximum(0, xx2 - xx1)
    w = np.maximum(0, yy2 - yy1)
    h = np.maximum(0, zz2 - zz1)

    inter = l * w * h
    o = (inter / (area[0] + area[1] - inter)).cuda()
    cls_mask = (cls_i == cls_j).squeeze(1).double()
    o = o * cls_mask

    label = (o>=thr).reshape((bs, proposal_num))

    return label.long()


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_obbs(end_points):
    pred_center = end_points['center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = DATASET_CONFIG.class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = DATASET_CONFIG.class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    return torch.from_numpy(pred_corners_3d_upright_camera)


def get_2d_iou_batch(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = torch.max(torch.stack((bb1[0], bb2[0]), 1), 1)[0]
    y_top = torch.max(torch.stack((bb1[1], bb2[1]), 1), 1)[0]
    x_right = torch.min(torch.stack((bb1[2], bb2[2]), 1), 1)[0]
    y_bottom = torch.min(torch.stack((bb1[3], bb2[3]), 1), 1)[0]

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = torch.clamp((x_right - x_left), min=0) * torch.clamp((y_bottom - y_top), min=0)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (bb1_area + bb2_area - intersection_area + 1e-16).double()

    return iou


def compute_spatial_relation(obbs_i, obbs_j, relation_type=1):
    bs, proposal_num, point_num, dim = obbs_i.shape
    obbs_i = torch.reshape(obbs_i, (bs*proposal_num, point_num, dim))
    obbs_j = torch.reshape(obbs_j, (bs*proposal_num, point_num, dim))

    obbs_i = obbs_i[:, :, [0, 2, 1]]
    obbs_j = obbs_j[:, :, [0, 2, 1]]
    obbs_i[:, :, 2] *= -1
    obbs_j[:, :, 2] *= -1

    distances = []
    for i in range(8):
        dist = (obbs_i - obbs_j).pow(2)
        dist = torch.sum(dist, -1)
        distances.append(dist)
        obbs_j = obbs_j.roll(shifts=1, dims=1)
    distances = torch.stack(distances, -1).reshape(bs*proposal_num, 8*8)
    min_dist, _ = torch.min(distances, dim=1)

    # label the proposals having large distace as 0
    labels = torch.zeros((bs*proposal_num))
    # iou between proposal's x-y surface must be lager than 0.1
    box_pair = torch.stack([obbs_i, obbs_j], 0).squeeze(1)
    min_x = box_pair[:, :, :, 0].min(2)[0]
    max_x = box_pair[:, :, :, 0].max(2)[0]
    min_y = box_pair[:, :, :, 1].min(2)[0]
    max_y = box_pair[:, :, :, 1].max(2)[0]
    min_z = box_pair[:, :, :, 2].min(2)[0]
    max_z = box_pair[:, :, :, 2].max(2)[0]
    iou_xy = get_2d_iou_batch([min_x[0], min_y[0], max_x[0], max_y[0]],
                              [min_x[1], min_y[1], max_x[1], max_y[1]])
    iou_xz = get_2d_iou_batch([min_x[0], min_z[0], max_x[0], max_z[0]],
                              [min_x[1], min_z[1], max_x[1], max_z[1]])
    iou_yz = get_2d_iou_batch([min_y[0], min_z[0], max_y[0], max_z[0]],
                              [min_y[1], min_z[1], max_y[1], max_z[1]])

    abs_height = torch.min(torch.stack((torch.abs(min_z[0] - max_z[1]), torch.abs(max_z[0] - min_z[1])), 1), 1)[0]
    abs_width = torch.min(torch.stack((torch.abs(min_x[0] - max_x[1]), torch.abs(max_x[0] - min_x[1])), 1), 1)[0]
    abs_length = torch.min(torch.stack((torch.abs(min_y[0] - max_y[1]), torch.abs(max_y[0] - min_y[1])), 1), 1)[0]
    # min height between proposals min_z/max_z must be smaller than 0.1
    indexes = ((iou_xy > 0.1).mul((abs_height < 0.1))).nonzero()
    indexes = torch.cat((indexes, ((iou_yz > 0.1).mul((abs_width < 0.1))).nonzero()), 0)
    indexes = torch.cat((indexes, ((iou_xz > 0.1).mul((abs_length < 0.1))).nonzero()), 0)

    labels[torch.squeeze(indexes.unique())] = 1

    labels = torch.where(min_dist > 0.5, torch.full_like(min_dist, 0), labels.double())
    labels = labels.reshape((bs, proposal_num)).cuda()

    return labels