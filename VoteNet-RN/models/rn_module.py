import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_distance import nn_distance, huber_loss
import numpy as np


class RNModule(nn.Module):
    def __init__(self, relation_pair=3, relation_type='semantic', random=False):
        super().__init__()

        self.relation_pair = relation_pair
        self.relation_type = relation_type
        self.random = random
        print('relation pair: {}'.format(self.relation_pair))
        print("FLAGS.relation_type: ", self.relation_type)
        print("random: ", self.random)
        self.gu_conv = torch.nn.Conv2d(128, 256, (1, 1))
        self.gu_bn = torch.nn.BatchNorm2d(256)
        self.rn_conv1 = torch.nn.Conv1d(256, 128, 1)
        self.rn_conv2 = torch.nn.Conv1d(128, 128, 1)

        self.rn_bn1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2 = torch.nn.BatchNorm1d(128)
        self.rn_conv3 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4 = torch.nn.Conv1d(128, 2, 1)

        self.rn_bn1_1 = torch.nn.BatchNorm1d(128)
        self.rn_bn2_1 = torch.nn.BatchNorm1d(128)
        self.rn_conv3_1 = torch.nn.Conv1d(128, 128, 1)
        self.rn_conv4_1 = torch.nn.Conv1d(128, 2, 1)

        # self.rn_bn1_2 = torch.nn.BatchNorm1d(128)
        # self.rn_bn2_2 = torch.nn.BatchNorm1d(128)
        # self.rn_conv3_2 = torch.nn.Conv1d(128, 128, 1)
        # self.rn_conv4_2 = torch.nn.Conv1d(128, 2, 1)

        # self.rn_bn1 = torch.nn.BatchNorm1d(256)
        # self.rn_bn2 = torch.nn.BatchNorm1d(128)
        # self.rn_conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.rn_conv4 = torch.nn.Conv1d(128, 2, 1)
        #
        # self.rn_bn1_1 = torch.nn.BatchNorm1d(256)
        # self.rn_bn2_1 = torch.nn.BatchNorm1d(128)
        # self.rn_conv3_1 = torch.nn.Conv1d(256, 128, 1)
        # self.rn_conv4_1 = torch.nn.Conv1d(128, 2, 1)


    def forward(self, feature, end_points):
        # prepare features for relation pairs
        bs, fea_in, num_proposal = feature.shape  # [B, feat_in, proposal_num]
        feature = feature.permute(0, 2, 1)  # [B, proposal_num, feat_in]  [8, 256, 128]

        # Use center point for distance computation
        # pred_center = end_points['center']
        pred_center = end_points['aggregated_vote_xyz']
        _pred_center_i = pred_center.view(bs, 1, num_proposal, 3).repeat(1, num_proposal, 1, 1)
        _pred_center_j = pred_center.view(bs, num_proposal, 1, 3).repeat(1, 1, num_proposal, 1)

        dist = (_pred_center_j[:, :, :, :3] - _pred_center_i[:, :, :, :3]).pow(2)
        dist = torch.sum(dist, -1) * (-1)

        # get index j for i
        self.max_paris = self.relation_pair
        if self.random:
            _idx_j_batch = torch.randint(0, num_proposal, (bs, num_proposal, self.max_paris), dtype=torch.int).cuda().long()
        else:
            # select the nearest proposals as relation pairs
            _, _idx_j = torch.topk(dist, k=self.relation_pair + 1)
            _idx_j_batch = _idx_j[:, :, 1:]
        _idx_j_batch_reshape = _idx_j_batch.reshape((bs * num_proposal * self.max_paris))

        # get pair of features
        _feature_i = feature.unsqueeze(2).repeat((1, 1, self.max_paris, 1))  # [B, proposal_num, self.max_pairs, feat_in]
        _range_for_bs = torch.arange(bs).unsqueeze(1).repeat((1, num_proposal * self.max_paris)).reshape((bs * num_proposal * self.max_paris))
        _feature_j = feature[_range_for_bs, _idx_j_batch_reshape, :].reshape((bs, num_proposal, self.max_paris, fea_in))
        relation_u = _feature_i.add(_feature_j)  # [B, proposal_num, max_pairs, feat_in]

        # print("idxes.shape: ", idxes.shape)
        end_points['nearest_n_index'] = _idx_j_batch

        """
            g_theta
        """
        # get relation feature for each object
        relation_u = relation_u.permute(0, 3, 1, 2)  # [B, feat_dim, proposal_num, pairs_num] [8, 128, 256, 3]
        # print("relation_u shape:", relation_u.shape)
        gu_output = self.gu_conv(relation_u)  # [B, feat_dim, proposal_num, pairs_num][8, 256, 256, 3]
        gu_output = self.gu_bn(gu_output)  # debug


        """
            f_phi
        """
        unsqueeze_h = torch.mean(gu_output, 3)  # [B, feat_dim, proposal_num] [8, 256, 256]
        output = self.rn_conv1(unsqueeze_h)  # [bs, fea_channel, proposal_num] [B, 128, 256]
        # output = torch.sigmoid(torch.log(torch.abs(output)))
        # output = F.sigmoid(output)
        end_points['rn_feature'] = output

        # get relationships for all pairs
        _, u_inchannel, _, _ = relation_u.shape
        relation_all = relation_u.view(bs, u_inchannel, num_proposal * self.max_paris)
        relation_all = self.rn_conv2(relation_all)
        # _, u_inchannel, _, _ = gu_output.shape
        # relation_all = gu_output.view(bs, u_inchannel, num_proposal * self.max_paris)

        if 'semantic' in self.relation_type:
            relation_all_0 = F.relu(self.rn_bn1(relation_all))
            relation_all_0 = self.rn_conv3(relation_all_0)
            relation_all_0 = F.relu(self.rn_bn2(relation_all_0))
            logits_0 = self.rn_conv4(relation_all_0)  # semantic
            end_points['rn_logits_0'] = logits_0

        if 'spatial' in self.relation_type:
            relation_all_1 = F.relu(self.rn_bn1_1(relation_all))
            relation_all_1 = self.rn_conv3_1(relation_all_1)
            relation_all_1 = F.relu(self.rn_bn2_1(relation_all_1))
            logits_1 = self.rn_conv4_1(relation_all_1)  # spatial
            # probs = torch.nn.softmax(logits, 2)
            end_points['rn_logits_1'] = logits_1

        return end_points
