from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pointcnn_feature import pointfly as pf
from pointcnn_feature.pointcnn import PointCNN
import tensorflow as tf
from keras import backend as K
import tensorflow.contrib.slim as slim
from keras.layers import Concatenate
import numpy as np

class Net:
    def __init__(self, points_A, features_A, points_B, features_B, points_CLS_A, features_CLS_A , points_A_with_context, features_A_with_context,  is_training, setting, input_box, train_flag=True):
        # Feature extracted from PointCNN
        points = tf.concat((points_A, points_B), axis=0) 
        features = tf.concat((features_A, features_B), axis=0)
        print("points_A:",points_A.shape)
        print("points_B:",points_B.shape)

        print("points:",points.shape)
        print("features:",features.shape)

        concat_points = tf.concat((points_CLS_A, points_A_with_context), axis=0)
        concat_features = tf.concat((features_CLS_A, features_A_with_context), axis=0)   #not features but normals of pointcloud

        b = int(points_CLS_A.shape[0]) #batch_size
        print("b:", b)

        b_rn = int(points_A.shape[0])  #batch_size * pair_num
        b_rn_pair = int(b_rn/b)  #pair_num
        print("b_rn:", b_rn)
        print("b_rn_pair:", b_rn_pair)
        pointcnn_for_cls = PointCNN(concat_points, concat_features, is_training, setting)
        fc_mean = tf.reduce_mean(pointcnn_for_cls.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')

        xconv_feature = tf.concat((fc_mean[0:b], fc_mean[b:]), axis=2) 

        #  Relation Module for reasoning
        with tf.name_scope("relation_reasoning"):
            pointcnn_for_rn = PointCNN(points, features, is_training, setting, name_scope='pointcnn_for_rn')
            print("pointcnn_for_rn:", pointcnn_for_rn.fc_layers[-1].shape)
            rn_fc_mean_0 = tf.reduce_mean(pointcnn_for_rn.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean') #relation PointCNN
            rn_fc_mean = tf.concat((rn_fc_mean_0[0:b_rn], rn_fc_mean_0[b_rn:]), axis=2)

            print("rn_fc_mean:",rn_fc_mean.shape)

            self.rn = RelationNet(rn_fc_mean_0, 2,b,b_rn_pair)
            rn_final = self.rn.fc_3
            rn_final = tf.expand_dims(rn_final, axis=1)

            self.logits_0_for_rn = self.rn.logits_0
            self.probs_0_for_rn = self.rn.probs_0
            self.logits_1_for_rn = self.rn.logits_1
            self.probs_1_for_rn = self.rn.probs_1
            self.logits_2_for_rn = self.rn.logits_2
            self.probs_2_for_rn = self.rn.probs_2
            self.logits_3_for_rn = self.rn.logits_3
            self.probs_3_for_rn = self.rn.probs_3

        concat_rn_fc_mean = rn_fc_mean[0:b_rn_pair]

        rn_add_list = []
        rnmlp_add_list = []
        for i in range(b):
            i_p_rn_fc_mean = rn_fc_mean[b_rn_pair*(i): b_rn_pair*(i+1)]
            i_p_rn_add = i_p_rn_fc_mean[0:1]

            for j in range(b_rn_pair-1):
                i_p_rn_add = tf.add(i_p_rn_add,i_p_rn_fc_mean[j+1:j+2])

            if i<1:
                concat_rn_fc_mean = i_p_rn_add

            else:
                concat_rn_fc_mean = tf.concat((concat_rn_fc_mean, i_p_rn_add) , axis= 0)


        print("xconv_feature:",xconv_feature.shape)
        print("rn_final:",rn_final.shape)
        self.concat_rn_fc_mean = concat_rn_fc_mean
        self.concat_rn_mlp = rn_final
        self.xconv_feature = xconv_feature
        concat_feature = tf.concat((xconv_feature, rn_final), axis=2)

        print("concat_feature:",concat_feature.shape)

        # Classification and regression
        with tf.name_scope("classification_and_regression"):
            cls_output = tf.layers.dense(inputs=concat_feature, units=256, name="cls_output")
            self.logits_for_cls = tf.layers.dense(inputs=cls_output, units=setting.num_class, name="cls_logits")
            print("cls_setting.num_class:",setting.num_class)

            #### Ref: HierarchyLayout https://github.com/yifeishi/HierarchyLayout/tree/master/vdrae
            output = tf.layers.dense(concat_feature, units=256, activation=tf.nn.tanh)
            output = tf.layers.dense(output, units=256, activation=tf.nn.tanh)
            output = tf.squeeze(output, 1)
            output_box = tf.layers.dense(input_box, units=100, activation=tf.nn.tanh)
            vector = tf.concat((output, output_box), 1)
            self.logits_for_reg = tf.layers.dense(vector, units=8, activation=None)

    def get_loss_op(self, singleobj_label=None, cls_label=None, reg_label=None, weights_2d=None, rn0_label=None,rn1_label=None,rn2_label=None,rn3_label=None):
        cls_loss = tf.losses.sparse_softmax_cross_entropy(labels=cls_label, logits=self.logits_for_cls, weights=weights_2d)

        reg_loss = tf.losses.mean_squared_error(reg_label, self.logits_for_reg) 
        reg_loss = reg_loss*10


        rn_loss_0, rn_accuracy_0 = self.rn.build_loss(self.rn.logits_0, rn0_label)  
        rn_loss_1, rn_accuracy_1 = self.rn.build_loss(self.rn.logits_1, rn1_label)  
        rn_loss_2, rn_accuracy_2 = self.rn.build_loss(self.rn.logits_2, rn2_label)  
        rn_loss_3, rn_accuracy_3 = self.rn.build_loss(self.rn.logits_3, rn3_label)  

        rn_loss = rn_loss_0 + rn_loss_1 + rn_loss_2 + rn_loss_3

        loss = cls_loss + reg_loss + 0.5*rn_loss

        rn_accuracy = (rn_accuracy_0 + rn_accuracy_1 + rn_accuracy_2+rn_accuracy_3) /4

        tf.summary.scalar('rn_loss', rn_loss)
        tf.summary.scalar('rn_loss_0', rn_loss_0)
        tf.summary.scalar('rn_loss_1', rn_loss_1)
        tf.summary.scalar('rn_loss_2', rn_loss_2)
        tf.summary.scalar('rn_loss_3', rn_loss_3)
        tf.summary.scalar('rn_accuracy_0', rn_accuracy_0)
        tf.summary.scalar('rn_accuracy_1', rn_accuracy_1)
        tf.summary.scalar('rn_accuracy_2', rn_accuracy_2)
        tf.summary.scalar('rn_accuracy_3', rn_accuracy_3)
        tf.summary.scalar('rn_accuracy', rn_accuracy)
        tf.summary.scalar('cls_loss', cls_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('loss', loss)

        return rn_loss, rn_loss_0 , rn_loss_1 , rn_loss_2 , rn_loss_3 ,rn_accuracy, cls_loss, reg_loss, loss,rn_accuracy_0,rn_accuracy_1,rn_accuracy_2,rn_accuracy_3

    def get_fl_loss_op(self, y, pred, alpha=0.25, gamma=2):

        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
            Args:
             pred: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
             y: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
             alpha: A scalar tensor for focal loss alpha hyper-parameter
             gamma: A scalar tensor for focal loss gamma hyper-parameter
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
        """
        alpha = tf.to_float(alpha)
        gamma = tf.to_float(gamma)
        y = tf.one_hot(y, 2)

        zeros = tf.zeros_like(pred, dtype=pred.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

        focal_loss = tf.reduce_sum(per_entry_cross_ent)
        tf.summary.scalar('focal_loss', focal_loss)

        return focal_loss

    def binary_focal_loss(self, y_true, y_pred, gamma=2, alpha=0.25):
        """
        Binary form of focal loss.
        适用于二分类问题的focal loss
        focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        loss = K.mean(focal_loss)
        tf.summary.scalar('focal_loss', loss)

        return loss

    def get_acc_op(self, prediction, labels):

        prediction = tf.cast(prediction, tf.int32)
        labels = tf.cast(labels, tf.int32)

        acc = tf.cast(tf.equal(prediction, labels), tf.float32)
        acc = tf.reduce_mean(acc)

        tf.summary.scalar('accuracy_top_1', acc)

        return acc


"""
    Relation Network
"""
class RelationNet:
    def __init__(self, pts_feature, class_num, b,b_rn_pair,is_train=True):
        self.is_train = is_train
        self.class_num = class_num
        g, self.logits_0, self.logits_1, self.logits_2, self.logits_3 = self.CONV(pts_feature, scope='CONV')
        self.rn_final  = self.f_phi(g, b,b_rn_pair, scope='f_phi')
        self.probs_0 = tf.nn.softmax(self.logits_0)
        self.probs_1 = tf.nn.softmax(self.logits_1)
        self.probs_2 = tf.nn.softmax(self.logits_2)
        self.probs_3 = tf.nn.softmax(self.logits_3)

    # Classifier: takes images as input and outputs class label [B, m]
    def CONV(self, pts, scope='CONV'):
        concat = Concatenate()
        with tf.variable_scope(scope) as scope:
            print(scope.name)

            # eq.1 in the paper
            # g_theta = (o_i, o_j, q)
            # conv_4 [B, d, d, k]
            b, _, _ = pts.get_shape().as_list()  #pts.shape: [128, 128, 192]
            pts = tf.reshape(pts, (b, -1))

            b = int(b/2)   
            pts_A = pts[0:b]
            pts_B = pts[b:]
            all_g = tf.concat((pts_A, pts_B), axis=-1)
            print("all_g_b:",all_g.shape)
            all_g = tf.stack(all_g, axis=0)
            print("all_g_a:",all_g.shape)
            all_g, g_5_0, g_5_1, g_5_2, g_5_3 = self.g_theta(all_g)
            # all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
            return all_g, g_5_0, g_5_1, g_5_2, g_5_3

    def g_theta(self, x_full, scope='g_theta', reuse=None):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            if not reuse: print(scope.name)
            g_1 = self.fc(x_full, 256, name='g_1')
            g_2 = self.fc(g_1, 256, name='g_2')
            g_3 = self.fc(g_2, 256, name='g_3')
            g_4 = self.fc(g_3, 256, name='g_4')
            print ("g_4:",g_4.shape)
            g_5_0 = self.fc(g_4, 2, name = 'g_5_0') #rn0
            g_5_1 = self.fc(g_4, 2, name = 'g_5_1') #rn1
            g_5_2 = self.fc(g_4, 2, name = 'g_5_2') #rn2
            g_5_3 = self.fc(g_4, 2, name = 'g_5_3') #rn3

            return g_4, g_5_0, g_5_1, g_5_2, g_5_3

    def f_phi(self, g, b, b_rn_pair, scope='f_phi',):
        with tf.variable_scope(scope) as scope:
            print(scope.name)
            g = tf.reshape(g,(b,b_rn_pair,-1))

            concat_g_A_B = tf.reduce_sum(g, axis = 1)

            fc_1 = self.fc(concat_g_A_B, 256, name='fc_1')
            self.fc_2 = fc_2 = self.fc(fc_1, 256, name='fc_2')
            fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_train, scope='fc_3/')  #fc_2.shape: (64, 256)
            self.fc_3 = self.fc(fc_2, 128, name='fc_3')


            return self.fc_3

    def build_loss(self, logits, labels):
        # Cross-entropy loss
        self.labels = labels 
        print("self.labels:",self.labels.shape)
        print("logits:",logits.shape)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)

        self.correct_prediction = tf.equal(tf.argmax(logits, 1), self.labels) 
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        return tf.reduce_mean(loss), accuracy

    def fc(self, input, output_shape, activation_fn=tf.tanh, name="fc"): 
        output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
        return output
    
    def get_fl_loss_op(self, y, pred, alpha=0.25, gamma=2):

        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
            Args:
             pred: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
             y: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
             alpha: A scalar tensor for focal loss alpha hyper-parameter
             gamma: A scalar tensor for focal loss gamma hyper-parameter
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
        """
        alpha = tf.to_float(alpha)
        gamma = tf.to_float(gamma)
        y = tf.one_hot(y, 2)

        zeros = tf.zeros_like(pred, dtype=pred.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

        focal_loss = tf.reduce_sum(per_entry_cross_ent)
        tf.summary.scalar('focal_loss', focal_loss)

        return focal_loss