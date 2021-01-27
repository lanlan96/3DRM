#!/usr/bin/python3
import math

num_class = 13
relation_num_class = 2

sample_num = 1024

batch_size = 16

num_epochs = 1024

label_weights = [1.0] * num_class

learning_rate_base = 1e-3
decay_steps = 2000
decay_rate = 0.8
learning_rate_min = 1e-15
step_val = 500

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.001, 0.001, 0.001, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 3

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]

with_global = True

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.8)]]

sampling = 'random'

optimizer = 'adam'
# epsilon = 1e-2
epsilon = 1e-08

data_dim = 6
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
