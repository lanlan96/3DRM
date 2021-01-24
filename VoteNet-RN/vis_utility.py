import numpy as np
import trimesh
import open3d
import torch


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


def draw_bbox(pred_dict, config=None, nms=False):

    scene_point_cloud = pred_dict['point_clouds']
    obb_nums = len(pred_dict['pred_center'])
    pred_center = pred_dict['pred_center']
    pred_heading_class = pred_dict['pred_heading_class']
    pred_heading_residual = pred_dict['pred_heading_residual']
    pred_size_class = pred_dict['pred_size_class']
    pred_size_residual = pred_dict['pred_size_residual']

    all_bbox = []
    all_lines = []
    all_colors = []
    count = 0
    for i in range(obb_nums):

        obb = config.param2obb(pred_center[i, 0:3], pred_heading_class[i], pred_heading_residual[i],
                               pred_size_class[i], pred_size_residual[i])
        bbox_mesh = convert_oriented_box_to_trimesh_fmt(obb)
        bbox = np.array(bbox_mesh.vertices)
        bbox = np.array([bbox[0], bbox[1], bbox[3], bbox[2], bbox[4], bbox[5], bbox[7], bbox[6]])

        label = pred_size_class[i]
        # lines = np.array(bbox_mesh.edges_unique)
        lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                 [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                 [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                 [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
        all_bbox.append(bbox)
        all_lines.append(lines)

        # repeat n times for bold the lines
        repeat_num = 20
        for j in range(int(repeat_num / 2)):
            all_bbox.append(bbox - 0.001 * (j + 1))
            all_bbox.append(bbox + 0.001 * (j + 1))

        for n in range(repeat_num):
            count = count + 1
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
            all_lines.append(lines)

        _color_rgb = label2color[label]
        all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
        count += 1

    all_lines = np.array(all_lines).astype(np.int)
    all_lines = np.reshape(all_lines, (-1, 2))

    all_bbox = np.array(all_bbox).astype(np.double)
    all_bbox = np.reshape(all_bbox, (-1, 3))
    all_colors = np.array(all_colors) / 255.0

    line_pcd = open3d.geometry.LineSet()
    line_pcd.lines = open3d.utility.Vector2iVector(all_lines)
    line_pcd.colors = open3d.utility.Vector3dVector(all_colors)
    line_pcd.points = open3d.utility.Vector3dVector(all_bbox)

    scene_points = scene_point_cloud[:, :3]
    scene_colors = scene_point_cloud[:, 3:6] / 255.0
    scene_pts = open3d.geometry.PointCloud()
    scene_pts.points = open3d.utility.Vector3dVector(scene_points)
    scene_pts.colors = open3d.utility.Vector3dVector(scene_colors)

    vis = open3d.Visualizer()
    vis.create_window(window_name='obbs')
    vis.add_geometry(scene_pts)
    vis.add_geometry(line_pcd)
    vis.run()

    if nms:
        pred_mask = pred_dict['pred_mask']
        pred_center = pred_dict['pred_center'][np.where(pred_mask)[0], :]
        pred_heading_class = pred_dict['pred_heading_class'][np.where(pred_mask)[0]]
        pred_heading_residual = pred_dict['pred_heading_residual'][np.where(pred_mask)[0]]
        pred_size_class = pred_dict['pred_size_class'][np.where(pred_mask)[0]]
        pred_size_residual = pred_dict['pred_size_residual'][np.where(pred_mask)[0], :]
        obb_nums = len(pred_center)

        all_bbox = []
        all_lines = []
        all_colors = []
        count = 0
        for i in range(obb_nums):

            obb = config.param2obb(pred_center[i, 0:3], pred_heading_class[i], pred_heading_residual[i],
                                   pred_size_class[i], pred_size_residual[i])
            bbox_mesh = convert_oriented_box_to_trimesh_fmt(obb)
            bbox = np.array(bbox_mesh.vertices)
            bbox = np.array([bbox[0], bbox[1], bbox[3], bbox[2], bbox[4], bbox[5], bbox[7], bbox[6]])

            label = pred_size_class[i]
            # lines = np.array(bbox_mesh.edges_unique)
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
            all_bbox.append(bbox)
            all_lines.append(lines)

            # repeat n times for bold the lines
            repeat_num = 20
            for j in range(int(repeat_num / 2)):
                all_bbox.append(bbox - 0.001 * (j + 1))
                all_bbox.append(bbox + 0.001 * (j + 1))

            for n in range(repeat_num):
                count = count + 1
                lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                         [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                         [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                         [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
                all_lines.append(lines)

            _color_rgb = label2color[label]
            all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
            count += 1

        all_lines = np.array(all_lines).astype(np.int)
        all_lines = np.reshape(all_lines, (-1, 2))

        all_bbox = np.array(all_bbox).astype(np.double)
        all_bbox = np.reshape(all_bbox, (-1, 3))
        all_colors = np.array(all_colors) / 255.0

        line_pcd = open3d.geometry.LineSet()
        line_pcd.lines = open3d.utility.Vector2iVector(all_lines)
        line_pcd.colors = open3d.utility.Vector3dVector(all_colors)
        line_pcd.points = open3d.utility.Vector3dVector(all_bbox)

        scene_points = scene_point_cloud[:, :3]
        scene_colors = scene_point_cloud[:, 3:6] / 255.0
        scene_pts = open3d.geometry.PointCloud()
        scene_pts.points = open3d.utility.Vector3dVector(scene_points)
        scene_pts.colors = open3d.utility.Vector3dVector(scene_colors)

        vis2 = open3d.Visualizer()
        vis2.create_window(window_name='obbs after nms')
        vis2.add_geometry(scene_pts)
        vis2.add_geometry(line_pcd)

        vis2.run()
        vis2.destroy_window()
        vis2.close()

    vis.destroy_window()
    vis.close()


def draw_bbox2(pred_dict, nms=False):

    scene_point_cloud = pred_dict['point_clouds']
    obbs = pred_dict['boxes_3d_with_prob']
    obb_nums = len(obbs)

    all_bbox = []
    all_lines = []
    all_colors = []
    count = 0
    for i in range(obb_nums):

        # x1 = obbs[i][0]
        # y1 = obbs[i][1]
        # z1 = obbs[i][2]
        # x2 = obbs[i][3]
        # y2 = obbs[i][4]
        # z2 = obbs[i][5]
        x1 = obbs[i][0]
        z1 = -obbs[i][1]
        y1 = obbs[i][2]
        x2 = obbs[i][3]
        z2 = -obbs[i][4]
        y2 = obbs[i][5]
        bbox = np.array([[x1, y1, z1],
                         [x1, y1, z2],
                         [x2, y1, z2],
                         [x2, y1, z1],
                         [x2, y2, z1],
                         [x2, y2, z2],
                         [x1, y2, z1],
                         [x1, y2, z2]])

        label = obbs[i][7]
        lines = [[0 + 8 * count, 1 + 8 * count], [1 + 8 * count, 2 + 8 * count], [2 + 8 * count, 3 + 8 * count],
                 [3 + 8 * count, 0 + 8 * count], [2 + 8 * count, 5 + 8 * count], [5 + 8 * count, 4 + 8 * count],
                 [4 + 8 * count, 3 + 8 * count], [5 + 8 * count, 7 + 8 * count], [4 + 8 * count, 6 + 8 * count],
                 [6 + 8 * count, 7 + 8 * count], [6 + 8 * count, 0 + 8 * count], [7 + 8 * count, 1 + 8 * count]]
        all_bbox.append(bbox)
        all_lines.append(lines)

        # repeat n times for bold the lines
        repeat_num = 20
        for j in range(int(repeat_num / 2)):
            all_bbox.append(bbox - 0.001 * (j + 1))
            all_bbox.append(bbox + 0.001 * (j + 1))

        for n in range(repeat_num):
            count = count + 1
            lines = [[0 + 8 * count, 1 + 8 * count], [1 + 8 * count, 2 + 8 * count], [2 + 8 * count, 3 + 8 * count],
                     [3 + 8 * count, 0 + 8 * count], [2 + 8 * count, 5 + 8 * count], [5 + 8 * count, 4 + 8 * count],
                     [4 + 8 * count, 3 + 8 * count], [5 + 8 * count, 7 + 8 * count], [4 + 8 * count, 6 + 8 * count],
                     [6 + 8 * count, 7 + 8 * count], [6 + 8 * count, 0 + 8 * count], [7 + 8 * count, 1 + 8 * count]]
            all_lines.append(lines)

        _color_rgb = label2color[label]
        all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
        count += 1

    all_lines = np.array(all_lines).astype(np.int)
    all_lines = np.reshape(all_lines, (-1, 2))

    all_bbox = np.array(all_bbox).astype(np.double)
    all_bbox = np.reshape(all_bbox, (-1, 3))
    all_colors = np.array(all_colors) / 255.0

    line_pcd = open3d.geometry.LineSet()
    line_pcd.lines = open3d.utility.Vector2iVector(all_lines)
    line_pcd.colors = open3d.utility.Vector3dVector(all_colors)
    line_pcd.points = open3d.utility.Vector3dVector(all_bbox)

    scene_points = scene_point_cloud[:, :3]
    scene_colors = scene_point_cloud[:, 3:6] / 255.0
    scene_pts = open3d.geometry.PointCloud()
    scene_pts.points = open3d.utility.Vector3dVector(scene_points)
    scene_pts.colors = open3d.utility.Vector3dVector(scene_colors)

    vis = open3d.Visualizer()
    vis.create_window(window_name='obbs')
    vis.add_geometry(scene_pts)
    vis.add_geometry(line_pcd)
    vis.run()

    if nms:
        pred_mask = pred_dict['pred_mask']
        pred_center = pred_dict['pred_center'][np.where(pred_mask)[0], :]
        pred_heading_class = pred_dict['pred_heading_class'][np.where(pred_mask)[0]]
        pred_heading_residual = pred_dict['pred_heading_residual'][np.where(pred_mask)[0]]
        pred_size_class = pred_dict['pred_size_class'][np.where(pred_mask)[0]]
        pred_size_residual = pred_dict['pred_size_residual'][np.where(pred_mask)[0], :]
        obb_nums = len(pred_center)

        all_bbox = []
        all_lines = []
        all_colors = []
        count = 0
        for i in range(obb_nums):

            obb = config.param2obb(pred_center[i, 0:3], pred_heading_class[i], pred_heading_residual[i],
                                   pred_size_class[i], pred_size_residual[i])
            bbox_mesh = convert_oriented_box_to_trimesh_fmt(obb)
            bbox = np.array(bbox_mesh.vertices)
            bbox = np.array([bbox[0], bbox[1], bbox[3], bbox[2], bbox[4], bbox[5], bbox[7], bbox[6]])

            label = pred_size_class[i]
            # lines = np.array(bbox_mesh.edges_unique)
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
            all_bbox.append(bbox)
            all_lines.append(lines)

            # repeat n times for bold the lines
            repeat_num = 20
            for j in range(int(repeat_num / 2)):
                all_bbox.append(bbox - 0.001 * (j + 1))
                all_bbox.append(bbox + 0.001 * (j + 1))

            for n in range(repeat_num):
                count = count + 1
                lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                         [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                         [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                         [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
                all_lines.append(lines)

            _color_rgb = label2color[label]
            all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
            count += 1

        all_lines = np.array(all_lines).astype(np.int)
        all_lines = np.reshape(all_lines, (-1, 2))

        all_bbox = np.array(all_bbox).astype(np.double)
        all_bbox = np.reshape(all_bbox, (-1, 3))
        all_colors = np.array(all_colors) / 255.0

        line_pcd = open3d.geometry.LineSet()
        line_pcd.lines = open3d.utility.Vector2iVector(all_lines)
        line_pcd.colors = open3d.utility.Vector3dVector(all_colors)
        line_pcd.points = open3d.utility.Vector3dVector(all_bbox)

        scene_points = scene_point_cloud[:, :3]
        scene_colors = scene_point_cloud[:, 3:6] / 255.0
        scene_pts = open3d.geometry.PointCloud()
        scene_pts.points = open3d.utility.Vector3dVector(scene_points)
        scene_pts.colors = open3d.utility.Vector3dVector(scene_colors)

        vis2 = open3d.Visualizer()
        vis2.create_window(window_name='obbs after nms')
        vis2.add_geometry(scene_pts)
        vis2.add_geometry(line_pcd)

        vis2.run()
        vis2.destroy_window()
        vis2.close()

    vis.destroy_window()
    vis.close()


def draw_relation_pairs(pred_dict, config, opt=None):
    scene_point_cloud = pred_dict['point_clouds']
    obb_nums = len(pred_dict['pred_center'])
    pred_center = pred_dict['pred_center']
    pred_heading_class = pred_dict['pred_heading_class']
    pred_heading_residual = pred_dict['pred_heading_residual']
    pred_size_class = pred_dict['pred_size_class']
    pred_size_residual = pred_dict['pred_size_residual']
    pairs_index = pred_dict['nearest_n_index']
    sem_labels = pred_dict['sem_cls_label']
    rn_labels = pred_dict['rn_label']

    proposal_num, pairs_num = pairs_index.shape
    rn_labels = np.reshape(rn_labels, (proposal_num, pairs_num))

    vis = open3d.VisualizerWithKeyCallback()
    vis.create_window(window_name='obbs')

    if 'scan_name' in pred_dict.keys():
        scan_p = './scannet/scans/' + pred_dict['scan_name'] + '/' + pred_dict['scan_name'] + '_vh_clean_2.ply'
        scene_pcd = open3d.io.read_triangle_mesh(scan_p)
    else:
        scene_points = scene_point_cloud[:, :3]
        scene_colors = scene_point_cloud[:, 3:6] / 255.0
        scene_pcd = open3d.geometry.PointCloud()
        scene_pcd.points = open3d.utility.Vector3dVector(scene_points)
        scene_pcd.colors = open3d.utility.Vector3dVector(scene_colors)
    vis.add_geometry(scene_pcd)

    i = 0
    line_pcd = None

    def next_callback(_vis):
        # for i in range(obb_nums):
        nonlocal i, line_pcd
        if i>0:
            _vis.remove_geometry(line_pcd)

        all_bbox = []
        all_lines = []
        all_colors = []
        count = 0

        obb_i = config.param2obb(pred_center[i, 0:3], pred_heading_class[i], pred_heading_residual[i],
                               pred_size_class[i], pred_size_residual[i])
        bbox_mesh_i = convert_oriented_box_to_trimesh_fmt(obb_i)
        bbox_i = np.array(bbox_mesh_i.vertices)
        bbox_i = np.array([bbox_i[0], bbox_i[1], bbox_i[3], bbox_i[2], bbox_i[4], bbox_i[5], bbox_i[7], bbox_i[6]])
        label_i = sem_labels[i]
        pairs_label = pairs_index.shape[1]

        pred_cls = pred_size_class[i]
        print("label: {},   pred: {}, relation_pairs: {}".format(list(filter(lambda  k: config.type2class.get(k)==label_i, config.type2class.keys())),
                                             list(filter(lambda  k: config.type2class.get(k)==pred_cls, config.type2class.keys())),
                                                                 pairs_label))
        print("rn label:{}".format(rn_labels[i]))

        # lines = np.array(bbox_mesh.edges_unique)
        lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                 [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                 [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                 [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
        all_bbox.append(bbox_i)
        all_lines.append(lines)

        # repeat n times for bold the lines
        repeat_num = 96
        gap = 0.003
        for r in range(int(repeat_num / 6)):
            if r >= int(repeat_num / 12):
                all_bbox.append(bbox_i - [np.array([gap, 0, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                all_bbox.append(bbox_i + [np.array([gap, 0, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                all_bbox.append(bbox_i - [np.array([0, gap, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                all_bbox.append(bbox_i + [np.array([0, gap, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                all_bbox.append(bbox_i - [np.array([0, 0, gap]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                all_bbox.append(bbox_i + [np.array([0, 0, gap]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
            else:
                all_bbox.append(bbox_i - [np.array([gap, 0, 0]) * (r + 1)] * 8)
                all_bbox.append(bbox_i + [np.array([gap, 0, 0]) * (r + 1)] * 8)
                all_bbox.append(bbox_i - [np.array([0, gap, 0]) * (r + 1)] * 8)
                all_bbox.append(bbox_i + [np.array([0, gap, 0]) * (r + 1)] * 8)
                all_bbox.append(bbox_i - [np.array([0, 0, gap]) * (r + 1)] * 8)
                all_bbox.append(bbox_i + [np.array([0, 0, gap]) * (r + 1)] * 8)

        for n in range(repeat_num):
            count = count + 1
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
            all_lines.append(lines)

        # _color_rgb = label2color[pred_cls]
        _color_rgb = [178, 34, 34]
        all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
        count += 1

        for j in range(pairs_index.shape[1]):
            if rn_labels[i][j] == 0 or pairs_index[i][j] == i:
                continue

            _idx_j = pairs_index[i][j]
            obb_j = config.param2obb(pred_center[_idx_j, 0:3], pred_heading_class[_idx_j], pred_heading_residual[_idx_j],
                                     pred_size_class[_idx_j], pred_size_residual[_idx_j])
            bbox_mesh_j = convert_oriented_box_to_trimesh_fmt(obb_j)
            bbox_j = np.array(bbox_mesh_j.vertices)
            bbox_j = np.array([bbox_j[0], bbox_j[1], bbox_j[3], bbox_j[2], bbox_j[4], bbox_j[5], bbox_j[7], bbox_j[6]])

            label = sem_labels[_idx_j]
            print(list(filter(lambda  k: config.type2class.get(k)==label, config.type2class.keys())))
            # lines = np.array(bbox_mesh.edges_unique)
            lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                     [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                     [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                     [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
            all_bbox.append(bbox_j)
            all_lines.append(lines)

            repeat_num = 24
            gap = 0.001
            for r in range(int(repeat_num / 6)):
                if r >= int(repeat_num / 12):
                    all_bbox.append(bbox_j - [np.array([gap, 0, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                    all_bbox.append(bbox_j + [np.array([gap, 0, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                    all_bbox.append(bbox_j - [np.array([0, gap, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                    all_bbox.append(bbox_j + [np.array([0, gap, 0]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                    all_bbox.append(bbox_j - [np.array([0, 0, gap]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                    all_bbox.append(bbox_j + [np.array([0, 0, gap]) * (r - repeat_num / 12 + 1) * (2 ** 0.5 / 2.0)] * 8)
                else:
                    all_bbox.append(bbox_j - [np.array([gap, 0, 0]) * (r + 1)] * 8)
                    all_bbox.append(bbox_j + [np.array([gap, 0, 0]) * (r + 1)] * 8)
                    all_bbox.append(bbox_j - [np.array([0, gap, 0]) * (r + 1)] * 8)
                    all_bbox.append(bbox_j + [np.array([0, gap, 0]) * (r + 1)] * 8)
                    all_bbox.append(bbox_j - [np.array([0, 0, gap]) * (r + 1)] * 8)
                    all_bbox.append(bbox_j + [np.array([0, 0, gap]) * (r + 1)] * 8)

            for n in range(repeat_num):
                count = count + 1
                lines = [[0 + 8 * count, 1 + 8 * count], [0 + 8 * count, 3 + 8 * count], [0 + 8 * count, 4 + 8 * count],
                         [1 + 8 * count, 2 + 8 * count], [1 + 8 * count, 5 + 8 * count], [4 + 8 * count, 5 + 8 * count],
                         [5 + 8 * count, 6 + 8 * count], [2 + 8 * count, 6 + 8 * count], [6 + 8 * count, 7 + 8 * count],
                         [2 + 8 * count, 3 + 8 * count], [7 + 8 * count, 3 + 8 * count], [7 + 8 * count, 4 + 8 * count]]
                all_lines.append(lines)

            # _color_rgb = label2color[label]
            _color_rgb = [0, 0, 255]
            all_colors += ([_color_rgb] * 12 * (repeat_num + 1))
            count += 1

        all_lines = np.array(all_lines).astype(np.int)
        all_lines = np.reshape(all_lines, (-1, 2))

        all_bbox = np.array(all_bbox).astype(np.double)
        all_bbox = np.reshape(all_bbox, (-1, 3))
        all_colors = np.array(all_colors) / 255.0

        line_pcd = open3d.geometry.LineSet()
        line_pcd.lines = open3d.utility.Vector2iVector(all_lines)
        line_pcd.colors = open3d.utility.Vector3dVector(all_colors)
        line_pcd.points = open3d.utility.Vector3dVector(all_bbox)

        _vis.add_geometry(line_pcd)

        i += 1

    def quit_callback(_vis):
        nonlocal i
        i = obb_nums

    vis.register_key_callback(ord("N"), next_callback)
    vis.register_key_callback(ord("Q"), quit_callback)

    while True:
        if i == obb_nums:
            vis.close()
            vis.destroy_window()
            return

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()


def draw_confusion_matrix(sem_label, sem_pred, DATASET_CONFIG, relation_pair, title=None):
    classid2name = DATASET_CONFIG.type2class
    classid2name = sorted(classid2name.items(), key=lambda kv: (kv[1], kv[0]))
    labels_ids = [int(item[1]) for item in classid2name]
    labels_names = [str(item[0]) for item in classid2name]

    import scikitplot as skplt
    import matplotlib.pyplot as plt
    y_true = torch.reshape(torch.stack(sem_label, 0), (1, -1)).cpu().numpy()
    y_pred = torch.reshape(torch.stack(sem_pred, 0), (1, -1)).cpu().numpy()
    plot = skplt.metrics.plot_confusion_matrix(y_true[0], y_pred[0], normalize=True,
                                               title='Ours(relation pair {}) confusion matrix'.format(
                                                   relation_pair),
                                               figsize=(12, 12))
    plt.xticks(labels_ids, labels_names, rotation=90)
    plt.yticks(labels_ids, labels_names)
    # plt.show()
    if title:
        plt.savefig('./ours_pair{}_{}_confusion_matrix.png'.format(relation_pair, title), dpi=300)
    else:
        plt.savefig('./ours_pair{}_confusion_matrix.png'.format(relation_pair), dpi=300)