import os
import random

ROOT = '../data/'
LOCAL_DATA = os.path.join(ROOT, 'local_data') + '/'
REMOTE_DATA = os.path.join(ROOT, 'data_release')

train_test_dict = {
    1: [[1, 2, 3, 4, 6], [5]],
    2: [[1, 3, 5, 6], [2, 4]],
    3: [[2, 4, 5], [1, 3, 6]]
}


def get_path_and_write_to_file(txt_path, id_list, isRemote=False):
    f = open(txt_path, 'w')

    if not isRemote:
        dirs = os.listdir(LOCAL_DATA)
        for dir in dirs:
            if int(dir.split('.')[0].split('_')[-1]) in id_list:
                for root, dirs, files in os.walk(LOCAL_DATA + dir):
                    for file in files:
                        pkl_path = os.path.join(root, file)
                        f.writelines(pkl_path + '\n')
    else:
        dirs = os.listdir(REMOTE_DATA)
        for dir in dirs:
            if int(dir.split('.')[0].split('_')[-1]) in id_list:
                anno_path = os.path.join(REMOTE_DATA, dir, 'data', 'files', 'node_info_yao.txt')
                f.writelines(anno_path+'\n')

    f.close()


def split(isRemote=False):
    if isRemote:
        txt_to_write = 'split_for_remote'
    else:
        txt_to_write = 'split_for_local'
    split_dir = os.path.join(ROOT, txt_to_write)

    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    for area_idx in range(1, 7):
        print('Split train data and val data for Area%d' % (area_idx))

        # area_idx for test and the rest areas for train
        train_areas = [idx for idx in range(1, 7) if idx != area_idx]
        train_txt = os.path.join(split_dir, 'train_files_for_Area_%d.txt' % area_idx)
        get_path_and_write_to_file(train_txt, train_areas, isRemote=isRemote)

        val_txt = os.path.join(split_dir, 'val_files_Area_%d.txt' % area_idx)
        get_path_and_write_to_file(val_txt, [area_idx], isRemote=isRemote)


def split2(isRemote=False):
    if isRemote:
        txt_to_write = 'split_for_remote'
    else:
        txt_to_write = 'split_for_local'
    split_dir = os.path.join(ROOT, txt_to_write)

    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    for (key, value) in train_test_dict.items():
        print('Split train data and val data for Area%d' % (key))

        train_areas = value[0]
        train_txt = os.path.join(split_dir, 'train_files_for_Area_%d.txt' % key)
        get_path_and_write_to_file(train_txt, train_areas, isRemote=isRemote)

        val_txt = os.path.join(split_dir, 'val_files_for_Area_%d.txt' % key)
        val_areas = value[1]
        get_path_and_write_to_file(val_txt, val_areas, isRemote=isRemote)

if __name__ == '__main__':
    split2(isRemote=False)
