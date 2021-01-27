import pickle
import numpy as np
import os
import shutil
import argparse
from tqdm import tqdm

ROOT = '../../data/'
LOCAL_DIR = os.path.join(ROOT, 'local_data')


def pickleFile(db, file):
    with open(file, "wb") as dbFile:
        pickle.dump(db, dbFile)


def save_data_to_local():

    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', '-o', help='', type=bool, default=False)
    parser.add_argument('--iou_threshold', '-iou', help='', type=float, default=0.5)
    args = parser.parse_args()

    if args.overwrite:
        # Clean the files generated before
        list_dir = os.listdir(LOCAL_DIR)
        if len(list_dir) != 0:
            for dir in list_dir:
                shutil.rmtree(os.path.join(LOCAL_DIR, dir))

    SSHF_DIR = os.path.join(ROOT, 'data_release')

    list_dir = os.listdir(SSHF_DIR)
    total_file = 0
    for dir in list_dir:
        filelist = os.path.join(SSHF_DIR, dir, 'data', 'files', 'node_info_yao.txt')
        files = open(filelist)
        lines = files.readlines()
        pps_txt_num = len(lines)

        # If pps .plk generation is interrupted, continue to generate; else skip
        if not os.path.exists(os.path.join(LOCAL_DIR, dir)) or len(os.listdir(os.path.join(LOCAL_DIR, dir))) < pps_txt_num:
            pps_count = 0
            max_ins_label = 0
            positive = 0
            negative = 0

            bar = tqdm(lines)
            bar.set_description("Processing %s" % dir)
            for line in bar:
                pps_obbs = []
                pps_obbsreg = []
                pps_sem_label = -1
                pps_ins_label = -1
                pps_iou = 0.0

                info = line.split(' ')
                info[-1] = info[-1].split('\n')[0]
                pps_count += 1

                # Read the infomation for each proposal
                pst_path = info[0]
                pps_obbs = info[1:9]
                if len(info) > 9:
                    pps_obbsreg = info[9:17]
                    pps_sem_label = int(info[17])
                    pps_ins_label = int(info[18])
                    pps_iou = float(info[19])

                    if int(info[18]) > max_ins_label:
                        max_ins_label = int(info[18])

                # Transfer the remote path to local path
                pst_path = pst_path.split('S3DIS')[-1]
                pst_path = ROOT + pst_path

                # Get the dir for proposal
                pst_dir = [name for name in pst_path.split('/') if 'Area' in name][0]
                pst_dir = os.path.join(ROOT, 'local_data', pst_dir)
                if not os.path.exists(pst_dir):
                    os.mkdir(pst_dir)

                # Get points for the proposal
                data = []
                with open(pst_path) as fpts:
                    count = 0
                    while 1:
                        line = fpts.readline()
                        if not line:
                            break
                        L = line.split(' ')
                        L = [float(i) for i in L]
                        data.append(np.array(L))
                        count = count + 1
                    data = np.array(data)

                if pps_iou > args.iou_threshold:
                    positive += 1
                else:
                    negative += 1

                ori_data_dict = {
                    'pst_data': np.array(data),
                    'pps_obbs': np.array(pps_obbs, dtype=np.float),
                    'pps_obbs_reg': np.array(pps_obbsreg, dtype=np.float),
                    'pps_sem_label': pps_sem_label,
                    'pps_ins_label': pps_ins_label,
                    'pps_iou': pps_iou
                }

                # Save to .pkl
                pst_dir = pst_dir + '/proposal_' + str(pps_count) + '.pkl'
                pickleFile(ori_data_dict, pst_dir)

            print("%-10s is completed! %-10d proposals %-10d instances     P/N: %-5d/%-5d" % (dir, pps_count, max_ins_label, positive, negative))
            #print(dir, " is completed!  ----- ", str(pps_count), " proposals    -----  ",  str(max_ins_label),
            #      " instances  -----  P/N: ", str(positive), "/", str(negative))
            total_file += 1

    print("Totally have ", total_file, " rooms.")


if __name__ == '__main__':
    save_data_to_local()
