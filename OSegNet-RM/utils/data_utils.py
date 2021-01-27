from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import plyfile
import numpy as np
from matplotlib import cm
from sklearn.neighbors import KDTree
import random
import pickle
from tqdm import tqdm
import time
from sklearn.neighbors import KDTree
import matplotlib
import  matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import random
import multiprocessing

def compute_class_weight(labels):
    print("use sklearn to balance weight")
    from sklearn.utils.class_weight import compute_class_weight
    
    # categories = [i for i in range(14)]
    categories = np.arange(13)
    
    _labels=[]

    for label in labels:
        # print("label:",label)
        if label in categories:
            # print("label:",label,"is in categories")
            _labels.append(int(label))
    # print("_labels:",_labels)
    print("labels:",np.unique(np.array(_labels)))
    weight = compute_class_weight('balanced', categories , _labels)

    train_weight = [weight[index] for index in labels]
    train_weight_np = np.array(train_weight)
    return train_weight_np,weight

def compute_single_class_weight(labels):
    print("use sklearn to balance weight")
    from sklearn.utils.class_weight import compute_class_weight
    
    # categories = [i for i in range(14)]
    categories = np.arange(2)
    # categories = [0,1]
    _labels=[]

    for label in labels:
        # print("label:",label)
        if label in categories:
            # print("label:",label,"is in categories")
            _labels.append(int(label))
    # print("_labels:",_labels)
    print("labels:",np.unique(np.array(_labels)))
    weight = compute_class_weight('balanced', categories , _labels)

    train_weight = [weight[index] for index in labels]
    train_weight_np = np.array(train_weight)
    return train_weight_np, weight

def random_room_files(Area_txt, batchsize):
    
    AREA_ROOM = {}

    filelists=[]

    for proposal in Area_txt:

        Area=proposal[proposal.find('Area'):proposal.find('Area')+6]
        rest_str=proposal[proposal.find('Area'):]
        gangpos=rest_str.find('/')
        room=rest_str[7:gangpos]
        tmp_path=Area+"/"+room+"/"+room+".txt"
        Area_room=Area+"."+room
        if not Area_room in AREA_ROOM.keys():
            AREA_ROOM[Area_room] = []
            AREA_ROOM[Area_room].append(proposal)
        else:
            AREA_ROOM[Area_room].append(proposal)
    for room in AREA_ROOM:
        tmp_files= np.random.choice(AREA_ROOM[room], batchsize)
        for file in tmp_files:
            filelists.append(file)
    #transform to np.array format
    filelists = np.array(filelists)

    return filelists


def all_cls_single_label_dict(Area_txt): #list
    sem_label_dict = {}
    single_label_dict ={}
    for proposal in Area_txt:
        proposal = proposal.split('\n')[0] #.pkl\n remove \n
        # print("proposal:",proposal)
        data = unpickeFile(proposal)
        pps_sem_label = int(data['pps_sem_label']) #
        if (pps_sem_label >=12):
            pps_sem_label-=1
        pps_sem_label-=1
        sem_label_dict[proposal] = pps_sem_label

        iou =  float(data['pps_iou'])
        single_label=0
        if iou>=0.5:
            single_label=1
        single_label_dict[proposal] = single_label
    
    return sem_label_dict, single_label_dict


def balance_classes(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts_max = np.amax(counts)


    repeat_num_avg_unique = counts_max / counts
    repeat_num_avg = repeat_num_avg_unique[inverse]
    repeat_num_floor = np.floor(repeat_num_avg)
    repeat_num_probs = repeat_num_avg - repeat_num_floor
    repeat_num = repeat_num_floor + (np.random.rand(repeat_num_probs.shape[0]) < repeat_num_probs)
    return repeat_num.astype(np.int64)

object_dict = {
            'clutter':    0,
            'wall':       1,
            'door':       2,
            'ceiling':    3,
            'floor':      4,
            'chair':      5,
            'bookcase':   6,
            'board':      7,
            'table':      8,
            'beam':       9,
            'column':     10,
            'window':     11,
            'sofa':       12}



def unpickeFile(file):
    with open(file,"rb") as dbFile:
        data = pickle.load(dbFile)
        dbFile.close()
        return data


def pickleFile(db, file): #save dictionary
    with open(file, "wb") as dbFile:
        pickle.dump(db, dbFile)


def load_local_data(filelists, iou_threshold,return_iou=False, debug=False, return_sem_label=False, return_ins_label=False):

    objectness_label = []
    pps_obbs = []
    pps_obbsreg = []
    pps_sem_label = []
    pps_ins_label = []
    pst_data = []
    pps_iou = []
    dataset = ''

    iou_list = [0.2, 0.3, 0.4, 0.5]
    p_or_n = np.zeros((len(iou_list), 2), dtype=int)  # [0] for positive, [1] for negative

    if 'train' in filelists:
        dataset = 'train'
    elif 'val' in filelists:
        dataset = 'val'

    start_time = time.time()
    pps_count = 0

    if isinstance(filelists, str):
        f = open(filelists)
        lines = f.readlines()

        bar = tqdm(lines)
        bar.set_description("Reading %s data.." % dataset)
    else:
        bar = filelists

    for filelist in bar:
        #if pps_count>3:
        #    break

        # Read the .pkl for each proposal
        
        filelist = filelist.split('\n')[0]
        # print(filelist)
        if debug:
            filelist = filelist.split('local_data')
            filelist = filelist[0] + '/' + 'local_data2' + '/' + filelist[-1]

        data = unpickeFile(filelist)

        # Read the infomation for each proposal
        _pps_obbsreg = data['pps_obbs_reg']
        _pps_sem_label = data['pps_sem_label']
        _pps_ins_label = data['pps_ins_label']
        _pps_iou = data['pps_iou']

        if not _pps_obbsreg.any() and _pps_sem_label == -1 and _pps_ins_label == -1 and _pps_iou == 0.0:
            continue

        pst_data.append(data['pst_data'])
        pps_obbs.append(data['pps_obbs'])
        pps_iou.append(float(data['pps_iou']))

        pps_obbsreg.append(data['pps_obbs_reg'])
        pps_sem_label.append(data['pps_sem_label'])
        pps_ins_label.append(data['pps_ins_label'])

        # Get the label for single object classification
        if _pps_iou < iou_threshold or (_pps_iou == 0 and data['pps_sem_label'] == -1 and data['pps_ins_label'] == -1):
            objectness_label.append(int(0))
        else:
            objectness_label.append(int(1))

        for i, iou in enumerate(iou_list):
            if _pps_iou >= iou:
                p_or_n[i][0] += 1
            else:
                p_or_n[i][1] += 1

        pps_count += 1

    end_time = time.time()

    if isinstance(filelists, str):
        print("Reading ", dataset, " data complete!  Proposals:", pps_count, "   Totally cost:", end_time-start_time)

        for i in range(len(iou_list)):
            print("iou_threshold:{:.2f}    P/N:{:08d}/{:08d}".format(iou_list[i], p_or_n[i][0], p_or_n[i][1]))

    if return_sem_label and return_iou:
        return np.array(pps_iou),np.array(pps_sem_label),

    if return_sem_label:
        return np.array(pps_sem_label)
    if return_ins_label:
        return np.array(pps_ins_label)
    if return_iou:
        return np.array(pps_iou)


    data_dict = {
        'ori_data':np.array(pst_data),
        'pts_data': sample_the_points(np.array(pst_data)),
        'pps_obbs': np.array(pps_obbs),
        'pps_obbs_reg': np.array(pps_obbsreg),
        'pps_sem_label': np.array(pps_sem_label),
        'pps_ins_label': np.array(pps_ins_label),
        'single_object_label': np.array(objectness_label),
        'pps_iou': np.array(pps_iou)
    }

    return data_dict, p_or_n


def load_local_data_and_contdata(filelists, iou_threshold, return_iou=False, debug=False, scale=None):

    objectness_label = []
    pps_obbs = []
    pps_obbsreg = []
    pps_sem_label = []
    pps_ins_label = []
    pst_data = []
    pps_iou = []
    pts_cont_data = []
    dataset = ''
    if debug:
        CONTEXT_ROOT = '../data/local_proposal2'
    else:
        if scale == 0.5:
            CONTEXT_ROOT = '../data/proposal05'
        elif scale == 0.3:
            CONTEXT_ROOT = '../data/proposal03'
        elif scale == 0.7:
            CONTEXT_ROOT = '../data/proposal07'
        elif scale == 1.0:
            CONTEXT_ROOT = '../data/proposal10'
        elif scale == 1.5:
            CONTEXT_ROOT = '../data/proposal15'

    iou_list = [0.2, 0.3, 0.4, 0.5]
    p_or_n = np.zeros((len(iou_list), 2), dtype=int)  # [0] for positive, [1] for negative

    if 'train' in filelists:
        dataset = 'train'
    elif 'val' in filelists:
        dataset = 'val'

    start_time = time.time()
    pps_count = 0

    if isinstance(filelists, str):
        f = open(filelists)
        lines = f.readlines()

        bar = tqdm(lines)
        bar.set_description("Reading %s data.." % dataset)
    else:
        bar = filelists  #real filelist is a list

    for filelist in bar:
        #if pps_count>3:
        #    break

        # Read the .pkl for each proposal
        filelist = filelist.split('\n')[0]

        # Get area_id and proposal_id and context data's path
        location = filelist.split('/')[3] #Area_room
        pps_id = filelist.split('/')[4]
        pps_id = pps_id.split('.')[0]
        pps_id = pps_id + '.pkl'
        cont_path = os.path.join(CONTEXT_ROOT, location, pps_id)

        if debug:
            filelist = filelist.split('local_data')
            filelist = filelist[0] + '/' + 'local_data2' + '/' + filelist[-1]

        data = unpickeFile(filelist)

        # Read the infomation for each proposal
        _pps_obbsreg = data['pps_obbs_reg']
        _pps_sem_label = data['pps_sem_label']
        _pps_ins_label = data['pps_ins_label']
        _pps_iou = data['pps_iou']
        _pps_cont_data = unpickeFile(cont_path)['pts_context_data']

        if not _pps_obbsreg.any() and _pps_sem_label == -1 and _pps_ins_label == -1 and _pps_iou == 0.0:
            continue


        pst_data.append(data['pst_data'])
        pts_cont_data.append(_pps_cont_data)
        pps_obbs.append(data['pps_obbs'])

        pps_iou.append(float(data['pps_iou']))
        pps_ins_label.append(data['pps_ins_label'])
        pps_sem_label.append(data['pps_sem_label'])
        pps_obbsreg.append(data['pps_obbs_reg'])

        # Get the label for single object classification
        if _pps_iou < iou_threshold or (_pps_iou == 0 and data['pps_sem_label'] == -1 and data['pps_ins_label'] == -1):
            objectness_label.append(int(0))
        else:
            objectness_label.append(int(1))

        for i, iou in enumerate(iou_list):
            if _pps_iou >= iou:
                p_or_n[i][0] += 1  #positive
            else:
                p_or_n[i][1] += 1  #negative

        pps_count += 1 

    

    if isinstance(filelists, str):
        print("Reading ", dataset, " data complete!  Proposals:", pps_count, "   Totally cost:", end_time-start_time)

        for i in range(len(iou_list)):
            print("iou_threshold:{:.2f}    P/N:{:08d}/{:08d}".format(iou_list[i], p_or_n[i][0], p_or_n[i][1]))

    if return_iou:
        return np.array(pps_iou)



    data_dict = {
        'ori_data' : np.array(pst_data),
        'pts_data': sample_the_points(np.array(pst_data)),
        'pts_context_data': sample_the_points(np.array(pts_cont_data)),
        'pps_obbs': np.array(pps_obbs),
        'pps_obbs_reg': np.array(pps_obbsreg),
        'pps_sem_label': np.array(pps_sem_label),
        'pps_ins_label': np.array(pps_ins_label),
        'single_object_label': np.array(objectness_label),
        'pps_iou': np.array(pps_iou)
    }
    end_time = time.time()
    # time_for_one_epoch = end_time - start_time
    # print("time_load_data:",time_for_one_epoch)
    return data_dict, p_or_n


def trans_pred_obb(obbs, obbs_reg):
    box_adjust = []
    for count in range(len(obbs)):
        _obb = np.zeros(8)
        _obb = obbs[count]+obbs_reg[count]
        box_adjust.append(_obb)
    return box_adjust

def multi_load(Batchsize, filelists, iou_threshold, stage=1, return_iou=False, debug=False, scale=None):
    start_time = time.time()

    process_num = int(Batchsize/2)
    pool = multiprocessing.Pool(processes=process_num)

    data_list=[]
    for i in range(0,process_num*2,2): #i:0-Batchsize, step=2
        # print(i,i+2)
        data_dict = pool.apply_async(load_local_data_and_contdata_multi, (filelists[i:i+2], i, iou_threshold, stage, return_iou, debug, scale))
        data_list.append(data_dict)
    pool.close()
    pool.join()

    index= []
    real_file = []
    objectness_label = []
    pps_obbs = []
    pps_obbsreg = []
    pps_sem_label = []
    pps_ins_label = []
    pst_data = []
    pps_iou = []
    pts_cont_data = []

    iou_list = [0.2, 0.3, 0.4, 0.5]
    p_or_n = np.zeros((len(iou_list), 2), dtype=int)  # [0] for positive, [1] for negative
    '''
                'pts_data': np.array(pst_data),
                'pts_context_data': np.array(pts_cont_data),
                'pps_obbs': np.array(pps_obbs),
                'pps_obbs_reg': np.array(pps_obbsreg),
                'pps_sem_label': np.array(pps_sem_label),
                'pps_ins_label': np.array(pps_ins_label),
                'single_object_label': np.array(objectness_label),
                'pps_iou': np.array(pps_iou)
    '''
    new_data = []
    for data_dict in data_list:
        real_data=data_dict.get()
        p_n_tmp=real_data[1]
        for i, iou in enumerate(iou_list):
            p_or_n[i][0] += p_n_tmp[i][0]  #positive
            p_or_n[i][1] += p_n_tmp[i][1]  #negative

        for i in range(2):
            # print()
            index.append(real_data[0]["num"]+int(i))
            real_file.append(real_data[0]["real_file"][i])
            pst_data.append(real_data[0]["pts_data"][i])
            pts_cont_data.append(real_data[0]["pts_context_data"][i])
            pps_obbs.append(real_data[0]["pps_obbs"][i])
            pps_obbsreg.append(real_data[0]["pps_obbs_reg"][i])
            pps_sem_label.append(real_data[0]["pps_sem_label"][i])
            pps_ins_label.append(real_data[0]["pps_ins_label"][i])
            objectness_label.append(real_data[0]["single_object_label"][i])
            pps_iou.append(real_data[0]["pps_iou"][i])
        
    #组装
    data_dict = {
    'pts_data': np.array(pst_data),
    'pts_context_data': np.array(pts_cont_data),
    'pps_obbs': np.array(pps_obbs),
    'pps_obbs_reg': np.array(pps_obbsreg),
    'pps_sem_label': np.array(pps_sem_label),
    'pps_ins_label': np.array(pps_ins_label),
    'single_object_label': np.array(objectness_label),
    'pps_iou': np.array(pps_iou)
    }
    # #end
    end_time = time.time()
    time_for_one_epoch = end_time - start_time
    print("time_load_data:",time_for_one_epoch)
    return data_dict, p_or_n

def load_local_filelist(filelists):

    f = open(filelists)
    lines = f.readlines()

    return lines


def load_remote_file(dataset_txt, iou_threshold, stage, load_filelists=True):
    ROOT = '../data/'
    f = open(dataset_txt)
    pps_paths = f.readlines()

    pts_data = []
    pps_obbs = []
    pps_obbsreg = []
    pps_sem_label = []
    pps_ins_label = []
    pps_iou = []
    objectness_label = []

    for path in pps_paths:

        files = open(path.split('\n')[0])
        lines = files.readlines()

        pps_count = 0
        max_ins_label = 0
        positive = 0
        negative = 0
        room_name = path.split('data_release/')[-1].split('/')[0]

        bar = tqdm(lines)
        bar.set_description("Reading %s" % room_name)
        for line in bar:
            info = line.split(' ')
            info[-1] = info[-1].split('\n')[0]
            pps_count += 1

            # Read the infomation for each proposal
            pst_path = info[0]
            obbs = info[1:9]

            if len(info) > 9:
                # Positive samples
                obbsreg = info[9:17]
                sem_label = int(info[17])
                ins_label = int(info[18])
                iou = float(info[19])

                if int(info[18]) > max_ins_label:
                    max_ins_label = int(info[18])
            else:
                # Negative samples
                obbsreg = []
                sem_label = -1
                ins_label = -1
                iou = 0.0

            # Get the label for single object classification
            if iou >= iou_threshold:
                objectness = int(1)
                positive += 1
            else:
                objectness = int(0)
                negative += 1

            pps_obbs.append(np.array(obbs, dtype=np.float))

            objectness_label.append(objectness)

            # Get the path of points
            pst_path = pst_path.split('S3DIS')[-1]
            pst_path = ROOT + pst_path

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
                pts_data.append(data)

        print("%-20s is completed!   proposals:%-5d  instances: %-5d  P/N: %d/%-15d" % (room_name, pps_count, max_ins_label, positive, negative))

    ori_data_dict = {
        'pts_data': np.array(pts_data),
        'single_object_label': objectness_label,
    }
    return ori_data_dict


def getObbVertices(obb):
    points = np.zeros((8,3))
    cen = np.zeros(3)
    leng = np.zeros(3)
    ax1 = np.zeros(3)
    ax2 = np.zeros(3)
    ax3 = np.zeros(3)
    cen[0] = float(obb[3])
    cen[1] = float(obb[4])
    cen[2] = float(obb[5])
    leng[0] = float(obb[0]) * 0.5
    leng[1] = float(obb[1]) * 0.5
    leng[2] = float(obb[2]) * 0.5
    ax1[0] = float(obb[6])
    ax1[1] = float(obb[7])
    ax1[2] = float(0)
    ax3[0] = float(0)
    ax3[1] = float(0)
    ax3[2] = float(1)
    ax2 = np.cross(ax1,ax3)

    offset = np.zeros((8,3))
    offset[0,0] = ax1[0] * leng[0] + ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[0,1] = ax1[1] * leng[0] + ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[0,2] = ax1[2] * leng[0] + ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[1,0] = ax1[0] * leng[0] - ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[1,1] = ax1[1] * leng[0] - ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[1,2] = ax1[2] * leng[0] - ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[2,0] = - ax1[0] * leng[0] - ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[2,1] = - ax1[1] * leng[0] - ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[2,2] = - ax1[2] * leng[0] - ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[3,0] = - ax1[0] * leng[0] + ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[3,1] = - ax1[1] * leng[0] + ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[3,2] = - ax1[2] * leng[0] + ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[4,0] = ax1[0] * leng[0] + ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[4,1] = ax1[1] * leng[0] + ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[4,2] = ax1[2] * leng[0] + ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[5,0] = ax1[0] * leng[0] - ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[5,1] = ax1[1] * leng[0] - ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[5,2] = ax1[2] * leng[0] - ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[6,0] = - ax1[0] * leng[0] - ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[6,1] = - ax1[1] * leng[0] - ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[6,2] = - ax1[2] * leng[0] - ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[7,0] = - ax1[0] * leng[0] + ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[7,1] = - ax1[1] * leng[0] + ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[7,2] = - ax1[2] * leng[0] + ax2[2] * leng[1]  - ax3[2] * leng[2]

    points[0,0] = cen[0] + offset[0,0]
    points[0,1] = cen[1] + offset[0,1]
    points[0,2] = cen[2] + offset[0,2]
    points[1,0] = cen[0] + offset[1,0]
    points[1,1] = cen[1] + offset[1,1]
    points[1,2] = cen[2] + offset[1,2]
    points[2,0] = cen[0] + offset[2,0]
    points[2,1] = cen[1] + offset[2,1]
    points[2,2] = cen[2] + offset[2,2]
    points[3,0] = cen[0] + offset[3,0]
    points[3,1] = cen[1] + offset[3,1]
    points[3,2] = cen[2] + offset[3,2]
    points[4,0] = cen[0] + offset[4,0]
    points[4,1] = cen[1] + offset[4,1]
    points[4,2] = cen[2] + offset[4,2]
    points[5,0] = cen[0] + offset[5,0]
    points[5,1] = cen[1] + offset[5,1]
    points[5,2] = cen[2] + offset[5,2]
    points[6,0] = cen[0] + offset[6,0]
    points[6,1] = cen[1] + offset[6,1]
    points[6,2] = cen[2] + offset[6,2]
    points[7,0] = cen[0] + offset[7,0]
    points[7,1] = cen[1] + offset[7,1]
    points[7,2] = cen[2] + offset[7,2]
    return points

def compute_rn_label_batch(batch_a_pps_info, batch_b_pps_info, relation=None, batch_a_pps_file=None, batch_b_pps_file=None):
    
    rn_label = []
    for i in range(len(batch_a_pps_info['pts_data'])):

        A_sem_label = batch_a_pps_info['pps_sem_label'][i]
        B_sem_label = batch_b_pps_info['pps_sem_label'][i]

        ptsA = batch_a_pps_info['ori_data'][i]
        ptsB = batch_b_pps_info['ori_data'][i]

        A_obbs = batch_a_pps_info['pps_obbs'][i]
        B_obbs = batch_b_pps_info['pps_obbs'][i]

        A_box = getObbVertices(A_obbs)
        B_box = getObbVertices(B_obbs)

        A_ins_label = batch_a_pps_info['pps_ins_label'][i]
        B_ins_label = batch_b_pps_info['pps_ins_label'][i]

        # Get the Area id and room name
        ppa_file = batch_a_pps_file[i]
        A_location = batch_a_pps_file[i].split('/')[3].split('.')
        A_area_id = A_location[0]
        A_room_name = A_location[1]

        ppb_file = batch_b_pps_file[i]
        B_location = batch_b_pps_file[i].split('/')[3].split('.')
        B_area_id = B_location[0]
        B_room_name = A_location[1]

        A_min_z = min(A_box[:,2])
        A_max_z = max(A_box[:,2])
        B_min_z = min(B_box[:,2])
        B_max_z = max(B_box[:,2])
        
        min_A_B_z = min( abs(A_min_z-B_max_z), abs(A_max_z-B_min_z) )

        A_min_x = min(A_box[:,0])
        A_max_x = max(A_box[:,0])
        B_min_x = min(B_box[:,0])
        B_max_x = max(B_box[:,0])
        # print (A_min_x,B_min_x)
        min_A_B_x = min(  abs(A_min_x-B_max_x), abs(A_max_x-B_min_x))

        A_min_y = min(A_box[:,1])
        A_max_y = max(A_box[:,1])
        B_min_y = min(B_box[:,1])
        B_max_y = max(B_box[:,1])
        min_A_B_y = min(  abs(A_min_y-B_max_y), abs(A_max_y-B_min_y) )

        dis_th = 0.1
        iou_th = 0.1

        aabbA = obb2Aabb(A_obbs)
        aabbB = obb2Aabb(B_obbs)
        ious = computeAabbProjectionIou_xyz(aabbA, aabbB)
        #[iou_xy,iou_xz,iou_yz]


        if relation == 0:  # group: have the same instance id

            if A_area_id == B_area_id and A_room_name == B_room_name and A_ins_label == B_ins_label:
                label = 1
            else:
                label = 0

            rn_label.append(label)

        elif relation == 1:  # adjacency: support or hang
            
            label =0

            if  A_ins_label == B_ins_label:
                rn_label.append(0)
                continue
            else:
                if min_A_B_z <= dis_th and ious[0]>iou_th:
                    print("support:",A_sem_label, B_sem_label)
                    label =1 
            rn_label.append(label)
        elif relation == 2: #hang
            label =0
            if  min_A_B_x <= dis_th and  min_A_B_y > dis_th:  #adjacent
                if ious[2]>iou_th:
                    label =1
                    print("hang on:",A_sem_label, B_sem_label)
            if min_A_B_y <= dis_th and  min_A_B_x > dis_th:  #adjacent
                if ious[1]>iou_th:
                    label =1 
                    print("hang on:",A_sem_label, B_sem_label)
            rn_label.append(label)
        elif relation == 3:  # same_as: have the same semantic label
            if A_sem_label == B_sem_label:
                label = 1
            else:
                label = 0
            rn_label.append(label)

        elif relation == 4:  # alignment: functionally meaningful objcect pairs
            functional_pairs_name = [['chair', 'table'],  # [chair, tabel]
                                    ['beam',  'column'],      # [beam, column]
                                    ['board',  'wall']]      # [board, wall]
            functional_pairs = []
            for pair in functional_pairs_name:
                id_0 = list(object_dict.values())[list(object_dict.keys()).index(pair[0])]
                id_1 = list(object_dict.values())[list(object_dict.keys()).index(pair[1])]
                functional_pairs.append([int(id_0), int(id_1)])

            if [A_sem_label, B_sem_label] in functional_pairs or [B_sem_label, A_sem_label] in functional_pairs:
                label = 1
            else:
                label = 0

            rn_label.append(label)
    return np.array(rn_label)

def computeAabbProjectionIou(boxA, boxB):  # size,center
    boxA = AabbFeatureTransformerReverse(boxA)
    boxB = AabbFeatureTransformerReverse(boxB)
    xA = max(boxA[3], boxB[3])
    yA = max(boxA[4], boxB[4])

    xB = min(boxA[0], boxB[0])
    yB = min(boxA[1], boxB[1])

    if xA - xB > 0 or yA - yB > 0:
        interArea = 0
    else:
        interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[0] - boxA[3]) * (boxA[1] - boxA[4])
    boxBArea = (boxB[0] - boxB[3]) * (boxB[1] - boxB[4])
    iou_for_A = interArea / float(boxAArea)
    iou_for_B = interArea / float(boxBArea)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return [iou, iou_for_A, iou_for_B]

def computeAabbProjectionIou_xyz(boxA, boxB):  # size,center
    boxA = AabbFeatureTransformerReverse(boxA)
    boxB = AabbFeatureTransformerReverse(boxB)
    xA = max(boxA[3], boxB[3])
    yA = max(boxA[4], boxB[4])
    zA = max(boxA[5], boxB[5])

    xB = min(boxA[0], boxB[0])
    yB = min(boxA[1], boxB[1])
    zB = min(boxA[2], boxB[2])
    if xA - xB > 0 or yA - yB > 0 or zA - zB > 0:
        interArea_xy = 0
        interArea_xz = 0
        interArea_yz = 0
    else:
        interArea_xy = (xB - xA) * (yB - yA)
        interArea_xz = (xB - xA) * (zB - zA)
        interArea_yz = (yB - yA) * (zB - zA)

    boxAArea_xy = (boxA[0] - boxA[3]) * (boxA[1] - boxA[4])
    boxBArea_xy = (boxB[0] - boxB[3]) * (boxB[1] - boxB[4])

    boxAArea_xz = (boxA[0] - boxA[3]) * (boxA[2] - boxA[5])
    boxBArea_xz = (boxB[0] - boxB[3]) * (boxB[2] - boxB[5])

    boxAArea_yz = (boxA[1] - boxA[4]) * (boxA[2] - boxA[5])
    boxBArea_yz = (boxB[1] - boxB[4]) * (boxB[2] - boxB[5])

    iou_for_A_xy = interArea_xy / float(boxAArea_xy)
    iou_for_B_xy = interArea_xy / float(boxBArea_xy)

    iou_for_A_xz = interArea_xz / float(boxAArea_xz)
    iou_for_B_xz = interArea_xz / float(boxBArea_xz)

    iou_for_A_yz = interArea_yz / float(boxAArea_yz)
    iou_for_B_yz = interArea_yz / float(boxBArea_yz)

    iou_xy=max(iou_for_A_xy,iou_for_B_xy)
    iou_xz=max(iou_for_A_xz,iou_for_B_xz)
    iou_yz=max(iou_for_A_yz,iou_for_B_yz)

    return [iou_xy,iou_xz,iou_yz]

# size,cen to xMax,yMax,zMax,xMin,yMin,zMin
#               0   1    2     3    4    5 
def AabbFeatureTransformerReverse(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[3] + obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[1] = obj_obb_fea_tmp[4] + obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[2] = obj_obb_fea_tmp[5] + obj_obb_fea_tmp[2]*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[3] - obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[4] = obj_obb_fea_tmp[4] - obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[5] = obj_obb_fea_tmp[5] - obj_obb_fea_tmp[2]*0.5
    return obj_obb_fea


def computeAabbIou(boxA,boxB): # size,center
    boxA = AabbFeatureTransformerReverse(boxA)
    boxB = AabbFeatureTransformerReverse(boxB)
    xA = max(boxA[3], boxB[3]) #xmin里最大的那个
    yA = max(boxA[4], boxB[4])
    zA = max(boxA[5], boxB[5])
    xB = min(boxA[0], boxB[0]) #xmax里最小的那个
    yB = min(boxA[1], boxB[1])
    zB = min(boxA[2], boxB[2])

    if xA - xB > 0 or yA - yB > 0 or zA - zB > 0:
        interArea = 0
        return 0
    else:
        interArea = (xB - xA) * (yB - yA) * (zB - zA)

    boxAArea = (boxA[0] - boxA[3]) * (boxA[1] - boxA[4]) * (boxA[2] - boxA[5])
    boxBArea = (boxB[0] - boxB[3]) * (boxB[1] - boxB[4]) * (boxB[2] - boxB[5])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def obb2Aabb(obb):
    aabb = np.zeros(6)
    if abs(obb[6]) > 0.9:
        aabb[0],aabb[1] = obb[0],obb[1]
    elif abs(obb[7]) > 0.9:
        aabb[0],aabb[1] = obb[1],obb[0]
    else:
        aabb[0] = aabb[1] = max(obb[1],obb[0])
    aabb[2] = obb[2]
    aabb[3:6] = obb[3:6]
    return aabb


def computePtsIou(pts0, pts1, obbA, obbB):
    iou = 0
    aabbA = obb2Aabb(obbA)
    aabbB = obb2Aabb(obbB)
    
    iou_aabb = computeAabbIou(aabbA,aabbB)
    
    if iou_aabb < 0.1:
        iou = 0
        return iou
    tree0 = KDTree(pts0[:,:3], leaf_size=2)
    tree1 = KDTree(pts1[:,:3], leaf_size=2)
    count_in0 = 0
    count_in1 = 0
    count_all0 = 0
    count_all1 = 0

    for i in range(pts0.shape[0]):
        #cancel random continue
        # if random.random() > 0.1:
        #     continue
        dist,ind = tree1.query(pts0[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in0 += 1
        count_all0 += 1
    for i in range(pts1.shape[0]):
        # if random.random() > 0.1:
        #     continue
        dist,ind = tree0.query(pts1[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in1 += 1
        count_all1 += 1
    intersection = (count_in0 + count_in1) * 0.5
    union = count_all0 + count_all1 - intersection
    # print("intersection:",intersection)
    # print("union:",union)
    if union == 0:
        iou = 0
    else:
        iou = float(intersection)/union
    # print("ioufinal:",iou)
    return iou


def computeSingleObject(proposals, gt):
    pass


def getPTS(pts_file,colorPara=1):
    fpts = open(pts_file)
    count = 0
    while 1:
        line = fpts.readline()
        if not line:
            break
        count = count + 1
    if count==0:
        return np.zeros(6)
    points = np.zeros((count,6))
    count = 0
    fpts = open(pts_file)
    while 1:
        line = fpts.readline()
        if not line:
            break
        L = line.split(' ')
        points[count,0] = float(L[0])
        points[count,1] = float(L[1])
        points[count,2] = float(L[2])
        points[count,3] = float(L[3])/colorPara
        points[count,4] = float(L[4])/colorPara
        points[count,5] = float(L[5])/colorPara
        count = count + 1
    return points


def draw_pr_curve(labels, scores, save_path):
    
    y_true = np.array(labels)
    y_scores = np.array(scores)

    plt.figure(1)
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.savefig(save_path)

def NMS_on_points_multiprocessing(dict_info, root_dir, categories, class_num):

    import open3d
    def downsample_points(points, ratio):
        _input_points0 = points[:, 0:3]
        _input_point_cloud0 = open3d.geometry.PointCloud()
        _input_point_cloud0.points = open3d.utility.Vector3dVector(_input_points0)
        _input_point_cloud0 = open3d.geometry.voxel_down_sample(_input_point_cloud0, ratio)
        _input_points0 = np.asarray(_input_point_cloud0.points)
        return _input_points0

    def computePtsIou2(pts0, pts1, obbA, obbB):
        iou = 0
        aabbA = obb2Aabb(obbA)
        aabbB = obb2Aabb(obbB)
        iou_aabb = computeAabbIou(aabbA, aabbB)
        if iou_aabb < 0.1:
            iou = 0
            return iou

        _pts = []
        for pts in [pts0, pts1]:
            if len(pts) < 2000:
                ratio = 0.02
            elif 2000 < len(pts) < 10000:
                ratio = 0.03
            elif 10000 < len(pts) < 20000:
                ratio = 0.04
            else:
                ratio = 0.05
            pts = downsample_points(pts, ratio)
            _pts.append(pts)

        pts0 = _pts[0]
        pts1 = _pts[1]
        tree0 = KDTree(pts0[:, :3], leaf_size=2)
        tree1 = KDTree(pts1[:, :3], leaf_size=2)
        count_in0 = 0
        count_in1 = 0
        count_all0 = 0
        count_all1 = 0
        for i in range(pts0.shape[0]):
            # norand
            # if random.random() > 0.1:
            #     continue
            dist, ind = tree1.query(pts0[i:i + 1, :3], k=1)
            if dist[0] < 0.1:
                count_in0 += 1
            count_all0 += 1
        for i in range(pts1.shape[0]):
            # norand
            # if random.random() > 0.1:
            #     continue
            dist, ind = tree0.query(pts1[i:i + 1, :3], k=1)
            if dist[0] < 0.1:
                count_in1 += 1
            count_all1 += 1
        intersection = (count_in0 + count_in1) * 0.5
        union = count_all0 + count_all1 - intersection
        if union == 0:
            iou = 0
        else:
            iou = float(intersection) / union
        return iou

    def compute_ious_between_pts(index_one, flags, visited, labels, pts, obbs, idx):

        for i in idx:
            if i == index_one:
                flags[i] = 1
                visited[i] = 1
                continue
            if visited[i] == 1:
                continue
            if labels[i] != labels[index_one]:
                continue
            iou = computePtsIou2(pts[index_one], pts[i], obbs[index_one], obbs[i])
            IOU = 0.5
            if iou > IOU:
                flags[i] = 0
                visited[i] = 1
        return [flags, visited]
    print('begin...')
    predictions = np.array(dict_info['cls_confidences'])
    obbs = np.array(dict_info['obbs'])
    pts = np.array(dict_info['pts'])
    labels = dict_info['cls_labels']
    ious = dict_info['ious']
    ins = dict_info['ins_label']
    test_f =dict_info["test_f"]

    visited = np.zeros(len(obbs))
    flags = np.ones(len(obbs))

    print('end...')

    countLabelCorrect = 0
    for i in range(len(labels)):
        prediction = predictions[i]
        validFlag = False
        # print('prediction',prediction)
        for category in categories:
            if prediction[category] > 0.01:
                validFlag = True
                break
        if validFlag == False:
            flags[i] = 0
            visited[i] = 1
        else:
            countLabelCorrect += 1
    print('node before NMS', countLabelCorrect)
    if len(obbs) == 0 or countLabelCorrect == 0:
        flags = np.zeros(len(obbs))
        return flags

    while True:
        # print('while')
        max_probs = np.zeros(len(obbs))
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = int(labels[i])
            if visited[i] == 0:
                max_probs[i] = prediction[label]
        index = np.where(max_probs == np.max(max_probs))[0]
        index_one = -1
        for i in range(len(index)):
            if visited[index[i]] == 0:
                index_one = index[i]
                break

        flags = multiprocessing.Array('d', flags)
        visited = multiprocessing.Array('d', visited)

        loop_num = len(predictions)
        _idxes = list(range(loop_num))
        step = int(np.ceil(loop_num/float(16)))

        p1 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[0:step],))
        p2 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step:step * 2],))
        p3 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 2:step * 3],))
        p4 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 3:step * 4],))
        p5 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 4:step * 5],))
        p6 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 5:step * 6],))
        p7 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 6:step * 7],))
        p8 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 7:step * 8],))
        p9 = multiprocessing.Process(target=compute_ious_between_pts,
                                     args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 8:step * 9],))
        p10 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 9:step * 10],))
        p11 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 10:step * 11],))
        p12 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 11:step * 12],))
        p13 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 12:step * 13],))
        p14 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 13:step * 14],))
        p15 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 14:step * 15],))
        p16 = multiprocessing.Process(target=compute_ious_between_pts,
                                      args=(index_one, flags, visited, labels, pts, obbs, _idxes[step * 15:step * 16],))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p9.start()
        p10.start()
        p11.start()
        p12.start()
        p13.start()
        p14.start()
        p15.start()
        p16.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
        p9.join()
        p10.join()
        p11.join()
        p12.join()
        p13.join()
        p14.join()
        p15.join()
        p16.join()

        flags = np.array(flags)
        visited = np.array(visited)

        stopFlag = 1
        num = 0
        for i in range(len(predictions)):
            if visited[i] == 0:
                stopFlag = 0
                num += 1
        if stopFlag:
            break

    print('node after NMS', int(flags.sum()))
    pred_box_file = os.path.join(root_dir, 'pred_boxes.txt')
    fw = open(pred_box_file, 'w')
    correct = wrong = 0
    count1 = 0
    count = len(obbs)
    pts_after_nms = []
    test_f_after_nms = []
    for i in range(count):
        if flags[i] == 0:
            continue
        prediction = predictions[i]
        #segment_id = ids[i]
        pred_array = prediction[0:class_num]
        pre_label = np.where(pred_array == np.max(pred_array))[0]
        obb = obbs[i]
        pts_after_nms.append(pts[i])
        test_f_after_nms.append(test_f[i])
        if pre_label[0] == labels[i]:
            correct += 1
        else:
            wrong += 1
        fw.write(
            '%f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s' \
            % (obb[0], obb[1], obb[2], obb[3], obb[4], obb[5], obb[6], obb[7], labels[i],
               pred_array[0], pred_array[1], pred_array[2], pred_array[3], pred_array[4],
               pred_array[5], pred_array[6], pred_array[7], pred_array[8], pred_array[9],
               pred_array[10], pred_array[11], pred_array[12], ious[i], test_f[i]))
        count1 += 1
    pts_dict = {
        'pts': np.array(pts_after_nms),
        'test_f': np.array(test_f_after_nms)
    }
    fw.close()
    return pts_dict
    #dont save pkl
    # pts_path = os.path.join(root_dir, 'pts_after_nms.pkl')
    # pickleFile(pts_dict, pts_path)



def box2AP2(pts_dict, pred_box_path, category, root_dir, broken=False, area=None):

    area_dict = {
        1: 'Area_1',
        2: 'Area_2',
        3: 'Area_3',
        4: 'Area_4',
        5: 'Area_5',
        6: 'Area_6'
    }

    area_id = area_dict[area]

    valid_pred_pts_list = []
    valid_f_list = []
    prob_list = []

    pts_path = os.path.join(root_dir, 'pts_after_nms.pkl')

    gt_info_root = '../data/data_release/'
    g_gt_path = os.path.join('../data/', 'gt')
    scene_names = os.listdir(gt_info_root)
    scene_names.sort()
    gt_info_path_list = []
    gt_pts_dirs_list = []
    pred_pts_list = pts_dict['pts']
    pred_f_list = pts_dict['test_f']

    gt_pts_dirs = []
    for scene_name in scene_names:

        if area_id not in scene_name:
            continue
        scene_dir = os.path.join(gt_info_root, scene_name)
        gt_info_path = os.path.join(scene_dir, 'gt_info.txt')
        gt_info_path_list.append(gt_info_path)

        gt_pts_num = len(open(gt_info_path, 'r').readlines())
        for i in range(gt_pts_num):   
            gt_pts_dirs.append(os.path.join(g_gt_path, scene_name.split('.')[0], scene_name.split('.')[1], 'Annotations',
                                            'objects_' + str(i) + '.pts'))
    gt_pts_dirs_list = gt_pts_dirs

    # Count the gt == category
    gt_num = 0
    gt_all = 0
    for i in range(len(gt_info_path_list)):
        gt_info_path = gt_info_path_list[i]

        lines = open(gt_info_path, 'r').readlines()
        gt_all += len(lines)
        for line in lines:
            L = line.split()
            L14 =int(L[14])
            if L14>=12:
                L14-=1
            L14 -=1
            if int(L14) == category:
                gt_num += 1
    lines = open(pred_box_path, 'r').readlines()
    # print("lines:",lines)
    pred_num = 0
    ious = []
    for line in lines:
        L = line.split()

        if float(L[9 + category]) > 0.001:  
            pred_num += 1

            prob_list.append(float(L[9 + category]))
            ious.append(float(L[22])) #ori 25
            #ious.append(random.random())

    gt_obbs = np.zeros((gt_num, 8))
    pred_obbs = np.zeros((pred_num, 8))
    pred_probs = np.zeros(pred_num)
    valid_gt_pts_dirs = []  #
    # lines.close()

    gt_num = 0
    count = 0
    for i in range(len(gt_info_path_list)):
        gt_info_path = gt_info_path_list[i]
        lines = open(gt_info_path, 'r').readlines()
        for line in lines:
            L = line.split()
            L14 =int(L[14])
            if L14>=12:
                L14-=1
            L14 -=1
            if int(L14) == category:
                # print("L14:",L14)
                gt_obbs[gt_num, 0] = float(L[6])
                gt_obbs[gt_num, 1] = float(L[7])
                gt_obbs[gt_num, 2] = float(L[8])
                gt_obbs[gt_num, 3] = float(L[9])
                gt_obbs[gt_num, 4] = float(L[10])
                gt_obbs[gt_num, 5] = float(L[11])
                gt_obbs[gt_num, 6] = float(L[12])
                gt_obbs[gt_num, 7] = float(L[13])
                valid_gt_pts_dirs.append(gt_pts_dirs_list[count])
                gt_num += 1
            count += 1
        # lines.close()

    lines = open(pred_box_path, 'r').readlines()
    pred_num = 0
    count = 0
    for line in lines:
        L = line.split()
        if float(L[9 + category]) > 0.001:
            pred_obbs[pred_num, 0] = float(L[0])
            pred_obbs[pred_num, 1] = float(L[1])
            pred_obbs[pred_num, 2] = float(L[2])
            pred_obbs[pred_num, 3] = float(L[3])
            pred_obbs[pred_num, 4] = float(L[4])
            pred_obbs[pred_num, 5] = float(L[5])
            pred_obbs[pred_num, 6] = float(L[6])
            pred_obbs[pred_num, 7] = float(L[7])
            pred_probs[pred_num] = float(L[9 + category])
            valid_pred_pts_list.append(pred_pts_list[count])
            valid_f_list.append(pred_f_list[count])
            pred_num += 1
        count += 1
    # lines.close()
    gt_obbs_list = np.array(gt_obbs)
    pred_obbs_list = np.array(pred_obbs)
    pred_probs_list = np.array(pred_probs)
    valid_gt_pts_dirs_list = np.array(valid_gt_pts_dirs)
    valid_pred_pts_list = np.array(valid_pred_pts_list)
    valid_f_list = np.array(valid_f_list)


    prob_list = np.array(prob_list)
    prob_list.sort()
    prob_list = np.unique(prob_list)

    # comput iou
    gt_num_all = 0
    iou_matrix = np.zeros((gt_obbs.shape[0], pred_obbs_list.shape[0]))

    for j in range(len(gt_obbs_list)):
        gt_obbs = gt_obbs_list[j]
        # pred_probs = pred_probs_list[j]
        valid_gt_pts = valid_gt_pts_dirs_list[j]

        for m in range(pred_obbs_list.shape[0]):
            if valid_gt_pts.split('/')[4] not in valid_f_list[m]: #如果gt和pred不是一个room的话，就跳过，只算同1个room里的iou
                iou_matrix[j, m] = 0
                continue
            pred_obbs = pred_obbs_list[m]
            # print("gt_obbs:",gt_obbs)
            # print("pred_obbs:",pred_obbs)
            iou = computePtsIou(getPTS(valid_gt_pts), valid_pred_pts_list[m], gt_obbs,
                                pred_obbs)
            if iou >0:
                print("iou:",iou)
            iou_matrix[j, m] = iou
        gt_num_all += len(gt_obbs)
    iou_matrix_list = iou_matrix

    recall_list = []
    precision_list = []
    IOU = 0.5
    for i in range(-1, prob_list.shape[0] + 1):
        TP = 0
        FP = 0
        FN = 0
        if i == -1:
            threshold = prob_list[i + 1] * 0.5
        elif i == prob_list.shape[0]:
            threshold = (1 - prob_list[i - 1]) * 0.5 + prob_list[i - 1]
        else:
            threshold = prob_list[i]

        gt_box_num = gt_obbs_list.shape[0]
        pred_box_num = pred_obbs_list.shape[0]

        print("threshold:",threshold)

        # true-pos
        for k in range(gt_obbs_list.shape[0]):
            for m in range(pred_obbs_list.shape[0]):
                iou = iou_matrix_list[k, m]

                if iou > IOU and pred_probs_list[m] >= threshold:
                    TP += 1
                    break

        # false-pos
        for m in range(pred_obbs_list.shape[0]):
            if pred_probs_list[m] < threshold:
                continue
            FPflag = True
            for k in range(gt_obbs_list.shape[0]):
                iou = iou_matrix_list[k, m]
                
                if iou > IOU:
                    FPflag = False
            if FPflag:
                FP += 1
        
        recall = float(TP) / gt_box_num
        if TP + FP > 0:
            precision = float(TP) / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
        print("TP:",TP,"FP:",FP)
    # print("TP:",TP,"FP:",FP)
    recall_list.append(0)
    precision_list.append(1)
    ap = 0
    length = 0
    for i in range(0, len(recall_list) - 1):
        r = recall_list[i]
        p = precision_list[i]
        r_n = recall_list[i + 1]
        p_n = precision_list[i + 1]
        ap += abs(r - r_n) * p_n
        length += (r - r_n)
    print(category, ap, gt_num)

    ap_path = os.path.join(root_dir, 'ap.txt')

    fw = open(ap_path, 'a')
    fw.write('category:%d  iou:%f  ap:%f\n' % (category, IOU, ap))
    fw.close()


def get_room_pkl_dict(filelists):
    room_ids = []
    ROOM_PKL_DICT = {}
    # Get all rooms
    for file in filelists:
        room_name = file.split('/')[3]
        if room_name not in ROOM_PKL_DICT.keys():
            ROOM_PKL_DICT[room_name]=[]
            ROOM_PKL_DICT[room_name].append(file)
        else:
            ROOM_PKL_DICT[room_name].append(file)

    return ROOM_PKL_DICT

# Count unique rooms' num and get the room id of filelists
def count_rooms(filelists):
    room_ids = []
    room_names = []
    # Get all rooms
    for file in filelists:
        room_names.append(file.split('/')[3])
    unique_rooms = np.unique(room_names)
    unique_rooms = np.sort(unique_rooms)

    # Get the room_name and room_id dict
    room_dict={}
    for i, r in enumerate(unique_rooms):
        room_dict[r] = i

    # Get the filelist's room id
    room_count = np.zeros(len(unique_rooms))
    for file in filelists:
        room_name = file.split('/')[3]
        room_id = room_dict[room_name]
        room_ids.append(room_id)

        room_count[room_id] += 1

    # print(np.where(room_ids==0))
    return room_ids, unique_rooms, room_count, room_dict


# Sample the input points before feed to the network
def sample_the_points(points):
    databatch = []
    batch_size = len(points) 
    for i in range(batch_size):
        # Process the pst
        data = points[i]  #each proposal
        count = len(data)
        data = np.array(data)
        data = data[:, :6]
        #color
        if max(data[:, 3])>1 or max(data[:, 4])>1 or max(data[:, 5])>1:
            data[:,3]=data[:,3] / 255
            data[:,4]=data[:,4] / 255
            data[:,5]=data[:,5] / 255
        #
        trans_x = (min(data[:, 0]) + max(data[:, 0])) / 2
        trans_y = (min(data[:, 1]) + max(data[:, 1])) / 2
        trans_z = (min(data[:, 2]) + max(data[:, 2])) / 2
        data = data - [trans_x, trans_y, trans_z, 0.5, 0.5, 0.5]  
        # Sample the points
        if (count >= 2048):
            index = np.random.choice(count, size=2048, replace=False)  
            dataset = data[index, :]
        else:
            # k = random.sample(range(0, count), count)
            index = np.random.choice(count, size=2048, replace=True)
            dataset = data[index, :]
        databatch.append(dataset)
    return databatch


def sample_pairs_for_rooms(room_dict, f_list, room_ids, debug=False, sample_nums=None):
    room_num = np.unique(list(room_dict.values()))
    room_pair_dict = {}
    total_pairs_num = 0
    for room_id in room_num:
        print('begin to process room: ', room_id, '...')
        _ids_for_room_id = np.where(room_ids == room_id)
        f_list_for_room_batch = f_list[_ids_for_room_id]

        if len(f_list_for_room_batch) < 2:
            continue

        sem_labels = load_local_data(f_list_for_room_batch, 0.5, 1, debug=debug, return_sem_label=True)
        relation_paris = np.zeros((len(f_list_for_room_batch), len(f_list_for_room_batch)))
        relation_paris = relation_paris - 1
        for i in range(len(sem_labels)):
            for j in range(len(sem_labels)):
                if i == j:
                    continue
                if sem_labels[i] == sem_labels[j]:
                    relation_paris[i][j] = 1
                else:
                    relation_paris[i][j] = 0

        ori_pair_count = len(sem_labels) * (len(sem_labels) - 1)
        pos_pairs  = np.where(relation_paris == 1)
        neg_pairs = np.where(relation_paris == 0)

        pos_pairs_num = len(pos_pairs[0])
        neg_pairs_num = len(neg_pairs[0])
        if ori_pair_count > (sample_nums*2):
            select_num = sample_nums

        else:
            select_num = ori_pair_count / 2

        if pos_pairs_num < select_num or neg_pairs_num < select_num:
            min_num = min([pos_pairs_num, neg_pairs_num])
            if min_num == 0:
                continue
            select_num = min_num
        print("pos pair num： ", pos_pairs_num)
        print("room id： ", room_id)
        idx = random.sample(range(pos_pairs_num), int(select_num))
        _pos_pairs_x = np.array(pos_pairs[0])[idx]
        _pos_pairs_y = np.array(pos_pairs[1])[idx]

        idx = random.sample(range(neg_pairs_num), int(select_num))
        _neg_pairs_x = np.array(neg_pairs[0])[idx]
        _neg_pairs_y = np.array(neg_pairs[1])[idx]

        _x = np.concatenate((_pos_pairs_x, _neg_pairs_x), axis=0)
        _y = np.concatenate((_pos_pairs_y, _neg_pairs_y), axis=0)
        obj_a_list = f_list_for_room_batch[_x]
        obj_b_list = f_list_for_room_batch[_y]

        room_pair_dict[room_id] = [obj_a_list, obj_b_list]

        total_pairs_num += len(obj_a_list)
        print('room', room_id, ' pairs: ', len(obj_a_list))

    return room_pair_dict, total_pairs_num


def sample_pairs_for_rooms_by_relation(room_dict, f_list, room_ids, debug=False, sample_nums=None, relation=None):
    room_num = np.unique(list(room_dict.values()))  # Get the room id
    room_pair_dict = {}
    total_pairs_num = 0
    for room_id in room_num:  # process the rooms one by one
        print('begin to process room: ', room_id, '...')
        _ids_for_room_id = np.where(room_ids == room_id)
        f_list_for_room_batch = f_list[_ids_for_room_id] #get all the pkl in the same room by room id
        if len(f_list_for_room_batch) < 2:
            continue

        if relation == 0:  # same instance
            sem_labels = load_local_data(f_list_for_room_batch, 0.5, 1, debug=debug, return_ins_label=True)
        elif relation == 2:  # same semantic label
            sem_labels = load_local_data(f_list_for_room_batch, 0.5, 1, debug=debug, return_sem_label=True)

        relation_paris = np.zeros((len(f_list_for_room_batch), len(f_list_for_room_batch)))
        relation_paris = relation_paris - 1
        for i in range(len(sem_labels)):
            for j in range(len(sem_labels)):
                if i == j:
                    continue
                if sem_labels[i] == sem_labels[j]:
                    relation_paris[i][j] = 1
                else:
                    relation_paris[i][j] = 0

        ori_pair_count = len(sem_labels) * (len(sem_labels) - 1)
        pos_pairs = np.where(relation_paris == 1) #pos_pairs: (array([  0,   0,   0, ..., 165, 165, 165]), array([  5,  22,  42, ...,  23, 114, 137]))
        neg_pairs = np.where(relation_paris == 0)

        pos_pairs_num = len(pos_pairs[0])
        neg_pairs_num = len(neg_pairs[0])
        
        if ori_pair_count > (sample_nums*2):
            select_num = sample_nums

        else:
            select_num = ori_pair_count / 2

        if pos_pairs_num < select_num or neg_pairs_num < select_num:
            min_num = min([pos_pairs_num, neg_pairs_num])
            if min_num == 0:
                continue
            select_num = min_num
        print("pos pair num： ", pos_pairs_num)
        print("room id： ", room_id)
        idx = random.sample(range(pos_pairs_num), int(select_num))
        _pos_pairs_x = np.array(pos_pairs[0])[idx]
        _pos_pairs_y = np.array(pos_pairs[1])[idx]

        idx = random.sample(range(neg_pairs_num), int(select_num))
        _neg_pairs_x = np.array(neg_pairs[0])[idx]
        _neg_pairs_y = np.array(neg_pairs[1])[idx]

        _x = np.concatenate((_pos_pairs_x, _neg_pairs_x), axis=0)  #pos: neg = 1: 1
        _y = np.concatenate((_pos_pairs_y, _neg_pairs_y), axis=0)
        obj_a_list = f_list_for_room_batch[_x]
        obj_b_list = f_list_for_room_batch[_y]

        room_pair_dict[room_id] = [obj_a_list, obj_b_list]

        total_pairs_num += len(obj_a_list)
        print('room', room_id, ' pairs: ', len(obj_a_list))

    return room_pair_dict, total_pairs_num

def relation_pair_generate(f_list, sem_labels, sample_nums, relation ): #per room

    if relation == 0: #same category
        relation_paris = np.zeros((len(f_list_for_room_batch), len(f_list_for_room_batch)))
        relation_paris = relation_paris - 1
    for i in range(len(sem_labels)):
        for j in range(len(sem_labels)):
            if i == j:
                continue
            if sem_labels[i] == sem_labels[j]:
                relation_paris[i][j] = 1
            else:
                relation_paris[i][j] = 0

        ori_pair_count = len(sem_labels) * (len(sem_labels) - 1)
        pos_pairs = np.where(relation_paris == 1) #pos_pairs: (array([  0,   0,   0, ..., 165, 165, 165]), array([  5,  22,  42, ...,  23, 114, 137]))
        neg_pairs = np.where(relation_paris == 0)

        pos_pairs_num = len(pos_pairs[0])
        neg_pairs_num = len(neg_pairs[0])
        
        if ori_pair_count > (sample_nums*2):
            select_num = sample_nums

        else:
            select_num = ori_pair_count / 2

        if pos_pairs_num < select_num or neg_pairs_num < select_num:
            min_num = min([pos_pairs_num, neg_pairs_num])
            if min_num == 0:
                continue
            select_num = min_num
        print("pos pair num： ", pos_pairs_num)
        # print("room id： ", room_id)
        idx = random.sample(range(pos_pairs_num), int(select_num))
        _pos_pairs_x = np.array(pos_pairs[0])[idx]
        _pos_pairs_y = np.array(pos_pairs[1])[idx]

        idx = random.sample(range(neg_pairs_num), int(select_num))
        _neg_pairs_x = np.array(neg_pairs[0])[idx]
        _neg_pairs_y = np.array(neg_pairs[1])[idx]

        _x = np.concatenate((_pos_pairs_x, _neg_pairs_x), axis=0)  #pos: neg = 1: 1
        _y = np.concatenate((_pos_pairs_y, _neg_pairs_y), axis=0)
        obj_a_list = f_list_for_room_batch[_x]
        obj_b_list = f_list_for_room_batch[_y]

        return [obj_a_list, obj_b_list]
        # room_pair_dict[room_id] = [obj_a_list, obj_b_list]

        # total_pairs_num += len(obj_a_list)
        # print('room', room_id, ' pairs: ', len(obj_a_list))
    
    # if relation == 1: #hang or support


    # if relation == 2: #same  instance
        
        

def sample_pairs_for_rooms_by_relation2(room_dict, f_list, room_ids, sem_labels, debug=False, sample_nums=None, relation=None):
    room_num = np.unique(list(room_dict.values()))  # Get the room id
    room_pair_dict = {}
    total_pairs_num = 0
    for room_id in room_num:  # process the rooms one by one
        print('begin to process room: ', room_id, '...')
        _ids_for_room_id = np.where(room_ids == room_id)
        f_list_for_room_batch = f_list[_ids_for_room_id] #get all the pkl in the same room by room id
        sem_labels_room_batch = sem_labels[_ids_for_room_id]
        if len(f_list_for_room_batch) < 2:
            continue

        if relation == 0:  # same instance
            sem_labels = sem_labels_room_batch
        elif relation == 2:  # same semantic label
            sem_labels = sem_labels_room_batch

        relation_paris = np.zeros((len(f_list_for_room_batch), len(f_list_for_room_batch)))
        relation_paris = relation_paris - 1
        for i in range(len(sem_labels)):
            for j in range(len(sem_labels)):
                if i == j:
                    continue
                if sem_labels[i] == sem_labels[j]:
                    relation_paris[i][j] = 1
                else:
                    relation_paris[i][j] = 0

        ori_pair_count = len(sem_labels) * (len(sem_labels) - 1)
        pos_pairs = np.where(relation_paris == 1) #pos_pairs: (array([  0,   0,   0, ..., 165, 165, 165]), array([  5,  22,  42, ...,  23, 114, 137]))
        neg_pairs = np.where(relation_paris == 0)

        pos_pairs_num = len(pos_pairs[0])
        neg_pairs_num = len(neg_pairs[0])
        
        if ori_pair_count > (sample_nums*2):
            select_num = sample_nums

        else:
            select_num = ori_pair_count / 2

        if pos_pairs_num < select_num or neg_pairs_num < select_num:
            min_num = min([pos_pairs_num, neg_pairs_num])
            if min_num == 0:
                continue
            select_num = min_num
        print("pos pair num： ", pos_pairs_num)
        print("room id： ", room_id)
        idx = random.sample(range(pos_pairs_num), int(select_num))
        _pos_pairs_x = np.array(pos_pairs[0])[idx]
        _pos_pairs_y = np.array(pos_pairs[1])[idx]

        idx = random.sample(range(neg_pairs_num), int(select_num))
        _neg_pairs_x = np.array(neg_pairs[0])[idx]
        _neg_pairs_y = np.array(neg_pairs[1])[idx]

        _x = np.concatenate((_pos_pairs_x, _neg_pairs_x), axis=0)  #pos: neg = 1: 1
        _y = np.concatenate((_pos_pairs_y, _neg_pairs_y), axis=0)
        obj_a_list = f_list_for_room_batch[_x]
        obj_b_list = f_list_for_room_batch[_y]

        room_pair_dict[room_id] = [obj_a_list, obj_b_list]

        total_pairs_num += len(obj_a_list)
        print('room', room_id, ' pairs: ', len(obj_a_list))

    return room_pair_dict, total_pairs_num