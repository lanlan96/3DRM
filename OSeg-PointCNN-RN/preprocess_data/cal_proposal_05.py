import numpy as np
from matplotlib import cm
from sklearn.neighbors import KDTree
import h5py
import os
import pickle
import re
import sys
import multiprocessing
import time
import math
sys.path.append('..')

def pickleFile(db, file):
    with open(file, "wb") as dbFile:
        pickle.dump(db, dbFile)

def unpickeFile(file):
    with open(file,"rb") as dbFile:
        data = pickle.load(dbFile)
        dbFile.close()
        print ("loaded"+file)
        return data

def load_data(filename, root_dir2):
    path=os.path.join(root_dir2, filename)
    data=unpickeFile(path)
    pst_data = data['pst_data']    ###(741,9)
    print ("data prepared")
    return pst_data

def calcualte_limit(data, n):
    ###calculate the proper limit
    longest=0
    for i in range(0, data.shape[0], n):
        for j in range(i + 1, data.shape[0], int(n/2)):
            point1 = data[i][:3]
            point2 = data[j][:3]
            dist = np.sqrt(np.sum(np.square(point1 - point2)))
            if dist > longest:
                longest=dist
    return longest
    ###

def retriv_rgb(data):
    for i in range(data.shape[0]):
        for j in range(3,6):
            data[i][j]=int(round(data[i][j]*256))
    return data[:, :6]

def compute_dist(filelist, root_dir2, root_dir3, limit):

    for filename in filelist:
        result_name = root_dir3 + "/" + filename[:-4] + '.pkl'    ##/Area_1.conferenceRoom_1/proposal_108.pkl
        if os.path.exists(result_name):
            print (filename+" exists")
            continue
        relation_point=[]
        start_time2 = time.time()
        O_data=load_data(filename, root_dir2)
        cal_gap=O_data.shape[0]
        end_time2 = time.time()
        print ("load cost:", end_time2-start_time2)
        treeO = KDTree(O_data[:, :3], leaf_size=40)
        start_time5 = time.time()

#        threshold=calcualte_limit(O_data, int(cal_gap/50))
 #       print ("threshold : ", threshold)
        end_time5 = time.time()
        print ("limit_Cal cost: ", end_time5 - start_time5)
        global S_data
        start_time3 = time.time()
        for i in range(S_data.shape[0]):
            dist, index=treeO.query(S_data[i:i+1, :3], k=1)
            if dist[0]<=limit:
                relation_point.append(S_data[i])
        end_time3 = time.time()
        print ("cal cost:", end_time3 - start_time3)
        relation_point = np.array(relation_point)

        data = {
            'pts_context_data': relation_point
        }
        pickleFile(data, result_name)
        # np.savetxt(result_name, relation_point, fmt='%.3f', newline='\n')
        print ("*********save "+result_name+"***********")

if __name__ == "__main__":
    """
    Dir tree:
    1. Path of the down sample points of each room for each area. 
    /home/dell/lyq/data/Area/
                            |---- Area_1_down/
                                            | ---------  hallway_1_down.txt (file to store the down sample points of this room)
                                            | ---------  office_1_down.txt
                                            | ---------  xxx.txt    
                                            | ---------  ... 
                            |---- Area_2_down/
                                            | ---------  xxxx.txt
                                            | ---------  xxx.txt
                            |---- Area_3_down/   
                                        .
                                        .
                                        .
                            |---- Area_6_down/
                                            | ---------  xxx.txt
                                            | ---------  ...
                                        
                                                                
    2. Path of the original proposals of each room for each area.
    /home/dell/dy/question-driven-3d-detection/data/local_data/
                                                              | ----- Area_1.conferenceRoom_1/
                                                                                             | ----- proposal1.ply
                                                                                             | ----- proposal2.ply
                                                                                             | ----- proposalN.ply
                                                              | ----- Area_1.hallway_1/
                                                                                      | ----- proposaln.ply
                                                                                      | ----- ...
                                                                            .
                                                                            .
                                                                            .
                                                              | ----- Area_6.pantry_1/
                                                                                      | ----- proposaln.ply
                                                              
    """


    root_path="/home/lthpc/lyq/3DRM/Area" # Path of the down sample points for each Area
    area_list=os.listdir(root_path)
    print("1")
    # area_list = [Area_1_down, Area_2_down, Area_3_down, Area_4_down, Area_5_down, Area_6_down]
    for area in area_list:

        # Dir of down sample points for Area_N
        root_dir=os.path.join(root_path, area)    ##/Area/Area_1_down/
        print(root_dir)
        # Files of down sample points for rooms which belongs to Area_N, such as "office_10_down.txt".
        scene_list=os.listdir(root_dir)
        for scene in scene_list:
            root_dir2 = '/home/lthpc/lyq/3DRM/data/local_data/'+area[:-5]+'.'  # Path of original proposals for each room
            root_dir3 = '/home/lthpc/lyq/3DRM/proposal05/'+area[:-5]+'.'  # Path to save context patches.
            scene_path=os.path.join(root_dir, scene)
            print (scene_path)
            root_dir2=root_dir2+scene[:-9]   ##/Area_1.conferenceRoom_1/
            root_dir3=root_dir3+scene[:-9]   ##/Area_1.conferenceRoom_1/
            # if(os.path.exists(root_dir3)):
            #     continue
            if not os.path.exists(root_dir2):
                continue
            if not (os.path.exists(root_dir3)):
                os.mkdir(root_dir3)

            start_time = time.time()
            filelist = os.listdir(root_dir2)
            start_time4 = time.time()
            S_data=np.loadtxt(scene_path)
            end_time4 = time.time()
            print ("load S_Data cost: ", end_time4 - start_time4)
            m=1

            n = int(math.ceil(len(filelist) / float(m)))
            pool = multiprocessing.Pool(processes=m)
            for i in range(0, len(filelist), n):
                pool.apply_async(compute_dist, (filelist[i: i+n], root_dir2, root_dir3,  0.5, ))
            pool.close()
            pool.join()
            end_time = time.time()
            print ("total cost: ", end_time-start_time)
