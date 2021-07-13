import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import math
import struct
import os

path_seq = "/home/doggy/SalsaNext/train/tasks/semantic/dataset/semantic_bin/" #資料夾目錄
file_seq = os.listdir(path_seq) #得到資料夾下的所有檔名稱
file_seq = sorted(file_seq)

for seq in file_seq :
    path_file = "/home/doggy/SalsaNext/train/tasks/semantic/dataset/semantic_bin/"+seq+'/'
    file_npy = os.listdir(path_file)
    file_npy = sorted(file_npy)
    
    for npy in file_npy:
        npy_5dim = np.load( path_file + npy )
        bin_file=open( "/home/doggy/SalsaNext/train/tasks/semantic/dataset/semantic_npy_to_bin/" + seq + '/' + npy.replace(".npy", ".bin"), "wb")
        
        for h in range(0,64):
            for w in range(0,2048):
                if npy_5dim[h,w,3] != -1  :
                    "output .bin"
                    bin_file.write(struct.pack("f", float( npy_5dim[h, w, 0] )))
                    bin_file.write(struct.pack("f", float( npy_5dim[h, w, 1] )))
                    bin_file.write(struct.pack("f", float( npy_5dim[h, w, 2] )))
                    bin_file.write(struct.pack("f", float( npy_5dim[h, w, 3] )))
                    bin_file.write(struct.pack("f", float( npy_5dim[h, w, 4] )))    
        bin_file.close()
        #print('\n========== Now we finish sequence ' + seq +' / ' + npy.replace(".npy", ".bin") + ' file ==========\n')
    print('\n========== Now we finish ' + seq + ' sequence ==========\n')
print('EVERY BIN FILE IS DONE')

