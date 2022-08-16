"""
__author__ = 'linlei'
__project__:split_data
__time__:2022/7/20 19:28
__email__:"919711601@qq.com"
"""
import os
import shutil
import random
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    random.seed(6)
    path_dir="./train_data_normal"
    files_list=os.listdir(path_dir)
    val_num=round(0.2*len(files_list))
    val_list=random.sample(files_list, k=val_num)
    for file in files_list:
        if file in val_list:
            shutil.copy(os.path.join(path_dir, file), os.path.join("./val",file))
        else:
            shutil.copy(os.path.join(path_dir, file), os.path.join("./train",file))
def test():
    file_path="./train/case_15.hdf5"
    curve_list = ["AC", "DEN", "GR", "RLLD", "CNL"]
    f = h5py.File(name=file_path, mode="r")
    fig = plt.figure(figsize=(15, 18))
    for i in range(1, len(curve_list) + 1):
        ax = fig.add_subplot(1, len(curve_list), i)
        curve_name = curve_list[i - 1]
        curve = np.array(f[curve_name]).reshape(-1, )
        # if curve_name=="RLLD":
        #     curve=np.log10(curve)
        depth = np.array(f["DEPTH"]).reshape(-1, )
        ax.plot(curve, depth)
        ax.set_xlabel(curve_name)
        if i != 1:
            ax.set_yticks([])
        else:
            ax.set_ylabel("DEPTH")
        ax.xaxis.tick_top()
        ax.invert_yaxis()  # reverse y
    plt.savefig("./case15.png", dpi=300)

if __name__ == "__main__":
    test()