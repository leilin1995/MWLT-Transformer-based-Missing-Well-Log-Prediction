"""
__author__ = 'linlei'
__project__:vision
__time__:2022/6/30 9:19
__email__:"919711601@qq.com"
"""
import os

import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib as mpl
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['Times New Roman'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号’-'显示为方块的问题


def show_compare(file_dir,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files_name=os.listdir(file_dir)
    for file in files_name:
        f=h5py.File(name=os.path.join(file_dir,file),mode="r")
        real=np.array(f.get("real")).reshape(-1,)
        pred=np.array(f.get("pred")).reshape(-1,)
        index=np.arange(len(real))
        plt.figure(figsize=(6,12))
        plt.plot(real,index,"r",label="real")
        plt.plot(pred,index,"b",label="pred")
        plt.legend()
        plt.gca().invert_yaxis()    # 翻转y轴
        plt.gca().xaxis.tick_top()  # 将x tick 放在上方
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel("DEN")
        plt.ylabel("Samples")
        plt.savefig(os.path.join(save_path,file.split(".")[0]+".png"),dpi=300)

def compare_loss(file_dir,save_path):
    files_name = os.listdir(file_dir)
    df={"file":[],"loss":[]}
    # count=0
    for file in files_name:
        f = h5py.File(name=os.path.join(file_dir, file), mode="r")
        real = np.array(f.get("real")).reshape(-1, )
        pred = np.array(f.get("pred")).reshape(-1, )
        loss=np.sqrt(np.sum((real-pred)**2))
        df["file"].append(file)
        df["loss"].append(loss)
        if loss>0.32:
            # if os.path.exists(os.path.join("../data/train",file)):
            os.remove(os.path.join("../data/train",file))
    # print(count)
    # df=pd.DataFrame(df)
    # df.to_csv(save_path)

def sort_rename():
    train_dir="../data/train"
    val_dir="../data/val"
    train_files=os.listdir(train_dir)
    val_files=os.listdir(val_dir)
    for step,file in enumerate(train_files):
        os.rename(os.path.join(train_dir,file),os.path.join(train_dir,"case_"+str(step).zfill(4)+".hdf5"))
    for step,file in enumerate(val_files):
        os.rename(os.path.join(val_dir,file),os.path.join(val_dir,"case_"+str(step).zfill(4)+".hdf5"))

if __name__ == "__main__":
    # compare_loss(file_dir="../result_org/pred_curve_train",save_path="../result_org/val_png")
    # sort_rename()
    show_compare(file_dir="../result/pred_val", save_path="../result/png_val")