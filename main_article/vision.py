"""
__author__ = 'linlei'
__project__:vision
__time__:2022/7/25 15:04
__email__:"919711601@qq.com"
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib as mpl
import random

mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False

random.seed(6)

def get_file_name():
    val_file_path="./Transformer/data_normal/val"
    files_list=os.listdir(val_file_path)
    files_name=random.sample(files_list,k=20)
    return files_name

def renormal_den(data):
    DEN_up = 3.3
    DEN_down = 1.9
    return data * (DEN_up - DEN_down) + DEN_down


def plt_well_figure():
    original_dir="./Transformer/data_nonormal/val"
    gardner_dir="./Gardner/pred_val"
    dnn_dir="./DNN/result_normal/pred_val"
    transformer_dir="./Transformer/result_base_normal/pred_val"
    files_name=["case_2273.hdf5","case_75.hdf5"]
    # print(files_name)
    for file in files_name:
        # gardner predict
        f_gardner=h5py.File(name=os.path.join(gardner_dir,file),mode="r")
        gardner_pred=(np.array(f_gardner.get("pred")).reshape(-1,))[:640,]
        f_gardner.close()
        # dnn predict
        f_dnn = h5py.File(name=os.path.join(dnn_dir, file), mode="r")
        dnn_pred = np.array(f_dnn.get("pred")).reshape(-1,)
        f_dnn.close()
        # base transformer predict
        f_transformer = h5py.File(name=os.path.join(transformer_dir, file), mode="r")
        transformer_pred = np.array(f_transformer.get("pred")).reshape(-1,)
        f_transformer.close()

        # renormalize
        dnn_pred=renormal_den(dnn_pred)
        transformer_pred=renormal_den(transformer_pred)

        # original_data
        f = h5py.File(name=os.path.join(original_dir, file), mode="r")
        AC = (np.array(f["AC"]).reshape(-1,))[:640]
        DEN = (np.array(f["DEN"]).reshape(-1,))[:640]
        GR = (np.array(f["GR"]).reshape(-1,))[:640]
        RLLD = (np.array(f["RLLD"]).reshape(-1,))[:640]
        RLLD=10**RLLD
        CNL = (np.array(f["CNL"]).reshape(-1,))[:640]
        DEPTH = (np.array(f["DEPTH"]).reshape(-1,))[:640]
        f.close()

        # plt_figure
        plt.figure(figsize=(16, 10))
        # figure 1
        ax1 = plt.subplot(171)
        ax1.plot(GR, DEPTH, "b")
        ax1.invert_yaxis()  # reverse y
        ax1.set_xlabel("GR (API)", size="16")
        ax1.set_ylabel("Depth (m)", size="16")
        ax1.set_xlim([45, 155])
        ax1.set_xticks([50, 100, 150])
        ax1.xaxis.tick_top()  # xtick top
        plt.tick_params(labelsize=16)
        ax1.xaxis.set_label_position('top')  # top xlabel

        # figure 2
        ax2 = plt.subplot(172)
        ax2.plot(AC, DEPTH, "b")
        ax2.invert_yaxis()  # reverse y
        ax2.set_xlabel("AC (us/m)", size="16")
        ax2.xaxis.tick_top()  # xtick top
        ax2.set_yticks([])
        plt.tick_params(labelsize=16)
        ax2.xaxis.set_label_position('top')  # top xlabel

        # figure3
        ax3 = plt.subplot(173)
        ax3.plot(CNL, DEPTH, "b")
        ax3.invert_yaxis()  # reverse y
        ax3.set_xlabel("CNL (%)", size="16")
        ax3.set_xlim([2, 47])
        ax3.set_xticks([10, 20, 30,40])
        ax3.xaxis.tick_top()  # xtick top
        ax3.set_yticks([])
        plt.tick_params(labelsize=16)
        ax3.xaxis.set_label_position('top')  # top xlabel

        # figure4
        ax4 = plt.subplot(174)
        ax4.plot(RLLD, DEPTH, "b")
        ax4.invert_yaxis()  # reverse y
        ax4.set_xlabel("RLLD (Ω·m)", size="16")
        ax4.set_yticks([])
        ax4.xaxis.tick_top()  # xtick top
        ax4.set_xscale('log')  # log x

        ax4.set_xticks([1, 10, 100, 1000])
        plt.tick_params(labelsize=16)
        ax4.xaxis.set_label_position('top')  # top xlabel

        # figure 5
        ax5 = plt.subplot(175)
        ax5.plot(DEN, DEPTH, c="b", label="True")
        ax5.plot(transformer_pred, DEPTH, c="r", label="Predict")
        ax5.set_xticks([2, 2.5, 3])
        ax5.set_xlim([1.95, 3.05])
        ax5.invert_yaxis()  # reverse y
        ax5.set_yticks([])
        ax5.set_xlabel("MWLT-Base (g/cm\u00b3)", size="16")
        ax5.xaxis.tick_top()  # xtick top
        plt.tick_params(labelsize=16)
        ax5.xaxis.set_label_position('top')  # top xlabel

        # figure 6
        ax6 = plt.subplot(176)
        ax6.plot(DEN, DEPTH, c="b")
        ax6.plot(dnn_pred, DEPTH, c="r")
        ax6.invert_yaxis()  # reverse y
        ax6.set_xticks([2, 2.5, 3])
        ax6.set_xlim([1.95, 3.05])
        ax6.set_yticks([])
        ax6.set_xlabel("FCDNN (g/cm\u00b3)", size="16")
        ax6.xaxis.tick_top()  # xtick top
        plt.tick_params(labelsize=16)
        ax6.xaxis.set_label_position('top')  # top xlabel

        # figure 7
        ax7 = plt.subplot(177)
        ax7.plot(DEN, DEPTH, c="b",label="True")
        ax7.plot(gardner_pred, DEPTH, c="r",label="Predict")
        plt.legend(loc="upper right")
        ax7.invert_yaxis()  # reverse y
        ax7.set_xticks([2, 2.5, 3])
        ax7.set_xlim([1.95, 3.05])
        ax7.set_yticks([])
        ax7.set_xlabel("Gardner (g/cm\u00b3)", size="16")
        ax7.xaxis.tick_top()  # xtick top
        plt.tick_params(labelsize=16)
        ax7.xaxis.set_label_position('top')  # top xlabel

        plt.subplots_adjust(left=0.07,
                            right=0.98,
                            top=0.9,
                            bottom=0.03,
                            wspace=0.15,
                            hspace=0.24)

        plt.savefig(file.split(".")[0]+",png",dpi=300)

def plt_crossplot():
    # pass
    files_name=get_file_name()
    original_dir = "./Transformer/data_nonormal/val"
    gardner_dir = "./Gardner/pred_val"
    dnn_dir = "./DNN/result_normal/pred_val"
    transformer_dir = "./Transformer/result_base_normal/pred_val"
    true=[]
    gardner=[]
    dnn=[]
    transformer=[]
    for file in files_name:
        # gardner predict
        f_gardner = h5py.File(name=os.path.join(gardner_dir, file), mode="r")
        gardner_pred = (np.array(f_gardner.get("pred")).reshape(-1, ))[:640, ]
        f_gardner.close()
        # dnn predict
        f_dnn = h5py.File(name=os.path.join(dnn_dir, file), mode="r")
        dnn_pred = np.array(f_dnn.get("pred")).reshape(-1, )
        f_dnn.close()
        # base transformer predict
        f_transformer = h5py.File(name=os.path.join(transformer_dir, file), mode="r")
        transformer_pred = np.array(f_transformer.get("pred")).reshape(-1, )
        f_transformer.close()

        # renormalize
        dnn_pred = renormal_den(dnn_pred)
        transformer_pred = renormal_den(transformer_pred)
        # original_data
        f = h5py.File(name=os.path.join(original_dir, file), mode="r")
        DEN = (np.array(f["DEN"]).reshape(-1, ))[:640]
        f.close()
        true.extend(list(DEN))
        gardner.extend(list(gardner_pred))
        dnn.extend(list(dnn_pred))
        transformer.extend(list(transformer_pred))
    plt.figure(figsize=(15,5))
    x=[1.9,3.0]
    y=[1.9,3.0]
    plt.subplot(131)
    plt.scatter(true, transformer, s=10,c="b")
    plt.plot(x,y,"black",linewidth=2)
    plt.xlabel("True (g/cm\u00b3)", size="14")
    plt.xlim([1.9, 3])
    plt.ylabel("MWLT-Base Predict (g/cm\u00b3)", size="14")
    plt.ylim([1.9, 3])
    plt.text(1.9, 3.05,"(a)", size="16")
    plt.tick_params(labelsize=14)
    plt.subplots_adjust(left=0.05,
                        right=0.98,
                        top=0.9,
                        bottom=0.15,
                        wspace=0.2,
                        hspace=0.2)
    plt.subplot(132)
    plt.scatter(true, dnn, s=10, c="b")
    plt.plot(x, y, "black", linewidth=2)
    plt.xlabel("True (g/cm\u00b3)", size="14")
    plt.xlim([1.9, 3])
    plt.ylabel("FCDNN Predict (g/cm\u00b3)", size="14")
    plt.ylim([1.9, 3])
    plt.text(1.9, 3.05, "(b)", size="16")
    plt.tick_params(labelsize=14)

    plt.subplot(133)
    plt.scatter(true, gardner, s=10, c="b")
    plt.plot(x, y, "black", linewidth=2)
    plt.xlabel("True (g/cm\u00b3)", size="14")
    plt.xlim([1.9, 3])
    plt.ylabel("Gardner Predict (g/cm\u00b3)", size="14")
    plt.ylim([1.9, 3])
    plt.tick_params(labelsize=14)
    plt.text(1.9, 3.05, "(c)", size="16")
    plt.savefig("val_crossplot.png",dpi=300)


if __name__ == "__main__":
    # plt_well_figure()
    plt_crossplot()