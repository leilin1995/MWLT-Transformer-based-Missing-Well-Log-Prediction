"""
__author__ = 'linlei'
__project__:cal_gardner
__time__:2022/7/10 11:31
__email__:"919711601@qq.com"
"""
import numpy as np
import h5py
import os


def cal(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    files_name = os.listdir(data_dir)
    for name in files_name:
        f = h5py.File(name=os.path.join(data_dir, name), mode="r")
        AC = np.array(f.get("AC"))
        DEN = np.array(f.get("DEN"))
        f.close()
        index=(AC<=0.).reshape(-1,)
        flag=sum(index)
        AC=np.delete(AC,index,axis=0)
        DEN =np.delete(DEN,index,axis=0)
        vp = (1 / AC) / 0.3048 * 10 ** 6
        GDN_den = 0.23 * vp ** 0.25
        f_gdn = h5py.File(name=os.path.join(save_dir, name), mode="w")
        f_gdn.create_dataset(name="real", data=DEN)
        f_gdn.create_dataset(name="pred", data=GDN_den)
        f_gdn.close()

def cal_RMSE(pred,real):
    MSE=np.mean((pred-real)**2)
    RMSE=np.sqrt(MSE)
    return RMSE

def MAPE(pred,real):
    mape=np.mean(abs((real-pred)/real))*100
    return mape

def cal_indictor():
    train_dir="./pred_train"
    val_dir = "./pred_val"
    # cal train
    train_files=os.listdir(train_dir)
    train_rmse=0.
    train_mape=0.
    count=0
    for file in train_files:
        file_path=os.path.join(train_dir,file)
        f=h5py.File(name=file_path,mode="r")
        pred=np.array(f.get("pred"))
        real=np.array(f.get("real"))
        f.close()
        real_total=np.sum(real)
        if real_total>1:
            rmse=cal_RMSE(pred,real)
            mape=MAPE(pred,real)
            train_rmse+=rmse
            train_mape+=mape
            count+=1
        else:
            print(file)
    train_rmse=train_rmse/count
    train_mape=train_mape/count
    
    # cal val
    val_files = os.listdir(val_dir)
    val_rmse = 0.
    val_mape = 0.
    for file in val_files:
        file_path = os.path.join(val_dir, file)
        f = h5py.File(name=file_path, mode="r")
        pred = np.array(f.get("pred"))
        real = np.array(f.get("real"))
        f.close()
        rmse = cal_RMSE(pred, real)
        mape = MAPE(pred, real)
        val_rmse += rmse
        val_mape += mape
    val_rmse = val_rmse / len(val_files)
    val_mape = val_mape / len(val_files)
    save_str = "train RMSE:{:.4f},train mape:{:.4f}\nval RMSE:{:.4f},val mape:{:.4f}".format(train_rmse, train_mape,
                                                                                             val_rmse, val_mape)
    with open("indicators.txt", "w") as f:
        f.write(save_str)
        f.close()
if __name__ == "__main__":
    # cal(data_dir="./data_nonormal/train", save_dir="./pred_train")
    # cal(data_dir="./data_nonormal/val", save_dir="./pred_val")
    cal_indictor()