"""
__author__ = 'linlei'
__project__:cal_gardner
__time__:2022/7/10 11:31
__email__:"919711601@qq.com"
"""
import numpy as np
import h5py
import os
from sklearn.metrics import r2_score


def cal_RMSE(pred, real):
    MSE = np.mean((pred - real) ** 2)
    RMSE = np.sqrt(MSE)
    return RMSE


def cal_MAPE(pred, real):
    mape = np.mean(abs((real - pred) / real))*100
    return mape


def renormal(data):
    DEN_up = 3.3
    DEN_down = 1.9
    return data * (DEN_up - DEN_down) + DEN_down


def filter_zero(pred, real):
    index = (real == 0).reshape(-1, )
    pred = np.delete(pred, index, axis=1)
    real = np.delete(real, index, axis=1)
    return pred, real


def cal_indictor():
    train_dir = "./pred_train"
    val_dir = "./pred_val"
    # cal train
    train_files = os.listdir(train_dir)
    train_rmse = 0.
    train_mape = 0.
    count = 0
    for file in train_files:
        file_path = os.path.join(train_dir, file)
        f = h5py.File(name=file_path, mode="r")
        pred = np.array(f.get("pred"))
        real = np.array(f.get("real"))
        pred, real = filter_zero(pred, real)
        pred = renormal(pred)
        real = renormal(real)
        f.close()
        real_total = np.sum(real)
        if real_total > 1:
            rmse = cal_RMSE(pred, real)
            mape = cal_MAPE(pred, real)
            train_rmse += rmse
            train_mape += mape
            count += 1
        else:
            print(file)
    train_rmse = train_rmse / count
    train_mape = train_mape / count

    # cal val
    val_files = os.listdir(val_dir)
    val_rmse = 0.
    val_mape = 0.
    count = 0
    for file in val_files:
        file_path = os.path.join(val_dir, file)
        f = h5py.File(name=file_path, mode="r")
        pred = np.array(f.get("pred"))
        real = np.array(f.get("real"))
        pred, real = filter_zero(pred, real)
        pred = renormal(pred)
        real = renormal(real)
        f.close()
        real_total = np.sum(real)
        if real_total >= 1:
            rmse = cal_RMSE(pred, real)
            mape = cal_MAPE(pred, real)
            val_rmse += rmse
            val_mape += mape
            count += 1
        else:
            print(file)
    val_rmse = val_rmse / count
    val_mape = val_mape / count
    save_str = "train RMSE:{:.4f},train mape:{:.4f}\nval RMSE:{:.4f},val mape:{:.4f}".format(train_rmse, train_mape,
                                                                                             val_rmse, val_mape)
    with open("indicators.txt", "w") as f:
        f.write(save_str)
        f.close()


if __name__ == "__main__":
    cal_indictor()
