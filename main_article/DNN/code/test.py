"""
__author__ = 'linlei'
__project__:test
__time__:2022/6/26 17:33
__email__:"919711601@qq.com"
"""
from utils import *
import torch.nn as nn
from model import *
import argparse
from dataset import WellDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
import os


def main(args):
    # device
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # save file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # prepare data
    # dataset
    test_dataset = WellDataset(root_path=args.test_files_path, input_curves=args.input_curves,
                               output_curves=args.output_curves,
                               transform=args.transform, total_seqlen=args.total_seqlen,
                               effect_seqlen=args.effect_seqlen, get_file_name=True)
    # dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # build model,optimizer,loss
    model = DNN(dropout=args.dropout)
    model = model.to(device)
    model_dict, epoch, loss = load_checkpoint(args.checkpoint_path)
    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            # [1,4,640]-->[640,4]
            conditions, targets, file_name = data
            conditions = torch.squeeze(conditions, dim=0).T
            targets = torch.squeeze(targets, dim=0).T
            conditions = conditions.to(device)
            preds = model(conditions)
            # [640,1]-->[1,640]
            targets = targets.numpy().T
            preds = preds.cpu().numpy().T
            f = h5py.File(name=os.path.join(args.save_path, file_name[0]), mode="w")
            f.create_dataset(name="real", data=targets)
            f.create_dataset(name="pred", data=preds)
            f.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Train a model to predict miss well log curve.")
    parse.add_argument("--device", type=str, default=0, help="The number of the GPU used,eg.0,1,2")
    parse.add_argument("--save_path", type=str, default="../result_nonormal/pred_val", help="Save files during training")
    parse.add_argument("--test_files_path", default="../data_nonormal/val", type=str, help="Path of the training set")
    parse.add_argument("--input_curves", default=["GR","AC","CNL","RLLD"], type=list,
                       help="Type of input curves")
    parse.add_argument("--dropout", type=int, default=0.1, help="The rate of dropout")
    parse.add_argument("--output_curves", default=["DEN"], type=list,
                       help="Type of output curves")
    parse.add_argument("--transform", default=False, type=bool,
                       help="Use data argumentation or not")
    parse.add_argument("--total_seqlen", default=720, type=int, help="Train seq total len")
    parse.add_argument("--effect_seqlen", default=640, type=int, help="Cropped randomly seq len from total seq")
    parse.add_argument("--drop", type=float, default=0.1,
                       help="Drop rate in linear")
    parse.add_argument("--checkpoint_path", type=str, default="../result_normal/best_model.pth",
                       help="Path to pre-trained model")
    args = parse.parse_args()
    main(args)
