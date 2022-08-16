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
    if args.model_type == "small":
        model = MWLT_Small(in_channels=len(args.input_curves), out_channels=len(args.output_curves),
                          feature_num=args.feature_num
                          , use_pe=args.use_pe, drop=args.drop, attn_drop=args.attn_drop,
                          position_drop=args.position_drop)

    if args.model_type == "base":
        model = MWLT_Base(in_channels=len(args.input_curves), out_channels=len(args.output_curves),
                         feature_num=args.feature_num
                         , use_pe=args.use_pe, drop=args.drop, attn_drop=args.attn_drop,
                         position_drop=args.position_drop)

    if args.model_type == "large":
        model = MWLT_Large(in_channels=len(args.input_curves), out_channels=len(args.output_curves),
                          feature_num=args.feature_num
                          , use_pe=args.use_pe, drop=args.drop, attn_drop=args.attn_drop,
                          position_drop=args.position_drop)
    model = model.to(device)
    model_dict, epoch, loss = load_checkpoint(args.checkpoint_path)
    model.load_state_dict(model_dict)
    model.eval()
    count = 0
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            conditions, targets, file_name = data
            conditions = conditions.to(device)
            preds = model(conditions)
            targets = targets.numpy().squeeze(axis=0)
            preds = preds.cpu().numpy().squeeze(axis=0)
            f = h5py.File(name=os.path.join(args.save_path, file_name[0]), mode="w")
            f.create_dataset(name="real", data=targets)
            f.create_dataset(name="pred", data=preds)
            f.close()
            count += 1


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Train a model to predict miss well log curve.")
    parse.add_argument("--device", type=str, default=0, help="The number of the GPU used,eg.0,1,2")
    parse.add_argument("--save_path", type=str, default="../result_base1/pred_val", help="Save files during training")
    parse.add_argument("--test_files_path", default="../data_normal/val", type=str, help="Path of the training set")
    parse.add_argument("--input_curves", default=["GR","AC","CNL","RLLD"], type=list,
                       help="Type of input curves")
    parse.add_argument("--output_curves", default=["DEN"], type=list,
                       help="Type of output curves")
    parse.add_argument("--transform", default=False, type=bool,
                       help="Use data argumentation or not")
    parse.add_argument("--total_seqlen", default=720, type=int, help="Train seq total len")
    parse.add_argument("--effect_seqlen", default=640, type=int, help="Cropped randomly seq len from total seq")
    parse.add_argument("--model_type", type=str, default="base",
                       help="Type of used model.options:'small','base','large'")
    parse.add_argument("--feature_num", type=int, default=64,
                       help="List of feature dimensions in the convolution layer")
    parse.add_argument("--use_pe", type=bool, default=True,
                       help="Use position embedding in network")
    parse.add_argument("--drop", type=float, default=0.1,
                       help="Drop rate in linear")
    parse.add_argument("--attn_drop", type=float, default=0.1,
                       help="Drop rate in self-attention")
    parse.add_argument("--position_drop", type=float, default=0.1,
                       help="Drop rate in position embedding")
    parse.add_argument("--checkpoint_path", type=str, default="../result_base/best_model.pth",
                       help="Path to pre-trained model")
    args = parse.parse_args()
    main(args)
