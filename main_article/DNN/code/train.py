"""
__author__ = 'linlei'
__project__:train
__time__:2022/7/6 21:20
__email__:"919711601@qq.com"
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DNN
import argparse
from dataset import WellDataset
from utils import *
import pandas as pd

def main(args):
    # device
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # save file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # save hyperparameter
    argsDict = args.__dict__
    with open(args.save_path + "/hyperparameter.txt","w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg,value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)
    # prepare data
    # dataset
    train_dataset=WellDataset(root_path=args.train_files_path,input_curves=args.input_curves,output_curves=args.output_curves,
                              transform=args.transform,total_seqlen=args.total_seqlen,effect_seqlen=args.effect_seqlen)
    val_dataset=WellDataset(root_path=args.val_files_path,input_curves=args.input_curves,output_curves=args.output_curves,
                              transform=args.transform,total_seqlen=args.total_seqlen,effect_seqlen=args.effect_seqlen)

    # dataloader
    train_loader=DataLoader(dataset=train_dataset,batch_size=1,shuffle=True)
    val_loader=DataLoader(dataset=val_dataset,batch_size=1,shuffle=False)
    # build model,optimizer,loss
    model=DNN(dropout=args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criteria = nn.MSELoss().to(device)
    # early stop to avoid overfitting
    early_stopping = EarlyStopping(patience=args.patience, path=args.save_path + "/best_model.pth")
    start_epoch=1
    # continute learning
    if args.continute_train:
        assert args.checkpoint_path==None,"You are missing the path to the pre-trained model"
        model_dict,epoch,loss=load_checkpoint(args.checkpoint_path)
        model.load_state_dict(model_dict)
        best_loss=loss
        start_epoch=epoch+1

    # training
    pd_log={"train_loss":[],"val_loss":[]}
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        train_total_loss=0.
        # train loader
        for step,data in enumerate(train_loader):
            conditions, targets = data
            # [1,4,640] --> [640,4]
            conditions=torch.squeeze(conditions,dim=0).T
            targets=torch.squeeze(targets,dim=0).T
            conditions = conditions.to(device)
            targets = targets.to(device)
            preds=model(conditions)
            loss=criteria(preds,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += loss
        train_loss_epoch = (train_total_loss / len(train_loader)).detach().cpu().numpy()
        # val loader
        model.eval()
        val_total_loss = 0.
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                # [1,4,640] --> [640,4]
                conditions, targets = data
                conditions = torch.squeeze(conditions, dim=0).T
                targets = torch.squeeze(targets, dim=0).T
                conditions = conditions.to(device)
                targets = targets.to(device)
                preds = model(conditions)
                loss = criteria(preds, targets)
                val_total_loss+= loss
            val_loss_epoch=(val_total_loss/len(val_loader)).detach().cpu().numpy()
        pd_log["train_loss"].append(train_loss_epoch)
        pd_log["val_loss"].append(val_loss_epoch)
        state = {"model_state_dict": model.state_dict(),
                 "loss": val_loss_epoch,
                 "epoch": epoch}
        frame_loss = pd.DataFrame(
            data=pd_log,
            index=range(start_epoch, epoch + 1)
        )
        # save loss
        frame_loss.to_csv(args.save_path + "/loss.csv", index_label="Epoch")
        print(
            "Epochs:[{}/{}],train loss:{:.4f},val loss:{:.4f}".format(epoch, args.epochs, train_loss_epoch, val_loss_epoch))
        # save best model
        early_stopping(state, model)
        # early stop
        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == "__main__":
    parse=argparse.ArgumentParser(description="Train a model to predict miss well log curve.")
    parse.add_argument("--device", type=str, default=0, help="The number of the GPU used,eg.0,1,2")
    parse.add_argument("--dropout", type=int, default=0.1, help="The rate of dropout")
    parse.add_argument("--save_path",type = str,default = "../result_normal",help = "Save files during training")
    parse.add_argument("--train_files_path",default = "../data_nonormal/train", type = str,help = "Path of the training set")
    parse.add_argument("--val_files_path",default = "../data_nonormal/val",type = str,help = "Path of the validation set")
    parse.add_argument("--input_curves", default=["GR","AC","CNL","RLLD"], type=list, help="Type of input curves")
    parse.add_argument("--output_curves", default=["DEN"], type=list,
                       help="Type of output curves")
    parse.add_argument("--transform", default=True, type=bool,
                       help="Use data argumentation or not")
    parse.add_argument("--total_seqlen",default=720,type=int,help="Train seq total len")
    parse.add_argument("--effect_seqlen", default=640, type=int, help="Cropped randomly seq len from total seq")
    parse.add_argument("--learning_rate", type=float, default=1e-5,
                       help="The value of the learning rate used for training")
    parse.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parse.add_argument("--continute_train", type=bool, default=False, help="Continue the previous training session")
    parse.add_argument("--checkpoint_path", type=str, default=None,help="Path to pre-trained model")
    parse.add_argument('--patience', type=int, default=150,
                        help="early stopping patience; how long to wait after last time validation loss improved")
    args=parse.parse_args()
    main(args)
