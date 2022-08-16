"""
__author__ = 'linlei'
__project__:dataset
__time__:2022/6/26 9:25
__email__:"919711601@qq.com"
"""
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class WellDataset(Dataset):
    def __init__(self, root_path: str, input_curves: list, output_curves: list, transform: bool = True,
                 total_seqlen: int = 720, effect_seqlen: int = 640,get_file_name: bool = False):
        """
        Generate training dataset
        Args:
            root_path: 曲线保存路径
            input_curves: 输入曲线类型
            output_curves: 输出曲线类型
            transform: 是否使用数据增广
            total_seqlen: 总序列长度
            effect_seqlen: 随机裁剪序列的长度
        """
        super().__init__()
        self.transform = transform
        self.root_path = root_path
        self.input_curves = input_curves
        self.output_curves = output_curves
        self.files_name = os.listdir(root_path)
        self.total_seqlen = total_seqlen
        self.effect_seqlen = effect_seqlen
        self.get_file_name= get_file_name

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, item):
        # read data
        well_curves = h5py.File(name=os.path.join(self.root_path, self.files_name[item]), mode="r")
        conditions = []
        targets = []
        # get input curves
        for curve in self.input_curves:
            conditions.append(well_curves.get(curve))
        # get output curves
        for curve in self.output_curves:
            targets.append(well_curves.get(curve))
        # seq to array
        conditions = np.array(conditions).reshape(len(conditions), self.total_seqlen)
        targets = np.array(targets).reshape(len(targets), self.total_seqlen)
        # data argumentation
        if self.transform:
            conditions, targets = self.transform_(conditions, targets)
        else:
            conditions, targets = conditions[:, :self.effect_seqlen], targets[:, :self.effect_seqlen]
        # array to tensor
        conditions = torch.from_numpy(conditions).type(torch.FloatTensor)
        targets = torch.from_numpy(targets).type(torch.FloatTensor)
        if self.get_file_name:
            return conditions, targets,self.files_name[item]
        else:
            return conditions, targets

    def transform_(self, conditions, targets):
        """
        Random crop
        Args:
            conditions: input seq of net
            targets: output seq of net

        Returns:
            cropped seqs
        """
        index = self.total_seqlen - self.effect_seqlen
        beg_index = np.random.randint(0, index + 1)
        return conditions[:, beg_index:beg_index + self.effect_seqlen], targets[:,
                                                                        beg_index:beg_index + self.effect_seqlen]


