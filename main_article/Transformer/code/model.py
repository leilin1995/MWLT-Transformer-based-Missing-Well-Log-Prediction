"""
__author__ = 'linlei'
__project__:model
__time__:2022/6/26 10:54
__email__:"919711601@qq.com"
"""
import torch
import torch.nn as nn
import math
from thop import profile


class Input_Embedding(nn.Module):
    def __init__(self, in_channels=1, res_num=4, feature_num=64):
        """
        Input_Embedding layer
        Args:
            in_channels: (int) input channels
            res_num: (int) resnet block number
            feature_num: (int) high-level feature number
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=feature_num, kernel_size=11, padding=5, stride=1),
            nn.BatchNorm1d(feature_num),
            nn.ReLU())
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=11, padding=5, stride=1) for i
             in range(res_num)])

    def forward(self, x):
        """
        conv1: [B,C,L]-->[B,feature_num,L]
        rescnn:[B,feature_num,L]-->[B,feature_num,L]
        Args:
            x:input sequence,shape[B,C,L] B:batch size，C：input channels,default=1，L:sequence len

        Returns:
            x:[B,feature_num,L]
        """
        # conv1: [B,C,L]-->[B,feature_num,L]
        x = self.conv1(x)
        # rescnn:[B,feature_num,L]-->[B,feature_num,L]
        for model in self.rescnn:
            x = model(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        """

        Args:
            in_channels: (int)
            out_channels: (int)
            kernel_size: (int)
            padding: (int)
            stride: (int)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """

        Args:
            x: [B,C,L]

        Returns:

        """
        identity = x
        out = self.conv(x)
        return identity + out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, seq_len):
        """
        Position Embedding
        Args:
            d_model: (int) feature dimension
            dropout: (float) drop rate
            seq_len: (int) sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model).float()
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x:[B,L,C] batch_size,sequence len,channel number

        Returns:

        """
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                                        requires_grad=False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """

        Args:
            dim: (int) feature dimension
            num_heads: (int) mutil heads number
            mlp_ratio: (float) hidden layer ratio in feedforward
            qkv_bias: (bool) use bias or not in qkv linear
            drop: (float) feedforward dropout
            attn_drop: (float) attention dropout
            act_layer: activation layer default (nn.GELU)
            norm_layer: nomrmalization layer default (nn.LayerNorm)
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feedforward = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                                       drop=drop)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.feedforward(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, block_num, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block = nn.ModuleList(
            [TransformerEncoder(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer) for i in
             range(block_num)])

    def forward(self, x):
        for model in self.block:
            x = model(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """

        Args:
            dim: (int) feature dimension
            num_heads: (int) mutil heads number
            qkv_bias:  (bool) use bias or not in qkv linear
            attn_drop: (float) attention drop rate
            proj_drop: (float) linear drop rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """

        Args:
            x: [B,L,C] batch_size,sequence len,channel number

        Returns:

        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """

        Args:
            in_features: (int) input feature number
            hidden_features: (int) hidden_features number
            out_features: (int) output feature number
            act_layer: activation function
            drop: drop rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, res_num=4, out_channels=1, feature_num=64):
        """

        Args:
            out_channels: (int) output feature number
            res_num: (int) resnet block number
            feature_num: (int) feature dimension
        """
        super().__init__()
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=11, padding=5, stride=1) for
             i in range(res_num)])
        self.out_layer = nn.Sequential(
            nn.Conv1d(in_channels=feature_num, out_channels=out_channels, kernel_size=11, padding=5, stride=1),
            nn.Sigmoid()    # normal
            # nn.ReLU()   # nonormal
        )

    def forward(self, x):
        """

        Args:
            x: input sequence,shape[B,C,L] B:batch size，C：特征维度,default=1，L序列长度

        Returns:

        """
        for model in self.rescnn:
            x = model(x)
        x = self.out_layer(x)
        return x


class MWL_Transformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_num=64, res_num=4, encoder_num=4, use_pe=True,
                 dim=64, seq_len=160, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 position_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm
                 ):
        """

        Args:
            in_channels: (int) 输入特征数 default=1
            out_channels: (int) 输出特征数 default=1
            channel_list: (list) 卷积层映射高维特征列表 default=[16,32,64]
            res_num: (int) rescnn中卷积块的个数,default=4
            encoder_num: (int) transformer encoder 中encoder block 的个数,default=4
            use_pe: (bool) 是否使用位置编码，default=True
            dim: (int) input embedding之后高维特征的维度 default=64
            seq_len: (int) input embedding之后高维特征的序列长度 default=160
            num_heads: (int) 多头注意力的头数目,default=4
            mlp_ratio: (float) feedforward中hidden layer的节点个数与输入特征的比例
            qkv_bias: (bool) use bias or not in qkv linear,default=False
            drop: (float) drop rate in linear,default=0.
            attn_drop: (float) drop rate in self-attention,default=0.
            position_drop: (float) drop rate in position embedding,default=0.
            act_layer: self-attention中的激活函数,default=nn.GELU
            norm_layer: transformer encdoer中的标准化层,default=nn.LayerNorm
        """
        super().__init__()
        self.feature_embedding = Input_Embedding(in_channels=in_channels, feature_num=feature_num, res_num=res_num)
        self.position_embedding = PositionalEncoding(d_model=dim, dropout=position_drop, seq_len=seq_len)
        self.use_pe = use_pe
        self.transformer_encdoer = TransformerBlock(block_num=res_num, dim=dim, num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                                    attn_drop=attn_drop,
                                                    act_layer=act_layer, norm_layer=norm_layer)
        self.decoder = Decoder(out_channels=out_channels, feature_num=feature_num, res_num=res_num)

    def forward(self, x):
        """

        Args:
            x: input sequence,shape[B,C,L] B:batch size，C：特征维度,default=1，L序列长度

        Returns:

        """
        # [B,in_channels,L] --> [B,feature_num,L]
        x = self.feature_embedding(x)
        # [B,feature_num,L]--> [B,L,feature_num]
        x = x.transpose(-2, -1)
        if self.use_pe:
            x = self.position_embedding(x)
        # [B, L, feature_num] --> [B, L, feature_num]
        x = self.transformer_encdoer(x)
        # [B,  L, feature_num] --> [B, feature_num,L]
        x = x.transpose(-2, -1)
        # [[B, feature_num,L] --> [B,out_channels,L]
        x = self.decoder(x)
        return x


# define Miss Well Logging Transformer Block
def MWLT_Small(in_channels=1, out_channels=1, feature_num=64, res_num=2, encoder_num=2, use_pe=True,
               dim=64, seq_len=640, num_heads=2, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., position_drop=0.,
               act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    model = MWL_Transformer(in_channels=in_channels, out_channels=out_channels, feature_num=feature_num,
                            res_num=res_num, encoder_num=encoder_num, use_pe=use_pe,
                            dim=dim, seq_len=seq_len, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            drop=drop, attn_drop=attn_drop, position_drop=position_drop,
                            act_layer=act_layer, norm_layer=norm_layer)
    return model


def MWLT_Base(in_channels=1, out_channels=1, feature_num=64, res_num=4, encoder_num=4, use_pe=True,
              dim=64, seq_len=640, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., position_drop=0.,
              act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    model = MWL_Transformer(in_channels=in_channels, out_channels=out_channels, feature_num=feature_num,
                            res_num=res_num, encoder_num=encoder_num, use_pe=use_pe,
                            dim=dim, seq_len=seq_len, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            drop=drop, attn_drop=attn_drop, position_drop=position_drop,
                            act_layer=act_layer, norm_layer=norm_layer)
    return model


def MWLT_Large(in_channels=1, out_channels=1, feature_num=128, res_num=6, encoder_num=6, use_pe=True,
               dim=128, seq_len=640, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., position_drop=0.,
               act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    model = MWL_Transformer(in_channels=in_channels, out_channels=out_channels, feature_num=feature_num,
                            res_num=res_num, encoder_num=encoder_num, use_pe=use_pe,
                            dim=dim, seq_len=seq_len, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            drop=drop, attn_drop=attn_drop, position_drop=position_drop,
                            act_layer=act_layer, norm_layer=norm_layer)
    return model


def claculate_flop_param():
    net_Base = MWLT_Base(in_channels=4)
    net_Large = MWLT_Large(in_channels=4)
    net_Small = MWLT_Small(in_channels=4)
    data = torch.randn(1, 4, 640)
    flops_small, params_small = profile(net_Small, inputs=(data,))
    flops_base, params_base = profile(net_Base, inputs=(data,))
    flops_large, params_large = profile(net_Large, inputs=(data,))
    with open("params_flops.txt", "w") as f:
        f.write("MWLT_Small: params={:.3f}M, flops={:.3f}G\n".format(params_small / 1000 ** 2, flops_small / 1000 ** 3))
        f.write("MWLT_Base: params={:.3f}M, flops={:.3f}G\n".format(params_base / 1000 ** 2, flops_base / 1000 ** 3))
        f.write("MWLT_Large: params={:.3f}M, flops={:.3f}G".format(params_large / 1000 ** 2, flops_large / 1000 ** 3))
        f.close()


if __name__ == "__main__":
    claculate_flop_param()
