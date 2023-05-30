import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .Transformer import TransformerModel
from ipdb import set_trace
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
import numpy as np

from pdb import set_trace
import warnings

__all__ = ['ViT_B16', 'ViT_B32', 'ViT_L16', 'ViT_L32', 'ViT_H14']

class Colar_static(nn.Module):
    def __init__(self, ch_in, ch_out, device, kmean_path):
        super(Colar_static, self).__init__()
        chennel = 1024
        self.conv1_3_Ek = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_Ev = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv2_1_W = nn.Conv1d(chennel, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv1d(chennel * 2, ch_out, kernel_size=1, stride=1, padding=0)
        self.opt = nn.ReLU()
        self.static_feature = list()
        self.device = device
        
        
        static = np.load(kmean_path, allow_pickle=True)
        for i in range(0, 5, 1):
            x = np.asarray(static[i])
            x = torch.from_numpy(x).squeeze().unsqueeze(0)
            self.static_feature.append(x.permute(0, 2, 1))
            self.static_feature[i] = self.static_feature[i].to(self.device)
    

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.matmul(value, y_weight)
        return y_sum

    def forward(self, x):
        # -1 to get most recent frame representation
        x = x[:, -1:, :]
        
        # permute so that temporal dimension is last
        x = x.permute(0, 2, 1)
        
        # convert frame to key and value space features
        k = self.conv1_3_k(x)
        v = self.conv1_3_v(x)
        
        feature_w = torch.empty(x.shape[0], 5).to(self.device)
        for i in range(0, 5, 1):
            static_feature = self.static_feature[i]

            # convert exemplar to key-value space
            Ek = self.conv1_3_Ek(static_feature)
            Ev = self.conv1_3_Ev(static_feature)

            weight = self.weight(Ek, k)
            sum = self.sum(Ev, weight)
            if i == 0:
                feature_E = sum
            else:
                feature_E = torch.cat((feature_E, sum), dim=-1)

            feature_w[:, i:i + 1] = self.conv2_1_W(sum).squeeze(-1)

        feature_E = feature_E.to(self.device)
        
        # changed from F.softmax to F.sigmoid to accomodate multilabel classification
        # feature_w = F.softmax(feature_w, dim=-1).unsqueeze(-1)
        
        feature_w = F.sigmoid(feature_w).unsqueeze(-1)
        feature_E = torch.bmm(feature_E, feature_w)
        out = torch.cat((v, feature_E), dim=1)
        out = self.opt(out)

        out = self.conv2_1(out)

        return out

class VisionTransformer_v4(nn.Module):
    def __init__(
        self,
        args,
        img_dim,
        patch_dim,
        out_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned", with_camera=True, with_motion=False, num_channels=3072,
    ):
        super(VisionTransformer_v4, self).__init__()
        
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.with_camera = with_camera
        
        # set to false cause features of 3D_ResNet are passed as camera inputs
        self.with_motion = False
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        # num_channels = img_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # self.num_patches = int((img_dim // patch_dim) ** 2)
        self.num_patches = int(img_dim // patch_dim)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.feature_dropout = nn.Dropout(p=self.dropout_rate)
        
        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
                
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        # print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        d_model = args.decoder_embedding_dim
        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim + d_model, hidden_dim//2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim + d_model, out_dim)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

        # Decoder
        factor = 1  # 5
        dropout = args.decoder_attn_dropout_rate
        # d_model = args.decoder_embedding_dim
        n_heads = args.decoder_num_heads
        d_layers = args.decoder_layers
        d_ff = args.decoder_embedding_dim_out  # args.decoder_embedding_dim_out or 4*args.decoder_embedding_dim None
        activation = 'gelu'  # 'gelu'
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),  # True
                                   d_model, n_heads),  # ProbAttention  FullAttention
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),  # False
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, args.query_num, d_model))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                args.query_num, self.embedding_dim, args.query_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        # print('position decoding :', positional_encoding_type)
        self.classifier = nn.Linear(d_model, out_dim)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)
        # self.merge_fc = nn.Linear(d_model, 1)
        # self.merge_sigmoid = nn.Sigmoid()

        # COLAR for exemplar consultation
        self.colar_static = Colar_static(num_channels,out_dim,device = args.device, kmean_path = '/workspace/pvc-meteor/features/colar/exemplar.pickle')
        self.colar_mlp = nn.Linear(2 * out_dim, out_dim)

    def forward(self, sequence_input_rgb, sequence_input_flow):
        if self.with_camera and self.with_motion:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2) # [128, 64, 4096]
        elif self.with_camera:
            x = sequence_input_rgb
        elif self.with_motion:
            x = sequence_input_flow
            
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        
        colar_x = self.colar_static(x[:, -1:, :])
        
        # my modification
        # x = self.feature_dropout(x)
        
        x = self.linear_encoding(x) # [128, 64, 1024]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1) # [128, 65, 1024]
        x = self.position_encoding(x) # [128, 65, 1024]
        x = self.pe_dropout(x)   # not delete
        
        # apply transformer
        x = self.encoder(x) # [128, 65, 1024]
        
        if not self.training:
            self.x_last_attention = x
        
        x = self.pre_head_ln(x)  # [128, 65, 1024]
        # x = self.after_dropout(x)  # add
        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1) # [128, 8, 1024]
        # decoder_cls_token = self.after_dropout(decoder_cls_token)  # add
        # decoder_cls_token = self.decoder_position_encoding(decoder_cls_token)  # [128, 8, 1024]
        dec = self.decoder(decoder_cls_token, x)   # [128, 8, 1024]
        dec = self.after_dropout(dec)  # add
        # merge_atte = self.merge_sigmoid(self.merge_fc(dec))  # [128, 8, 1]
        # dec_for_token = (merge_atte*dec).sum(dim=1)  # [128, 1024]
        # dec_for_token = (merge_atte*dec).sum(dim=1)/(merge_atte.sum(dim=-2) + 0.0001)
        dec_for_token = dec.mean(dim=1) # [128, 1024]
        # dec_for_token = dec.max(dim=1)[0]
        dec_cls_out = self.classifier(dec) # [128, 8, 7]
        # set_trace()
        # x = self.to_cls_token(x[:, 0])
        x = torch.cat((self.to_cls_token(x[:, -1]), dec_for_token), dim=1) # [128, 2048]
        x = self.mlp_head(x) # [128,7]
        
        x = torch.cat((x, colar_x.squeeze(-1)), dim=1)
        x = self.colar_mlp(x)
        # x = F.relu(x)
        # dec_cls_out = F.relu(dec_cls_out)

        return x, dec_cls_out # [128,7], [128, 8, 7]

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


def ViT_B16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_B32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_H14(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 14
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )
