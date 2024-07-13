# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import repeat, rearrange
from timm.models.layers.weight_init import trunc_normal_
from timm.models.layers import DropPath
from .tokenpose_base import Residual, PreNorm, FeedForward

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, groups * channels_per_group, height, width)

    return x


class InvertedResidual(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.stride = stride

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True) 
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):

        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='Normal', std=0.01, layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    bias=0.0001
                    layer=['_BatchNorm', 'GroupNorm'])
            ]``
    """

    def __init__(self,
                 channels=244, # 48, 116, 176, 244
                ):
        super().__init__()
        self.stage_blocks = 4

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True) 
            )

        self.layer = self._make_layer(channels, self.stage_blocks)


    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)


    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1.0 / m.weight.shape[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        return x


class ResidualAttn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out['x'] = out['x'] + x
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return {
            'x':out,
            'attn_logit':dots
        }


class GroupAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_sparse = 6, num_patches = 256, scale_with_head = False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        mask_q = torch.ones([1, heads, 3, num_patches + num_sparse], dtype=torch.bool)
        mask_q[:,:,0,0] = 0
        mask_k = torch.ones([1, heads, 3, num_patches + num_sparse * 3], dtype=torch.bool)
        mask_k[:,:,:,0] = 0
        mask = mask_q[:,:,:,:, None] * mask_k[:,:,:, None, :]
        self.register_buffer('mask', ~mask)

    def forward(self, x, mask = None): # [B G (K+N) C]
        B, G, N, _, H, K = *x.shape, self.heads, 6
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b g n (h d) -> b h g n d', h = H), qkv)
        
        k_kp = k[:,:,:,:K].flatten(2,3).unsqueeze(2).repeat(1,1,G,1,1)
        v_kp = v[:,:,:,:K].flatten(2,3).unsqueeze(2).repeat(1,1,G,1,1)
        k = torch.cat([k_kp, k[:,:,:,K:]], 3)
        v = torch.cat([v_kp, v[:,:,:,K:]], 3)

        dots = torch.einsum('bhgid,bhgjd->bhgij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        dots.masked_fill_(self.mask[..., :q.size(3), :k.size(3)].repeat(B,1,1,1,1), mask_value)

        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('bhgij,bhgjd->bhgid', attn, v)
        
        out = rearrange(out, 'b h g n d -> b g n (h d)')
        out =  self.to_out(out)
        return {
            'x':out,
            'attn_logit':dots
        }


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, num_patches=None, all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.group_layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        self.group_mask = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim*3),
            nn.Sigmoid()
        )
        for _ in range(depth//2):
            self.layers.append(nn.ModuleList([
                ResidualAttn(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
            self.group_layers.append(nn.ModuleList([
                ResidualAttn(PreNorm(dim, GroupAttention(dim, heads = heads, dropout = dropout, num_sparse=6, num_patches=num_patches, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def softmax_pool(self, x, dim):
        weight = x.softmax(dim)
        return (x * weight).sum(dim)

    def forward(self, x, mask = None, pos = None, sparse_keypoint_token = None, keep_rate = None):
        K = 1
        B, N, C = x.shape
        N_img = N - K
        pos = pos.repeat(B,1,1)
        attns = []
        for idx,(attn, ff) in enumerate(self.layers):
            if idx>0 and self.all_attn:
                x[:,K:] += pos
            out = attn(x, mask = mask)
            x , attn_logit = out['x'], out['attn_logit']
            attns.append(attn_logit[:,:,:K,K:])

            ### prune img token
            now_img_token = x.size(1) - K
            tgt_img_token = int(N_img * keep_rate[idx])
            if now_img_token != tgt_img_token:
                global_attn = attn_logit[:,:,:K,K:] # [B H K N]
                kp_token, img_token = x[:,:K], x[:,K:]          # [B K C] [B N C]

                now_attn = self.softmax_pool(global_attn.flatten(1,2),1)
                _, sorted_idx = now_attn.topk(tgt_img_token, dim=1)
                img_token = torch.gather(img_token, 1, sorted_idx[..., None].repeat(1,1,C))
                pos = torch.gather(pos, 1, sorted_idx[..., None].repeat(1,1,C))
                x = torch.cat([kp_token, img_token], 1)

            if idx == 3:
                kp, img = x[:,:1], x[:,1:]
                kp = kp + sparse_keypoint_token
                x = torch.cat([kp, img], 1)
                K = kp.size(1)
                
            x = ff(x)

        kp, img = x[:,:K], x[:,K:]
        kp = torch.cat([kp[:,:1].clone(), kp], 1)
        kp = rearrange(kp, 'b (g k) c->b g k c',g=3)

        channel_mask = rearrange(self.group_mask(img), 'b n (g c)->b g n c', g=3)
        img = img[:, None].repeat(1,3,1,1) * channel_mask
        pos = pos[:, None].repeat(1,3,1,1) * channel_mask

        now_img_token = img.size(2)
        tgt_img_token = int(N_img * keep_rate[6])
        if now_img_token != tgt_img_token:
            attn_logit = attn_logit[:,:,:K,K:]
            global_attn = torch.cat([-1e5*torch.ones_like(attn_logit[:,:,:1]), attn_logit], 2) # [B H G K N]
            now_attn = self.softmax_pool(rearrange(global_attn, 'b h (g k) n->b g (h k) n', g=3),2)
            _, sorted_idx = now_attn.topk(tgt_img_token, dim=2, largest=True, sorted=False)
            img = torch.gather(img, 2, sorted_idx[..., None].repeat(1,1,1,C))
            pos = torch.gather(pos, 2, sorted_idx[..., None].repeat(1,1,1,C))

        x = torch.cat([kp, img], 2)
        K = 6
        for idx,(attn, ff) in enumerate(self.group_layers):
            if self.all_attn:
                x[:,:,K:] += pos
            out = attn(x, mask = mask)
            x , attn_logit = out['x'], out['attn_logit']

            ### prune img token
            now_img_token = x.size(2) - K
            tgt_img_token = int(N_img * keep_rate[idx+6])
            if now_img_token != tgt_img_token:
                global_attn = attn_logit[:,:,:,:K,3*K:] # [B H G K N]
                kp_token, img_token = x[:,:,:K], x[:,:,K:]          # [B G K C] [B G N C]

                now_attn = self.softmax_pool(global_attn.permute(0,2,1,3,4).flatten(2,3),2)
                _, sorted_idx = now_attn.topk(tgt_img_token, dim=2, largest=True, sorted=False)
                img_token = torch.gather(img_token, 2, sorted_idx[..., None].repeat(1,1,1,C))
                pos = torch.gather(pos, 2, sorted_idx[..., None].repeat(1,1,1,C))
                x = torch.cat([kp_token, img_token], 2)

            x = ff(x)

        kp_body = x[:,:,:K].flatten(1,2)[:,1:]
        return kp_body, attns


class TokenPose_S_base(nn.Module):
    def __init__(self, *, image_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False, channels = 3, dropout = 0., emb_dropout = 0.,pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(image_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size[0] // (4*patch_size[0])) * (image_size[1] // (4*patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['none', 'learnable', 'learnable-full', 'sine', 'sine-full']

        self.patch_size = patch_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = ("full" in self.pos_embedding_type)

        self.human_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = image_size[0] // (4*self.patch_size[0]), image_size[1] // (4* self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        # stem net
        self.stem = ShuffleNetV2(channels)

        self.patch2emb_norm = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.patch2emb_mlp = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,num_keypoints=num_keypoints,num_patches=num_patches,all_attn=self.all_attn)

        trunc_normal_(self.human_token, std=.02)
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            print("Initialization...")
            self.apply(self._init_weights)
            
    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'learnable-full', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=True)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_pretrain(self, pretrained):
        print('=> init from pretrain: ' + pretrained)
        pretrained_state_dict = torch.load(pretrained)['model']
        self.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, img):
        p = self.patch_size
        x = self.stem(img)

        # transformer
        x = self.patch2emb_norm(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch2emb_mlp(x)
        b, n, _ = x.shape

        human_token = repeat(self.human_token, '() n d -> b n d', b = b)
        sparse_keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        x += self.pos_embedding[:, :n]
        x = torch.cat((human_token, x), dim=1)
        x = self.dropout(x)
        return x, sparse_keypoint_tokens

    def trans(self, x, sparse_keypoint_tokens, mask = None, keep_rate = None):
        x, attns = self.transformer(x, mask,self.pos_embedding, sparse_keypoint_tokens, keep_rate)
        return x, attns      


class TokenPose_S(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(TokenPose_S, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.features = TokenPose_S_base(
                            image_size = [cfg.MODEL.IMAGE_SIZE[1],cfg.MODEL.IMAGE_SIZE[0]],
                            patch_size = [cfg.MODEL.PATCH_SIZE[1],cfg.MODEL.PATCH_SIZE[0]],
                            num_keypoints = cfg.MODEL.NUM_JOINTS, dim = cfg.MODEL.DIM, channels = 128,
                            depth = cfg.MODEL.TRANSFORMER_DEPTH,heads = cfg.MODEL.TRANSFORMER_HEADS,
                            mlp_dim = cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
                            pos_embedding_type = cfg.MODEL.POS_EMBEDDING_TYPE)
        self.ema_features = copy.deepcopy(self.features)
        ###################################################

        self.keep_rate = cfg.MODEL.EXTRA.KEEP_RATE
        # head
        dim = cfg.MODEL.DIM
        self.mlp_head_x = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
        )
        self.mlp_head_y = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))
        )
        self.ema_mlp_head_x = copy.deepcopy(self.mlp_head_x)
        self.ema_mlp_head_y = copy.deepcopy(self.mlp_head_y)

    def forward(self, img):
        outputs = {}
        stu_keep_rate = [self.keep_rate[i] for i in range(4) for _ in range(3)]
        if self.training:
            feat, sparse_kp_tokens = self.features(img)
            x, _ = self.features.trans(feat, sparse_kp_tokens, keep_rate=stu_keep_rate)
            outputs['pred_x'] = self.mlp_head_x(x)
            outputs['pred_y'] = self.mlp_head_y(x)
            tch_keep_rate = [1 for _ in range(12)]
            x, _ = self.features.trans(feat, sparse_kp_tokens, keep_rate=tch_keep_rate)
            outputs['entire_pred_x'] = self.mlp_head_x(x)
            outputs['entire_pred_y'] = self.mlp_head_y(x)
            return outputs
        with torch.no_grad():
            feat, sparse_kp_tokens = self.ema_features(img)
            x, _ = self.ema_features.trans(feat, sparse_kp_tokens, keep_rate=stu_keep_rate)
            outputs['ema_pred_x'] = self.ema_mlp_head_x(x)
            outputs['ema_pred_y'] = self.ema_mlp_head_y(x)
        return outputs['ema_pred_x'], outputs['ema_pred_y']

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            self.features._init_pretrain(pretrained)
            logger.info('=> init final weights from normal distribution')
            for m in self.mlp_head_x.modules():
                if isinstance(m, nn.Linear):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format('mlp_head_x'))
                    logger.info('=> init {}.bias as 0'.format('mlp_head_x'))
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            for m in self.mlp_head_y.modules():
                if isinstance(m, nn.Linear):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format('mlp_head_y'))
                    logger.info('=> init {}.bias as 0'.format('mlp_head_y'))
                    trunc_normal_(m.weight, std=0.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

        self.ema_features = copy.deepcopy(self.features)
        self.ema_mlp_head_x = copy.deepcopy(self.mlp_head_x)
        self.ema_mlp_head_y = copy.deepcopy(self.mlp_head_y)


def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_S(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model