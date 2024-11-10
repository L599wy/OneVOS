
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_
from itertools import repeat
import collections.abc

from lib.utils.misc import is_main_process
import math

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,cfg=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear_ID_KV = nn.Linear(dim, dim + self.num_heads)
        self._init_weights(self.linear_ID_KV)
        self.dim=dim

        self.topk = cfg.is_topk
        # print("istopk:", self.topk)
        if self.topk is True:
            self.topk_template = cfg.topk_tempalte
            self.topk_search = cfg.topk_search
            self.topk_percent = cfg.is_topk_percent

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def softmax_w_top(self,x, top):
        top=int(top)
        values, indices = torch.topk(x, k=top, dim=-1)
        x_exp = torch.exp(values[:,:,:,:] - values[:, :,:,0:1])
        x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(-1, indices, x_exp.type(x.dtype))  # B * THW * HW
        return x

    def key_value_id(self, id_emb,k_m,v_m):
        ID_KV = self.linear_ID_KV(id_emb)
        ID_K, ID_V = torch.split(ID_KV, [self.num_heads, self.dim], dim=2)
        k_m = k_m * ((1 + torch.tanh(ID_K)).transpose(1, 2).unsqueeze(-1))
        v_m = v_m.permute(0, 2, 1, 3).flatten(2) + ID_V
        return k_m, v_m

    def forward(self, x,id_total,mem_k,mem_v,id_add=False):
        B, N, C = x.shape
        N_m = id_total.shape[1]
        N_s = N - id_total.shape[1]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        v_m, v_s = torch.split(v, [N_m, N_s], dim=2)
        k_m, k_s = torch.split(k, [N_m, N_s], dim=2)
        v_add_id=v
        if id_add:
            k_m, v_m=self.key_value_id(id_total,k_m,v_m)
            v_m = v_m.reshape(B, N_m, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_add_id = torch.cat([v_m, v_s], dim=2)

        q = q * self.scale
        # divided q
        q_m, q_s = torch.split(q, [N_m, N_s], dim=2)

        # template attention
        attn = (q_m @ k_m.transpose(-2, -1))
        if self.topk:
            if self.topk_percent:
                topk=int(self.topk_template/100*k_m.shape[2])
            else:
                topk=self.topk_template
            attn = self.softmax_w_top(attn, top=topk)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_m = (attn @ v_m).transpose(1, 2).reshape(B, N_m, C)

        # search attention
        if id_add and mem_k is not None:
            k=torch.cat((mem_k,k),dim=2)
            v_add_id=torch.cat((mem_v,v_add_id),dim=2)

        attn = (q_s @ k.transpose(-2, -1))
        if self.topk:
            if self.topk_percent:
                topk = int(self.topk_search/100 * k.shape[2])
            else:
                topk = self.topk_search
            attn = self.softmax_w_top(attn, top=topk)
        else:
            #print("no topk")
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v_add_id).transpose(1, 2).reshape(B, N_s, C)


        x = torch.cat([x_m, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if id_add:
            return x,k_m,v_m
        else:
            return x,None,None


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,cfg=cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, template,search,id_template,bs,mem_k,mem_v,id_add):
        B, N, C = search.shape
        BT = template.shape[0]
        t_N = N
        if BT != bs:
            template = template.view(bs, -1, C)
            t_N = int(BT / bs) * N

        x = torch.cat([template, search], dim=1)

        # new_memk=None
        x1, new_memk, new_memv = self.attn(self.norm1(x), id_template.transpose(0, 1), mem_k, mem_v, id_add=id_add)
        x = x + self.drop_path1(x1)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        template, search = torch.split(x, [t_N, N], dim=1)

        if template.shape[0] != BT:
            template = template.view(BT, N, C)

        return template, search, new_memk, new_memv



class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=464, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None,cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer,cfg=cfg)
            for i in range(depth[2])])

        self.norm = norm_layer(embed_dim[-1])

        self.apply(self._init_weights)

        self.grid_size = img_size // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches = self.grid_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[2]))

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed


    def forward(self, template_patch,id_template, search,is_train,mem_k=None,mem_v=None,bs=None):
        if template_patch.shape[2]==search.shape[2]:
            template_patch = self.patch_embed1(template_patch)
            template_patch = self.pos_drop(template_patch)
            for blk in self.blocks1:
                template_patch = blk(template_patch)
            template_patch = self.patch_embed2(template_patch)

            for blk in self.blocks2:
                template_patch= blk(template_patch)
            template_patch = self.patch_embed3(template_patch)

            B, C = template_patch.size(0), template_patch.size(-1)
            H_s = template_patch.shape[2]
            W_s = template_patch.shape[3]

            template_patch = template_patch.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
            template_patch = self.patch_embed4(template_patch)

            pos_embed = self.pos_embed
            if pos_embed.shape[1] != H_s * W_s:
                pos_embed = self.interpolate_pos_encoding(template_patch, H_s, W_s)
            template_patch = template_patch + pos_embed

            return template_patch,None,None,None

        search_features = []
        search = self.patch_embed1(search)

        search = self.pos_drop(search)
        for blk in self.blocks1:
            search = blk(search)
        search_features.append(search)

        search = self.patch_embed2(search)

        for blk in self.blocks2:
            search = blk(search)
        search_features.append(search)
        search = self.patch_embed3(search)

        search_features.append(search)
        H_s,W_s = search.shape[2], search.shape[3]
        search = search.flatten(2).permute(0, 2, 1) #BCHW --> BNC
        search = self.patch_embed4(search)


        pos_embed = self.pos_embed
        if pos_embed.shape[1] != H_s * W_s:
            pos_embed = self.interpolate_pos_encoding(search, H_s, W_s)
        search  = search  + pos_embed

        self.search_patch_record=search
        self.search_patch = self.pos_drop(search)
        self.template_patch = self.pos_drop(template_patch)

        now_layer=0
        new_memks = []
        new_memkv = []
        for blk in self.blocks3:
            if mem_k is not None:
                memk = mem_k[now_layer]
                memv = mem_v[now_layer]
            else:
                memk = None
                memv = None
            self.template_patch, self.search_patch,new_memk,new_memv  = blk(self.template_patch,self.search_patch,id_template,bs=bs,mem_k=memk,mem_v=memv,id_add=True)
            new_memks.append(new_memk)
            new_memkv.append(new_memv)
            now_layer += 1

        search_features.append(self.search_patch.transpose(1, 2).reshape(bs, -1, int(H_s), int(W_s)))
        return self.search_patch_record, search_features,new_memks,new_memkv

class Seg_Convmae(nn.Module):
    def __init__(self, backbone):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, template_patch,id_template, search,mem_k=None,mem_v=None,is_train=True):
        # search: (b, c, h, w)
        bs=search.shape[0]
        if search.dim() == 5:
            search = torch.flatten(search, start_dim=0, end_dim=1)

        if isinstance(id_template,list):
            id_template=id_template[0]

        search_patch_record, search_features,new_memks,new_memkv = self.backbone(template_patch,id_template, search,is_train,mem_k=mem_k,mem_v=mem_v,bs=bs)
        return search_patch_record,  search_features,new_memks,new_memkv



def get_convmae_model(config, **kwargs):
    msvit_spec = config.BACKBONE
    img_size = config.DATA_RANDOMCROP[0]

    if msvit_spec.VIT_TYPE == 'convmae_base':
        convViT = ConvViT(
            img_size=img_size, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
            depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=config)
    elif msvit_spec.VIT_TYPE == 'convmae_large':
        convViT = ConvViT(
            img_size=img_size, patch_size=[4, 2, 2], embed_dim=[384, 768, 1024],
            depth=[2, 2, 20], num_heads=16, mlp_ratio=[4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),cfg=config)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'convmae_base' or 'convmae_large'")

    return convViT


def build_onevos_convmae(cfg):
    print("build onevos convmae uni-hybrid attention")
    backbone = get_convmae_model(cfg)  
    model = Seg_Convmae(
        backbone,
    )

    return model
