# lib/models/vgst/vl_transformer.py
import copy
import torch
from typing import Optional, List
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from lib.models.vgst.transformer import _get_activation_fn
from lib.utils.misc import NestedTensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_device(mod):
    return next(mod.parameters()).device if any(True for _ in mod.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_mha3(x: Tensor) -> Tensor:
    """确保输入是 [S,B,C]，供 MultiheadAttention 使用"""
    if x.dim() == 3:
        return x
    if x.dim() == 4:  # (B,C,H,W)->(S,B,C)
        B, C, H, W = x.shape
        return x.flatten(2).permute(2, 0, 1).contiguous()
    if x.dim() == 2:  # (B,C)->(1,B,C)
        return x.unsqueeze(0)
    raise RuntimeError(f"expect 3D or 4D tensor, got {tuple(x.shape)}")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            x = self.norm1(src)
            q = k = self.with_pos(x, pos)
            q = _to_mha3(q); k = _to_mha3(k); x = _to_mha3(x)
            x2 = self.self_attn(q, k, value=x, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(x2)
            x = self.norm2(src)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
            src = src + self.dropout2(x2)
            return src

        q = k = self.with_pos(src, pos)
        q = _to_mha3(q); k = _to_mha3(k); src = _to_mha3(src)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        out = src
        for l in self.layers:
            out = l(out, mask, src_key_padding_mask, pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class VisionLanguageEncoder(nn.Module):
    """
    RGB 分支视觉编码器（**显存优化版**）

    输入:  NestedTensor
        .tensors: [B,C_in,H,W]
        .mask   : [B,H,W] (bool, True=padding)

    输出:  out_list=[NestedTensor(B,d_model,H',W')], pos_list=[B,d_model,H',W']
          其中 H'=H/16, W'=W/16 —— 与 VGST 里 divisor=16 的假设一致
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, num_channels=1024,
                 activation="relu", normalize_before=False, init='trunc_normal',
                 patch_size=16):
        super().__init__()
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
        enc_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, enc_norm)
        self.num_encoder_layers = num_encoder_layers

        self.d_model = d_model
        self.nhead = nhead
        self.num_channels = num_channels
        self.patch_size = patch_size

        # 懒创建：patch 向量 -> d_model
        self.input_proj = None  # nn.Linear(C_in*ps*ps, d_model)
        # 懒创建：可学习位置编码 [L,1,d_model]
        self.pos_embed = None

        if init == 'xavier':
            self.apply(self._init_xavier)
        elif init == 'trunc_normal':
            self.apply(self._init_trunc_normal)
        elif init == 'xavier_stable':
            self.apply(self._init_xavier_stable)
        else:
            raise RuntimeError(f"init method should be xavier/trunc_normal/xavier_stable, not {init}.")

    # ---- inits ----
    def _init_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def _init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def _init_xavier_stable(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=self.num_encoder_layers + 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    # ---- forward ----
    def forward(self, tensor_list: NestedTensor, mode=None):
        assert isinstance(tensor_list, NestedTensor), "expect NestedTensor"
        x = tensor_list.tensors        # [B,C_in,H,W]
        m = tensor_list.mask           # [B,H,W] or None

        device = x.device
        B, C_in, H, W = x.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, f"Input {H}x{W} must be divisible by patch_size={ps}"

        # 1) unfold 成 patch tokens
        patches = F.unfold(x, kernel_size=ps, stride=ps)      # [B, C_in*ps*ps, L]
        L = patches.shape[-1]
        Hp, Wp = H // ps, W // ps
        patches = patches.permute(2, 0, 1).contiguous()       # [L,B,C_in*ps^2]
        patch_dim = C_in * ps * ps

        # 2) 懒创建线性投影
        if (self.input_proj is None) or (self.input_proj.in_features != patch_dim):
            self.input_proj = nn.Linear(patch_dim, self.d_model).to(device)
        tokens = self.input_proj(patches)                     # [L,B,d_model]

        # 3) 下采样 mask -> [B,Hp,Wp] -> flatten [B,L]
        if m is not None:
            m_ds = F.interpolate(m.float().unsqueeze(1), size=(Hp, Wp), mode='nearest').to(torch.bool).squeeze(1)
            m_flat = m_ds.flatten(1)                          # [B,L]
        else:
            m_ds = None
            m_flat = None

        # 4) 可学习位置编码 [L,1,d_model]
        if (self.pos_embed is None) or (self.pos_embed.shape[0] != L):
            self.pos_embed = nn.Parameter(torch.zeros(L, 1, self.d_model, device=device))
        pos = self.pos_embed

        # 5) Transformer 编码
        tokens = self.encoder(tokens, src_key_padding_mask=m_flat, pos=pos)  # [L,B,d_model]

        # 6) 复原 2D feature+pos，接口与旧版保持一致
        feat_2d = tokens.permute(1, 2, 0).reshape(B, self.d_model, Hp, Wp)  # [B,C,Hp,Wp]
        pos_2d = pos.permute(1, 2, 0).reshape(1, self.d_model, Hp, Wp).expand(B, -1, -1, -1)

        out_list: List[NestedTensor] = [NestedTensor(feat_2d, m_ds)]
        pos_list: List[Tensor] = [pos_2d.to(feat_2d.dtype)]
        return out_list, pos_list


def build_vl_transformer(cfg):
    # 强制 patch_size=16，与 VGST.divisor=16 对齐，避免 token 尺寸不一致
    return VisionLanguageEncoder(
        d_model=cfg.MODEL.VL.HIDDEN_DIM,
        dropout=cfg.MODEL.VL.DROPOUT,
        nhead=cfg.MODEL.VL.NHEAD,
        dim_feedforward=cfg.MODEL.VL.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.VL.ENC_LAYERS,
        num_channels=1024,
        normalize_before=cfg.MODEL.VL.NORM_BEFORE,
        activation=cfg.MODEL.VL.ACTIVATION,
        init=cfg.MODEL.VL.INIT,
        patch_size=16,
    )
