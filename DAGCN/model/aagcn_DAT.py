import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import einops
from timm.models.layers import to_2tuple, trunc_normal_
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

            # self.bn = nn.BatchNorm2d(out_channels)
            # bn_init(self.bn, 1)
        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)

                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V

                A1 = A[i] + A1 * self.alpha

                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))

                y = z + y if y is not None else z
        else:
            A = self.A.cuda(x.get_device()) * self.mask
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.bn(y)
        y = y + self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))

            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            # unified attention
            # y = y * self.Attention + y
            # y = y + y * ((a2 + a3) / 2)
            # y = self.bn(y)
        return y


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DAttentionBaseline(nn.Module):

    def __init__(
            self,
            q_size,
            kv_size,
            n_heads=8,
            n_head_channels=16,
            n_groups=1,
            attn_drop=0.9,
            proj_drop=0.9,
            stride=2,
            offset_range_factor=[-1, -1, 2, 2],
            use_pe=False,
            dwc_pe=False,
            no_off=False,
            fixed_pe=False,
            stage_idx=0

    ):

        super().__init__()

        # 是否需要相对位置
        self.dwc_pe = dwc_pe

        # 设置多头的通道数
        self.n_head_channels = n_head_channels

        # 设置尺寸
        self.scale = self.n_head_channels ** -0.5

        # 设置多头数量
        self.n_heads = n_heads

        # query的高(h)与宽(w)大小
        self.q_h, self.q_w = q_size

        # key和value的h和w
        self.kv_h, self.kv_w = kv_size

        # channel数和heads数的乘积，为之后合并提供shape大小
        self.nc = n_head_channels * n_heads

        # 分组
        self.n_groups = n_groups

        # 每组的channel数
        self.n_group_channels = self.nc // self.n_groups

        # 每组的head数
        self.n_group_heads = self.n_heads // self.n_groups

        # 是否使用位置编码
        self.use_pe = use_pe

        self.fixed_pe = fixed_pe

        self.no_off = no_off

        # 偏移范围因子
        self.offset_range_factor = offset_range_factor

        self.bn = nn.BatchNorm2d(256)

        # 卷积核大小
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        # 设置偏移网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),  # 可以调整替换
            nn.GELU(),  # 可以调整替换
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
        )

        # 对proj_q进行高斯嵌入
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        # 对proj_k进行高斯嵌入
        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # 对proj_v进行高斯嵌入
        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=False)
        self.attn_drop = nn.Dropout(attn_drop, inplace=False)

        # 使用位置编码的几种选择
        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    # 设置参考点
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        # 设置生成网格
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref



    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 经过self.proj_q的高斯嵌入后得到q
        q = self.proj_q(x)

        # 将张量q进行形状变换
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        # 计算位置偏移量offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg

        # 计算位置偏移量offset的长和宽
        Hk, Wk = offset.size(2), offset.size(3)

        # 计算采样的数量
        n_sample = Hk * Wk

        # 判断位置偏移参数offset_range_factor是否大于0
        if self.offset_range_factor[0] > 0:
            # 初始化位置偏移量范围
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)

            # 对位置偏移量offset进行范围调整
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 对位置偏移量offset进行形状变换
        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        # 获取位置参考点
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        # 判断偏移量是否大于等于0
        if self.offset_range_factor[0] >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        # 进行采样
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        # 重新变换采样的形状
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # 重新变换采样的形状
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns

        # 调整注意力权重尺度
        attn = attn.mul(self.scale)

        # 判断是否使用位置编码
        if self.use_pe:
            # 判断是否使用深度可分离卷积位置编码
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, dtype, device)

                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

                attn = attn + attn_bias

        # 进行 softmax 操作，以便将注意力分数转换为概率分布
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        # 使用 Einstein Summation 的公式计算加权和，其中 attn 是注意力分布，v 是值张量，生成输出 out
        out = torch.einsum('b m n, b c n -> b c m', attn,
                           v)  # 例如:torch.Size([72, 8, 120]) torch.Size([72, 120, 16]) torch.Size([72, 8, 16]) out, attn, v
        # print(out.shape,attn.shape, v.shape,'out, attn, v')
        if self.use_pe and self.dwc_pe:
            # 将输出 out 与残差位置编码相加，以增加空间维度上的位置敏感性
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.bn(self.proj_drop(self.proj_out(out)))


        # 返回三个值：y 是输出；pos 是位置编码的坐标；reference 是位置编码的值。
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=False)

        # if attention:
        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.beta = nn.Parameter(torch.ones(1))
        # temporal attention
        # self.conv_ta1 = nn.Conv1d(out_channels, out_channels//rt, 9, padding=4)
        # self.bn = nn.BatchNorm2d(out_channels)
        # bn_init(self.bn, 1)
        # self.conv_ta2 = nn.Conv1d(out_channels, 1, 9, padding=4)
        # nn.init.kaiming_normal_(self.conv_ta1.weight)
        # nn.init.constant_(self.conv_ta1.bias, 0)
        # nn.init.constant_(self.conv_ta2.weight, 0)
        # nn.init.constant_(self.conv_ta2.bias, 0)

        # rt = 4
        # self.inter_c = out_channels // rt
        # self.conv_ta1 = nn.Conv2d(out_channels, out_channels // rt, 1)
        # self.conv_ta2 = nn.Conv2d(out_channels, out_channels // rt, 1)
        # nn.init.constant_(self.conv_ta1.weight, 0)
        # nn.init.constant_(self.conv_ta1.bias, 0)
        # nn.init.constant_(self.conv_ta2.weight, 0)
        # nn.init.constant_(self.conv_ta2.bias, 0)
        # s attention
        # num_jpts = A.shape[-1]
        # ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        # pad = (ker_jpt - 1) // 2
        # self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        # nn.init.constant_(self.conv_sa.weight, 0)
        # nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        # rr = 16
        # self.fc1c = nn.Linear(out_channels, out_channels // rr)
        # self.fc2c = nn.Linear(out_channels // rr, out_channels)
        # nn.init.kaiming_normal_(self.fc1c.weight)
        # nn.init.constant_(self.fc1c.bias, 0)
        # nn.init.constant_(self.fc2c.weight, 0)
        # nn.init.constant_(self.fc2c.bias, 0)
        #
        # self.softmax = nn.Softmax(-2)
        # self.sigmoid = nn.Sigmoid()
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.attention:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

            # spatial attention
            # se = y.mean(-2)  # N C V
            # se1 = self.sigmoid(self.conv_sa(se))
            # y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            # se = y.mean(-1)  # N C T
            # # se1 = self.relu(self.bn(self.conv_ta1(se)))
            # se2 = self.sigmoid(self.conv_ta2(se))
            # # y = y * se1.unsqueeze(-1) + y
            # a2 = se2.unsqueeze(-1)

            # se = y  # NCTV
            # N, C, T, V = y.shape
            # se1 = self.conv_ta1(se).permute(0, 2, 1, 3).contiguous().view(N, T, self.inter_c * V)  # NTCV
            # se2 = self.conv_ta2(se).permute(0, 1, 3, 2).contiguous().view(N, self.inter_c * V, T)  # NCVT
            # a2 = self.softmax(torch.matmul(se1, se2) / np.sqrt(se1.size(-1)))  # N T T
            # y = torch.matmul(y.permute(0, 1, 3, 2).contiguous().view(N, C * V, T), a2) \
            #         .view(N, C, V, T).permute(0, 1, 3, 2) * self.alpha + y

            # channel attention
            # se = y.mean(-1).mean(-1)
            # se1 = self.relu(self.fc1c(se))
            # se2 = self.sigmoid(self.fc2c(se1))
            # # y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
            #
            # y = y * ((a2 + a3) / 2) + y
            # y = self.bn(y)
        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,num_set=3,
                 drop_out=0, adaptive=True, attention=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        # A = np.stack([np.eye(num_point)] * num_set, axis=0)
        A = self.graph.A
        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.DAT = DAttentionBaseline(q_size=(10, 5), kv_size=(10, 5), n_heads=8, n_head_channels=32, n_groups=1,
                                      attn_drop=0.9, proj_drop=0.9, stride=2,
                                      offset_range_factor=[-1, -1, 2, 2], use_pe=True, dwc_pe=True,
                                      no_off=False, fixed_pe=False, stage_idx=0)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        x,_,_ = self.DAT(x)
        x,_,_ = self.DAT(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x),self.fc(x),self.fc(x)
# if __name__ == '__main__':
#     input = torch.randn(2, 3, 300,25,2)
#     model = Model()
#     model.eval()
#     out = model(input)
#     print(out.shape)