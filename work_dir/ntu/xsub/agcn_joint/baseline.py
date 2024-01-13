import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ntu_rgb_d  import Graph
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
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

'''
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A,stride=1,kernel_size=5 ,adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        pad = int((kernel_size - 1) / 2)
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),stride=(stride,1),padding=(pad, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, -1, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, stride,adaptive=adaptive)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.gcn1(x) + self.residual(x))
        return y

'''
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        num_jpts = A.shape[-1]
        self.Graph =torch.tensor(Graph('spatial').get_adjacency_matrix(), dtype=torch.float32, requires_grad=False).to('cuda')



        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.Conv2d(in_channels,out_channels,1)
        self.conv_b =  nn.Conv2d(in_channels,out_channels,1)
        self.conv_c = nn.Conv2d(out_channels, out_channels, 1)
        self.soft = nn.Softmax(0)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        # unified attention
        self.Attention = nn.Parameter(torch.ones(num_jpts))

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels , requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels , 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)


        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()

        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        N, C, T, V = x0.size()

        # learn_A = self.DecoupleA.repeat(
        #     1, self.out_channels // self.groups, 1, 1)
        #print(learn_A.shape)
        # norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
        #     learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        A1 = self.conv_a(x0).permute(1,3,0,2).contiguous().view(self.out_channels,V,N*T)
        A2 = self.conv_b(x0).permute(1,0, 2, 3).contiguous().view(self.out_channels,N * T,V)


        A1 = self.tan(torch.matmul(A1, A2) / A2.size(-1))#.unsqueeze(0).repeat(3,1,1,1)
        #     1, self.out_channels // self.groups, 1, 1))
        # A1 = self.norm((torch.matmul(A1,A2)/A1.size(-1))#.unsqueeze(0).repeat(
        #     1, self.out_channels // self.groups, 1, 1)) # c v v


        norm_learn_A = self.conv_c(A1.unsqueeze(0)).squeeze(0)

        n, kc, t, v = x.size()

        # x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('ndtv,dvw->ndtw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        y =x
        '''
        # spatial attention
        se = y.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y
        a1 = se1.unsqueeze(-2)

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        a2 = se1.unsqueeze(-1)

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        a3 = se2.unsqueeze(-1).unsqueeze(-1)

        # unified attention
        y = y * self.Attention + y
        y = y + y * ((a2 + a3) / 2)
        y = self.bn(y)
        '''

        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point, groups, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)
        # self.dropSke = DropBlock_Ske(num_point=num_point)
        # self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x))+ self.residual(x)
        return self.relu(x)

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3,groups = 8):
        super(Model, self).__init__()


        # Graph = import_class(graph)
        # self.graph = Graph(**graph_args)


        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        # self.triplet = TripletAttention()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, num_point,groups,stride=1,residual=False)

        self.l2 = TCN_GCN_unit(64, 64, A,num_point,groups,stride=1)
        self.l3 = TCN_GCN_unit(64, 64, A,num_point,groups)
        self.l4 = TCN_GCN_unit(64, 64, A, num_point,groups)
        self.l5 = TCN_GCN_unit(64, 128, A, num_point,groups,stride=2, )
        self.l6 = TCN_GCN_unit(128, 128, A, num_point,groups)
        self.l7 = TCN_GCN_unit(128, 128, A, num_point,groups)
        self.l8 = TCN_GCN_unit(128, 256, A, num_point,groups,stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, num_point,groups)
        self.l10 = TCN_GCN_unit(256, 256, A, num_point,groups)

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
        x = self.l9(x)
        x = self.l10(x)
        #x = self.triplet(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # x = self.drop_out(x)

        return self.fc(x)