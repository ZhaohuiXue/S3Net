import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
class SEnet(nn.Module):
    def __init__(self, channel, r):
        super(SEnet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Linear1 = nn.Linear(channel, channel//r)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(channel//r, channel)
    def forward(self, x):
        avg_x = self.avgpool(x)
        lin1_x = self.Linear1(avg_x)
        lin1_x = self.relu(lin1_x)
        lin2_x = self.Linear2(lin1_x)
        lin2_x = torch.unsqueeze(lin2_x, 1)
        res = lin2_x*x
        return res
class ConvBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=(1,1,1)):
        super(ConvBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
        super(ConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class dialiationConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0, rate=0):
        super(dialiationConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.rate = rate
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                            kernel_size=self.kernel_size, stride=self.stride,padding=self.padding, dilation=self.rate)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConvBNRelu1D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=20, stride=1,padding=0):
        super(ConvBNRelu1D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm1d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()


    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y
class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, d, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, d)
        y = self.fc(y).view(b, d, 1, 1)
        return x * y.expand_as(x)
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=128, dim=64, alpha=1.0,
                 normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)#
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim).cuda().requires_grad_())
        self.hidden_weight = nn.Parameter(torch.rand(self.num_clusters*self.dim, 1024).cuda().requires_grad_())
        self._init_params()

    def _init_params(self):
        # self.weight = nn.Parameter(
        #     (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1).cuda().requires_grad_()
        # )
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1).cuda().requires_grad_()
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1).cuda().requires_grad_()
        )

    def forward(self, x):
        N, C = x.shape[:2]#x

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # x [?, N, W, H] -> [?, N, S] -> [?, S, N]
        # C: dim
        # K: num_clusters
        # W [K, N] -> [N, K]
        # A [?, K, W, H] -> [?, K, S]

        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)#
        x_flatten = x.view(N, C, -1).cuda()#

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).cuda()
        residual *= soft_assign.unsqueeze(2)#
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
class HyperCLR(nn.Module):
    def __init__(self, channel, output_units):  # channel就是K
        # 调用Module的初始化
        super(HyperCLR, self).__init__()
        self.channel = channel
        self.output_units = output_units
        self.fcl_size = 1024
        self.conv1 = ConvBNRelu3D(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0)
        # self.res_conv1 = ConvBNRelu3D(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = ConvBNRelu3D(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.res_conv2 = ConvBNRelu3D(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = ConvBNRelu3D(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.res_conv3 = ConvBNRelu3D(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1,
                                  padding=0)
        self.res_conv4 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                                  padding=1)
        self.cam = Channel_Attention(channel=64, r=16)
        self.sam = Spartial_Attention(kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.max_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc_r = nn.Linear(1024, 512)
        self.netvlad = NetVLAD(num_clusters=64, dim=64, alpha=1.0, normalize_input=True)
        self.projector = nn.Sequential(
            nn.Linear(self.fcl_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(self.fcl_size, 512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(512, self.output_units)

    def forward(self, x):
        x = self.conv1(x)
        # res_x1 = self.res_conv1(x)
        # res_x1 = self.res_conv1(res_x1)
        # x = x + res_x1
        x = self.conv2(x)
        # res_x2 = self.res_conv2(x)
        # res_x2 = self.res_conv2(res_x2)
        # x = x + res_x2
        x = self.conv3(x)
        res_x3 = self.res_conv3(x)
        res_x3 = self.res_conv3(res_x3)
        x = res_x3 + x
        # netvlad = NetVLAD(num_clusters=64, dim=128, alpha=100.0,normalize_input=True)
        # vlad = netvlad.forward(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.conv4(x)
        # res_x4 = self.res_conv4(x)
        # res_x4 = self.res_conv4(res_x4)
        # x = x + res_x4
        x_f = self.pool(x)
        # x_max = self.max_pool(x)
        # x_f = (x_max + x_f)/2
        x = self.cam(x_f)
        x_f = self.sam(x)
        # vlad = self.netvlad.forward(x)
        # vlad = vlad.view(vlad.shape[0], 64, -1)
        x_vlad = x_f.reshape([x_f.shape[0], -1])
        # x = self.fc_r(x)
        # h = self.projector(x_vlad)  # 要一个（batch，1024）的输入
        # h = torch.softmax(h,1)
        # vlad *= h
        x = self.fc(x_vlad)
        x = self.relu1(x)
        x = self.dropout(x)
        # x = self.fc1(x)
        # x = self.relu2(x)
        # x_clus = x.view(x.shape[0], self.output_units, -1)
        z = self.fc2(x)
        # x = x.view(x.shape[0], self.output_units, -1)
        # z = torch.sum(torch.pow(x, 2), 2)
        return x_f, z, x_vlad
class HyperCLR2(nn.Module):
    def __init__(self, channel, output_units):  # channel就是K
        # 调用Module的初始化
        super(HyperCLR2, self).__init__()
        self.channel = channel
        self.output_units = output_units
        self.fcl_size = 1024
        self.conv1_1 = ConvBNRelu3D(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv1_2 = ConvBNRelu3D(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv1_3 = ConvBNRelu3D(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.AvgPool3d((2, 2, 2))
        self.res_conv1 = ConvBNRelu3D(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2_1 = ConvBNRelu3D(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv2_2 = ConvBNRelu3D(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2_3 = ConvBNRelu3D(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.AdaptiveAvgPool3d((7, 3, 3))
        self.conv3_1 = ConvBNRelu3D(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv3_2 = ConvBNRelu3D(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3_3 = ConvBNRelu3D(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3_4 = ConvBNRelu3D(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32 * (self.channel-12), out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.cam = Channel_Attention(channel=64, r=16)
        self.sam = Spartial_Attention(kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.max_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc_r = nn.Linear(1024, 512)
        self.netvlad = NetVLAD(num_clusters=64, dim=64, alpha=1.0, normalize_input=True)
        self.projector = nn.Sequential(
            nn.Linear(self.fcl_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(self.fcl_size, 512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(80, 40)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(512, self.output_units)

    def forward(self, x):
        x_1 = self.conv1_1(x)
        x = self.conv1_2(x_1)
        x = self.conv1_3(x)
        x = x_1 + x
        # x = self.pool1(x)
        # res_x1 = self.res_conv1(x)
        # res_x1 = self.res_conv1(res_x1)
        # x = x + res_x1
        x_2 = self.conv2_1(x)
        x = self.conv2_2(x_2)
        x = self.conv2_3(x)
        x = x_2 + x
        # x = self.pool2(x)
        x_3 = self.conv3_1(x)
        x = self.conv3_2(x_3)
        x = self.conv3_3(x)
        x = x + x_3
        # x_4 = self.conv3_4(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])

        x = self.conv4(x)
        # x_4 = x_4.reshape([x_4.shape[0], -1, x.shape[2], x.shape[3]])
        # x = x_4 + x
        # x_m = self.max_pool(x)
        x = self.pool(x)
        # x = x + x_m
        x = self.cam(x)
        x = self.sam(x)
        # vlad = self.netvlad.forward(x)
        # vlad = vlad.view(vlad.shape[0], 64, -1)
        x_l = x.reshape([x.shape[0], -1])
        # x = self.fc_r(x)
        # h = self.projector(x_l)  # 要一个（batch，1024）的输入
        # h = torch.softmax(h,1)
        # vlad *= h
        x = self.fc(x_l)
        x = self.relu1(x)
        # x_c = self.dropout(x)

        z = self.fc2(x)
        return z, x_l
class HyperCLR1(nn.Module):
    def __init__(self, channel, output_units):  # channel就是K
        # 调用Module的初始化
        super(HyperCLR1, self).__init__()
        self.channel = channel
        self.output_units = output_units
        self.fcl_size = 1024
        self.conv1 = ConvBNRelu3D(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=0)
        self.conv1_1 = ConvBNRelu3D(in_channels=8, out_channels=8, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=0)
        self.conv2_1 = ConvBNRelu3D(in_channels=16, out_channels=16, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=0)
        self.conv3 = ConvBNRelu3D(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.conv3_1 = ConvBNRelu3D(in_channels=32, out_channels=32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv4 = ConvBNRelu3D(in_channels=32, out_channels=32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv5 = ConvBNRelu3D(in_channels=16, out_channels=8, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv6 = ConvBNRelu3D(in_channels=16, out_channels=8, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv4 = ConvBNRelu2D(in_channels=int(32*((self.channel-6)/2 -6)), out_channels=64, kernel_size=(3, 3), stride=1,
        #                           paddng=0)
        self.conv7 = ConvBNRelu2D(in_channels=32*(self.channel-24), out_channels=64, kernel_size=(3, 3), stride=1,
                                  padding=0)

        # self.conv5 = ConvBNRelu2D(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        # self.conv6 = ConvBNRelu2D(in_channels=32, out_channels=self.output_units, kernel_size=(3, 3), stride=1, padding=0)
         # self.conv4 = ConvBNRelu2D(in_channels=32*(math.ceil((math.ceil((((self.channel-6)/2)-4)/2)-2)/2)), out_channels=64, kernel_size=(3, 3), stride=1,
        #                           padding=0)
        # self.conv5 = ConvBNRelu2D(in_channels=64, out_channels=1, kernel_size=(1,1), stride=1, padding=0)
        self.netvlad = NetVLAD(num_clusters=256, dim=64, alpha=1.0, normalize_input=True)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pool1 = nn.AdaptiveAvgPool1d(512)
        self.pool2 = nn.AdaptiveAvgPool1d(self.output_units)
        self.cam = Channel_Attention(channel=64, r=8)
        self.sam = Spartial_Attention(kernel_size=3)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.fcl_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.output_units, self.output_units)
    def forward(self, x):#只用pool，效果可以，fc不能和pool混用
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv3(x)
        x = self.conv3_1(x)

        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x_4 = self.conv7(x)
        x = self.pool(x_4)
        x = self.cam(x)
        x = self.sam(x)
        x = x.reshape([x.shape[0], -1])
        # x = self.fc1(x)
        # x = self.relu(x)
        # z_p = self.fc2(x)
        x = torch.unsqueeze(x, 0)
        z_p = self.pool1(x)
        z_p = self.pool2(z_p)
        z_p = torch.squeeze(z_p, 0)
        # z_p = self.fc2(z_p)
        return z_p#平均池化可以保证精度的误差很小

class DFSL(nn.Module):
    def __init__(self,channel,output_units):#channel就是K
        # 调用Module的初始化
        super(DFSL, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.fcl_size = 1024
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(3,3,3),stride=(1,1,1),padding=0)#kernel_size（depth，heigth，width）
        self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=8,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv3 = ConvBNRelu3D(in_channels=8,out_channels=8,kernel_size=(3,3,3),stride=1,padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv4 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(3,3,3),stride=(1,1,1),padding=0)
        self.conv5 = ConvBNRelu3D(in_channels=16,out_channels=16,kernel_size=(3,3,3),stride=(1,1,1),padding=1)
        self.conv6 = ConvBNRelu3D(in_channels=16,out_channels=16,kernel_size=(3,3,3),stride=1,padding=1)
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv7 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0)
        self.conv8 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv9 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv9_1 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv9_2 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv9_3 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        # self.pool3 = nn.AdaptiveAvgPool3d((24,6,6))
        self.conv10 = ConvBNRelu2D(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1,
                                  padding=0)
        self.conv11 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,padding=1)
        self.conv12 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,padding=1)
        # self.conv9 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
        #                           padding=1)
        # self.conv1 = ConvBNRelu1D(in_channels=1,out_channels=20,kernel_size=24,stride=1,padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cam = Channel_Attention(channel=64, r=8)
        self.sam = Spartial_Attention(kernel_size=3)
        self.pool_c1 = nn.AdaptiveAvgPool1d(512)
        self.pool_c2 = nn.AdaptiveAvgPool1d(self.output_units)
        # self.conv8 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,1,1),stride=1,padding=0)
        # self.conv9 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,1,1),stride=1,padding=0)
        self.projector = nn.Sequential(
            nn.Linear(self.fcl_size, 512),
            nn.ReLU(),
            nn.Linear(512, 80),
        )
        self.fc=nn.Linear(self.fcl_size,512)
        self.relu1=nn.ReLU()
        self.fc3=nn.Linear(self.fcl_size, self.output_units)
    def forward(self, x):
        x = self.conv1(x)

        x_r_1 = self.conv2(x)
        x_r_1 = self.conv3(x_r_1)
        x = x_r_1 + x
        x = self.pool1(x)

        x = self.conv4(x)
        x_r_1 = self.conv5(x)

        x_r_1 = self.conv6(x_r_1)
        x = x + x_r_1
        x = self.pool2(x)

        x = self.conv7(x)
        # x = self.pool3(x)
        # x = self.conv8(x_r_1)
        # x = self.conv9(x)
        # x_r_2 = x + x_r_1
        # x = self.conv9_1(x_r_2)
        # x = self.conv9_1(x)
        # x = x + x_r_2
        # x = self.pool3(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.conv10(x)
        # x = self.conv9(x)
        x = self.cam(x)
        x = self.sam(x)
        x = self.pool(x)
        # x = self.conv11(x_c)
        # x = self.conv12(x)
        # x = x_c + x
        # x_9 = self.conv9(x_8)
        x_result = x.reshape([x.shape[0], -1])
        # self.fcl_size = x.shape[-1]
        # h = self.projector(x_result)
        x_result_p = torch.unsqueeze(x_result, 0)
        x_result_p=self.pool_c1(x_result_p)
        z=self.pool_c2(x_result_p)
        z = torch.squeeze(z, 0)
        # z_1=self.fc3(x_result)
        return z

class SSRN(nn.Module):
    def __init__(self,channel,output_units):#channel就是K
        # 调用Module的初始化
        super(SSRN, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.fcl_size = 1024
        self.f_bn =nn.BatchNorm3d(num_features=1)
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=16,kernel_size=(3,1,1),stride=(1,1,1),padding=0)#kernel_size（depth，heigth，width）
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        # self.conv3 = ConvBNRelu3D(in_channels=24,out_channels=24,kernel_size=(7,1,1),stride=1,padding=(3,0,0))
        self.conv4 = ConvBNRelu3D(in_channels=32,out_channels=32,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        # self.conv5 = ConvBNRelu3D(in_channels=24,out_channels=24,kernel_size=(7,1,1),stride=1,padding=(3,0,0))
        self.conv6 = ConvBNRelu3D(in_channels=32,out_channels=64,kernel_size=(math.ceil(int((self.channel-2)/2)),1,1),stride=1,padding=0)
        self.conv7 = ConvBNRelu2D(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1,padding=0)
        # self.conv7_1 = ConvBNRelu2D(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1,padding=1)
        # self.conv7_2 = ConvBNRelu2D(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=1,padding=2)
        #
        # self.conv7_3 = ConvBNRelu2D(in_channels=64, out_channels=32, kernel_size=(7, 7), stride=1,padding=3)
        self.netvlad = NetVLAD(num_clusters=64, dim=32, alpha=1.0, normalize_input=True)
        # self.conv8 = ConvBNRelu2D(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=1,padding=1)
        self.conv9 = ConvBNRelu2D(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1,padding=0)
        self.conv10 = ConvBNRelu2D(in_channels=16, out_channels=self.output_units, kernel_size=(3, 3), stride=1,padding=1)
        # self.conv10 = ConvBNRelu2D(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=1,padding=1)
        # self.conv11 = ConvBNRelu2D(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=1,padding=1)
        # self.conv12 = ConvBNRelu2D(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=1,padding=1)

        # self.conv9 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
        #                           padding=1)
        # self.conv1 = ConvBNRelu1D(in_channels=1,out_channels=20,kernel_size=24,stride=1,padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cam = Channel_Attention(channel=16, r=8)
        self.sam = Spartial_Attention(kernel_size=3)
        self.pool_c1 = nn.AdaptiveAvgPool1d(512)
        self.pool_c2 = nn.AdaptiveAvgPool1d(self.output_units)
        self.fc1=nn.Linear(self.fcl_size,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(2704, self.output_units)
        self.netvlad = NetVLAD(num_clusters=32, dim=32, alpha=1.0, normalize_input=True)
        self.netvlad2 = NetVLAD(num_clusters=32, dim=16, alpha=1.0, normalize_input=True)
    def forward(self, x0, x1, classiorcontrast):

        # x = F.normalize(x)
        # x0 = self.f_bn(x0)
        # x1 = self.f_bn(x1)
        # x0 = x0[:, :, :, int((x0.shape[2] - x1.shape[2]) / 2):int(
        #     (x0.shape[2] - x1.shape[2]) / 2) + x1.shape[2],
        #               int((x0.shape[2] - x1.shape[2]) / 2):int(
        #                   (x0.shape[2] - x1.shape[2]) / 2) + x1.shape[2]]
        # print(x0==t)
        x0 = self.conv1(x0)
        x1 = self.conv1(x1)

        x0 = self.pool1(x0)
        x1 = self.pool1(x1)

        x0 = self.conv2(x0)
        x1 = self.conv2(x1)

        x0 = self.conv4(x0)
        x1 = self.conv4(x1)

        x0 = self.conv6(x0)#(?,128,1,25,25)
        x1 = self.conv6(x1)#(?,128,1,25,25)

        x0 = torch.squeeze(x0,2)
        x1 = torch.squeeze(x1,2)
        x0 = self.conv7(x0)
        x1 = self.conv7(x1)

        x0_contrast = self.conv9(x0)
        x1_contrast = self.conv9(x1)

        x0 = x0_contrast.reshape([x0_contrast.shape[0], -1])
        x1 = x1_contrast.reshape([x1_contrast.shape[0], -1])
        x0 = torch.unsqueeze(x0, 0)
        x1 = torch.unsqueeze(x1, 0)
        z0=self.pool_c2(x0)
        z1=self.pool_c2(x1)
        z0 = torch.squeeze(z0, 0)
        z1 = torch.squeeze(z1, 0)

        # z0 = self.fc2(x0)
        # z1 = self.fc2(x1)
        return z0, z1, x0_contrast, x1_contrast

class S3Net(nn.Module):
    def __init__(self,channel,output_units):#
        #
        super(S3Net, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5] * self.output_units), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([0.5] * self.output_units), requires_grad=True)
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=16,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        self.conv_pool = ConvBNRelu3D(in_channels=16,out_channels=16,kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)

        self.conv2 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,1,1),stride=1, padding=(1,0,0))
        self.conv4 = ConvBNRelu3D(in_channels=32,out_channels=64,kernel_size=(3,1,1),stride=1, padding=(1,0,0))
        self.conv6 = ConvBNRelu3D(in_channels=64,out_channels=32,kernel_size=(math.ceil(self.channel//2),1,1),stride=1,padding=0)
        self.conv6_1 = ConvBNRelu3D(in_channels=64,out_channels=32,kernel_size=(math.ceil(self.channel//2),1,1),stride=1,padding=0)
        self.conv7 = ConvBNRelu2D(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1,padding=1)
        self.conv9 = ConvBNRelu2D(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1,padding=1)
        self.conv10 = ConvBNRelu2D(in_channels=32, out_channels=self.output_units, kernel_size=(3, 3), stride=1,padding=1)
        self.pool_c2 = nn.AdaptiveAvgPool1d(self.output_units)

    def forward(self, x0, x1):
        x0 = self.conv1(x0)#两个conv1的卷积的参数是一样的
        x1 = self.conv1(x1)

        x0 = self.conv_pool(x0)
        x1 = self.conv_pool(x1)
        x0 = self.conv2(x0)
        x1 = self.conv2(x1)

        x0 = self.conv4(x0)
        x1 = self.conv4(x1)

        x0 = self.conv6(x0)
        x1 = self.conv6_1(x1)

        x0 = torch.squeeze(x0,2)
        x1 = torch.squeeze(x1,2)

        x0 = self.conv7(x0)
        x1 = self.conv7(x1)


        x0 = self.conv9(x0)
        x1 = self.conv9(x1)
        #
        x0 = self.conv10(x0)
        x1 = self.conv10(x1)

        x0_ = x0[:, :,x0.shape[2]//2, x0.shape[3]//2]
        x1_ = x1[:, :,x1.shape[2]//2, x1.shape[3]//2]
        x0 = x0.reshape([x0.shape[0], -1])
        x1 = x1.reshape([x1.shape[0], -1])
        x0 = torch.unsqueeze(x0, 0)
        x1 = torch.unsqueeze(x1, 0)

        z0=self.pool_c2(x0)
        z1=self.pool_c2(x1)
        z0 = torch.squeeze(z0, 0)
        z1 = torch.squeeze(z1, 0)

        return z0*self.scale1, z1*self.scale2,x0_,x1_
class D3CNN(nn.Module):
    def __init__(self,channel,output_units,windowSize):#channel就是K
        # 调用Module的初始化
        super(D3CNN, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.fcl_size = 1024
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=64,kernel_size=(3,1,1),stride=(3,1,1),padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=64,out_channels=64,kernel_size=(3,1,1),stride=(2,1,1),padding=0)
        # self.conv3 = ConvBNRelu3D(in_channels=64,out_channels=64,kernel_size=(3,1,1),stride=1,padding=0)
        self.conv4 = ConvBNRelu3D(in_channels=64,out_channels=128,kernel_size=(3,3,3),stride=(1,1,1),padding=0)
        self.conv5 = ConvBNRelu3D(in_channels=128,out_channels=128,kernel_size=(3,3,3),stride=(2,1,1),padding=0)
        # self.conv6 = ConvBNRelu3D(in_channels=128,out_channels=128,kernel_size=(3,1,1),stride=1,padding=(1, 0, 0))
        self.conv7 = ConvBNRelu3D(in_channels=128,out_channels=256,kernel_size=(3,3,3),stride=(1,1,1),padding=0)
        self.conv8 = ConvBNRelu3D(in_channels=256,out_channels=256,kernel_size=(3,3,3),stride=(1,1,1),padding=0)
        self.conv9 = ConvBNRelu3D(in_channels=256,out_channels=512,kernel_size=(3,1,1),stride=(2,1,1),padding=(1,0,0))#(256, 1, 1)
        self.conv10 = ConvBNRelu3D(in_channels=512,out_channels=512,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))#(512, 1, 1)
        self.conv11 = ConvBNRelu3D(in_channels=512,out_channels=1024,kernel_size=(3,1,1),stride=(1,1,1),padding=(1, 0, 0))#(1024, 1, 1)
        # self.conv12 = ConvBNRelu3D(in_channels=512,out_channels=1024,kernel_size=(3,3,3),stride=(1,1,1),padding=(1, 0, 0))#(1024, 1, 1)
        self.projector = nn.Sequential(
            nn.Linear(self.fcl_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.fc=nn.Linear(self.fcl_size,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(512, self.output_units)

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        # x = self.conv6(x)

        x = self.conv7(x)

        x = self.conv8(x)

        x = self.conv9(x)
        #
        x = self.conv10(x)
        x = self.conv11(x)
        x = x.reshape([x.shape[0], -1])

        # self.fcl_size = x.shape[-1]
        h = self.projector(x)
        x=self.fc(x)
        x=self.relu1(x)
        z=self.fc2(x)
        return h, z
class D1CNN(nn.Module):
    def __init__(self,channel,output_units):#channel就是K
        # 调用Module的初始化
        super(D1CNN, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.fcl_size = 1024

        self.pool = nn.MaxPool1d(5)
        self.fc_c = nn.Linear(700, self.fcl_size)
        # self.conv3 = ConvBNRelu3D(in_channels=64,out_channels=64,kernel_size=(3,1,1),stride=1,padding=0)
       # self.conv12 = ConvBNRelu3D(in_channels=512,out_channels=1024,kernel_size=(3,3,3),stride=(1,1,1),padding=(1, 0, 0))#(1024, 1, 1)
        self.projector = nn.Sequential(
            nn.Linear(self.fcl_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.fc=nn.Linear(self.fcl_size,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(512, self.output_units)

    def forward(self, x):
        # x = x.permute(0,2,1)
        x = self.conv1(x)

        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        x_c = self.fc_c(x)
        h = self.projector(x_c)
        x=self.fc(x_c)
        x=self.relu1(x)
        z=self.fc2(x)
        return x_c, h, z
class SpectralRLC(nn.Module):
    def __init__(self,channel,output_units,windowSize):#channel就是K
        # 调用Module的初始化
        super(SpectralRLC, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=1)
        self.conv4 = ConvBNRelu2D(in_channels=32*self.channel, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.pool=nn.AdaptiveAvgPool2d((4, 4))
        self.fcl = nn.Linear(64*3*3, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
        x = self.conv4(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fcl(x)
        return x
class HyperRLC(nn.Module):
    def __init__(self,channel,output_units,windowSize):#channel就是K
        # 调用Module的初始化
        super(HyperRLC, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(5,3,3),stride=1,padding=0)
        self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool=nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
        x = self.conv4(x)
        x = self.pool(x)

        return x
class fcl(nn.Module):
    def __init__(self, channel, output_units,windowSize1, windowSize2):#channel就是K
        # 调用Module的初始化
        super(fcl, self).__init__()
        self.output_units=output_units
        self.channel=channel
        self.windowSize1=windowSize1
        self.windowSize2=windowSize2
        self.HyperCLR = HyperRLC(self.channel,self.output_units,self.windowSize1)
        self.SpectralRLC = HyperRLC(self.channel,self.output_units,self.windowSize2)
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.fc=nn.Linear(1024,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(512, 256)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(256, self.output_units)
    def forward(self, x1, x2, x3, x4):

        x1 = self.HyperCLR(x1).reshape([x1.shape[0],-1])#大
        x3 = self.HyperCLR(x3).reshape([x3.shape[0],-1])#大
        x2 = self.SpectralRLC(x2).reshape([x2.shape[0],-1])#小512
        x4 = self.SpectralRLC(x4).reshape([x4.shape[0],-1])#小
        # x1 = self.HyperCLR(x1)
        # x2 = self.SpectralRLC(x2)
        input_dis1 = x1-x3#大
        input_dis2 = x2-x4#小
        # input_dis = torch.cat((input_dis1, input_dis2), 0)
        # classi_dis = torch.cat((x1, x2), 1)
        classi_dis = x1*x2
        h1 = self.projector(input_dis1)
        h2 = self.projector(input_dis2)
        x=self.fc(classi_dis)
        x=self.relu1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        z=self.fc3(x)
        return h1, h2, z