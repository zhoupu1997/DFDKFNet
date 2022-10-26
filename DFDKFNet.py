import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class Nonlocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, factor=16):
        super(Nonlocal, self).__init__()

        self.sub_sample = sub_sample
        self.factor = factor
        self.pool = nn.MaxPool2d(factor, factor)
        self.in_channels = in_channels
        self.pool2x = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, ndwi, pooling=True):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch_size = x.size(0)
        if pooling:
            x = self.pool2x(x)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  #(b, input_challes, h, w)=>(b, input_challes/2, h, w)=>(b, input_challes/2, h*w)
        g_x = g_x.permute(0, 2, 1)    #(b, h*w, input_challes/2)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   ##(b, input_challes, h, w)=>(b, input_challes/2, h, w)=>(b, input_challes/2, h*w)
        theta_x = theta_x.permute(0, 2, 1)    ##(b, h*w, input_challes/2)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)    ##(b, input_challes/2, h*w)
        f = torch.matmul(theta_x, phi_x)     # (b, h*w, input_challes/2) * (b, input_challes/2, h*w)=>(b, h*w, h*w)
        f_div_C = F.softmax(f, dim=-1)     #(b, h*w, 1)
        ndwi = self.pool(ndwi)   #(b, 1, 16*h, 16*w)=>(b, 1, h, w)
        ndwi = ndwi.view(batch_size, ndwi.shape[1], -1).permute(0, 2, 1)      ##(b, 1, h, w)=>(b, 1, h*w)=>(b, h*w, 1)
        f_div_C = torch.mul(f_div_C, ndwi)

        y = torch.matmul(f_div_C, g_x)   #(b, h*w, 1)* (b, h*w, input_challes/2) = >(b, h*w, input_challes/2)
        y = y.permute(0, 2, 1).contiguous()   #(b, h*w, input_challes/2)=>(b, input_challes/2, h*w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) #(b, input_challes/2, h*w)=>(b, input_challes/2, h, w)
        W_y = self.W(y)   #(b, input_challes/2, h, w)=>(b, input_challes, h, w)
        z = W_y + x   #(b, input_challes, h, w)+(b, input_challes, h, w)=(b, input_challes, h, w)

        if pooling:
            z= self.up(z)

        return z


class ChannelSELayer(nn.Module):   ##high-level features supervise low-level features


    def __init__(self, high_channels, low_channels):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = low_channels//2
        # self.k_size = k_size
        self.conv = nn.Conv2d(high_channels + low_channels, low_channels, 1, bias=False)
        # self.eca = eca_layer(k_size = k_size)
        self.fc1 = nn.Linear(low_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, low_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, low_features, high_features):   #[b, 256, h/8, w/8], [b, 512, h/16, w/16]
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, high_channels, H1, W1 = high_features.size()
        batch_size, low_channels, H2, W2 = low_features.size()
        # Average along each channel
        features = torch.cat((self.up(high_features), low_features), 1)  ##[b, low+high, h, w]
        features = self.conv(features)   ##[b, low+high, h, w] =>[b, low, h, w]
        squeeze_tensor = features.view(batch_size, low_channels, -1).mean(dim=2)  ##[b, low, h, w]=>[b, low, h*w]=>[b, low, 1]
        # # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))  ##[batch_size, low, 1]=>[batch_size, low/2, 1]
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))   ##[batch_size, low/2, 1]=>[batch_size, low_channels, 1]
        output_tensor = torch.mul(low_features, fc_out_2.view(batch_size, low_channels, 1, 1))    #batch_size, low_channels, H2, W2
        # output_tensor = self.eca(features)

        return output_tensor

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),                                   ##这里输出的是[B,in_channels,1,1]
            nn.Conv2d(in_channels, out_channels, 1, bias=False),       #[B,in_channels,h, w]=>[B,in_channels,1,1]=>[B,out_channels,1,1]=>[B,out_channels,h, w]
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        # print(size, "size")
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, factor=8):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(factor, factor)

    def forward(self, x, ndwi):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        ndwi = self.pool(ndwi)
        sa = torch.cat([avg_out, max_out, ndwi], dim=1)
        sa = self.conv1(sa)
        sa= torch.mul(self.sigmoid(sa), x)
        return sa

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y

class ASPP(nn.Module):                        ##[B, in_channels, H, W] =>[B, 256, H, W]
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.Nonlocal = Nonlocal(in_channels, factor=16)
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3, rate4 = (3, 6, 12, 18)    ##这里是卷积率的大小
        # self.atrous_block1 = ASPPConv(in_channels, out_channels, rate0)
        self.atrous_block2 = ASPPConv(in_channels, out_channels, rate1)
        self.atrous_block3 = ASPPConv(in_channels, out_channels, rate2)
        self.atrous_block4 = ASPPConv(in_channels, out_channels, rate3)
        self.atrous_block5 = ASPPConv(in_channels, out_channels, rate4)
        self.atrous_block6 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, 2 * out_channels, 1, bias=False),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, x):
        # atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block5 = self.atrous_block5(x)
        atrous_block6 = self.atrous_block6(x)
        # print(self.conv(x).shape, ndwi.shape)
        res = self.project(torch.cat((atrous_block2, atrous_block3, atrous_block4, atrous_block5, atrous_block6), dim=1))
        # nonlocal_ = self.Nonlocal(res, ndwi, pooling=False)

        return res

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):    #  [b, in_channels, h, w]=>[b, out_channels, h, w]
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)     #[b, c, h, w]=>[b, c, 1, 1]
        # print(y.shape)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  #[b, c, 1, 1]=>[b, c, 1]=>[b, 1, c]=>[b, c, 1]=>[b, c, 1, 1]

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)   #[b, c, h, w]

class SpatialAttentionv2(nn.Module):
    def __init__(self, in_channels, factor=8):
        super(SpatialAttentionv2, self).__init__()

        self.conv = VGGBlock(in_channels, 32, 32)
        self.conv2 = VGGBlock(6, 6, 3)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.in_channels = in_channels
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(factor, factor)
        self.con1x1 = nn.Conv2d(32, 2, kernel_size=1)


    def forward(self, x, ndwi, ndvi):

        x = self.conv(x)
        # avg_out = self.sigmoid(torch.mean(x, dim=1, keepdim=True))
        avg_out = self.sigmoid(torch.mean(x, dim=1, keepdim=True))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = self.sigmoid(max_out)
        conv_out = self.con1x1(x)
        ndwi = self.pool(ndwi)
        ndvi = self.pool(ndvi)
        sa = torch.cat([avg_out, max_out, ndwi, conv_out, ndvi], dim=1)
        sa = self.conv2(self.bn(sa))
        sa = self.sigmoid(self.conv1(sa))
        return sa

class SpatialAttentionv3(nn.Module):
    def __init__(self, in_channels, factor=8):
        super(SpatialAttentionv3, self).__init__()

        self.conv = VGGBlock(in_channels, 32, 32)
        self.conv2 = VGGBlock(6, 6, 3)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.in_channels = in_channels
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(factor, factor)
        self.con1x1 = nn.Conv2d(32, 2, kernel_size=1)


    def forward(self, x, ndwi, ndvi):

        x = self.conv(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        conv_out = self.con1x1(x)
        ndwi = self.pool(ndwi)
        ndvi = self.pool(ndvi)
        sa = torch.cat([avg_out, max_out, ndwi, conv_out, ndvi], dim=1)
        sa = self.conv2(self.bn(sa))
        sa = self.sigmoid(self.conv1(sa))
        return sa

class DFDKFNet_(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.pool3 = nn.MaxPool2d(8, 8)
        self.pool4 = nn.MaxPool2d(16, 16)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])


        self.SA1 = SpatialAttentionv2(in_channels=256, factor=8)
        self.SA2 = SpatialAttentionv2(in_channels=128, factor=4)
        self.SA3 = SpatialAttentionv2(in_channels=64, factor=2)
        self.SA4 = SpatialAttentionv2(in_channels=32, factor=1)

        self.conv_cat1 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv_cat2 = nn.Conv2d(384, 128, kernel_size=1)
        self.conv_cat3 = nn.Conv2d(192, 64, kernel_size=1)
        self.conv_cat4 = nn.Conv2d(96, 32, kernel_size=1)

        self.con3x1 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.con2x2 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.con1x3 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.con0x4 = nn.Conv2d(32, n_classes, kernel_size=1)

        self.conv_final = nn.Conv2d(32, n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi, ndvi):

        x0_0 = self.conv0_0(input)    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0))   # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0))   # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0))   # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0))   # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        # x3_1_out = self.SA1(x3_1, ndwi, ndvi)  # [b, 256, h/8, w/8]=>[b, 1, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))  # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        # x2_2_out = self.SA2(x2_2, ndwi, ndvi)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))  # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        # x1_3_out = self.SA3(x1_3, ndwi, ndvi)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))  # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x0_4_out = self.SA4(x0_4, ndwi, ndvi)
        # output = self.LS(self.final(x0_4_sa))  # [b, 32, h, w]=>[b, 1, h, w]
        return x0_4_out