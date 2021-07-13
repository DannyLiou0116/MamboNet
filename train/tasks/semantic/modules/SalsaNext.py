# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F


# padding mode = circular 時，padding會被除以二，所以有事先把padding*2
# Conv2d default: stride=1, padding=0, dilation=1, groups=1

##############################
#        se block
##############################

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print("----------x_size: ",x.shape)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

##############################
#        pre processing
##############################

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        # input dimension = [i, j, k]
        super(ResContextBlock, self).__init__()

        # self.conv0 = nn.Conv2d(in_filters, out_filters*4, kernel_size=(1, 1), stride=1)  # [i, j, k]
        # self.act0 = nn.LeakyReLU(0.01, inplace=True)
        # self.bn0 = nn.BatchNorm2d(out_filters)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)  # [i, j, k]
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)             # [i, j, k]
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)  # [i, j, k]
        self.act3 = nn.LeakyReLU(0.01, inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)           # [5, 64, 2048]
        shortcut = self.act1(shortcut)

        # print("========== shortcut.upsample_bilinear before: ", shortcut.shape)
        # shortcut = F.interpolate(shortcut, size = [128,2048] , mode = "nearest")  # [5, 128, 2048]
        # print("========== shortcut.upsample_bilinear after: ", shortcut.shape)
        # x = nn.PixelShuffle(2)(x)
        # print("========== x.shape: ", x.shape)

        resA = self.conv2(shortcut)           # [5, 128, 2048]
        resA = self.act2(resA)         # [5, 128, 2048]
        resA1 = self.bn1(resA)         # [5, 128, 2048]
        # resA1 = self.pool(resA1)
        # print("========== resA1.shape: ", resA1.shape)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)
        # resA2 = self.pool(resA2)
        # print("========== resA2 .shape: ", resA2.shape)

        output = shortcut + resA2
        return output


##############################
#        Encoder
##############################

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        # input dimension = [i, j, k]
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU(0.01, inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU(0.01, inplace=True)
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))

        # out_filters*3 because concat below
        self.act5 = nn.LeakyReLU(0.01, inplace=True)
        self.bn4 = nn.BatchNorm2d(out_filters)

        # 暫定把se_block 放在 1x1 conv 後，照著論文把它放在殘差的相加之前
        self.se1 = SELayer(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # input dimension = [i, j, k]
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)

        resA = self.se1(resA)

        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

##############################
#        Bottleneck
##############################


class ASPP(nn.Module):
    def __init__(self, in_channel = 256, depth = 256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):

        # print("========== before aspp:", x.shape)

        size = x.shape[2:]
        # print("========== size:", size)
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

##############################
#        Decoder
##############################

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU(0.01, inplace=True)
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU(0.01, inplace=True)
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)
        # print("================before upB", upA.shape)
        upB = torch.cat((upA,skip),dim=1)
        # print("================after upB", upB.shape)

        # 某一層 up block
        # upA before PixelShuffle:  torch.Size([3, 256, 4, 128])
        # upA after  PixelShuffle:  torch.Size([3, 64, 8, 256])
        # skip.shape:               torch.Size([3, 256, 8, 256])
        # upB.shape:                torch.Size([3, 320, 8, 256])

        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

##############################
#        Generator
##############################

class SalsaNext(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses
        #self.in_channel = in_channel
        # print("self.nclasses",self.nclasses)
        self.downCntx = ResContextBlock(5, 32)     
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False) 
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.aspp = ASPP(in_channel = 256)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.fit_conv = nn.Conv2d(nclasses, nclasses, kernel_size=(3, 3), padding = 1, stride = (2, 1) )

    def forward(self, x):
        # input dimension = [2048 x 64 x 5]
        in_channel = int(x.shape[1])
        # print("in_channel: ", in_channel)
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)          # [2048 x 64 x 32]
        # print("downCntx.shape: ", downCntx.shape)

        down0c, down0b = self.resBlock1(downCntx)    # [1024 x 32 x 64], [2048 x 64 x 64]
        down1c, down1b = self.resBlock2(down0c)      # [512 x 16 x 128], [1024 x 32 x 128]
        down2c, down2b = self.resBlock3(down1c)      # [256 x 8 x 256], [512 x 16 x 256]
        down3c, down3b = self.resBlock4(down2c)      # [128 x 4 x 256], [256 x 8 x 256]
        down5c = self.resBlock5(down3c)              # [128 x 4 x 256]

        # print("\n\n========== down0c: ",down0c.shape)
        # print("========== down0b: ",down0b.shape)
        # print("========== down1c: ",down1c.shape)
        # print("========== down1b: ",down1b.shape)
        # print("========== down2c: ",down2c.shape)
        # print("========== down2b: ",down2b.shape)
        # print("========== down3c: ",down3c.shape)
        # print("========== down3b: ",down3b.shape)
        # print("========== down5c: ",down5c.shape)
        down5c = self.aspp(down5c)
        # print("\n========== down5c after aspp: ",down5c.shape)


        "up"
        up4e = self.upBlock1(down5c,down3b)            # [256 x 8 x 128]
        # print("========== up4e: ",up4e.shape)
        up3e = self.upBlock2(up4e, down2b)           # [512 x 16 x 128]
        # print("========== up3e: ",up3e.shape)
        up2e = self.upBlock3(up3e, down1b)           # [1024 x 32 x 64]
        # print("========== up2e: ",up2e.shape)
        up1e = self.upBlock4(up2e, down0b)           # [2048 x 64 x 32]

        # print("========== up1e: ",up1e.shape)


        logits = self.logits(up1e)                   # [2048 x  x 20]
        # logits = self.fit_conv(logits)               # [2048 x 64 x 20]
        # print("========= logits: ",logits.shape)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits



##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()

        self.cnn0 = nn.Sequential(                # [2048 x 64 x 6]
            nn.Conv2d(6, 32, 3, 1, 1),            # [2048 x 64 x 32]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2)                       # [1024 x 32 x 32]
        )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),           # [1024 x 32 x 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2)                       # [512 x 16 x 64]
        )   
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),          # [512 x 16 x 128]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2)                       # [256 x 8 x 128]
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),         # [256 x 8 x 256]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),    
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),         # [256 x 8 x 256]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2)                       # [128 x 4 x 256]
        )        
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 4, 3, 1, 1),           # [128 x 4 x 4]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.fc0 = nn.Sequential(                        
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(                        
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(                        
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )



    def forward(self, feature):
        # Concatenate image and condition image by channels to produce input
        # print("feature.shape",feature.shape)  # [2, 256, 4, 128]
        feature = self.cnn0(feature)
        feature = self.cnn1(feature)
        feature = self.cnn2(feature)
        feature = self.cnn3(feature)
        feature = self.cnn4(feature)
        feature = self.cnn5(feature)
        # print("feature.shape",feature.shape)  
        feature = feature.view(len(feature), -1)
        # print("flatten.shape",feature.shape)  
        digit = self.fc0(feature)
        digit = self.fc1(digit)
        digit = self.fc2(digit)
        # print("digit.shape",digit.shape)  
        return digit





