"""
PyTorch implementation for the 3D version of the SonoNet model proposed in:
Baumgartner et al. "SonoNet: real-time detection and localisation of fetal standard scan planes in freehand ultrasound."
IEEE transactions on medical imaging 36.11 (2017): 2204-2215.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _triple


class SonoNet3D(nn.Module):
    """
    PyTorch implementation for a 3D version of the SonoNet model. This network takes as input a video clip with D frames
    and assigns a single label to the whole clip by exploiting space-time information. Thus, if input clip is (NxDxHxW),
    the corresponding output would be (NxC), where C is the number of output classes, N the batch size,
    D the number of frames in a clip (that should be small), and H,W the spatial dimension of each frame.
    Note that the target labels fed at training time should therefore be a vector of N elements. 

    Args:
        in_channels (int, optional): Number of input channels in the data.
            Default is 1.
        hid_features (int, optional): Number of features in the first hidden layer that defines the network arhcitecture.
            In fact, features in all subsequent layers are set accordingly by using multiples of this value,
            (i.e. x2, x4 and x8).
            Default is 16.
        out_labels (int, optional): Number of output labels (length of output vector after adaptation).
            Default is 7. Ignored if features_only=True
        features_only (bool, optional): If True, only feature layers are initialized and the forward method returns the features.
            Default is False.
        train_classifier_only (bool, optional): If True, only the adaptation layers are trainable.

    Attributes:
        _features (torch.nn.Sequential): Feature extraction CNN
        _adaptation (torch.nn.Sequential): Adaption layers for classification

    """

    def __init__(self, in_channels: int = 1, hid_features: int = 32, out_labels: int = 7,
                 features_only: bool = False, init: str = 'uniform', train_classifier_only: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.hid_features = hid_features  # number of filters in the first layer (then x2, x4 and x8)
        self.out_labels = out_labels
        self.features_only = features_only
        #self.dropout = nn.Dropout(p=0.25)

        self._features = self._make_feature_layers()
        if not features_only:
            self._adaptation = self._make_adaptation_layers()

        assert init in ['normal', 'uniform'], 'The init parameter may only be one between "normal" and "uniform"'
        full_init = self._initialize_normal if init == 'normal' else self._initialize_uniform
        self.apply(full_init)
        if not features_only:
            last_init = nn.init.xavier_normal_ if init == 'normal' else nn.init.xavier_uniform_
            last_init(self._adaptation[3].weight)  # last conv layer has no ReLu, hence Kaiming init is not suitable
            if train_classifier_only:
                for param in self._features.parameters():
                    param.requires_grad = False
                for param in self._adaptation.parameters():
                    param.requires_grad = True

    def forward(self, x):

        x = self._features(x)

        if not self.features_only:
            x = self._adaptation(x)
            try:
                (batch, channel, t, h, w) = x.size()    #t is the depth (number of frames in the clip)
            except ValueError:
                (channel, t, h, w) = x.size()
                batch = 1
           
            #x = self.dropout(x) 
            x = F.avg_pool3d(x, kernel_size=(t, h, w)).view(batch, channel)  # in=(N,C,D,H,W) & out=(N,C)

        return x

    @staticmethod
    def _initialize_normal(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()

    @staticmethod
    def _initialize_uniform(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()

    @staticmethod
    def _conv_layer(in_channels, out_channels):
        layer = [
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=(3, 3, 3), padding="same", bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layer)

    def _make_feature_layers(self):
        layers = [
            # Convolution stack 1
            self._conv_layer(self.in_channels, self.hid_features),
            self._conv_layer(self.hid_features, self.hid_features),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 2
            self._conv_layer(self.hid_features, self.hid_features * 2),
            self._conv_layer(self.hid_features * 2, self.hid_features * 2),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 3
            self._conv_layer(self.hid_features * 2, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 4
            self._conv_layer(self.hid_features * 4, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 5
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
        ]
        return nn.Sequential(*layers)

    def _make_adaptation_layers(self):
        layers = [
            # Adaptation layer 1
            nn.Conv3d(self.hid_features * 8, self.hid_features * 4,
                      kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.hid_features * 4),
            nn.ReLU(inplace=True),
            # Adaptation layer 2
            nn.Conv3d(self.hid_features * 4, self.out_labels,
                      kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.out_labels)
        ]
        return nn.Sequential(*layers)
    

class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        stride = _triple(stride)

        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        
        
        ######################## v1 ##########################
        if first_conv:
            intermed_channels = 45
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
            
        ############### COMPLETE ###############################
        '''
        self.spatial_conv = nn.Sequential(nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias),
                                      nn.BatchNorm3d(intermed_channels), 
                                      nn.ReLU(inplace=True)) 
        self.temporal_conv = nn.Sequential(nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(inplace=True))
        '''
        
        ############### NO FORMULA ###############################
        # The formula is used to reproduce the number of parameter of the 3D version of SonoNet as closely as possible
        self.spatial_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias),
                                      nn.BatchNorm3d(out_channels), 
                                      nn.ReLU(inplace=True)) 
        self.temporal_conv = nn.Sequential(nn.Conv3d(out_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(inplace=True))
        '''
        ############### NO BATCH ###############################
        self.spatial_conv = nn.Sequential(nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias),
                                      nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(inplace=True))
        '''
        '''
        
        ############### NO FORMULA NO BATCH ###############################
        self.spatial_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias),
                                      nn.ReLU(inplace=True)) 
        self.temporal_conv = nn.Sequential(nn.Conv3d(out_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(inplace=True))
        '''

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
 
        return x
    

class SonoNet3D_2_1d(nn.Module):
    """
    PyTorch implementation for a (2+1)D version of the SonoNet model. This network takes as input a video clip with D frames
    and assigns a single label to the whole clip by exploiting space-time information. Thus, if input clip is (NxDxHxW),
    the corresponding output would be (NxC), where C is the number of output classes, N the batch size,
    D the number of frames in a clip (that should be small), and H,W the spatial dimension of each frame.
    Note that the target labels fed at training time should therefore be a vector of N elements. 

    Args:
        in_channels (int, optional): Number of input channels in the data.
            Default is 1.
        hid_features (int, optional): Number of features in the first hidden layer that defines the network arhcitecture.
            In fact, features in all subsequent layers are set accordingly by using multiples of this value,
            (i.e. x2, x4 and x8).
            Default is 16.
        out_labels (int, optional): Number of output labels (length of output vector after adaptation).
            Default is 7. Ignored if features_only=True
        features_only (bool, optional): If True, only feature layers are initialized and the forward method returns the features.
            Default is False.
        train_classifier_only (bool, optional): If True, only the adaptation layers are trainable.

    Attributes:
        _features (torch.nn.Sequential): Feature extraction CNN
        _adaptation (torch.nn.Sequential): Adaption layers for classification

    """

    def __init__(self, in_channels: int = 1, hid_features: int = 16, out_labels: int = 7,
                 features_only: bool = False, init: str = 'uniform', train_classifier_only: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.hid_features = hid_features  # number of filters in the first layer (then x2, x4 and x8)
        self.out_labels = out_labels
        self.features_only = features_only
        self.dropout = nn.Dropout(p=0.25)

        self._features = self._make_feature_layers()
        if not features_only:
            self._adaptation = self._make_adaptation_layers()

        assert init in ['normal', 'uniform'], 'The init parameter may only be one between "normal" and "uniform"'
        full_init = self._initialize_normal if init == 'normal' else self._initialize_uniform
        self.apply(full_init)
        if not features_only:
            if train_classifier_only:
                for param in self._features.parameters():
                    param.requires_grad = False
                for param in self._adaptation.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self._features(x)

        if not self.features_only:
            x = self._adaptation(x)
            try:
                (batch, channel, t, h, w) = x.size()    #it is the depth (number of frames in the clip)
            except ValueError:
                (channel, t, h, w) = x.size()
                batch = 1

            k_size = (t, h, w)
            #print("aaa ", x.size())
            #x = self.dropout(x) 
            x = F.avg_pool3d(x, kernel_size=k_size).view(batch, channel)  # in=(N,C,D,H,W) & out=(N,C)

        return x

    @staticmethod
    def _initialize_normal(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()

    @staticmethod
    def _initialize_uniform(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)  # m.weight.data.fill_(1)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)   # m.bias.data.zero_()


    @staticmethod
    def _conv_layer(in_channels, out_channels, first_conv = False):
        kernel_size = (3,3,3)
        spatial_padding = kernel_size[1] // 2
        temporal_padding = kernel_size[0] // 2
        padding = (temporal_padding, spatial_padding, spatial_padding)
        
        layer = [
            SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, first_conv = first_conv)
        ]
        return nn.Sequential(*layer)

    def _make_feature_layers(self):
        layers = [
            # Convolution stack 1
            self._conv_layer(self.in_channels, self.hid_features, first_conv= True),
            self._conv_layer(self.hid_features, self.hid_features),
            #nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  v2
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 2
            self._conv_layer(self.hid_features, self.hid_features * 2),
            self._conv_layer(self.hid_features * 2, self.hid_features * 2),
            #nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 3
            self._conv_layer(self.hid_features * 2, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            self._conv_layer(self.hid_features * 4, self.hid_features * 4),
            #nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 4
            self._conv_layer(self.hid_features * 4, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            #nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # Convolution stack 5
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
            self._conv_layer(self.hid_features * 8, self.hid_features * 8),
        ]
        return nn.Sequential(*layers)

    def _make_adaptation_layers(self):
        layers = [
            # Adaptation layer 1
            SpatioTemporalConv(self.hid_features * 8, self.hid_features * 4, kernel_size=(1,1,1)),
            
            # Adaptation layer 2
            SpatioTemporalConv(self.hid_features * 4, self.out_labels, kernel_size=(1,1,1)),
        ]
        return nn.Sequential(*layers)

