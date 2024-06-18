import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from src.train import N_FRAMES

def conv2D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
	return output_shape

def conv3D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
				  np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
	return output_shape 

class EncoderCNN(nn.Module):
    def __init__(self,
                 img_x=200,
                 img_y=200,
                 input_channels=1,
                 fc_hidden=1024,
                 dropout_pct=0.5,
                 CNN_embed_dim=128):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # Fully connected layer hidden nodes
        self.fc_hidden = fc_hidden
        self.dropout_pct = dropout_pct

        # CNN architectures
        self.ch1, self.ch2, self.ch3, self.ch4, self.ch5 = 32, 64, 128, 256, 512
        self.k1, self.k2, self.k3, self.k4, self.k5 = (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4, self.s5 = (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4, self.pd5 = (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y),
                                                 self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2,
                                                 self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3,
                                                 self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4,
                                                 self.k4, self.s4)
        self.conv5_outshape = conv2D_output_size(self.conv4_outshape, self.pd5,
                                                 self.k5, self.s5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=self.ch1,
                      kernel_size=self.k1,
                      stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1,
                      out_channels=self.ch2,
                      kernel_size=self.k2,
                      stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2,
                      out_channels=self.ch3,
                      kernel_size=self.k3,
                      stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3,
                      out_channels=self.ch4,
                      kernel_size=self.k4,
                      stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch4,
                      out_channels=self.ch5,
                      kernel_size=self.k5,
                      stride=self.s5,
                      padding=self.pd5),
            nn.BatchNorm2d(self.ch5),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout(self.dropout_pct)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(
            self.ch5,
            self.fc_hidden
        ) # Fully connected layer, output k classes
        self.fc2 = nn.Linear(
            self.fc_hidden,
            self.CNN_embed_dim
        )  # Output = CNN embedding latent variables

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten the output of conv
        x = self.drop(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc2(x) # CNN embedding
        return x

class DecoderFC(nn.Module):
    def __init__(self,
                input_dim=3 * 512,
                FC_layer_nodes=[256, 256, 64],
                dropout_pct=0.5,
                output_dim=1):
        super(DecoderFC, self).__init__()
    
        self.FC_input_size = input_dim
        self.FC_layer_nodes = FC_layer_nodes
        self.dropout_pct = dropout_pct
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.FC_input_size, self.FC_layer_nodes[0])
        self.fc2 = nn.Linear(self.FC_layer_nodes[0], self.FC_layer_nodes[1])
        self.fc3 = nn.Linear(self.FC_layer_nodes[1], self.FC_layer_nodes[2])
        if len(self.FC_layer_nodes) == 3:
            self.fc4 = nn.Linear(self.FC_layer_nodes[2], self.output_dim)
        elif len(self.FC_layer_nodes) == 4:
            self.fc4 = nn.Linear(self.FC_layer_nodes[2], self.FC_layer_nodes[3])
            self.fc5 = nn.Linear(self.FC_layer_nodes[3], self.output_dim)
        self.drop = nn.Dropout(self.dropout_pct)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc4(x)
        if len(self.FC_layer_nodes) == 4:
            x = F.elu(x)
            x = self.drop(x)
            x = self.fc5(x)
        if self.output_dim == 1:
            return torch.sigmoid(x)
        else:
            return x
    
class EstimationDecoderFC(nn.Module):
    def __init__(self,
                input_dim=6,
                FC_layer_nodes=[64, 64, 32],
                dropout_pct=0.0,
                output_dim=1):
        super(EstimationDecoderFC, self).__init__()
    
        self.FC_input_size = input_dim
        self.FC_layer_nodes = FC_layer_nodes
        self.dropout_pct = dropout_pct
        self.output_dim = output_dim

        assert len(FC_layer_nodes) == 3

        self.fc1 = nn.Linear(self.FC_input_size, self.FC_layer_nodes[0])
        self.fc2 = nn.Linear(self.FC_layer_nodes[0], self.FC_layer_nodes[1])
        self.fc3 = nn.Linear(self.FC_layer_nodes[1], self.FC_layer_nodes[2])
        self.fc4 = nn.Linear(self.FC_layer_nodes[2], self.output_dim)
        self.drop = nn.Dropout(self.dropout_pct)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.drop(x)
        x = self.fc4(x)
        return torch.sigmoid(x)

class ForceFC(nn.Module):
    def __init__(self, input_dim=1, hidden_size=16, output_dim=16, dropout_pct=0.5):
        super(ForceFC, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_pct = dropout_pct

        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.output_dim)
        # self.drop = nn.Dropout(self.dropout_pct)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.elu(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = F.elu(x)
        # x = self.drop(x)
        # x = self.fc3(x)
        return x
    
class WidthFC(nn.Module):
    def __init__(self, input_dim=1, hidden_size=16, output_dim=16, dropout_pct=0.5):
        super(WidthFC, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_pct = dropout_pct

        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.output_dim)
        # # self.fc3 = nn.Linear(self.hidden_size, self.output_dim)
        # self.drop = nn.Dropout(self.dropout_pct)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.elu(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = F.elu(x)
        # x = self.drop(x)
        # x = self.fc3(x)
        return x
