import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def conv5x5(in_planes, out_planes):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=0, bias=False, dilation = 1),
		nn.ReLU())

def deconv5x5(in_planes, out_planes):
	return nn.ConvTranspose2d(in_planes, out_planes, 5, stride = 1, padding = 0)


class RED_CNN(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(RED_CNN, self).__init__()

		self.conv1 = conv5x5(in_ch, 96)
		self.conv2 = conv5x5(96, 96)
		self.conv3 = conv5x5(96, 96)
		self.conv4 = conv5x5(96, 96)
		self.conv5 = conv5x5(96, 96)

		self.deconv1 = deconv5x5(96, 96)
		self.relu1 = nn.ReLU(inplace = True)
		self.deconv2 = deconv5x5(96, 96)
		self.relu2 = nn.ReLU(inplace = True)
		self.deconv3 = deconv5x5(96, 96)
		self.relu3 = nn.ReLU(inplace = True)
		self.deconv4 = deconv5x5(96, 96)
		self.relu4 = nn.ReLU(inplace = True)
		self.deconv5 = deconv5x5(96, out_ch)
		self.relu5 = nn.ReLU(inplace = True)


	def forward(self, x):

		feat1 = x
		x = self.conv1(x)
		x = self.conv2(x)
		feat2 = x
		x = self.conv3(x)
		x = self.conv4(x)
		feat3 = x
		x = self.conv5(x)

		x = self.deconv1(x)
		x = self.relu1(x+feat3)
		x = self.deconv2(x)
		x = self.relu2(x)
		x = self.deconv3(x)
		x = self.relu3(x+feat2)
		x = self.deconv4(x)
		x = self.relu4(x)
		x = self.deconv5(x)
		x = self.relu5(x+feat1)

		return x













