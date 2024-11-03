"""
U-Net architecture as introduced in https://arxiv.org/abs/1505.04597
modified:
- use 3D convolution instead of 2D.
- added padding to convolutions to prevent rapid growth of window in upsampling. 
  Without this, the minimum size of context patches (input) is 76x76x76. 
  With it, the minimum input size is 16x16x16.
"""

import torch
import torch.nn as nn

class Conv(nn.Module):
	"""
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(Conv, self).__init__()
		assert kernel_size % 2 == 1
		padding = (kernel_size - 1) // 2

		self.op = nn.Sequential(
			nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
			nn.BatchNorm3d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
			nn.BatchNorm3d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.op(x)


class DownStage(nn.Module):
	"""
	"""
	def __init__(self, in_channels, out_channels):
		super(DownStage, self).__init__()

		self.op = nn.Sequential(
			nn.MaxPool3d(2),
			Conv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.op(x)


class UpStage(nn.Module):
	"""
	"""
	def __init__(self, in_channels, out_channels):
		super(UpStage, self).__init__()

		self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
		self.conv = Conv(in_channels, out_channels)

	def forward(self, x, res):
		x = self.upsample(x)
		y = torch.cat((x, res), dim=1) # concatenate on channel dimension
		return self.conv(y)

class UNet(nn.Module):
	"""
	"""
	def __init__(self, channel_sequence, c=14):
		super(UNet, self).__init__()
		assert len(channel_sequence) == 6

		self.in_stage = Conv(channel_sequence[0], channel_sequence[1])
		self.down_stage2 = DownStage(channel_sequence[1], channel_sequence[2])
		self.down_stage3 = DownStage(channel_sequence[2], channel_sequence[3])
		self.down_stage4 = DownStage(channel_sequence[3], channel_sequence[4])
		self.bottom_stage = DownStage(channel_sequence[4], channel_sequence[5])
		self.up_stage4 = UpStage(channel_sequence[5]+channel_sequence[4], channel_sequence[4])
		self.up_stage3 = UpStage(channel_sequence[4]+channel_sequence[3], channel_sequence[3])
		self.up_stage2 = UpStage(channel_sequence[3]+channel_sequence[2], channel_sequence[2])
		self.up_stage1 = UpStage(channel_sequence[2]+channel_sequence[1], channel_sequence[1])
		self.out_stage = Conv(channel_sequence[1], c)
	
	def forward(self, x):
		s1 = self.in_stage(x)
		s2 = self.down_stage2(s1)
		s3 = self.down_stage3(s2)
		s4 = self.down_stage4(s3)
		sb = self.bottom_stage(s4)
		s4_ = self.up_stage4(sb, s4)
		s3_ = self.up_stage3(s4_, s3)
		s2_ = self.up_stage2(s3_, s2)
		s1_ = self.up_stage1(s2_, s1)
		return self.out_stage(s1_)

def unet_predictor(size=0):
	channel_sequences = {
		0: [1, 64, 128, 256, 512, 1024],
		1: [1,  2,   4,   8,  16,   32],
		2: [1,  4,   8,  16,  32,   64],
		3: [1,  8,  16,  32,  64,  128],
		4: [1, 16,  32,  64, 128,  256],
		5: [1, 32,  64, 128, 256,  512]
	}
	assert size in channel_sequences.keys()
	return UNet(channel_sequences[size])