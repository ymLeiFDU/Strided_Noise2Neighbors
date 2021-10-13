
from __future__ import division

import torch
import torch.nn as nn
import json
import os
import sys
import math
import numpy as np
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from utils.losses import MSELoss, SSIM
from torch.autograd import Variable
from shutil import copyfile


def save_args(ckpt_dir, args, train_type):
	# -----------------------------------
	# save running files (.py, .sh, etc.)
	# -----------------------------------
	if not os.path.exists('ckpt_dir'):
		os.mkdir(ckpt_dir)

	os.mkdir(ckpt_dir + '/models')
	os.mkdir(ckpt_dir + '/utils')
	os.mkdir(ckpt_dir + '/data')
	os.mkdir(ckpt_dir + '/results')
	os.mkdir(ckpt_dir + '/denoising_images_' + train_type)

	print('Saving running files ...')
	for f in args:
		print(f)
		copyfile(f, ckpt_dir + '/' + f)


class Tee(object):
	def __init__(self, name, mode):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		sys.stdout = self

	def __def__(self):
		sys.stdout = self.stdout
		self.file.close()

	def write(self, data):
		if not '...' in data:
			self.file.write(data)

		self.stdout.write(data)
	
	def flush(self):
		self.file.flush()


class Evaluator(object):

	def __init__(self, interval_list = None):
		super(Evaluator, self).__init__()
		self.mse = MSELoss()

	def psnr(self, input, target):
		input = input.data.cpu().numpy()
		target = target.data.cpu().numpy()

		mse = np.mean((input - target)**2)
		MAX = np.max(input)
		return 10 * math.log10(MAX**2/mse)

	def ssim(self, input, target):
		ssim = SSIM(window_size=11, size_average=True, val_range=None)

		return ssim(input, target)

		













