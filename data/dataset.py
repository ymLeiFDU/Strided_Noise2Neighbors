# coding: utf-8
import numpy as np
import torch
import os
import glob
import cv2
import pickle
import random
import torchvision.transforms as transforms
from skimage.util import random_noise
from torch.utils.data import DataLoader

from PIL import Image
from torch.utils.data import Dataset
from data.interpolation import Interpolation
from scipy.misc import imsave, toimage


class DataSet(Dataset):
	def __init__(self, data_path, ratio, mode = 'train'):
		# mode: train 0.8, test 0.2
		self.bilinear = Interpolation()

		inst_C_dicts = []
		inst_L_dicts = []
		inst_N_dicts = []

		for dirname, subdirlist, filelist in os.walk('../processed_LDCT_data/train_dicts_inter8pixel_1ch_128_largeStride'):
			for f in filelist:
				if 'C' in f:
					inst_C_dicts.append(dirname + '/' + f)
				elif 'L' in f:
					inst_L_dicts.append(dirname + '/' + f)
				elif 'N' in f:
					inst_N_dicts.append(dirname + '/' + f)
		inst_C_dicts.sort()
		inst_L_dicts.sort()
		inst_N_dicts.sort()
		self.total_running_dicts = inst_C_dicts + inst_L_dicts + inst_N_dicts
		
		print('>> {} numbers: C {}, L {}, N {}, total {}'.format(
			mode, len(inst_C_dicts), len(inst_L_dicts), len(inst_N_dicts), len(self.total_running_dicts)))

	def __getitem__(self, index):

		img_dict = self.total_running_dicts[index % len(self.total_running_dicts)]
		img_dict = pickle.load(open(img_dict, 'rb'))

		img = img_dict['low']
		img = img[:, :].unsqueeze(0)

		label = img_dict['full']
		label = label[:, :].unsqueeze(0)

		img_path = img_dict['path']

		interplations = []
		inter_1_tensor = img_dict['inter_1']
		inter_1_tensor = inter_1_tensor[:, :]

		inter_2_tensor = img_dict['inter_2']
		inter_2_tensor = inter_2_tensor[:, :]

		inter_3_tensor = img_dict['inter_3']
		inter_3_tensor = inter_3_tensor[:, :]

		interplations = [inter_1_tensor.unsqueeze(0), inter_2_tensor.unsqueeze(0), inter_3_tensor.unsqueeze(0)]
		return img.unsqueeze(0), label.unsqueeze(0), interplations, img_path


class TestDataSet(Dataset):
	def __init__(self, data_path, ratio, mode = 'train'):
		# mode: train 0.8, test 0.2

		LD_path = data_path + '/LD_data'
		ND_path = data_path + '/ND_data'

		LD_cases = []
		ND_cases = []

		for dirname, subdirlist, filelist in os.walk(LD_path):
			for f in filelist:
				LD_cases.append(dirname + '/' + f)

		for dirname, subdirlist, filelist in os.walk(ND_path):
			for f in filelist:
				ND_cases.append(dirname + '/' + f)
		LD_cases.sort()
		ND_cases.sort()

		inst_C_dicts = []
		inst_L_dicts = []
		inst_N_dicts = []
		for x in zip(LD_cases, ND_cases):
			# low case == full case
			low_case = x[0].split('/')[-1].split('.')[0].split('_')[0]
			full_case = x[1].split('/')[-1].split('.')[0].split('_')[0]
			assert low_case == full_case
			if 'C' in low_case and 'C' in full_case:
				inst_C_dicts.append({'low_data': x[0], 'full_data': x[1]})
			elif 'L' in low_case and 'L' in full_case:
				inst_L_dicts.append({'low_data': x[0], 'full_data': x[1]})
			elif 'N' in low_case and 'N' in full_case:
				inst_N_dicts.append({'low_data': x[0], 'full_data': x[1]})

		running_dicts_C = self._split_train_test(dicts = inst_C_dicts, ratio = ratio, mode = mode)
		running_dicts_L = self._split_train_test(dicts = inst_L_dicts, ratio = ratio, mode = mode)
		running_dicts_N = self._split_train_test(dicts = inst_N_dicts, ratio = ratio, mode = mode)
		self.total_running_dicts = running_dicts_C + running_dicts_L + running_dicts_N

		self.total_data = []
		for i, item in enumerate(self.total_running_dicts):
			low_data = pickle.load(open(item['low_data'], 'rb'))
			image = torch.from_numpy(low_data).type(torch.FloatTensor)
			full_data = pickle.load(open(item['full_data'], 'rb'))
			label = torch.from_numpy(full_data).type(torch.FloatTensor)
			self.total_data.append({'low':image, 'full':label, 'path':item['low_data']})
		

		print('>> {} numbers: C {}, L {}, N {}, total {}'.format(
			mode, len(running_dicts_C), len(running_dicts_L), len(running_dicts_N), len(self.total_running_dicts)))

	def __getitem__(self, index):

		img_dict = self.total_data[index % len(self.total_data)]
		img_tensor = img_dict['low']

		label = img_dict['full']

		channel = 1
		img_tensor = img_tensor[img_tensor.size(0)//2, :, :].unsqueeze(0)
		label = label[label.size(0)//2, :, :].unsqueeze(0)
		
		img_path = img_dict['path']

		return img_tensor.unsqueeze(0), label.unsqueeze(0), img_path

	def _split_train_test(self, dicts, mode, ratio):

		if mode == 'train':
			return dicts[0 : int(ratio * len(dicts))]
		elif mode == 'test':
			return dicts[int((1 - ratio) * len(dicts)):]

	def __len__(self):
		return len(self.total_running_dicts)




if __name__ == '__main__':
	
	train_dataset = DataSet(data_path = '/mnt/pami14/ymlei/processed_LDCT_data', ratio = 0.8, mode = 'train')
	# train_loader = DataLoader(
	# 	train_dataset,
	# 	batch_size = 1,
	# 	shuffle = False,
	# 	num_workers = 16
	# 	)

	# for img, label, interplations, img_path in train_loader:
		
	# 	print(img.size(), label.size(), interplations[0].size())


		

















