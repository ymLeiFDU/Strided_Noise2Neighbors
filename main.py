
import yaml
import torch
import argparse
import configs
import datetime
import sys
import os
import pickle

from train import train
from torch.utils.data import DataLoader
from data.dataset import DataSet, TestDataSet, DataSetN2N
from models.unet import UNet
from models.unet3d import UNet_3d
from models.red_cnn import RED_CNN
from models.cnn10 import CNN10
from utils.losses import MSELoss
from evaluate import evaluate
from torch.autograd import Variable
from utils.utils import *
from utils import utils
from torch.optim.lr_scheduler import MultiStepLR, StepLR



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 10, help = 'num of epochs')
	parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
	parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
	parser.add_argument('--weight_decay', type = float, default = 0.0001, help = 'weight decay')
	parser.add_argument('--train_batch_size', type = int, default = 8, help = 'train batch size')
	parser.add_argument('--step_size', type = int, default = 10, help = 'lr decay step')

	parser.add_argument('--test_batch_size', type = int, default = 8, help = 'test batch size')
	parser.add_argument('--workers', type = int, default = 4, help = 'num of workers')
	parser.add_argument('--use_gpu', type = bool, default = True, help = 'whether or not using gpus')
	parser.add_argument('--gpus', type = str, default = 2, help = 'gpu numbers')
	parser.add_argument('--set_device', type = int, default = 0, help = 'torch.cuda.set_device')
	parser.add_argument('--ckpt_interval', type = int, default = 10, help = 'save ckpt per ckpt_interval')
	parser.add_argument('--model', type = str, default = 'fcn', help = 'save ckpt per ckpt_interval')
	parser.add_argument('--train_type', type = str, default = 'sup', help = 'sup or self')
	parser.add_argument('--pretrained', type = int, default = 0)
	opt = parser.parse_args()
	print('< Settings > :')
	print(opt)
	print('Training torch version: ', torch.__version__)
	with open('configs/config.yaml', 'r') as f:
		cfgs = yaml.load(f)

	# ----------
	# Save files
	# ----------
	project_dir = '/your project path/'
	if not os.path.exists(project_dir + '/ckpt/' + opt.model):
		os.mkdir(project_dir + '/ckpt/' + opt.model)
	t = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	checkpoint_dir = project_dir + '/ckpt/' + opt.model + '/' + t
	print('Files saving dir: ', checkpoint_dir)
	args = cfgs['project_files']

	# save files
	save_args(checkpoint_dir, args, opt.train_type)
	logger = Tee(checkpoint_dir + '/log.txt', 'a')

	# ---------
	# Load data
	# ---------
	print('> load data ...')
	interval = 0.5

	train_data = DataSet(data_path = cfgs['data_path'], ratio = 0.8, mode = 'train')
	train_loader = DataLoader(
		train_data,
		batch_size = opt.train_batch_size,
		shuffle = True,
		num_workers = opt.workers
		)

	test_data = TestDataSet(data_path = cfgs['data_path'], ratio = 0.2, mode = 'test')
	test_loader = DataLoader(
		test_data,
		batch_size = opt.test_batch_size,
		shuffle = True,
		num_workers = opt.workers
		)
	train_num = train_data.__len__()
	test_num = test_data.__len__()

	# ----------------
	# Running settings
	# ----------------
	cuda = torch.cuda.is_available() and opt.use_gpu
	
	if cuda:
		torch.cuda.set_device(opt.set_device)
		if opt.model == 'UNet':
			model = UNet(in_channel = 1, out_channel = 1).cuda()
		elif opt.model == 'RED_CNN':
		# model = UNet_3d(in_ch = 32, out_ch = 32).cuda()
			model = RED_CNN(in_ch = 1, out_ch = 1).cuda()
		# model = CNN10(in_ch = 32, out_ch = 32).cuda()
		if len(opt.gpus) > 1:
			model = nn.DataParallel(model, device_ids = [int(x) for x in opt.gpus.split(',')])

	optimizer_main = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
	scheduler_main = StepLR(optimizer_main, step_size = opt.step_size, gamma=0.1)
	criterion = MSELoss()

	# -------
	#  train
	# -------
	save_time = datetime.datetime.now()
	best_psnr = 0
	psnr_epoch = 0
	best_ssim = 0
	ssim_epoch = 0
	best_model_wts = model.state_dict()
	best_model = model

	epoch_psnr_list = []
	epoch_ssim_list = []
	epoch_loss_list = []

	for epoch in range(opt.epochs):

		start_time = datetime.datetime.now()
		print('Epoch {} / {} ({}) {} | Bests: PSNR: {:.4f}(ep:{}) SSIM: {:.4f}(ep:{})'.format(
			epoch, opt.epochs, opt.model, t, best_psnr, psnr_epoch, best_ssim, ssim_epoch))
		print('== Model: {}, Method: {} =='.format(opt.model, opt.train_type))
		model = train(
			model = model,
			optimizer_main = optimizer_main,
			scheduler_main = scheduler_main,
			train_loader = train_loader,
			criterion = criterion,
			train_num = train_num,
			epoch = epoch,
			opt = opt,
			)

		best_model_wts, best_model, epoch_psnr, best_psnr, psnr_epoch, epoch_ssim, best_ssim, ssim_epoch, epoch_loss = evaluate(
			epoch = epoch, 
			train_type = opt.train_type,
			test_loader = test_loader, 
			model = model,
			criterion = criterion,
			best_model_wts = best_model_wts,
			best_model = best_model,
			best_psnr = best_psnr,
			psnr_epoch = psnr_epoch,
			best_ssim = best_ssim,
			ssim_epoch = ssim_epoch,
			ckpt_dir = checkpoint_dir,
			cuda = cuda)

		epoch_psnr_list.append(epoch_psnr)
		epoch_ssim_list.append(epoch_ssim)
		epoch_loss_list.append(epoch_loss)

		end_time = datetime.datetime.now()
		print('elapsed time: {} s'.format((end_time - start_time).seconds))
		print()

		if epoch == opt.epochs - 1:
			if not os.path.exists(checkpoint_dir + '/trained_models'):
				os.mkdir(checkpoint_dir + '/trained_models')
			torch.save(best_model_wts, checkpoint_dir + '/trained_models/' + opt.train_type + '_{}_epoch_{}_weights.pth'.format(opt.model, psnr_epoch))
			torch.save(best_model, checkpoint_dir + '/trained_models/' + opt.train_type + '_{}_epoch_{}_model.pth'.format(opt.model, psnr_epoch))

	pickle.dump(epoch_psnr_list, open(checkpoint_dir + '/results/epoch_psnr_list.pkl', 'wb'))
	pickle.dump(epoch_ssim_list, open(checkpoint_dir + '/results/epoch_ssim_list.pkl', 'wb'))
	pickle.dump(epoch_loss_list, open(checkpoint_dir + '/results/epoch_loss_list.pkl', 'wb'))

























