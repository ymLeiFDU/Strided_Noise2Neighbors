	# coding: utf-8
import torch
import torch.nn.functional as F
import math
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from data.dataset import *
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils.losses import MSELoss
from utils.utils import Evaluator
from utils.mask import Masker
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR, StepLR


def train(
		model,
		criterion,
		optimizer_main,
		scheduler_main,
		train_num,
		train_loader,
		epoch,
		opt
		):

	model.train()
	adjust_learning_rate(optimizer_main, epoch, opt)

	evaluator = Evaluator()
	masker = Masker(width = 3, mode='interpolate')

	scheduler_main.step()
	
	losses = []
	epoch_psnr = []
	epoch_ssim = []

	for batch_i, (img, label, interplations, img_path) in tqdm(enumerate(train_loader)):
		input_var = Variable(img.cuda()).squeeze(1)
		# input_var_2 = Variable(img_2.cuda())
		label = Variable(label.cuda()).squeeze(1)

		inter_1_tensor = Variable(interplations[0].cuda())
		inter_2_tensor = Variable(interplations[1].cuda())
		inter_3_tensor = Variable(interplations[2].cuda())

		optimizer_main.zero_grad()
		output = model(input_var)

		loss = criterion(output, inter_1_tensor) + criterion(output, inter_1_tensor) + criterion(output, inter_1_tensor)

		loss.backward()
		optimizer_main.step()

		losses.append(loss.item())

		# -------------------
		#  calculate stats
		# -------------------
		psnr = evaluator.psnr(output, label)
		epoch_psnr.append(psnr)
		ssim = evaluator.ssim(output, label)
		epoch_ssim.append(ssim)

	

	# -------------------
	#  calculate metrics
	# -------------------
	print('-'*80)
	print('<Train> [MSE Loss]: {:.4f} [PSNR]: {:.4f} [SSIM]: {:.4f} <Train>'.format(
		np.mean(losses), np.mean(epoch_psnr), np.mean(epoch_ssim))
		)

	return model


def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr
    if True:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.epochs))
    # else:  # stepwise lr schedule
    #     for milestone in opt.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    print('lr: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

















