import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data.dataset import *
from torch.autograd import Variable
from terminaltables import AsciiTable
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils.losses import MSELoss
from tqdm import tqdm
from utils.utils import Evaluator
from scipy.misc import imsave, toimage


def evaluate(epoch, 
	train_type,
	test_loader, 
	model, 
	criterion,
	best_model_wts,
	best_model,
	best_psnr,
	psnr_epoch,
	best_ssim,
	ssim_epoch,
	ckpt_dir,
	cuda = True):

	device = torch.device("cuda" if cuda else "cpu")
	epoch_loss = 0

	evaluator = Evaluator()
	losses = []
	epoch_psnr = []
	epoch_ssim = []
	outputs = []

	for batch_i, (img, label, img_path) in tqdm(enumerate(test_loader)):
		input_var = Variable(img.cuda()).squeeze(1)
		label = Variable(label.cuda()).squeeze(1)

		with torch.no_grad():
			output = model(input_var)
			outputs.append((output.squeeze(0), img_path[0].split('/')[-1].split('.')[0]))

		loss = criterion(output, label)
		losses.append(loss.item())

		# -------------------
		#  calculate metrics
		# -------------------
		psnr = evaluator.psnr(output, label)
		epoch_psnr.append(psnr)
		ssim = evaluator.ssim(output, label)
		epoch_ssim.append(ssim)


	if best_psnr < np.mean(epoch_psnr):
		best_psnr = np.mean(epoch_psnr)
		psnr_epoch = epoch
		best_model_wts = model.state_dict()
		best_model = model

		print('Save results ...')
		predict(outputs, train_type, ckpt_dir)

	if best_ssim < np.mean(epoch_ssim):
		best_ssim = np.mean(epoch_ssim)
		ssim_epoch = epoch



	print('<Eval> [MSE Loss]: {:.4f} [PSNR]: {:.4f} [SSIM]: {:.4f} <Eval>'.format(
		np.mean(losses), np.mean(epoch_psnr), np.mean(epoch_ssim)))
	print('-'*80)

	return best_model_wts, best_model, np.mean(epoch_psnr), best_psnr, psnr_epoch, np.mean(epoch_ssim), best_ssim, ssim_epoch, np.mean(losses)


def predict(outputs, train_type, ckpt_dir):
	names = [item[1] for item in outputs]
	arrays = [item[0][item[0].size(0)//2, :, :].unsqueeze(0) for item in outputs]
	outputs = torch.cat(arrays, dim = 0).data.cpu().numpy()
	print(outputs.shape)
	for i in range(outputs.shape[0]):
		imsave(ckpt_dir + '/denoising_images_' + train_type + '/' + names[i] + '.png', outputs[i, :, :])
		imsave('./predict_results_' + train_type + '/' + names[i] + '.png', outputs[i, :, :])













