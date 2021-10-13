import numpy as np
import cv2
from PIL import Image
from scipy.misc import imsave, toimage

class Interpolation():

	def __init__(self, kernel_size = 3, padding = False):
		self.kernel_size = kernel_size
		self.padding = padding


	def _bilinear(self, patch_array):
		center_x, center_y = self.kernel_size//2, self.kernel_size//2
		patch_array[center_x, center_y] = 0
		f = patch_array

		u, v = 0.5, 0.5

		value1 = (1-u)*(1-v)*f[0,0] + (1-u)*v*f[0,2] + u*(1-v)*f[2,0] + u*v*f[2,2]

		value2 = (1-u)*(1-v)*f[0,1] + (1-u)*v*f[1,2] + u*(1-v)*f[2,1] + u*v*f[1,0]

		value = (value1 + value2) / 2

		patch_array[center_x, center_y] = value
		return patch_array

	def _mean(self, patch_array):
		center_x, center_y = self.kernel_size//2, self.kernel_size//2
		patch_array[center_x, center_y] = 0
		f = patch_array
		value = np.mean([f[center_x-1, center_y-1], f[center_x-1, center_y+1], f[center_x+1, center_y-1], f[center_x+1, center_y+1]])

		patch_array[center_x, center_y] = value
		return patch_array

	def interplate(self, image, idxs = [1, 1], patch_size = 1):
		if self.padding == False:

			# image = _tuple[0]
			# idxs = _tuple[1]
			# patch_size = _tuple[2]
			ph, pw = patch_size, patch_size # selected local patch
			h, w = (image.shape[0]-2)//ph, (image.shape[1]-2)//pw

			ph_idx, pw_idx = idxs[0], idxs[1] # choose the target dropping point in loacl patch
			for i in range(h):
				ii = i + ph_idx
				for j in range(w):
					jj = j + pw_idx
					center_x, center_y = ii + i*(ph-1), jj + j*(pw-1)
					image[center_x, center_y] = 0 # dropping point
					patch = image[center_x-1:center_x+2, center_y-1:center_y+2]
					patch = self._bilinear(patch)
					image[center_x-1:center_x+2, center_y-1:center_y+2] = patch
		
		elif self.padding == True:
			pass

		return image























