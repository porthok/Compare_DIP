import scipy.io
import numpy as np
from PIL import Image
from DIP_main import *


def load_3D(fichier, index):
  mat = scipy.io.loadmat(fichier)
  images = mat[index]
  return np.asarray(images)


if __name__ == '__main__':
    test = sys.argv[1]    
    if test == 1:
		img_noisy_np = load_3D('data/shepp_logan.mat', "x")
		denoised_image, parameters = DIP_3D(img_noisy_np, num_iter=25, LR=0.005, osirim = True, PLOT=False)
	if test == 2:
		img_np = load_3D('data/18am_T2MS_MCT_norm.mat', "MCT18am_norm")
		img_noisy_np = load_3D('data/18am_T2MS_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_18am_norm")
		denoised_image, parameters = DIP_3D(img_noisy_np, img_np=img_np, num_iter=25, LR=0.005, osirim = True, PLOT=False)
	if test == 3:
		img_np = load_3D('data/37c_T3M1_MCT_norm.mat', "MCT37c_norm")
		img_noisy_np = load_3D('data/37c_T3M1_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_37c_norm")
		denoised_image, parameters = DIP_3D(img_noisy_np, img_np=img_np, num_iter=25, LR=0.005, osirim = True, PLOT=False)
