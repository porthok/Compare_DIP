
#Import libs

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from models.skip import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from utils.denoising_utils import *
import sys
import time
from PIL import Image


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Redefine DIP functions

"""
Entrees : 
- size     : dimensions de img_noisy_torchl'image en entree
- INPUT       : 'noise' pour filtrer les tenseurs avec du bruitn, et 'meshgrid' pour utiliser np.meshgrid
- pad         : type de padding utilisé
Sorties :
- net             : reseau de neurones utilisé par l'algorithme
- net_input       : tenseur contenant l'image 2D de bruit blanc avec des valeurs entre 0 et 1, taille = (1,1,taille,taille)
- mse             : mean squared error utilisé pour la loss du réseau
"""  
def Setup(size, INPUT = 'noise', pad = 'reflection'):
  # Deduction du nombre de canaux en entree
  if len(size) == 2:
    input_depth = 1
  else:
    input_depth = size[2]
  
  # Creation du reseau de neurones
  net = skip(
              input_depth, 1, 
              num_channels_down = [128, 128, 128, 128, 128], 
              num_channels_up   = [128, 128, 128, 128, 128],
              num_channels_skip = [4, 4, 4, 4, 4], 
              upsample_mode='bilinear',
              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
  net = net.type(dtype)      
  
  # Creation de l'image d'entree (bruit blanc) de taille (1,1,taille[0],taille[1])  
  net_input = get_noise(input_depth, INPUT, (size[0], size[1])).type(dtype).detach()
  
  # Loss
  mse = torch.nn.MSELoss().type(dtype)
  return net, net_input, mse


"""
Entrees : 
- size     : dimensions de img_noisy_torchl'image en entree
- INPUT       : 'noise' pour filtrer les tenseurs avec du bruitn, et 'meshgrid' pour utiliser np.meshgrid
- pad         : type de padding utilisé
Sorties :
- net             : reseau de neurones utilisé par l'algorithme
- net_input       : tenseur contenant l'image 2D de bruit blanc avec des valeurs entre 0 et 1, taille = (1,1,taille,taille, taille)
- mse             : mean squared error utilisé pour la loss du réseau
"""  
def Setup3D(size, INPUT = 'noise', pad = 'reflection'):
  # Deduction du nombre de canaux en entree
  if len(size) == 3:
    input_depth = 1
  else:
    input_depth = size[3]
  
  # Creation du reseau de neurones
  net = skip3D(
              input_depth, 1, 
              num_channels_down = [128, 128, 128, 128, 128], 
              num_channels_up   = [128, 128, 128, 128, 128],
              num_channels_skip = [4, 4, 4, 4, 4], 
              upsample_mode='trilinear',
              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
  net = net.type(dtype)
  
  # Creation de l'image d'entree (bruit blanc) de taille (1,1,taille[0],taille[1])  
  net_input = get_noise3D(input_depth, INPUT, (size[0], size[1], size[2])).type(dtype).detach()
  #net_input = np.expand_dims(net_input, axis=(0))
  
  # Loss
  mse = torch.nn.MSELoss().type(dtype)
  return net, net_input, mse


"""
Entree : 
- params     : Dictionnaire de parametres comportant :
                  reg_noise_std     : deviation standard du bruit qui va etre utilisee pour bruite l'image de depart donnee au reseau. Si 0, alors aucun bruit n'est ajoute
                  net_input_saved   : sauvegarde de l'image donnee a DIP en entree lors de l'iteration precedente
                  noise             : image donnee à DIP lors de l'iteration precedente en entree qui va etre normalisee
                  net               : reseau de neurones utilise pour DIP
                  out_avg           : moyenne ponderee de toutes les sorties du reseau au cours du temps pour une itération du RED
                  mse               : mean squared error utilisé pour la loss du réseau
                  psrn_noisy_last   : sauvegarde du psnr de l'image renvoyee par DIP lors de l'affichage precedent
                  i                 : nombre d'iterations faites par DIP 
                  img_noisy_np      : image 2D avec des valeurs entre 0 et 1 en format array de numpy et bruitee que DIP va essayer d'estimer
                  img_noisy_torch   : img_noisy_np mise sous forme de Tenseur pour pouvoir etre utilisee par le reseau de neurones
                  net_input         : bruit blanc donne à DIP. C'est l'image de depart du reseau qu'il va essayer de faire converger vers img_noisy_np
                  ar                : image de base au format numpy array avec des valeurs entre 0 et 1 avant quelle soit bruitee
                  PLOT              : booleen pour plot les images sorties par DIP toutes les 'show_every' itérations
                  show_every        : nombre d'itérations entre chaque affichage de l'image de DIP
                  figsize           : taille d'affichage des images de sorties de DIP
                  exp_weight        : poids pour pondérer l'image moyenne (moyenne = moyenne * exp_weight + nouvelle valeur * (1-exp_weight))
Sorties :
- total_loss : Loss renvoyee par le reseau DIP à la fin des iterations entre l'image renvoyee et l'image a approximer
"""
def closure(params):
  if params['reg_noise_std'] > 0:
      params['net_input'] = params['net_input_saved'] + params['noise'].normal_() * params['reg_noise_std']
  
  # out : Tenseur renvoye par le reseau de neurones avec net_input le bruit blanc donne en entre au reseau
  out = params['net'](params['net_input'])

  # out_avg : Tenseur moyen des sorties du reseau au cours du temps
  out_avg = None

  # Met à jour l'image moyenne renvoyee
  # regle de mise a jour :
  # image moyenne = image moyenne                                                       si image moyenne = Rien
  # image moyenne = image moyenne * coefficient + sortie du reseau * (1 - coefficient)  sinon
  if params['out_avg'] is None:
      params['out_avg'] = out.detach()
  else:
      params['out_avg'] = params['out_avg'] * params['exp_weight'] + out.detach() * (1 - params['exp_weight'])
  
  # Calcul de la Loss par rapport à la sortie du reseau et img_noisy_torch qui est le Tenseur renvoye par la fonction Setup()
  total_loss = params['mse'](out, params['img_noisy_torch'])
  # Backtracking
  total_loss.backward()

  # caclul du PSNR entre l'image donnee a RED et la sortie du reseau
  # img_noisy_np : image 2D avec valeurs entre 0 et 1 que DIP essaie d'approximer sous format array de numpy
  # out.detach().cpu().numpy() : de taille (1, nb canaux = 1,nb_colonnes,nb_lignes) = sortie du reseau remis sous forme array numpy
  psrn_noisy = peak_signal_noise_ratio(params['img_noisy_np'], out.detach().cpu().numpy()[0][0]) 
  # ar : de taille (1,taille,taille) car 1 canal
  psrn_gt    = peak_signal_noise_ratio(params['ar'], out.detach().cpu().numpy()[0]) 
  psrn_gt_sm = peak_signal_noise_ratio(params['ar'], params['out_avg'].detach().cpu().numpy()[0]) 
  
  # Affiche les résultats Loss et PSNR
  print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (params['i'], total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
  # Plot l'image renvoyee par DIP a la derniere iteration et la moyenne de toutes les images renvoyees jusque là
  if  params['PLOT'] and params['i'] % params['show_every'] == 0:    
      out_np = torch_to_np(out)
      if type(params['img_np']) is np.ndarray :
        plot_image_grid([np.clip(params['expanded_img_noisy_np'], 0, 1), np.clip(out_np, 0, 1), np.clip(params['img_np'], 0, 1)], factor=params['figsize'], nrow=1)
      else:        
        plot_image_grid([np.clip(params['expanded_img_noisy_np'], 0, 1), np.clip(out_np, 0, 1)], factor=params['figsize'], nrow=1)

      print("noisy / output : ", psrn_noisy, "initial / output : ", psrn_gt, "initial / avg_output : " , psrn_gt_sm)

  # Backtrackingthe generated image to
  if params['i'] % params['show_every']:
      if psrn_noisy - params['psrn_noisy_last'] < -5: 
          print('Falling back to previous checkpoint.')
          for new_param, net_param in zip(params['last_net'], params['net'].parameters()):
              net_param.data.copy_(new_param.cuda())
          return total_loss*0
      else:
          params['last_net'] = [x.detach().cpu() for x in params['net'].parameters()]
          params['psrn_noisy_last'] = psrn_noisy
          
  params['i'] += 1
  return total_loss


"""
Entree : 
- params     : Dictionnaire de parametres comportant :
                  reg_noise_std     : deviation standard du bruit qui va etre utilisee pour bruite l'image de depart donnee au reseau. Si 0, alors aucun bruit n'est ajoute
                  net_input_saved   : sauvegarde de l'image donnee a DIP en entree lors de l'iteration precedente
                  noise             : image donnee à DIP lors de l'iteration precedente en entree qui va etre normalisee
                  net               : reseau de neurones utilise pour DIP
                  out_avg           : moyenne ponderee de toutes les sorties du reseau au cours du temps pour une itération du RED
                  mse               : mean squared error utilisé pour la loss du réseau
                  psrn_noisy_last   : sauvegarde du psnr de l'image renvoyee par DIP lors de l'affichage precedent
                  i                 : nombre d'iterations faites par DIP 
                  img_noisy_np      : image 2D avec des valeurs entre 0 et 1 en format array de numpy et bruitee que DIP va essayer d'estimer
                  img_noisy_torch   : img_noisy_np mise sous forme de Tenseur pour pouvoir etre utilisee par le reseau de neurones
                  net_input         : bruit blanc donne à DIP. C'est l'image de depart du reseau qu'il va essayer de faire converger vers img_noisy_np
                  ar                : image de base au format numpy array avec des valeurs entre 0 et 1 avant quelle soit bruitee
                  PLOT              : booleen pour plot les images sorties par DIP toutes les 'show_every' itérations
                  show_every        : nombre d'itérations entre chaque affichage de l'image de DIP
                  figsize           : taille d'affichage des images de sorties de DIP
                  exp_weight        : poids pour pondérer l'image moyenne (moyenne = moyenne * exp_weight + nouvelle valeur * (1-exp_weight))
Sorties :
- total_loss : Loss renvoyee par le reseau DIP à la fin des iterations entre l'image renvoyee et l'image a approximer
"""
def closure3D(params):
  if params['reg_noise_std'] > 0:
      params['net_input'] = params['net_input_saved'] + params['noise'].normal_() * params['reg_noise_std']
  
  # out : Tenseur renvoye par le reseau de neurones avec net_input le bruit blanc donne en entre au reseau
  out = params['net'](params['net_input'])
  # out_avg : Tenseur moyen des sorties du reseau au cours du temps
  out_avg = None

  # Met à jour l'image moyenne renvoyee
  # regle de mise a jour :
  # image moyenne = image moyenne                                                       si image moyenne = Rien
  # image moyenne = image moyenne * coefficient + sortie du reseau * (1 - coefficient)  sinon
  if params['out_avg'] is None:
      params['out_avg'] = out.detach()
  else:
      params['out_avg'] = params['out_avg'] * params['exp_weight'] + out.detach() * (1 - params['exp_weight'])
  # Calcul de la Loss par rapport à la sortie du reseau et img_noisy_torch qui est le Tenseur renvoye par la fonction Setup()
  total_loss = params['mse'](out, params['img_noisy_torch'])
  # Backtracking
  total_loss.backward()
  
  # caclul du PSNR entre l'image donnee a RED et la sortie du reseau
  # img_noisy_np : image 2D avec valeurs entre 0 et 1 que DIP essaie d'approximer sous format array de numpy
  # out.detach().cpu().numpy() : de taille (1, nb canaux = 1,nb_colonnes,nb_lignes, nb_couches) = sortie du reseau remis sous forme array numpy
  out_numpy = out.detach().cpu().numpy()
  psrn_noisy = peak_signal_noise_ratio(params['img_noisy_np'], out_numpy[0][0])
  # ar : de taille (1,taille,taille) car 1 canal
  psrn_gt    = peak_signal_noise_ratio(params['ar'], out_numpy[0]) 
  psrn_gt_sm = peak_signal_noise_ratio(params['ar'], params['out_avg'].detach().cpu().numpy()[0])  

  # Plot l'image renvoyee par DIP a la derniere iteration et la moyenne de toutes les images renvoyees jusque là
  if  params['PLOT'] and params['i'] % params['show_every'] == 0:    
      #out_np = torch_to_np(out)
      slice_index = params['expanded_img_noisy_np'].shape[3]//2
      if params['osirim']:
        im1 = Image.fromarray(out_numpy[0, 0, :, :, slice_index] * 255).convert('L').save(params['save_directory'] + '/DIP3D_Unfolded_DIPImageJpg_Tr%s_epoch%s_lr%.2e.jpg' % (params['i'], params['num_iter'], params['LR']))
      else:
        if type(params['img_np']) is np.ndarray :
          plot_image_grid([np.clip(params['expanded_img_noisy_np'][:, :, :, slice_index], 0, 1), np.clip(out_numpy[0, :, :, :, slice_index], 0, 1), np.clip(params['img_np'][:, :, :, slice_index], 0, 1)], factor=params['figsize'], nrow=1)
        else:
          plot_image_grid([np.clip(params['expanded_img_noisy_np'][:, :, :, slice_index], 0, 1), np.clip(out_numpy[0, :, :, :, slice_index], 0, 1)], factor=params['figsize'], nrow=1)

  params['evo_mse'][params['i']] = total_loss
  params['evo_psnr'][params['i']] = psrn_noisy
  params['evo_ssim'][params['i']] = ssim(params['img_noisy_np'], out_numpy[0, 0, :, :, :], data_range=params['img_noisy_np'].max() - params['img_noisy_np'].min())
  # Backtracking the generated image to
  if params['i'] % params['show_every']:
      # Affiche les résultats Loss et PSNR
      if params['osirim']:
        params['log'].write('Iteration %05d    Loss: %f   SSIM: %f   PSNR_noisy: %f   PSRN_gt: %f   PSNR_gt_sm: %f\n' % (params['i'], total_loss.item(), params['evo_ssim'][params['i']], psrn_noisy, psrn_gt, psrn_gt_sm))
      else:
        print('Iteration %05d    Loss %f   SSIM: %f   PSNR_noisy: %f   PSRN_gt:   %f PSNR_gt_sm: %f' % (params['i'], total_loss.item(), params['evo_ssim'][params['i']], psrn_noisy, psrn_gt, psrn_gt_sm))    
      #print("noisy / output : ", psrn_noisy, "initial / output : ", psrn_gt, "initial / avg_output : " , psrn_gt_sm)
      if psrn_noisy - params['psrn_noisy_last'] < -5: 
          print('Falling back to previous checkpoint.')
          for new_param, net_param in zip(params['last_net'], params['net'].parameters()):
              net_param.data.copy_(new_param.cuda())
          return total_loss*0
      else:
          params['last_net'] = [x.detach().cpu() for x in params['net'].parameters()]
          params['psrn_noisy_last'] = psrn_noisy
          
  params['i'] += 1
  
  return total_loss

# Main functions

def DIP_couche(img_noisy_np, img_np = None, PLOT = True, target = 0, num_iter = 250, LR = 0.01, OPTIMIZER='adam'): #Main function
  if len(img_noisy_np.shape) == 2:
    nb_couches = 1;
  else:
    nb_couches = img_noisy_np.shape[2]
    
  out_np = np.zeros(img_noisy_np.shape)
  parameters = []

  if target != 0:
    out_np[0], parameters[0] = DIP_2D(img_noisy_np[target], img_np[target], num_iter = num_iter, LR = LR, OPTIMIZER = OPTIMIZER)
  else:
    if len(img_noisy_np.shape) == 2:
      out_np[0], parameters[0] = DIP_2D(img_noisy_np, img_np, num_iter = num_iter, LR = LR, OPTIMIZER = OPTIMIZER)
    else:
      for i in range(nb_couches):
        if type(img_np) is np.ndarray:
          out_np[i, :, :], param = DIP_2D(img_noisy_np[i], img_np[i], num_iter = num_iter, LR = LR, OPTIMIZER = OPTIMIZER)
        else:
          out_np[i, :, :], param = DIP_2D(img_noisy_np[i], num_iter = num_iter, LR = LR, OPTIMIZER = OPTIMIZER)
        parameters.append(param)
  return out_np, parameters
  

def DIP_2D(img_noisy_np, img_np = None, PLOT = True, num_iter = 250, LR = 0.01, OPTIMIZER='adam'): #Main function
  #Set parameters  
  closure_params = {}
  ar = np.array(img_noisy_np)[None, ...]
  ar = ar.astype(np.float32) / 255

  # conversion de l'image donnee en entree en Tenseur pour Torch
  closure_params['img_noisy_torch'] = np_to_torch(img_noisy_np).type(dtype)

  # Setup the DIP
  net, net_input, closure_params['mse'] = Setup(img_noisy_np.shape)
  p = get_params('net', net, net_input)
  closure_params['net'] = net
  closure_params['last_net'] = net
  closure_params['net_input'] = net_input
  closure_params['reg_noise_std'] = 1./30. #set to 1./20. for sigma=50
  closure_params['net_input_saved'] = closure_params['net_input'].detach().clone()
  closure_params['noise'] = closure_params['net_input'].detach().clone()
  closure_params['out_avg'] = None
  closure_params['psrn_noisy_last'] = 0
  closure_params['i'] = 0
  if type(img_np) is np.ndarray:
    closure_params['img_np'] = np.expand_dims(img_np, axis=(0))
  else:
    closure_params['img_np'] = None
  closure_params['img_noisy_np'] = img_noisy_np
  closure_params['expanded_img_noisy_np'] = np.expand_dims(img_noisy_np, axis=(0))
  closure_params['ar'] = ar
  closure_params['PLOT'] = PLOT
  closure_params['show_every'] = 10
  closure_params['figsize'] = 5
  closure_params['exp_weight'] = 0.99
  closure_initialized = lambda : closure(closure_params)

  # Optimize
  optimize(OPTIMIZER, p, closure_initialized, LR, num_iter)

  #Display final results
  out_np = torch_to_np(net(net_input))
  #q = plot_image_grid([np.clip(out_np, 0, 1), ar, img_noisy_np], factor=13);
  return out_np, net.parameters()

def DIP_3D(img_noisy_np, img_np = None, PLOT = True, num_iter = 250, LR = 0.01, OPTIMIZER = 'adam', osirim = False, save_directory = "Results"): #Main function
  start_time = time.time()
  #Set parameters  
  closure_params = {}
  ar = np.array(img_noisy_np)[None, ...]
  ar = ar.astype(np.float32) / 255

  if osirim:
    if not os.path.isdir(save_directory):
      os.mkdir(save_directory)
    now = time.strftime("_date%d-%m-%y_hour%H-%M-%S", time.localtime())
    save_directory += "/Results_DIP3D_epochs%s_lr%.3e" % (num_iter ,LR) + now
    if not os.path.isdir(save_directory):
      os.mkdir(save_directory)
    closure_params['save_directory'] = save_directory
    closure_params['log'] = open(save_directory + '/DIP3D_Unfolded_Log_epochs%s_lr%.3e.txt' % (num_iter ,LR),'w+')
    closure_params['LR'] = LR
    closure_params['num_iter'] = num_iter

  # Setup the DIP
  net, net_input, closure_params['mse'] = Setup3D(img_noisy_np.shape)
  p = get_params('net', net, net_input)
  closure_params['net'] = net
  closure_params['net_input'] = net_input
  closure_params['reg_noise_std'] = 1./30. #set to 1./20. for sigma=50
  closure_params['net_input_saved'] = closure_params['net_input'].detach().clone()
  closure_params['noise'] = net_input.detach().clone()
  closure_params['out_avg'] = None
  closure_params['psrn_noisy_last'] = 0
  closure_params['i'] = 0
  closure_params['evo_mse'] = np.zeros((num_iter))
  closure_params['evo_psnr'] = np.zeros((num_iter))    
  closure_params['evo_ssim'] = np.zeros((num_iter))    
  if type(img_np) is np.ndarray:
    closure_params['img_np'] = np.expand_dims(img_np, axis=(0))
  else:
    closure_params['img_np'] = None
  closure_params['img_noisy_np'] = np.float32(img_noisy_np)
  closure_params['expanded_img_noisy_np'] = np.expand_dims(np.float32(img_noisy_np), axis=(0))
  # conversion de l'image donnee en entree en Tenseur pour Torch
  closure_params['img_noisy_torch'] = np_to_torch(closure_params['expanded_img_noisy_np']).type(dtype)
  closure_params['ar'] = ar
  closure_params['PLOT'] = PLOT
  closure_params['show_every'] = 10
  closure_params['figsize'] = 5
  closure_params['exp_weight'] = 0.99
  closure_params['osirim'] = osirim
  
  closure_initialized = lambda : closure3D(closure_params)

  # Optimize
  optimize(OPTIMIZER, p, closure_initialized, LR, num_iter, osirim = osirim)

  #Display final results
  out_np = torch_to_np(net(net_input))
  #q = plot_image_grid([np.clip(out_np, 0, 1), ar, img_noisy_np], factor=13);
  
  _ = plt.figure()
  plt.plot(closure_params['evo_mse'])
  plt.xlabel('epoch')
  plt.ylabel('MSE')
  plt.title("Mean squared error as function of epochs")
  if osirim:
    plt.savefig(save_directory + "/DIP3D_Unfolded_MsePng_epoch%s_lr%.2e.png" % (num_iter,LR))
  _ = plt.figure()
  plt.plot(closure_params['evo_psnr'])  
  plt.xlabel('epoch')
  plt.ylabel('PSNR')
  plt.title("Peak signal to noise ratio as function of epochs")
  if osirim:
    plt.savefig(save_directory + "/DIP3D_Unfolded_PsnrPng_epoch%s_lr%.2e.png" % (num_iter,LR))
  _ = plt.figure()
  plt.plot(closure_params['evo_ssim'])
  plt.xlabel('epoch')
  plt.ylabel('SSIM')
  plt.title("Structural similarity as function of epochs")
  if osirim:
    plt.savefig(save_directory + "/DIP3D_Unfolded_SsimPng_epoch%s_lr%.2e.png" % (num_iter,LR))
    closure_params['log'].write("\nDIP_3D took %s seconds\n" % (time.time() - start_time))
    closure_params['log'].close()
  else:
    print("DIP_3D took %s seconds" % round(time.time() - start_time, 2))

  return out_np, net.parameters()
