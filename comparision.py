# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:40:43 2020
"""
#comparision of models
from PIL import Image
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import os, shutil, time
class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(pretrained = True) 
    for param in resnet.parameters():
        param.requires_grad = False
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output

class FourLayerUpsampling(nn.Module):
  def __init__(self, input_size=128):
    super(FourLayerUpsampling, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(pretrained = True) 
    for param in resnet.parameters():
        param.requires_grad = False
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output
    
use_gpu = torch.cuda.is_available() 
model1 = FourLayerUpsampling()
model2 = ColorizationNet()
model1.load_state_dict(torch.load("resnet_models/train_with_mixed_dataset/FourLayerUpsampling/model-epoch-20-losses-0.003.pth"))
model2.load_state_dict(torch.load("resnet_models/train_with_mixed_dataset/ColorizationNet/model-epoch-17-losses-0.003.pth"))
#model.cuda()
class GrayscaleImageFolder(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab, target

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

#testing
for filename in glob.iglob( 'images/test/class/*.jpg', recursive=True):
    print(filename)
    im = Image.open(filename)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save(filename , 'JPEG', quality=90)
    
    
def test(val_loader, model, save_images, model_name ):
  model.eval()

  # Prepare value counters and timers
  already_saved_images = False
  for i, (input_gray, input_ab, target) in enumerate(val_loader):

    # Use GPU
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(len(output_ab)): # save at most 5 images
        save_path = {'colorized': 'checkpoints/comparisonResults2/'}
        save_name = 'img-{}-model_name-{}.jpg'.format(i * val_loader.batch_size + j, model_name)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

#testing
for filename in glob.iglob( 'images/loss_check/class/*.jpg', recursive=True):
    print(filename)
    im = Image.open(filename)
    imResize = im.resize((256,256), Image.ANTIALIAS)
    imResize.save(filename , 'JPEG', quality=90)

val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder('images/test' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=100, shuffle=False)
with torch.no_grad():
    model1.cuda()
    test(val_loader, model1, True, "FourLayerUpsampling")
    model2.cuda()
    test(val_loader, model2, True, "ColorizationNet")
    
import cv2
img1 = cv2.imread("images/test/class/Anitkabir-fotografi.jpg") #original photo
img1 = cv2.resize(img1, (224,224), interpolation = cv2.INTER_AREA)
img2 = cv2.imread("checkpoints/comparisonResults2/img-9-model_name-ColorizationNet.jpg")
img3 = cv2.imread("checkpoints/comparisonResults2/img-9-model_name-FourLayerUpsampling.jpg")
print(type(img2))
err1 = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
err1 /= float(img2.shape[0] * img2.shape[1])
err2 = np.sum((img1.astype("float") - img3.astype("float")) ** 2)
err2 /= float(img2.shape[0] * img2.shape[1]*3)

print("MSE of ColorizationNet model output: " + str(err1))
print("MSE of FourLayerUpsampling model output: " + str(err2))

