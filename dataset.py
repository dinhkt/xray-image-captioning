import cv2
import numpy as np
import torch
import pandas as pd
import os
from PIL import Image

class Dataset():
  #here we will get the images converted to vector form and the corresponding captions
  def __init__(self,df_path,transform,vocab): 
    """
    df  = dataframe containing image_1,image_2 and impression
    """
    df = pd.read_pickle(df_path)
    self.image1 = list(df.image_1)
    self.image2 = list(df.image_2)
    self.caption = list(df.impression)
    self.transform = transform
    self.vocab=vocab

  def __getitem__(self,i):
    #gets the datapoint at i th index, we will extract the feature vectors of images after resizing the image  and apply augmentation
    image_folder="dataset/images/"
    image1 = self.transform(Image.open(os.path.join(image_folder,self.image1[i]))) 
    image2 = self.transform(Image.open(os.path.join(image_folder,self.image2[i])))

    caption = []
    tokens=self.caption[i].split()
    caption.append(self.vocab('<start>'))
    caption.extend([self.vocab(token) for token in tokens])
    caption.append(self.vocab('<end>'))
    caption = torch.LongTensor(caption)

    return image1,image2,caption,caption.shape[0]

    
  def __len__(self):
    return len(self.image1)
