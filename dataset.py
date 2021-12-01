import cv2
import numpy as np
import torch
import pandas as pd

class Dataset():
  #here we will get the images converted to vector form and the corresponding captions
  def __init__(self,df_path,input_size,vocab, augmentation = True): 
    """
    df  = dataframe containing image_1,image_2 and impression
    """
    df = pd.read_pickle(df_path)
    self.image1 = df.image_1
    self.image2 = df.image_2
    self.caption = df.impression
    self.input_size = input_size #tuple ex: (512,512)
    self.augmentation = augmentation
    self.vocab=vocab

  def __getitem__(self,i):
    #gets the datapoint at i th index, we will extract the feature vectors of images after resizing the image  and apply augmentation
    image1 = cv2.imread(self.image1[i],cv2.IMREAD_UNCHANGED)/255 
    image2 = cv2.imread(self.image2[i],cv2.IMREAD_UNCHANGED)/255 #here there are 3 channels
    image1 = cv2.resize(image1,self.input_size,interpolation = cv2.INTER_NEAREST)
    image2 = cv2.resize(image2,self.input_size,interpolation = cv2.INTER_NEAREST)
    if image1.any()==None:
      print("%i , %s image sent null value"%(i,self.image1[i]))
    if image2.any()==None:
      print("%i , %s image sent null value"%(i,self.image2[i]))

    ### How should we merge image1 and image2 to image?
    image=image1

    caption = []
    caption.append(self.vocab('<start>'))
    caption.extend([self.vocab(token) for token in self.caption[i]])
    caption.append(self.vocab('<end>'))
    caption = torch.Tensor(caption)

    if self.augmentation: #we will not apply augmentation that crops the image 
          a = np.random.uniform()
          if a<0.333:
              image1 = self.aug1.augment_image(image1)
              image2 = self.aug1.augment_image(image2)
          elif a<0.667:
              image1 = self.aug2.augment_image(image1)
              image2 = self.aug2.augment_image(image2)

    return image,caption

    
  def __len__(self):
    return len(self.image1)
