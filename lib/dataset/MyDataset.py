from logging import NullHandler
import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import datasets, transforms
import numpy as np

class MyDatasetClass(Dataset):  
  def __init__(self, main_dir, transform):
         
        # Set the loading directory
        self.main_dir = main_dir
        self.transform = transform
         
        # List all images in folder and count them
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

  def __len__(self):
    # Return the previously computed number of images
    return len(self.total_imgs)

  def __getitem__(self, idx):
    
    img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    # Use PIL for image loading
    image = Image.open(img_loc).convert("RGB")
  
    # Apply the transformations
    tensor_image = self.transform(image)

    return tensor_image