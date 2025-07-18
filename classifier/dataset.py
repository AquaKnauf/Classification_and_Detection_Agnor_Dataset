import torch
import numpy as np
import os
import collections

from PIL import Image
from torchvision import transforms
from numpy import random
import numpy as np
import random
from typing import Tuple, Callable
import pandas as pd

# Set up augmentation class
import torchvision.transforms.functional as TF
import random

class MyTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [-30, -15, 0, 15, 30]
        self.brightness_levels = [0.9, 1.1, 1.2]

    def __call__(self, x):
        angle = random.choice(self.angles)
        x = TF.rotate(x, angle)
        x = TF.adjust_brightness(x, random.choice(self.brightness_levels))
        return x

tranformation = MyTransform()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_frame: pd.DataFrame, path_to_slides: str, image_size: Tuple[int, int],
                transformation_fn: Callable = None, pseudo_epoch_length=1000, mode=""):
        super().__init__()
        self._image_size = image_size

        self._resize = transforms.Resize(size=image_size, antialias=True)
        
        self.annotations_frame = annotation_frame
        self.mode = mode
        if not os.path.isdir(path_to_slides):
            raise IOError("Image path is not set correctly")
            
        self.path_to_slides = path_to_slides
        self.pseudo_epoch_length = pseudo_epoch_length
        
        self.transformation = transformation_fn
        self.transform_to_tensor = transforms.ToTensor()
        self.slide_list, self.annotation_list = self._initialize()
        

    def __len__(self):
        return len(self.annotation_list)

        
    def _initialize(self):
        """
        Initilize the internal dictionary
        """

        slide_list = []
        annotation_list = []
        for _, row in self.annotations_frame.iterrows():
            image = row['filename']
            label = row['label']
            open_image = Image.open(os.path.join(self.path_to_slides, image)).convert('RGB')
            open_image = self.transform_to_tensor(open_image)
            if self.mode == "train":
                if label == 1:
                    # Undersample class 1 with a probability of 25%
                    if random.choice([1, 2, 3, 4]) == 1:
                        continue
                    slide_list.append(open_image)
                    annotation_list.append(label)
                # Oversample classes with label 3 using transformations
                elif label == 3:
                        
                    slide_list.append(open_image)
                    annotation_list.append(label)

                    slide_list.append(open_image)
                    annotation_list.append(label)


                elif label in [4, 5]:
                    for i in range(3):
                        slide_list.append(open_image)
                        annotation_list.append(label)

                elif label in [0, 2]:

                    slide_list.append(open_image)
                    annotation_list.append(label)
                    
    
                elif label in  [9 , 10, 11]:
                    for i in range(20):
                        slide_list.append(open_image)
                        annotation_list.append(label)
                else:
                    for i in range(15):
                        slide_list.append(open_image)
                        annotation_list.append(label)
            else:
                slide_list.append(open_image)
                annotation_list.append(label)
                    
        
        print(f"The counts after adjustment: {collections.Counter(annotation_list)}")
        print(f"The number of images in the dataset : {len(annotation_list)}")


        return slide_list, annotation_list
    
    
    def __getitem__(self,index):
        """
        Load an image
        """

        img = self.slide_list[index]
        label = self.annotation_list[index]

        # Perform necessary conversions
        img = self._resize.forward(img)
        img = self.transformation(img) if self.transformation else img
        
           
        return img.to(torch.device("cuda")), torch.tensor(label, device=torch.device("cuda"), dtype=torch.int64)