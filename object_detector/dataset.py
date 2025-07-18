# Define dataset
import torch
import numpy as np
import platform
from os import path
import os

from PIL import Image
from torchvision import transforms
from numpy import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, annotations_frame,
                 path_to_slides,
                 crop_size = (128,128),
                 pseudo_epoch_length:int = 1000,
                 transformations = None):
        
        super().__init__()
        
        if platform.system() == 'Linux':
            self.separator = '/'
        else:
            self.separator = '\\'

        self.anno_frame = annotations_frame
        self.path_to_slides = path_to_slides
        self.crop_size = crop_size
        self.pseudo_epoch_length = pseudo_epoch_length
        
        # list which holds annotations of all slides in slide_names in the format
        # slide_name, annotation, label, min_x, max_x, min_y, max_y
        
        self.slide_dict, self.annotations_list = self._initialize()
        self.sample_cord_list = self._sample_cord_list()

        # set up transformations
        self.transformations = transformations
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])


    def _initialize(self):
        # open all images and store them in self.slide_dict with their name as key value
        slide_dict = {}
        annotations_list = []
        for slide in self.anno_frame['filename'].unique():
            # open slide
            slide_dict[slide] =  Image.open(path.join(self.path_to_slides, slide)).convert('RGB')
            im_obj = Image.open(self.path_to_slides + self.separator + slide).convert('RGB')
            slide_dict[slide] = im_obj
            # setting up a list with all bounding boxes
            for idx,annotations in self.anno_frame[self.anno_frame.filename == slide][['max_x','max_y','min_x','min_y','label']].iterrows():
                annotations_list.append([slide, annotations['label'], annotations['min_x'], annotations['min_y'], annotations['max_x'], annotations['max_y']])

        return slide_dict, annotations_list


    def __getitem__(self,index):
        slide, x_cord, y_cord = self.sample_cord_list[index]
        x_cord = np.int64(x_cord)
        y_cord = np.int64(y_cord)
        # load image
        img = self.slide_dict[slide].crop((x_cord,y_cord,x_cord + self.crop_size[0],y_cord + self.crop_size[1]))
        # transform image
        #img = self.transformations(img)
        
        # load boxes for the image
        labels_boxes = self._get_boxes_and_label(slide,x_cord,y_cord)
        
        labels_boxes = [[i[1] - x_cord, i[2] - y_cord, i[3] - x_cord, i[4] - y_cord] + [i[0]] for i in labels_boxes]
        is_crowd = torch.zeros(len(labels_boxes))
        area = [torch.tensor((coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])) for coordinates in labels_boxes]
        # applay transformations
        if self.transformations != None:
            if len(labels_boxes) > 0:
                transformed = self.transformations(image = np.array(img), bboxes = labels_boxes)
                boxes = torch.tensor([line[:-1] for line in transformed['bboxes']], dtype = torch.float32)
                labels = torch.ones(boxes.shape[0], dtype = torch.int64)
                img = self.transform_to_tensor(transformed['image'])
                
            # check if there is no labeld instanceb on the image
            if len(labels_boxes) == 0:
                labels = torch.tensor([0], dtype = torch.int64)
                boxes = torch.zeros((0,4),dtype = torch.float32)
                img = self.transform_to_tensor(img)

        else:
            if len(labels_boxes) == 0:
                labels = torch.tensor([0], dtype = torch.int64)
                boxes = torch.zeros((0,4),dtype = torch.float32)
                img = self.transform_to_tensor(img)
            else:
                # now, you need to change the originale box cordinates to the cordinates of the image
                boxes = torch.tensor([line[:-1] for line in labels_boxes],dtype=torch.float32)
                labels = torch.ones(boxes.shape[0], dtype = torch.int64)
                img = self.transform_to_tensor(img)
        
        num_boxes = len(labels_boxes)
        boxes = torch.reshape(boxes, shape=(num_boxes, 4))

        
        target = {
            'image_id': torch.tensor(index).to(device),
            'boxes': boxes.to(device),
            'labels':labels.to(device),
            'iscrowd' : is_crowd.to(device),
            'area' : torch.tensor(area).to(device)
        }


        return img.to(device), target
        

    def _sample_cord_list(self):
        # select slides from which to sample an image
        slide_names = np.array(list(self.slide_dict.keys()))
        slide_indice = random.choice(np.arange(len(slide_names)), size = self.pseudo_epoch_length, replace = True)
        slides = slide_names[slide_indice]
        # select coordinates from which to load images
        # only works if all images have the same size
        width,height = self.slide_dict[slides[0]].size
        cordinates = random.randint(low = (0,0), high=(width - self.crop_size[0], height - self.crop_size[1]), size = (self.pseudo_epoch_length,2))
        return np.concatenate((slides.reshape(-1,1),cordinates), axis = -1)

    def __len__(self):
        return self.pseudo_epoch_length

    def _get_boxes_and_label(self,slide,x_cord,y_cord):
        return [line[1::] for line in self.annotations_list if line[0] == slide and line[2] > x_cord and line [3] > y_cord and line[4] < x_cord + self.crop_size[0] and line[5] < y_cord + self.crop_size[1]]
       
    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __iter__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])
            
            
        images = torch.stack(images, dim=0)

        return images, targets

    def trigger_sampling(self):
        self.sample_cord_list = self._sample_cord_list()