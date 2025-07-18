"""
Object Detection Training Script for AgNOR Cell Detection
=========================================================

This script trains a Faster R-CNN model with MobileNet-V3 backbone for cell detection.
The model is fine-tuned on a custom dataset with object detection annotations.

Requirements:
- PyTorch
- torchvision
- albumentations
- pandas
- numpy
- matplotlib
- Custom Dataset class and utility functions

Author: [Your Name]
Date: [Date]
"""

import os
import math
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import albumentations as A

# Custom imports (ensure these modules are available)
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from dataset_objectdetector import Dataset  # Assuming you have a custom Dataset class


class ObjectDetectionTrainer:
    """
    A class for training object detection models on custom datasets.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 device: str = None,
                 learning_rate: float = 0.0001,
                 batch_size: int = 5):
        """
        Initialize the trainer.
        
        Args:
            num_classes: Number of classes including background
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        
        print(f"Using device: {self.device}")
    
    def create_model(self) -> None:
        """
        Create and configure the Faster R-CNN model.
        """
        # Load pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        
        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.learning_rate, amsgrad=True)
        
        print(f"Model created with {self.num_classes} classes")
    
    def prepare_data(self, 
                     annotation_frame_path: str, 
                     images_path: str, 
                     train_split: float = 0.8,
                     crop_size: Tuple[int, int] = (128, 128)) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and testing data loaders.
        
        Args:
            annotation_frame_path: Path to annotation pickle file
            images_path: Path to images directory
            train_split: Fraction of data to use for training
            crop_size: Size for image cropping
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Load annotations
        annotation_frame = pd.read_pickle(annotation_frame_path)
        
        # Define augmentations
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        
        # Split data
        total_unique_images = annotation_frame['filename'].unique()
        train_images_count = math.floor(train_split * len(total_unique_images))
        
        train_data_images = total_unique_images[:train_images_count]
        test_data_images = total_unique_images[train_images_count:]
        
        # Create dataframes
        training_frame = annotation_frame[annotation_frame['filename'].isin(train_data_images)]
        testing_frame = annotation_frame[annotation_frame['filename'].isin(test_data_images)]
        
        # Create datasets
        train_dataset = Dataset(training_frame, images_path, 
                              crop_size=crop_size, transformations=transform)
        test_dataset = Dataset(testing_frame, images_path, crop_size=crop_size)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=0, collate_fn=Dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=0, collate_fn=Dataset.collate_fn)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train_one_epoch(self, data_loader: DataLoader, epoch: int, 
                       print_freq: int = 10, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Train the model for one epoch.
        
        Args:
            data_loader: Training data loader
            epoch: Current epoch number
            print_freq: Frequency of printing training stats
            scaler: GradScaler for mixed precision training
            
        Returns:
            Metric logger with training statistics
        """
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        # Learning rate warmup
        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                losses.backward()
                self.optimizer.step()
                
            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        return metric_logger
    
    @torch.inference_mode()
    def evaluate(self, data_loader: DataLoader):
        """
        Evaluate the model on the test dataset.
        
        Args:
            data_loader: Test data loader
            
        Returns:
            COCO evaluator with evaluation results
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self._get_iou_types()
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(self.device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            model_time = time.time()
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Gather stats and evaluate
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        torch.set_num_threads(n_threads)
        return coco_evaluator
    
    def _get_iou_types(self) -> List[str]:
        """Get IoU types for evaluation."""
        model_without_ddp = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
            
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
              num_epochs: int = 10, save_path: str = None):
        """
        Train the model for specified number of epochs.
        
        Args:
            train_loader: Training data loader
            test_loader: Testing data loader
            num_epochs: Number of epochs to train
            save_path: Path to save the trained model
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_stats = self.train_one_epoch(train_loader, epoch, print_freq=10)
            
            # Evaluate on test set
            eval_stats = self.evaluate(test_loader)
            
            print(f"Training stats: {train_stats}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.model, path)
    
    def load_model(self, path: str):
        """Load a pre-trained model."""
        self.model = torch.load(path, map_location=self.device)
        self.model.eval()
    
    def visualize_predictions(self, test_loader: DataLoader, num_samples: int = 4):
        """
        Visualize model predictions on test samples.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to visualize
        """
        self.model.eval()
        
        plt.rcParams["savefig.bbox"] = 'tight'
        
        def show_images(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 5))
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.tight_layout()
            plt.show()
        
        with torch.no_grad():
            for i in range(num_samples):
                images, targets = next(iter(test_loader))
                images = list(image.to(self.device) for image in images)
                
                predictions = self.model(images)
                
                for index, img in enumerate(images):
                    # Move image back to CPU for visualization
                    img_cpu = img.cpu()
                    boxes = predictions[index]['boxes'].cpu()
                    
                    # Draw bounding boxes
                    result = draw_bounding_boxes(
                        F.convert_image_dtype(img_cpu, dtype=torch.uint8), 
                        boxes, 
                        width=2,
                        colors="red"
                    )
                    show_images(result)
                    
                    if index == 0:  # Show only first image from batch
                        break


def main():
    """
    Main function to run the training pipeline.
    """
    # Configuration
    config = {
        'num_classes': 2,  # 1 class (cell) + background
        'learning_rate': 0.0001,
        'batch_size': 5,
        'num_epochs': 10,
        'train_split': 0.8,
        'crop_size': (128, 128),
        'annotation_frame_path': "/path/to/your/annotation_frame.p",
        'images_path': "/path/to/your/images/",
        'model_save_path': "/path/to/save/model"
    }
    
    # Initialize trainer
    trainer = ObjectDetectionTrainer(
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size']
    )
    
    # Create model
    trainer.create_model()
    
    # Prepare data
    train_loader, test_loader = trainer.prepare_data(
        annotation_frame_path=config['annotation_frame_path'],
        images_path=config['images_path'],
        train_split=config['train_split'],
        crop_size=config['crop_size']
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['num_epochs'],
        save_path=config['model_save_path']
    )
    
    # Visualize some predictions
    print("\nVisualizing predictions...")
    trainer.visualize_predictions(test_loader, num_samples=2)


if __name__ == "__main__":
    main()