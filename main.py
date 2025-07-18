"""
Image Classification Training and Evaluation Script

This script implements a complete pipeline for training and evaluating 
an image classification model using PyTorch.

Dependencies:
- torch
- torchvision
- sklearn
- albumentations
- pandas
- numpy
- matplotlib
- tqdm
- torchmetrics
- PIL
"""

import os
import pickle
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
import matplotlib.pyplot as plt

# Custom imports
from Agnor.classifier.dataset import Dataset
from Agnor.classifier.model import Classifier


def load_and_split_data(annotation_path, test_size=0.2):
    """
    Load annotation data and split into train/test sets.
    
    Args:
        annotation_path (str): Path to the annotations pickle file
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (df_train, df_test) dataframes
    """
    annotation_frames = pd.read_pickle(annotation_path)
    
    # Stratified sampling
    df_train, df_test = train_test_split(
        annotation_frames, 
        test_size=test_size, 
        stratify=annotation_frames[["label"]]
    )
    
    print("Training set label distribution:")
    print(df_train['label'].value_counts())
    
    return df_train, df_test


def create_data_loaders(df_train, df_test, slides_path, image_size=(32, 32)):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        df_train (pd.DataFrame): Training annotations
        df_test (pd.DataFrame): Testing annotations
        slides_path (str): Path to slide images
        image_size (tuple): Target image size
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Note: 'tranformation' should be defined elsewhere or passed as parameter
    train_dataset = Dataset(
        path_to_slides=slides_path,
        image_size=image_size,
        annotation_frame=df_train,
        transformation_fn=transformation,  # Fix typo: was 'tranformation'
        mode="train"
    )
    
    test_dataset = Dataset(
        path_to_slides=slides_path,
        image_size=image_size,
        annotation_frame=df_test,
        mode="test"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, n_epochs=10, lr=0.001):
    """
    Train the classification model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Testing data loader
        n_epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    print(f"Training on device: {device}")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        acc = 0
        count = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
        
        acc /= count
        print(f"Epoch {epoch+1}: Model accuracy {acc*100:.2f}%")
    
    return model


def evaluate_model(model, test_loader, num_classes=12):
    """
    Comprehensive model evaluation with confusion matrix and metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Testing data loader
        num_classes (int): Number of classification classes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = list(range(num_classes))
    
    # Initialize confusion matrix
    CM = np.zeros((num_classes, num_classes))
    model.eval()
    
    # Collect predictions
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            
            CM += confusion_matrix(
                labels.cpu().numpy(), 
                preds.cpu().numpy(),
                labels=classes
            )
    
    # Calculate metrics (for binary classification)
    if num_classes == 2:
        tn, fp, fn, tp = CM.ravel()
        
        accuracy = np.sum(np.diag(CM)) / np.sum(CM)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision)
        
        print(f'\nTestset Accuracy: {100 * accuracy:.2f}%')
        print('\nConfusion Matrix:')
        print(CM)
        print(f'- Sensitivity: {sensitivity * 100:.2f}%')
        print(f'- Specificity: {specificity * 100:.2f}%')
        print(f'- Precision: {precision * 100:.2f}%')
        print(f'- NPV: {npv * 100:.2f}%')
        print(f'- F1 Score: {f1_score * 100:.2f}%')
    
    else:
        # Multi-class metrics
        accuracy = np.sum(np.diag(CM)) / np.sum(CM)
        print(f'\nTestset Accuracy: {100 * accuracy:.2f}%')
        print('\nConfusion Matrix:')
        print(CM)
    
    # Per-class accuracy
    print('\nPer-class Accuracy:')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    for classname, correct_count in correct_pred.items():
        total_count = total_pred[classname]
        if total_count > 0:
            accuracy = 100 * float(correct_count) / total_count
            print(f'Accuracy for class {classname}: {accuracy:.1f}%')


def main():
    """
    Main execution function.
    """
    # Configuration
    ANNOTATION_PATH = "annotations.p"
    SLIDES_PATH = "/home/ESPL_001/user/Downloads/fourth_milestore/crops"
    IMAGE_SIZE = (32, 32)
    N_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 12
    
    # Load and split data
    df_train, df_test = load_and_split_data(ANNOTATION_PATH)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        df_train, df_test, SLIDES_PATH, IMAGE_SIZE
    )
    
    # Initialize and train model
    model = Classifier()
    trained_model = train_model(
        model, train_loader, test_loader, N_EPOCHS, LEARNING_RATE
    )
    
    # Evaluate model
    evaluate_model(trained_model, test_loader, NUM_CLASSES)


if __name__ == "__main__":
    main()