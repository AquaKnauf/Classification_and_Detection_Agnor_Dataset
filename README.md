# Object Detection and Classification Training for AgNOR Cell Detection

This repository contains code for training a Faster R-CNN model with MobileNet-V3 backbone for cell detection in AgNOR (Argyrophilic Nucleolar Organizer Regions) images.

## Overview

The project implements a complete pipeline for training an object detection model to identify and localize cells in microscopy images. The model uses transfer learning from a pre-trained Faster R-CNN model and fine-tunes it on a custom dataset.

## Features

- **Model Architecture**: Faster R-CNN with MobileNet-V3 Large FPN backbone
- **Data Augmentation**: Horizontal flipping using Albumentations
- **Training Pipeline**: Complete training and evaluation pipeline
- **Visualization**: Tools for visualizing model predictions
- **Modular Design**: Object-oriented implementation for easy customization

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the complete list of dependencies.

## Project Structure

```
.
├── object_detector    # Main training script
    ├──dataset.py
├──classifier
    ├── dataset.py             # Custom dataset class (to be implemented)
    ├── model.py
├── main.py                    # Classifier Model training
├── model_objectdetector.py    # Object Detection training
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### 1. Prepare Your Data

Ensure your data is organized as follows:
- Images in a directory (e.g., `images/`)
- Annotations in a pandas DataFrame saved as a pickle file (`annotation_frame.p`)

The annotation DataFrame should contain at least:
- `filename`: Image filename
- Bounding box coordinates in Pascal VOC format

### 2. Configure the Training

Edit the configuration dictionary in the `main()` function:

```python
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
```

### 3. Run Training

```bash
python main.py
```

OR

```bash
python model_objectdetector.py
```

### 4. Using the ObjectDetectionTrainer Class

You can also use the trainer class directly:

```python
from train_object_detector import ObjectDetectionTrainer

# Initialize trainer
trainer = ObjectDetectionTrainer(
    num_classes=2,
    learning_rate=0.0001,
    batch_size=5
)

# Create model
trainer.create_model()

# Prepare data
train_loader, test_loader = trainer.prepare_data(
    annotation_frame_path="path/to/annotations.p",
    images_path="path/to/images/",
    train_split=0.8
)

# Train
trainer.train(train_loader, test_loader, num_epochs=10)

# Visualize predictions
trainer.visualize_predictions(test_loader)
```

## Model Architecture

The model uses a Faster R-CNN architecture with:
- **Backbone**: MobileNet-V3 Large with Feature Pyramid Network (FPN)
- **Region Proposal Network (RPN)**: For generating object proposals
- **ROI Head**: For classification and bounding box regression

## Training Details

- **Optimizer**: Adam with AMSGrad
- **Learning Rate**: 0.0001 with warmup for the first epoch
- **Loss Function**: Combined classification and regression losses from Faster R-CNN
- **Evaluation**: COCO-style evaluation metrics

## Data Augmentation

The training pipeline includes:
- Random horizontal flipping (50% probability)
- Image normalization using ImageNet statistics

## Evaluation

The model is evaluated using COCO-style metrics:
- Average Precision (AP) at different IoU thresholds
- Average Recall (AR) at different IoU thresholds

## Customization

### Adding New Augmentations

Modify the `prepare_data` method to include additional augmentations:

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.5),
    # Add more augmentations as needed
], bbox_params=A.BboxParams(format='pascal_voc'))
```

### Changing the Model Architecture

Replace the model initialization in `create_model()`:

```python
# For ResNet-50 backbone
self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
```

### Custom Dataset Class

Implement your own dataset class by inheriting from `torch.utils.data.Dataset`:

```python
class CustomDataset(Dataset):
    def __init__(self, annotation_frame, images_path, crop_size=None, transformations=None):
        # Implementation here
        pass
        
    def __getitem__(self, idx):
        # Return image and target dictionary
        pass
        
    def __len__(self):
        # Return dataset size
        pass
        
    @staticmethod
    def collate_fn(batch):
        # Custom collate function for DataLoader
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image crop size
2. **Module Not Found**: Ensure all dependencies are installed
3. **Data Loading Errors**: Check file paths and data format

### Performance Tips

- Use mixed precision training with `torch.cuda.amp.GradScaler()`
- Increase number of workers in DataLoader for faster data loading
- Use larger batch sizes if GPU memory allows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the detection models
- Torchvision for pre-trained weights
- COCO evaluation utilities

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{agnor_object_detection,
  title={Object Detection Training for AgNOR Cell Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/your-repo}
}
```
