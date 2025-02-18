# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms

# class NewspaperDataset(Dataset):
#     def __init__(self, dataset_dir, transform=None):
#         self.dataset_dir = dataset_dir
#         self.transform = transform

#         # Supported image formats
#         self.valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

#         # Get all valid images, ignoring masks
#         self.images = [f for f in os.listdir(dataset_dir) if "_m" not in f and f.lower().endswith(self.valid_exts)]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.dataset_dir, img_name)

#         # Extract base filename without extension
#         base_name, _ = os.path.splitext(img_name)

#         # Find the mask with any supported extension
#         mask_path = None
#         for ext in self.valid_exts:  # Possible mask formats
#             possible_mask = os.path.join(self.dataset_dir, f"{base_name}_m{ext}")
#             if os.path.exists(possible_mask):
#                 mask_path = possible_mask
#                 break

#         if mask_path is None:
#             raise FileNotFoundError(f"Mask not found for {img_name}")

#         # Load image and mask in grayscale mode
#         image = Image.open(img_path).convert("L")  
#         mask = Image.open(mask_path).convert("L")  

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  
#     transforms.ToTensor()  
# ])

# # Set dataset path
# dataset_path = r"C:/Users/Het/Desktop/Het/Method-2/dataset_segmentation"
# train_dataset = NewspaperDataset(dataset_path, transform)





# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# import cv2
# import numpy as np
# import os
# from PIL import Image

# # U-Net Model
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# # Dataset Loader
# class NewspaperDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir)
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.images[idx])
#         mask_path = os.path.join(self.mask_dir, self.images[idx])
        
#         image = Image.open(img_path).convert("L")
#         mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale
        
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
        
#         return image, mask

# # Transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Update dataset paths
# image_dir = "path/to/UCI_dataset/images"
# mask_dir = "path/to/UCI_dataset/masks"

# # DataLoader
# train_dataset = NewspaperDataset(image_dir, mask_dir, transform)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # Model Training
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()

# # Training Loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for images, masks in train_loader:
#         images, masks = images.to(device), masks.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# # Save Model
# torch.save(model.state_dict(), "unet_newspaper.pth")












#U-net Model training 


import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewspaperDataset(Dataset):
    """Dataset class for newspaper image segmentation."""
    
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    def __init__(self, 
                 dataset_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 mask_suffix: str = "_m"):
        """
        Initialize the dataset.
        
        Args:
            dataset_dir: Directory containing images and masks
            transform: Optional transforms to be applied
            mask_suffix: Suffix used to identify mask files
        """
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.mask_suffix = mask_suffix
        
        # Get all valid images, excluding masks
        self.images = [
            f for f in os.listdir(dataset_dir) 
            if self.mask_suffix not in f and f.lower().endswith(self.VALID_EXTENSIONS)
        ]
        
        if not self.images:
            raise ValueError(f"No valid images found in {dataset_dir}")
        
        logger.info(f"Found {len(self.images)} images in dataset")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = self.dataset_dir / img_name
        
        # Find corresponding mask
        base_name = Path(img_name).stem
        mask_path = None
        
        for ext in self.VALID_EXTENSIONS:
            possible_mask = self.dataset_dir / f"{base_name}{self.mask_suffix}{ext}"
            if possible_mask.exists():
                mask_path = possible_mask
                break
                
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_name}")
            
        try:
            # Load images in grayscale mode
            image = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                
            return image, mask
            
        except Exception as e:
            logger.error(f"Error loading {img_name}: {str(e)}")
            raise

class UNet(nn.Module):
    """Simple U-Net architecture for image segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self._conv_block(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            self._conv_block(32, out_channels),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class NewspaperSegmentation:
    """Main class for training and inference of the newspaper segmentation model."""
    
    def __init__(self, 
                 dataset_dir: str,
                 batch_size: int = 8,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize the segmentation pipeline.
        
        Args:
            dataset_dir: Directory containing the dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Initialize dataset and model
        self.dataset = NewspaperDataset(dataset_dir, self.transform)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if self.device == 'cuda' else 0
        )
        
        self.model = UNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        logger.info(f"Model initialized on {self.device}")

    def train(self, num_epochs: int, save_path: str = "model_weights.pth"):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save the model weights
        """
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (images, masks) in enumerate(self.dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} "
                              f"Batch {batch_idx}/{len(self.dataloader)} "
                              f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} Average Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    """Main function to run the training pipeline."""
    # Configuration
    CONFIG = {
        'dataset_dir': r"C:/Users/Het/Desktop/Het/Method-2/dataset_segmentation",
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'save_path': 'newspaper_unet.pth'
    }
    
    try:
        # Initialize and train the model
        segmentation = NewspaperSegmentation(
            dataset_dir=CONFIG['dataset_dir'],
            batch_size=CONFIG['batch_size'],
            learning_rate=CONFIG['learning_rate']
        )
        
        segmentation.train(
            num_epochs=CONFIG['num_epochs'],
            save_path=CONFIG['save_path']
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()