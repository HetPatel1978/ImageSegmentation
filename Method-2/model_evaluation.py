# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import logging
# from pathlib import Path
# from typing import Optional
# from try1 import UNet, NewspaperDataset  # Ensure the model is imported correctly

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelEvaluator:
#     """Class to evaluate the trained U-Net model on test data."""
    
#     def __init__(self, model_path: str, dataset_dir: str, device: Optional[str] = None):
#         """
#         Initialize the evaluator.
        
#         Args:
#             model_path: Path to the trained model file.
#             dataset_dir: Directory containing the test dataset.
#             device: Device to run the model on ('cuda' or 'cpu').
#         """
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load model
#         self.model = UNet().to(self.device)
#         self._load_model(model_path)
        
#         # Define transformations for evaluation
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor()
#         ])
        
#         # Load dataset
#         self.dataset = NewspaperDataset(dataset_dir, transform=self.transform)
#         self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

#     def _load_model(self, model_path: str):
#         """Load the trained model from the given path."""
#         try:
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#             self.model.eval()
#             logger.info(f"Model successfully loaded from {model_path}")
#         except Exception as e:
#             logger.error("Error loading model. Ensure the model architecture matches the saved state_dict.")
#             raise e

#     @staticmethod
#     def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
#         """Compute Dice Coefficient."""
#         pred = (pred > 0.5).float()
#         target = (target > 0.5).float()
#         intersection = (pred * target).sum()
#         return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
#     @staticmethod
#     def iou(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
#         """Compute Intersection over Union (IoU)."""
#         pred = (pred > 0.5).float()
#         target = (target > 0.5).float()
#         intersection = (pred * target).sum()
#         union = pred.sum() + target.sum() - intersection
#         return (intersection + smooth) / (union + smooth)

#     def evaluate(self):
#         """Evaluate the model on test data."""
#         total_dice, total_iou = 0.0, 0.0
#         num_samples = len(self.dataloader)
        
#         logger.info("Starting evaluation...")
#         with torch.no_grad():
#             for idx, (image, mask) in enumerate(self.dataloader):
#                 image, mask = image.to(self.device), mask.to(self.device)
#                 output = self.model(image)
                
#                 dice = self.dice_coefficient(output, mask).item()
#                 iou = self.iou(output, mask).item()
#                 total_dice += dice
#                 total_iou += iou
                
#                 logger.info(f"Sample {idx+1}: Dice = {dice:.4f}, IoU = {iou:.4f}")
        
#         avg_dice = total_dice / num_samples
#         avg_iou = total_iou / num_samples
#         logger.info(f"Final Evaluation: Avg Dice = {avg_dice:.4f}, Avg IoU = {avg_iou:.4f}")
#         return avg_dice, avg_iou

# if __name__ == "__main__":
#     CONFIG = {
#         'model_path': r'C:/Users/Het/Desktop/Het/Method-2/newspaper_unet.pth',  # Path to saved model
#         'dataset_dir': r'C:/Users/Het/Desktop/Het/Method-1/PRImA Layout Analysis Dataset/Images'  # Path to test dataset
#     }
    
#     try:
#         evaluator = ModelEvaluator(CONFIG['model_path'], CONFIG['dataset_dir'])
#         evaluator.evaluate()
#     except Exception as e:
#         logger.error(f"Evaluation failed: {str(e)}")









import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from try1 import UNet, NewspaperDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewspaperDataset(Dataset):
    """Dataset class for newspaper image segmentation."""
    
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    def __init__(self, dataset_dir: str, transform: Optional[transforms.Compose] = None, mask_suffix: str = "_m"):
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
            image = Image.open(img_path)
            if hasattr(image, "n_frames") and image.n_frames > 1:
                image.seek(0)
            image = image.convert("L")
            
            mask = Image.open(mask_path)
            if hasattr(mask, "n_frames") and mask.n_frames > 1:
                mask.seek(0)
            mask = mask.convert("L")
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                
            return image, mask
            
        except Exception as e:
            logger.error(f"Error loading {img_name}: {str(e)}")
            raise

class ModelEvaluator:
    """Class to evaluate the trained U-Net model on test data."""
    
    def __init__(self, model_path: str, dataset_dir: str, device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model file.
            dataset_dir: Directory containing the test dataset.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = UNet().to(self.device)
        self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.dataset = NewspaperDataset(dataset_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def _load_model(self, model_path: str):
        """Load the trained model from the given path."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model successfully loaded from {model_path}")
        except Exception as e:
            logger.error("Error loading model. Ensure the model architecture matches the saved state_dict.")
            raise e

    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    @staticmethod
    def iou(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def evaluate(self):
        total_dice, total_iou = 0.0, 0.0
        num_samples = len(self.dataloader)
        
        logger.info("Starting evaluation...")
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.dataloader):
                image, mask = image.to(self.device), mask.to(self.device)
                output = self.model(image)
                
                dice = self.dice_coefficient(output, mask).item()
                iou = self.iou(output, mask).item()
                total_dice += dice
                total_iou += iou
                
                logger.info(f"Sample {idx+1}: Dice = {dice:.4f}, IoU = {iou:.4f}")
        
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        logger.info(f"Final Evaluation: Avg Dice = {avg_dice:.4f}, Avg IoU = {avg_iou:.4f}")
        return avg_dice, avg_iou

if __name__ == "__main__":
    CONFIG = {
        'model_path': r'C:/Users/Het/Desktop/Het/Method-2/newspaper_unet.pth',
        'dataset_dir': r'C:/Users/Het/Desktop/Het/Method-2/images'
    }
    
    try:
        evaluator = ModelEvaluator(CONFIG['model_path'], CONFIG['dataset_dir'])
        evaluator.evaluate()
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")