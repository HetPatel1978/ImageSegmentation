import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from tqdm import tqdm

class PRImADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory containing the PRImA dataset
            transform: Optional transforms to be applied
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Look for images and their corresponding XML files
        self.image_files = []
        self.xml_files = []
        
        # Walk through the directory to find image-XML pairs
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.png', '.jpg')):
                    img_path = Path(root) / file
                    xml_path = img_path.with_suffix('.xml')
                    
                    if xml_path.exists():
                        self.image_files.append(img_path)
                        self.xml_files.append(xml_path)
        
        print(f"Found {len(self.image_files)} image-XML pairs")

    def __len__(self):
        return len(self.image_files)
    
    def parse_page_xml(self, xml_path, image_size):
        """Parse PAGE XML format and create segmentation mask."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create empty mask with image dimensions
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # Convert (w,h) to (h,w)
        
        # Handle different XML namespace possibilities
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {}  # No namespace
        ]
        
        for ns in namespaces:
            # Try to find TextRegion elements with this namespace
            regions = root.findall('.//TextRegion', ns) or root.findall('.//page:TextRegion', ns)
            if regions:
                for region in regions:
                    # Find coordinates
                    coords = region.find('.//Coords', ns) or region.find('.//page:Coords', ns)
                    if coords is not None:
                        points_str = coords.get('points')
                        if points_str:
                            # Convert points string to numpy array
                            points = [tuple(map(int, p.split(','))) for p in points_str.split()]
                            points = np.array(points)
                            
                            # Fill polygon in mask
                            cv2.fillPoly(mask, [points], 1)
                break  # Found the correct namespace, no need to try others
        
        return mask

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Create mask from XML
        xml_path = self.xml_files[idx]
        mask = self.parse_page_xml(xml_path, image.size)
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Training function with progress tracking."""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/(batch_idx+1)})
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = PRImADataset(
        root_dir=r'C:\Users\Het\Desktop\Het\PRImA Layout Analysis Dataset',
        transform=transform
    )
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save final model
    torch.save(model.state_dict(), 'prima_segmentation_final.pth')
    print("Training completed!")

def predict(model, image_path, device):
    """Make prediction on a single image."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_mask = model(image_tensor)
    
    # Convert prediction to binary mask
    pred_mask = (pred_mask > 0.5).float()
    return pred_mask.squeeze().cpu().numpy()

if __name__ == '__main__':
    main()