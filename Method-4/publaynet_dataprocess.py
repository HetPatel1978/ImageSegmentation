import os
import random
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image

# Define dataset paths
BASE_PATH = r"D:/5101_D/Newspaper/publaynet"  # Change this to the path of your dataset
IMAGE_DIRS = {
    "train": os.path.join(BASE_PATH, "train"),
    "val": os.path.join(BASE_PATH, "val"),
    "test": os.path.join(BASE_PATH, "test")  # Test has no annotations
}
ANNOTATION_FILES = {
    "train": os.path.join(BASE_PATH, "train.json"),
    "val": os.path.join(BASE_PATH, "val.json")
}

# Select which dataset split to visualize
SPLIT = "train"  # Change to 'val' if needed
IMAGE_DIR = IMAGE_DIRS[SPLIT]
ANNOTATION_FILE = ANNOTATION_FILES[SPLIT]

# Load COCO Annotations
coco = COCO(ANNOTATION_FILE)

# Get all image IDs
img_ids = coco.getImgIds()
random.shuffle(img_ids)  # Shuffle images for better visualization

# Function to visualize images with annotations
def visualize_image(img_id):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGE_DIR, img_info['file_name'])

    # Load image
    image = Image.open(img_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Plot bounding boxes
    ax = plt.gca()
    for ann in anns:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']

        # Draw rectangle
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Label the bounding box
        plt.text(bbox[0], bbox[1] - 5, category_name, color='red', fontsize=12, weight='bold')

    plt.show()

# Display a few images with their annotations
for i in range(3):  # Change this number to see more samples
    visualize_image(img_ids[i])
