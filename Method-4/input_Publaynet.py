import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ✅ Load Trained Model 1
MODEL_DIR = r"D:\5101_D\Newspaper\ImageSegmentation\Method-4\output_publaynet"  # Adjust path to Model 1
MODEL_PATH = os.path.join(MODEL_DIR, "model_final.pth")  # Trained model path

# ✅ Path to Input Newspaper Image
INPUT_IMAGE_PATH = r"D:\5101_D\Newspaper\ImageSegmentation\imageprocessor.jpg"  # Change to your test image path

# ✅ Set Model Configuration
cfg = get_cfg()
cfg.merge_from_file(r"D:\5101_D\Newspaper\ImageSegmentation\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml")  # Use the same config as training
cfg.MODEL.WEIGHTS = MODEL_PATH  # Load Model 1's trained weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.DEVICE = "cpu"  # Use "cuda" if running on GPU
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Ensure the correct number of classes

# ✅ Load Model for Inference
predictor = DefaultPredictor(cfg)

# ✅ Read Input Image
image = cv2.imread(INPUT_IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# ✅ Perform Inference (Object Detection & Segmentation)
outputs = predictor(image)

# ✅ Visualize Results
metadata = MetadataCatalog.get("publaynet_val")  # Use validation dataset metadata
visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
visualized_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

# ✅ Save & Show Output
OUTPUT_IMAGE_PATH = os.path.join(MODEL_DIR, "output_newspaper.jpg")
cv2.imwrite(OUTPUT_IMAGE_PATH, visualized_output.get_image()[:, :, ::-1])  # Save result
print(f"\n✅ Output saved at: {OUTPUT_IMAGE_PATH}")

# ✅ Display Output
plt.figure(figsize=(12, 8))
plt.imshow(visualized_output.get_image())
plt.axis("off")
plt.title("Newspaper Image Segmentation Output")
plt.show()
