import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

def setup_config():
    """Setup Detectron2 configuration for training."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Set dataset paths
    dataset_root = r"D:/5101_D/Newspaper/publaynet"
    train_json = os.path.join(dataset_root, "train.json")
    train_images = os.path.join(dataset_root, "train")
    val_json = os.path.join(dataset_root, "val.json")
    val_images = os.path.join(dataset_root, "val")

    # Register dataset
    register_coco_instances("publaynet_train", {}, train_json, train_images)
    register_coco_instances("publaynet_val", {}, val_json, val_images)

    # Model settings
    cfg.DATASETS.TRAIN = ("publaynet_train",)
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # Avoid multiprocessing issues on Windows
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in PubLayNet

    # Output directory
    cfg.OUTPUT_DIR = "./output_publaynet"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Load pre-trained weights while avoiding mismatched layers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    return cfg

def train_model():
    """Train the Detectron2 model."""
    cfg = setup_config()
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    train_model()
