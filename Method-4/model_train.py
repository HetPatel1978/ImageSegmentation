# import os
# import torch
# from detectron2.data.datasets import register_coco_instances
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer, hooks
# from detectron2 import model_zoo
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.modeling import build_model

# # Force CPU usage right at the beginning
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# torch.cuda.is_available = lambda: False

# class SaveModelHook(hooks.HookBase):
#     """Custom hook to save model periodically during training."""
#     def __init__(self, cfg, save_period=1000):
#         super().__init__()
#         self.cfg = cfg
#         self.save_period = save_period

#     def after_step(self):
#         if self.trainer.iter % self.save_period == 0:
#             save_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_{self.trainer.iter}.pth")
#             torch.save(self.trainer.model.state_dict(), save_path)
#             self.trainer.checkpointer.save(f"model_{self.trainer.iter}")

# class CustomTrainer(DefaultTrainer):
#     """Custom trainer that adds the model saving hook."""
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.append(SaveModelHook(self.cfg, save_period=1000))
#         return hooks

# def setup_config():
#     """Setup Detectron2 configuration for training."""
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

#     # Set dataset paths
#     dataset_root = r"D:\5101_D\Newspaper\ImageSegmentation\publaynet"
#     train_json = os.path.join(dataset_root, "train.json")
#     train_images = os.path.join(dataset_root, "train")
#     val_json = os.path.join(dataset_root, "val.json")
#     val_images = os.path.join(dataset_root, "val")

#     # Register dataset
#     register_coco_instances("publaynet_train", {}, train_json, train_images)
#     register_coco_instances("publaynet_val", {}, val_json, val_images)

#     # Model settings
#     cfg.DATASETS.TRAIN = ("publaynet_train",)
#     cfg.DATASETS.TEST = ("publaynet_val",)
#     cfg.DATALOADER.NUM_WORKERS = 0  # Avoid multiprocessing issues on Windows
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.00025
#     cfg.SOLVER.MAX_ITER = 5000
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in PubLayNet

#     # Explicitly force CPU
#     cfg.MODEL.DEVICE = "cpu"

#     # Output directory
#     cfg.OUTPUT_DIR = "./output_publaynet"
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # Load pre-trained weights while avoiding mismatched layers
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     model = build_model(cfg)
#     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

#     return cfg

# def train_model():
#     """Train the Detectron2 model."""
#     cfg = setup_config()
    
#     trainer = CustomTrainer(cfg)
#     trainer.resume_or_load(resume=False)
#     trainer.train()

# if __name__ == '__main__':
#     train_model()





















#improved one

import os
import cv2
import torch
import shutil
import numpy as np
import datetime

from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

# Force CPU usage (remove these lines if using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

# ------------------------------------------------------------------------------
# STEP 1: Create a New Unique Output Folder
# ------------------------------------------------------------------------------
base_output_dir = "./output_publaynet"
unique_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_output_dir, unique_folder)

if os.path.exists(output_dir):
    print(f"🗑️ Deleting old output directory: {output_dir}")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"✅ New output directory created: {output_dir}")

# ------------------------------------------------------------------------------
# STEP 2: Custom Hook to Save Model Checkpoints Periodically
# ------------------------------------------------------------------------------
class SaveModelHook(hooks.HookBase):
    """Custom hook to save model checkpoints periodically during training."""
    def __init__(self, cfg, save_period=1000):
        super().__init__()
        self.cfg = cfg
        self.save_period = save_period

    def after_step(self):
        if self.trainer.iter % self.save_period == 0:
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_{self.trainer.iter}.pth")
            torch.save(self.trainer.model.state_dict(), save_path)
            self.trainer.checkpointer.save(f"model_{self.trainer.iter}")
            print(f"✅ Saved checkpoint at iteration {self.trainer.iter}")

# ------------------------------------------------------------------------------
# STEP 3: Custom Data Mapper with Augmentations (Fixed Version)
# ------------------------------------------------------------------------------
def custom_mapper(dataset_dict):
    """
    A custom mapper that reads an image, applies augmentations, and processes instance annotations.
    """
    dataset_dict = dataset_dict.copy()  # avoid modifying the original dict
    
    # Read the image in BGR format then convert to RGB.
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # Define the list of augmentations: increase resolution, random flip, brightness, rotation.
    aug_list = [
        T.Resize((1600, 1600)),
        T.RandomFlip(prob=0.5, horizontal=True),
        T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
        T.RandomRotation(angle=[-15, 15]),
    ]
    
    # Apply augmentations using the helper function. 
    # This returns the augmented image and the applied transforms.
    image_aug, transforms = T.apply_transform_gens(aug_list, image)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_aug.transpose(2, 0, 1)))
    
    # Process annotations if available (transforms will also be applied to annotations)
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_aug.shape[:2])
            for obj in dataset_dict.pop("annotations")
        ]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image_aug.shape[:2])
    
    return dataset_dict

# ------------------------------------------------------------------------------
# STEP 4: Custom Trainer Using the Custom Mapper and Hook
# ------------------------------------------------------------------------------
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.append(SaveModelHook(self.cfg, save_period=1000))
        return hooks_list

# ------------------------------------------------------------------------------
# STEP 5: Setup the Detectron2 Configuration
# ------------------------------------------------------------------------------
def setup_config():
    """Setup Detectron2 configuration for training from scratch."""
    cfg = get_cfg()
    # Load the base config from the model zoo.
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset paths
    dataset_root = r"D:\5101_D\Newspaper\ImageSegmentation\publaynet"
    train_json = os.path.join(dataset_root, "train.json")
    train_images = os.path.join(dataset_root, "train")
    val_json = os.path.join(dataset_root, "val.json")
    val_images = os.path.join(dataset_root, "val")
    
    # Register training and validation datasets
    register_coco_instances("publaynet_train", {}, train_json, train_images)
    register_coco_instances("publaynet_val", {}, val_json, val_images)
    
    # Assign datasets for training and testing
    cfg.DATASETS.TRAIN = ("publaynet_train",)
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # For Windows
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4  
    cfg.SOLVER.BASE_LR = 0.0001  
    cfg.SOLVER.MAX_ITER = 20000  # Extended training iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in PubLayNet
    
    # Adjust anchor sizes for narrow columns
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    
    # Increase input resolution for training and testing
    cfg.INPUT.MIN_SIZE_TRAIN = (1600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    
    # Set new output directory for this training run
    cfg.OUTPUT_DIR = output_dir
    
    # Save configuration for reproducibility
    config_save_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(cfg.dump())
    print(f"✅ Configuration saved to {config_save_path}")
    
    # Train a new model from scratch: do NOT load any pre-trained weights.
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.DEVICE = "cpu"  # Force CPU
    
    return cfg

# ------------------------------------------------------------------------------
# STEP 6: Train the Model
# ------------------------------------------------------------------------------
def train_model():
    """Train the Detectron2 model from scratch and save all outputs in a new folder."""
    cfg = setup_config()
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Start fresh training
    trainer.train()

if __name__ == '__main__':
    train_model()
