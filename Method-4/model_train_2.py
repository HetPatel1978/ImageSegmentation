# import os
# import torch
# from detectron2.data.datasets import register_coco_instances
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer, hooks
# from detectron2 import model_zoo
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.modeling import build_model
# from detectron2.data import transforms as T
# from detectron2.data import DatasetMapper, build_detection_train_loader

# # Force CPU usage right at the beginning
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# torch.cuda.is_available = lambda: False

# # Custom hook to save model periodically
# class SaveModelHook(hooks.HookBase):
#     def __init__(self, cfg, save_period=1000):
#         super().__init__()
#         self.cfg = cfg
#         self.save_period = save_period

#     def after_step(self):
#         if self.trainer.iter % self.save_period == 0:
#             save_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_{self.trainer.iter}.pth")
#             torch.save(self.trainer.model.state_dict(), save_path)
#             self.trainer.checkpointer.save(f"model_{self.trainer.iter}")

# # Custom trainer with data augmentation and saving
# class CustomTrainer(DefaultTrainer):
#     @classmethod
#     def build_train_loader(cls, cfg):
#         return build_detection_train_loader(cfg, 
#             mapper=DatasetMapper(cfg, is_train=True, augmentations=[
#                 T.Resize((1024, 1024)),  # 🔥 Higher resolution for small object detection
#                 T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # 🔄 Flip augmentation
#                 T.RandomBrightness(0.8, 1.2),  # ☀️ Brightness adjustment
#                 T.RandomRotation(angle=[-10, 10]),  # 🔄 Small rotations
#             ])
#         )

#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.append(SaveModelHook(self.cfg, save_period=1000))
#         return hooks

# # Function to configure the model
# def setup_config():
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
#     cfg.SOLVER.IMS_PER_BATCH = 2  # Keep small to fit on CPU
#     cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
#     cfg.SOLVER.MAX_ITER = 10000  # 🔥 Extended training iterations
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # 🔥 Higher batch size for better segmentation
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in PubLayNet

#     # Small object detection improvements
#     cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1024)  # 🔥 Higher resolution for smaller objects
#     cfg.INPUT.MIN_SIZE_TEST = 1024
#     cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]  # 🔥 Reduced anchor sizes

#     # Explicitly force CPU
#     cfg.MODEL.DEVICE = "cpu"

#     # Output directory
#     cfg.OUTPUT_DIR = "./output_publaynet"
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # Load from last checkpoint instead of COCO weights
#     last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     if os.path.exists(last_checkpoint):
#         cfg.MODEL.WEIGHTS = last_checkpoint
#         print(f"🔄 Resuming training from: {last_checkpoint}")
#     else:
#         cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#         print("🆕 Starting fresh training with COCO pre-trained weights")

#     model = build_model(cfg)
#     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

#     return cfg

# # Function to start training
# def train_model():
#     cfg = setup_config()
    
#     trainer = CustomTrainer(cfg)
#     trainer.resume_or_load(resume=True)  # 🔥 Resume training
#     trainer.train()

# if __name__ == '__main__':
#     train_model()
























import os
import torch
import shutil
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

# Force CPU usage (remove if using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

# 🛑 STEP 1: SET NEW OUTPUT DIRECTORY (AND DELETE OLD CONTENTS)
output_dir = r"D:\5101_D\Newspaper\ImageSegmentation\Method-4\output_publaynet_2"

if os.path.exists(output_dir):
    print(f"🗑️ Deleting old training checkpoints from {output_dir}...")
    shutil.rmtree(output_dir)  # Delete the entire folder

os.makedirs(output_dir, exist_ok=True)  # Recreate empty output directory

class SaveModelHook(hooks.HookBase):
    """Custom hook to save model periodically during training."""
    def __init__(self, cfg, save_period=1000):
        super().__init__()
        self.cfg = cfg
        self.save_period = save_period

    def after_step(self):
        if self.trainer.iter % self.save_period == 0:
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_{self.trainer.iter}.pth")
            torch.save(self.trainer.model.state_dict(), save_path)
            self.trainer.checkpointer.save(f"model_{self.trainer.iter}")

class CustomTrainer(DefaultTrainer):
    """Custom trainer that adds the model saving hook."""
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(SaveModelHook(self.cfg, save_period=1000))
        return hooks

def setup_config():
    """Setup Detectron2 configuration for training from scratch."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Dataset paths
    dataset_root = r"D:\5101_D\Newspaper\ImageSegmentation\publaynet"
    train_json = os.path.join(dataset_root, "train.json")
    train_images = os.path.join(dataset_root, "train")
    val_json = os.path.join(dataset_root, "val.json")
    val_images = os.path.join(dataset_root, "val")

    # Register dataset
    register_coco_instances("publaynet_train", {}, train_json, train_images)
    register_coco_instances("publaynet_val", {}, val_json, val_images)

    # Train from scratch (Do NOT load old weights)
    cfg.DATASETS.TRAIN = ("publaynet_train",)
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  
    cfg.SOLVER.IMS_PER_BATCH = 4  
    cfg.SOLVER.BASE_LR = 0.0001  
    cfg.SOLVER.MAX_ITER = 10000  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  

    # Ensure only CPU is used
    cfg.MODEL.DEVICE = "cpu"

    # Increase image resolution
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Apply data augmentation
    cfg.INPUT.AUGMENTATIONS = [
        "Resize((1024,1024))",
        "RandomFlip(prob=0.5)",
        "RandomBrightness(intensity_min=0.8, intensity_max=1.2)",
        "RandomRotation(angle=[-10, 10])"
    ]

    # Set new output directory
    cfg.OUTPUT_DIR = r"D:\5101_D\Newspaper\ImageSegmentation\Method-4\output_publaynet_2"
    
    # Ensure no pretrained weights are loaded
    cfg.MODEL.WEIGHTS = ""

    return cfg

def train_model():
    """Train the Detectron2 model from scratch."""
    cfg = setup_config()
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Ensure training starts from scratch
    trainer.train()

if __name__ == '__main__':
    train_model()
