import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from multiprocessing import freeze_support

# ✅ Windows Fix: Prevent Multiprocessing Error
if __name__ == "__main__":
    freeze_support()

    # ✅ Register dataset (Necessary before evaluation)
    dataset_root = r"D:\5101_D\Newspaper\ImageSegmentation\publaynet"
    val_json = os.path.join(dataset_root, "val.json")
    val_images = os.path.join(dataset_root, "val")

    register_coco_instances("publaynet_val", {}, val_json, val_images)

    # ✅ Verify Registration
    print("\n✅ Available Datasets:", list(DatasetCatalog.keys()))

    # ✅ Load configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # ✅ Set correct number of classes to avoid parameter mismatch
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Set this to match the trained model

    # ✅ Load trained weights
    cfg.MODEL.WEIGHTS = r"D:\5101_D\Newspaper\ImageSegmentation\Method-4\output_publaynet_2\model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for predictions
    cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if using GPU

    # ✅ Set dataset for evaluation
    cfg.DATASETS.TEST = ("publaynet_val",)

    # ✅ Initialize predictor
    predictor = DefaultPredictor(cfg)

    # ✅ Create evaluator
    evaluator = COCOEvaluator("publaynet_val", cfg, False, output_dir="./output_publaynet")
    val_loader = build_detection_test_loader(cfg, "publaynet_val")

    # ✅ Run inference and evaluation
    print("\n🚀 Running Evaluation on `publaynet_val`...\n")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
